"""
BGAN module.

This module implements the Bayesian Generative Adversarial Network (BGAN) for tabular data,
with extensions for uncertainty estimation, Bayesian regularization, and optional Bayesian
Network (BN) structure integration. It is inspired by CTGAN (https://github.com/sdv-dev/CTGAN)
and TableGAN, but introduces several key differences:

- Uncertainty Estimation: The generator outputs both mean and log-variance, enabling
  aleatoric uncertainty estimation for each feature.
- Bayesian Regularization: KL divergence is used to regularize the latent space.
- MC Dropout: The discriminator uses dropout at inference for Bayesian uncertainty.
- BN Integration: Optionally, a Bayesian Network structure can guide the generator
  via soft parent embedding transformations.
- Flexible Optimizers: Supports Adam, AdamW, RMSprop, and SGD.
- PAC Discriminator: The discriminator operates on grouped ("packed") samples for
  improved stability, as in CTGAN.

This code is inspired by and partially adapted from CTGAN (MIT License), but is
substantially modified for Bayesian and uncertainty-aware imputation and synthesis.

MIT License applies to portions derived from CTGAN:
https://github.com/sdv-dev/CTGAN/blob/master/LICENSE
"""

import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm
import torch.nn as nn

from bgan.data_sampler import DataSampler
from bgan.data_transformer import DataTransformer
from bgan.errors import InvalidDataError
from bgan.synthesizers.base import BaseSynthesizer, random_state


class Discriminator(nn.Module):
    """
    Bayesian Discriminator with MC Dropout and PAC grouping.

    Differences from standard GAN:
    - Uses MC Dropout for Bayesian uncertainty.
    - Operates on "packed" samples (PAC) for improved stability (as in CTGAN).
    """

    def __init__(self, input_dim, discriminator_dim, pac=10, dropout_prob=0.2):
        super(Discriminator, self).__init__()

        self.pac = pac
        self.pacdim = input_dim * pac

        self.input_dim = input_dim
        self.dropout_prob = dropout_prob

        dim = self.pacdim
        layers = []
        for item in discriminator_dim:
            layers.append(nn.Linear(dim, item))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(p=dropout_prob))
            dim = item

        self.seq = nn.ModuleList(layers)
        # Improved Discriminator
        self.hidden = nn.Sequential(
            nn.Linear(input_dim * pac, discriminator_dim[0]),
            nn.LeakyReLU(0.2),
            #nn.BatchNorm1d(discriminator_dim[0]),
            nn.Dropout(p=dropout_prob),
            *[nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(0.2),
                #nn.BatchNorm1d(out_dim),
                nn.Dropout(p=dropout_prob)
            ) for in_dim, out_dim in zip(discriminator_dim[:-1], discriminator_dim[1:])],
        )

        self.output_layer = nn.Linear(dim, 1)

    def forward(self, x):
        """
        Forward pass with PAC grouping.
        """
        # Reshape for PAC
        assert x.size()[0] % self.pac == 0, 'Batch size must be divisible by pac.'
        x = x.view(-1, self.input_dim * self.pac)

        h = self.hidden(x)
        return self.output_layer(h)

    def calc_gradient_penalty(self, real_data, fake_data, device, pac):
        """
        Compute gradient penalty for WGAN-GP.
        """
        alpha = torch.rand(real_data.size(0), 1, device=device).expand(real_data.size())
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad_(True)
        
        disc_interpolates = self(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates), create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and skip connection.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act2 = nn.LeakyReLU(0.2)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(self.fc1(x)))
        out = self.act2(self.bn2(self.fc2(out)))
        return out + self.skip(x)

class Generator(nn.Module):
    """
    Generator with uncertainty estimation and optional BN structure guidance.

    Differences from standard GAN generator:
    - Outputs mean and log-variance for each feature (aleatoric uncertainty).
    - Outputs an uncertainty map (Softplus).
    - Optionally integrates Bayesian Network structure by transforming parent embeddings.
    """
    def __init__(self, embedding_dim, generator_dim, data_dim, bn_structure=None, bn_influence = 0.1):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.data_dim = int(data_dim)
        self.bn_structure = bn_structure
        self.bn_influence = bn_influence

        # Standard generator backbone
        dim = self.embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [
                nn.Linear(dim, item),
                nn.LayerNorm(item),
                nn.LeakyReLU(0.2)
            ]
            dim = item
        self.hidden = nn.Sequential(*seq)
         # Separate outputs for mean, log_var, and uncertainty
        self.mean_layer = nn.Linear(dim, self.data_dim)
        self.log_var_layer = nn.Linear(dim, self.data_dim)
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(dim, self.data_dim),
            nn.Softplus()
        )

        # BN-guided soft guidance
        if bn_structure:
            self.bn_transforms = nn.ModuleDict()
            for node, parents in bn_structure.items():
                if parents:
                    # Create a linear layer to transform the parents' embeddings
                    self.bn_transforms[node] = nn.Linear(
                        len(parents) * embedding_dim, embedding_dim
                    )
        else:
            self.bn_transforms = None


    def forward(self, z):
        """
        Forward pass with optional BN guidance and uncertainty estimation.
        """
        # Soft BN guidance: transform parents' embeddings and add to node's embedding
        if self.bn_structure and self.bn_transforms is not None:
            for node, parents in self.bn_structure.items():
                if parents:
                    # Concatenate parents' embeddings
                    parent_embeddings = torch.cat([z[:, i] for i in parents], dim=1)
                    # Transform parents' embeddings
                    transformed_embeddings = self.bn_transforms[node](parent_embeddings)
                    # Add transformed embeddings to node's embedding
                    #CHANGE THE CONSTANT HERE TO IMPACT THE INFLUENCE OF THE INFLUENCE EMBEDDING ON THE NODE'S ORIGINAL EMBEDDING
                    z[:, node] = z[:, node] + self.bn_influence * transformed_embeddings

        h = self.hidden(z)
        # Generate mean and log_var
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sampled = mean + std * eps
        uncertainty = self.uncertainty_layer(h)
        return sampled, uncertainty, (mean, log_var)



class BGAN(BaseSynthesizer):
    """
    Bayesian Generative Adversarial Network (BGAN) for Tabular Data.

    This class implements a GAN-based synthesizer for tabular data with several
    Bayesian and uncertainty-aware extensions, as proposed in our thesis.

    Key features and differences from standard GANs and prior work:
    ----------------------------------------------------------------
    - Uncertainty Estimation: The generator outputs both mean and log-variance,
      enabling aleatoric uncertainty estimation for each feature. This is not present
      in standard GANs or in CTGAN/TableGAN.
    - Bayesian Regularization: KL divergence is used to regularize the latent space,
      encouraging the generator to produce samples from a distribution close to standard normal.
    - MC Dropout in Discriminator: The discriminator uses dropout at inference time,
      providing Bayesian uncertainty estimates for the real/fake decision.
    - Optional Bayesian Network (BN) Integration:* The generator can be guided by a
      learned Bayesian Network structure, allowing soft parent embedding transformations
      to influence feature generation. This is a novel extension for structure-aware synthesis.
    - PAC Discriminator: The discriminator operates on grouped ("packed") samples for
      improved stability, as introduced in CTGAN [Xu et al., 2019].
    - Flexible Optimizer Selection: Supports Adam, AdamW, RMSprop, and SGD.
    - Ablation Controls: Uncertainty loss and KL loss can be toggled for ablation studies.

    References:
    -----------
    - Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
      Modeling Tabular data using Conditional GAN. NeurIPS 2019.
      [https://arxiv.org/abs/1907.00503]
    - CTGAN codebase (MIT License): https://github.com/sdv-dev/CTGAN
      Portions of this code are adapted from CTGAN, with substantial modifications
      for Bayesian and uncertainty-aware synthesis. MIT License applies to those portions.

    Args:
        embedding_dim (int): Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints): Hidden layer sizes for the generator. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints): Hidden layer sizes for the discriminator. Defaults to (256, 256).
        generator_lr (float): Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float): Generator weight decay for the optimizer. Defaults to 1e-6.
        discriminator_lr (float): Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float): Discriminator weight decay for the optimizer. Defaults to 1e-6.
        batch_size (int): Number of data samples to process in each step.
        discriminator_steps (int): Number of discriminator updates per generator update.
        log_frequency (bool): Whether to use log frequency of categorical levels in conditional sampling.
        verbose (bool): Whether to print progress results.
        epochs (int): Number of training epochs.
        pac (int): Number of samples to group together when applying the discriminator (PAC).
        cuda (bool or str): Whether to use CUDA for GPU computation.
        beta (float): Weight for the uncertainty loss term.
        bn_structure (dict): Optional Bayesian network structure for generator guidance.
        kl_weight (float): Weight for the KL divergence loss term.
        bn_influence (float): Influence of BN-guided parent embeddings in the generator.
        use_uncertainty_loss (bool): Whether to include uncertainty loss in the generator objective.
        use_kl_loss (bool): Whether to include KL divergence loss in the generator objective.
        optimizer_type (str): Optimizer type ("adam", "adamw", "rmsprop", "sgd").
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=0.0002,
        generator_decay=0.000001,
        discriminator_lr=0.0002,
        discriminator_decay=0.000001,
        batch_size=200,
        discriminator_steps=5,
        log_frequency=True,
        verbose=False,
        epochs=50,
        pac=10,
        cuda=True,
        beta=1e-3,  # control uncertainty loss weight
        bn_structure=None,
        kl_weight=1e-3,  # KL divergence weight
        bn_influence = 0.1,
        use_uncertainty_loss=True,
        use_kl_loss=True,
        optimizer_type="adam"
    ):
        assert batch_size % 2 == 0
        super().__init__()
        #self.embedding_dim = embedding_dim
        self.bn_structure = bn_structure
        self.optimizer_type = optimizer_type

        self.bn_influence = bn_influence
        self.use_uncertainty_loss = use_uncertainty_loss
        self.use_kl_loss = use_kl_loss

        #self._embedding_dim = embedding_dim
         # Convert dimensions to lists
        self._embedding_dim = embedding_dim
        #self.bn_structure = bn_structure
        self._generator_dim = (
            generator_dim if isinstance(generator_dim, list) 
            else list(generator_dim) if isinstance(generator_dim, tuple)
            else [generator_dim]
        )
        self._discriminator_dim = (
            discriminator_dim if isinstance(discriminator_dim, list)
            else list(discriminator_dim) if isinstance(discriminator_dim, tuple)
            else [discriminator_dim]
        )

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._kl_weight = kl_weight  

        self._beta = beta  
        


        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    def _get_optimizer(self, params, lr, betas=None, weight_decay=0):
        """
        Flexible optimizer selection.
        """
        if self.optimizer_type == "adam":
            return optim.Adam(params, lr=lr, betas=betas or (0.5, 0.9), weight_decay=weight_decay)
        elif self.optimizer_type == "adamw":
            return optim.AdamW(params, lr=lr, betas=betas or (0.9, 0.999), weight_decay=weight_decay)
        elif self.optimizer_type == "rmsprop":
            return optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay)
        elif self.optimizer_type == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _kl_divergence(self, mean, log_var):
        """
        Compute KL divergence between N(mean, std) and N(0, 1)
        """
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        # Convert tuple to tensor if needed
        if isinstance(data, tuple):
            data = data[0]
            
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def _validate_null_data(self, train_data, discrete_columns):
        """Check whether null values exist in continuous ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

        if any_nulls:
            raise InvalidDataError(
                'BGAN does not support null values in the continuous training data. '
                'Please remove all null values from your continuous training data.'
            )

    def fit(self, train_data, discrete_columns=(), epochs=None, bn_structure=None):
            """
            Fit the BGAN model to the training data.
            """
            # Validate columns and data
            self._validate_discrete_columns(train_data, discrete_columns)
            self._validate_null_data(train_data, discrete_columns)

            if epochs is None:
                epochs = self._epochs
            else:
                warnings.warn(
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead',
                    DeprecationWarning,
                )

            # Data Transformation and Sampling
            self._transformer = DataTransformer()
            self._transformer.fit(train_data, discrete_columns)
            train_data = self._transformer.transform(train_data)

            self._data_sampler = DataSampler(
                train_data, self._transformer.output_info_list, self._log_frequency
            )

            data_dim = self._transformer.output_dimensions
            cond_dim = self._data_sampler.dim_cond_vec()

            # Initialize Generator and Discriminator
            self._generator = Generator(
                self._embedding_dim + cond_dim,
                self._generator_dim,
                data_dim,
                bn_structure=bn_structure,
                bn_influence = self.bn_influence
            ).to(self._device)

            self._discriminator = Discriminator(
                data_dim + cond_dim,
                self._discriminator_dim,
                pac=self.pac
            ).to(self._device)

            optimizerG = self._get_optimizer(self._generator.parameters(), self._generator_lr, betas=(0.5, 0.9), weight_decay=self._generator_decay)
            optimizerD = self._get_optimizer(self._discriminator.parameters(), self._discriminator_lr, betas=(0.5, 0.9), weight_decay=self._discriminator_decay)

            embedding_dim = self._embedding_dim
            batch_size = self._batch_size
            mean = torch.zeros(batch_size, embedding_dim, device=self._device)
            std = torch.ones(batch_size, embedding_dim, device=self._device)  # Use ones instead of mean + 1
            kl_anneal_epochs = 15  # you can tune this (MAKE SURE TO KEEP CONSISTENT THROUGHOUT CLASS)


            # Initialize loss values dataframe
            self.loss_values = pd.DataFrame({
                'Epoch': pd.Series(dtype='int64'),
                'Generator Loss': pd.Series(dtype='float64'),
                'Discriminator Loss': pd.Series(dtype='float64')
            })

            # Loop over epochs
            epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
            if self._verbose:
                description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
                epoch_iterator.set_description(description.format(gen=0, dis=0))

            steps_per_epoch = max(len(train_data) // self._batch_size, 1)

            for i in epoch_iterator:
                for _ in range(steps_per_epoch):
                    # === Train Discriminator ===
                    for _ in range(self._discriminator_steps):
                        condvec = self._data_sampler.sample_condvec(self._batch_size)
                        if condvec is None:
                            noise_dim = self._embedding_dim
                        else:
                            c1, m1, col, opt = condvec
                            noise_dim = self._embedding_dim + cond_dim
                        mean = torch.zeros(self._batch_size, noise_dim, device=self._device)
                        std = torch.ones(self._batch_size, noise_dim, device=self._device)
                        fakez = torch.normal(mean=mean, std=std)

                        if condvec is None:
                            c1, m1, col, opt = None, None, None, None
                            real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                        else:
                            c1, m1, col, opt = condvec
                            c1 = torch.from_numpy(c1).to(self._device)
                            m1 = torch.from_numpy(m1).to(self._device)
                            # fakez already has correct shape, so no need to concatenate c1 again

                            perm = np.arange(self._batch_size)
                            np.random.shuffle(perm)
                            real = self._data_sampler.sample_data(
                                train_data, self._batch_size, col[perm], opt[perm]
                            )
                            c2 = c1[perm]

                        fake, uncertainty_map, _ = self._generator(fakez)  # Ignore mean/log_var for discriminator
                        fakeact = self._apply_activate(fake)
                        real = torch.from_numpy(real.astype('float32')).to(self._device)

                        if c1 is not None:
                            fake_cat = torch.cat([fakeact, c1], dim=1)
                            real_cat = torch.cat([real, c2], dim=1)
                        else:
                            real_cat = real
                            fake_cat = fakeact

                        y_fake = self._discriminator(fake_cat)
                        y_real = self._discriminator(real_cat)

                        pen = self._discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, self._device, self.pac
                        )
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                        optimizerD.zero_grad(set_to_none=False)
                        pen.backward(retain_graph=True)
                        loss_d.backward()
                        optimizerD.step()

                    # Train Generator
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        # Do NOT concatenate c1 to fakez here

                    fake, uncertainty_map, (mean, log_var) = self._generator(fakez)  # Unpack all outputs
                    fakeact = self._apply_activate(fake)

                     # Calculate KL divergence
                    kl_loss = self._kl_divergence(mean, log_var) if self.use_kl_loss else 0
                    kl_annealed_weight = self._kl_weight * min(1.0, i / kl_anneal_epochs) if self.use_kl_loss else 0

                    if c1 is not None:
                        y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                    else:
                        y_fake = self._discriminator(fakeact)

                    cross_entropy = 0 if condvec is None else self._cond_loss(fake, c1, m1)

                    # ABLATION STUDY -- REMOVE UNCERTAINTY AND KL LOSSES HERE
                    uncertainty_loss = torch.mean(torch.log(uncertainty_map ** 2 + 1e-8)) if self.use_uncertainty_loss else 0.0 # avoid log(0)

                    #kl_annealed_weight = self._kl_weight * min(1.0, i / kl_anneal_epochs)
                    loss_g = -torch.mean(y_fake) + cross_entropy 
                    if self.use_uncertainty_loss:
                        loss_g += self._beta * uncertainty_loss
                    if self.use_kl_loss:
                        loss_g += kl_annealed_weight * kl_loss

                    optimizerG.zero_grad(set_to_none=False)
                    loss_g.backward()
                    optimizerG.step()

                # Record and print epoch stats
                generator_loss = loss_g.detach().cpu().item()
                discriminator_loss = loss_d.detach().cpu().item()

                # Update loss values
                epoch_loss_df = pd.DataFrame({
                    'Epoch': pd.Series([i], dtype='int64'),
                    'Generator Loss': [generator_loss],
                    'Discriminator Loss': [discriminator_loss],
                })

                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df],
                    ignore_index=True
                )

                if self._verbose:
                    epoch_iterator.set_description(
                        description.format(gen=generator_loss, dis=discriminator_loss)
                    )

            self.generator = self._generator
            self.transformer = self._transformer
            self.data_sampler = self._data_sampler
            self.embedding_dim = self._embedding_dim

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def get_discriminator(self):
        """
        Return the discriminator network.
        """
        return self._discriminator
    
    def eval(self):
        """
        Set generator and discriminator to evaluation mode.
        """
        if self._generator:
            self._generator.eval()
        if self._discriminator:
            self._discriminator.eval()
