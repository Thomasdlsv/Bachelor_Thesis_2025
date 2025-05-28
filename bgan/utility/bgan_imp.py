import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from bgan.data_sampler import DataSampler
from bgan.data_transformer import DataTransformer
from bgan.errors import InvalidDataError
from bgan.synthesizers.base import BaseSynthesizer
from bgan.synthesizers.bgan import Generator
from bgan.synthesizers.bgan import Discriminator
from torch import optim


class BGAIN(BaseSynthesizer):
    """
    Bayesian Generative Adversarial Imputation Network (BGAIN).
    GAN-based imputer for missing data using Bayesian regularization and uncertainty.
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
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=1,
        pac=10,
        cuda=True,
        beta=1e-3,
        bn_structure=None,
        kl_weight=1e-3,
    ):
        
        """
        Initialize the BGAIN imputer.

        Args:
            embedding_dim (int): Embedding dimension for generator input.
            generator_dim (tuple): Hidden layer sizes for generator.
            discriminator_dim (tuple): Hidden layer sizes for discriminator.
            generator_lr (float): Learning rate for generator.
            generator_decay (float): Weight decay for generator optimizer.
            discriminator_lr (float): Learning rate for discriminator.
            discriminator_decay (float): Weight decay for discriminator optimizer.
            batch_size (int): Batch size for training.
            discriminator_steps (int): Discriminator steps per generator step.
            log_frequency (bool): Whether to log progress.
            verbose (bool): Verbosity flag.
            epochs (int): Number of training epochs.
            pac (int): Number of samples per discriminator batch.
            cuda (bool): Use CUDA if available.
            beta (float): Uncertainty loss weight.
            bn_structure (dict): Optional Bayesian network structure.
            kl_weight (float): KL divergence loss weight.
        """

        super().__init__()
        self._embedding_dim = embedding_dim
        self._generator_dim = list(generator_dim)
        self._discriminator_dim = list(discriminator_dim)
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
        self.bn_structure = bn_structure

        if not cuda or not torch.cuda.is_available():
            self._device = torch.device('cpu')
        else:
            self._device = torch.device('cuda')

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
        self.loss_values = None

    def _kl_divergence(self, mean, log_var):
        """Compute KL divergence for Bayesian regularization."""
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()

    def _gumbel_softmax(self, logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Apply Gumbel-Softmax for categorical outputs."""
        for _ in range(10):
            transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed
        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """
        Apply activation functions to generator output based on transformer info.
        """
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
        """
        Compute cross-entropy loss for discrete columns.
        """
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
                    tmp = F.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """
        Ensure discrete columns exist in the training data.
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
        """
        Ensure no nulls in continuous columns of training data.
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
            Fit the BGAIN model to the training data.

            Args:
                train_data (pd.DataFrame or np.ndarray): Training data.
                discrete_columns (list): List of discrete columns.
                epochs (int, optional): Number of epochs to train.
                bn_structure (dict, optional): Bayesian network structure.
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
                bn_structure=bn_structure
            ).to(self._device)

            self._discriminator = Discriminator(
                data_dim + cond_dim,
                self._discriminator_dim,
                pac=self.pac
            ).to(self._device)

            # Optimizers for Generator and Discriminator
            optimizerG = optim.Adam(
                self._generator.parameters(),
                lr=self._generator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._generator_decay,
            )

            optimizerD = optim.Adam(
                self._discriminator.parameters(),
                lr=self._discriminator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._discriminator_decay,
            )

            embedding_dim = self._embedding_dim
            batch_size = self._batch_size
            mean = torch.zeros(batch_size, embedding_dim, device=self._device)
            std = torch.ones(batch_size, embedding_dim, device=self._device)  
            kl_anneal_epochs = 15  # CAN BE TUNED AND EXPERIMENTED WITH, KEPT AT 15 THROUGHOUT THIS EXPERIMENTAITON

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
                    kl_loss = self._kl_divergence(mean, log_var)
                    kl_annealed_weight = self._kl_weight * min(1.0, i / kl_anneal_epochs)

                    if c1 is not None:
                        y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                    else:
                        y_fake = self._discriminator(fakeact)

                    cross_entropy = 0 if condvec is None else self._cond_loss(fake, c1, m1)

                    uncertainty_loss = torch.mean(torch.log(uncertainty_map ** 2 + 1e-8))  # avoid log(0)

                    #kl_annealed_weight = self._kl_weight * min(1.0, i / kl_anneal_epochs)
                    loss_g = -torch.mean(y_fake) + cross_entropy + self._beta * uncertainty_loss + kl_annealed_weight * kl_loss

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

    def transform(model, original_df, num_rows=5):
        """
        Impute missing rows using the trained BGAIN model.

        Args:
            original_df (pd.DataFrame): DataFrame with missing values.
            num_rows (int): Number of rows to impute and display.

        Returns:
            np.ndarray: Imputed data.
        """
        # Find rows with missing values
        missing_rows = original_df[original_df.isna().any(axis=1)].copy()
        sample_rows = missing_rows.head(num_rows)

        print("\nOriginal rows with missing values:")
        print(sample_rows)

        # Transform using the same transformer
        transformed = model._transformer.transform(sample_rows)

        # Sample noise
        noise = torch.randn(len(transformed), model.embedding_dim).to(model._device)

        # Optional: add conditional vector
        condvec = model.data_sampler.sample_condvec(len(transformed))
        if condvec is not None:
            c, _, _, _ = condvec
            c = torch.from_numpy(c).to(model._device)
            noise = torch.cat([noise, c], dim=1)

        # Generate imputed data
        with torch.no_grad():
            synthetic, _, _ = model.generator(noise)
            synthetic_activated = model._apply_activate(synthetic)

        # Convert back to original data space
        imputed = model.transformer.inverse_transform(synthetic_activated.cpu().numpy())

        print("\nImputed rows:")
        print(pd.DataFrame(imputed, columns=original_df.columns).head(num_rows))

        return imputed

    def impute_all_missing(self, df_with_missing):
        """
        Impute all missing rows in the DataFrame using the trained BGAIN model.

        Args:
            df_with_missing (pd.DataFrame): DataFrame with missing values.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        df_filled = df_with_missing.copy()
        missing_rows = df_filled[df_filled.isna().any(axis=1)]
        if missing_rows.empty:
            return df_filled

        # Transform missing rows
        transformed = self._transformer.transform(missing_rows)
        noise = torch.randn(len(transformed), self.embedding_dim).to(self._device)
        condvec = self.data_sampler.sample_condvec(len(transformed))
        if condvec is not None:
            c, _, _, _ = condvec
            c = torch.from_numpy(c).to(self._device)
            noise = torch.cat([noise, c], dim=1)

        with torch.no_grad():
            synthetic, _, _ = self.generator(noise)
            synthetic_activated = self._apply_activate(synthetic)
        imputed = self.transformer.inverse_transform(synthetic_activated.cpu().numpy())
        imputed_df = pd.DataFrame(imputed, columns=df_with_missing.columns, index=missing_rows.index)

        # Fill only missing values in the original DataFrame
        for idx in missing_rows.index:
            for col in df_with_missing.columns:
                if pd.isnull(df_filled.at[idx, col]):
                    df_filled.at[idx, col] = imputed_df.at[idx, col]
        return df_filled