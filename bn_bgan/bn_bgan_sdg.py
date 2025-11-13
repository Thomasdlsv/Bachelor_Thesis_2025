import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.metrics import mutual_info_score
import torch
from bgan.data_transformer import DataTransformer
from bgan.synthesizers.bgan import BGAN
import pandas as pd
import torch.nn as nn
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BIC
import networkx as nx
import matplotlib.pyplot as plt

# =================================
# Generator with Batch Normalization
# ==================================

class GeneratorWithBN(nn.Module):

    """
    Generator neural network with batch normalization and optional Bayesian Network structure integration.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, bn_structure=None, use_spectral_norm=False):
        super(GeneratorWithBN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bn_structure = bn_structure
        self.use_spectral_norm = use_spectral_norm

        # Double the hidden size for deeper capacity
        hidden_dims = [dim * 2 for dim in hidden_dims]
        self.gate_log = {}  # Track gate values per node
        
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            main_layer = nn.Sequential(
                self._linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            )
            self.layers.append(main_layer)
            self.skip_layers.append(nn.Sequential(
                self._linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ))
            prev_dim = hidden_dim

        self.output_layer = nn.Sequential(
            self._linear(prev_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Tanh()
        )

        if bn_structure:
            self.bn_transforms = nn.ModuleDict()
            self.bn_attention = nn.ModuleDict()
            self.bn_combine = nn.ModuleDict()
            
            for node, parents in bn_structure.items():
                if parents:
                    self.bn_transforms[str(node)] = nn.Sequential(
                        self._linear(len(parents) * input_dim, input_dim * 2),
                        nn.LayerNorm(input_dim * 2),
                        nn.ReLU(),
                        nn.Linear(input_dim * 2, input_dim),
                        nn.LayerNorm(input_dim)
                    )
                    
                    self.bn_attention[str(node)] = nn.Sequential(
                        nn.Linear(input_dim, len(parents) * 8),
                        nn.LayerNorm(len(parents) * 8),
                        nn.ReLU(),
                        nn.Linear(len(parents) * 8, len(parents)),
                        nn.Sigmoid()  # Changed from softmax to sigmoid for gating
                    )
                    
                    self.bn_combine[str(node)] = nn.Sequential(
                        nn.Linear(input_dim * 2, input_dim),
                        nn.LayerNorm(input_dim),
                        nn.ReLU()
                    )

    def _linear(self, in_dim, out_dim):

        """
        Create a (spectral) linear layer.
        """

        layer = nn.Linear(in_dim, out_dim)
        return nn.utils.spectral_norm(layer) if self.use_spectral_norm else layer

    def forward(self, x):

        """
        Forward pass through the generator with BN and skip connections.
        """

        original_x = x.clone()
        
        for idx, (layer, skip) in enumerate(zip(self.layers, self.skip_layers)):
            main_path = layer(x)
            skip_path = skip(original_x)

            # Adaptive skip-main gate
            gate = torch.sigmoid(torch.mean(parent_influence, dim=1, keepdim=True))

            # Log gate values
            if str(node) not in self.gate_log:
                self.gate_log[str(node)] = []
            self.gate_log[str(node)].append(gate.detach().cpu().mean().item())

            x = gate * main_path + (1 - gate) * skip_path

            # BN integration
            if self.bn_structure and hasattr(self, 'bn_transforms'):
                if not hasattr(self, 'gate_log'):
                    self.gate_log = {}
                for node, parents_with_weights in self.bn_structure.items():
                    if str(node) not in self.bn_transforms:
                        continue
                    
                    node_idx = int(node)
                    node_features = x[:, node_idx:node_idx + 1]

                    parents, weights = zip(*parents_with_weights)
                    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=x.device)

                    parent_embeddings = torch.cat([
                        x[:, int(p):int(p) + 1] * w
                        for p, w in zip(parents, weights)
                    ], dim=1)

                    attention = self.bn_attention[str(node)](node_features)
                    attention = attention * weights_tensor.unsqueeze(0)  # Apply gating
                    parent_influence = self.bn_transforms[str(node)](parent_embeddings * attention)

                    combined = torch.cat([node_features, parent_influence], dim=1)
                    update = self.bn_combine[str(node)](combined)

                    influence_gate = torch.sigmoid(torch.mean(parent_influence, dim=1, keepdim=True))
                    x[:, node_idx:node_idx + 1] = influence_gate * update + (1 - influence_gate) * node_features

        return self.output_layer(x)

# =====================================
# Discriminator with Batch Normalization
# =====================================

class DiscriminatorWithBN(nn.Module):

    """
    Discriminator neural network with batch normalization.
    """

    def __init__(self, input_dim, hidden_dims, use_spectral_norm=False, return_features=False):
        super(DiscriminatorWithBN, self).__init__()
        
        self.use_spectral_norm = use_spectral_norm
        self.return_features = return_features
        self.hidden_layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Sequential(
                self._linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Sequential(
            self._linear(prev_dim, 1),
            nn.Sigmoid()
        )

    def _linear(self, in_dim, out_dim):

        """
        Create a (spectral) linear layer.
        """

        layer = nn.Linear(in_dim, out_dim)
        return nn.utils.spectral_norm(layer) if self.use_spectral_norm else layer

    def forward(self, x):

        """
        Forward pass through the discriminator.
        """

        features = []
        for layer in self.hidden_layers:
            x = layer(x)
            if self.return_features:
                features.append(x)
        out = self.output_layer(x)
        return (out, features) if self.return_features else out

# ==================================
# BGANWithBN: BGAN with BN integration
# ===================================
    
class BGANWithBN(BGAN):

    """
    BGAN with batch normalization and Bayesian Network structure integration.
    """

    def __init__(self, epochs, batch_norm=True, optimizer_type="adam", use_uncertainty_loss=True,
        use_kl_loss=True,
        bn_influence=0.1,
        embedding_dim=256,
        **kwargs):
        self._embedding_dim = embedding_dim
        self.use_bn = batch_norm
        self.bn_structure = None
        self._model_built = False
        self.optimizer_type = optimizer_type
        self.use_uncertainty_loss = use_uncertainty_loss
        self.use_kl_loss = use_kl_loss
        self.bn_influence = bn_influence

        
        super().__init__(
            embedding_dim=self._embedding_dim,
            epochs=epochs,
            optimizer_type=optimizer_type,
            use_uncertainty_loss=use_uncertainty_loss,
            use_kl_loss=use_kl_loss,
            bn_influence=bn_influence,
            **kwargs
        )

    def _build_model(self, data, discrete_columns):

        """
        Build the generator and discriminator models with BN structure.
        """

        if self._model_built:
            return

        # First build transformer to get data dimensions
        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_columns)
        
        self._data_dim = self._transformer.output_dimensions
        if self._data_dim is None:
            self._data_dim = data.shape[1]
            print(f"Warning: Using fallback data dimension: {self._data_dim}")

        print("Building BN-enhanced generator...")
        self._generator = GeneratorWithBN(
            input_dim=self._embedding_dim,
            output_dim=self._data_dim,
            hidden_dims=[512, 512],
            bn_structure=self.bn_structure if self.use_bn else None,
        )
        self._discriminator = DiscriminatorWithBN(
            input_dim=self._data_dim,
            hidden_dims=[512, 512]
        )
        
        self._model_built = True

    def fit(self, data, discrete_columns):

        """
        Fit the BGAN model, ensuring the model is built first.
        """

        if not self._model_built:
            self._build_model(data, discrete_columns)
        return super().fit(data, discrete_columns)
    
class BN_AUG_SDG:

    """
    Bayesian Network-augmented Synthetic Data Generator (SDG).
    Combines a learned BN structure with a BGAN for improved synthetic data generation and imputation.
    """

    def __init__(self, epochs=100, batch_norm=True, embedding_dim=256, bn_influence=0.1, use_uncertainty_loss=True, use_kl_loss=True, optimizer_type="adam"):
        self.epochs = epochs
        self.batch_norm = batch_norm
        self.embedding_dim = embedding_dim
        self.bn_influence = bn_influence
        self.use_uncertainty_loss = use_uncertainty_loss
        self.use_kl_loss = use_kl_loss
        self.optimizer_type = optimizer_type

        self.bgan = BGAN(
            epochs = epochs,
            bn_influence = bn_influence,
            use_uncertainty_loss=use_uncertainty_loss,
            use_kl_loss=use_kl_loss,
            optimizer_type=optimizer_type
        )
        self.bn_structure = None
        self.node_importance = {}

    def _calculate_mutual_information(self, x, y):

        """
        Compute mutual information between two variables.
        """

        if x.dtype == 'object' or y.dtype == 'object':
            x = LabelEncoder().fit_transform(x)
            y = LabelEncoder().fit_transform(y)
        return mutual_info_score(x, y)

    def learn_bn_structure(self, data):

        """
        Learn a Bayesian Network structure using mutual information-based weights.
        Returns a weighted BN structure.
        """

        estimator = HillClimbSearch(data)
        model = estimator.estimate(scoring_method=BIC(data), max_iter=50, epsilon=1e-4)

        self.bn_structure = {}
        raw_weights = {}

        for parent, child in model.edges():
            if child not in self.bn_structure:
                self.bn_structure[child] = []
                raw_weights[child] = []

            self.bn_structure[child].append(parent)
            score = self._calculate_mutual_information(data[parent], data[child])
            raw_weights[child].append(score)

        # Normalize weights
        for node, weights in raw_weights.items():
            norm_weights = np.array(weights) / np.sum(weights)
            self.node_importance[node] = {
                parent: float(w) for parent, w in zip(self.bn_structure[node], norm_weights)
            }

        # Construct final weighted BN structure
        weighted_structure = {
            node: [(parent, self.node_importance[node][parent]) for parent in parents]
            for node, parents in self.bn_structure.items()
        }

        print("Learned Bayesian Network structure with weighted edges:")
        for node, edges in weighted_structure.items():
            print(f"  {node}: {[f'{p} ({w:.2f})' for p, w in edges]}")

        return weighted_structure

    def sample_conditionally(self, X: pd.DataFrame, missing_mask: pd.DataFrame) -> pd.DataFrame:

        """
        Sample synthetic data conditionally, filling missing values in X.
        """

        n_samples = X.shape[0]
        device = self.bgan._device
        embedding_dim = self.bgan._embedding_dim

        # Generate random noise for the generator
        noise = torch.randn(n_samples, embedding_dim, device=device)

        self.bgan._generator.eval()
        with torch.no_grad():
            gen_output = self.bgan._generator(noise)
            if isinstance(gen_output, tuple):
                gen_output = gen_output[0]  # If generator returns (data, uncertainty, ...)

        # Transform the (initially-filled) data into matrix space
        transformed = self.bgan._transformer.transform(X)
        output_np = transformed.copy()
        gen_np = gen_output.cpu().numpy()

        # Build a mask in the transformed (matrix) space that corresponds
        # to the original missing positions (passed in `missing_mask`). The
        # transformer expands each original column into multiple output
        # dimensions; we mark all output dimensions for a column as
        # replaceable where the original column was missing.
        try:
            # Accept either DataFrame or array-like missing_mask
            if isinstance(missing_mask, pd.DataFrame):
                miss_arr = missing_mask.reindex(columns=X.columns).values.astype(bool)
            else:
                miss_arr = np.asarray(missing_mask).astype(bool)
                # ensure shape (n_rows, n_cols)
                if miss_arr.ndim == 1:
                    # single-column mask -> expand
                    miss_arr = miss_arr.reshape(-1, 1)

            transformed_mask = np.zeros_like(output_np, dtype=bool)
            st = 0
            # The transformer keeps a list of ColumnTransformInfo objects
            for col_info in self.bgan._transformer._column_transform_info_list:
                dim = col_info.output_dimensions
                col_name = col_info.column_name
                # If column name is present in X, get its missing vector; else default False
                if col_name in X.columns:
                    col_idx = list(X.columns).index(col_name)
                    col_missing = miss_arr[:, col_idx]
                else:
                    col_missing = np.zeros(n_samples, dtype=bool)
                # Broadcast the per-row missing flag across the column's output dims
                transformed_mask[:, st:st + dim] = np.repeat(col_missing.reshape(-1, 1), dim, axis=1)
                st += dim

            # Replace only originally-missing positions in transformed space
            combined = np.where(transformed_mask, gen_np, output_np)
            return self.bgan._transformer.inverse_transform(combined)
        except Exception as e:
            # Fallback: if anything goes wrong, return inverse transform of
            # the generator output (best-effort) to avoid silent failures.
            try:
                combined = gen_np
                return self.bgan._transformer.inverse_transform(combined)
            except Exception:
                raise RuntimeError(f"Sampling failed in sample_conditionally: {e}")


    def fit(self, data: pd.DataFrame, discrete_columns: list):

        """
        Learn BN structure and fit the BGAN with BN integration.
        """

        # Learn BN structure and initialize BGAN
        weighted_bn = self.learn_bn_structure(data)

        self.bgan = BGANWithBN(epochs=self.epochs, batch_norm=self.batch_norm, optimizer_type=self.optimizer_type, use_uncertainty_loss=self.use_uncertainty_loss, use_kl_loss=self.use_kl_loss, bn_influence=self.bn_influence)
        self.bgan.bn_structure = weighted_bn
        self.bgan.fit(data, discrete_columns)

    def sample(self, n: int) -> pd.DataFrame:

        """
        Sample n synthetic data points from the trained model.
        """

        if self.bgan is None:
            raise RuntimeError("Model not trained. Call `fit` first.")
        return self.bgan.sample(n)

    def get_gate_log(self):

        """
        Get the gate log from the generator (for BN integration analysis).
        """

        if not self.bgan or not hasattr(self.bgan, '_generator'):
            raise AttributeError("BGAN model not initialized or trained yet.")
        
        gen = self.bgan._generator
        if hasattr(gen, 'gate_log'):
            return gen.gate_log
        else:
            raise AttributeError("Gate log not found in generator.")

    def get_bn_structure(self):
        """
        Return the learned Bayesian Network structure as a dict.
        Each key is a node, and the value is a list of (parent, weight) tuples.
        """
        if self.bn_structure is None:
            raise RuntimeError("BN structure not learned. Call fit() first.")
        return self.bn_structure

    def print_bn_structure(self):
        """
        Pretty-print the learned Bayesian Network structure.
        """
        if self.bn_structure is None:
            raise RuntimeError("BN structure not learned. Call fit() first.")
        print("Learned Bayesian Network structure (DAG):")
        for node, parents in self.bn_structure.items():
            if isinstance(parents[0], tuple):
                # Weighted edges
                print(f"  {node}: {[f'{p} ({w:.2f})' for p, w in parents]}")
            else:
                print(f"  {node}: {parents}")

        
    def plot_bn_structure(self, weighted=True):
        """
        Plot the learned Bayesian Network structure as a DAG.
        If weighted=True, edge weights are shown.
        """
        if self.bn_structure is None:
            raise RuntimeError("BN structure not learned. Call fit() first.")

        G = nx.DiGraph()
        for node, parents in self.bn_structure.items():
            if isinstance(parents[0], tuple):
                # Weighted edges
                for parent, weight in parents:
                    if weighted:
                        G.add_edge(parent, node, weight=weight)
                    else:
                        G.add_edge(parent, node)
            else:
                for parent in parents:
                    G.add_edge(parent, node)

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 7))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1200, arrowsize=20)
        if weighted and any('weight' in d for u, v, d in G.edges(data=True)):
            edge_labels = nx.get_edge_attributes(G, 'weight')
            edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Learned Bayesian Network Structure (DAG)")
        plt.tight_layout()
        plt.show()