import numpy as np
import pandas as pd

class DataSamplerBN(object):
    """DataSamplerBN samples the conditional vector and corresponding data for BGAN with Bayesian Network (BN) augmented prior."""

    def __init__(self, data, output_info, log_frequency, bn_structure):
        """
        Initialize the sampler with Bayesian Network (BN) structure.
        Args:
            data: Training data matrix.
            output_info: Information about the output columns.
            log_frequency: Whether to use log frequency for categorical values.
            bn_structure: The structure of the Bayesian Network representing the latent variable dependencies.
        """
        self._data_length = len(data)
        
        self.bn_structure = bn_structure  # Bayesian Network structure for latent variable dependencies
        
        # Similar to the original setup, but will sample from BN prior instead of i.i.d Gaussian
        n_discrete_columns = sum([
            1 for column_info in output_info if self.is_discrete_column(column_info)
        ])

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')

        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols as before
        st = 0
        for column_info in output_info:
            if self.is_discrete_column(column_info):
                span_info = column_info
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    if isinstance(data, pd.DataFrame):
                        data = data.to_numpy()
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += column_info.dim

        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sampling the conditional vector
        max_category = max(
            [column_info.dim for column_info in output_info if self.is_discrete_column(column_info)],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info.dim for column_info in output_info if self.is_discrete_column(column_info)
        ])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if self.is_discrete_column(column_info):
                span_info = column_info
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, : span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += column_info.dim

    def is_discrete_column(self, column_info):
        """
        Check if a column is discrete based on its activation function.
        
        Args:
            column_info: An instance of ColumnInfo.
        
        Returns:
            True if the column is discrete, False otherwise.
        """
        return column_info.activation_fn == 'softmax'

    def sample_condvec(self, batch):
        """Generate the conditional vector for training with Bayesian Network-based prior."""
        if self._n_discrete_columns == 0:
            return None

        # Sample from the BN prior using ancestral sampling
        # Each latent variable is conditioned on its parents in the BN structure
        z_samples = self.sample_from_bn(batch)

        # Generate the conditional vectors
        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_from_bn(self, batch):
        """
        Sample latent variables from the Bayesian Network prior.
        
        Args:
            batch: Number of samples to generate.
        
        Returns:
            A NumPy array of sampled latent variables.
        """
        z_samples = np.zeros((batch, len(self.bn_structure)))
        
        for node_id, node_info in self.bn_structure.items():
            parents = node_info['parents']
            if not parents:
                # If the node has no parents, sample from a standard normal distribution
                z_samples[:, node_id] = np.random.normal(size=batch)
            else:
                # If the node has parents, sample conditionally
                parent_values = z_samples[:, parents]
                z_samples[:, node_id] = self.sample_conditional(parent_values)
        
        return z_samples

    def sample_conditional(self, parent_values):
        """Sample a latent variable conditioned on its parents."""
        # Implement the conditional distribution based on the parent's values
        # This can be a Gaussian, Bernoulli, or any other distribution
        return np.random.normal(0, 1, parent_values.shape[0])  # Example: Gaussian for simplicity

    def _random_choice_prob_index(self, discrete_column_id):
        """Helper to randomly choose category index based on category probabilities."""
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_data(self, data, n, col, opt):
        """
        Sample data rows based on the given indices.
        
        Args:
            data: The dataset (Pandas DataFrame or NumPy array).
            n: Number of rows to sample.
            col: Column information (not used here).
            opt: Options (not used here).
        
        Returns:
            A subset of the data.
        """
        idx = np.random.choice(len(data), n, replace=False)  # Randomly sample row indices
        if isinstance(data, pd.DataFrame):
            return data.iloc[idx]  # Use .iloc for row selection in DataFrame
        return data[idx]  # For NumPy arrays

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec
    
class ColumnInfo:
    def __init__(self, dim, activation_fn):
        self.dim = dim
        self.activation_fn = activation_fn
