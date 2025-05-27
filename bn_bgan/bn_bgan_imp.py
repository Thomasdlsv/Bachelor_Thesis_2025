import numpy as np
import pandas as pd

from bn_bgan.bn_bgan_sdg import BN_AUG_SDG


class BN_AUG_Imputer:

    """
    Imputer based on the BN-AUG-SDG model.
    Trains a Bayesian Network-augmented SDG on complete rows and imputes missing values
    using conditional sampling and iterative refinement.
    """

    def __init__(self, epochs=50, embedding_dim=256, batch_norm=True, random_state=42):

        """
        Initialize the imputer.

        Args:
            epochs (int): Number of training epochs for the SDG model.
            embedding_dim (int): Embedding dimension for the SDG model.
            batch_norm (bool): Whether to use batch normalization.
            random_state (int): Random seed for reproducibility.
        """

        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.batch_norm = batch_norm
        self.random_state = random_state
        self.model = None
        self.discrete_columns = []
        self.original_columns = []
        self._missing_mask = None

    def fit(self, X: pd.DataFrame):

        """
        Fit the BN-AUG-SDG model on complete rows of X.

        Args:
            X (pd.DataFrame): Input data with possible missing values.

        Returns:
            self
        """

        X = X.copy()
        self.original_columns = X.columns.tolist()
        self.discrete_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self._missing_mask = X.isnull()

        complete_rows = X.dropna()
        if complete_rows.empty:
            raise ValueError("No complete rows available to train imputer.")

        self.model = BN_AUG_SDG(epochs=self.epochs, embedding_dim=self.embedding_dim, batch_norm=self.batch_norm)
        self.model.fit(complete_rows, self.discrete_columns)
        return self

    def _initial_impute(self, X: pd.DataFrame) -> pd.DataFrame:

        """
        Perform an initial imputation using parent-based grouping and fallback to mode/median.

        Args:
            X (pd.DataFrame): Data to impute.

        Returns:
            pd.DataFrame: Data with initial imputation.
        """

        X_filled = X.copy()
        for col in X.columns:
            if col in self.model.node_importance:
                parents = self.model.node_importance[col].keys()
                if all(p in X.columns for p in parents):
                    try:
                        complete_parent_rows = X[parents + [col]].dropna()
                        if complete_parent_rows.empty:
                            raise ValueError()
                        grouped = complete_parent_rows.groupby(list(parents))[col].agg(
                            lambda x: x.mode()[0] if x.dtype == 'object' else x.median()
                        )
                        for idx in X_filled.index[X[col].isnull()]:
                            parent_vals = tuple(X.loc[idx, parents])
                            if parent_vals in grouped:
                                X_filled.loc[idx, col] = grouped[parent_vals]
                                continue
                    except:
                        pass

            if X[col].dtype == 'object' or col in self.discrete_columns:
                mode = X[col].mode(dropna=True)
                X_filled[col] = X_filled[col].fillna(mode[0] if not mode.empty else 'missing')
            else:
                median = X[col].median()
                X_filled[col] = X_filled[col].fillna(median)

        return X_filled

    def _postprocess(self, X_filled: pd.DataFrame) -> pd.DataFrame:

        """
        Postprocess imputed data (e.g., clip numeric columns to non-negative).

        Args:
            X_filled (pd.DataFrame): Imputed data.

        Returns:
            pd.DataFrame: Postprocessed data.
        """

        for col in X_filled.select_dtypes(include=np.number).columns:
            X_filled[col] = X_filled[col].clip(lower=0)
        return X_filled

    def sdg_impute(self, X: pd.DataFrame, n_iter: int = 30, refine_passes: int = 3) -> pd.DataFrame:

        """
        Impute missing values using SDG logic: multiple stochastic samples and averaging.

        Args:
            X (pd.DataFrame): Data to impute.
            n_iter (int): Number of stochastic samples per refinement pass.
            refine_passes (int): Number of refinement passes.

        Returns:
            pd.DataFrame: Imputed data.
        """

        if self.model is None:
            raise RuntimeError("Model not trained. Call `fit` first.")

        X = X.copy()
        missing_mask = X.isnull()
        X_filled = self._initial_impute(X)

        for pass_num in range(refine_passes):
            imputations = []
            for _ in range(n_iter):
                # Each sample_conditionally call generates a new stochastic imputation
                X_imp = self.model.sample_conditionally(X_filled, missing_mask)
                if not isinstance(X_imp, pd.DataFrame):
                    X_imp = pd.DataFrame(X_imp, columns=X.columns, index=X.index)
                imputations.append(X_imp)

            # Stack and average imputations for missing values
            imputations = np.stack([imp.values for imp in imputations], axis=0)  # shape: (n_iter, n_rows, n_cols)
            imputed_mean = np.nanmean(imputations, axis=0)

            # Fill only missing values with the mean, keep observed values as is
            for i, col in enumerate(X.columns):
                X_filled.loc[missing_mask[col], col] = imputed_mean[missing_mask[col].values, i]

        return self._postprocess(X_filled)

    def impute_all_missing(self, X):

        """
        Impute all missing values in X using the SDG imputer.

        Args:
            X (pd.DataFrame): Data to impute.

        Returns:
            pd.DataFrame: Imputed data.
        """

        return self.sdg_impute(X)

    def fit_transform(self, X: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:

        """
        Fit the imputer and transform the data.

        Args:
            X (pd.DataFrame): Data to fit and transform.
            max_iter (int): Not used, for compatibility.

        Returns:
            pd.DataFrame: Imputed data.
        """

        self.fit(X)
        return self.transform(X, max_iter=max_iter)

    def get_gate_log(self):

        """
        Get the gate log from the underlying BN_AUG_SDG model.

        Returns:
            Any: Gate log from the model.

        Raises:
            AttributeError: If the model has not been trained yet.
        """
        
        if self.model:
            return self.model.get_gate_log()
        else:
            raise AttributeError("Model has not been trained yet.")
