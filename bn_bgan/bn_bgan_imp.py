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
        
        # First check if we have any missing values
        missing_cols = X.columns[X.isnull().any()].tolist()
        if not missing_cols:
            print("No missing values found in initial imputation")
            return X_filled
            
        print(f"Performing initial imputation for columns: {missing_cols}")
        
        for col in missing_cols:
            print(f"Imputing column: {col}")
            if col in self.model.node_importance:
                parents = self.model.node_importance[col].keys()
                if all(p in X.columns for p in parents):
                    try:
                        complete_parent_rows = X[parents + [col]].dropna()
                        if complete_parent_rows.empty:
                            raise ValueError(f"No complete rows found for {col} and its parents")
                        grouped = complete_parent_rows.groupby(list(parents))[col].agg(
                            lambda x: x.mode()[0] if x.dtype == 'object' else x.median()
                        )
                        missing_idx = X_filled.index[X[col].isnull()]
                        for idx in missing_idx:
                            parent_vals = tuple(X.loc[idx, parents])
                            if parent_vals in grouped:
                                X_filled.loc[idx, col] = grouped[parent_vals]
                                print(f"Parent-based imputation successful for {len(missing_idx)} values in {col}")
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
            
        # Verify we have missing values to impute
        missing_mask = X.isnull()
        if not missing_mask.any().any():
            print("Warning: No missing values found in input data")
            return X.copy()

        X = X.copy()
        missing_mask = X.isnull()
        
        # Early return if no missing values
        if not missing_mask.any().any():
            print("Warning: No missing values found in input data")
            return X

        X_filled = self._initial_impute(X)
        
        # Keep track of which values were originally missing
        original_missing = missing_mask.copy()

        for pass_num in range(refine_passes):
            imputations = []
            for _ in range(n_iter):
                try:
                    # Each sample_conditionally call generates a new stochastic imputation
                    X_imp = self.model.sample_conditionally(X_filled, original_missing)
                    if not isinstance(X_imp, pd.DataFrame):
                        X_imp = pd.DataFrame(X_imp, columns=X.columns, index=X.index)
                    imputations.append(X_imp)
                except Exception as e:
                    print(f"Warning: Sampling iteration failed: {str(e)}")
                    continue

            if not imputations:
                print("Warning: All sampling iterations failed")
                continue

            # Stack and average imputations for missing values
            try:
                imputations = np.stack([imp.values for imp in imputations], axis=0)  # shape: (n_iter, n_rows, n_cols)
                imputed_mean = np.nanmean(imputations, axis=0)

                # Fill only missing values with the mean, keep observed values as is
                for i, col in enumerate(X.columns):
                    if original_missing[col].any():  # Only update if column had missing values
                        missing_idx = original_missing[col]
                        if pd.api.types.is_numeric_dtype(X[col].dtype):
                            X_filled.loc[missing_idx, col] = imputed_mean[missing_idx.values, i]
                        else:
                            # For categorical columns, use mode of imputations
                            imp_vals = np.array([imp[missing_idx.values, i] for imp in imputations])
                            modes = [pd.Series(imp_vals[:, j]).mode()[0] for j in range(imp_vals.shape[1])]
                            X_filled.loc[missing_idx, col] = modes
            except Exception as e:
                print(f"Warning: Failed to update imputations in pass {pass_num}: {str(e)}")
                continue

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

    def transform(self, X, max_iter=10):
        """
        Transform method for imputing missing values.
        
        Args:
            X: Input data with missing values
            max_iter: Maximum iterations (not used, for compatibility)
            
        Returns:
            Imputed DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.original_columns)
            
        # Check for missing values
        missing_mask = X.isnull()
        if not missing_mask.any().any():
            print("Warning: No missing values found in transform input")
            return X.copy()
            
        print(f"Transforming data with {missing_mask.sum().sum()} missing values")
        return self.impute_all_missing(X)

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
