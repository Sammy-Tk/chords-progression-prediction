from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# from taxifare.ml_logic.encoders import (transform_time_features,
#                                               transform_lonlat_features,
#                                               compute_geohash)

import numpy as np
import pandas as pd

from colorama import Fore, Style


def preprocess_features(X: pd.DataFrame) -> np.ndarray:




    print(Fore.BLUE + "\nPreprocess features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()

    X_processed = preprocessor.fit_transform(X)

    print("\n✅ X_processed, with shape", X_processed.shape)

    return X_processed

def preprocess(source_type = 'train'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    print("\n⭐️ Use case: preprocess")



    X_processed_chunk = preprocess_features(X_chunk)



    print(f"\n✅ Data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None
