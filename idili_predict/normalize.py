"""Feature normalization using MAD robustize via pycytominer."""

import numpy as np
import pandas as pd
from pycytominer import normalize
from tqdm import tqdm


def normalize_plate_mad(plate_data, feature_cols, metadata_cols):
    """MAD-robustize features for a single plate using pycytominer.

    For each feature: (x - median) / MAD, where median and MAD are computed
    from all wells on the plate.

    Args:
        plate_data: DataFrame containing one plate of data.
        feature_cols: List of feature column names to normalize.
        metadata_cols: List of metadata column names to preserve.

    Returns:
        DataFrame with normalized feature values.
    """
    plate_data = plate_data.copy()
    present_meta = [c for c in metadata_cols if c in plate_data.columns]

    normalized = normalize(
        profiles=plate_data,
        features=feature_cols,
        meta_features=present_meta,
        samples="all",
        method="mad_robustize",
        output_file=None,
    )

    return normalized


def normalize_all_plates(df, feature_cols, metadata_cols, plate_col="Metadata_PlateID"):
    """Apply MAD robustize normalization to each plate independently.

    After normalization, replaces NaN and Inf values with 0 (MAD robustize
    produces Inf when MAD=0 for constant features).

    Args:
        df: Full DataFrame with all plates.
        feature_cols: List of feature column names.
        metadata_cols: List of metadata column names.
        plate_col: Column identifying plate membership.

    Returns:
        DataFrame with normalized features across all plates.
    """
    plates = df[plate_col].unique()
    normalized_plates = []

    for plate in tqdm(plates, desc="  Normalizing plates"):
        plate_data = df[df[plate_col] == plate].copy()
        normalized_plate = normalize_plate_mad(plate_data, feature_cols, metadata_cols)
        normalized_plates.append(normalized_plate)

    df_normalized = pd.concat(normalized_plates, ignore_index=True)

    # Clean up NaN/Inf from MAD=0 features
    n_nan = df_normalized[feature_cols].isna().sum().sum()
    n_inf = np.isinf(df_normalized[feature_cols].values).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  Replacing {n_nan} NaN and {n_inf} Inf values with 0")
        df_normalized[feature_cols] = (
            df_normalized[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

    return df_normalized
