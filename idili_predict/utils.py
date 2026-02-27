"""Shared data utilities for iDILI-Predict pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_column_types(dtypes_file):
    """Load column_dtypes.csv and return (feature_cols, metadata_cols) lists.

    The file must have two columns: one for column names and one for types
    ('feature' or 'metadata'). Accepts both 'ColumnName'/'ColumnType' and
    'Column_Name'/'Column_Type' header formats.
    """
    dtypes_file = Path(dtypes_file)
    if not dtypes_file.exists():
        raise FileNotFoundError(f"Column dtypes file not found: {dtypes_file}")

    dtypes_df = pd.read_csv(dtypes_file)

    name_col = "ColumnName" if "ColumnName" in dtypes_df.columns else "Column_Name"
    type_col = "ColumnType" if "ColumnType" in dtypes_df.columns else "Column_Type"

    if name_col not in dtypes_df.columns or type_col not in dtypes_df.columns:
        raise ValueError(
            f"Expected columns 'ColumnName'/'ColumnType' or 'Column_Name'/'Column_Type' "
            f"in {dtypes_file}. Found: {list(dtypes_df.columns)}"
        )

    feature_cols = dtypes_df.loc[dtypes_df[type_col] == "feature", name_col].tolist()
    metadata_cols = dtypes_df.loc[dtypes_df[type_col] == "metadata", name_col].tolist()

    return feature_cols, metadata_cols


def validate_required_columns(df, required_cols):
    """Raise ValueError if any required columns are missing from df."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns include: {list(df.columns[:10])}..."
        )


def filter_to_present_numeric(df, feature_cols):
    """Filter feature_cols to those present in df and numeric."""
    present = [c for c in feature_cols if c in df.columns]
    numeric = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
    n_dropped = len(present) - len(numeric)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} non-numeric feature columns")
    return numeric


def encode_labels(df, cond_col="Metadata_COND", pos_label="PC", neg_label="NC",
                  cmpd_col="Metadata_CMPD", dmso_compound="DMSO"):
    """Add binary 'Metadata_Label_Encoded' column (PC=1, NC=0, DMSO=0).

    Returns the dataframe with the new column added. Rows with unmapped
    condition values get NaN.
    """
    df = df.copy()
    df["Metadata_Label_Encoded"] = df[cond_col].map({neg_label: 0, pos_label: 1})

    if cmpd_col in df.columns:
        dmso_mask = df[cmpd_col] == dmso_compound
        df.loc[dmso_mask, "Metadata_Label_Encoded"] = 0
        n_dmso = dmso_mask.sum()
        if n_dmso > 0:
            print(f"  DMSO controls labeled as NC: {n_dmso} wells")

    return df


def impute_plate_medians(df, feature_cols, plate_col="Metadata_PlateID"):
    """Impute missing feature values with per-plate medians, then global medians."""
    df = df.copy()
    nan_total = df[feature_cols].isna().sum().sum()
    if nan_total == 0:
        return df

    n_cols = (df[feature_cols].isna().sum() > 0).sum()
    print(f"  Imputing {nan_total} missing values across {n_cols} columns...")

    for plate in df[plate_col].unique():
        mask = df[plate_col] == plate
        plate_medians = df.loc[mask, feature_cols].median()
        df.loc[mask, feature_cols] = df.loc[mask, feature_cols].fillna(plate_medians)

    remaining = df[feature_cols].isna().sum().sum()
    if remaining > 0:
        global_medians = df[feature_cols].median()
        df[feature_cols] = df[feature_cols].fillna(global_medians)

    print(f"  Missing after imputation: {df[feature_cols].isna().sum().sum()}")
    return df
