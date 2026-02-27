"""iDILI-Predict analysis pipeline: load -> impute -> normalize -> train -> evaluate."""

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from idili_predict.normalize import normalize_all_plates
from idili_predict.utils import (
    encode_labels,
    filter_to_present_numeric,
    impute_plate_medians,
    load_column_types,
    validate_required_columns,
)


def run_pipeline(
    input_file,
    dtypes_file,
    output_dir,
    *,
    plate_col="Metadata_PlateID",
    well_col="Metadata_WellID",
    cmpd_col="Metadata_CMPD",
    conc_col="Metadata_CONC",
    cond_col="Metadata_COND",
    dmso_compound="DMSO",
    positive_label="PC",
    negative_label="NC",
    time_limit=2000,
    presets="best_quality",
    num_bag_folds=5,
):
    """Run the full iDILI-Predict analysis pipeline for a single input CSV.

    Pipeline steps:
        1. Load data and validate required columns
        2. Impute missing values (per-plate medians)
        3. Normalize features (MAD robustize per plate)
        4. Train AutoGluon with compound-grouped cross-validation
        5. Export OOF predictions, leaderboard, and metrics

    Args:
        input_file: Path to input CSV with well-level CellProfiler features.
        dtypes_file: Path to column_dtypes.csv defining feature vs metadata columns.
        output_dir: Directory for all output files.
        plate_col: Column name for plate identifier.
        well_col: Column name for well identifier.
        cmpd_col: Column name for compound name.
        conc_col: Column name for concentration.
        cond_col: Column name for condition (PC/NC).
        dmso_compound: Value in cmpd_col identifying DMSO controls.
        positive_label: Value in cond_col for DILI-positive.
        negative_label: Value in cond_col for DILI-negative.
        time_limit: AutoGluon training time limit in seconds.
        presets: AutoGluon presets string.
        num_bag_folds: Number of bagging folds for OOF predictions.

    Returns:
        dict with keys 'auc', 'accuracy', 'f1_positive', 'f1_negative',
        'oof_predictions_file', 'leaderboard_file'.
    """
    input_file = Path(input_file)
    dtypes_file = Path(dtypes_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"iDILI-Predict Pipeline")
    print(f"  Input:  {input_file.name}")
    print(f"  Output: {output_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading {input_file.name}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Shape: {df.shape}")

    # Remove duplicate columns (e.g., Metadata_Plate.1)
    dup_cols = [c for c in df.columns if c.endswith(".1")]
    if dup_cols:
        print(f"  Removing {len(dup_cols)} duplicate columns")
        df = df.drop(columns=dup_cols)

    required_cols = [plate_col, well_col, cmpd_col, conc_col, cond_col]
    validate_required_columns(df, required_cols)

    print(f"  Conditions: {df[cond_col].value_counts().to_dict()}")

    # Encode labels
    df = encode_labels(
        df, cond_col=cond_col, pos_label=positive_label,
        neg_label=negative_label, cmpd_col=cmpd_col, dmso_compound=dmso_compound,
    )

    # Load and filter feature columns
    feature_cols_all, metadata_cols_all = load_column_types(dtypes_file)
    feature_cols = filter_to_present_numeric(df, feature_cols_all)
    metadata_cols = [c for c in metadata_cols_all if c in df.columns]
    metadata_cols = list(set(metadata_cols + ["Metadata_Label_Encoded"]))

    print(f"  Feature columns: {len(feature_cols)}")

    # Copy dtypes file to output
    shutil.copyfile(dtypes_file, output_dir / "column_dtypes.csv")

    # Checkpoint 1
    df.to_csv(output_dir / "01_raw_data_with_labels.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Impute missing values
    # ------------------------------------------------------------------
    print("\n[2/5] Checking for missing values...")
    df = impute_plate_medians(df, feature_cols, plate_col=plate_col)

    # ------------------------------------------------------------------
    # 3. Normalize features (plate-based MAD robustize)
    # ------------------------------------------------------------------
    print("\n[3/5] Normalizing features (MAD robustize)...")
    df_normalized = normalize_all_plates(
        df, feature_cols, metadata_cols, plate_col=plate_col,
    )

    # Checkpoint 2
    df_normalized.to_csv(output_dir / "02_normalized_features.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Train AutoGluon
    # ------------------------------------------------------------------
    print(f"\n[4/5] Training AutoGluon (time_limit={time_limit}s, "
          f"presets={presets}, folds={num_bag_folds})...")

    from autogluon.tabular import TabularPredictor
    from sklearn.model_selection import GroupKFold

    df_for_ag = df_normalized.copy()
    df_for_ag["log_Metadata_CONC"] = np.log(df_for_ag[conc_col].replace(0, 0.01))

    ag_feature_cols = feature_cols + ["log_Metadata_CONC"]
    target_col = "Metadata_Label_Encoded"

    labeled_mask = df_for_ag[target_col].isin([0, 1])
    train_df = df_for_ag.loc[labeled_mask].copy()

    print(f"  Training rows: {len(train_df)}")
    print(f"  Unique compounds: {train_df[cmpd_col].nunique()}")
    print(f"  Classes: {train_df[target_col].value_counts().to_dict()}")

    # Compound-grouped fold assignments
    gkf = GroupKFold(n_splits=num_bag_folds)
    fold_assignments = np.zeros(len(train_df), dtype=int)
    for fold_id, (_, val_idx) in enumerate(
        gkf.split(train_df, groups=train_df[cmpd_col])
    ):
        fold_assignments[val_idx] = fold_id
    train_df["AG_fold"] = fold_assignments

    # Sample weights for class balancing
    class_counts = train_df[target_col].value_counts()
    n_samples = len(train_df)
    n_classes = len(class_counts)
    weight_map = {
        cls: n_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }
    train_df["sample_weight"] = train_df[target_col].map(weight_map)

    train_data = train_df[ag_feature_cols + [target_col, "AG_fold", "sample_weight"]]

    model_dir = output_dir / "autogluon_model"
    predictor = TabularPredictor(
        label=target_col,
        eval_metric="roc_auc",
        problem_type="binary",
        path=str(model_dir),
        groups="AG_fold",
        sample_weight="sample_weight",
    )

    predictor.fit(
        train_data=train_data,
        presets=presets,
        time_limit=time_limit,
        verbosity=2,
        num_bag_folds=num_bag_folds,
    )

    # ------------------------------------------------------------------
    # 5. OOF evaluation and export
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating and exporting OOF predictions...")

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    oof_proba = predictor.predict_proba_oof()
    oof_preds = predictor.predict_oof()

    positive_col_name = 1 if 1 in oof_proba.columns else oof_proba.columns[-1]
    prob_pos = oof_proba[positive_col_name]
    y_true = train_data[target_col].loc[oof_proba.index]

    auc = roc_auc_score(y_true, prob_pos)
    acc = accuracy_score(y_true, oof_preds)
    f1_pos = f1_score(y_true, oof_preds, pos_label=1)
    f1_neg = f1_score(y_true, oof_preds, pos_label=0)

    print(f"  ROC AUC:       {auc:.4f}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  F1 (positive): {f1_pos:.4f}")
    print(f"  F1 (negative): {f1_neg:.4f}")

    # Save leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard_file = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_file, index=False)

    # Build and save OOF predictions
    oof_df = train_df[
        [plate_col, well_col, cmpd_col, conc_col, cond_col, target_col]
    ].copy()
    oof_df["Predicted_Prob_Positive"] = prob_pos.values
    oof_df["Predicted_Label"] = (prob_pos.values >= 0.5).astype(int)
    oof_df["Actual_Label"] = train_df[target_col].values

    oof_file = output_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_file, index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {input_file.name}")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Output:  {output_dir}")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"    {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return {
        "auc": auc,
        "accuracy": acc,
        "f1_positive": f1_pos,
        "f1_negative": f1_neg,
        "oof_predictions_file": str(oof_file),
        "leaderboard_file": str(leaderboard_file),
    }
