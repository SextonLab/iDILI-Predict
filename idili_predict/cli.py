"""Command-line interface for iDILI-Predict."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="iDILI-Predict: Morphological profiling for DILI prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run with example data:
  python -m idili_predict \\
      --input example_data/63D_well_median_fixed.csv \\
      --dtypes example_data/column_dtypes.csv

  # Custom settings:
  python -m idili_predict \\
      --input my_data.csv \\
      --dtypes my_column_dtypes.csv \\
      --output-dir results/ \\
      --time-limit 3600
""",
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="Input CSV file with well-level CellProfiler features.",
    )
    parser.add_argument(
        "--dtypes", "-d", required=True,
        help="Path to column_dtypes.csv defining feature vs metadata columns.",
    )
    parser.add_argument(
        "--output-dir", "-o", default="results",
        help="Output directory (default: results/).",
    )
    parser.add_argument(
        "--time-limit", "-t", type=int, default=2000,
        help="AutoGluon training time limit in seconds (default: 2000).",
    )
    parser.add_argument(
        "--presets", default="best_quality",
        help="AutoGluon presets (default: best_quality).",
    )
    parser.add_argument(
        "--num-bag-folds", type=int, default=5,
        help="Number of bagging folds for cross-validation (default: 5).",
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    dtypes_file = Path(args.dtypes)

    if not input_file.exists():
        print(f"Error: input file not found: {input_file}")
        sys.exit(1)
    if not dtypes_file.exists():
        print(f"Error: dtypes file not found: {dtypes_file}")
        sys.exit(1)

    from idili_predict.pipeline import run_pipeline

    run_pipeline(
        input_file=input_file,
        dtypes_file=dtypes_file,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        presets=args.presets,
        num_bag_folds=args.num_bag_folds,
    )
