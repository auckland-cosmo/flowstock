"""
Restrict a csv of telco data to some regions
"""
import argparse
import sys
from typing import List

import distances  # type: ignore
import load_csv  # type: ignore

import pandas as pd  # type: ignore  # NOQA


def parse_args(args):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(args)

    parser.add_argument(
        "telco_file_name",
        help="A csv file of location data",
    )
    parser.add_argument(
        "location_file_name",
        help="A csv file from Stats NZ containing locations of centroids",
    )
    parser.add_argument(
        "out",
        help="Where to write a CSV with output",
    )
    parser.add_argument(
        "dist",
        help="Limit to areas less than dist from loc",
    )
    parser.add_argument(
        "loc",
        help="Limit to areas less than dist from loc",
    )

    args = parser.parse_args()

    args = parser.parse_args()

    return args


def main(argv: List[str]) -> None:
    """
    The main function
    """
    args: argparse.Namespace = parse_args(argv)

    print("Loading ", args.location_file_name)
    centroid_data = load_csv.load_centroid_data_2018(args.location_file_name)

    print("Finding regions")
    codes = distances.locations_in_range(centroid_data, int(args.dist), str(args.loc))

    print("Filtering telco data by region")
    telco = load_csv.load_telco_data(args.telco_file_name)

    telco = pd.pivot_table(
        telco[telco["region_code"].isin(codes)],
        index="time",
        columns="region_code",
        values="count",
    )

    telco.to_csv(args.out, index=True)


if __name__ == "__main__":
    main(sys.argv[1:])
