"""
Restrict a csv of telco data to some regions
"""
import argparse
import sys
from typing import List

import pandas  # type: ignore

import distances
import load_csv


def parse_args(args):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(args)

    parser.add_argument(
        "telco_file_name", help="A csv file of location data",
    )
    parser.add_argument(
        "location_file_name",
        help="A csv file from Stats NZ containing locations of centroids",
    )
    parser.add_argument(
        "out", help="Where to write a CSV with output",
    )
    parser.add_argument(
        "dist", help="Limit to areas less than dist from loc",
    )
    parser.add_argument(
        "loc", help="Limit to areas less than dist from loc",
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
    telco = pandas.read_csv(args.telco_file_name, dtype={"sa2_2018_code": str})

    telco[telco["sa2_2018_code"].isin(codes)].to_csv(args.out, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
