"""
Transform a csv of centroids to a list of distances between them
"""
import argparse
import sys
from typing import List

import geopandas  # type: ignore
import pandas  # type: ignore

from . import load_csv


def calculate_distances(df: geopandas.GeoDataFrame) -> pandas.DataFrame:
    """
    """

    out = df.copy()

    out.reset_index(inplace=True)
    out.drop("index", axis=1, inplace=True)
    out.index.name = "index"

    for i, row in out.iterrows():
        out[i] = out.distance(row.centroid)

    out.drop("centroid", axis=1, inplace=True)

    return out


def remove_furthest_points(
    df: geopandas.GeoDataFrame, dist: float, loc: str
) -> geopandas.GeoDataFrame:
    """
    Remove points further than `dist` from `loc`, inplace
    """

    # Find index of desired location
    center_index_list = df.index[df["region_name"] == loc].tolist()
    assert len(center_index_list) == 1
    center_index = center_index_list[0]

    out = df[df.distance(df.loc[center_index]["centroid"]) < dist]

    return out


def locations_in_range(
    centroid_data: geopandas.GeoDataFrame, dist: float, loc: str
) -> List[str]:

    shrunk = remove_furthest_points(centroid_data, dist, loc)

    out = shrunk.region_code.tolist()

    return out


def parse_args(args):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(args)

    parser.add_argument(
        "file_name", help="A csv file from Stats NZ containing locations of centroids",
    )
    parser.add_argument(
        "out", help="Where to write a CSV with output",
    )
    parser.add_argument(
        "-d",
        nargs=2,
        metavar=("dist", "loc"),
        help="Limit to areas less than dist from loc",
    )

    args = parser.parse_args()

    return args


def main(argv: List[str]) -> None:
    """
    The main function
    """
    args: argparse.Namespace = parse_args(argv)

    print("Loading ", args.file_name)

    centroid_data = load_csv.load_centroid_data_2018(args.file_name)

    if args.d is not None:
        shrunk = remove_furthest_points(centroid_data, int(args.d[0]), str(args.d[1]))
    else:
        shrunk = centroid_data

    distance_table = calculate_distances(shrunk)

    # output save file
    distance_table.to_csv(args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
