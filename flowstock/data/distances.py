"""
Transform a csv of centroids to a list of distances between them
"""
import argparse
import sys
from typing import List

import numpy as np  # type: ignore
import pandas  # type: ignore

import geopandas  # type: ignore

from . import load_csv


class AreaSubset:
    """
    Representation of a subset of all SA2 regions
    """

    def __init__(
        self,
        centroid_file_name="data/area/statistical-area-2-2018-centroid-true.csv",
        hierarchy_file_name="data/area/statistical-area-2-higher-geographies-2018-generalised.csv",  # noqa
    ):
        """
        Load data for all SA2 regions

        Other methods can be used to remove regions by various criteria.
        """

        centroid_data = load_csv.load_centroid_data_2018(centroid_file_name)
        hierarchy_data = load_csv.load_hierarchy_data_2018(hierarchy_file_name)

        self.data: geopandas.GeoDataFrame = centroid_data.merge(
            hierarchy_data, on=["sa2_code", "sa2_name"]
        )
        self.reset_index()

    def filter_radius(
        self,
        loc: str,
        dist: float,
    ) -> geopandas.GeoDataFrame:
        """
        Remove points further than `dist` from `loc`, inplace
        """

        df = self.data

        # Find index of desired location
        center_index_list = df.index[df["sa2_name"] == loc].tolist()
        assert len(center_index_list) == 1

        center_index = center_index_list[0]

        self.data = df[df.distance(df.loc[center_index]["centroid"]) < dist]

    def filter_rc_code(self, codes: List[str]):
        """
        Filter to only regional council codes in `codes`

        Note that the codes should be strings, not integers.
        """

        self.data = self.data[self.data["rc_code"].isin(codes)].copy()
        self.reset_index()

    def filter_rc_name(self, names: List[str]):
        """
        Filter to only regional council names in `names`
        """

        self.data = self.data[self.data["rc_name"].isin(names)].copy()
        self.reset_index()

    def filter_ta_code(self, codes: List[str]):
        """
        Filter to only territorial authority codes in `codes`

        Note that the codes should be strings, not integers.
        """

        self.data = self.data[self.data["ta_code"].isin(codes)].copy()
        self.reset_index()

    def filter_ta_name(self, names: List[str]):
        """
        Filter to only territorial authority names in `names`
        """

        self.data = self.data[self.data["ta_name"].isin(names)].copy()
        self.reset_index()

    def filter_sa2_code(self, codes: List[str]):
        """
        Filter to only SA2 codes in `codes`

        Note that the codes should be strings, not integers.
        """

        self.data = self.data[self.data["sa2_code"].isin(codes)].copy()
        self.reset_index()

    def filter_sa2_name(self, names: List[str]):
        """
        Filter to only SA2 names in `names`
        """

        self.data = self.data[self.data["sa2_name"].isin(names)].copy()
        self.reset_index()

    def remove_sa2(self, codes: List[str]):
        """
        Filter to only SA2 codes not in `codes`

        Note that the codes should be strings, not integers.
        """

        self.data = self.data[~self.data["sa2_code"].isin(codes)].copy()
        self.reset_index()

    def reset_index(self):
        self.data.reset_index(inplace=True)
        self.data.sort_values(by="sa2_code", inplace=True)
        self.data.drop("index", axis=1, inplace=True)
        self.data.index.name = "index"

    def sa2_codes(self) -> List[str]:

        out = self.data.sa2_code.tolist()

        return out

    def sa2_names(self) -> List[str]:

        out = self.data.sa2_name.tolist()

        return out

    def distance_table(self, units="km") -> pandas.DataFrame:
        """
        Calculate a table of distances between centroids
        """

        n = len(self.data.index)
        out = np.empty((n, n))

        for i, row in self.data.iterrows():
            out[i] = self.data.distance(row.centroid).astype(float)

        if units == "km":
            out /= 1000
        else:
            raise ValueError("Unknown length unit")

        return out


def parse_args(args):
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(args)

    parser.add_argument(
        "centroid_file_name",
        help="A csv file from Stats NZ containing locations of centroids",
    )
    parser.add_argument(
        "hierarchy_file_name",
        help="A csv file from Stats NZ containing hierarchical data about SA2 regions",
    )
    parser.add_argument(
        "out",
        help="Where to write a CSV with output",
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

    areas = AreaSubset(args.centroid_file_name, args.hierarchy_file_name)

    if args.d is not None:
        areas.filter_radius(str(args.d[1]), int(args.d[0]))

    distance_table = areas.distance_table()

    # output save file
    distance_table.to_csv(args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
