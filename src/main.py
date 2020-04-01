import load_csv
import pandas as pd  # type: ignore


def main():
    area_data = load_csv.load_area_data_2018(
        "../data/area/statistical-area-2-2018-generalised.csv"
    )

    telco_data = load_csv.load_telco_data(
        "../data/telco/pop_data_2020-04-01.dat"
    )

    data = pd.merge(area_data, telco_data, on="region_code")

    print(data)


if __name__ == "__main__":
    main()
