import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def handmade_features(df_metadata):
    df = df_metadata.copy()
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    # df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    # df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2
    )
    # df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    # df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    #df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]

    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    # df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df[
        "tbp_lv_eccentricity"]) / 3
    # df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df[
        "tbp_lv_deltaLBnorm"]
    # df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    # df["std_dev_contrast"] = np.sqrt(
    #     (df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df[
        "tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df[
        "tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

    new_num_cols = [
        "lesion_size_ratio", "hue_contrast",
        "lesion_color_difference",
        "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
        "lesion_visibility_score", "color_consistency",

        "size_age_interaction", "lesion_severity_index",
        "color_contrast_index",
        "normalized_lesion_size", "mean_hue_difference",
        "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
        "symmetry_perimeter_interaction", "comprehensive_lesion_index",
    ]
    new_cat_cols = []

    return df, new_num_cols, new_cat_cols


def grouping_agg(df_metadata, grouping, agg, agg_cols):
    df_metadata = df_metadata.copy()

    column_names = ["_".join(grouping) + "_" + agg_col + "_" + agg for agg_col in agg_cols]
    if agg == "mean":
        df_agg = (
            df_metadata
            .groupby(grouping)[agg_cols]
            .mean()
        )
    elif agg == "median":
        df_agg = (
            df_metadata
            .groupby(grouping)[agg_cols]
            .median()
        )
    elif agg == "std":
        df_agg = (
            df_metadata
            .groupby(grouping)[agg_cols]
            .std()
        )
    elif agg == "min":
        df_agg = (
            df_metadata
            .groupby(grouping)[agg_cols]
            .min()
        )
    elif agg == "max":
        df_agg = (
            df_metadata
            .groupby(grouping)[agg_cols]
            .max()
        )
    elif agg == "max_min":
        df_max = (
            df_metadata
            .groupby(grouping)[agg_cols]
            .max()
        )
        df_min = (
            df_metadata
            .groupby(grouping)[agg_cols]
            .min()
        )
        df_agg = df_max - df_min
    else:
        raise Exception(f"unknown agg: {agg}")

    df_agg.columns = column_names
    df_agg = df_agg.reset_index()
    return df_agg, column_names


def encode_categorical(
    df_metadata,
    cat_cols,
    categorical_encoder,
):
    if categorical_encoder is None:
        categorical_encoder = OrdinalEncoder(
            categories="auto",
            dtype=int,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        X_cat = categorical_encoder.fit_transform(df_metadata[cat_cols]) + 2
    else:
        X_cat = categorical_encoder.transform(df_metadata[cat_cols]) + 2

    for c, cat_col in enumerate(cat_cols):
        df_metadata[cat_col] = X_cat[:, c]
    return df_metadata, categorical_encoder


class FeatureEngineer:
    def __init__(
        self,
        df_metadata: pd.DataFrame,
        num_cols: list[str],
        cat_cols: list[str],
        groupings: list[list[str]],
        aggs: list[str],
        categorical_encoder: OrdinalEncoder | None,
    ):
        self.df_metadata = df_metadata
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.groupings = groupings
        self.aggs = aggs

        self.categorical_encoder = categorical_encoder

    def main(self):
        self.add_handmade_features()
        self.grouping_process()
        self.encode_categorical_features()

    def add_handmade_features(self):
        df, new_num_cols, new_cat_cols = handmade_features(self.df_metadata)

        self.df_metadata = df
        self.num_cols += new_num_cols
        self.cat_cols += new_cat_cols

    def encode_categorical_features(self):
        df_metadata, categorical_encoder = encode_categorical(
            self.df_metadata,
            self.cat_cols,
            categorical_encoder=self.categorical_encoder,
        )
        self.df_metadata = df_metadata
        self.categorical_encoder = categorical_encoder

    def grouping_process(self):
        new_num_cols = []
        for grouping in self.groupings:

            column_name = self.grouping_cnt(grouping)
            new_num_cols.append(column_name)

            for agg in self.aggs:
                new_cols = self.grouping_agg(
                    grouping=grouping,
                    agg=agg,
                    agg_cols=self.num_cols,
                )
                if grouping not in [["attribution"]]:
                    new_num_cols += new_cols

            new_cols = self.grouping_normalize(
                grouping=grouping,
                columns=self.num_cols,
            )
            new_num_cols += new_cols

        self.num_cols += new_num_cols

    def grouping_cnt(self, grouping):
        df_metadata = self.df_metadata.copy()

        column_name = "_".join(grouping) + "_" + "cnt"
        sr_agg = (
            df_metadata
            .groupby(grouping)["isic_id"]
            .count()
            .rename(column_name)
        )
        df_agg = sr_agg.reset_index()
        df_metadata = df_metadata.merge(
            df_agg,
            on=grouping,
            how="left",
            validate="many_to_one"
        )
        self.df_metadata = df_metadata
        return column_name

    def grouping_agg(self, grouping, agg, agg_cols):
        df_metadata = self.df_metadata.copy()
        df_agg, column_names = grouping_agg(
            df_metadata=df_metadata,
            grouping=grouping,
            agg=agg,
            agg_cols=agg_cols
        )
        df_metadata = df_metadata.merge(
            df_agg,
            on=grouping,
            how="left",
            validate="many_to_one"
        )
        self.df_metadata = df_metadata
        return column_names

    def diff_features(
        self,
        columns_1,
        columns_2
    ):
        df_metadata = self.df_metadata.copy()

        column_names = [term1 + "_minus_" + term2 for term1, term2 in zip(columns_1, columns_2)]
        df_metadata[column_names] = df_metadata[columns_1].values - df_metadata[columns_2].values

        self.df_metadata = df_metadata
        return column_names

    def grouping_normalize(
        self,
        grouping,
        columns
    ):
        df_metadata = self.df_metadata.copy()
        mean_column_names = ["_".join(grouping) + "_" + col + "_" + "mean" for col in columns]
        std_column_names = ["_".join(grouping) + "_" + col + "_" + "std" for col in columns]
        column_names = ["_".join(grouping) + "_" + col + "_" + "normalize" for col in columns]

        df_mean = df_metadata[mean_column_names]
        df_std = df_metadata[std_column_names]

        df_metadata[column_names] = (df_metadata[columns].values - df_mean.values) / (df_std.values + 1e-8)

        self.df_metadata = df_metadata
        return column_names

    def grouping_frequency_encoding(
        self,
        grouping,
        cat_col,
    ):
        df_metadata = self.df_metadata.copy()

        column_name = f"grouping_{'_'.join(grouping)}_frequency_encoding_{cat_col}"
        df_encoded = (
            df_metadata
            .groupby(grouping)[cat_col]
            .value_counts()
            .reset_index()
            .rename(columns={"count": column_name})
        )
        df_metadata = df_metadata.merge(
            df_encoded,
            on=grouping + [cat_col],
            how="left",
            validate="many_to_one",
        )
        self.df_metadata = df_metadata
        return column_name
