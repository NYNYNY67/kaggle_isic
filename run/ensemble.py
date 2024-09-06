import pathlib
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd

from src.cross_validation import cross_validation
from src.comp_score import comp_score


@hydra.main(version_base=None, config_path="conf", config_name="ensemble")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"

    df_train = pd.read_csv(data_dir / "original" / "train-metadata.csv")
    df_train = cross_validation(
        df_train=df_train,
        mode=cfg.cv_mode,
        n_splits=cfg.n_splits,
    )

    members = OmegaConf.to_container(cfg.members)
    dict_df_preds = {
        key: pd.read_parquet(members[key]["path"]) for key in members.keys()
    }

    results = []
    for fold in range(cfg.n_splits):
        df_oof = df_train[df_train["fold"] == fold].copy()
        df_oof["average_rank"] = 0
        fold_result = []
        for i, key in enumerate(members.keys()):
            df_preds = dict_df_preds[key].rename(columns={"pred": f"meta_pred_{key}"})
            df_oof = df_oof.merge(
                df_preds,
                on="isic_id",
                how="left",
                validate="one_to_one",
            ).sort_values(f"meta_pred_{key}", ascending=False)
            df_oof[f"rank_{key}"] = np.arange(len(df_oof)) + 1
            df_oof["average_rank"] += members[key]["weight"] * df_oof[f"rank_{key}"]

            score = comp_score(
                df_oof["target"],
                pd.DataFrame(df_oof[f"meta_pred_{key}"].values, columns=["prediction"]),
                "",
            )
            print(f"model: {key}, fold: {fold}, score: {score}")
            fold_result.append(score)

        ranks = df_oof[[f"rank_{key}" for key in members.keys()]].values
        ranks.sort(axis=1)

        score = comp_score(
            df_oof["target"],
            1 / pd.DataFrame(df_oof["average_rank"].values, columns=["prediction"]),
            "",
        )
        fold_result.append(score)

        results.append(fold_result)
    results.append(list(np.mean(results, axis=0)))
    results.append([members[key]["weight"] for key in members.keys()])

    df_result = pd.DataFrame(
        results,
        columns=list(members.keys()) + ["ensemble_average"],
        index=[f"fold_{i}" for i in range(cfg.n_splits)] + ["mean"] + ["weight"]
    )
    print(df_result)


if __name__ == "__main__":
    main()
