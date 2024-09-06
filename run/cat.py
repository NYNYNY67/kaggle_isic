import pathlib
import pickle
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd

from src.cross_validation import cross_validation
from src.feature_engineer import FeatureEngineer
from src.train_cat import train_cat
from src.comp_score import comp_score


@hydra.main(version_base=None, config_path="conf", config_name="cat")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    df_train = pd.read_csv(data_dir / "original" / "train-metadata.csv")
    df_train = cross_validation(
        df_train=df_train,
        mode=cfg.cv_mode,
        n_splits=cfg.n_splits,
    )

    meta_features = cfg.meta_features
    meta_feature_cols = []

    if meta_features:
        meta_features = OmegaConf.to_container(cfg.meta_features)
        for i, key in enumerate(meta_features.keys()):
            df_preds = pd.read_parquet(meta_features[key]["path"])
            df_preds = df_preds[["isic_id", "pred"]].rename(columns={"pred": f"meta_pred_{i}"})

            df_train = df_train.merge(
                df_preds,
                on=meta_features[key]["join_keys"],
                how="left",
                validate="many_to_one",
            )
            meta_feature_cols.append(f"meta_pred_{i}")

    fe = FeatureEngineer(
        df_metadata=df_train,
        num_cols=OmegaConf.to_container(cfg.num_cols),
        cat_cols=OmegaConf.to_container(cfg.cat_cols),
        groupings=OmegaConf.to_container(cfg.groupings),
        aggs=OmegaConf.to_container(cfg.aggs),
        categorical_encoder=None,
    )
    fe.main()

    with open(out_dir / "cat_cols.pickle", mode="wb") as f:
        pickle.dump(fe.cat_cols, f)

    with open(out_dir / "num_cols.pickle", mode="wb") as f:
        pickle.dump(fe.num_cols, f)

    with open(out_dir / "categorical_encoder.pickle", mode="wb") as f:
        pickle.dump(fe.categorical_encoder, f)

    df_train = fe.df_metadata
    num_cols = fe.num_cols + meta_feature_cols
    cat_cols = fe.cat_cols
    train_cols = num_cols + cat_cols

    print(f"n_features: {len(train_cols)}")

    list_df_preds = []
    scores = []
    models = []
    for fold in range(cfg.n_splits):
        df_train_fold = df_train[df_train["fold"] != fold].reset_index(drop=True)
        df_valid_fold = df_train[df_train["fold"] == fold].reset_index(drop=True)

        model, preds = train_cat(
            df_train=df_train_fold,
            df_valid=df_valid_fold,
            cat_params=OmegaConf.to_container(cfg.cat_params),
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        score = comp_score(df_valid_fold["target"], pd.DataFrame(preds, columns=["prediction"]), "")
        scores.append(score)
        print(f"fold: {fold} - Partial AUC Score: {score:.5f}")

        df_preds = df_valid_fold[["isic_id"]].copy()
        df_preds["pred"] = preds
        list_df_preds.append(df_preds)

        model.save_model(out_dir / f"cat_{fold}.json")
        models.append(model)

    print(f"over all score: {np.mean(scores)}")

    df_preds = pd.concat(list_df_preds)
    df_preds.to_parquet(out_dir / "preds.parquet")


if __name__ == "__main__":
    main()
