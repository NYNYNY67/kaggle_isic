import pathlib
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd

from src.cross_validation import cross_validation
from src.feature_engineer import FeatureEngineer
from src.train_lgb import train_lgb
from src.comp_score import comp_score


@hydra.main(version_base=None, config_path="conf", config_name="lgb")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    df_train = pd.read_csv(data_dir / "original" / "train-metadata.csv")
    df_train = cross_validation(
        df_train=df_train,
        mode=cfg.cv_mode,
        n_splits=cfg.n_splits,
    )

    fe = FeatureEngineer(
        df_metadata=df_train,
        num_cols=OmegaConf.to_container(cfg.num_cols),
        cat_cols=OmegaConf.to_container(cfg.cat_cols),
        categorical_encoder=None,
    )
    fe.main()

    df_train = fe.df_metadata
    cat_cols = fe.cat_cols
    num_cols = fe.num_cols

    list_df_preds = []
    scores = []
    models = []
    for fold in range(cfg.n_splits):
        df_train_fold = df_train[df_train["fold"] != fold].reset_index(drop=True).copy()
        df_valid_fold = df_train[df_train["fold"] == fold].reset_index(drop=True).copy()

        model, preds = train_lgb(
            df_train=df_train_fold,
            df_valid=df_valid_fold,
            lgb_params=OmegaConf.to_container(cfg.lgb_params),
            cat_cols=cat_cols,
            num_cols=num_cols,
        )
        score = comp_score(df_valid_fold["target"], pd.DataFrame(preds, columns=["prediction"]), "")
        scores.append(score)
        print(f"fold: {fold} - Partial AUC Score: {score:.5f}")

        df_preds = df_valid_fold[["isic_id"]].copy()
        df_preds["pred"] = preds
        list_df_preds.append(df_preds)

        model.save_model(out_dir / f"lgb_{fold}.json")
        models.append(model)

    print(f"over all score: {np.mean(scores)}")

    importance = np.mean([model.feature_importance() for model in models], 0)
    df_importance = pd.DataFrame(
        {
            "feature": models[0].feature_name(),
            "importance": importance,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    df_importance.to_csv(out_dir / "importance.csv")

    df_preds = pd.concat(list_df_preds)
    df_preds.to_parquet(out_dir / "preds.parquet")

    df = df_preds.merge(df_train[["isic_id", "target"]], on="isic_id")
    score = comp_score(df["target"], pd.DataFrame(df["pred"].values, columns=["predinction"]), "")
    print(f"cocnat score: {score}")


if __name__ == "__main__":
    main()
