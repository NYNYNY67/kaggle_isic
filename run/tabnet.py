import pathlib
import pickle
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from src.cross_validation import cross_validation
from src.feature_engineer import FeatureEngineer
from src.comp_score import comp_score


@hydra.main(version_base=None, config_path="conf", config_name="tabnet")
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
    df_train = df_train.replace([np.inf, -np.inf], np.nan).fillna(-1)
    train_cols = fe.cat_cols + fe.num_cols
    print(f"n_features: {len(train_cols)}")

    list_df_preds = []
    scores = []
    models = []
    for fold in range(cfg.n_splits):
        df_train_fold = df_train[df_train["fold"] != fold].reset_index(drop=True)
        df_valid_fold = df_train[df_train["fold"] == fold].reset_index(drop=True)

        model = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            cat_idxs=[i for i in range(len(fe.cat_cols))],
            cat_dims=[df_train[cat_col].nunique() + 2 for cat_col in fe.cat_cols],
            cat_emb_dim=4,
            seed=42,
        )
        model.fit(
            df_train_fold[train_cols].values,
            df_train_fold["target"].values,
            eval_set=[(df_valid_fold[train_cols].values, df_valid_fold["target"].values)],
            weights={0: 1, 1: 1},
        )
        model.save_model(str(out_dir / f"tabnet_{fold}"))
        preds = model.predict_proba(df_valid_fold[train_cols].values)[:, 1]

        score = comp_score(df_valid_fold["target"], pd.DataFrame(preds, columns=["prediction"]), "")
        scores.append(score)
        print(f"fold: {fold} - Partial AUC Score: {score:.5f}")

        df_preds = df_valid_fold[["isic_id"]].copy()
        df_preds["pred"] = preds
        list_df_preds.append(df_preds)

        models.append(model)

    df_preds = pd.concat(list_df_preds)
    df_preds.to_parquet(out_dir / "preds.parquet")

    print(f"over all score: {np.mean(scores)}")


if __name__ == "__main__":
    main()
