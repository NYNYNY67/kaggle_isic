from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def cross_validation(df_train, mode, n_splits, random_state=42):
    if mode == "gkf":
        kfold = GroupKFold(n_splits=n_splits)
    elif mode == "sgkf":
        kfold = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    else:
        raise Exception(f"unknown mode: {mode}")

    df_train["fold"] = -1
    for idx, (train_idx, val_idx) in enumerate(
        kfold.split(
            df_train,
            df_train["target"],
            groups=df_train["patient_id"]
        )
    ):
        df_train.loc[val_idx, "fold"] = idx

    return df_train
