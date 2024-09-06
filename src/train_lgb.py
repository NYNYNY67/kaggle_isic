import lightgbm


def train_lgb(
    df_train,
    df_valid,
    lgb_params,
    cat_cols,
    num_cols,
    early_stopping_rounds,
):
    train_cols = cat_cols + num_cols

    df_train["weight"] = 1

    # df_train["target"] = df_train["iddx_1"].map({
    #     "Benign": 0,
    #     "Indeterminate": 1,
    #     "Malignant": 1,
    # })
    # df_train["weight"] = df_train["iddx_1"].map({
    #     "Benign": 1,
    #     "Indeterminate": 2,
    #     "Malignant": 12,
    # })
    # df_train["has_lesion_id"] = ~df_train["lesion_id"].isna()
    # df_train["weight"] = df_train.apply(
    #     lambda row: 0.3 if (row["iddx_1"] == "Benign" and row["has_lesion_id"]) else row["weight"],
    #     axis=1,
    # )

    train_set = lightgbm.Dataset(
        data=df_train[train_cols],
        label=df_train["target"],
        feature_name=train_cols,
        categorical_feature=cat_cols,
        weight=df_train["weight"].values,
    )
    valid_set = lightgbm.Dataset(
        data=df_valid[train_cols],
        label=df_valid["target"],
        reference=train_set,
        feature_name=train_cols,
        categorical_feature=cat_cols,
    )
    model = lightgbm.train(
        train_set=train_set,
        valid_sets=[valid_set],
        callbacks=[lightgbm.early_stopping(stopping_rounds=early_stopping_rounds)],
        **lgb_params,
    )
    preds = model.predict(df_valid[train_cols])
    return model, preds
