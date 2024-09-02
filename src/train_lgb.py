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

    train_set = lightgbm.Dataset(
        data=df_train[train_cols],
        label=df_train["target"],
        feature_name=train_cols,
        categorical_feature=cat_cols,
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
