import catboost


def train_cat(
    df_train,
    df_valid,
    cat_params,
    num_cols,
    cat_cols,
):
    train_pool = catboost.Pool(
        data=df_train[num_cols + cat_cols],
        label=df_train["target"],
        cat_features=cat_cols,
        feature_names=num_cols + cat_cols,
    )
    valid_pool = catboost.Pool(
        data=df_valid[num_cols + cat_cols],
        label=df_valid["target"],
        cat_features=cat_cols,
        feature_names=num_cols + cat_cols,
    )

    model = catboost.CatBoostClassifier(**cat_params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        verbose=100,
    )
    preds = model.predict_proba(valid_pool)[:, 1]
    return model, preds
