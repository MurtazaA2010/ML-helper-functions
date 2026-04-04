def encoder(train, test, col, target, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train_encoded = np.zeros(len(train))
    
    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        
        means = X_train.groupby(col)[target].mean()
        train_encoded[val_idx] = X_val[col].map(means)
    
    global_mean = train[target].mean()
    train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
    
    means = train.groupby(col)[target].mean()
    test_encoded = test[col].map(means).fillna(global_mean)
    
    train[col] = train_encoded
    test[col] = test_encoded
    
    return train, test
