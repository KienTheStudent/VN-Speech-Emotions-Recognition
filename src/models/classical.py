from sklearn.ensemble import RandomForestClassifier

def get_rf_model(seed):
    """Return a Random Forest classifier configured for SER."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )
