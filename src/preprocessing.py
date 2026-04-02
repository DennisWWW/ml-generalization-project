import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_ionosphere(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    y = (y == "g").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy()