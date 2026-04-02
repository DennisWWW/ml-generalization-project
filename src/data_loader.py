import pandas as pd


def load_ionosphere(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None)