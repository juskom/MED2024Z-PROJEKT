import pandas as pd
import seaborn as sns
from numpy import arange, float64, int64
import matplotlib.pyplot as plt

def to_normal(df: pd.DataFrame, attr: str) -> pd.DataFrame:
    # słownik wartości dobranych przez left_skewed_to_normal.py
    value_map = {
        "White": 5.0,
        "Drive": 4.5,
        "PrivateWork": 4.0
    }
    
    best_exp = value_map.get(attr, None)

    n_attr = 'exp_' + attr
    if df[attr].dtype == float64:
        df = df.assign(**{f'{n_attr}': ((df[attr] + 0.1) ** best_exp)})
    elif df[attr].dtype == int64:    
        df = df.assign(**{f'{n_attr}': ((df[attr] + 1) ** best_exp)})

    plt.figure(figsize=(10, 25))
    fig, axs = plt.subplots(1, 2)
    sns.histplot(data=df, x=attr, kde=True, ax=axs[0], color="salmon")
    axs[0].set_title(f"Histogram dla {attr}")
    axs[0].set_xlabel(attr)

    sns.histplot(data=df, x=n_attr, kde=True, ax=axs[1], color="salmon")
    axs[1].set_title(f"Histogram dla {attr}**{best_exp}")
    axs[1].set_xlabel(f"{attr}**{best_exp}")

    plt.tight_layout()
    plt.show()

    return df
