import pandas as pd
import seaborn as sns
from numpy import log, float64, int64
import matplotlib.pyplot as plt

def to_normal(df: pd.DataFrame, attr: str) -> pd.DataFrame:
    # słownik wartości dobranych przez right_skewed_to_normal.py
    value_map = {
        "Average_House_Price": 6.0,
        "FamilyWork": 1.5,
        "SelfEmployed": 3.0,
        "PublicWork": 3.0,
        "WorkAtHome": 1.5,
        "OtherTransp": 6.0,
        "Walk": 6.5,
        "Transit": 4.5,
        "Pacific": 1.0,
        "Asian": 1.5,
        "Native": 3.5,
        "Black": 2.5,
        "Hispanic": 7.5,
        "TotalPop": 5.5
    }
    
    best_base = value_map.get(attr, None)

    n_attr = 'log' + attr

    if df[attr].dtype == float64 and best_base != 1.0:
        df = df.assign(**{f'{n_attr}': (log(df[attr] + 0.1) / log(best_base))})
    elif df[attr].dtype == int64 and best_base != 1.0:
        df = df.assign(**{f'{n_attr}': (log(df[attr] + 1) / log(best_base))})

    plt.figure(figsize=(10, 25))
    fig, axs = plt.subplots(1, 2)
    sns.histplot(data=df, x=attr, kde=True, ax=axs[0], color="salmon")
    axs[0].set_title(f"Histogram dla {attr}")
    axs[0].set_xlabel(attr)

    if best_base != 1.0:
        sns.histplot(data=df, x=n_attr, kde=True, ax=axs[1], color="salmon")
        axs[1].set_title(f"Histogram dla log_{best_base}({attr})")
        axs[1].set_xlabel(f"log_{best_base}({attr})")

    plt.tight_layout()
    plt.show()

    return df
