import pandas as pd
import seaborn as sns
from numpy import arange, log, float64, int64
import matplotlib.pyplot as plt
from scipy.stats import shapiro

def to_normal(df: pd.DataFrame, attr: str) -> pd.DataFrame:
    """
    Wyświetla histogramy, przed i po wykonaniu transformacji logarytmicznej. 
    Uwaga: jeżeli nie udało się dobrać podstawy przybliżającej rozkład do normalnego,
    wyświetlany jest tylko wykres przed transformacją. Jeżeli udało dobrać się podstawę
    przybliżającą rozkład do normalnego, dodawany jest przetransformowany atrybut.

    Uwaga: dane będące liczbami zmiennoprzecinkowymi podlegają dodaniu o 0.1, aby uniknąć
    logarytmowania zera. Podobnie do danych całkowitoliczbowych dodawane jest 1. 
    Dla danych zmiennoprzecinkowych przyjęto 0.1, ponieważ jest to największa dokładność
    opisywanych w zbiorze danych, które przedstawiane są jako %. Odpowiednio dla danych
    całkowitoliczbowych dodawane jest 1.

    Parametry:
    df (pd.DataFrame): Ramka danych do analizy.
    attr (str): nazwa atrybutu, który ma podlegać transformacji

    Zwraca:
    df (pd.DataFrame): Ramkę danych z dodanymi zlogarytmowanymi atrybutami.
    Wyświetla histogramy.
    """

    # sprawdzany zakres podstaw logarytmu
    base = arange(1.5, 10.1, 0.5)
    n_attr = 'log' + attr
    if df[attr].dtype == float64:
        stats, best_p_value = shapiro(df[attr] + 0.1)
    elif df[attr].dtype == int64:    
        stats, best_p_value = shapiro(df[attr] + 1)
    best_base = 1.0
    for i in range(len(base)):
        if df[attr].dtype == float64:
            log_attr = log(df[attr] + 0.1) / log(base[i])
        elif df[attr].dtype == int64:
            log_attr = log(df[attr] + 1) / log(base[i])
        stat, p_value = shapiro(log_attr)
        if p_value > best_p_value:
            best_p_value = p_value
            best_base = base[i]
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
