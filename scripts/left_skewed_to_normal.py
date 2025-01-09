import pandas as pd
import seaborn as sns
from numpy import arange, float64, int64
import matplotlib.pyplot as plt
from scipy.stats import shapiro

def to_normal(df: pd.DataFrame, attr: str) -> pd.DataFrame:
    """
    Wyświetla histogramy, przed i po wykonaniu transformacji przez podniesienie do potęgi. 
    Uwaga: jeżeli nie udało się dobrać wykładnika przybliżającej rozkład do normalnego,
    wyświetlany jest tylko wykres przed transformacją. Jeżeli udało dobrać się wykładnik
    przybliżając rozkład do normalnego, dodawany jest przetransformowany atrybut.

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

    # sprawdzany zakres wykładników
    exponents = arange(1.5, 10.1, 0.5)
    n_attr = 'exp_' + attr
    if df[attr].dtype == float64:
        stats, best_p_value = shapiro(df[attr] + 0.1)
    elif df[attr].dtype == int64:    
        stats, best_p_value = shapiro(df[attr] + 1)
    best_exponent = 1.0
    for i in range(len(exponents)):
        if df[attr].dtype == float64:
            exp_attr = (df[attr] + 0.1) ** exponents[i]
        elif df[attr].dtype == int64:
            exp_attr = (df[attr] + 1) ** exponents[i]
        stat, p_value = shapiro(exp_attr)
        if p_value > best_p_value:
            best_p_value = p_value
            best_exponent = exponents[i]
    if df[attr].dtype == float64 and best_exponent != 1.0:
        df = df.assign(**{f'{n_attr}': ((df[attr] + 0.1) ** best_exponent)})
    elif df[attr].dtype == int64 and best_exponent != 1.0:
        df = df.assign(**{f'{n_attr}': ((df[attr] + 1) ** best_exponent)})

    plt.figure(figsize=(10, 25))
    fig, axs = plt.subplots(1, 2)
    sns.histplot(data=df, x=attr, kde=True, ax=axs[0], color="salmon")
    axs[0].set_title(f"Histogram dla {attr}")
    axs[0].set_xlabel(attr)

    if best_exponent != 1.0:
        sns.histplot(data=df, x=n_attr, kde=True, ax=axs[1], color="salmon")
        axs[1].set_title(f"Histogram dla {attr}**{best_exponent}")
        axs[1].set_xlabel(f"{attr}**{best_exponent}")

    plt.tight_layout()
    plt.show()

    return df
