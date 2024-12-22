import pandas as pd

def extended_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przeprowadza rozszerzoną względem funkcji df.describe() analizę podstawowych statystyk,
    tj. poza odchyleniem stand., śrędnia, ... liczy również współczynnik zmienności, który
    pozwala łatwiej porównać zmienność danych dla różnych atrybutów.

    Liczony jako: s/x, gdzie s - odchylenie standardowe, x - wartość średnia.

    Parametry:
    df (pd.DataFrame): Ramka danych do analizy.

    Zwraca:
    description - rozszerzone df.describe() o współczynnik zmienności.
    """
    description = df.describe()

    coeff_of_variation = description.loc["std"]/description.loc["mean"]
    coeff_of_variation = coeff_of_variation.to_frame().T
    coeff_of_variation.index = ['coeff_of_variation']

    description = pd.concat([description, coeff_of_variation])

    return description