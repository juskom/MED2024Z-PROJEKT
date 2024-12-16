import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Tuple

def analyze(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Przeprowadza kompletną analizę braków w ramce danych i je wizualizuje.

    Parametry:
    df (pd.DataFrame): Ramka danych do analizy.

    Zwraca:
    (missing_attributes, missing_objects): Krotkę zawierającą dwie pd.Series:
        - Pierwsza pd.Series zawierającą sumę braków w danych liczonych wg. atrybutów (kolumn).
        - Druga pd.Series zawierającą sumę braków w danych liczonych wg. obiektów (wierszy).
    """
    visualize_missing_values(df)
    return identify_missing_values(df)


def identify_missing_values(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Wypisuje sumę braków danych dla każdego atrybutu i obiektu w ramce danych

    Parametry:
    df (pd.DataFrame): Ramka danych do analizy.

    Zwraca:
    (missing_attributes_filtred, missing_objects_filtred): Krotkę zawierającą dwie pd.Series:
        - Pierwsza pd.Series zawierającą sumę braków w danych liczonych wg. atrybutów (kolumn).
        - Druga pd.Series zawierającą sumę braków w danych liczonych wg. obiektów (wierszy).
    Wypisuje braki do konsoli.
    """
    missing_attributes = df.isnull().sum()
    missing_attributes_filtred = missing_attributes[missing_attributes > 0]
    if missing_attributes_filtred:
        print("\nBraki w danych liczone wg. atrybutów")
        print(missing_attributes_filtred)
    else:
        print("\nBrak braków w danych liczonych wg. atrybutów")

    print("\nBraki w danych liczone wg. obiektów")
    missing_objects = df.isnull().sum(axis = 1)
    missing_objects_filtred = missing_objects[missing_objects > 0]
    if missing_objects_filtred:
        print("\nBraki w danych liczone wg. obiektów")
        print(missing_objects_filtred)
    else:
        print("\nBrak braków w danych liczonych wg. obiektów")

    return missing_attributes_filtred, missing_objects_filtred

def visualize_missing_values(df: pd.DataFrame) -> None:
    """
    Tworzy mapę ciepła do wizualizacji braków w ramce danych.

    Parametry:
    df (pd.DataFrame): Ramka danych do wizualizacji.

    Zwraca:
    None: Wyświetla mapę ciepła dla braków w ramce danych.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Mapa ciepła, dla braków danych")
    plt.show()