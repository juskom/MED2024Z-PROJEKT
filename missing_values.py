import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def analyze(df: pd.DataFrame) -> None:
    identify_missing_values(df)
    visualize_missing_values(df)


def identify_missing_values(df: pd.DataFrame):
    print("\nBraki w danych liczone wg. kolumn")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nBraki w danych liczone wg. obiektów")
    missing_objects = df.isnull().sum(axis = 1)
    print(missing_objects[missing_objects > 0])

def visualize_missing_values(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Mapa ciepła, dla braków danych")
    plt.show()