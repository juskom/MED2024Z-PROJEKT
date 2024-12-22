import pandas as pd
import numpy as np

def custom_describe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyświetla i zwraca ramkę danych z wartościami średniej, mediany i mody,
    dla wszystkich atrybutów numerycznych.

    Parametry:
    df (pd.DataFrame): Ramka danych do analizy.

    Zwraca:
    pd.Dataframe - ramkę danych z wartościami średniej, mediany i mody,
    dla atrybutów numerycznych z ramki danych df.
    """
    stats = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    data = {
       'mean': [df[col].mean() for col in stats],
       'median': [df[col].median() for col in stats],
       'mode': [df[col].mode().iloc[0] if not df[col].mode().empty else np.nan for col in stats]
    }
    return pd.DataFrame(data, index=stats).T
