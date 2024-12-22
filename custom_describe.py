import pandas as pd
import numpy as np

def custom_describe(df: pd.DataFrame) -> pd.DataFrame:
    stats = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    data = {
       'mean': [df[col].mean() for col in stats],
       'median': [df[col].median() for col in stats],
       'mode': [df[col].mode().iloc[0] if not df[col].mode().empty else np.nan for col in stats]
    }
    return pd.DataFrame(data, index=stats).T
