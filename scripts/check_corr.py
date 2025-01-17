import pandas as pd

def check_corr(df: pd.DataFrame, start_idx: int, end_idx: int, dec_attr: str) -> None:
    attributes = df.columns.to_list()[start_idx:end_idx]
    transf_attr = ""

    for attr in attributes:
        for _ in df.columns.to_list()[end_idx:]:
            if _.endswith(attr):
                transf_attr = _
        if transf_attr != "":
            print(attr, "|", transf_attr)
            correlation_attr = df[dec_attr].corr(df[attr])
            correlation_transf_attr = df[dec_attr].corr(df[transf_attr])
            print("Korelacja z atrybutem decyzyjnym\n - Przed transf.:", correlation_attr, "\n - Po transf.:   ", correlation_transf_attr)
            if abs(correlation_transf_attr) > abs(correlation_attr):
                print("Dla", attr, "korelacja po transformacji jest większa")
            print("\n")
        transf_attr = ""

