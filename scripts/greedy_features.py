# Miejce na kod
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

def find_best_features(df: pd.DataFrame, dec_attr: str, is_standarized = False, std_dev = 0, is_normalized = False, range_minmax=0) -> None:
    features = df.columns.to_list()
    features.remove(dec_attr)
    n = len(features)

    best_features = []
    min_error = float('inf')

    tmp_features = []
    for j in range(n):
        errors = [0] * (n-j)
        for i, value in enumerate(features):
            tmp_features.append(value)
            result = cross_validate(LinearRegression(), df[tmp_features].values, df[dec_attr].values, cv = 5, scoring = 'neg_mean_squared_error')
            errors[i] = np.mean(-result['test_score'])
            tmp_features.remove(value)
        idx = np.argmin(errors)
        tmp_features.append(features[idx])
        print(tmp_features, errors[idx])
        features.pop(idx)
        print()
        if errors[idx] < min_error:
            min_error = errors[idx]
            best_features = tmp_features

    print("Najmniejszy błąd otrzymujemy dla następującej pary atrybutów:")
    print(best_features, min_error)

    if is_standarized:
        print("Po przeliczeniu błędu w porównaniu do danych bez przeskalowania:")
        print(best_features, min_error * (std_dev ** 2))
    elif is_normalized:
        print("Po przeliczeniu błędu w porównaniu do danych bez przeskalowania:")
        print(best_features, min_error * (range_minmax ** 2))