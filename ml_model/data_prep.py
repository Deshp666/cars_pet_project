from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    numeric_features = ['year', 'mileage', 'engine', 'max_power', 'km_driven',
                        'distance_by_year', 'age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def get_features_and_target(frame):
    """
    Выделяет X и Y и применяет PowerTransformer к целевой переменной.
    """
    df = frame.copy()

    X = df.drop(columns=['selling_price'])
    y = df['selling_price']

    # PowerTransformer требует 2D-массив
    power_trans = PowerTransformer()
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))

    # Возвращаем X, Y_scale, power_trans
    return X, Y_scale, power_trans