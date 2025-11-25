from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from math import ceil


class FeatureEngineerAndCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, drop_duplicates=False, use_statistic_method=True):
        self.drop_duplicates = drop_duplicates
        self.use_statistic_method = use_statistic_method
        self.encoders = {}
        self.outlier_borders = {}
        self.median_values = []
        self.known_categories = {}
        self.cat_columns = ['name', 'fuel', 'transmission', 'owner', 'seller_type']

    def calculate_outliers(self, column):
        inter_quantile = column.quantile(0.75) - column.quantile(0.25)
        lower_border = column.quantile(0.25) - inter_quantile * 1.5
        upper_border = inter_quantile * 1.5 + column.quantile(0.75)
        return ceil(lower_border), ceil(upper_border)

    def fit(self, X, y=None):
        df = X.copy()

        df = df.drop(['torque'], axis=1)
        df['engine'] = df['engine'].str.replace('CC', '').apply(pd.to_numeric, errors='coerce')
        df['mileage'] = df['mileage'].str.replace('kmpl', '').apply(pd.to_numeric, errors='coerce')
        df['max_power'] = df['max_power'].str.replace('bhp', '').apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        df['name'] = df['name'].apply(lambda x: str(x).split()[0])
        for col in self.cat_columns:
            encoder = LabelEncoder()
            encoder.fit(df[col])
            self.encoders[col] = encoder
            self.known_categories[col] = set(df[col].unique())

        df_col_to_prepare = ['year', 'mileage', 'engine', 'max_power', 'km_driven']
        if self.use_statistic_method:
            for col in df_col_to_prepare:
                self.outlier_borders[col] = self.calculate_outliers(df[col])

        self.median_values = df.median(numeric_only=True)

        return self

    def transform(self, X):
        df = X.copy()

        if 'torque' in df.columns:
            df = df.drop(['torque'], axis=1)
        df['engine'] = df['engine'].str.replace('CC', '').apply(pd.to_numeric, errors='coerce')
        df['mileage'] = df['mileage'].str.replace('kmpl', '').apply(pd.to_numeric, errors='coerce')
        df['max_power'] = df['max_power'].str.replace('bhp', '').apply(pd.to_numeric, errors='coerce')

        if hasattr(self, 'median_values'):
            df = df.fillna(self.median_values)

        if self.use_statistic_method and self.outlier_borders:
            for col, (lower_border, upper_border) in self.outlier_borders.items():
                df.loc[df[col] > upper_border, col] = upper_border
                df.loc[df[col] < lower_border, col] = lower_border

        current_year = 2025
        df['distance_by_year'] = round(df['km_driven'] / (2021 - df['year']))
        df['age'] = current_year - df['year']

        df['name'] = df['name'].apply(lambda x: str(x).split()[0])
        for col in self.cat_columns:
            if col in self.encoders:
                unseen_mask = ~df[col].isin(self.known_categories[col])
                if df[col].dtype == 'object':
                    mode_val = self.encoders[col].classes_[0]
                    df.loc[unseen_mask, col] = mode_val

                df[col] = self.encoders[col].transform(df[col])
        if self.drop_duplicates:
            df = df.drop_duplicates()

        return df