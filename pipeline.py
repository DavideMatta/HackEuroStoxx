from feature_engineering import *
from sklearn.pipeline import Pipeline


# Pipeline creation
financial_data_pipeline = Pipeline([
    ('financial_indicators', FinancialIndicators()),
    ('date_handling', DateHandling()),
    ('target_feature', CreateTarget()),
    ('drop_na', DropNA())  # Drop NA rows after all transformations
])