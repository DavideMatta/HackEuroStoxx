from sklearn.pipeline import Pipeline
from processing.feature_engineering import *


# Pipeline creation
financial_data_pipeline = Pipeline([
    ('financial_indicators', FinancialIndicators()),
    ('date_handling', DateHandling()),
    ('target_feature', CreateTarget()),
    ('drop_na', DropNA())  # Drop NA rows after all transformations
])