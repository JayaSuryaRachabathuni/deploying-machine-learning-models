from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from regression_model.processing import preprocessors as pp
from regression_model.processing import features
from regression_model.config import config

import logging


_logger = logging.getLogger(__name__)


# price_pipe = Pipeline(
#     [
#         (
#             "categorical_imputer",
#             pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA),
#         ),
#         (
#             "numerical_inputer",
#             pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA),
#         ),
#         (
#             "temporal_variable",
#             pp.TemporalVariableEstimator(
#                 variables=config.TEMPORAL_VARS, reference_variable=config.DROP_FEATURES
#             ),
#         ),
#         (
#             "rare_label_encoder",
#             pp.RareLabelCategoricalEncoder(tol=0.01, variables=config.CATEGORICAL_VARS),
#         ),
#         (
#             "categorical_encoder",
#             pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS),
#         ),
#         (
#             "log_transformer",
#             features.LogTransformer(variables=config.NUMERICALS_LOG_VARS),
#         ),
#         (
#             "drop_features",
#             pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES),
#         ),
#         ("scaler", MinMaxScaler()),
#         ("Linear_model", Lasso(alpha=0.005, random_state=0)),
#     ]
# )

price_pipe = Pipeline(
    [
        ("drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES),
        ),

        ('categorical_imputer',
         pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),

        ('missing indicator',
            pp.MissingIndicator(variables=config.NUMERICAL_VARS)),

        ('numerical_inputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
         
        ('cabin_variable',
            pp.ExtractFirstLetter(
                variables=config.CABIN)),
         
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.05,
                variables=config.CATEGORICAL_VARS)),
         
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        
        ('scaler', StandardScaler()),
        ('Linear_model', Lasso(C=0.0005, random_state=0))
    ]
)
