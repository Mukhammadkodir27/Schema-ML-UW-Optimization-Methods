'''
It always follows the same schema
1. Initialization of the model
2. Pipeline
3. param_grid -> for hyperparameter tuning
4. GridSearchCV with cross-validation
5. gridsearchcv model .fit(X_train, y_train)
6. model results from gridsearchcv (.best_params_ and .best_score_)
7. get the model_tuned.best_estimator_
'''

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from imblearn.over_sampling import SMOTE

# Example feature lists
numeric_features = ['age', 'income', 'years_experience']
categorical_features = ['gender', 'department']

# Preprocessing for numeric columns
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # handle missing values
    ('scaler', StandardScaler()),                  # scale numeric features
    # optional feature generation
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Preprocessing for categorical columns
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # fill missing categories
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numeric + categorical preprocessing
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline: preprocessing + oversampling + model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE()),               # optional if target is imbalanced
    ('model', ElasticNet())           # regularized regression
])

# Hyperparameter grid
param_grid = {
    'model__alpha': [0.01, 0.1, 1, 10],
    'model__l1_ratio': [0.2, 0.5, 0.8]
}

# Cross-validation
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)


'''
Features of this example:
Handles missing values
Scales numeric features
Encodes categorical features
Generates polynomial features
Handles imbalanced target with SMOTE
Uses regularized regression (ElasticNet)
Fully cross-validated and reproducible via pipeline
'''


'''
Pipeline is not a model itself — it’s a way to chain steps (preprocessing → model) together.
GridSearchCV uses the pipeline to repeatedly train models with different hyperparameters and CV splits.
Without a pipeline, you’d have to manually scale, encode, transform data for each CV fold and each hyperparameter combination.
Think of it like a smart for-loop: GridSearchCV loops over hyperparameters + CV folds, 
and the pipeline ensures every iteration applies all preprocessing steps correctly.
So pipeline = automation + reproducibility for repeated model training.
'''

'''
#? this is example from my simple pipeline - above is advanced pipeline with preprocessing 
Right now, you preprocessed separately (scaling, encoding, etc.), so your pipeline just sees clean data.
That works, but it can cause data leakage in cross-validation, because preprocessing was done on the full dataset, 
not separately on each CV fold.
To fix this, you can move preprocessing into the pipeline:
'''


#! --------- Pipeline Continued --- with GridSearchCV + CV KFold

# Create pipeline
logit_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Define hyperparameters to tune
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],       # regularization strength
    'model__penalty': ['l1', 'l2'],       # regularization type
    'model__solver': ['liblinear']        # solver compatible with l1
}

# Setup GridSearchCV
grid = GridSearchCV(logit_pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Best hyperparameters
print("Best parameters:", grid.best_params_)

# Best model performance
print("Best CV score:", grid.best_score_)

# Predict on test set
y_pred = grid.predict(X_test)


# ? This combines pipeline + cross-validation + hyperparameter tuning in one workflow.


# ! or -------------


logit_pipeline = Pipeline([
    ('scaler', StandardScaler()),        # optional, scales features
    ('model', LogisticRegression())      # logistic regression model
])

logit_pipeline.fit(X_train, y_train)
y_pred = logit_pipeline.predict(X_test)


# 5-fold cross-validation
scores = cross_val_score(logit_pipeline, X_train,
                         y_train, cv=5, scoring='accuracy')

print("CV scores:", scores)
print("Mean CV score:", scores.mean())
