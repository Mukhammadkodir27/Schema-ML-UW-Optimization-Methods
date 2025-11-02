# it is good to note that you define Ridge and Lasso in param_grid section

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LogisticRegression


'''
Ridge and Lasso are models themselves — they are regularized linear regression models.
You cannot directly apply them to SVM. SVM has its own regularization via the C parameter (and epsilon for SVR).
Think of Ridge/Lasso as linear regression with penalties, not general-purpose regularizers for other models.
'''


#! Ridge and Lasso in Logistic Regression

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # important for regularization
    ('model', LogisticRegression(max_iter=1000))
])

param_grid = {
    'model__penalty': ['l1', 'l2'],  # Lasso = l1, Ridge = l2
    'model__C': [0.01, 0.1, 1, 10]  # inverse of regularization strength
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv5,
    scoring='roc_auc',
    n_jobs=-1
)


#! Ridge and Lasso in Polynomial Regression

poly_pipeline = Pipeline([
    ('generator', PolynomialFeatures()),
    # placeholder, will be overwritten in param_grid
    ('model', LinearRegression())
])

# define param grid
polyreg_grid = {
    'generator__degree': degrees,        # degrees of polynomial
    'model': [Ridge(), Lasso()],         # choose Ridge or Lasso
    'model__alpha': [0.01, 0.1, 1, 10]  # regularization strength
}

poly_grid_search = GridSearchCV(
    poly_pipeline,
    param_grid=polyreg_grid,
    cv=cv5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)


#! Elastic Net

param_grid = {
    'model': [Ridge(), Lasso(), ElasticNet()],
    'model__alpha': [0.01, 0.1, 1, 10],
    'model__l1_ratio': [0.2, 0.5, 0.8]  # only used by ElasticNet
}

'''
Yes ✅ — Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties.
It can give better results when you have many correlated features, because:
Lasso alone may pick only one feature from a group
Ridge keeps all, shrinks them
You can definitely add it to your GridSearchCV along with Ridge and Lasso, 
and tune its alpha (strength) and l1_ratio (mix of L1/L2).

GridSearchCV will handle it correctly.
'''


#! Additional Information:
'''
Ridge, Lasso, and Elastic Net are mainly regression models - used for Regression problems.
They are designed to predict continuous outcomes and shrink coefficients.
For classification, you would use Logistic Regression with L1/L2 penalty, which is similar in spirit but adapted for classification.
So:
Regression → Ridge, Lasso, ElasticNet
Classification → Logistic Regression with L1/L2 penalty (sometimes called “regularized logistic regression”)
'''
