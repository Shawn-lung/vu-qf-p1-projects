from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

all_features = [
    'Recent_return', 'Size', 'Trading_activity', 'Momentum_6m', 'Momentum_12m',
    'Volatility', 'beta', 'bm', 'GProf', 'debt_at', 'accrual', 'divyield'
]
ols3_features = ['Size', 'bm', 'Momentum_12m']

train_end = '2020-12-01'
validation_end = '2022-12-01'
forecast_month = '2025-12-01'

def r2(realized, predicted):
    """
    Out-of-sample R^2
    """
    realized = np.asarray(realized)
    predicted = np.asarray(predicted)

    return 1 - np.sum((realized - predicted) ** 2) / np.sum(realized ** 2)

def load_model_data(outputs_dir):
    """
    Read the ranked stock-month panel used for estimation
    """
    df = pd.read_csv(outputs_dir / 'model_dataset_monthly.csv', parse_dates = ['Month', 'Target_month'])
    return df.sort_values(['Target_month', 'PERMNO'])

def data_split(df):
    """
    Separate data into training, validation, and testing sets by target month
    """
    target = df['Target_month']
    train = df.loc[target <= train_end]
    validation = df.loc[(target > train_end) & (target <= validation_end)]
    test = df.loc[target > validation_end]
    return train, validation, test

def fit_ols(train, features):
    """
    Fit an OLS return model with intercept
    """
    X = train[features]
    y = train['Target_excess_return']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def predict(model, data, features):
    """
    Predict monthly excess returns for any stock-month
    """
    if hasattr(model, 'params'):
        X = sm.add_constant(data[features], has_constant='add')
        return np.asarray(model.predict(X))
    return model.predict(data[features])

def tune_random_forest(train, validation):
    """
    Choose the tree depth and feature share on the validation set. Suggestion by GKX (2020) is that 
    depth = 1 ~ 6, trees = 300, feature share = (3, 5, 10, 20, 30, 50,...)
    """
    records = []
    best_score = -np.inf
    best_depth = None
    best_feature_share = None
    for depth in range(4, 7):
        for feature_share in [3, 5, 10]:
            model = RandomForestRegressor(
                n_estimators = 300,
                min_samples_leaf = 50,
                max_depth = depth,
                max_features = feature_share,
                random_state = 23,
                n_jobs = -1
            )
            model.fit(train[all_features], train['Target_excess_return'])
            predicted = predict(model, validation, all_features)
            score = r2(validation['Target_excess_return'], predicted)
            records.append({
                'Depth': depth,
                'Feature_share': feature_share,
                'Score': score
            })
            if score > best_score:
                best_score = score
                best_depth = depth
                best_feature_share = feature_share
    return best_depth, best_feature_share, pd.DataFrame(records)

def fit_random_forest(train, depth, feature_share):
    """
    Fit the RF with the validation-chosen settings
    """
    model = RandomForestRegressor(
        n_estimators = 300,
        min_samples_leaf = 50,
        max_depth = depth,
        max_features = feature_share,
        random_state = 23,
        n_jobs = -1
    )
    model.fit(train[all_features], train['Target_excess_return'])
    return model

def comparison_row(name, model, features, estimation, test):
    """
    In-sample and test R^2 for a given model
    """
    y_est = estimation['Target_excess_return']
    y_test = test['Target_excess_return']
    r2_est = r2(y_est, predict(model, estimation, features))
    r2_test = r2(y_test, predict(model, test, features))
    
    return {
        'Model': name,
        'N_estimation': len(estimation),
        'N_test': len(test),
        'R^2 in-sample (%)': r2_est*100,
        'R^2 out-of-sample (%)': r2_test*100,
    }

def coefficient_table(ols3, ols):
    """
    Collect coefficients from OLS3 and OLS models
    """
    rows = []
    for name, model, features in [('OLS-3', ols3, ols3_features), ('OLS', ols, all_features)]:
        rows.append(
            {
                'Model': name,
                'Features': 'Intercept',
                'Coefficients': model.params.iloc[0]
            }
        )
        for feature, slope in zip(features, model.params.iloc[1:]):
            rows.append(
                {
                    'Model': name,
                    'Features': feature,
                    'Coefficients': slope
                }
            )
    return pd.DataFrame(rows)

def forecast_latest(outputs_dir, history, rf_params):
    """
    Forecast January 2026 excess returns from December 2025 characteristics.

    The models are refitted on all realized target months. Hyperparameters are fixed at the validation-chosen 
    settings.

    Each prediction is an expected monthly excess return.
    """
    features = pd.read_csv(outputs_dir / 'ranked_features_monthly.csv', parse_dates = ['Month'])
    latest = features.loc[features['Month'] == forecast_month]
    if len(latest) == 0:
        raise ValueError(f"No data found for {forecast_month}")
    
    depth, feature_share = rf_params
    models = {
        'pred_ols3': (fit_ols(history, ols3_features), ols3_features),
        'pred_ols': (fit_ols(history, all_features), all_features),
        'pred_rf': (fit_random_forest(history, depth, feature_share), all_features),
    }
    forecast = latest[['PERMNO', 'Month']].copy()
    forecast['Target_month'] = pd.Timestamp('2026-01-01')
    for name, (model, features) in models.items():
        forecast[name] = predict(model, latest, features)
    return forecast

def run_models(outputs_dir: Path):
    """
    Compare OLS-3, OLS, and RF models on the same out-of-sample months.

    OLS-3 and OLS need no hyperparameter tuning. RF is tuned on the validation set. But all three test forecasts
    come from models refitted on the same estimation sample (training + validation).

    Returns:
        Comparison table, test predictions and the JAN 2026 forecasts.
    """
    df = load_model_data(outputs_dir)
    train, validation, test = data_split(df)
    estimation = pd.concat([train, validation], ignore_index = True)

    best_depth, best_feature_share, rf_records = tune_random_forest(train, validation)
    models = {
        'OLS-3': (fit_ols(estimation, ols3_features), ols3_features),
        'OLS': (fit_ols(estimation, all_features), all_features),
        'Random Forest': (fit_random_forest(estimation, best_depth, best_feature_share), all_features),
    }
    comparison = pd.DataFrame([
        comparison_row(name, model, features, estimation, test)
        for name, (model, features) in models.items()
    ])
    comparison['rf_max_depth'] = best_depth
    comparison['rf_max_features'] = best_feature_share
    
    predictions = test[['PERMNO', 'Month', 'Target_month', 'Target_excess_return']].copy()
    for name, (model, features) in models.items():
        column = {'OLS-3': 'pred_ols3', 'OLS': 'pred_ols', 'Random Forest': 'pred_rf'}[name]
        predictions[column] = predict(model, test, features)

    forecast = forecast_latest(outputs_dir, df, (best_depth, best_feature_share))

    comparison.to_csv(outputs_dir / 'model_comparison.csv', index = False)
    rf_records.to_csv(outputs_dir / 'rf_validation.csv', index = False)
    coefficient_table(models['OLS-3'][0], models['OLS'][0]).to_csv(outputs_dir / 'ols_coefficients.csv', index = False)
    predictions.to_csv(outputs_dir / 'test_predictions.csv', index = False)
    forecast.to_csv(outputs_dir / 'jan_2026_forecast.csv', index = False)

    return comparison, predictions, forecast


if __name__ == '__main__':
    outputs_dir = Path(__file__).resolve().parent.parent / 'outputs'
    comparison, _, forecast = run_models(outputs_dir)
    print(comparison.round(4).to_string(index = False))
    print(
        f"January 2026 forecasts: {forecast['PERMNO'].nunique()} stocks, "
        f"mean RF excess return {forecast['pred_rf'].mean():.2%}"
    )



