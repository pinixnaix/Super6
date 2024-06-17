import os
import logging
import numpy as np
import optuna
import pandas as pd
import joblib
from influxdb_client import InfluxDBClient
from typing import Dict, Any, List
from keras.src.layers import Dense, BatchNormalization, Dropout
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import make_scorer, mean_absolute_error, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier, XGBRegressor
from keras import Sequential
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress Convergence Warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "super6")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "international")

# File paths for model persistence
MODEL_FILES = {
    'home_goals': 'home_goals_model.pkl',
    'away_goals': 'away_goals_model.pkl',
    'both_teams_to_score': 'both_teams_to_score_model.pkl',
    'match_result': 'match_result_model.pkl',
    'poly_features': 'poly_features.joblib',
    'scaler': 'scaler.joblib',
    'nn_match_results': 'nn_match_result_model'
}

UEFA_TEAM_RANKINGS = {
    "Germany": 15, "Scotland": 31, "Hungary": 32, "Switzerland": 12, "Spain": 10, "Croatia": 6,
    "Italy": 8, "Albania": 59, "Poland": 25, "Netherlands": 7, "Slovenia": 63, "Denmark": 19,
    "Serbia": 22, "England": 4, "Romania": 29, "Ukraine": 21, "Belgium": 5, "Slovakia": 44,
    "Austria": 27, "France": 2, "Turkey": 36, "Georgia": 48, "Portugal": 9, "Czech Republic": 28
}


class DataLoadError(Exception):
    pass


def init_influxdb_client() -> Any:
    logging.info('Initializing InfluxDB client')
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        logging.info('InfluxDB client initialized successfully')
        return client.query_api()
    except Exception as e:
        logging.error('Error initializing InfluxDB client: %s', e)
        raise DataLoadError('Failed to initialize InfluxDB client')


query_api = init_influxdb_client()


def query_influxdb(query: str) -> Any:
    try:
        return query_api.query(query=query)
    except Exception as e:
        logging.error('Error querying data: %s', e)
        return []


def get_team_games(team_name: str) -> pd.DataFrame:
    logging.info(f'Querying games for team: {team_name}')
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -5y)
      |> filter(fn: (r) => r._measurement == "match_results")
      |> filter(fn: (r) => r.home_team == "{team_name}" or r.away_team == "{team_name}")
      |> sort(columns: ["_time"], desc: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    tables = query_influxdb(query)
    games = [
        {
            "time": record.get_time(),
            "home_team": record.values.get("home_team"),
            "away_team": record.values.get("away_team"),
            "tournament": record.values.get("tournament"),
            "country": record.values.get("country"),
            "neutral": record.values.get("neutral"),
            "home_score": int(record.values.get("home_score", 0)),
            "away_score": int(record.values.get("away_score", 0))
        }
        for table in tables for record in table.records
    ]
    df_games = pd.DataFrame(games).sort_values(by="time").reset_index(drop=True)
    logging.info(f'Found {len(df_games)} games for team: {team_name}')
    return df_games


def get_all_teams() -> List[str]:
    logging.info('Querying for all unique teams')
    query_teams = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -5y)
      |> filter(fn: (r) => r._measurement == "match_results")
      |> keep(columns: ["home_team", "away_team"])
      |> distinct(column: "home_team")
      |> distinct(column: "away_team")
    '''
    tables = query_influxdb(query_teams)
    teams = {record.values.get("home_team") or record.values.get("away_team") for table in tables for record in
             table.records}
    logging.info('Found %d unique teams', len(teams))
    return list(teams)


def create_dataset() -> tuple:

    all_teams = get_all_teams()
    all_games = pd.concat([get_team_games(team) for team in all_teams], ignore_index=True)

    if all_games.empty:
        logging.error('No games found for any team.')
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None, None

    logging.info(f'Total games collected: {len(all_games)}')

    features, targets_home, targets_away, targets_bt, targets_result = [], [], [], [], []

    for index, row in all_games.iterrows():
        logging.info(f'Processing game {index} of {len(all_games)}')
        team1, team2 = row['home_team'], row['away_team']

        feature = create_features(team1, team2)
        features.append(feature)
        targets_home.append(row['home_score'])
        targets_away.append(row['away_score'])
        targets_bt.append(1 if row['home_score'] > 0 and row['away_score'] > 0 else 0)
        if row['home_score'] > row['away_score']:
            targets_result.append(1)  # Home win
        elif row['home_score'] < row['away_score']:
            targets_result.append(2)  # Away win
        else:
            targets_result.append(0)  # Draw

    features = np.random.rand(100, 10)
    home_goals = np.random.randint(0, 5, 100)
    away_goals = np.random.randint(0, 5, 100)
    both_teams_to_score = np.random.randint(0, 2, 100)
    match_results = np.random.randint(0, 3, 100)

    poly = PolynomialFeatures(degree=2)
    features_poly = poly.fit_transform(features)
    scaler = StandardScaler().fit(features_poly)
    features_poly_scaled = scaler.transform(features_poly)

    return features_poly_scaled, home_goals, away_goals, both_teams_to_score, match_results, poly, scaler


def get_team_form(team: str, num_games: int = 4) -> Dict[str, float]:
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -3y)
      |> filter(fn: (r) => r._measurement == "match_results")
      |> filter(fn: (r) => r.home_team == "{team}" or r.away_team == "{team}")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: {num_games})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    tables = query_influxdb(query)
    games = [
        {
            "time": record.get_time(),
            "home_team": record.values.get("home_team"),
            "away_team": record.values.get("away_team"),
            "tournament": record.values.get("tournament"),
            "country": record.values.get("country"),
            "neutral": record.values.get("neutral"),
            "home_score": int(record.values.get("home_score", 0)),
            "away_score": int(record.values.get("away_score", 0))
        }
        for table in tables for record in table.records
    ]
    games.sort(key=lambda x: x["time"])  # Sort games by time

    form = {
        "wins": 0, "draws": 0, "losses": 0,
        "avg_scored": 0, "avg_conceded": 0, "avg_goal_dif": 0,
        "home_total_scored": 0, "home_total_conceded": 0,
        "away_total_scored": 0, "away_total_conceded": 0,
        "home_over_0_5_scored": 0, "home_over_1_5_scored": 0, "home_over_2_5_scored": 0, "home_over_3_5_scored": 0,
        "home_over_0_5_conceded": 0, "home_over_1_5_conceded": 0, "home_over_2_5_conceded": 0,
        "home_over_3_5_conceded": 0,
        "away_over_0_5_scored": 0, "away_over_1_5_scored": 0, "away_over_2_5_scored": 0, "away_over_3_5_scored": 0,
        "away_over_0_5_conceded": 0, "away_over_1_5_conceded": 0, "away_over_2_5_conceded": 0,
        "away_over_3_5_conceded": 0
    }

    for game in games:
        if not valid_scores(game["home_score"], game["away_score"]):
            continue
        if game["home_team"] == team:
            form["avg_scored"] += game["home_score"]
            form["avg_conceded"] += game["away_score"]
            form["avg_goal_dif"] += game["home_score"] - game["away_score"]
            form["home_total_scored"] += game["home_score"]
            form["home_total_conceded"] += game["away_score"]
            if game["home_score"] > game["away_score"]:
                form["wins"] += 1
            elif game["home_score"] == game["away_score"]:
                form["draws"] += 1
            else:
                form["losses"] += 1

            if game["home_score"] > 0.5:
                form["home_over_0_5_scored"] += 1
            if game["home_score"] > 1.5:
                form["home_over_1_5_scored"] += 1
            if game["home_score"] > 2.5:
                form["home_over_2_5_scored"] += 1
            if game["home_score"] > 3.5:
                form["home_over_3_5_scored"] += 1

            if game["away_score"] > 0.5:
                form["home_over_0_5_conceded"] += 1
            if game["away_score"] > 1.5:
                form["home_over_1_5_conceded"] += 1
            if game["away_score"] > 2.5:
                form["home_over_2_5_conceded"] += 1
            if game["away_score"] > 3.5:
                form["home_over_3_5_conceded"] += 1
        else:
            form["avg_scored"] += game["away_score"]
            form["avg_conceded"] += game["home_score"]
            form["avg_goal_dif"] += game["away_score"] - game["home_score"]
            form["away_total_scored"] += game["away_score"]
            form["away_total_conceded"] += game["home_score"]
            if game["away_score"] > game["home_score"]:
                form["wins"] += 1
            elif game["away_score"] == game["home_score"]:
                form["draws"] += 1
            else:
                form["losses"] += 1

            if game["away_score"] > 0.5:
                form["away_over_0_5_scored"] += 1
            if game["away_score"] > 1.5:
                form["away_over_1_5_scored"] += 1
            if game["away_score"] > 2.5:
                form["away_over_2_5_scored"] += 1
            if game["away_score"] > 3.5:
                form["away_over_3_5_scored"] += 1

            if game["home_score"] > 0.5:
                form["away_over_0_5_conceded"] += 1
            if game["home_score"] > 1.5:
                form["away_over_1_5_conceded"] += 1
            if game["home_score"] > 2.5:
                form["away_over_2_5_conceded"] += 1
            if game["home_score"] > 3.5:
                form["away_over_3_5_conceded"] += 1

    num_games = len(games)
    if num_games > 0:
        form["avg_scored"] /= num_games
        form["avg_conceded"] /= num_games
        form["avg_goal_dif"] /= num_games

    return form


def get_games_between_teams(team1: str, team2: str) -> List[Dict]:
    logging.info(f'Querying head-to-head games between {team1} and {team2}')
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: 0)
      |> filter(fn: (r) => r._measurement == "match_results")
      |> filter(fn: (r) => (r.home_team == "{team1}" and r.away_team == "{team2}") or (r.home_team == "{team2}" and r.away_team == "{team1}"))
      |> sort(columns: ["_time"], desc: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    tables = query_influxdb(query)
    games = [
        {
            "time": record.get_time(),
            "home_team": record.values.get("home_team"),
            "away_team": record.values.get("away_team"),
            "tournament": record.values.get("tournament"),
            "country": record.values.get("country"),
            "neutral": record.values.get("neutral"),
            "home_score": int(record.values.get("home_score", 0)),
            "away_score": int(record.values.get("away_score", 0))
        }
        for table in tables for record in table.records
    ]
    games.sort(key=lambda x: x["time"])  # Sort games by time
    logging.info(f'Found {len(games)} head-to-head games between {team1} and {team2}')
    return games


def valid_scores(home_score: int, away_score: int) -> bool:
    return home_score is not None and away_score is not None


def create_features(team1: str, team2: str) -> np.ndarray:
    logging.info(f'Creating features for match: {team1} vs {team2}')
    form_team1 = get_team_form(team1)
    form_team2 = get_team_form(team2)
    games_between = get_games_between_teams(team1, team2)

    h2h_games = len(games_between)
    h2h_team1_wins = sum(1 for game in games_between if valid_scores(game["home_score"], game["away_score"]) and (
            (game["home_team"] == team1 and game["home_score"] > game["away_score"]) or (
            game["away_team"] == team1 and game["away_score"] > game["home_score"])))
    h2h_team2_wins = sum(1 for game in games_between if valid_scores(game["home_score"], game["away_score"]) and (
            (game["home_team"] == team2 and game["home_score"] > game["away_score"]) or (
            game["away_team"] == team2 and game["away_score"] > game["home_score"])))
    h2h_draws = h2h_games - h2h_team1_wins - h2h_team2_wins

    ranking_team1 = UEFA_TEAM_RANKINGS.get(team1, 30)
    ranking_team2 = UEFA_TEAM_RANKINGS.get(team2, 30)

    features = [
        form_team1["avg_scored"], form_team1["avg_goal_dif"], form_team1["avg_conceded"],
        form_team1["wins"], form_team1["draws"], form_team1["losses"],
        form_team1["home_total_scored"], form_team1["home_total_conceded"],
        form_team1["away_total_scored"], form_team1["away_total_conceded"],
        form_team1["home_over_0_5_scored"], form_team1["home_over_1_5_scored"], form_team1["home_over_2_5_scored"],
        form_team1["home_over_3_5_scored"],
        form_team1["home_over_0_5_conceded"], form_team1["home_over_1_5_conceded"],
        form_team1["home_over_2_5_conceded"], form_team1["home_over_3_5_conceded"],
        form_team1["away_over_0_5_scored"], form_team1["away_over_1_5_scored"], form_team1["away_over_2_5_scored"],
        form_team1["away_over_3_5_scored"],
        form_team1["away_over_0_5_conceded"], form_team1["away_over_1_5_conceded"],
        form_team1["away_over_2_5_conceded"], form_team1["away_over_3_5_conceded"],
        form_team2["avg_scored"], form_team2["avg_goal_dif"], form_team2["avg_conceded"],
        form_team2["wins"], form_team2["draws"], form_team2["losses"],
        form_team2["home_total_scored"], form_team2["home_total_conceded"],
        form_team2["away_total_scored"], form_team2["away_total_conceded"],
        form_team2["home_over_0_5_scored"], form_team2["home_over_1_5_scored"], form_team2["home_over_2_5_scored"],
        form_team2["home_over_3_5_scored"],
        form_team2["home_over_0_5_conceded"], form_team2["home_over_1_5_conceded"],
        form_team2["home_over_2_5_conceded"], form_team2["home_over_3_5_conceded"],
        form_team2["away_over_0_5_scored"], form_team2["away_over_1_5_scored"], form_team2["away_over_2_5_scored"],
        form_team2["away_over_3_5_scored"],
        form_team2["away_over_0_5_conceded"], form_team2["away_over_1_5_conceded"],
        form_team2["away_over_2_5_conceded"], form_team2["away_over_3_5_conceded"],
        h2h_team1_wins, h2h_team2_wins, h2h_draws, ranking_team1, ranking_team2
    ]
    logging.info(f'Features created for match: {team1} vs {team2}')
    return np.array(features)


def tune_hyperparameters(features: np.ndarray, targets: np.ndarray, model_type: str) -> Dict:
    logging.info(f'Tuning hyperparameters for {model_type} model')

    def objective(trial: optuna.trial.Trial, features: np.ndarray, targets: np.ndarray, model_type: str) -> float:
        if model_type == 'ridge':
            alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
            model = Ridge(alpha=alpha, positive=True)  # Ensure non-negative predictions
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        elif model_type == 'logistic':
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            model = LogisticRegression(C=C, max_iter=1000)
            scorer = make_scorer(accuracy_score)
        elif model_type == 'xgboost':
            if len(np.unique(targets)) > 2:
                model = XGBRegressor(objective='reg:squarederror')
                scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            else:
                model = XGBClassifier(objective='binary:logistic')
                scorer = make_scorer(accuracy_score)
            model.max_depth = trial.suggest_int('max_depth', 2, 10)
            model.learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            model.n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        kf = KFold(n_splits=5)
        scores = cross_val_score(model, features, targets, cv=kf, scoring=scorer)
        return scores.mean()

    study = optuna.create_study(direction='maximize' if model_type != 'ridge' else 'minimize')
    study.optimize(lambda trial: objective(trial, features, targets, model_type), n_trials=50)
    return study.best_params


def create_nn_model(input_shape: int, layers: list, dropout_rate: float) -> Sequential:
    model = Sequential()
    for units in layers:
        model.add(Dense(units, activation='relu', input_dim=input_shape if len(model.layers) == 0 else None))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes for match result
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_nn_models(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    input_shape: int) -> list:
    architectures = [
        {'layers': [128, 64, 32], 'dropout_rate': 0.5},
        {'layers': [256, 128, 64], 'dropout_rate': 0.3},
        {'layers': [64, 32, 16], 'dropout_rate': 0.4}
    ]
    models = []
    for arch in architectures:
        model = create_nn_model(input_shape, arch['layers'], arch['dropout_rate'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32,
                  callbacks=[early_stopping])
        models.append(model)
    return models


def train_models() -> None:
    features, home_goals, away_goals, both_teams_to_score, match_results, poly, scaler = create_dataset()
    if features.size == 0:
        logging.error('No data available for training models.')
        return

    X_train, X_val, y_train_home, y_val_home = train_test_split(features, home_goals, test_size=0.2, random_state=42)
    _, _, y_train_away, y_val_away = train_test_split(features, away_goals, test_size=0.2, random_state=42)
    _, _, y_train_bt, y_val_bt = train_test_split(features, both_teams_to_score, test_size=0.2, random_state=42)
    _, _, y_train_result, y_val_result = train_test_split(features, match_results, test_size=0.2, random_state=42)

    # Train traditional models
    logging.info('Tuning hyperparameters for home goals model')
    best_params_home_goals = tune_hyperparameters(X_train, y_train_home, 'ridge')
    logging.info('Tuning hyperparameters for away goals model')
    best_params_away_goals = tune_hyperparameters(X_train, y_train_away, 'ridge')
    logging.info('Tuning hyperparameters for both teams to score model')
    best_params_bt = tune_hyperparameters(X_train, y_train_bt, 'logistic')
    logging.info('Tuning hyperparameters for match result model')
    best_params_result = tune_hyperparameters(X_train, y_train_result, 'xgboost')

    home_goals_model = Ridge(**best_params_home_goals, positive=True).fit(X_train, y_train_home)
    away_goals_model = Ridge(**best_params_away_goals, positive=True).fit(X_train, y_train_away)
    both_teams_to_score_model = LogisticRegression(**best_params_bt, max_iter=1000).fit(X_train, y_train_bt)
    match_result_model = XGBClassifier(**best_params_result).fit(X_train, y_train_result)

    # Train neural network models for match result prediction
    nn_models_result = train_nn_models(X_train, y_train_result, X_val, y_val_result, X_train.shape[1])

    models = {
        'home_goals': home_goals_model,
        'away_goals': away_goals_model,
        'both_teams_to_score': both_teams_to_score_model,
        'match_result': match_result_model,
        'nn_match_results': nn_models_result
    }

    save_models(models, poly, scaler)
    logging.info('Models trained and saved successfully')


def save_models(models: Dict[str, Any], poly_features: PolynomialFeatures, scaler: StandardScaler) -> None:
    for name, model in models.items():
        if name == 'nn_match_results':
            for i, nn_model in enumerate(model):
                nn_model.save(f'{MODEL_FILES[name]}_{i}')
        else:
            joblib.dump(model, MODEL_FILES[name])
    joblib.dump(poly_features, MODEL_FILES['poly_features'])
    joblib.dump(scaler, MODEL_FILES['scaler'])


def load_models() -> Dict[str, Any]:
    models = {}
    for name in MODEL_FILES.keys():
        if name == 'nn_match_results':
            nn_models = []
            for i in range(3):  # Number of neural network models you saved
                nn_models.append(keras.models.load_model(f'{MODEL_FILES[name]}_{i}'))
            models[name] = nn_models
        else:
            models[name] = joblib.load(MODEL_FILES[name])
    return models


def predict_match_outcome(team1: str, team2: str) -> Dict[str, Any]:
    try:
        models = load_models()
        poly = joblib.load(MODEL_FILES['poly_features'])
        scaler = joblib.load(MODEL_FILES['scaler'])

        feature = create_features(team1, team2).reshape(1, -1)
        feature = poly.transform(feature)
        feature = scaler.transform(feature)

        home_goals = models['home_goals'].predict(feature)[0]
        away_goals = models['away_goals'].predict(feature)[0]
        bt_score = models['both_teams_to_score'].predict(feature)[0]
        match_result = models['match_result'].predict(feature)[0]

        nn_results = []
        for nn_model in models['nn_match_results']:
            nn_results.append(nn_model.predict(feature).argmax(axis=1)[0])

        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'both_teams_to_score': bt_score,
            'match_result': match_result,
            'nn_match_results': nn_results
        }
    except Exception as e:
        logging.error('Error predicting match outcome: %s', e)
        return {}


if __name__ == '__main__':
    train_models()
    matches = [
        ("Germany", "Scotland"),
        ("Hungary", "Switzerland"),
        ("Spain", "Croatia"),
        ("Italy", "Albania"),
        ("Poland", "Netherlands"),
        ("Slovenia", "Denmark"),
        ("Serbia", "England"),
        ("Romania", "Ukraine"),
        ("Belgium", "Slovakia"),
        ("Austria", "France"),
        ("Turkey", "Georgia"),
        ("Portugal", "Czechia")
    ]
    for match in matches:
        team1, team2 = match
        predictions = predict_match_outcome(team1, team2)
        print(f"Predicted outcome for {team1} vs {team2}")
        print(f"  Game Result Prediction: {predictions.get('match_result', 'N/A')}")
        print(f"  Game Result Prediction: {predictions.get('nn_match_results', 'N/A')}")
        print(f"  Predicted Home Goals: {predictions.get('home_goals', 'N/A'):.2f}")
        print(f"  Predicted Away Goals: {predictions.get('away_goals', 'N/A'):.2f}")
        print(f"  Both Teams to Score: {predictions.get('both_teams_to_score', 'N/A')}\n")