import os
import logging
import numpy as np
import pandas as pd
import joblib
from influxdb_client import InfluxDBClient
from typing import Dict, Any, List
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import load_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress Convergence Warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "PinaCode")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "Euros2024")

# File paths for model persistence
MODEL_FILES = {
    'home_goals': 'home_goals_model.h5',
    'away_goals': 'away_goals_model.h5',
    'both_teams_to_score': 'both_teams_to_score_model.h5',
    'match_result': 'match_result_model.h5',
    'poly_features': 'poly_features.joblib',
    'scaler': 'scaler.joblib'
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
      |> range(start: -9y)
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
    teams = {record.values.get("home_team") or record.values.get("away_team") for table in tables for record in table.records}
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

    features = np.array(features).reshape(len(features), -1)

    poly = PolynomialFeatures(degree=2, interaction_only=True)
    features_poly = poly.fit_transform(features)
    scaler = StandardScaler().fit(features_poly)
    features_poly_scaled = scaler.transform(features_poly)

    return features_poly_scaled, np.array(targets_home), np.array(targets_away), np.array(targets_bt), np.array(targets_result), poly, scaler

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
        "home_over_0_5_conceded": 0, "home_over_1_5_conceded": 0, "home_over_2_5_conceded": 0, "home_over_3_5_conceded": 0,
        "away_over_0_5_scored": 0, "away_over_1_5_scored": 0, "away_over_2_5_scored": 0, "away_over_3_5_scored": 0,
        "away_over_0_5_conceded": 0, "away_over_1_5_conceded": 0, "away_over_2_5_conceded": 0, "away_over_3_5_conceded": 0
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
            if game["home_score"] > 3.5":
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

def create_nn_model(input_shape: int, layers: list, dropout_rate: float, output_units: int, output_activation: str) -> Sequential:
    model = Sequential()
    for units in layers:
        model.add(Dense(units, activation='relu', input_dim=input_shape if len(model.layers) == 0 else None))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(output_units, activation=output_activation))
    model.compile(optimizer='adam', loss='mean_squared_error' if output_activation == 'linear' else 'sparse_categorical_crossentropy', metrics=['accuracy'] if output_activation == 'softmax' else ['mse'])
    return model

def train_nn_models(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    input_shape: int, output_units: int, output_activation: str) -> Sequential:
    architectures = [
        {'layers': [128, 64, 32], 'dropout_rate': 0.5},
        {'layers': [256, 128, 64], 'dropout_rate': 0.3},
        {'layers': [64, 32, 16], 'dropout_rate': 0.4}
    ]
    best_model = None
    best_val_loss = float('inf')
    for arch in architectures:
        model = create_nn_model(input_shape, arch['layers'], arch['dropout_rate'], output_units, output_activation)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    return best_model

def train_models() -> None:
    features, home_goals, away_goals, both_teams_to_score, match_results, poly, scaler = create_dataset()
    if features.size == 0:
        logging.error('No data available for training models.')
        return

    X_train, X_val, y_train_home, y_val_home = train_test_split(features, home_goals, test_size=0.2, random_state=42)
    _, _, y_train_away, y_val_away = train_test_split(features, away_goals, test_size=0.2, random_state=42)
    _, _, y_train_bt, y_val_bt = train_test_split(features, both_teams_to_score, test_size=0.2, random_state=42)
    _, _, y_train_result, y_val_result = train_test_split(features, match_results, test_size=0.2, random_state=42)

    logging.info('Training neural network for home goals')
    home_goals_model = train_nn_models(X_train, y_train_home, X_val, y_val_home, X_train.shape[1], 1, 'linear')
    logging.info('Training neural network for away goals')
    away_goals_model = train_nn_models(X_train, y_train_away, X_val, y_val_away, X_train.shape[1], 1, 'linear')
    logging.info('Training neural network for both teams to score')
    both_teams_to_score_model = train_nn_models(X_train, y_train_bt, X_val, y_val_bt, X_train.shape[1], 1, 'sigmoid')
    logging.info('Training neural network for match result')
    match_result_model = train_nn_models(X_train, y_train_result, X_val, y_val_result, X_train.shape[1], 3, 'softmax')

    models = {
        'home_goals': home_goals_model,
        'away_goals': away_goals_model,
        'both_teams_to_score': both_teams_to_score_model,
        'match_result': match_result_model
    }

    save_models(models, poly, scaler)
    logging.info('Models trained and saved successfully')

def save_models(models: Dict[str, Any], poly_features: PolynomialFeatures, scaler: StandardScaler) -> None:
    for name, model in models.items():
        model.save(MODEL_FILES[name])
    joblib.dump(poly_features, MODEL_FILES['poly_features'])
    joblib.dump(scaler, MODEL_FILES['scaler'])

def load_models() -> Dict[str, Any]:
    models = {}
    for name in MODEL_FILES.keys():
        if name in ['poly_features', 'scaler']:
            models[name] = joblib.load(MODEL_FILES[name])
        else:
            models[name] = load_model(MODEL_FILES[name])
    return models

def predict_match_outcome(team1: str, team2: str) -> Dict[str, Any]:
    try:
        models = load_models()
        poly = models['poly_features']
        scaler = models['scaler']

        feature = create_features(team1, team2).reshape(1, -1)
        feature = poly.transform(feature)
        feature = scaler.transform(feature)

        home_goals = models['home_goals'].predict(feature)[0][0]
        away_goals = models['away_goals'].predict(feature)[0][0]
        bt_score = models['both_teams_to_score'].predict(feature)[0][0]
        match_result = models['match_result'].predict(feature).argmax(axis=1)[0]

        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'both_teams_to_score': bt_score,
            'match_result': match_result
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
        print(f"  Predicted Home Goals: {predictions.get('home_goals', 'N/A'):.2f}")
        print(f"  Predicted Away Goals: {predictions.get('away_goals', 'N/A'):.2f}")
        print(f"  Both Teams to Score: {predictions.get('both_teams_to_score', 'N/A')}\n")
