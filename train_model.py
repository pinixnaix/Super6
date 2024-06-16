import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
from influxdb_client import InfluxDBClient
from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import optuna
from sklearn.metrics import make_scorer, mean_absolute_error, accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.ensemble import VotingClassifier

# Suppress Convergence Warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "super6")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "international")
MODEL_FILES = {
    'home_goals': 'home_goals_model.joblib',
    'away_goals': 'away_goals_model.joblib',
    'both_teams_to_score': 'both_teams_to_score_model.joblib',
    'match_result': 'match_result_model.joblib',
    'poly_features': 'poly_features.joblib',
    'scaler': 'scaler.joblib',
    'dataset': 'dataset.joblib'
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


def save_models(models: Dict[str, Any], poly_features: PolynomialFeatures, scaler: StandardScaler) -> None:
    for name, model in models.items():
        joblib.dump(model, MODEL_FILES[name])
    joblib.dump(poly_features, MODEL_FILES['poly_features'])
    joblib.dump(scaler, MODEL_FILES['scaler'])


def load_dataset() -> Tuple:
    try:
        return joblib.load(MODEL_FILES['dataset'])
    except Exception as e:
        logging.error('Error loading dataset: %s', e)
        raise DataLoadError('Failed to load dataset')


def get_team_games(team_name: str) -> pd.DataFrame:
    logging.info(f'Querying games for team: {team_name}')
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -4y)
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
      |> range(start: -4y)
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


def create_dataset(force_recreate: bool = False) -> Tuple:
    if os.path.exists(MODEL_FILES['dataset']) and not force_recreate:
        logging.info('Loading existing dataset')
        return load_dataset()

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
    features = poly.fit_transform(features)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    dataset = (
    features, np.array(targets_home), np.array(targets_away), np.array(targets_bt), np.array(targets_result), poly,
    scaler)
    joblib.dump(dataset, MODEL_FILES['dataset'])
    logging.info('Dataset saved successfully')
    return dataset


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


def get_team_form(team: str, num_games: int = 4) -> Dict[str, float]:
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -2y)
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


def valid_scores(home_score: int, away_score: int) -> bool:
    return home_score is not None and away_score is not None


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


def train_models() -> None:
    features, home_goals, away_goals, both_teams_to_score, match_results, poly, scaler = create_dataset()
    if features.size == 0:
        logging.error('No data available for training models.')
        return

    X_train, X_test, y_train_home, y_test_home = train_test_split(features, home_goals, test_size=0.2, random_state=42)
    _, _, y_train_away, y_test_away = train_test_split(features, away_goals, test_size=0.2, random_state=42)
    _, _, y_train_bt, y_test_bt = train_test_split(features, both_teams_to_score, test_size=0.2, random_state=42)
    _, _, y_train_result, y_test_result = train_test_split(features, match_results, test_size=0.2, random_state=42)

    logging.info('Tuning hyperparameters for home goals model')
    best_params_home_goals = tune_hyperparameters(X_train, y_train_home, 'ridge')
    logging.info('Tuning hyperparameters for away goals model')
    best_params_away_goals = tune_hyperparameters(X_train, y_train_away, 'ridge')
    logging.info('Tuning hyperparameters for both teams to score model')
    best_params_bt = tune_hyperparameters(X_train, y_train_bt, 'logistic')
    logging.info('Tuning hyperparameters for match result model')
    best_params_result = tune_hyperparameters(X_train, y_train_result, 'xgboost')

    home_goals_model = Ridge(**best_params_home_goals, positive=True).fit(X_train,
                                                                          y_train_home)  # Ensure non-negative predictions
    away_goals_model = Ridge(**best_params_away_goals, positive=True).fit(X_train,
                                                                          y_train_away)  # Ensure non-negative predictions
    both_teams_to_score_model = LogisticRegression(**best_params_bt, max_iter=1000).fit(X_train, y_train_bt)
    match_result_model = XGBClassifier(**best_params_result).fit(X_train, y_train_result)

    voting_classifier = VotingClassifier(estimators=[
        ('lr', LogisticRegression(**best_params_bt, max_iter=1000)),
        ('xgb', XGBClassifier(**best_params_result)),
    ], voting='soft')

    voting_classifier.fit(X_train, y_train_result)

    models = {
        'home_goals': home_goals_model,
        'away_goals': away_goals_model,
        'both_teams_to_score': both_teams_to_score_model,
        'match_result': voting_classifier
    }

    save_models(models, poly, scaler)

    logging.info('Models trained and saved successfully')


def predict_match_outcome(team1: str, team2: str) -> Dict[str, Any]:
    try:
        models = {
            'home_goals': joblib.load(MODEL_FILES['home_goals']),
            'away_goals': joblib.load(MODEL_FILES['away_goals']),
            'both_teams_to_score': joblib.load(MODEL_FILES['both_teams_to_score']),
            'match_result': joblib.load(MODEL_FILES['match_result']),
        }
        poly = joblib.load(MODEL_FILES['poly_features'])
        scaler = joblib.load(MODEL_FILES['scaler'])
    except FileNotFoundError as e:
        logging.error('Model file not found: %s', e)
        return {}

    feature = create_features(team1, team2).reshape(1, -1)
    feature = poly.transform(feature)
    feature = scaler.transform(feature)

    home_goals = max(0, round(models['home_goals'].predict(feature)[0], 2))  # Ensure non-negative predictions
    away_goals = max(0, round(models['away_goals'].predict(feature)[0], 2))  # Ensure non-negative predictions
    both_teams_to_score = models['both_teams_to_score'].predict(feature)[0]
    match_result = models['match_result'].predict(feature)[0]

    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "both_teams_to_score": bool(both_teams_to_score),
        "match_result": ["Draw", team1, team2][int(match_result)]
    }


if __name__ == '__main__':
    # train_models()
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
