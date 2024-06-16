import logging
from flask import Flask, jsonify, request, render_template
import joblib
from train_model import create_features, get_games_between_teams, get_team_form, get_team_games

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Model file paths
MODEL_FILES = {
    'home_goals': 'home_goals_model.joblib',
    'away_goals': 'away_goals_model.joblib',
    'both_teams_to_score': 'both_teams_to_score_model.joblib',
    'match_result': 'match_result_model.joblib'
}
POLY_FEATURES_FILE = 'poly_features.joblib'
SCALER_FILE = 'scaler.joblib'

# Placeholder for models and preprocessing objects
models = {}
poly_features = None
scaler = None
models_loaded = False


def load_models():
    global models_loaded, poly_features, scaler
    if not models_loaded:
        try:
            for key, file in MODEL_FILES.items():
                models[key] = joblib.load(file)
                logging.info(f'{key} model loaded successfully.')
            poly_features = joblib.load(POLY_FEATURES_FILE)
            scaler = joblib.load(SCALER_FILE)
            models_loaded = True
        except Exception as e:
            logging.error(f'Error loading models: {e}')
            raise


@app.before_request
def initialize_models():
    load_models()


def make_predictions(team1, team2):
    try:
        features = create_features(team1, team2).reshape(1, -1)

        # Apply the same preprocessing steps as during training
        features_scaled = scaler.transform(features)
        features_poly = poly_features.transform(features_scaled)

        predictions = {
            'home_goals': round(models['home_goals'].predict(features_poly)[0], 2),
            'away_goals': round(models['away_goals'].predict(features_poly)[0], 2),
            'both_teams_to_score_prob': round((models['both_teams_to_score'].predict_proba(features_poly)[0][1]) * 100, 2),
            'match_result': models['match_result'].predict(features_poly)[0]
        }

        result_map = {0: "Draw", 1: f"{team1}", 2: f"{team2}"}
        predictions['match_result_text'] = result_map[predictions['match_result']]
        return predictions
    except Exception as e:
        logging.error(f'Error making predictions: {e}')
        return None


@app.route('/api/predict', methods=['GET'])
def api_predict():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')
    if not team1 or not team2:
        return jsonify({"error": "Missing team parameters"}), 400

    predictions = make_predictions(team1, team2)
    if predictions:
        return jsonify(predictions)
    else:
        return jsonify({"error": "Prediction error"}), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/groups')
def groups():
    # Data for groups and fixtures
    groups_data = {
        "Group A": [
            {"team": "Germany", "pl": 1, "w": 1, "d": 0, "l": 0, "f": 5, "a": 1, "gd": 4, "pts": 3},
            {"team": "Switzerland", "pl": 1, "w": 1, "d": 0, "l": 0, "f": 3, "a": 1, "gd": 2, "pts": 3},
            {"team": "Hungary", "pl": 1, "w": 0, "d": 0, "l": 1, "f": 1, "a": 3, "gd": -2, "pts": 0},
            {"team": "Scotland", "pl": 1, "w": 0, "d": 0, "l": 1, "f": 1, "a": 5, "gd": -4, "pts": 0}
        ],
        "Group B": [
            {"team": "Spain", "pl": 1, "w": 1, "d": 0, "l": 0, "f": 3, "a": 0, "gd": 3, "pts": 3},
            {"team": "Italy", "pl": 1, "w": 1, "d": 0, "l": 0, "f": 2, "a": 1, "gd": 1, "pts": 3},
            {"team": "Albania", "pl": 1, "w": 0, "d": 0, "l": 1, "f": 1, "a": 2, "gd": -1, "pts": 0},
            {"team": "Croatia", "pl": 1, "w": 0, "d": 0, "l": 1, "f": 0, "a": 3, "gd": -3, "pts": 0}
        ],
        "Group C": [
            {"team": "Slovenia", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Denmark", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Serbia", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "England", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0}
        ],
        "Group D": [
            {"team": "Poland", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Netherlands", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Austria", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "France", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0}
        ],
        "Group E": [
            {"team": "Belgium", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Slovakia", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Romania", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Ukraine", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0}
        ],
        "Group F": [
            {"team": "Turkey", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Georgia", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Portugal", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0},
            {"team": "Czech Republic", "pl": 0, "w": 0, "d": 0, "l": 0, "f": 0, "a": 0, "gd": 0, "pts": 0}
        ]
    }
    fixtures = [
        {"date": "Friday June 14", "match": "Germany 5-1 Scotland"},
        {"date": "Saturday June 15", "matches": ["Hungary 1-3 Switzerland", "Spain 3-0 Croatia", "Italy 2-1 Albania"]},
        {"date": "Sunday June 16", "matches": ["Poland vs Netherlands (Hamburg, kick-off 2pm UK time)", "Slovenia vs Denmark (Stuttgart, kick-off 5pm UK time)", "Serbia vs England (Gelsenkirchen, kick-off 8pm UK time)"]},
        {"date": "Monday June 17", "matches": ["Romania vs Ukraine (Munich, kick-off 2pm UK time)", "Belgium vs Slovakia (Frankfurt, kick-off 5pm UK time)", "Austria vs France (Dusseldorf, kick-off 8pm UK time)"]},
        {"date": "Tuesday June 18", "matches": ["Turkey vs Georgia (Dortmund, kick-off 5pm UK time)", "Portugal vs Czech Republic (Leipzig, kick-off 8pm UK time)"]},
        {"date": "Wednesday June 19", "matches": ["Croatia vs Albania (Hamburg, kick-off 2pm UK time)", "Germany vs Hungary (Stuttgart, kick-off 5pm UK time)", "Scotland vs Switzerland (Cologne, kick-off 8pm UK time)"]},
        {"date": "Thursday June 20", "matches": ["Slovenia vs Serbia (Munich, kick-off 2pm UK time)", "Denmark vs England (Frankfurt, kick-off 5pm UK time)", "Spain vs Italy (Gelsenkirchen, kick-off 8pm UK time)"]},
        {"date": "Friday June 21", "matches": ["Slovakia vs Ukraine (Dusseldorf, kick-off 2pm UK time)", "Poland vs Austria (Berlin, kick-off 5pm UK time)", "Netherlands vs France (Leipzig, kick-off 8pm UK time)"]},
        {"date": "Saturday June 22", "matches": ["Georgia vs Czech Republic (Hamburg, kick-off 2pm UK time)", "Turkey vs Portugal (Dortmund, kick-off 5pm UK time)", "Belgium vs Romania (Cologne, kick-off 8pm UK time)"]},
        {"date": "Sunday June 23", "matches": ["Switzerland vs Germany (Frankfurt, kick-off 8pm UK time)", "Scotland vs Hungary (Stuttgart, kick-off 8pm UK time)"]},
        {"date": "Monday June 24", "matches": ["Croatia vs Italy (Leipzig, kick-off 8pm UK time)", "Albania vs Spain (Dusseldorf, kick-off 8pm UK time)"]},
        {"date": "Tuesday June 25", "matches": ["Netherlands vs Austria (Berlin, kick-off 5pm UK time)", "France vs Poland (Dortmund, kick-off 5pm UK time)", "England vs Slovenia (Cologne, kick-off 8pm UK time)", "Denmark vs Serbia (Munich, kick-off 8pm UK time)"]},
        {"date": "Wednesday June 26", "matches": ["Slovakia vs Romania (Frankfurt, kick-off 5pm UK time)", "Ukraine vs Belgium (Stuttgart, kick-off 5pm UK time)", "Czech Republic vs Turkey (Hamburg, kick-off 8pm UK time)", "Georgia vs Portugal (Gelsenkirchen, kick-off 8pm UK time)"]}
    ]
    return render_template('groups.html', groups_data=groups_data, fixtures=fixtures)


if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.critical(f'Application startup failed: {e}')
