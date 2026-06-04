from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)

DATA_PATH = "energydata_complete.xlsx"
INPUT_FEATURES = ["T1", "T2", "T6", "RH_1", "RH_2", "Windspeed"]
DEFAULT_VALUES = {
    "T1": 20.0,
    "T2": 20.0,
    "T6": 10.0,
    "RH_1": 50.0,
    "RH_2": 50.0,
    "Windspeed": 5.0,
}


def load_data():
    df = pd.read_excel(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df.drop(columns=["date"], inplace=True)
    return df


def train_model(df):
    X = df.drop(columns=["Appliances"])
    y = df["Appliances"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    r2 = r2_score(y_test, model.predict(X_test_sc))
    return model, scaler, r2, X


df = load_data()
model, scaler, r2_score_value, X = train_model(df)
feature_means = X.mean()


@app.route("/", methods=["GET", "POST"])
def index():
    values = {name: DEFAULT_VALUES[name] for name in INPUT_FEATURES}
    prediction = None

    if request.method == "POST":
        for feature in INPUT_FEATURES:
            raw_value = request.form.get(feature, values[feature])
            try:
                values[feature] = float(raw_value)
            except ValueError:
                values[feature] = values[feature]

        input_data = feature_means.to_frame().T.copy()
        input_data.loc[0, INPUT_FEATURES] = [values[name] for name in INPUT_FEATURES]
        input_data = input_data[X.columns]
        prediction = model.predict(scaler.transform(input_data))[0]
        prediction = float(max(0.0, prediction))

    return render_template(
        "index.html",
        values=values,
        prediction=prediction,
        r2_score=r2_score_value,
        input_features=INPUT_FEATURES,
    )


if __name__ == "__main__":
    app.run(debug=True)
