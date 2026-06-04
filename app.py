import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load & prepare data 

def load_data():
    data_file = "energydata_complete.xlsx"

    # If the file exists in the project, read it.so 
    if os.path.exists(data_file):
        df = pd.read_excel(data_file)
    else:
        # Offer a Streamlit uploader so the app can run in the browser without manual file copy.
        uploaded = st.file_uploader(
            "Upload dataset (energydata_complete.xlsx) or place the file in the project root.",
            type=["xlsx", "csv"],
        )
        if uploaded is not None:
            try:
                # read uploaded file (works for both xlsx and csv)
                content = uploaded.read()
                try:
                    df = pd.read_excel(BytesIO(content))
                except Exception:
                    df = pd.read_csv(BytesIO(content))
            except Exception as e:
                st.error("Failed to read uploaded file: " + str(e))
                st.stop()
        else:
            st.error(
                f"Dataset '{data_file}' not found. Please upload it here or place it in: {os.getcwd()}"
            )
            st.stop()

    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        st.error("Failed to parse 'date' column from dataset: " + str(e))
        st.stop()

    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df.drop('date', axis=1, inplace=True)
    return df

df = load_data()

X = df.drop('Appliances', axis=1)
y = df['Appliances']

# Train model (cached – runs ONCE, not on every slider move) 

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    r2 = r2_score(y_test, model.predict(X_test_sc))
    return model, scaler, r2

model, scaler, r2 = train_model()

# UI 

st.title("⚡ Appliance Energy Consumption Prediction")
st.markdown(f"**Model R² Score:** `{r2:.2f}`")
st.markdown("---")

st.markdown("### Enter sensor values:")

col1, col2 = st.columns(2)

with col1:
    t1  = st.slider("T1 – Kitchen temp (°C)",     0.0, 50.0, 20.0, 0.5)
    t2  = st.slider("T2 – Living room temp (°C)",  0.0, 50.0, 20.0, 0.5)
    t6  = st.slider("T6 – Outside temp (°C)",     -10.0, 45.0, 10.0, 0.5)

with col2:
    rh1  = st.slider("RH_1 – Kitchen humidity (%)",      0.0, 100.0, 50.0, 1.0)
    rh2  = st.slider("RH_2 – Living room humidity (%)",  0.0, 100.0, 50.0, 1.0)
    wind = st.slider("Wind speed (m/s)",                 0.0, 15.0,  5.0, 0.1)

# Build input row using column MEANS for unspecified features 
#     (much better than filling with 0 which breaks scaling)

col_means = X.mean()

input_data = col_means.to_frame().T.copy()

# Override the 6 features the user actually controls
input_data['T1']        = t1
input_data['T2']        = t2
input_data['T6']        = t6
input_data['RH_1']      = rh1
input_data['RH_2']      = rh2
input_data['Windspeed'] = wind   # exact column name from dataset

input_data = input_data[X.columns]          # enforce column order
input_scaled = scaler.transform(input_data)

#Predict (live – no button needed) 

prediction = model.predict(input_scaled)[0]
prediction = max(0, prediction)             # energy can't be negative

st.markdown("---")
st.metric(label="Predicted Energy Consumption", value=f"{prediction:.1f} Wh")

# Dynamic chart: highlight where the prediction falls 

st.markdown("### Energy Distribution")
st.caption("The red line shows where your predicted value sits in the dataset.")

fig = px.histogram(
    df,
    x="Appliances",
    nbins=60,
    labels={"Appliances": "Energy (Wh)", "count": "Count"},
    color_discrete_sequence=["#5b8dee"],
)

fig.add_vline(x=prediction, line_color="red", line_width=2)

fig.add_annotation(
    x=prediction,
    y=0.95,
    xref="x",
    yref="paper",
    text=f"Your prediction: {prediction:.1f} Wh",
    showarrow=False,
    font=dict(color="red"),
)

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)

#  Feature importance (bonus insight) 

with st.expander("Show top feature importances"):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top10 = importances.nlargest(10).sort_values()

    fig2 = px.bar(
        top10,
        orientation="h",
        labels={"value": "Importance", "index": "Feature"},
        color_discrete_sequence=["#5b8dee"],
    )
    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

