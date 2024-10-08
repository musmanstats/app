import streamlit as st
import seaborn as sns
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Machine Learning App for Titanic Data")
st.subheader("This will use Random Forest Regression to predict Fare based on other features")
st.subheader("`Muhammad Usman`")

# make containers
data_sets = st.container()
features = st.container()
model_training = st.container()

with data_sets:
    st.header("Data Set and Data Visualizations")
    st.text("We will work with `titanic` data set")
    # import data
    df = sns.load_dataset('titanic')
    st.write(df.head())
    df['survived'] = pd.Categorical(df['survived'], categories=[0, 1], ordered=True)
    df['survived_label'] = df['survived'].map({0: 'Not Survived', 1: 'Survived'})
    st.subheader("Data file after dropping `missing` observations from `age` data")
    df = df.dropna(subset=['age'])
    st.write(df.shape)

# Sidebar for plot selections
    st.sidebar.header("Visualization Settings")
    # Scatter Plot with Hue
    st.subheader("Scatter Plot with Hue")
    hue_scatter = st.sidebar.selectbox("Hue for Scatter Plot", ['survived_label', 'sex', 'class', 'who'])

    fig_scatter = px.scatter(df, x='age', y='fare', color=hue_scatter,
                             title=f"Scatter Plot of Age vs Fare with Hue {hue_scatter}")
    st.plotly_chart(fig_scatter)

    # Bar Chart with Hue
    st.subheader("Bar Chart with Hue")
    hue_bar = st.sidebar.selectbox("Hue for Bar Chart", ['sex', 'class', 'who', 'embark_town'])

    fig_bar = px.bar(df, x='survived_label', color=hue_bar, 
                     title=f"Bar Chart of sex Hue {hue_bar}", barmode="group")
    st.plotly_chart(fig_bar)

with features:
    st.header("Features")
    st.text("This section showes the featuers of the `titanic` data set")

with model_training:
    st.header("Model training")
    st.subheader("Prediction model to predict fare")
    # making columns
    input, display = st.columns(2)
    max_depth = input.slider("How many people do you know?", min_value=10, max_value=100, value=20, step=5)
    n_estimators = input.selectbox("How many trees in RF?", options=[50, 100, 150, 200, 'No limit'])

    # input features
        # input_features = input.text_input('Which feature to be used?')
        # input_features = input.multiselect('Select features to be used (e.g., age, class, sex)', options=df.columns.tolist())
    
    # Input features: allow multiple numeric features to be selected
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Use the filtered numeric columns in the multiselect widget
    if 'fare' in numeric_features:
        numeric_features.remove('fare')
    input_features = input.multiselect('Select numeric features to be used (e.g., age, fare)', options=numeric_features)

    # ML model
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    if n_estimators == 'No limit':
        model = RandomForestRegressor(max_depth=max_depth)
    else:
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    # define X and y
    y=df[['fare']]
    X=df[input_features]

    model.fit(X, y)
    pred = model.predict(X)

    # Display metrics
    display.subheader("Mean absolute error: ")
    display.write(mean_absolute_error(y, pred))
    display.subheader("Mean squared error: ")
    display.write(mean_squared_error(y, pred))
    display.subheader("R squared score: ")
    display.write(r2_score(y, pred))
