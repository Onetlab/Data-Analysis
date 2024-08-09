import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import r2_score
import joblib
import os
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.signal import find_peaks
import re
import json

st.set_page_config(layout="wide")

# Function to find peaks and troughs
def find_peaks_troughs(data, distance=20, prominence=1.0, width=1):
    peaks, _ = find_peaks(data, distance=distance, prominence=prominence, width=width)
    troughs, _ = find_peaks(-data, distance=distance, prominence=prominence, width=width)
    return peaks, troughs

# Function to calculate distances between positions
def calculate_distances(positions):
    if len(positions) < 2:
        return []
    return np.abs(np.diff(positions)).tolist()

# Function to calculate alignment between peaks and troughs of two data sets
def calculate_alignment(peaks1, peaks2, troughs1, troughs2):
    peaks1 = np.array(peaks1)
    peaks2 = np.array(peaks2)
    troughs1 = np.array(troughs1)
    troughs2 = np.array(troughs2)
    
    peak_diffs = np.abs(peaks1[:, np.newaxis] - peaks2).min(axis=1)
    trough_diffs = np.abs(troughs1[:, np.newaxis] - troughs2).min(axis=1)
    
    peak_alignment = peak_diffs.mean()
    trough_alignment = trough_diffs.mean()
    
    return peak_alignment, trough_alignment

# Function to add rolling features
def add_rolling_features(df, window=5):
    df['rolling_mean'] = df['dBm'].rolling(window=window).mean()
    df['rolling_std'] = df['dBm'].rolling(window=window).std()
    return df

# Function to extract metadata from filename
def extract_metadata(filename):
    pattern = r"delay_(\d+\.?\d*)_run_(\d+)_step_(\d+\.?\d*)_(forward|reverse)_(\d+\.?\d*)m_(rough|smooth)_port_(\d+)"
    match = re.search(pattern, filename)
    if match:
        return {
            "delay": float(match.group(1)),
            "run": int(match.group(2)),
            "step": float(match.group(3)),
            "direction": match.group(4),
            "base_distance": float(match.group(5)),
            "surface": match.group(6),
            "port": int(match.group(7))
        }
    else:
        return {}

# Function to analyze trends
def analyze_trends(df_list, metadata_list, run_names, params_dict):
    peaks_troughs_data = []
    distances_data = []
    for df, metadata, run_name in zip(df_list, metadata_list, run_names):
        step_size = metadata['step']
        params = params_dict.get(str(step_size), {'distance': 20, 'prominence': 1.0, 'width': 1})
        peaks, troughs = find_peaks_troughs(df['dBm'], distance=params['distance'], prominence=params['prominence'], width=params['width'])
        peaks_positions = df['Position'][peaks].tolist()
        troughs_positions = df['Position'][troughs].tolist()
        
        peak_distances = calculate_distances(peaks_positions)
        trough_distances = calculate_distances(troughs_positions)
        
        for i, pos in enumerate(peaks_positions):
            data = {**metadata, 'Run': run_name, 'Type': 'Peak', 'Position': pos}
            if i < len(peak_distances):
                data['Next_Peak_Distance'] = peak_distances[i]
                distances_data.append({'Run': run_name, 'Type': 'Peak Distance', 'Distance': peak_distances[i], 'Index': i, 'step': metadata['step']})
            peaks_troughs_data.append(data)
        
        for i, pos in enumerate(troughs_positions):
            data = {**metadata, 'Run': run_name, 'Type': 'Trough', 'Position': pos}
            if i < len(trough_distances):
                data['Next_Trough_Distance'] = trough_distances[i]
                distances_data.append({'Run': run_name, 'Type': 'Trough Distance', 'Distance': trough_distances[i], 'Index': i, 'step': metadata['step']})
            peaks_troughs_data.append(data)
    
    peaks_troughs_df = pd.DataFrame(peaks_troughs_data)
    distances_df = pd.DataFrame(distances_data)
    return peaks_troughs_df, distances_df

# Read params from JSON file
def load_params():
    if os.path.exists("params.json"):
        with open("params.json", "r") as file:
            return json.load(file)
    else:
        return {}



# Save params to JSON file
def save_params(params):
    with open("params.json", "w") as file:
        json.dump(params, file, indent=4)

st.title("Interactive Motor Sweep Data Analyzer with ML and DL")

# Load initial parameters
params_dict = load_params()

uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

if uploaded_files:
    data_frames = [add_rolling_features(pd.read_csv(file)) for file in uploaded_files]
    run_names = [file.name for file in uploaded_files]
    metadata_list = [extract_metadata(file.name) for file in uploaded_files]

    st.sidebar.title("Select Runs to Overlay")
    selected_runs = [st.sidebar.checkbox(run, value=True) for run in run_names]

    selected_data_frames = [df for df, selected in zip(data_frames, selected_runs) if selected]
    selected_metadata_list = [metadata for metadata, selected in zip(metadata_list, selected_runs) if selected]
    selected_run_names = [name for name, selected in zip(run_names, selected_runs) if selected]

    st.sidebar.title("Filter Options")
    delay_times = sorted(set(metadata['delay'] for metadata in selected_metadata_list if 'delay' in metadata))
    selected_delay = st.sidebar.multiselect("Select Delay Time", delay_times, delay_times)

    step_sizes = sorted(set(metadata['step'] for metadata in selected_metadata_list if 'step' in metadata))
    selected_step_size = st.sidebar.multiselect("Select Step Size", step_sizes, step_sizes)

    directions = sorted(set(metadata['direction'] for metadata in selected_metadata_list if 'direction' in metadata))
    selected_direction = st.sidebar.multiselect("Select Direction", directions, directions)

    surfaces = sorted(set(metadata['surface'] for metadata in selected_metadata_list if 'surface' in metadata))
    selected_surface = st.sidebar.multiselect("Select Surface", surfaces, surfaces)

    ports = sorted(set(metadata['port'] for metadata in selected_metadata_list if 'port' in metadata))
    selected_port = st.sidebar.multiselect("Select Port", ports, ports)

    st.sidebar.title("Set Peak/Trough Finding Parameters")
    
    # Initialize parameters
    for step in selected_step_size:
        if str(step) not in params_dict:
            params_dict[str(step)] = {'distance': 20, 'prominence': 1.0, 'width': 1}

    # Update parameters with sliders
    for step in selected_step_size:
        st.sidebar.subheader(f"Parameters for Step Size {step}")
        distance = st.sidebar.slider(f"Distance for {step}", 1, 100, params_dict[str(step)]['distance'], key=f"distance_{step}")
        prominence = st.sidebar.slider(f"Prominence for {step}", 0.1, 10.0, params_dict[str(step)]['prominence'], key=f"prominence_{step}")
        width = st.sidebar.slider(f"Width for {step}", 1, 20, params_dict[str(step)]['width'], key=f"width_{step}")
        params_dict[str(step)] = {'distance': distance, 'prominence': prominence, 'width': width}

    st.sidebar.title("Parameters for Averaged Plot")
    if 'avg_plot' not in params_dict:
        params_dict['avg_plot'] = {'distance': 20, 'prominence': 1.0, 'width': 1}

    avg_distance = st.sidebar.slider("Distance for Averaged Plot", 1, 100, params_dict['avg_plot']['distance'])
    avg_prominence = st.sidebar.slider("Prominence for Averaged Plot", 0.1, 10.0, params_dict['avg_plot']['prominence'])
    avg_width = st.sidebar.slider("Width for Averaged Plot", 1, 20, params_dict['avg_plot']['width'])

    params_dict['avg_plot'] = {'distance': avg_distance, 'prominence': avg_prominence, 'width': avg_width}

    # Save updated parameters
    save_params(params_dict)

    filtered_data_frames = []
    filtered_run_names = []
    filtered_metadata_list = []

    for df, metadata, name in zip(selected_data_frames, selected_metadata_list, selected_run_names):
        if ('delay' in metadata and metadata['delay'] in selected_delay and
            metadata['step'] in selected_step_size and
            metadata['direction'] in selected_direction and
            metadata['surface'] in selected_surface and
            metadata['port'] in selected_port):
            filtered_data_frames.append(df)
            filtered_metadata_list.append(metadata)
            filtered_run_names.append(name)

    fig = go.Figure()

    for df, name, metadata in zip(filtered_data_frames, filtered_run_names, filtered_metadata_list):
        step_size = metadata['step']
        params = params_dict.get(str(step_size), {'distance': 20, 'prominence': 1.0, 'width': 1})
        position = df['Position']
        dbm = df['dBm']
        peaks, troughs = find_peaks_troughs(dbm, distance=params['distance'], prominence=params['prominence'], width=params['width'])
        
        fig.add_trace(go.Scatter(x=position, y=dbm, mode='lines', name=name))
        fig.add_trace(go.Scatter(x=position[peaks], y=dbm[peaks], mode='markers', name=f'Peaks {name}', marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=position[troughs], y=dbm[troughs], mode='markers', name=f'Troughs {name}', marker=dict(color='blue')))

    fig.update_layout(title='Overlay of dBm vs Position for Selected Runs',
                      xaxis_title='Position (mm)',
                      yaxis_title='dBm',
                      legend_title='Runs',
                      hovermode='closest',
                      width=1800,  # Set the width of the plot
                      height=900   # Set the height of the plot
                      )

    st.plotly_chart(fig)
    
    if len(filtered_data_frames) >= 2:
        st.subheader("Alignment Calculations")
        alignment_results = []
        
        for i in range(len(filtered_data_frames)):
            for j in range(i + 1, len(filtered_data_frames)):
                df1, df2 = filtered_data_frames[i], filtered_data_frames[j]
                name1, name2 = filtered_run_names[i], filtered_run_names[j]
                
                peaks1, troughs1 = find_peaks_troughs(df1['dBm'], distance=params['distance'], prominence=params['prominence'], width=params['width'])
                peaks2, troughs2 = find_peaks_troughs(df2['dBm'], distance=params['distance'], prominence=params['prominence'], width=params['width'])
                
                peak_alignment, trough_alignment = calculate_alignment(df1['Position'][peaks1], df2['Position'][peaks2],
                                                                       df1['Position'][troughs1], df2['Position'][troughs2])
                
                alignment_results.append({
                    'Run 1': name1,
                    'Run 2': name2,
                    'Peak Alignment (mm)': peak_alignment,
                    'Trough Alignment (mm)': trough_alignment
                })
        
        if alignment_results:
            alignment_df = pd.DataFrame(alignment_results)
            st.dataframe(alignment_df)  # Making the table scrollable
    
    st.subheader("Run Metadata")
    metadata_df = pd.DataFrame(filtered_metadata_list, index=filtered_run_names)
    st.dataframe(metadata_df)  # Making the table scrollable

    st.subheader("Peak and Trough Trend Analysis")
    trends_df, distances_df = analyze_trends(filtered_data_frames, filtered_metadata_list, filtered_run_names, params_dict)
    st.dataframe(trends_df)  # Making the table scrollable
    filtered_distances_df = distances_df[distances_df['step'].isin(selected_step_size)]

    st.subheader("Distances Between Consecutive Peaks and Troughs")

    fig_distances = go.Figure()
    for run_name in filtered_distances_df['Run'].unique():
        df_run = filtered_distances_df[filtered_distances_df['Run'] == run_name]
        for distance_type in ['Peak Distance', 'Trough Distance']:
            df_type = df_run[df_run['Type'] == distance_type]
            if not df_type.empty:
                fig_distances.add_trace(go.Scatter(
                    x=df_type['Index'],
                    y=df_type['Distance'],
                    mode='lines+markers',
                    name=f'{distance_type} {run_name}'
                ))

    fig_distances.update_layout(title='Distances Between Consecutive Peaks and Troughs',
                                xaxis_title='Index',
                                yaxis_title='Distance (mm)',
                                legend_title='Type and Run',
                                hovermode='closest',
                                width=1800,  # Set the width of the plot
                                height=900   # Set the height of the plot
                                )
    st.plotly_chart(fig_distances)

    st.subheader("Box Plot of Distances Between Consecutive Peaks and Troughs")

    fig_box = go.Figure()
    for distance_type in ['Peak Distance', 'Trough Distance']:
        df_box_type = filtered_distances_df[filtered_distances_df['Type'] == distance_type]
        fig_box.add_trace(go.Box(
            y=df_box_type['Distance'],
            x=df_box_type['Index'],
            name=distance_type
        ))

    fig_box.update_layout(title='Box Plot of Distances Between Consecutive Peaks and Troughs',
                        xaxis_title='Index',
                        yaxis_title='Distance (mm)',
                        legend_title='Type',
                        hovermode='closest',
                        width=1800,  # Set the width of the plot
                        height=900   # Set the height of the plot
                        )
    st.plotly_chart(fig_box)

    st.subheader("Average Plot of dBm vs Position")
    all_positions = np.linspace(
        min(df['Position'].min() for df in filtered_data_frames),
        max(df['Position'].max() for df in filtered_data_frames),
        1000
    )

    avg_dbm = np.zeros_like(all_positions)
    for df in filtered_data_frames:
        avg_dbm += np.interp(all_positions, df['Position'], df['dBm'])

    avg_dbm /= len(filtered_data_frames)

    peaks, troughs = find_peaks_troughs(
        avg_dbm,
        distance=avg_distance,
        prominence=avg_prominence,
        width=avg_width
    )

    peak_positions = all_positions[peaks]
    trough_positions = all_positions[troughs]

    peak_distances = calculate_distances(peak_positions)
    trough_distances = calculate_distances(trough_positions)

    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(
        x=all_positions,
        y=avg_dbm,
        mode='lines',
        name='Average dBm',
        line=dict(color='green')
    ))

    fig_avg.add_trace(go.Scatter(
        x=peak_positions,
        y=avg_dbm[peaks],
        mode='markers',
        name='Peaks',
        marker=dict(color='red', size=10, symbol='triangle-up')
    ))

    fig_avg.add_trace(go.Scatter(
        x=trough_positions,
        y=avg_dbm[troughs],
        mode='markers',
        name='Troughs',
        marker=dict(color='blue', size=10, symbol='triangle-down')
    ))

    fig_avg.update_layout(
        title='Average dBm vs Position with Detected Peaks and Troughs',
        xaxis_title='Position (mm)',
        yaxis_title='dBm',
        legend_title='Components',
        hovermode='closest',
        width=1800,
        height=900
    )

    st.plotly_chart(fig_avg)

    if len(peak_positions) > len(trough_positions):
        trough_positions = np.pad(trough_positions, (0, len(peak_positions) - len(trough_positions)), 'constant', constant_values=np.nan)
    else:
        peak_positions = np.pad(peak_positions, (0, len(trough_positions) - len(peak_positions)), 'constant', constant_values=np.nan)

    st.subheader("Peak and Trough Positions Data")
    peak_trough_data = pd.DataFrame({
        'Peak Positions': peak_positions,
        'Trough Positions': trough_positions
    })
    st.dataframe(peak_trough_data)

    fig_distances_avg = go.Figure()
    fig_distances_avg.add_trace(go.Scatter(
        x=np.arange(len(peak_distances)),
        y=peak_distances,
        mode='lines+markers',
        name='Peak Distances',
        line=dict(color='red'),
        marker=dict(color='red', size=8)
    ))

    fig_distances_avg.add_trace(go.Scatter(
        x=np.arange(len(trough_distances)),
        y=trough_distances,
        mode='lines+markers',
        name='Trough Distances',
        line=dict(color='blue'),
        marker=dict(color='blue', size=8)
    ))

    fig_distances_avg.update_layout(
        title='Consecutive Peak and Trough Distances (Average Plot)',
        xaxis_title='Index',
        yaxis_title='Distance (mm)',
        legend_title='Type',
        hovermode='closest',
        width=1800,
        height=900
    )

    st.plotly_chart(fig_distances_avg)

    fig_box_avg = go.Figure()
    fig_box_avg.add_trace(go.Box(
        y=peak_distances,
        name='Peak Distances',
        marker=dict(color='red')
    ))

    fig_box_avg.add_trace(go.Box(
        y=trough_distances,
        name='Trough Distances',
        marker=dict(color='blue')
    ))

    fig_box_avg.update_layout(
        title='Box Plot of Consecutive Peak and Trough Distances (Average Plot)',
        yaxis_title='Distance (mm)',
        width=1800,
        height=900
    )

    st.plotly_chart(fig_box_avg)

if st.button('Run Machine Learning Analysis'):
    st.subheader("Machine Learning Analysis")
    
    # Prepare data for ML models
    all_data = []
    for df, metadata in zip(filtered_data_frames, filtered_metadata_list):
        for index, row in df.iterrows():
            all_data.append({
                'Position': row['Position'],
                'dBm': row['dBm'],
                'step': metadata['step'],
                'delay': metadata['delay'],
                'base_distance': metadata['base_distance'],
                'surface': metadata['surface'],
                'port': metadata['port']
            })

    all_data_df = pd.DataFrame(all_data)
    
    # Encode categorical feature 'surface'
    all_data_df = pd.get_dummies(all_data_df, columns=['surface'], drop_first=True)

    features = all_data_df[['Position', 'step', 'delay', 'base_distance', 'surface_smooth', 'port']]
    target = all_data_df['dBm']

    # Remove infinite and NaN values from features and target
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    target = target.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure all columns are numeric
    features = features.apply(pd.to_numeric, errors='coerce')

    # Align target with features indices
    features = features.loc[target.index].dropna()
    target = target.loc[features.index]

    # Sanity check for extremely large values
    if (features > 1e10).any().any():
        st.write("Warning: Features contain extremely large values. Please check your data.")
    if (target > 1e10).any():
        st.write("Warning: Target contains extremely large values. Please check your data.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Option to upload pre-trained models
    st.sidebar.subheader("Upload Pre-trained Models")
    rf_model_file = st.sidebar.file_uploader("Upload Random Forest Model", type=["pkl"])
    dl_model_file = st.sidebar.file_uploader("Upload Deep Learning Model", type=["h5"])

    # Random Forest Model
    if rf_model_file:
        best_rf = joblib.load(rf_model_file)
    else:
        param_grid = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10],
            'max_features': [None, 'sqrt', 'log2']
        }

        rf = RandomForestRegressor()

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        
        grid_search.fit(X_train, y_train)

        st.write("Best parameters found for Random Forest: ", grid_search.best_params_)

        best_rf = grid_search.best_estimator_

        model_path = "best_rf_model.pkl"
        joblib.dump(best_rf, model_path)
        st.write(f"Random Forest model saved at {os.path.abspath(model_path)}")

        with open(model_path, "rb") as f:
            st.download_button(label="Download Random Forest Model", data=f, file_name="best_rf_model.pkl", mime="application/octet-stream")

    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    st.write("Cross-validation scores for Random Forest: ", cv_scores)
    st.write("Mean cross-validation score: ", np.mean(cv_scores))

    best_rf.fit(X_train, y_train)

    y_pred_rf = best_rf.predict(X_test)

    st.write("Random Forest R^2 score on test set: ", r2_score(y_test, y_pred_rf))

    # Gradient Boosting Regressor
    param_grid_gbr = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }

    gbr = GradientBoostingRegressor()

    grid_search_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid_gbr, cv=3, n_jobs=-1, verbose=2)
    grid_search_gbr.fit(X_train, y_train)

    st.write("Best parameters found for Gradient Boosting Regressor: ", grid_search_gbr.best_params_)

    best_gbr = grid_search_gbr.best_estimator_

    cv_scores_gbr = cross_val_score(best_gbr, X_train, y_train, cv=5)
    st.write("Cross-validation scores for Gradient Boosting Regressor: ", cv_scores_gbr)
    st.write("Mean cross-validation score: ", np.mean(cv_scores_gbr))

    best_gbr.fit(X_train, y_train)

    y_pred_gbr = best_gbr.predict(X_test)

    st.write("Gradient Boosting Regressor R^2 score on test set: ", r2_score(y_test, y_pred_gbr))

    # Support Vector Regressor
    param_grid_svr = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf']
    }

    svr = SVR()

    grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=3, n_jobs=-1, verbose=2)
    grid_search_svr.fit(X_train, y_train)

    st.write("Best parameters found for Support Vector Regressor: ", grid_search_svr.best_params_)

    best_svr = grid_search_svr.best_estimator_

    cv_scores_svr = cross_val_score(best_svr, X_train, y_train, cv=5)
    st.write("Cross-validation scores for Support Vector Regressor: ", cv_scores_svr)
    st.write("Mean cross-validation score: ", np.mean(cv_scores_svr))

    best_svr.fit(X_train, y_train)

    y_pred_svr = best_svr.predict(X_test)

    st.write("Support Vector Regressor R^2 score on test set: ", r2_score(y_test, y_pred_svr))

    # Predict dBm values for a new set of parameters
    st.subheader("Predict dBm Values for New Parameters")

    step = st.number_input('Step Size (mm)', min_value=0.01, max_value=10.0, value=0.1)
    delay = st.number_input('Delay (s)', min_value=1, max_value=10, value=2)
    base_distance = st.number_input('Base Distance (m)', min_value=0.1, max_value=10.0, value=0.5)
    surface = st.selectbox('Surface', ['rough', 'smooth'])
    port = st.number_input('Port Number', min_value=1, max_value=10, value=1)
    positions = np.linspace(all_data_df['Position'].min(), all_data_df['Position'].max(), 1000)

    new_features = pd.DataFrame({
        'Position': positions,
        'step': [step] * len(positions),
        'delay': [delay] * len(positions),
        'base_distance': [base_distance] * len(positions),
        'surface_smooth': [1 if surface == 'smooth' else 0] * len(positions),
        'port': [port] * len(positions)
    })

    # Random Forest predictions
    predicted_dbm_rf = best_rf.predict(new_features)

    # Gradient Boosting Regressor predictions
    predicted_dbm_gbr = best_gbr.predict(new_features)

    # Support Vector Regressor predictions
    predicted_dbm_svr = best_svr.predict(new_features)

    # Function to create and display plot
    def create_and_display_plot(positions, predicted_dbm, model_name):
        st.write(f"Debug: About to create {model_name} plot")
        try:
            predicted_peaks, predicted_troughs = find_peaks_troughs(predicted_dbm, distance=1, prominence=0.5, width=1)
            
            st.write(f"Predicted Peaks at Positions ({model_name}): {positions[predicted_peaks]}")
            st.write(f"Predicted Troughs at Positions ({model_name}): {positions[predicted_troughs]}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=positions,
                y=predicted_dbm,
                mode='lines',
                name=f'Predicted dBm ({model_name})'
            ))
            fig.add_trace(go.Scatter(
                x=positions[predicted_peaks],
                y=predicted_dbm[predicted_peaks],
                mode='markers',
                name=f'Predicted Peaks ({model_name})',
                marker=dict(color='red', size=10)
            ))
            fig.add_trace(go.Scatter(
                x=positions[predicted_troughs],
                y=predicted_dbm[predicted_troughs],
                mode='markers',
                name=f'Predicted Troughs ({model_name})',
                marker=dict(color='blue', size=10)
            ))
            fig.update_layout(
                title=f'Predicted dBm Values with Peaks and Troughs ({model_name})',
                xaxis_title='Position (mm)',
                yaxis_title='dBm',
                legend_title='Type',
                hovermode='closest',
                width=1800,
                height=900
            )
            st.write(f"Debug: {model_name} plot object created")
            st.plotly_chart(fig)
            st.write(f"Debug: {model_name} plot should be displayed now")
        except Exception as e:
            st.write(f"Error creating or displaying {model_name} plot: {str(e)}")

    # Create and display plots for each model
    create_and_display_plot(positions, predicted_dbm_rf, "Random Forest")
    create_and_display_plot(positions, predicted_dbm_gbr, "Gradient Boosting Regressor")
    create_and_display_plot(positions, predicted_dbm_svr, "Support Vector Regressor")

    # Correlation Analysis for Peak and Trough Positions
    st.subheader("Correlation Analysis for Peak and Trough Positions")
    
    # Extracting metadata for peaks and troughs
    peaks_troughs_metadata = []
    for df, metadata in zip(filtered_data_frames, filtered_metadata_list):
        peaks, troughs = find_peaks_troughs(df['dBm'])
        peaks_positions = df['Position'][peaks].tolist()
        troughs_positions = df['Position'][troughs].tolist()
        
        for pos in peaks_positions:
            peaks_troughs_metadata.append({**metadata, 'Type': 'Peak', 'Position': pos})
        for pos in troughs_positions:
            peaks_troughs_metadata.append({**metadata, 'Type': 'Trough', 'Position': pos})

    peaks_troughs_df = pd.DataFrame(peaks_troughs_metadata)
    
    # Prepare data for correlation analysis
    corr_features = ['delay', 'step', 'base_distance', 'surface_smooth', 'port']
    peaks_troughs_df = pd.get_dummies(peaks_troughs_df, columns=['surface'], drop_first=True)
    X_corr = peaks_troughs_df[corr_features]
    y_corr = peaks_troughs_df['Position']

    # Ensure all columns are numeric
    X_corr = X_corr.apply(pd.to_numeric, errors='coerce').dropna()

    # Train Random Forest model for correlation analysis
    rf_corr = RandomForestRegressor()
    rf_corr.fit(X_corr, y_corr)

    # Extract feature importances using permutation importance
    perm_importance = permutation_importance(rf_corr, X_corr, y_corr, n_repeats=30, random_state=42)

    # Visualize feature importances using a bar chart
    fig_corr_bar, ax_corr_bar = plt.subplots()
    feature_importance_df = pd.DataFrame({'Feature': corr_features, 'Importance': perm_importance.importances_mean})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    ax_corr_bar.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    ax_corr_bar.set_title('Feature Importances for Peak and Trough Positions')
    ax_corr_bar.set_ylabel('Importance')
    st.pyplot(fig_corr_bar)

    # Partial Dependence Plot
    st.subheader("Partial Dependence Plot")
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(best_rf, X_train, features=[0, 1, 2, 3], ax=ax)
    st.pyplot(fig)

st.write("Upload CSV files to start analyzing data.")

