from flask import Flask, render_template, request
import pandas as pd
import os
# Sample DataFrame
df=pd.read_excel('Task1and2/train.xlsx')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib 
def calculation(i):
    features = df.iloc[:, :-1]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    saved_cluster_centers = joblib.load('kmeans_weights.pkl')
    predicted_cluster = kmeans.fit_predict(scaled_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    # Create a scatter plot of the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=predicted_cluster, cmap='viridis', alpha=0.6)
    # Highlight the selected point
    plt.scatter(reduced_features[i, 0], reduced_features[i, 1], 
                c='red', label=f'Selected Row (Cluster {predicted_cluster[i]})', edgecolors='black')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.title('Clusters Visualized in 2D')
    plt.legend()
    plt.savefig('static/sample_plot.png')

df2=pd.read_excel('results/Classification_Dicition_tree_result.xlsx')
df3=pd.read_excel('results/Classification_logistic_regression_result.xlsx')
df4=pd.read_excel('results/Classification_naive_bias_result.xlsx')

main_df=pd.read_excel('Task3/rawdata.xlsx')

def result_df(df):
    # Ensure 'date' and 'time' are strings
    df['date'] = df['date'].astype(str)
    df['time'] = df['time'].astype(str)

    # Convert 'date' and 'time' columns to datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    # Sort the DataFrame by the datetime column
    df = df.sort_values(by='datetime')

    # Calculate the duration in minutes for each entry
    df['duration in mintes'] = df['datetime'].diff().dt.total_seconds() / 60

    # Fill the first row's NaN duration with 0
    df['duration in mintes'].fillna(0, inplace=True)

    # 1. Datewise total duration
    total_duration_agg = df.groupby('date')['duration in mintes'].sum().reset_index()
    print("Datewise total duration:")
    result_df1=total_duration_agg

    # 2. Datewise number of picking and placing activity done
    activity_agg = df.groupby(['date', 'activity']).size().reset_index(name='count')
    print("\nDatewise number of picking and placing activity done:")
    result_df2=activity_agg
    return result_df1,result_df2
result_df1,result_df2=result_df(main_df)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
@app.route('/')
def index():
    return render_template('index.html', table=df.head().to_html(index=False), selected_row=None, selected_df=None)

@app.route('/select_row', methods=['POST'])
def select_row():
    row_index = int(request.form['row_index'])
    if 0 <= row_index < len(df):
        selected_row = df.iloc[row_index]
        selected_df = pd.DataFrame(selected_row).transpose().to_html(index=True)
        calculation(row_index)
    else:
        selected_row = None
        selected_df = None
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_plot.png')
    return render_template('index.html', table=df.head().to_html(index=False), selected_row=selected_row, selected_df=selected_df,selected_image=full_filename)
@app.route('/task2')
def task2():
    return render_template('task2.html', table1=df2.head().to_html(index=True), table2=df3.head().to_html(index=True), table3=df4.head().to_html(index=True))
@app.route('/task3')
def task3():
    return render_template('task3.html', main_table=main_df.head().to_html(index=False), result_table1=result_df1.to_html(index=False), result_table2=result_df2.to_html(index=False))

if __name__ == '__main__':
    app.run(debug=True)