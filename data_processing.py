# data_processing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def preparar_datos():
    os.makedirs('static/images', exist_ok=True)

    # Cargar el dataset
    sales_df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')
    sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'])

    # Eliminar columnas innecesarias
    df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY',
               'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
    sales_df = sales_df.drop(df_drop, axis=1)

    # Barplots de variables categóricas
    def barplot_visualization(col):
        fig = px.bar(x=sales_df[col].value_counts().index, 
                     y=sales_df[col].value_counts(), 
                     color=sales_df[col].value_counts().index)
        fig.write_image(f'static/images/barplot_{col}.png')

    barplot_visualization('COUNTRY')
    barplot_visualization('STATUS')
    sales_df.drop(columns=['STATUS'], inplace=True)
    barplot_visualization('PRODUCTLINE')
    barplot_visualization('DEALSIZE')

    # Dummy variables
    for col in ['COUNTRY', 'PRODUCTLINE', 'DEALSIZE']:
        dummies = pd.get_dummies(sales_df[col])
        sales_df = pd.concat([sales_df.drop(columns=[col]), dummies], axis=1)

    # Dummy específico
    if 'Motorcycles' in sales_df.columns:
        barplot_visualization('Motorcycles')

    # Codificar PRODUCTCODE
    sales_df['PRODUCTCODE'] = pd.Categorical(sales_df['PRODUCTCODE']).codes

    # Agrupar por fecha
    sales_df_group = sales_df.groupby('ORDERDATE').sum()

    fig = px.line(x=sales_df_group.index, y=sales_df_group.SALES, title='Ventas por fecha')
    fig.write_image('static/images/line_ventas_fecha.png')

    # Heatmap de correlaciones
    plt.figure(figsize=(20,20))
    sns.heatmap(sales_df.iloc[:, :9].corr(), annot=True, cbar=False)
    plt.title('Heatmap Correlaciones')
    plt.savefig('static/images/heatmap_corr.png')
    plt.close()

    # Scatter Matrix
    fig = px.scatter_matrix(sales_df, dimensions=sales_df.columns[:8], color='MONTH_ID')
    fig.update_layout(width=1100, height=1100)
    fig.write_image('static/images/scatter_matrix.png')

    # Eliminar QTR_ID
    if 'QTR_ID' in sales_df.columns:
        sales_df.drop('QTR_ID', axis=1, inplace=True)

    # Escalar
    scaler = StandardScaler()
    sales_df_scaled = scaler.fit_transform(sales_df.drop(columns=['ORDERDATE']))

    # WCSS - método del codo
    scores = []
    range_values = range(1,15)
    for i in range_values:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(sales_df_scaled)
        scores.append(kmeans.inertia_)

    plt.plot(range_values, scores, 'bx-')
    plt.title('Número óptimo de clusters')
    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.savefig('static/images/elbow_kmeans.png')
    plt.close()

    # KMeans normal
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(sales_df_scaled)
    sale_df_cluster = pd.concat([sales_df, pd.DataFrame({'cluster': labels})], axis=1)

    # Cluster centers antes del autoencoder
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=sales_df.drop(columns=['ORDERDATE']).columns)
    cluster_centers = pd.DataFrame(scaler.inverse_transform(cluster_centers), columns=sales_df.drop(columns=['ORDERDATE']).columns)

    # PCA 3D antes
    pca = PCA(n_components=3)
    pca_comp = pca.fit_transform(sales_df_scaled)
    pca_df = pd.DataFrame(pca_comp, columns=['pca1', 'pca2', 'pca3'])
    pca_df['cluster'] = labels

    fig = px.scatter_3d(pca_df, x='pca1', y='pca2', z='pca3', color='cluster')
    fig.write_image('static/images/pca_3d_before_autoencoder.png')

    # PCA 2D antes
    pca2d = PCA(n_components=2).fit_transform(sales_df_scaled)
    pca2d_df = pd.DataFrame(pca2d, columns=['pca1', 'pca2'])
    pca2d_df['cluster'] = labels

    fig = px.scatter(pca2d_df, x='pca1', y='pca2', color='cluster')
    fig.write_image('static/images/pca_2d_before_autoencoder.png')

    # Histograma por cluster (primer clustering)
    for col in sales_df.columns[:8]:
        plt.figure(figsize=(30,6))
        for j in range(5):
            plt.subplot(1,5,j+1)
            cluster = sale_df_cluster[sale_df_cluster['cluster'] == j]
            cluster[col].hist()
            plt.title(f'{col}\nCluster {j}')
        plt.savefig(f'static/images/hist_{col}_before.png')
        plt.close()

    # ============== AUTOENCODER ==============

    input_df = Input(shape=(sales_df_scaled.shape[1],))
    x = Dense(50, activation='relu')(input_df)
    x = Dense(500, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(2000, activation='relu')(x)
    encoded = Dense(8, activation='relu')(x)
    x = Dense(2000, activation='relu')(encoded)
    x = Dense(500, activation='relu')(x)
    decoded = Dense(sales_df_scaled.shape[1])(x)

    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)

    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

    # ✅ Cargar pesos previamente guardados
    autoencoder.load_weights('autoencoder_1.weights.h5')

    # Codificación
    pred = encoder.predict(sales_df_scaled)

    # WCSS después del autoencoder
    scores = []
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(pred)
        scores.append(kmeans.inertia_)

    plt.plot(range_values, scores, 'bx-')
    plt.title('Número óptimo de clusters (después del Autoencoder)')
    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.savefig('static/images/elbow_autoencoder.png')
    plt.close()

    # KMeans después del autoencoder
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(pred)

    df_cluster_dr = pd.concat([sales_df, pd.DataFrame({'cluster': labels})], axis=1)

    cluster_centers_after = pd.DataFrame(kmeans.cluster_centers_, columns=[f"feature_{i}" for i in range(pred.shape[1])])

    # PCA 3D después
    pca = PCA(n_components=3)
    pca_comp = pca.fit_transform(pred)
    pca_df_after = pd.DataFrame(pca_comp, columns=['pca1', 'pca2', 'pca3'])
    pca_df_after['cluster'] = labels

    fig = px.scatter_3d(pca_df_after, x='pca1', y='pca2', z='pca3', color='cluster')
    fig.write_image('static/images/pca_3d_after_autoencoder.png')

    # PCA 2D después
    pca2d = PCA(n_components=2).fit_transform(pred)
    pca2d_df = pd.DataFrame(pca2d, columns=['pca1', 'pca2'])
    pca2d_df['cluster'] = labels

    fig = px.scatter(pca2d_df, x='pca1', y='pca2', color='cluster')
    fig.write_image('static/images/pca_2d_after_autoencoder.png')

    # Histograma por cluster (después del autoencoder)
    for col in sales_df.columns[:8]:
        plt.figure(figsize=(30,6))
        for j in range(3):
            plt.subplot(1,3,j+1)
            cluster = df_cluster_dr[df_cluster_dr['cluster'] == j]
            cluster[col].hist()
            plt.title(f'{col}\nCluster {j}')
        plt.savefig(f'static/images/hist_{col}_after.png')
        plt.close()

    # Diccionario de DataFrames
    todos_los_dataframes = {
        'Sales Data': sales_df.head(10),
        'Grouped Sales': sales_df_group.head(10),
        'Clustered Sales (KMeans)': sale_df_cluster.head(10),
        'PCA Before Autoencoder': pca_df.head(10),
        'Cluster Centers Before Autoencoder': cluster_centers.head(10),
        'Clustered Sales After Autoencoder': df_cluster_dr.head(10),
        'PCA After Autoencoder': pca_df_after.head(10),
        'Cluster Centers After Autoencoder': cluster_centers_after.head(10)
    }

        # Crear carpeta para guardar DataFrames si no existe
    os.makedirs('static/data', exist_ok=True)

    sales_df.head(10).to_csv('static/data/sales_df.csv', index=False)
    sales_df_group.head(10).to_csv('static/data/sales_df_group.csv', index=False)
    sale_df_cluster.head(10).to_csv('static/data/sale_df_cluster.csv', index=False)
    pca_df.head(10).to_csv('static/data/pca_df.csv', index=False)
    cluster_centers.head(10).to_csv('static/data/cluster_centers.csv', index=False)
    df_cluster_dr.head(10).to_csv('static/data/df_cluster_dr.csv', index=False)
    pca_df_after.head(10).to_csv('static/data/pca_df_after.csv', index=False)
    cluster_centers_after.head(10).to_csv('static/data/cluster_centers_after.csv', index=False)

preparar_datos()
