from flask import Flask, render_template, Response
import os
import pandas as pd
from flask import send_file
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

app = Flask(__name__)

# Leer los DataFrames guardados
def cargar_dataframes():
    dataframes = {}
    try:
        dataframes['Sales Data'] = pd.read_csv('static/data/sales_df.csv')
        dataframes['Grouped Sales'] = pd.read_csv('static/data/sales_df_group.csv')
        dataframes['Clustered Sales (KMeans)'] = pd.read_csv('static/data/sale_df_cluster.csv')
        dataframes['PCA Before Autoencoder'] = pd.read_csv('static/data/pca_df.csv')
        dataframes['Cluster Centers Before Autoencoder'] = pd.read_csv('static/data/cluster_centers.csv')
        dataframes['Clustered Sales After Autoencoder'] = pd.read_csv('static/data/df_cluster_dr.csv')
        dataframes['PCA After Autoencoder'] = pd.read_csv('static/data/pca_df_after.csv')
        dataframes['Cluster Centers After Autoencoder'] = pd.read_csv('static/data/cluster_centers_after.csv')
    except Exception as e:
        print(f"Error cargando CSVs: {e}")
    return dataframes

dataframes = cargar_dataframes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dataframes')
def mostrar_dataframes():
    return render_template('dataframes.html', dataframes=dataframes)

@app.route('/graphs')
def mostrar_graphs():
    return render_template('graphs.html')


from flask import send_file
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

@app.route('/dataframes-text')
def api_dataframes_pdf():
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50  # posición inicial vertical
    p.setFont("Helvetica", 10)

    for nombre, df in dataframes.items():
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y, f"📊 {nombre}")
        y -= 20

        p.setFont("Helvetica", 9)
        preview = df.head(3).to_string(index=False).split('\n')
        for line in preview:
            if y < 50:
                p.showPage()
                y = height - 50
                p.setFont("Helvetica", 9)
            p.drawString(50, y, line)
            y -= 15

        y -= 10  # Espacio entre dataframes

    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name='dataframes_preview.pdf',
        mimetype='application/pdf'
    )

@app.route('/download/graphs')
def download_graphs():
    # Crear un buffer en memoria
    pdf_buffer = io.BytesIO()
    # PdfPages para agrupar varias figuras
    with PdfPages(pdf_buffer) as pdf:
        # Ejemplo: una gráfica por cada DataFrame (puedes ajustar según tus necesidades)
        for name, df in dataframes.items():
            plt.figure(figsize=(6,4))
            # Si contiene columnas numéricas, plotea las primeras 5
            num_cols = df.select_dtypes('number').columns.tolist()
            if num_cols:
                df[num_cols].head(5).plot.bar(title=name)
            else:
                plt.text(0.5, 0.5, f'No hay datos numéricos en "{name}"',
                         ha='center', va='center')
            plt.tight_layout()
            pdf.savefig()   # guarda la figura en el PDF
            plt.close()
    pdf_buffer.seek(0)
    # Enviar el PDF al cliente
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name='todas_las_graficas.pdf',
        mimetype='application/pdf'
    )



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
