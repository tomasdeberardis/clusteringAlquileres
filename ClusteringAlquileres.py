# ==================================================================================
#  PIPELINE FINAL Y DEFINITIVO (V21 - EDICIÓN COMPLETA CON EVALUACIÓN DE CALIDAD)
# ==================================================================================
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
# --- Importación para la evaluación ---
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# --- PASOS 0, 1 y 2: CONFIGURACIÓN, FUNCIONES Y DATOS HISTÓRICOS ---
print("--- Iniciando Pasos 0, 1 y 2: Configuración, Funciones y Dólar ---")
warnings.filterwarnings('ignore')
tu_archivo_de_scraping = 'original_info.csv'
file_usd_xls = 'usd_historic_price.xlsx'
df_usd = pd.read_excel(file_usd_xls, header=0)
df_usd.columns = ['date', 'usd_price']
df_usd['date'] = pd.to_datetime(df_usd['date'])
print("Pasos 0, 1 y 2: Listos.\n")

# --- PASO 3: CREACIÓN DEL DATASET MAESTRO (CON LOGS DETALLADOS) ---
print("--- Iniciando Paso 3: Creación del Dataset Maestro (con logs detallados) ---")
try:
    print("\n[LOG] 3.1.1 - Cargando datos crudos de scraping desde 'original_info.csv'...")
    df_scraped = pd.read_csv(tu_archivo_de_scraping, encoding='utf-8-sig')
    df_scraped.rename(columns={'BARRIO': 'Barrio','FECHA_INGRESO': 'Fecha','PRECIO_ORIGINAL': 'Precio_Original','MONEDA': 'Moneda','METROS': 'Superficie_m2','AMBIENTES': 'Ambientes'}, inplace=True)
    print(f"      -> Se cargaron {len(df_scraped)} registros iniciales.")

    print("\n[LOG] 3.1.2 - Realizando limpieza inicial y conversión de tipos...")
    df_scraped['Precio_Original'] = pd.to_numeric(df_scraped['Precio_Original'], errors='coerce')
    df_scraped['Fecha'] = pd.to_datetime(df_scraped['Fecha'], errors='coerce')
    df_scraped.dropna(subset=['Fecha', 'Precio_Original', 'Superficie_m2', 'Ambientes', 'Barrio'], inplace=True)
    df_scraped['Barrio'] = df_scraped['Barrio'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)

    print("\n[LOG] 3.2.1 - Uniendo con datos del dólar y clasificando Alquiler/Venta...")
    df_full = pd.merge_asof(df_scraped.sort_values('Fecha'), df_usd.sort_values('date'), left_on='Fecha', right_on='date', direction='nearest')
    df_full['Precio_USD'] = np.where(df_full['Moneda'] == 'USD', df_full['Precio_Original'], df_full['Precio_Original'] / df_full['usd_price'])
    
    print("\n[LOG] 3.2.2 - Clasificando propiedades en 'Alquiler' o 'Venta' (Umbral: < 30,000 USD para Alquiler)...")
    df_full['Tipo_Operacion'] = np.where(df_full['Precio_USD'] < 30000, 'Alquiler', 'Venta')
    print("      -> Distribución Alquiler/Venta encontrada:")
    print(df_full['Tipo_Operacion'].value_counts())

    print("\n[LOG] 3.3.1 - Pivote estratégico: Filtrando para analizar solo ALQUILERES...")
    df_alquiler = df_full[df_full['Tipo_Operacion'] == 'Alquiler'].copy()
    df_alquiler['Precio_USD_m2'] = df_alquiler['Precio_USD'] / df_alquiler['Superficie_m2']
    
    print("\n[LOG] 3.4.1 - Eliminando outliers extremos para asegurar la calidad del modelo...")
    print(f"      - Partiendo de {len(df_alquiler)} propiedades en alquiler.")
    p_low, p_high = df_alquiler['Precio_USD_m2'].quantile([0.01, 0.99])
    s_low, s_high = df_alquiler['Superficie_m2'].quantile([0.01, 0.99])
    df_final = df_alquiler[(df_alquiler['Precio_USD_m2'].between(p_low, p_high)) & (df_alquiler['Superficie_m2'].between(s_low, s_high))].copy()
    print(f"      -> Quedan {len(df_final)} alquileres para el análisis tras la limpieza.")

    print("\n[LOG] 3.5.1 - [FEATURE ENGINEERING] Creando 'Indice_Valor_Barrio'...")
    indice_valor_barrio = df_final.groupby('Barrio')['Precio_USD_m2'].mean()
    df_final['Indice_Valor_Barrio'] = df_final['Barrio'].map(indice_valor_barrio)
    
    print("\n[LOG] 3.6.1 - Aplicando Transformación Logarítmica...")
    df_final['log_Precio_USD_m2'] = np.log1p(df_final['Precio_USD_m2'])
    df_final['log_Superficie_m2'] = np.log1p(df_final['Superficie_m2'])
    
    print("\nPASO 3 COMPLETADO: El Dataset Maestro está listo.")
    
except Exception as e:
    df_final = pd.DataFrame()
    print(f"  - ERROR en el paso 3: {e}\n")


# --- PASO 4: MODELADO FINAL CON DBSCAN ---
print("\n--- Iniciando Paso 4: Modelado Final con DBSCAN (Robusto a Outliers) ---")
if not df_final.empty and len(df_final) > 10:
    numeric_features = ['log_Precio_USD_m2', 'log_Superficie_m2', 'Ambientes', 'Indice_Valor_Barrio']
    X = df_final[numeric_features]

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

    eps_valor = 0.75
    min_samples_valor = 15
    dbscan = DBSCAN(eps=eps_valor, min_samples=min_samples_valor)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clusterer', dbscan)])
    
    print("  - [Entrenamiento] Entrenando el modelo DBSCAN...")
    pipeline.fit(X)
    df_final['Cluster'] = pipeline.named_steps['clusterer'].labels_
    
    n_clusters = len(set(df_final['Cluster'])) - (1 if -1 in df_final['Cluster'] else 0)
    n_noise = list(df_final['Cluster']).count(-1)
    
    print(f"\n  - Modelo final entrenado.")
    print(f"  - DBSCAN encontró {n_clusters} clústeres distintos y {n_noise} outliers (ruido).")
    print("\nPaso 4: Modelado completado.\n")
else:
    print("  - ERROR: No hay datos suficientes para el modelo.\n")

# --- PASO 5: EVALUACIÓN CUANTITATIVA Y ANÁLISIS DE CLUSTERS ---
if 'Cluster' in df_final.columns:
    print("\n--- Iniciando Paso 5: Evaluación y Análisis de Clusters ---")
    
# --- PASO 5: EVALUACIÓN CUANTITATIVA Y ANÁLISIS DE CLUSTERS ---
if 'Cluster' in df_final.columns:
    print("\n--- Iniciando Paso 5: Evaluación Cuantitativa y Análisis de Clusters ---")
    
    X_preprocessed = pipeline.named_steps['preprocessor'].transform(X)
    cluster_labels = pipeline.named_steps['clusterer'].labels_
    
   # --- PASO 5: EVALUACIÓN CUANTITATIVA Y ANÁLISIS DE CLUSTERS ---
if 'Cluster' in df_final.columns:
    print("\n--- Iniciando Paso 5: Evaluación Cuantitativa y Análisis de Clusters ---")
    
    X_preprocessed = pipeline.named_steps['preprocessor'].transform(X)
    cluster_labels = pipeline.named_steps['clusterer'].labels_
    
    # --- EVALUACIÓN DE CALIDAD DEL CLUSTERING ---
    print("\n[LOG] 5.1 - Evaluando la calidad de la segmentación con un panel de métricas...")
    mask = cluster_labels != -1
    
    if sum(mask) > 1 and len(set(cluster_labels[mask])) > 1:
        # --- Métrica 1: Índice de Silueta ---
        silhouette_avg = silhouette_score(X_preprocessed[mask], cluster_labels[mask])
        print(f"  - [Métrica 1] Índice de Silueta: {silhouette_avg:.3f}")
        print("    (Cercano a 1 es mejor. Mide la calidad desde la perspectiva de cada punto).")

        # --- Métrica 2: Índice de Calinski-Harabasz ---
        ch_score = calinski_harabasz_score(X_preprocessed[mask], cluster_labels[mask])
        print(f"  - [Métrica 2] Índice de Calinski-Harabasz: {ch_score:,.2f}")
        print("    (Más alto es mejor. Mide la separación general de los clústeres).")

        # --- Métrica 3: Índice de Davies-Bouldin ---
        db_score = davies_bouldin_score(X_preprocessed[mask], cluster_labels[mask])
        print(f"  - [Métrica 3] Índice de Davies-Bouldin: {db_score:.3f}")
        print("    (Más bajo es mejor, cercano a 0 es ideal. Mide la separación en el 'peor de los casos').")

    else:
        print("\n  - No se pudo calcular las métricas de evaluación (se necesitan al menos 2 clústeres).")
        
    df_clusters_only = df_final[df_final['Cluster'] != -1].copy()
    if not df_clusters_only.empty:
        interpret_features = ['Precio_USD_m2', 'Superficie_m2', 'Ambientes', 'Indice_Valor_Barrio']
        cluster_summary = df_clusters_only.groupby('Cluster')[interpret_features].mean()
        top_barrios = df_clusters_only.groupby('Cluster')['Barrio'].apply(lambda x: x.mode().head(2).tolist())
        cluster_summary['Barrios_Comunes'] = top_barrios
        
        print("\n========================= PERFIL DE CLUSTERS FINAL =========================")
        display(cluster_summary.round(2))
        
        cluster_summary_sorted = cluster_summary.sort_values('Indice_Valor_Barrio')
        for i, row in cluster_summary_sorted.iterrows():
            print(f"\n--- CLÚSTER {i}: Análisis Detallado ---")
            if i == cluster_summary_sorted.index[0]: valor_desc = "el más ECONÓMICO"
            elif i == cluster_summary_sorted.index[-1]: valor_desc = "el más PREMIUM"
            else: valor_desc = "de VALOR INTERMEDIO"
            
            print(f"  - PERFIL DE VALOR: Representa el segmento {valor_desc} de los clústeres.")
            print(f"  - PERFIL DE TAMAÑO: Se enfoca en propiedades de tamaño {'GRANDE' if row['Superficie_m2'] > 80 else 'MEDIANO' if row['Superficie_m2'] > 45 else 'PEQUEÑO'}.")
            print(f"  - BARRIOS TÍPICOS: {', '.join(row['Barrios_Comunes'])}.")
        print("\n============================================================================")

        print("\n  - [Visualización] Generando gráficos para entender la distribución de cada clúster...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Análisis Visual de Clusters de Alquiler (DBSCAN)', fontsize=20)
        sns.scatterplot(ax=axes[0, 0], data=df_final, x='Superficie_m2', y='Precio_USD_m2', hue='Cluster', palette='viridis', s=20, alpha=0.6)
        axes[0, 0].set_title('Visión General (Clúster -1 = Outliers)'); axes[0, 0].grid(True)
        sns.boxplot(ax=axes[0, 1], data=df_clusters_only, x='Cluster', y='Precio_USD_m2', palette='viridis'); axes[0, 1].set_title('Distribución de Precio/m² por Cluster')
        sns.boxplot(ax=axes[1, 0], data=df_clusters_only, x='Cluster', y='Superficie_m2', palette='viridis'); axes[1, 0].set_title('Distribución de Superficie por Cluster')
        sns.boxplot(ax=axes[1, 1], data=df_clusters_only, x='Cluster', y='Indice_Valor_Barrio', palette='viridis'); axes[1, 1].set_title('Distribución de Valor de Barrio por Cluster')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    else:
        print("  - No se encontraron clústeres suficientemente densos.")

# --- PASO 6: VALUACIÓN DETALLADA Y EXPORTACIÓN ---
if 'Cluster' in df_final.columns and not df_clusters_only.empty:
    print("\n--- Iniciando Paso 6: Valuación Detallada y Exportación ---")
    # (El código de valuación y exportación se mantiene igual)
    stats_clusters = df_clusters_only.groupby('Cluster')['Precio_USD_m2'].agg(['mean', 'std']).reset_index()
    stats_clusters.rename(columns={'mean': 'Precio_Promedio_Cluster_m2', 'std': 'Precio_Std_Cluster_m2'}, inplace=True)
    stats_clusters.fillna(0, inplace=True)
    df_para_powerbi = pd.merge(df_final, stats_clusters, on='Cluster', how='left')
    df_para_powerbi['Valuacion'] = df_para_powerbi.apply(lambda row: 'Outlier/Ruido' if row['Cluster'] == -1 else ('N/A' if pd.isna(row['Precio_Promedio_Cluster_m2']) else ('Sobrevaluada' if row['Precio_USD_m2'] > row['Precio_Promedio_Cluster_m2'] + 0.5 * row['Precio_Std_Cluster_m2'] else ('Infravalorada' if row['Precio_USD_m2'] < row['Precio_Promedio_Cluster_m2'] - 0.5 * row['Precio_Std_Cluster_m2'] else 'Precio Justo'))), axis=1)
    
    columnas_export = ['ID', 'URL', 'Fecha', 'Barrio', 'Ambientes', 'Superficie_m2', 'Precio_USD', 'Precio_USD_m2', 'Cluster', 'Valuacion', 'Indice_Valor_Barrio']
    df_export_final = df_para_powerbi[[col for col in columnas_export if col in df_para_powerbi.columns]]
    
    output_filename = 'alquileres_valuados_para_powerbi.csv'
    df_export_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n- Archivo '{output_filename}' generado con el análisis final.")
    print("\n======================= PIPELINE COMPLETADO, REFINADO Y EVALUADO =======================")