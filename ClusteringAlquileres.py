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
# --- Importaciones para la evaluación ---
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

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

    # --- EVALUACIÓN DE CALIDAD DEL CLUSTERING ---
    print("\n[LOG] 5.1 - Evaluando la calidad de la segmentación con un panel de métricas...")
    X_preprocessed = pipeline.named_steps['preprocessor'].transform(X)
    cluster_labels = pipeline.named_steps['clusterer'].labels_
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

    # --- ANÁLISIS CUALITATIVO ---
    df_clusters_only = df_final[df_final['Cluster'] != -1].copy()
    if not df_clusters_only.empty:
        interpret_features = ['Precio_USD_m2', 'Superficie_m2', 'Ambientes', 'Indice_Valor_Barrio']
        cluster_summary = df_clusters_only.groupby('Cluster')[interpret_features].mean()
   # Lógica para barrios distintivos
        overall_barrio_dist = df_final['Barrio'].value_counts(normalize=True)
        distinctive_barrios_list = []
        for i in cluster_summary.index:
            cluster_barrio_dist = df_clusters_only[df_clusters_only['Cluster'] == i]['Barrio'].value_counts(normalize=True)
            lift = (cluster_barrio_dist / overall_barrio_dist).dropna()
            distinctive_barrios_list.append(lift.nlargest(2).index.tolist())
        cluster_summary['Barrios_Distintivos'] = distinctive_barrios_list

        print("\n========================= PERFIL DE CLUSTERS FINAL =========================")
        display(cluster_summary.round(2))

       # --- LÓGICA DE DESCRIPCIÓN NARRATIVA CON NÚMEROS ---
        def generar_descripcion_numerica(row, sorted_summary):
            # Perfil de TAMAÑO y TIPO
            if row['Ambientes'] < 1.8: tipo_prop = "Monoambientes y estudios"
            elif row['Ambientes'] < 2.8: tipo_prop = "Apartamentos de 2 ambientes"
            elif row['Ambientes'] < 3.8: tipo_prop = "Apartamentos familiares (3 amb.)"
            else: tipo_prop = "Apartamentos grandes (4+ amb.)"

            # Perfil de VALOR (ranking)
            rank = sorted_summary.index.get_loc(row.name)
            total_clusters = len(sorted_summary)
            if rank == 0: valor_desc = "en las zonas más accesibles"
            elif rank < total_clusters / 3: valor_desc = "en zonas de valor intermedio-bajo"
            elif rank < total_clusters * 2 / 3: valor_desc = "en zonas de valor intermedio-alto"
            else: valor_desc = "en el nicho más Premium y exclusivo"

            return (f"Este clúster define el segmento de '{tipo_prop}' ({row['Ambientes']:.1f} amb. promedio) "
                    f"con una superficie promedio de {row['Superficie_m2']:.0f} m², "
                    f"ubicados {valor_desc} (Índice de Valor: ${row['Indice_Valor_Barrio']:.2f}). \n"
                    f"Sus barrios más característicos son **{', '.join(row['Barrios_Distintivos'])}**.\n")
        
        print("\n--- Descripción Detallada y Numérica de Cada Segmento Encontrado ---")      
        cluster_summary_sorted = cluster_summary.sort_values('Indice_Valor_Barrio')
        for i, row in cluster_summary_sorted.iterrows():
            print(f"\n- CLÚSTER {i}:\n {generar_descripcion_numerica(row, cluster_summary_sorted)}")
            print(f"\n--- CLÚSTER {i}: Análisis Detallado ---")
            if i == cluster_summary_sorted.index[0]: valor_desc = "el más ECONÓMICO"
            elif i == cluster_summary_sorted.index[-1]: valor_desc = "el más PREMIUM"
            else: valor_desc = "de VALOR INTERMEDIO"

            print(f"  - PERFIL DE VALOR: Representa el segmento {valor_desc} de los clústeres.")
            print(f"    (Su Índice de Valor Promedio es de ${row['Indice_Valor_Barrio']:.2f}).")
            print(f"  - PERFIL DE TAMAÑO: Se enfoca en propiedades de tamaño {'GRANDE' if row['Superficie_m2'] > 80 else 'MEDIANO' if row['Superficie_m2'] > 45 else 'PEQUEÑO'}.")
            print(f"    (La superficie promedio es de {row['Superficie_m2']:.0f} m² y tienen alrededor de {row['Ambientes']:.1f} amb.)")
            print(f"  - BARRIOS TÍPICOS: {', '.join(row['Barrios_Distintivos'])}.")
        print("\n============================================================================")

       # --- VISUALIZACIÓN COMPLETA ---
        print("\n  - [Visualización] Generando dashboard de gráficos para entender la distribución de cada clúster...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Análisis Visual de Clusters de Alquiler (DBSCAN)', fontsize=20)

        sns.scatterplot(ax=axes[0, 0], data=df_final, x='Superficie_m2', y='Precio_USD_m2', hue='Cluster', palette='viridis', s=20, alpha=0.6, legend='full')
        axes[0, 0].set_title('Visión General: Precio/m² vs. Superficie', fontsize=14); axes[0, 0].grid(True)

        sns.boxplot(ax=axes[0, 1], data=df_clusters_only, x='Cluster', y='Precio_USD_m2', palette='viridis'); axes[0, 1].set_title('Distribución de Precio/m² por Cluster', fontsize=14)
        sns.boxplot(ax=axes[1, 0], data=df_clusters_only, x='Cluster', y='Superficie_m2', palette='viridis'); axes[1, 0].set_title('Distribución de Superficie por Cluster', fontsize=14)
        sns.boxplot(ax=axes[1, 1], data=df_clusters_only, x='Cluster', y='Indice_Valor_Barrio', palette='viridis'); axes[1, 1].set_title('Distribución de Valor de Barrio por Cluster', fontsize=14)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
        
    else:
        print("  - No se encontraron clústeres suficientemente densos.")

# --- PASO 6: VALUACIÓN DETALLADA Y EXPORTACIÓN ---
if 'Cluster' in df_final.columns and not df_clusters_only.empty:
    print("\n--- Iniciando Paso 6: Valuación Detallada y Exportación ---")
    stats_clusters = df_clusters_only.groupby('Cluster')['Precio_USD_m2'].agg(['mean', 'std']).reset_index()
    stats_clusters.rename(columns={'mean': 'Precio_Promedio_Cluster_m2', 'std': 'Precio_Std_Cluster_m2'}, inplace=True)
    stats_clusters.fillna(0, inplace=True)
    df_para_powerbi = pd.merge(df_final, stats_clusters, on='Cluster', how='left')
    df_para_powerbi['Valuacion'] = df_para_powerbi.apply(lambda row: 'Outlier/Ruido' if row['Cluster'] == -1 else ('N/A' if pd.isna(row['Precio_Promedio_Cluster_m2']) else ('Sobrevaluada' if row['Precio_USD_m2'] > row['Precio_Promedio_Cluster_m2'] + 0.5 * row['Precio_Std_Cluster_m2'] else ('Infravalorada' if row['Precio_USD_m2'] < row['Precio_Promedio_Cluster_m2'] - 0.5 * row['Precio_Std_Cluster_m2'] else 'Precio Justo'))), axis=1)

    print("\n====================== CRITERIOS DE VALUACIÓN POR CLÚSTER ======================")
    print("A continuación se muestra la lógica de precios para CADA clúster encontrado:")
    for index, row in stats_clusters.iterrows():
        cluster_id = int(row['Cluster'])
        mean_price = row['Precio_Promedio_Cluster_m2']
        std_price = row['Precio_Std_Cluster_m2']
        margen = 0.5 * std_price
        lim_inf = mean_price - margen
        lim_sup = mean_price + margen
        print(f"\n--- Lógica para Clúster {cluster_id} ---")
        print(f"  - Precio Promedio por m²: ${mean_price:,.2f} USD")
        print(f"  - RANGO DE 'PRECIO JUSTO': de ${lim_inf:,.2f} a ${lim_sup:,.2f} USD/m²")
    print("\n============================================================================")

    columnas_export = ['ID', 'URL', 'Fecha', 'Barrio', 'Ambientes', 'Superficie_m2', 'Precio_USD', 'Precio_USD_m2', 'Cluster', 'Valuacion', 'Indice_Valor_Barrio']
    df_export_final = df_para_powerbi[[col for col in columnas_export if col in df_para_powerbi.columns]]

    output_filename = 'alquileres_valuados_para_powerbi.csv'
    df_export_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n- Archivo '{output_filename}' generado con el análisis final.")
    print("\n======================= PIPELINE COMPLETADO Y REFINADO =======================")
    
# ==================================================================================
#  PASO FINAL: HERRAMIENTA DE VALUACIÓN INTERACTIVA EN COLAB
# ==================================================================================
print("\n--- Iniciando Herramienta de Valuación Interactiva ---")

def valuar_propiedad(barrio, ambientes, superficie, precio_usd, 
                      preprocessor_entrenado, cluster_centroids, cluster_stats,
                      indice_barrios):
    # (código de la función sin cambios)
    if preprocessor_entrenado is None or cluster_centroids.empty: return {"error": "El modelo o las estadísticas no están disponibles."}
    data = {'Barrio': [barrio], 'Ambientes': [float(ambientes)], 'Superficie_m2': [float(superficie)], 'Precio_USD': [float(precio_usd)]}
    nueva_prop = pd.DataFrame(data)
    nueva_prop['Precio_USD_m2'] = nueva_prop['Precio_USD'] / nueva_prop['Superficie_m2']
    media_indice_general = indice_barrios.mean()
    nueva_prop['Indice_Valor_Barrio'] = nueva_prop['Barrio'].map(indice_barrios).fillna(media_indice_general)
    nueva_prop['log_Precio_USD_m2'] = np.log1p(nueva_prop['Precio_USD_m2'])
    nueva_prop['log_Superficie_m2'] = np.log1p(nueva_prop['Superficie_m2'])
    features_usadas = ['log_Precio_USD_m2', 'log_Superficie_m2', 'Ambientes', 'Indice_Valor_Barrio']
    nueva_prop_scaled = preprocessor_entrenado.transform(nueva_prop[features_usadas])
    centroids_scaled = preprocessor_entrenado.transform(cluster_centroids)
    distancias = distance.cdist(nueva_prop_scaled, centroids_scaled, 'euclidean')
    cluster_asignado = cluster_centroids.index[np.argmin(distancias)]
    stats_del_cluster = stats_clusters[stats_clusters['Cluster'] == cluster_asignado]
    if stats_del_cluster.empty: return {"cluster_asignado": int(cluster_asignado), "valuacion_final": "N/A"}
    mean_price = stats_del_cluster['Precio_Promedio_Cluster_m2'].iloc[0]
    std_price = stats_del_cluster['Precio_Std_Cluster_m2'].iloc[0]
    lim_sup = mean_price + (0.5 * std_price)
    lim_inf = mean_price - (0.5 * std_price)
    precio_actual_m2 = nueva_prop['Precio_USD_m2'].iloc[0]
    if precio_actual_m2 > lim_sup: valuacion = 'SOBREVALUADA'
    elif precio_actual_m2 < lim_inf: valuacion = 'INFRAVALORADA'
    else: valuacion = 'PRECIO JUSTO'
    return {"cluster_asignado": int(cluster_asignado), "valuacion_final": valuacion, "precio_ingresado_m2": precio_actual_m2, "rango_justo_inf": lim_inf, "rango_justo_sup": lim_sup}

# --- EJECUCIÓN DE LA HERRAMIENTA ---
if 'pipeline' in locals() and pipeline is not None:

    # ======================================================
    # ▼▼▼ INGRESA LOS DATOS DE LA PROPIEDAD A VALUAR AQUÍ ▼▼▼
    # ======================================================
    propiedad_a_valuar = {
        "barrio": "Caballito",
        "ambientes": 3,
        "superficie": 75,
        "precio_usd": 300  # Alquiler mensual en USD
    }
    # ======================================================

    resultado = valuar_propiedad(
        preprocessor_entrenado=pipeline.named_steps['preprocessor'],
        cluster_centroids=cluster_centroids_model,
        cluster_stats=stats_clusters,
        indice_barrios=indice_valor_barrio,
        **propiedad_a_valuar
    )

    # --- PRESENTACIÓN VISUAL DEL RESULTADO ---
    if 'error' not in resultado:
        width = 62
        line_propiedad = f"PROPIEDAD: {propiedad_a_valuar['ambientes']} amb de {propiedad_a_valuar['superficie']}m² en {propiedad_a_valuar['barrio']}"
        line_precio = f"PRECIO INGRESADO: ${propiedad_a_valuar['precio_usd']:,.2f} USD (${resultado['precio_ingresado_m2']:.2f} USD/m²)"
        line_segmento = f"SEGMENTO ASIGNADO: Clúster {resultado['cluster_asignado']}"
        line_rango = f"RANGO JUSTO DEL SEGMENTO: de ${resultado['rango_justo_inf']:.2f} a ${resultado['rango_justo_sup']:.2f} USD/m²"
        veredict_text = f">>> {resultado['valuacion_final']} <<<"
        print("\n\n" + "╔" + "═"*width + "╗"); print("║" + " FICHA DE VALUACIÓN DEL MERCADO".center(width) + "║"); print("╠" + "═"*width + "╣")
        print(f"║ {line_propiedad.ljust(width-2)} ║"); print(f"║ {line_precio.ljust(width-2)} ║"); print("║" + "-"*width + "║")
        print(f"║ {line_segmento.ljust(width-2)} ║"); print(f"║ {line_rango.ljust(width-2)} ║"); print("╠" + "═"*width + "╣")
        print("║" + "VEREDICTO DEL MODELO".center(width) + "║"); print("╠" + "═"*width + "╣")
        print("║" + veredict_text.center(width) + "║"); print("╚" + "═"*width + "╝")

        # --- Gráfico de Valuación (CON ANOTACIÓN LATERAL) ---
        fig, ax = plt.subplots(figsize=(14, 3.5))
        
        lim_inf = resultado['rango_justo_inf']
        lim_sup = resultado['rango_justo_sup']
        precio_actual = resultado['precio_ingresado_m2']
        
        chart_min = min(lim_inf * 0.90, precio_actual * 0.90)
        chart_max = max(lim_sup * 1.10, precio_actual * 1.10)
        
        ax.barh([0], [lim_inf - chart_min], left=chart_min, color='#d9534f', height=1, alpha=0.6)
        ax.barh([0], [lim_sup - lim_inf], left=lim_inf, color='#5cb85c', height=1, alpha=0.7)
        ax.barh([0], [chart_max - lim_sup], left=lim_sup, color='#d9534f', height=1, alpha=0.6)

        ax.text((chart_min + lim_inf) / 2, 0, 'Infravalorado', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax.text((lim_inf + lim_sup) / 2, 0, 'Precio Justo', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax.text((lim_sup + chart_max) / 2, 0, 'Sobrevaluado', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        
        ax.axvline(x=precio_actual, color='yellow', linestyle='--', linewidth=2.5)
 
    # --- Lógica de Posicionamiento de la Anotación (NUEVA Y MEJORADA) ---
        # Decidimos si el texto va a la derecha o a la izquierda del marcador para evitar que se salga del gráfico
        posicion_relativa = (precio_actual - chart_min) / chart_range
        
        if posicion_relativa < 0.5:
            # Si el punto está en la mitad izquierda, ponemos el texto a la derecha
            ha_align = 'left'
            text_x_pos = precio_actual + (chart_range * 0.2)
        else:
            # Si el punto está en la mitad derecha, ponemos el texto a la izquierda
            ha_align = 'right'
            text_x_pos = precio_actual - (chart_range * 0.1)
        
        ax.annotate(f'Tu Propiedad\n${precio_actual:.2f}/m²',
                    xy=(precio_actual, 0), 
                    xytext=(text_x_pos, 0), # La clave: la coordenada Y del texto es la misma que la del punto
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='black'),
                    ha=ha_align, va='center',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9))
        ax.set_xlim(chart_min, chart_max)
        ax.set_yticks([])
        ax.set_title('Análisis de Valuación: Posición del Precio vs. Rango del Segmento', fontsize=16, pad=20)
        ax.set_xlabel('Precio por m² (USD)', fontsize=12)
        plt.show()

    else:
        print(f"ERROR: {resultado['error']}")
else:
    print("\nNo se puede ejecutar la función de valuación porque el modelo no se entrenó.")    
