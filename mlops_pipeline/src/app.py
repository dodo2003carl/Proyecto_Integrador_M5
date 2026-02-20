import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from model_monitoring import ModelMonitor
from ft_engineering import cargar_datos, feature_engineering
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="Monitor de Modelos MLOps",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una apariencia profesional
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Dashboard de Monitoreo de Modelos y Data Drift")
st.markdown("Verificación continua de la salud del modelo y la estabilidad de los datos.")

@st.cache_data
def load_and_process_data():
    """Carga datos, procesa y simula datos actuales (producción)"""
    try:
        # Cargar datos originales (Referencia)
        if os.path.exists("Base_de_datos.xlsx"):
            df = cargar_datos("Base_de_datos.xlsx")
        elif os.path.exists("../../Base_de_datos.xlsx"):
             df = cargar_datos("../../Base_de_datos.xlsx")
        else:
            st.error("No se encontró el archivo de base de datos.")
            return None, None
            
        # 1. Obtener datos procesados (Numéricos/OneHot) + TARGETS REALES
        # feature_engineering usa test_size=0.2 por defecto (ver ft_engineering.py)
        X_train_proc, X_test_proc, y_train_proc, y_test_proc, _ = feature_engineering(df)
        
        # 2. Obtener datos CRUDOS (Categorías originales) para monitoreo legible
        from sklearn.model_selection import train_test_split
        target_col = 'Pago_atiempo'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # IMPORTANTE: Replicamos el split EXACTO de ft_engineering.py
        X_train_raw, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Asegurar 0.2
        
        # --- CAMBIO IMPORTANTE: PRIORIZAR DATOS CRUDOS ---
        # El usuario quiere ver gráficas "Correctas" (Valores reales: $5000, 30 años), no escalados (0.5, -1.2).
        # Por eso, usamos X_train_raw como base y le adjuntamos lo procesado solo como extra.
        
        X_ref_final = X_train_raw.copy()
        X_curr_final = X_test_raw.copy()
        
        # Eliminar columnas de identificación o fechas que causan falsos positivos en Drift (por alta cardinalidad)
        cols_to_drop = ['id_cliente', 'fecha_prestamo']
        X_ref_final = X_ref_final.drop(columns=[c for c in cols_to_drop if c in X_ref_final.columns])
        X_curr_final = X_curr_final.drop(columns=[c for c in cols_to_drop if c in X_curr_final.columns])
        
        # Opcional: Adjuntar procesados con sufijo por si se requiere debug técnico (pero ocultos del usuario general)
        # X_ref_final = X_ref_final.join(X_train_proc, rsuffix='_proc')
        # X_curr_final = X_curr_final.join(X_test_proc, rsuffix='_proc')
        
        # NOTA: Al usar datos crudos, el monitor calculará Drift sobre el Dinero Real, Edad Real, etc. 
        # Esto es mucho más valioso para el negocio.
        
        # Simular datos actuales (Current) con Drift (EN VARIABLES REALES)
        # if 'salario_cliente' in X_curr_final.columns:
        #     # Simulamos que los nuevos clientes tienen ingresos mucho mayores (+40%)
        #     X_curr_final['salario_cliente'] = X_curr_final['salario_cliente'] * 1.40 
        #     
        # if 'edad_cliente' in X_curr_final.columns:
        #     # Simulamos que los nuevos clientes son más jóvenes (-5 años)
        #     X_curr_final['edad_cliente'] = X_curr_final['edad_cliente'] - 5
        #     
        # if 'tipo_laboral' in X_curr_final.columns:
        #     # Simulamos un cambio drástico en las categorías educativas o laborales
        #     mask = np.random.rand(len(X_curr_final)) < 0.3
        #     # Asignamos una categoría que sepamos que hace ruido o la primera disponible
        #     X_curr_final.loc[mask, 'tipo_laboral'] = X_curr_final['tipo_laboral'].mode()[0]

        return X_ref_final, X_curr_final, y_train_proc, y_test_proc
        
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None, None, None

df_ref, df_curr, y_ref, y_curr = load_and_process_data()

if df_ref is not None:
    # Sidebar
    st.sidebar.title("Configuración")
    st.sidebar.markdown("---")
    # Ajustamos el default a 0.001 para ser menos sensibles al "ruido" natural entre splits
    confidence_level = st.sidebar.slider("Nivel de Confianza (p-value)", 0.001, 0.10, 0.001, format="%.3f")
    psi_threshold = st.sidebar.slider("Umbral PSI (Alerta)", 0.1, 0.5, 0.2)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Este dashboard compara los datos de entrenamiento (Referencia) con los datos más recientes (Actual) para detectar degradación del modelo.")

    # Instanciar Monitor
    monitor = ModelMonitor(df_ref, df_curr)
    
    # Calcular Métricas
    with st.spinner('Realizando análisis estadístico...'):
        results = monitor.run_all_checks()
        ks_results = results['ks_test']
        psi_results = results['psi_numeric']
        chi_results = results['chi_square']
    
    # --- KPIs Generales ---
    st.header("1. Estado de Salud del Modelo")
    
    # Variables con drift
    vars_drift_ks = [var for var, res in ks_results.items() if res['p_value'] < confidence_level]
    vars_drift_psi = [var for var, psi in psi_results.items() if psi > psi_threshold]
    vars_drift_chi = [var for var, res in chi_results.items() if res.get('p_value', 1) < confidence_level]
    
    all_drift_vars = list(set(vars_drift_ks + vars_drift_psi + vars_drift_chi))
    
    drift_score = len(all_drift_vars) / (len(ks_results) + len(chi_results)) if (len(ks_results) + len(chi_results)) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables Numéricas", len(ks_results))
    col1.metric("Variables Categóricas", len(chi_results))
    
    col3.metric("Variables con Drift", len(all_drift_vars), delta_color="inverse")
    
    status_text = "Saludable 🟢"
    status_color = "green"
    if drift_score > 0.1:
        status_text = "Advertencia 🟡"
        status_color = "orange"
    if drift_score > 0.3:
        status_text = "Crítico 🔴"
        status_color = "red"
        
    col4.markdown(f"<div style='background-color: {status_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{status_text}</div>", unsafe_allow_html=True)
    
    # --- NUEVO: Target Drift Analysis ---
    st.markdown("---")
    st.header("1.1 Análisis del Objetivo (Target Drift)")
    
    col_tgt1, col_tgt2 = st.columns([1, 2])
    
    with col_tgt1:
        st.info("Distribución de la Variable Objetivo: 'Pago_atiempo'")
        
        # Dataframe para gráfico
        tgt_df = pd.DataFrame({
            'Clase': np.concatenate([y_ref, y_curr]),
            'Dataset': ['Referencia']*len(y_ref) + ['Actual']*len(y_curr)
        })
        
        # Calcular proporciones
        y_ref_prop = y_ref.value_counts(normalize=True)
        y_curr_prop = y_curr.value_counts(normalize=True)
        
        diff = (y_ref_prop - y_curr_prop).abs().max()
        st.metric("Máxima Desviación (Clases)", f"{diff:.2%}", delta=f"{'⚠️ Drift' if diff > 0.1 else 'Estable'}", delta_color="inverse")

    with col_tgt2:
        fig_tgt = px.histogram(tgt_df, x="Clase", color="Dataset", barmode="group", 
                             color_discrete_map={'Referencia': '#3498db', 'Actual': '#e74c3c'},
                             text_auto=True)
        st.plotly_chart(fig_tgt, use_container_width=True)

    # Alertas
    if len(all_drift_vars) > 0:
        st.error(f"⚠️ **Drift Detectado**: {', '.join(all_drift_vars)}")
        with st.expander("🔍 Ver Detalles y Recomendaciones", expanded=True):
            st.markdown(f"""
            **Diagnóstico:**
            - Se ha detectado un cambio significativo en la distribución de **{len(all_drift_vars)}** variables de entrada.
            - Esto puede indicar cambios en el comportamiento de los clientes o problemas en la recolección de datos.
            
            **Acciones Recomendadas:**
            1.  **Investigar Fuente**: Validar pipelines de datos e ingesta.
            2.  **Evaluar Impacto**: Verificar si el rendimiento del modelo (Accuracy/F1) ha decaído en el set actual.
            3.  **Reentrenar**: Si el drift es legítimo, incorporar los datos recientes al entrenamiento y generar la versión **v1.2.0**.
            """)
    else:
        st.success("✅ El modelo opera dentro de los parámetros estables.")

    # --- Análisis Detallado ---
    st.markdown("---")
    st.header("2. Análisis Detallado")
    
    tab_num, tab_cat, tab_viz = st.tabs(["📊 Variables Numéricas", "📋 Variables Categóricas", "📈 Visualización Comparativa"])
    
    with tab_num:
        st.subheader("Métricas de Drift Numérico (KS & PSI)")
        
        summary_data = []
        for var in ks_results.keys():
            psi_val = psi_results.get(var, 0)
            p_val = ks_results[var]['p_value']
            
            is_drift = p_val < confidence_level or psi_val > psi_threshold
            
            summary_data.append({
                "Variable": var,
                "KS Statistic": f"{ks_results[var]['ks_statistic']:.3f}",
                "P-Value": f"{p_val:.4e}",
                "PSI": f"{psi_val:.3f}",
                "Estado": "🔴 DRIFT" if is_drift else "🟢 OK"
            })
            
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.style.applymap(lambda x: 'color: red; font-weight: bold' if 'DRIFT' in str(x) else 'color: green', subset=['Estado']), use_container_width=True)
        
    with tab_cat:
        st.subheader("Test de Chi-Cuadrado")
        if not chi_results:
            st.info("No hay variables categóricas para analizar.")
        else:
            cat_data = []
            for var, res in chi_results.items():
                if 'error' in res:
                    continue
                p_val = res['p_value']
                is_drift = p_val < confidence_level
                cat_data.append({
                    "Variable": var,
                    "Chi2 Stat": f"{res['chi2_statistic']:.3f}",
                    "P-Value": f"{p_val:.4e}",
                    "Estado": "🔴 DRIFT" if is_drift else "🟢 OK"
                })
            cat_df = pd.DataFrame(cat_data)
            st.dataframe(cat_df.style.applymap(lambda x: 'color: red; font-weight: bold' if 'DRIFT' in str(x) else 'color: green', subset=['Estado']), use_container_width=True)

    with tab_viz:
        st.subheader("3.1. Comparación Visual Detallada")
        
        col_sel, col_empty = st.columns([1, 2])
        all_cols = df_ref.columns.tolist()
        
        # Priorizar variables con drift
        default_idx = 0
        if len(all_drift_vars) > 0:
            try:
                default_idx = all_cols.index(all_drift_vars[0])
            except:
                pass
                
        with col_sel:
            selected_var = st.selectbox("Seleccionar Variable Principal", all_cols, index=default_idx)
        
        if selected_var in df_ref.select_dtypes(include=[np.number]).columns:
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.markdown("**Distribución Acumulada (ECDF)**")
                # Gráfico ECDF (Más profesional que el histograma simple para estadística)
                fig_ecdf = px.ecdf(pd.DataFrame({
                    'Valor': np.concatenate([df_ref[selected_var], df_curr[selected_var]]),
                    'Dataset': ['Referencia']*len(df_ref) + ['Actual']*len(df_curr)
                }), x="Valor", color="Dataset", color_discrete_map={'Referencia': '#3498db', 'Actual': '#2ecc71'})
                st.plotly_chart(fig_ecdf, use_container_width=True)
                
            with col_graph2:
                st.markdown("**Violin Plot (Densidad + Box)**")
                data_combined = pd.DataFrame({
                    'Valor': np.concatenate([df_ref[selected_var], df_curr[selected_var]]),
                    'Dataset': ['Referencia'] * len(df_ref) + ['Actual'] * len(df_curr)
                })
                # Violin plot es más "profesional" y denso que boxplot
                fig2 = px.violin(data_combined, x="Dataset", y="Valor", color="Dataset", box=True, points="all",
                               color_discrete_map={'Referencia': '#3498db', 'Actual': '#e74c3c'})
                st.plotly_chart(fig2, use_container_width=True)
            
            # Gráfico de dispersión 3D (Solo por elegancia visual si hay otra variable numérica)
            st.markdown("**Interacción Multivariable (3D Scatter)**")
            nums = df_ref.select_dtypes(include=[np.number]).columns.tolist()
            if len(nums) > 2:
                var2 = nums[1] if nums[1] != selected_var else nums[0]
                var3 = nums[2] if nums[2] != selected_var else nums[0]
                
                # Samplear para performance
                sample_idx = np.random.choice(len(df_ref), min(500, len(df_ref)), replace=False)
                df_3d = df_ref.iloc[sample_idx].copy()
                df_3d['Color'] = y_ref.iloc[sample_idx].astype(str).values # Colorear por target real
                
                fig_3d = px.scatter_3d(df_3d, x=selected_var, y=var2, z=var3, color='Color',
                                     title=f"Interacción 3D: {selected_var} vs {var2} vs {var3}",
                                     opacity=0.7, color_discrete_sequence=px.colors.qualitative.Bold)
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
                st.plotly_chart(fig_3d, use_container_width=True)
            
        else:
            # Gráfico para categóricas
            st.markdown(f"**Distribución de Categorías: {selected_var}**")
            
            # Asegurar que la variable se trata como string para el gráfico axis
            # Limitar a top 15 para evitar saturación
            top_n = 15
            
            # Limpieza específica: Filtrar valores que parecen números (basura en columnas categóricas)
            def is_valid_category(val):
                s = str(val)
                if s.lower() == 'nan': return False
                try:
                    float(s) # Si se puede convertir a número, es basura en una col categórica
                    return False
                except:
                    return True

            # Filtrar DF para gráfico
            df_ref_clean = df_ref[df_ref[selected_var].apply(is_valid_category)]
            df_curr_clean = df_curr[df_curr[selected_var].apply(is_valid_category)]

            val_counts_ref = df_ref_clean[selected_var].astype(str).value_counts(normalize=True).head(top_n).reset_index()
            val_counts_ref.columns = [selected_var, 'Proporción']
            val_counts_ref['Dataset'] = 'Referencia'
            
            val_counts_curr = df_curr_clean[selected_var].astype(str).value_counts(normalize=True).head(top_n).reset_index()
            val_counts_curr.columns = [selected_var, 'Proporción']
            val_counts_curr['Dataset'] = 'Actual'
            
            prop_df = pd.concat([val_counts_ref, val_counts_curr])
            
            fig = px.bar(prop_df, x=selected_var, y="Proporción", color="Dataset", barmode="group", 
                         color_discrete_map={'Referencia': '#3498db', 'Actual': '#e74c3c'})
            
            # Forzar eje X a categoría para evitar que Plotly trate números como rango continuo
            fig.update_xaxes(type='category', categoryorder='total descending')
            st.plotly_chart(fig, use_container_width=True)

    # --- NUEVA SECCIÓN: Galería de Gráficos (Para cumplir meta de 20 gráficos) ---
    st.markdown("---")
    st.header("4. Galería de Variables (Vista Rápida)")
    st.markdown("Visualización compacta de todas las variables monitoreadas.")
    
    # Seleccionar top 20 variables (o todas si son menos) para mostrar
    cols_to_show = all_cols[:24] # Grid de 4x6
    
    # Dividir en filas de 4 columnas
    rows = [cols_to_show[i:i + 4] for i in range(0, len(cols_to_show), 4)]
    
    for row in rows:
        cols = st.columns(4)
        for i, var_name in enumerate(row):
            with cols[i]:
                # Mini gráfico ligero
                if var_name in df_ref.select_dtypes(include=[np.number]).columns:
                    # Usar datos muestreados para velocidad si es necesario
                    ref_sample = df_ref[var_name].sample(min(500, len(df_ref)))
                    curr_sample = df_curr[var_name].sample(min(500, len(df_curr)))
                    
                    mini_df = pd.DataFrame({
                        'Val': np.concatenate([ref_sample, curr_sample]),
                        'Set': ['Ref']*len(ref_sample) + ['Act']*len(curr_sample)
                    })
                    
                    # Sparkline style histogram
                    fig_mini = px.histogram(mini_df, x="Val", color="Set", nbins=20, 
                                          color_discrete_map={'Ref': '#3498db', 'Act': '#e74c3c'},
                                          barmode="overlay", opacity=0.6)
                    fig_mini.update_layout(
                        title=dict(text=var_name, font=dict(size=10)),
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=150,
                        xaxis=dict(showticklabels=False, title=None),
                        yaxis=dict(showticklabels=False, title=None)
                    )
                    st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})
                else:
                    # Mini bar chart for categorical
                    st.caption(f"📊 {var_name} (Cat)")

    # --- NUEVA SECCIÓN: Mapa de Calor ---
    st.markdown("---")
    st.header("5. Análisis de Correlaciones")
    
    # Correlación Numérica
    numeric_ref = df_ref.select_dtypes(include=[np.number])
    if not numeric_ref.empty:
        # Importancia de Variables (Correlación con Target)
        # Unimos X e y temporalmente para calcular correlación
        full_train = numeric_ref.copy()
        # Tratamos de pegar el target numérico si es posible
        try:
            full_train['TARGET'] = y_ref.values
            corrs = full_train.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)
            
            st.subheader("Importancia de Variables (Correlación con Target)")
            fig_imp = px.bar(x=corrs.index, y=corrs.values, color=corrs.values, 
                           color_continuous_scale='Viridis', title="Ranking de Influencia Global")
            st.plotly_chart(fig_imp, use_container_width=True)
        except:
            st.warning("No se pudo calcular correlación con target (tipo de dato incompatible).")

        corr_ref = numeric_ref.corr()
        corr_curr = df_curr[numeric_ref.columns].corr()
        
        col_corr1, col_corr2 = st.columns(2)
        with col_corr1:
            st.subheader("Matriz Correlación (Referencia)")
            fig_corr1 = px.imshow(corr_ref, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_corr1.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_corr1, use_container_width=True)
            
        with col_corr2:
            st.subheader("Matriz Correlación (Actual)")
            fig_corr2 = px.imshow(corr_curr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_corr2.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_corr2, use_container_width=True)
    else:
        st.info("No hay suficientes variables numéricas para correlación.")

else:
    st.info("Iniciando sistema... Por favor espere mientras se cargan y procesan los datos.")
