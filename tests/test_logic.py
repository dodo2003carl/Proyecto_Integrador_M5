import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from mlops_pipeline.src.ft_engineering import cargar_datos, feature_engineering
from mlops_pipeline.src.model_training_evaluation import evaluate_model
from sklearn.ensemble import RandomForestClassifier

def test_cargar_datos_excel(tmp_path):
    """Prueba la carga de datos desde un archivo Excel (simulado con CSV para simplicidad en test)"""
    # En realidad cargar_datos usa pd.read_excel para .xlsx, pero para el test creamos un CSV
    d = {'col1': [1, 2], 'col2': [3, 4], 'Pago_atiempo': [1, 0]}
    df = pd.DataFrame(data=d)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    
    loaded_df = cargar_datos(str(file_path))
    assert loaded_df.shape == (2, 3)
    assert 'Pago_atiempo' in loaded_df.columns

def test_cargar_datos_not_found():
    """Prueba el error al no encontrar el archivo"""
    with pytest.raises(FileNotFoundError):
        cargar_datos("archivo_inexistente.xlsx")

def test_feature_engineering_basic():
    """Prueba el pipeline de ingeniería de características con datos suficientes para estratificar"""
    data = {
        'tipo_laboral': ['Fijo', 'Temporal'] * 10,
        'tendencia_ingresos': ['Estable', 'Baja', 'Crece', 'Estable'] * 5,
        'salario_cliente': [1000, 2000, 1500, 3000] * 5,
        'edad_cliente': [20, 30, 40, 50] * 5,
        'Pago_atiempo': [1, 0, 1, 0] * 5
    }
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test, preprocessor = feature_engineering(df)
    
    assert not X_train.empty
    assert not X_test.empty
    assert len(y_train) == 16
    assert len(y_test) == 4
    
    # Probar transformación de nuevos datos
    X_new = preprocessor.transform(df.drop(columns=['Pago_atiempo']).head(1))
    assert X_new.shape[0] == 1

def test_feature_engineering_invalid_target():
    """Prueba error al no encontrar columna target"""
    df = pd.DataFrame({'A': [1, 2]})
    with pytest.raises(ValueError, match="La columna objetivo"):
        feature_engineering(df, target_col='Inexistente')

def test_evaluate_model_comprehensive():
    """Prueba la función de evaluación con métricas variadas"""
    X = pd.DataFrame({'feat1': np.random.rand(10), 'feat2': np.random.rand(10)})
    y = pd.Series([0, 1] * 5)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    
    metrics, report = evaluate_model("LogReg", model, X, y)
    assert metrics['Accuracy'] >= 0
    assert 'Modelo' in metrics
    assert 'ROC-AUC' in metrics

def test_cargar_datos_csv(tmp_path):
    """Prueba carga explícita de CSV"""
    df = pd.DataFrame({'A': [1], 'Pago_atiempo': [1]})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    assert cargar_datos(str(p)).shape == (1, 2)

def test_cargar_datos_unsupported(tmp_path):
    """Prueba formato no soportado"""
    p = tmp_path / "data.txt"
    p.write_text("dummy")
    with pytest.raises(ValueError, match="Formato de archivo no soportado"):
        cargar_datos(str(p))

def test_training_main_logic(tmp_path, monkeypatch):
    """Prueba la ejecución de la función main de entrenamiento (simulada)"""
    # Mock de cargar_datos y feature_engineering para no depender de archivos externos pesados
    df_dummy = pd.DataFrame({
        'salario_cliente': [1000] * 10,
        'edad_cliente': [30] * 10,
        'tipo_laboral': ['Fijo'] * 10,
        'tendencia_ingresos': ['Estable'] * 10,
        'Pago_atiempo': [1, 0] * 5
    })
    
    # Usar tmp_path para los modelos
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Cambiar al directorio temporal para que main cree los modelos allí
    monkeypatch.chdir(tmp_path)
    
    from mlops_pipeline.src.model_training_evaluation import main as train_main
    
    # Mock de cargar_datos para que devuelva nuestro df dummy
    import mlops_pipeline.src.model_training_evaluation as mte
    monkeypatch.setattr(mte, "cargar_datos", lambda x: df_dummy)
    
    # Ejecutar main
    train_main()
    
    assert (model_dir / "modelo_final.pkl").exists()
    assert (model_dir / "preprocesador.pkl").exists()

def test_model_monitor_logic():
    """Prueba la lógica de ModelMonitor con datos simulados"""
    from mlops_pipeline.src.model_monitoring import ModelMonitor
    df_ref = pd.DataFrame({
        'a': [1, 2, 3, 4, 5], 
        'b': ['x', 'y', 'x', 'y', 'x']
    })
    df_curr = pd.DataFrame({
        'a': [1, 2, 10, 11, 12], 
        'b': ['x', 'z', 'z', 'z', 'z']
    })
    monitor = ModelMonitor(df_ref, df_curr)
    
    ks_res = monitor.calculate_ks_test()
    assert 'a' in ks_res
    
    psi_res = monitor.calculate_psi_numeric(buckets=2)
    assert 'a' in psi_res
    
    chi_res = monitor.calculate_chi_square()
    assert 'b' in chi_res
    
    js_res = monitor.calculate_jensen_shannon(buckets=2)
    assert 'a' in js_res
    
    all_res = monitor.run_all_checks()
    assert "ks_test" in all_res

def test_save_model_main_logic(tmp_path, monkeypatch):
    """Prueba la función de guardado de modelo"""
    from mlops_pipeline.src.save_model import train_and_save
    df_dummy = pd.DataFrame({
        'salario_cliente': [1000] * 10,
        'edad_cliente': [30] * 10,
        'tipo_laboral': ['Fijo'] * 10,
        'tendencia_ingresos': ['Estable'] * 10,
        'Pago_atiempo': [1, 0] * 5
    })
    
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    
    import mlops_pipeline.src.save_model as sm
    monkeypatch.setattr(sm, "cargar_datos", lambda x: df_dummy)
    
    train_and_save()
    assert (tmp_path / "models" / "modelo_final.pkl").exists()

def test_dashboard_load_data_logic(monkeypatch):
    """Prueba la lógica de carga de datos del dashboard (app.py)"""
    # 1. Mock streamlit cache BEFORE any app imports
    # Handle both @st.cache_data and @st.cache_data()
    def mock_cache(f=None, **kwargs):
        if f is None:
            return lambda func: func
        return f
    monkeypatch.setattr("streamlit.cache_data", mock_cache)
    
    # 2. Mock os.path.exists to simulate file presence selectively
    real_exists = os.path.exists
    def mock_exists(path):
        if "Base_de_datos.xlsx" in str(path):
            return True
        return real_exists(path)
    monkeypatch.setattr("os.path.exists", mock_exists)

    df_dummy = pd.DataFrame({
        'salario_cliente': [1000, 2000, 3000, 4000, 5000, 1000, 2000, 3000, 4000, 5000],
        'edad_cliente': [20, 30, 40, 50, 60, 20, 30, 40, 50, 60],
        'tipo_laboral': ['Fijo', 'Fijo', 'Fijo', 'Fijo', 'Fijo', 'Fijo', 'Fijo', 'Fijo', 'Fijo', 'Fijo'],
        'tendencia_ingresos': ['Estable', 'Estable', 'Estable', 'Estable', 'Estable', 'Estable', 'Estable', 'Estable', 'Estable', 'Estable'],
        'id_cliente': range(10),
        'fecha_prestamo': pd.date_range('2023-01-01', periods=10),
        'Pago_atiempo': [1, 0] * 5
    })
    
    # 3. Mock ft_engineering.cargar_datos globally
    import mlops_pipeline.src.ft_engineering as fte_mod
    monkeypatch.setattr(fte_mod, "cargar_datos", lambda x: df_dummy)

    # 4. Now import app (this applies the mocks to app at definition time)
    import mlops_pipeline.src.app as app_mod
    
    # 5. Explicitly patch app's internal aliases
    monkeypatch.setattr(app_mod, "_cargar_datos", lambda x: df_dummy)
    
    # 6. Execute from the module to ensure we use the patched version
    df_ref, df_curr, y_ref, y_curr = app_mod.load_and_process_data()
    
    assert df_ref is not None
    assert 'id_cliente' not in df_ref.columns
    # The counts should match the split logic (80/20 of 10 rows)
    assert len(df_ref) == 8
    assert len(df_curr) == 2
