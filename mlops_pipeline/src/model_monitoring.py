import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare, entropy
from typing import Dict, List, Tuple, Union

class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """
        Inicializa el monitor de modelo.
        
        Args:
            reference_data: DataFrame con datos de entrenamiento (referencia).
            current_data: DataFrame con datos actuales (producción/test).
        """
        self.reference_data = reference_data
        self.current_data = current_data
        self.numeric_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = reference_data.select_dtypes(exclude=[np.number]).columns.tolist()

    def calculate_ks_test(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula el test de Kolmogorov-Smirnov para variables numéricas.
        Retorna un diccionario con el estadístico KS y el p-value para cada variable.
        """
        results = {}
        for col in self.numeric_columns:
            ref_values = self.reference_data[col].dropna()
            curr_values = self.current_data[col].dropna()
            
            statistic, p_value = ks_2samp(ref_values, curr_values)
            
            results[col] = {
                "ks_statistic": statistic,
                "p_value": p_value,
                "drift_detected": p_value < 0.05  # Umbral típico
            }
        return results

    def calculate_psi_numeric(self, buckets: int = 10) -> Dict[str, float]:
        """
        Calcula el Population Stability Index (PSI) para variables numéricas.
        """
        results = {}
        for col in self.numeric_columns:
            ref_values = self.reference_data[col].dropna()
            curr_values = self.current_data[col].dropna()
            
            # Crear bins basados en la referencia
            try:
                ref_percentiles = np.percentile(ref_values, np.linspace(0, 100, buckets + 1))
                ref_percentiles[0] = -np.inf
                ref_percentiles[-1] = np.inf
                
                # Calcular frecuencias
                ref_counts, _ = np.histogram(ref_values, bins=ref_percentiles)
                curr_counts, _ = np.histogram(curr_values, bins=ref_percentiles)
                
                # Calcular proporciones
                ref_ratio = (ref_counts + 1e-6) / len(ref_values) # Evitar división por cero
                curr_ratio = (curr_counts + 1e-6) / len(curr_values)
                
                psi_values = (ref_ratio - curr_ratio) * np.log(ref_ratio / curr_ratio)
                psi = np.sum(psi_values)
                
                results[col] = psi
            except Exception as e:
                print(f"Error calculando PSI para {col}: {e}")
                results[col] = np.nan
                
        return results

    def calculate_chi_square(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula el test de Chi-cuadrado para variables categóricas.
        """
        results = {}
        for col in self.categorical_columns:
            try:
                # Alinear categorías
                ref_counts = self.reference_data[col].value_counts(normalize=True).sort_index()
                curr_counts = self.current_data[col].value_counts(normalize=True).sort_index()
                
                # Asegurar que ambos tengan las mismas categorías (índices)
                all_categories = sorted(list(set(ref_counts.index) | set(curr_counts.index)))
                ref_counts = ref_counts.reindex(all_categories, fill_value=0)
                curr_counts = curr_counts.reindex(all_categories, fill_value=0)
                
                # Convertir a frecuencias absolutas esperadas para el test (usando tamaño de current)
                expected_freq = ref_counts * len(self.current_data)
                observed_freq = curr_counts * len(self.current_data)
                
                # Chi-square requiere frecuencias absolutas, evitamos ceros
                expected_freq = expected_freq + 1e-6
                observed_freq = observed_freq + 1e-6

                statistic, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)
                
                results[col] = {
                    "chi2_statistic": statistic,
                    "p_value": p_value,
                    "drift_detected": p_value < 0.05
                }
            except Exception as e:
                print(f"Error calculando Chi-square para {col}: {e}")
                results[col] = {"error": str(e)}
        return results

    def calculate_jensen_shannon(self, buckets: int = 10) -> Dict[str, float]:
        """
        Calcula la divergencia de Jensen-Shannon.
        """
        results = {}
        # Implementación simplificada para numéricas usando histogramas
        for col in self.numeric_columns:
            ref_values = self.reference_data[col].dropna()
            curr_values = self.current_data[col].dropna()
            
            try:
                ref_percentiles = np.percentile(ref_values, np.linspace(0, 100, buckets + 1))
                ref_counts, _ = np.histogram(ref_values, bins=ref_percentiles)
                curr_counts, _ = np.histogram(curr_values, bins=ref_percentiles)
                
                p = ref_counts / len(ref_values)
                q = curr_counts / len(curr_values)
                
                m = (p + q) / 2
                js_divergence = (entropy(p, m) + entropy(q, m)) / 2
                results[col] = js_divergence
            except Exception as e:
                results[col] = np.nan
        return results

    def run_all_checks(self) -> Dict[str, any]:
        """
        Ejecuta todas las verificaciones y consolida resultados.
        """
        return {
            "ks_test": self.calculate_ks_test(),
            "psi_numeric": self.calculate_psi_numeric(),
            "chi_square": self.calculate_chi_square(),
            "jensen_shannon": self.calculate_jensen_shannon()
        }

if __name__ == "__main__":
    # Prueba rápida
    # Generar datos dummy
    np.random.seed(42)
    df_ref = pd.DataFrame({
        'edad': np.random.normal(30, 5, 1000),
        'ingresos': np.random.exponential(5000, 1000),
        'categoria': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Datos actuales con drift en edad (media desplazada) y categoria
    df_curr = pd.DataFrame({
        'edad': np.random.normal(35, 5, 1000), 
        'ingresos': np.random.exponential(5000, 1000),
        'categoria': np.random.choice(['A', 'B', 'B'], 1000) # Más B
    })
    
    monitor = ModelMonitor(df_ref, df_curr)
    results = monitor.run_all_checks()
    print("Resultados Monitor:")
    print(results)
