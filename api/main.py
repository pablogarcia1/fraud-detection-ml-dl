from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Inicializamos la aplicación FastAPI
app = FastAPI(
    title="API de Detección de Fraude",
    description="Motor de inferencia con XGBoost para transacciones bancarias",
    version="1.0.0"
)

# 2. Carga de los artefactos en memoria
try:
    modelo_xgb = joblib.load('../models/xgboost_fraude.pkl')
    # scaler = joblib.load('../modelos_exportados/robust_scaler.pkl')
    print("Modelo cargado correctamente en memoria.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")


# 3. Definimos el "Contrato" de los datos esperados
class Transaccion(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# 4. Creamos el Endpoint principal de predicción
@app.post("/predecir/")
def predecir_fraude(datos: Transaccion):
    try:
        # Convertimos los datos a un diccionario
        datos_dict = datos.model_dump()

        # Lo pasamos a DataFrame
        df_entrada = pd.DataFrame([datos_dict])

        # --- ZONA DE INFERENCIA ---
        probabilidad = modelo_xgb.predict_proba(df_entrada)[0][1]
        prediccion = modelo_xgb.predict(df_entrada)[0]

        # Preparamos la respuesta
        resultado = {
            "es_fraude": bool(prediccion == 1),
            "probabilidad_fraude": round(float(probabilidad), 4),
            "alerta_sistema": "BLOQUEAR TARJETA" if prediccion == 1 else "TRANSACCION APROBADA"
        }

        return resultado

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 5. Endpoint de salud
@app.get("/")
def health_check():
    return {"status": "API Activa", "modelo": "XGBoost Optimizado"}