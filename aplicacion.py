from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ===== INICIALIZACIÓN DEL MODELO =====
breast_cancer = load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
df = pd.DataFrame(X, columns=breast_cancer.feature_names)

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=42, stratify=Y)

# Normalización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementación de Red Neuronal con TensorFlow
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Crear y entrenar el modelo
model = create_model()

# Entrenamiento del modelo
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Predicciones de prueba para métricas
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# ===== TRADUCCIÓN DE CARACTERÍSTICAS AL ESPAÑOL =====
traducciones = {
    'mean radius': 'Radio medio',
    'mean texture': 'Textura media',
    'mean perimeter': 'Perímetro medio',
    'mean area': 'Área media',
    'mean smoothness': 'Suavidad media',
    'mean compactness': 'Compacidad media',
    'mean concavity': 'Concavidad media',
    'mean concave points': 'Puntos cóncavos medios',
    'mean symmetry': 'Simetría media',
    'mean fractal dimension': 'Dimensión fractal media',
    'radius error': 'Error del radio',
    'texture error': 'Error de textura',
    'perimeter error': 'Error del perímetro',
    'area error': 'Error del área',
    'smoothness error': 'Error de suavidad',
    'compactness error': 'Error de compacidad',
    'concavity error': 'Error de concavidad',
    'concave points error': 'Error de puntos cóncavos',
    'symmetry error': 'Error de simetría',
    'fractal dimension error': 'Error de dimensión fractal',
    'worst radius': 'Peor radio',
    'worst texture': 'Peor textura',
    'worst perimeter': 'Peor perímetro',
    'worst area': 'Peor área',
    'worst smoothness': 'Peor suavidad',
    'worst compactness': 'Peor compacidad',
    'worst concavity': 'Peor concavidad',
    'worst concave points': 'Peores puntos cóncavos',
    'worst symmetry': 'Peor simetría',
    'worst fractal dimension': 'Peor dimensión fractal'
}

# ===== APLICACIÓN FLASK =====
app = Flask(__name__)

@app.route('/')
def index():
    """Muestra la página principal con el formulario"""
    # Crear una lista de características con información sobre rangos y nombres traducidos
    caracteristicas_info = []
    for i, feature in enumerate(breast_cancer.feature_names):
        nombre_traducido = traducciones.get(feature, feature)  # Usar traducción o original si no existe
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        caracteristicas_info.append({
            'nombre_original': feature,
            'nombre_traducido': nombre_traducido,
            'min': min_val,
            'max': max_val,
            'ejemplo': float(df[feature].iloc[0])  # valor de ejemplo
        })
    
    return render_template('formulario.html', 
                         caracteristicas=caracteristicas_info,
                         accuracy=accuracy*100)

@app.route('/predecir', methods=['POST'])
def predecir():
    """Procesa los datos del formulario y devuelve el resultado"""
    try:
        print("Datos recibidos del formulario:", request.form)
        
        # Obtener datos del formulario (usando nombres originales en inglés)
        nuevos_datos = []
        for feature in breast_cancer.feature_names:
            valor = request.form.get(feature, '').strip()
            if not valor:
                return f"Error: Falta el valor para {feature}", 400
            try:
                nuevos_datos.append(float(valor))
            except ValueError:
                return f"Error: Valor no válido para {feature}: {valor}", 400
        
        print("Datos procesados:", nuevos_datos)
        
        # Realizar predicción con la red neuronal
        nuevo_paciente_array = np.array([nuevos_datos])
        nuevo_paciente_scaled = scaler.transform(nuevo_paciente_array)
        
        prediccion_proba = model.predict(nuevo_paciente_scaled, verbose=0)
        prediccion = (prediccion_proba > 0.5).astype(int).flatten()[0]
        confianza = float(prediccion_proba[0][0] if prediccion == 1 else 1 - prediccion_proba[0][0])
        
        # Preparar resultado
        resultado = "POSITIVO" if prediccion == 1 else "NEGATIVO"
        
        return render_template('resultado.html', 
                             resultado=resultado, 
                             confianza=confianza*100,
                             prediccion_num=prediccion,
                             sensibilidad=sensitivity*100,
                             especificidad=specificity*100)
    
    except Exception as e:
        print(f"Error completo: {str(e)}")
        return f"Error al procesar los datos: {str(e)}", 500

@app.route('/ejemplo')
def ejemplo():
    """Muestra un ejemplo real del dataset"""
    ejemplo_idx = 0
    ejemplo_real = X_test.iloc[ejemplo_idx]
    verdad_real = y_test[ejemplo_idx]
    
    # Convertir y predecir con la red neuronal
    ejemplo_array = np.array([ejemplo_real])
    ejemplo_scaled = scaler.transform(ejemplo_array)
    prediccion_proba = model.predict(ejemplo_scaled, verbose=0)
    prediccion = (prediccion_proba > 0.5).astype(int).flatten()[0]
    
    resultado_real = "MALIGNO" if verdad_real == 1 else "BENIGNO"
    resultado_pred = "MALIGNO" if prediccion == 1 else "BENIGNO"
    correcto = verdad_real == prediccion
    
    # Preparar datos del ejemplo para mostrar con nombres traducidos
    ejemplo_datos = []
    for i, feature in enumerate(breast_cancer.feature_names):
        nombre_traducido = traducciones.get(feature, feature)
        ejemplo_datos.append({
            'nombre_original': feature,
            'nombre_traducido': nombre_traducido,
            'valor': float(ejemplo_real[i])
        })
    
    return render_template('ejemplo.html',
                         ejemplo_datos=ejemplo_datos[:10],  # Mostrar solo las primeras 10
                         resultado_real=resultado_real,
                         resultado_pred=resultado_pred,
                         correcto=correcto,
                         accuracy=accuracy*100)

@app.route('/metricas')
def metricas():
    """Muestra las métricas del modelo"""
    return render_template('metricas.html',
                         accuracy=accuracy*100,
                         sensitivity=sensitivity*100,
                         specificity=specificity*100,
                         tn=tn, fp=fp, fn=fn, tp=tp)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8000)