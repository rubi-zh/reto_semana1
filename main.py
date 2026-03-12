"""
Reto 01: Humano vs Máquina - Clasificación de Pingüinos
Autor: Zhang Tan Rubi
Descripción: Clasificador experto humano vs Modelo de Machine Learning
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ═══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS 
# ═══════════════════════════════════════════════════════════════════════════

# Cargar dataset desde seaborn
df_original = sns.load_dataset('penguins')

# Limpiar datos (eliminar rows con NaN)
df = df_original.dropna().reset_index(drop=True)

# Definir features y target
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]
y = df['species']

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLASIFICADOR HUMANO (versión mejorada v2)
# ═══════════════════════════════════════════════════════════════════════════

def clasificador_humano_v2(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Versión mejorada del clasificador humano.
    """
    
    # Regla 1: Adelie con pico muy profundo (aunque sea largo)
    if bill_depth_mm > 20.0 and flipper_length_mm < 200:
        return 'Adelie'
    
    # Regla 2: Gentoo - aletas muy largas
    if flipper_length_mm > 207:
        return 'Gentoo'
    
    # Regla 3: Gentoo con flipper límite + bill_depth muy bajo
    elif flipper_length_mm >= 203 and bill_depth_mm < 15.0:
        return 'Gentoo'
    
    # Regla 4: Chinstrap - pico largo
    elif bill_length_mm > 45:
        return 'Chinstrap'
    
    # Regla 5: Adelie - el resto
    else:
        return 'Adelie'

# ═══════════════════════════════════════════════════════════════════════════
# 3. CLASIFICADOR ML (Machine Learning)
# ═══════════════════════════════════════════════════════════════════════════

# Entrenar modelo (Decision Tree con random_state=42)
modelo_ml = DecisionTreeClassifier(random_state=42)
modelo_ml.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════════════════════════
# 4. GENERAR SALIDA 
# ═══════════════════════════════════════════════════════════════════════════

# Iterar sobre cada muestra del conjunto de test
for i in range(len(X_test)):
    # Obtener datos de la fila actual
    fila = X_test.iloc[i]
    
    # Predicción Humana
    pred_humano = clasificador_humano_v2(
        fila['bill_length_mm'], 
        fila['bill_depth_mm'], 
        fila['flipper_length_mm'], 
        fila['body_mass_g']
    )
    
    # Predicción ML
    pred_ml = modelo_ml.predict(fila.to_frame().T)[0]
    
    # IMPRIMIR SOLO LA SALIDA REQUERIDA (Humano,ML sin espacio)
    print(f"{pred_humano},{pred_ml}")