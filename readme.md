# Proyecto de Clasificación: Éxito de Ventas de Videojuegos (MLPClassifier con Scikit-learn)

Este proyecto implementa una Red Neuronal Artificial (RNA) utilizando `MLPClassifier` de Scikit-learn para clasificar si un videojuego tendrá éxito en ventas a nivel global (Ventas Globales $\ge 1.0$ millón).

## 1. Definición de la Columna Objetivo

### Pregunta: ¿Qué es la columna objetivo (`Exito_Ventas`) y cómo se define?

La columna objetivo (`Exito_Ventas`) es una variable **binaria (0 o 1)** que transforma el problema de regresión de ventas en un problema de **clasificación** (éxito/no éxito).

* **Valor 1 (Éxito):** Asignado si las `Global_Sales` (Ventas Globales) son mayores o iguales a $1.0$ millón de unidades.
* **Valor 0 (No Éxito):** Asignado si las `Global_Sales` son menores a $1.0$ millón de unidades.

---

## 2. Origen de Datos, Variables y Análisis Exploratorio

El código procesa el dataset `"video games sales.csv"`.

### Origen y Contexto de los Datos

El dataset contiene información histórica sobre ventas de videojuegos, incluyendo:

* **Ventas Regionales:** Múltiples variables de ventas desagregadas por ubicación: `NA_Sales` (Norteamérica), `EU_Sales` (Europa), `JP_Sales` (Japón), y `Other_Sales`.
* **Características del Juego:** Variables como `Platform`, `Genre`, `Publisher`, `Rank` y `Year`.

### Pasos del Análisis Exploratorio (AE) y Transformaciones

| Paso | Descripción de la Tarea | Observación Clave |
| :--- | :--- | :--- |
| **Creación Objetivo** | Se crea la variable binaria `Exito_Ventas`. | Permite abordar el problema con un modelo de clasificación. |
| **Detección de Outliers (IQR)** | Se evalúa la presencia de valores extremos en las columnas de ventas. | Se detecta la **presencia de numerosos outliers** en las métricas de ventas, lo cual es típico en datos de ingresos/popularidad. |
| **Correlación** | Se analiza la correlación de variables numéricas con `Exito_Ventas`. | Se espera una alta correlación entre las ventas regionales y el éxito global. |
| **Normalización** | Se aplica `MinMaxScaler` a las columnas de ventas. | Las características se escalan al rango $[0, 1]$, mejorando la estabilidad y velocidad de convergencia de la Red Neuronal. |
| **Codificación** | Se aplica `pd.get_dummies` a `Platform`, `Genre`, y `Publisher`. | Las variables categóricas se convierten a un formato numérico (*one-hot encoding*) utilizable por el `MLPClassifier`. |

---

## 3. Modelado, Validaciones y Optimizaciones

Se utiliza un clasificador de Red Neuronal Artificial para el problema de clasificación binaria.

### Modelo Elegido: `MLPClassifier`

El modelo utilizado es el **Multi-layer Perceptron Classifier** (`MLPClassifier`), que es una RNA tipo *Feedforward*.

* **Arquitectura:** Dos capas ocultas con tamaño `(64, 32)`, proporcionando una capacidad significativa para modelar relaciones **no lineales**.
* **Función de Activación:** Se utiliza **ReLU** (`activation='relu'`).

### Validaciones, Optimizaciones y Regularizaciones

| Técnica | Implementación en el Código | Propósito |
| :--- | :--- | :--- |
| **Validación** | **División de datos 80/20** (`train_test_split`). | Crea un conjunto de validación (`X_val, y_val`) independiente para evaluar el modelo y aplicar *Early Stopping*. |
| **Optimización/Regularización** | **`Early Stopping`** (Implementación manual con `patience = 20`). | Previene el **sobreajuste (overfitting)** deteniendo el entrenamiento si la precisión en el conjunto de validación no mejora durante un número (`patience`) de épocas. |
| **Optimización** | **`warm_start=True`** y `max_iter=1` en el bucle. | Permite el **entrenamiento incremental** época por época, reutilizando los pesos de la época anterior en lugar de re-inicializar el modelo en cada llamada a `clf.fit()`. |
| **Regularización** | **Ajuste de Hiperparámetros** (`learning_rate_init=0.05`). | Controla el tamaño de los pasos durante el descenso de gradiente, afectando la estabilidad y la velocidad de convergencia. |