# Proyecto de Clasificación: Éxito de Ventas de Videojuegos (MLPClassifier con Scikit-learn)

Este proyecto implementa una Red Neuronal Artificial (RNA) utilizando `MLPClassifier` de Scikit-learn para clasificar si un videojuego tendrá éxito en ventas a nivel global (Ventas Globales $\ge 1.0$ millón).

## 1. Definición de la Columna Objetivo

### Pregunta: ¿Qué es la columna objetivo (`Exito_Ventas`) y cómo se define?

La columna objetivo (`Exito_Ventas`) es una variable de **clasificación binaria (0 o 1)** que transforma el problema de regresión de ventas en un problema de **clasificación** (éxito/no éxito).

* **Valor 1 (Éxito):** Asignado si las `Global_Sales` (Ventas Globales) son mayores o iguales a $1.0$ millón de unidades.
* **Valor 0 (No Éxito):** Asignado si las `Global_Sales` son menores a $1.0$ millón de unidades.

---

## 2. Origen de Datos, Variables y Análisis Exploratorio

El código procesa el dataset `"video games sales.csv"` obtenido de Kaggle.

### Origen y Contexto de los Datos

El dataset contiene información histórica sobre ventas de videojuegos, incluyendo:

* **Ventas Regionales:** Múltiples variables de ventas clasificadas por ubicación: `NA_Sales` (Norteamérica), `EU_Sales` (Europa), `JP_Sales` (Japón), y `Other_Sales`.
* **Características del Juego:** Variables como `Platform`, `Genre`, `Publisher`, `Rank` y `Year`.

### Pasos del Análisis Exploratorio (AE) y Transformaciones

| Paso | Descripción de la Tarea | Observación Clave |
| :--- | :--- | :--- |
| **Creación Objetivo** | Se crea la variable binaria `Exito_Ventas`. | Permite abordar el problema con un modelo de clasificación. |
| **Detección de Outliers (IQR)** | Se evalúa la presencia de valores extremos en las columnas de ventas. | Se detecta la **presencia de numerosos outliers** en las métricas de ventas para detectar valores atípicos |
| **Correlación** | Se analiza la correlación de variables numéricas con `Exito_Ventas`. | Se espera una alta correlación entre las ventas regionales y el éxito global. |
| **Normalización** | Se aplica `MinMaxScaler` a las columnas de ventas. | Las características se escalan al rango $[0, 1]$, mejorando la estabilidad y velocidad de convergencia de la Red Neuronal. |
| **Codificación** | Se aplica `pd.get_dummies` a `Platform`, `Genre`, y `Publisher`. | Las variables categóricas se convierten a un formato numérico (*one-hot encoding*) utilizable por el `MLPClassifier`. |

---

## 3. Modelado, Validaciones y Optimizaciones

Se utiliza un clasificador de Red Neuronal Artificial para el problema de clasificación binaria.

Una **Red Neuronal Artificial (RNA) tipo Feedforward** es la arquitectura de red neuronal **más básica y fundamental**. Se caracteriza porque el flujo de información es **unidireccional** y **sin ciclos**.

### Características Clave

1.  **Flujo Unidireccional:** La información solo viaja hacia adelante (de "izquierda" a "derecha") :
    * Comienza en la **capa de entrada**.
    * Pasa por una o más **capas ocultas**.
    * Termina en la **capa de salida**.
2.  **Sin Bucles:** No hay conexiones que permitan que la salida de una neurona retroceda a una capa anterior o a la misma capa, lo que la diferencia de las redes recurrentes (RNN).
3.  **Aplicación:** Es ideal para tareas de **clasificación** (como la utilizada en el proyecto) y **regresión**, donde se mapea una entrada a una salida sin depender de la secuencia o el tiempo.

El modelo **`MLPClassifier`** (Multi-layer Perceptron Classifier) utilizado en este proyecto es un ejemplo canónico de una RNA Feedforward.

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

## 📚 Explicación de Términos Clave (MLPClassifier)

A continuación, hay conceptos clave de optimización y validación implementados en el código, importantes para el entrenamiento de la Red Neuronal (`MLPClassifier`).

---

### **1. Validación y Regularización**

#### **Early Stopping** (Parada Temprana)

El **Early Stopping** es una técnica de **regularización** y **optimización** que tiene como objetivo principal evitar el **sobreajuste (overfitting)**.

* **Mecanismo:** El entrenamiento se detiene antes de alcanzar el máximo número de épocas (`max_epochs`).
* **Criterio:** La detención ocurre cuando el rendimiento del modelo (generalmente la precisión o la pérdida) en el conjunto de **validación** deja de mejorar durante un número predefinido de iteraciones, conocido como **paciencia** (`patience`).
* **Beneficio:** Evita que el modelo aprenda demasiado los ruidos del conjunto de entrenamiento, preservando su capacidad de **generalización** sobre datos no vistos. 

#### **Overfitting** (Sobreajuste)

El **sobreajuste** es un fenómeno que ocurre cuando un modelo de Machine Learning aprende los datos de entrenamiento **demasiado bien**, incluyendo el ruido o los detalles irrelevantes de esos datos.

* **Resultado:** El modelo muestra un rendimiento **excelente** en el conjunto de **entrenamiento**, pero un rendimiento **pobre** y resultados poco confiables en el conjunto de **validación** o de prueba.

---

### **2. Optimización y Control del Entrenamiento**

#### **Warm Start**

El parámetro `warm_start=True` en `MLPClassifier` se utiliza para permitir el **entrenamiento incremental** del modelo.

* **Mecanismo:** Cuando se llama al método `fit()` varias veces, el modelo **reutiliza los pesos y sesgos** aprendidos en la llamada anterior en lugar de re-inicializarlos aleatoriamente.
* **Uso en el Código:** Combinado con `max_iter=1`, esto simula un entrenamiento paso a paso, época por época, lo cual es necesario para la implementación manual del **Early Stopping**.

#### **Learning Rate Init** (Tasa de Aprendizaje Inicial)

El `learning_rate_init` es un **hiperparámetro** que define el tamaño del paso que el algoritmo de optimización (como Adam o SGD) da para ajustar los pesos del modelo durante el descenso del gradiente.

* **Descenso del Gradiente:** Es el proceso por el cual el modelo minimiza su función de pérdida.
* **Impacto:**
    * **Tasa Alta:** El modelo converge más rápido, pero puede **saltarse el mínimo** o volverse inestable.
    * **Tasa Baja:** El modelo es más estable, pero tarda mucho más en converger o puede quedarse atascado en un mínimo local.
* **Ajuste:** El valor `0.05` es una **tasa de aprendizaje** específica seleccionada para encontrar un buen equilibrio entre estabilidad y velocidad de convergencia.