# Spam Email Classifier

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Graphics-4c8cbf?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![KaggleHub](https://img.shields.io/badge/KaggleHub-Dataset%20Download-20beff?logo=kaggle&logoColor=white)](https://github.com/KaggleHub)

Este proyecto implementa un sistema de clasificación de correos electrónicos para identificar mensajes de spam utilizando técnicas de Machine Learning. El análisis se realiza sobre el conjunto de datos "Spambase", ampliamente utilizado en la investigación de filtrado de spam. El objetivo principal es desarrollar un clasificador robusto, reproducible y fácil de adaptar a otros conjuntos de datos de spam.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Uso](#uso)
- [Resultados](#resultados)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Descripción

El proyecto utiliza Python y Jupyter Notebook para cargar, explorar, visualizar y modelar el dataset de correos electrónicos. Se emplean técnicas de preprocesamiento y dos clasificadores principales: Random Forest y XGBoost para la detección de spam. El flujo de trabajo incluye análisis exploratorio de datos, preparación de datos, entrenamiento de modelos y evaluación de resultados.

## Características

- Descarga automática del dataset “Spambase” desde KaggleHub.
- Limpieza y visualización exploratoria de datos.
- Entrenamiento y evaluación de modelos Random Forest y XGBoost.
- Reporte detallado de métricas de rendimiento como accuracy, matriz de confusión y clasificación.
- Código modular y fácil de extender a nuevos experimentos o modelos.

## Estructura del proyecto

```
├── spam_email_clas.ipynb    # Notebook principal con todo el flujo de trabajo
└── README.md                # Este archivo
```

## Requisitos

- Python 3.7+
- Librerías de Python:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - kagglehub
    - xgboost

Puedes instalar los requisitos con:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub xgboost
```

## Instalación y ejecución

1. Clona este repositorio:
    ```bash
    git clone https://github.com/0xfabrica/spam_email.git
    cd spam_email
    ```

2. Instala los requisitos:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn kagglehub xgboost
    ```

3. Ejecuta el notebook:
    - Abre `spam_email_clas.ipynb` en Jupyter Notebook, JupyterLab o Google Colab.
    - Sigue las celdas en orden para reproducir el flujo completo.

## Uso

- El notebook está preparado para descargar automáticamente el dataset, visualizarlo y ajustar los modelos.
- Puedes modificar los parámetros de los clasificadores o probar otros modelos fácilmente.
- Los resultados de cada etapa se muestran en las celdas de salida del notebook.

## Resultados

Se evaluaron dos modelos principales: Random Forest y XGBoost. A continuación, se presenta una tabla resumen con sus métricas de desempeño:

| Modelo         | Precisión | Recall | F1-score | Accuracy | Score Entrenamiento |
|----------------|-----------|--------|----------|----------|---------------------|
| **Random Forest** | 0.95/0.96 | 0.97/0.93 | 0.96/0.94 | 0.95    | 0.99                |
| **XGBoost**       | 0.96/0.96 | 0.97/0.94 | 0.96/0.95 | 0.96    | 0.95                |

- **Random Forest**: El modelo presenta un excelente rendimiento en las métricas de validación, pero muestra signos de overfitting, como se observa en el score de entrenamiento (0.998). Esto indica que el modelo aprende demasiado bien los datos de entrenamiento, perdiendo cierta capacidad de generalización.
- **XGBoost**: Demuestra ser un modelo más preciso y eficaz, alcanzando una mejor precisión en validación y un mejor equilibrio entre precisión y recall. Además, muestra mayor robustez y menor tendencia al overfitting, siendo preferible para entornos productivos o escenarios donde la generalización es crítica.

<details>
<summary>Métricas completas</summary>

### Random Forest

```
              precision    recall  f1-score   support

         0.0       0.95      0.97      0.96       804
         1.0       0.96      0.93      0.94       577

    accuracy                           0.95      1381
   macro avg       0.95      0.95      0.95      1381
weighted avg       0.95      0.95      0.95      1381

Score de entrenamiento: 0.9981366459627329
```

### XGBoost

```
Precisión: 0.9587255611875453

              precision    recall  f1-score   support

         0.0       0.96      0.97      0.96       804
         1.0       0.96      0.94      0.95       577

    accuracy                           0.96      1381
   macro avg       0.96      0.96      0.96      1381
weighted avg       0.96      0.96      0.96      1381
```
</details>

## Contribuciones

Las contribuciones son bienvenidas. Si deseas proponer mejoras, corregir errores o añadir nuevas funcionalidades:

1. Haz un fork del repositorio.
2. Crea una rama feature/tu-mejora.
3. Abre un Pull Request con una descripción clara de los cambios realizados.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

---

**Contacto:**  
Desarrollado por [0xfabrica](https://github.com/0xfabrica)
