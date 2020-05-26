#Cada que el modelo se vuelve a entrenar se obtienen resultados diferentes para la predicción
#ya que el modelo utiliza aleatoriedad para entrenar
#Para que los resultados sean consistentes es NECESARIO ejecutar el entrenamiento solamente una vez
#Ya entrenado el modelo se puede utilizar para predecir alcance en cualquier número de publicaciones

#Librerías necesarias:
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

#Definición de funciones para entrenar y predecir:
def entrenador(arreglo,alcances):
    """Entrena el modelo utilizando un arreglo de publicaciones o un dataframe y sus alcances.
    
    Parameters:
        arreglo (arreglo de numpy, también puede ser un dataframe):
            Arreglo multidimensional con los valores de las métricas para cada publicación.
            Cada publicación está en una fila del arreglo.
            El orden de las métricas debe ser el siguiente [likes,love,angry,wow,haha,sad,shares].
        alcances (arreglo de numpy, tambien puede ser una serie):
            Arreglo unidimensional con los valores de los alcances para cada publicación.
            Cada publicación está en una fila del arreglo.
        
    Returns:
        red (red neuronal de Sklearn):
            Modelo de red neuronal entrenada para predecir los alcances de publicaciones.
            
    """
    logtrain = np.log1p(arreglo)
    logpredi = np.log1p(alcances)
    
    red = MLPRegressor(alpha=0.01, hidden_layer_sizes = (10,), max_iter = 50000, 
                 activation = 'logistic', learning_rate = 'adaptive',solver= 'lbfgs')
    
    red.fit(logtrain,logpredi)
    
    return red


def predictor(arreglo,modelo):
    """Predice los alcances para un arreglo de publicaciones o un dataframe.
    
    Parameters:
        arreglo (arreglo de numpy, también puede ser un dataframe):
            Arreglo multidimensional con los valores de las métricas para cada publicación.
            Cada publicación está en una fila del arreglo.
            El orden de las métricas debe ser el siguiente [likes,love,angry,wow,haha,sad,shares].
            
        modelo (modelo de sklearn):
            El modelo de predicción entrenado previamente
            
    Returns:
        alcances (arreglo de numpy):
            Arreglo con los alcances para cada publicación.
            
    """
    logdata = np.log1p(arreglo)
    predata = modelo.predict(logdata)
    bacdata = np.expm1(predata)
    
    return bacdata

#Carga de los datos utilizados para entrenar
#Siempre se tiene que entrenar con estos datos, SIEMPRE.
data = pd.read_csv("./posts.csv")

#Procesado y filtrado de los datos para entrenar:
metricas = ['likes', 'love', 'angry', 'wow', 'haha', 'sad', 'shares']
data["reacciones"] = data[metricas].sum(1)
fdata=data[(data["scope"]!=0)&(data["reacciones"]>10)&(data["reacciones"]<=data["scope"])]
mdata = fdata[metricas + ["scope"]]

#Preparación de los datos para el entrenamiento:
arr_metricas = mdata[metricas].values
arr_alcances = mdata["scope"].values

#Entrenamiento con los datos de las publicaciones:
red = entrenador(arr_metricas,arr_alcances)

datap = pd.read_csv("./PostsConIdUnicamente.tsv",sep='\t')

datafp = datap.fillna(0).set_index("id")[['likes', 'love', 'angry', 'wow', 'haha', 'sad', 'shares']]

datapv = datafp.values


#Ejemplo de como se obtendría la predicción.
#Esta función se puede ejecutar las veces que se requieran
#En este ejemplo se usan los mismos datos que se usaron para entrenar
#En la practica se puede usar cualquier arreglo de publicaciones
#Siempre y cuando tenga el mismo formato y orden en las métricas
predicciones = predictor(datapv,red)
prediccion = pd.DataFrame(predicciones,columns=["Prediccion"],index=datafp.index).reset_index().rename(columns={"index":"id"})


#Se exportan los datos y la predicción para comparación
#Cuando se realice la predicción con datos a predecir no se tendrá esta comparación
prediccion.to_csv("./prediccion.csv",index=False)
print("listo")