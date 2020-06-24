
# Carga de librerías necesarias

import pandas as pd
import numpy as np
import scipy as sci
from sklearn.neural_network import MLPRegressor



# Definición de funciones necesarias para entrenar el modelo de alcance


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


# Carga y procesado de datos para entrenar la red para predecir alcance

data = pd.read_csv("./inputs/posts.csv")

metricas = ['likes', 'love', 'angry', 'wow', 'haha', 'sad', 'shares']

data["reacciones"] = data[metricas].sum(1)

fdata=data[(data["scope"]!=0)&(data["reacciones"]>10)&(data["reacciones"]<=data["scope"])]

mdata = fdata[metricas + ["scope"]]

arr_metricas = mdata[metricas].values

arr_alcances = mdata["scope"].values


# Entrenado de la red predictora de alcance

red = entrenador(arr_metricas,arr_alcances)

#Parámetros para la distribución objetivo
escala = 20
a = 0.7

# Carga de datos de publicaciones y temas

datap = pd.read_csv("./inputs/Posts_CDMX.tsv",sep='\t')
datat = pd.read_csv("./inputs/Temas_CDMX.tsv",sep='\t')


# Proceso de datos de temas y publicaciones

datat.columns=[cadena + "_T" for cadena in datat.columns]

datat.rename(columns={"id_T":"idTema"},inplace=True)


# Mezcla de los datos de publicaciones y de temas

datamix = pd.merge(datap,datat,how="left",on="idTema")


# Llenado de datos vacíos a cero

datafp = datamix[metricas].fillna(0)


# Predicción de alcances para todas las publicaciones

datapv = datafp.values

predicciones = predictor(datapv,red)

# Conversión a DataFrame e incorporación al frame general

prediccion = pd.DataFrame(predicciones,columns=["Alcance_estimado"],index=datafp.index)
prediccion["Alcance_estimado"] = prediccion["Alcance_estimado"].apply(lambda x: max(x,10))

datamix["Alcance_estimado"] = prediccion

datamix["reacciones"] = datamix[metricas].sum(1)


# Definición de Estado a calificar y parámetros para el modelo, alcmax es el alcance máximo posible, es decir la población total a alcanzar
# pubmax es el número máximo de publicaciones para fijar el tope de la calificación, usualmente 100 0 200 funciona bien, dependiendo de la ciudad.

#Para CDMX:
estado = ["Ciudad de México"]
alcmax = 20000000

#Para Tamaulipas:
#estado = ["Tamaulipas"]
#alcmax = 5000000

#Para Sinaloa:
#estado = ["Sinaloa"]
#alcmax = 4000000

#Para Baja California Sur:
#estado = ["Baja California Sur"]
#alcmax = 1500000

#Para Baja California:
#estado = ["Baja California"]
#alcmax = 4000000

#Para Sonora:
#estado = ["Sonora"]
#alcmax = 3500000

#Para NACIONAL:
#estado = ["NACIONAL"]
#alcmax = 100000000

#Para Querétaro:
#estado = ["Querétaro"]
#alcmax = 3000000


# Filtrado de las publicaciones a aquellas que están en el estado y selección de columnas importantes

data_filt=datamix[(datamix["estado_T"].isin(estado))][metricas+["score_T","Alcance_estimado","reacciones","idTema","nombre_T"]]


# Sustitución de los alcances mayores al alcance máximo por el alcance máximo

data_filt["Alcance_est_top"] = data_filt["Alcance_estimado"].apply(lambda x: min(x,alcmax))


# Definición de la función para calcular el alcance extra al alcance de la publicación con mayor alcance para cada tema

def alcance_extra(serie,atope):
    """Calcula el alcance extra para un conjunto de publicaciones correspondientes al mismo tema.
    
    Parameters:
        serie (serie de pandas):
            Serie que contiene los alcances estimados para cada publicación.
        atope (real):
            Valor del alcance tope en el estado o ciudad, usualmente este valor es la población total del estado.
        
    Returns:
        red (real):
            Valor del alcance extra al alcance de la publicación con mayor alcance en el conjunto de publicaciones.
            
    """
    nserie = serie.apply(lambda x: min(x,atope))
    serie_s = nserie.sort_values(ascending=False)
    index = serie_s.index
    a_max  = min(atope,serie_s.max())
    r = (atope - a_max)/atope
    rango = pd.Series(pd.RangeIndex(0,len(serie_s)),index=index)
    mults = np.power(r,rango)
    return (serie_s*mults).sum()-a_max


# Agrupación de publicaciones por tema

grupos = data_filt.groupby(["idTema","nombre_T"])


# Cálculo del alcance extra para cada tema

por_tema = grupos.apply(lambda x: alcance_extra(x["Alcance_est_top"],alcmax)).to_frame("Alcance_extra")


# Cálculos de alcances máximos, suma de alcances y número de publicaciones para cada tema

por_tema["Alcance_max_top"] = grupos["Alcance_est_top"].max()
por_tema["Alcance_suma"] = por_tema["Alcance_max_top"] + por_tema["Alcance_extra"]
por_tema["Publicaciones"] = grupos["Alcance_estimado"].size()
por_tema["reacciones"] = grupos["reacciones"].sum()


# Construcción de la distribución objetivo

x = np.linspace(0,100, 99)
y = sci.stats.gamma.cdf(x,a=a,scale=escala)


# Determinación del límite de la distribución

lim100 = sci.stats.gamma.cdf(100,a=a,scale=escala)


# Obtención de datos para estimar la distribución empírica

variable = "Alcance_suma"
serie_data = por_tema[variable].sort_values()


#Determinación de la distribución empírica

percentiles = np.percentile(serie_data,range(1,100))
percentiles = np.append(percentiles,[serie_data.max()])
rang1 = (np.array(range(1,100))/100)
rang1 = np.append(rang1,[lim100])


# Interpolación de los datos a la distribución empírica

interp1 = np.interp(serie_data,percentiles,rang1)


# Proyección a la distribución objetivo

interp2 = sci.stats.gamma.ppf(interp1,a=a,scale=escala)


#Construcción de la serie con las calificaciones

serie_calif = pd.Series(interp2,index=serie_data.index)


# Cálculo de las calificaciones por alcance, por temas y total

por_tema["Calificacion"] = serie_calif


# Definición del frame de salida

salida = por_tema[["reacciones","Alcance_max_top","Alcance_extra","Alcance_suma","Publicaciones","Calificacion"]]


# Ordenando de mayor a menor por calificación y alcance total

lista_final = salida.sort_values(["Calificacion","Alcance_suma"],ascending=False)


# Exportando el archivo de salida

lista_final.to_csv("./outputs/salida.csv")


