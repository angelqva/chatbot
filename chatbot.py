from ast import While
import random
import json
import pickle
from unittest import result
import numpy as np
import spacy
import os
import codecs
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""
nlp = spacy.load('es_dep_news_trf')

intentos = json.loads(codecs.open(
    'intentos.json', 'r', encoding='utf-8').read())

palabras = pickle.load(open('palabras.pkl', 'rb'))
categorias = pickle.load(open('categorias.pkl', 'rb'))
model: tf.keras.models.Model = tf.keras.models.load_model('chatbot_model.h5')
ignorar = ["?", "!", ",", ".", ":"]


def entradas(sentencia):
    tokens = nlp(sentencia)
    lemmas = set([tok.lemma_.lower()
                  for tok in tokens if tok.text not in ignorar])
    entrada_palabras = [0] * len(palabras)
    for word in lemmas:
        for i, palabra in enumerate(palabras):
            if word == palabra:
                entrada_palabras[i] = 1
    return np.array(entrada_palabras)


def prediccion(sentencia):
    req = entradas(sentencia)
    res = model.predict(np.array([req]))[0]
    error = 0.25
    resultados = [[i, r] for i, r in enumerate(res) if r > error]
    resultados.sort(key=lambda x: x[1], reverse=True)
    listado = []
    for r in resultados:
        listado.append(
            {'intento': categorias[r[0]], 'probabilidad': str(r[1])})
    print("prediccion-listado: ", listado)
    return listado


def respuesta(sentencia):

    categoria = prediccion(sentencia)[0]["intento"]
    datos = intentos["intentos"]
    res = "No tengo muchos temas para hablar, hablemos de otra cosa"
    for i in datos:
        if i["categoria"] == categoria:
            res = random.choice(i["respuestas"])
            break
    return res


print("ChatBot Listo")

while True:
    message = input("")
    print(respuesta(message))
