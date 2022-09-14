import numpy as np
import tensorflow as tf
import spacy
import os
import json
import codecs
import pickle
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
nlp = spacy.load('es_dep_news_trf')


intentos = json.loads(codecs.open(
    'categorias.json', 'r', encoding='utf-8').read())

documentos = []
palabras = set()
categorias = []
ignorar = ["?", "!", ",", ".", ":"]

for intento in intentos["categorias"]:
    categoria = intento["categoria"]
    lematizacion = set()
    categorias.append(categoria)
    for sentencia in intento["sentencias"]:

        tokens = nlp(sentencia)
        print(sentencia)
        lemmas = [tok.lemma_.lower()
                  for tok in tokens if tok.text not in ignorar]
        lematizacion = lematizacion.union(set(lemmas))
        palabras = palabras.union(lematizacion)
    documentos.append((sorted(lematizacion), categoria))

categorias = sorted(categorias)
palabras = sorted(palabras)

pickle.dump(palabras, open('palabras.pkl', 'wb'))
pickle.dump(categorias, open('categorias.pkl', 'wb'))

training = []
salida_categorias = [0] * len(categorias)

for documento in documentos:
    entrada_palabras = []
    palabras_sentencias = documento[0]
    for palabra in palabras:
        if palabra in palabras_sentencias:
            entrada_palabras.append(1)
        else:
            entrada_palabras.append(0)
    salida_fila = list(salida_categorias)
    salida_fila[categorias.index(documento[1])] = 1
    training.append([entrada_palabras, salida_fila])

random.shuffle(training)
print("training array: ", training)
training = np.array(training)

train_x = list(training[:, 0])
print("training x: ", train_x)
train_y = list(training[:, 1])
print("training x: ", train_y)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(len(palabras),)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Dense(len(categorias), activation='softmax'))

opt = tf.keras.optimizers.SGD(
    learning_rate=0.04, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y),
                 batch_size=5, epochs=380, verbose=1)
model.save('chatbot_model.h5', hist)
print("Done")
