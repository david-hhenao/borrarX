import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import requests

# Cargar el texto de 'reviews_booking_limpio'
url = 'https://raw.githubusercontent.com/Izainea/nlp_ean/refs/heads/main/Datos/Datos%20Crudos/reviews_booking_limpio.csv'
DF=pd.read_csv(url)

# Tokenizar el texto
tokenizer = Tokenizer()
tokenizer.fit_on_texts(DF['Comentarios'])
total_words = len(tokenizer.word_index) + 1
max_sequence_len = 100

# Prueba 2
for x,y in enumerate(range(5,13)):
    print(x,y)

# sequences

sequences=tokenizer.texts_to_sequences(DF['Comentarios'])
padded = pad_sequences(sequences, maxlen=max_sequence_len, padding='post', truncating='post')

padded.shape

# Prueba 1
for i in range(10):
    j = i *2.65
    print(j,i)