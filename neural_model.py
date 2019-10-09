import pandas as pd
from keras.engine.saving import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras import layers, models, regularizers
import numpy as np
import logging
import time
from googletrans import Translator

logging.getLogger('tensorflow').disabled = True
translator = Translator()


filepath_dict = {
    'yelp': 'sentiment_labeled_sentences/yelp_labelled.txt',
    'amazon': 'sentiment_labeled_sentences/amazon_cells_labelled.txt',
    'imdb': 'sentiment_labeled_sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)
# print(df.iloc[0])

# df_yelp = df[df['source'] == 'yelp']
# df_amazon = df[df['source']]

sentences = df['sentence'].values
y = df['label'].values

# print(sentences[0])
# print(y[0])

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y,
                                                                    test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

# X_train = vectorizer.transform(sentences_train)
# X_test = vectorizer.transform(sentences_test)
#
# in_shape = X_train.shape[1]
#
# # train
# regularizer = regularizers.l2(0.001)
# model = models.Sequential()
# model.add(layers.Dense(32, input_shape=(in_shape, ), activation='relu',
#                        kernel_regularizer=regularizer))
# model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizer))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# epochs = 35
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=16)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")


sentence_new = input("Frase: ")
sentence_new = translator.translate(sentence_new, dest="en").text
print("Traduccion:", sentence_new)
sentence_new_vec = vectorizer.transform([sentence_new])
predictions = model.predict(sentence_new_vec[:1])
# print(predictions)

threshold = 0.5
if predictions[0] > 0.5:
    print("Buena en un", round(predictions[0][0] * 100, 1), "%")
else:
    print("Mala en un", round((1 - predictions[0][0]) * 100, 1), "%")
