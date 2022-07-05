import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
import random
import numpy as np
from keras.layers import LSTM, TimeDistributed, Dense, Activation, RepeatVector
from numpy import argmax


morse_dict = { 'a':'.-', 'b':'-...',
              'c':'-.-.', 'd':'-..', 'e':'.',
              'f':'..-.', 'g':'--.', 'h':'....',
              'i':'..', 'j':'.---', 'k':'-.-',
              'l':'.-..', 'm':'--', 'n':'-.',
              'o':'---', 'p':'.--.', 'q':'--.-',
              'r':'.-.', 's':'...', 't':'-',
              'u':'..-', 'v':'...-', 'w':'.--',
              'x':'-..-', 'y':'-.--', 'z':'--..'}

morse_alphabet = ".-| "
alphabet = "abcdefghijklmnopqrstuvwxyz "


def morse_encode(letterString):
    encoded_morse = []
    for letter in list(letterString):
        if letter in morse_dict:
            for val, morse in morse_dict.items():
                if val == letter:
                    encoded_morse.append(morse)

    return "|".join(encoded_morse)


word_length = 7
max_morse_length = 4 * word_length + (word_length - 1)
max_word_length = word_length


def generate_data(n):
    with open('englischWords.txt', 'r') as f:
        read_words = f.read().lower().split('\n')
        words = [word for word in read_words if len(word) == n]
        random.shuffle(words)
        out_w = lambda x: ' ' * (max_word_length - len(x)) + x
        output_list = [out_w(word) for word in words]
        in_w = lambda x: morse_encode(x) + ' ' * (max_morse_length- len(morse_encode(x)))
        input_list = [in_w(word) for word in words]

        return output_list, input_list


output_list, input_list = generate_data(word_length)

print(len(output_list))
print(len(input_list))
print(output_list[:10])
print(input_list[:10])


def alphabet_integer_hashing():
    encoded_alphabet = [char for char in alphabet]
    hashed_alphabet_dict = {}
    cnt = 0
    for item in encoded_alphabet:
        hashed_alphabet_dict[item] = cnt
        cnt = cnt+1

    encoded_morse = [char for char in morse_alphabet]
    hashed_morse_dict = {}
    cnt2 = 0
    for item in encoded_morse:
        hashed_morse_dict[item] = cnt2
        cnt2 = cnt2+1

    return hashed_alphabet_dict, hashed_morse_dict


hashed_alphabet, hashed_morse = alphabet_integer_hashing()
print(hashed_alphabet)
print(hashed_morse)


def onehot_encoding(encoder,data,input_alphabet):
    integer_encoded = [encoder[char] for char in data]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(input_alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return onehot_encoded


all_encoded_outputs = []
all_encoded_inputs = []
for item in output_list:
    all_encoded_outputs.append(onehot_encoding(hashed_alphabet, item, alphabet))
for item in input_list:
    all_encoded_inputs.append(onehot_encoding(hashed_morse, item, morse_alphabet))

print(all_encoded_inputs[:5])
print(all_encoded_outputs[:5])

data_train, data_test, label_train, label_test = train_test_split(all_encoded_inputs, all_encoded_outputs, test_size=0.20, random_state=0)


model = Sequential()

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

latent_dim = 256
model.add(layers.InputLayer((max_morse_length, len(morse_alphabet))))
model.add(layers.LSTM(latent_dim))
model.add(layers.RepeatVector(max_word_length))
model.add(layers.LSTM(latent_dim, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(alphabet))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# set Parameters
batch_size = 8
num_epochs = 30

model.fit(
        data_train,
        label_train,
        batch_size,
        verbose=1,
        epochs=num_epochs,
        validation_data=(data_test, label_test),
        callbacks=[tensorboard_callback]
)

print(model.metrics_names, model.evaluate(data_test,label_test))
model.summary()
model.save('saved_model/project2')


def invert_encoding(data,encoder):
    int_to_char = dict((i, c) for i, c in enumerate(encoder))
    result = ""
    for i in range(len(data))[:10]:
        result += " "
        for a in range(len(data[i])):
            result += int_to_char[argmax(data[i][a])]

    result = result.split()
    return result


def doing_test_pred(input):
    print("Training Input <-> Training Output")
    for i in range(10):
        print(invert_encoding(data_train, morse_alphabet)[i], "<->", invert_encoding(label_train, alphabet)[i])

    test_list = []
    for item in input:
        enc_item = morse_encode(item) + ' ' * (max_morse_length - len(morse_encode(item)))
        test_list.append(onehot_encoding(hashed_morse, enc_item, morse_alphabet))

    pred_test = model.predict(test_list)

    results = []
    for item in pred_test:
        result = []
        for inner_item in item:
            result.append(np.argmax(inner_item))
        results.append(result)

    final_pred = ""
    for item in results:
        final_pred += " "
        for inner_item in item:
            final_pred += str(next((k for k in hashed_alphabet if hashed_alphabet[k] == inner_item), None))

    print("your input words were: ",input)
    print("This words got converted into this Morse-Code: ",invert_encoding(test_list,morse_alphabet))
    print("The Neural-Net did the following predictions for this Morse-Code: ", final_pred.split())


doing_test_pred(["testtes","welthey","wortabc","abcdefg"])

