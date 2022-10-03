import numpy as np
import pickle
import re

import tensorflow as tf
import keras.api._v2.keras as keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Flatten , Embedding, Input, LSTM, Dropout
from keras.models import Model
from keras.initializers import Constant

with open('Saved/vocab_dict.pickle', 'rb') as f:
    vocab_dict = pickle.load(f)

# load the weights
w_embeddings = np.load('Saved/embedding.npz', allow_pickle=True)
w_encoder_lstm = np.load('Saved/encoder_lstm.npz', allow_pickle=True)
w_decoder_lstm = np.load('Saved/decoder_lstm.npz', allow_pickle=True)
w_dense = np.load('Saved/dense.npz', allow_pickle=True)


### PREREQUISITES

maxi = 23
vocab = list(vocab_dict.keys())
vocab_size = len(vocab)
embed_dim = 300

embed_matrix=np.zeros(shape=(vocab_size,embed_dim))
for i,word in enumerate(vocab):
  embed_vector=vocab_dict.get(word)
  if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
    embed_matrix[i]=embed_vector

def clean_text(text):
  text = text.lower()
  text = re.sub(r"\'m", " am", text)
  text = re.sub(r"\'s", " is", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r" 'bout", " about", text)
  text = re.sub(r"gonna", "going to", text)
  text = re.sub(r"gotta", "got to", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can't", "can not", text)
  text = re.sub(r"n't", " not", text)
  text = re.sub("\d+","",text) # remove numbers
  text = re.sub(r"[^\w\s]", "", text) # remove punctuations
  return text

  # input of encoder and decoder
enc_inp = Input(shape=(maxi,))
dec_inp = Input(shape=(maxi,))

# this layer embeds the english word in vector form
embed = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxi, embeddings_initializer=Constant(embed_matrix))
enc_embed = embed(enc_inp)

# LSTM takes all the words at the same time but processes it one by one. The last _,_,_ is related to its previous state. So only final o/p is enough.
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

# embedding is done in decoding and the LSTM return a series of sequence
dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(vocab_size, activation='softmax')
dense_op = dense(dec_op)

load_model = Model([enc_inp, dec_inp], dense_op)
load_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['acc'],optimizer='adam')

# set the weights of the model

load_model.layers[2].set_weights(w_embeddings['arr_0'])
load_model.layers[3].set_weights(w_encoder_lstm['arr_0'])
load_model.layers[4].set_weights(w_decoder_lstm['arr_0'])
load_model.layers[5].set_weights(w_dense['arr_0'])

enc_model = Model([enc_inp], enc_states)

decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = dec_lstm(dec_embed, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
dec_model = Model([dec_inp]+ decoder_states_inputs, [decoder_outputs]+ decoder_states)

def predict(text):
  text = clean_text(text)                                    # ---------> 'hi how are you'
  text = text.split(' ')                                     # ---------> ['hi', 'how', 'are', 'you']
  temp = []
  for each in text:
    if each in vocab:
      temp.append(vocab.index(each))                         # ---------> vocab = list(vocab_dict.keys())
    else:
      temp.append(vocab.index('<OUT>'))
  text = [temp]                                              # ---------> [[32, 45, 65 ,77]]
  text = pad_sequences(text, padding='post', maxlen=maxi)    # ---------> [[32, 45, 65 ,77, 0, 0,0,.....,0]]
  enc_model = Model([enc_inp], enc_states)
  enc_pred = enc_model.predict(text)                         # ---------> return h,c after running LSTM. Shape: (400) each

  empty_target_seq = np.zeros((1, 1))                        # ---------> [[0]] 2D list
  empty_target_seq[0, 0] = vocab.index('<START>')            # ---------> index of 'start' so that model know it's time to predict 
  
  stop = False
  decoded_translation = ''

  while not stop:
    dec_outputs , h, c= dec_model.predict([empty_target_seq] + enc_pred )       # ---------> takes h,c with 'start' sequence as input
    decoder_concat_input = dense(dec_outputs)                                   # ---------> shape is of vocab with softmax probab (1,1,13015)
    sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])              # ---------> reshapes into (13015,) and argmax

    sampled_word = vocab[sampled_word_index] + ' '
    if sampled_word != '<END> ':
      decoded_translation += sampled_word
    if sampled_word == '<END> ' or len(decoded_translation.split()) > maxi:
      stop = True

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = sampled_word_index
    enc_pred = [h,c]

  return decoded_translation


# print("##########################################")
# print("#          Welcome to Chatbot            #")
# print("##########################################")

# while True:
#   text = input("You : ")
#   if text.lower() == 'quit':
#     break
#   answer = predict(text)
#   print(f'Bot : {answer}')
#   print("==============================================")