# model training is taking...time (even on EC2)
# to free up my computer and help it run uninteruppted, want to port from jupyter notebook into a .py file
# then we can tmux, run the script, sit back and not worry about disconnecting and losing progress
# note - data preprocessing already done so will just run the model initiation and training
# will output the weights for future use
import logging
import time
import os
import pickle


import numpy as np
import matplotlib.pyplot as plt

#import tensorflow_datasets as tfds
import tensorflow as tf

# Import tf_text to load the ops used by the tokenizer saved model
#import tensorflow_text  # pylint: disable=unused-import
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib as plt


from sklearn.model_selection import train_test_split


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model,  Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dropout, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding, Concatenate
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam, Adagrad
from keras.losses import sparse_categorical_crossentropy
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
import random

# pulling data and setting baseline model parameters
out_pkl = '/home/ec2-user/PichiaCLM/Model_PichiaCLM/Training/AllData/Pichia_All_2Target.pkl'
parameter_settings_csv = "/home/ec2-user/PichiaCLM/Model_PichiaCLM/Training/BO_forHyperParameter/Arch1/Round3.csv"

with open(out_pkl, "rb") as fp:
    Data_AllOrg = pickle.load(fp)

AA_tr = Data_AllOrg['AA_tr']
Cds_tr = Data_AllOrg['Cds_tr']
AA_ts = Data_AllOrg['AA_ts']
Cds_ts = Data_AllOrg['Cds_ts']

Settings = pd.read_csv(parameter_settings_csv).iloc[:, 1:]
Setting_no = 1

Max_length = 1000
learning_rate = 0.001
batch_size = 32
epochs = 50
aa_vocab_size = 25
dna_vocab_size = 67

hidden_size_enc = int(Settings['Enc hidden size'][Setting_no])
hidden_size_enc_aa = int(Settings['Enc hidden size'][Setting_no])
embedding_size_enc = int(Settings['Enc Embedding size'][Setting_no])
embedding_size_dec = int(Settings['Dec Embedding size'][Setting_no])
Dense_layer_size = int(Settings['Dense Layer size'][Setting_no])
Dense_layer_size_aa = int(Settings['Dense Layer size aa'][Setting_no])

drop_rate = Settings['Drop rate'][Setting_no]
drop_rate_aa = Settings['Drop rate aa'][Setting_no]

# setting network parameters (2 outputs)
input_sequence = Input(shape=(Max_length-1,))
encod_emb = Embedding(input_dim= aa_vocab_size, output_dim = embedding_size_enc,trainable=True, mask_zero = True)
embedding = encod_emb(input_sequence)

encoder = Bidirectional(GRU(hidden_size_enc, return_sequences=True, return_state = True),
                        merge_mode="concat", weights=None)

encoder_sequence, encoder_final_f, encoder_final_b  = encoder(embedding)

encoder_final = Concatenate(axis=-1)([encoder_final_f, encoder_final_b])



decoder_inputs = Input(shape=(Max_length -1, ))
decoder_inputs_aa = Input(shape=(Max_length-1, ))

dex=  Embedding(input_dim = dna_vocab_size, output_dim = embedding_size_dec, trainable=True, mask_zero = True)


final_dex= dex(decoder_inputs)
final_dex_aa =  encod_emb(decoder_inputs_aa)


decoder = GRU(2*hidden_size_enc, return_sequences = True, return_state = True)
decoder_aa =  GRU(2*hidden_size_enc_aa, return_sequences = True, return_state = True)

decoder_sequence, decoder_final = decoder(final_dex, initial_state=encoder_final)
decoder_sequence_aa, decoder_final_aa = decoder_aa(final_dex_aa, initial_state=encoder_final)


attn_layer = Attention()
attn_out = attn_layer([decoder_sequence, encoder_sequence])
attn_layer_aa = Attention()
attn_out_aa = attn_layer_aa([decoder_sequence_aa, encoder_sequence])

decoder_concat_input = Concatenate(axis=-1)([decoder_sequence, attn_out]) #decoder_sequence,
decoder_concat_input_aa = Concatenate(axis=-1)([decoder_sequence_aa, attn_out_aa]) #decoder_sequence,


Intermediate_layer = TimeDistributed(Dense(Dense_layer_size, activation='tanh'))
Intermediate_layer_aa= TimeDistributed(Dense(Dense_layer_size_aa, activation='tanh'))

Intemediate_output = Intermediate_layer(decoder_concat_input) #decoder_concat_input
Intemediate_output_aa = Intermediate_layer_aa(decoder_concat_input_aa) #decoder_concat_input


dropout_layer = Dropout(drop_rate)
dropout_output = dropout_layer(Intemediate_output)

dropout_layer_aa = Dropout(drop_rate_aa)
dropout_output_aa = dropout_layer_aa(Intemediate_output_aa)

dense_layer = TimeDistributed(Dense(dna_vocab_size, activation='softmax'))
logits = dense_layer(dropout_output)

dense_layer_aa = TimeDistributed(Dense(aa_vocab_size, activation='softmax'))
logits_aa = dense_layer_aa(dropout_output_aa)

enc_dec_model = Model([input_sequence, decoder_inputs, decoder_inputs_aa], [logits, logits_aa])

enc_dec_model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(learning_rate = learning_rate),
              metrics=['accuracy','accuracy'])
enc_dec_model.summary()

# model training
checkpoint_path = "/home/ec2-user/PichiaCLM/Model_PichiaCLM/2Target_AllData/Arch1_weights_11Mar2026.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience = 3,
    verbose=0, mode="auto", baseline=None, restore_best_weights=False)
## Train the model
model_results = enc_dec_model.fit([AA_tr[:,1:Max_length], Cds_tr[:,0:Max_length-1], AA_tr[:,0:Max_length-1]],
                                  [Cds_tr[:,1:Max_length],  AA_tr[:,1:Max_length]],
                                  batch_size= batch_size,
                                  epochs= epochs,
                                  validation_split=0.2, callbacks=[cp_callback, early_stop])