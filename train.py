from re import X
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from loss import *

index_inputs, index_outputs, index_targets, data_configs=prepro_dataset()

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

kargs={
  "batch_size":2,
  "units":512,
  "sequence_length":25,
  "vocab_size": data_configs['vocab_size'],
  'ffd':2048,
  'rate':0.1,
  "enc_num_layers":2,
  'end_token_idx':2,
  'epochs':2,
  'val_split':0.1,
  'num_layers':2
}



model = Transformer(**kargs)
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.001), metrics=[accuracy])

checkpoint_path = "./weights.h5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

earlystop_cb=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)

history = model.fit([index_inputs, index_outputs], index_targets, batch_size = kargs['batch_size'], epochs = kargs['epochs'], validation_split=kargs['val_split'], callbacks=[earlystop_cb, cp_callback])

