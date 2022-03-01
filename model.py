from re import X
import tensorflow as tf
import numpy as np
from utils import *
from loss import *



def scaled_dot_product_attention(q,k,v,mask):
  #q,k,v의 형태는 (batch, num_head, -1, out_dim) 형태이다.

  # key를 transpose하여 연산해야하므로 transpose_b를 true로 해준다.
  matmul = tf.matmul(q,k, transpose_b = True)

  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  # print(matmul/tf.math.sqrt(dk).numpy())
  scaled_attention = matmul/tf.math.sqrt(dk)
  # print("mask shape", mask.shape)
  print("scaled attention shape", scaled_attention.shape)
  if mask is not None:
    scaled_attention += (mask * -1e9)
    print("scaled attention shape", scaled_attention.shape)
  print("valud shap", v.shape)
  attention_weight = tf.nn.softmax(scaled_attention, axis=-1)
  # print(scaled.numpy())
  output = tf.matmul(attention_weight, v)

  return output, attention_weight



class MultiheadAttention(tf.keras.layers.Layer):
  def __init__(self, **kargs):
    super(MultiheadAttention, self).__init__()

    self.num_heads = 8
    self.d_model = kargs['units']

    assert self.d_model % self.num_heads == 0

    self.depth = self.d_model // self.num_heads

    self.l1 = tf.keras.layers.Dense(kargs['units'])
    self.l2 = tf.keras.layers.Dense(kargs['units'])
    self.l3 = tf.keras.layers.Dense(kargs['units'])

    self.l4 = tf.keras.layers.Dense(kargs['units'])

  def split_heads(self,x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v,k,q,mask):
    batch_size = tf.shape(q)[0]

    q = self.l1(q)
    k = self.l2(k)
    v = self.l3(v)

    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)
    print("input query shape",q.shape)

    attention, attention_weights = scaled_dot_product_attention(q,k,v,mask)

    concat_attention = tf.reshape(attention, (batch_size, -1, self.d_model))

    concat_attention = self.l4(concat_attention)

    return concat_attention, attention_weights




def feed_forward_network(d_model, ffd):
  return tf.keras.Sequential([
    tf.keras.layers.Dense(ffd, activation='relu'),
    tf.keras.layers.Dense(d_model)
  ])

def get_angles(pos, i, d_model):
  angle = 1/np.power(10000, (2*i)/np.float32(d_model))
  return pos*angle

def positional_encoding(position, d_model):
  angle = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis,:], d_model)

  angle[:, 0::2] = np.sin(angle[:, 0::2])
  angle[:, 0::1] = np.cos(angle[:, 0::1])

  pos_encoding = angle[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, **kargs):
    super(EncoderLayer, self).__init__()

    self.multiheadattention = MultiheadAttention(**kargs)
    self.ffn = feed_forward_network(kargs['units'], kargs['ffd'])
    self.normal1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.normal2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
    self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])

  def call(self, x,mask):
    attn_output,_ = self.multiheadattention(x,x,x,mask)
    attn_output = self.dropout1(attn_output)

    normal = self.normal1(attn_output + x)
    ffn = self.ffn(normal)
    ffn = self.dropout2(ffn)
    output = self.normal2(normal + ffn)

    return output


class Encoder(tf.keras.layers.Layer):
  def __init__(self, **kargs):
    super(Encoder, self).__init__()
    self.d_model =kargs['units']
    self.num_layers= kargs['num_layers']
    self.embedding = tf.keras.layers.Embedding(kargs['vocab_size'], kargs['units'])
    self.pos_encoding = positional_encoding(kargs['sequence_length'], kargs['units'])
    self.dropout = tf.keras.layers.Dropout(kargs['rate'])

    self.enc_layers = [EncoderLayer(**kargs) for _ in range(kargs['enc_num_layers'])]

    


  def call(self, x, mask):
    seq_len = tf.shape(x)[1]
    embedding = self.embedding(x)
    pos_encoding = self.pos_encoding[:,:seq_len,:]
    
    embedding *=tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    input = tf.add(embedding, pos_encoding)

    x=self.dropout(input)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, mask)

    return x


    
def create_look_ahead_mask(size):
  mask = 1-tf.linalg.band_part(tf.ones((size,size)),-1,0)
  return mask

#패딩값을 지워주는 mask이다.
def create_padding_mask(seq):
    #tf equal은 seq의 값이 0(pad index)이랑 같으면 True로 반환
    #여기에 들어오는 seq는 embedding 되기 전의 input이다. (batch, seq_len)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, **kargs):
    super(DecoderLayer, self).__init__()

    self.multiheadattention1= MultiheadAttention(**kargs)
    self.multiheadattention2=MultiheadAttention(**kargs)

    self.ffn = feed_forward_network(kargs['units'], kargs['ffd'])

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
    self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])
    self.dropout3 = tf.keras.layers.Dropout(kargs['rate'])

  def call(self, x, enc_output, look_ahead_mask, padding_mask):

    mha, attention_weights1 = self.multiheadattention1(x,x,x, look_ahead_mask)
    mha = self.dropout1(mha)
    out1 = self.layernorm1(tf.add(x,mha))

    mha1,attention_weights2 = self.multiheadattention2(enc_output, enc_output, out1, padding_mask)
    mha1 = self.dropout2(mha1)
    out2 = self.layernorm2(mha1 + out1)

    ffn_output = self.ffn(out2)
    ffn_output = self.dropout3(ffn_output)
    out3  = self.layernorm3(ffn_output + out2)

    return out3, attention_weights1, attention_weights2

class Decoder(tf.keras.layers.Layer):
  def __init__(self, **kargs):
    super(Decoder, self).__init__()

    self.d_model = kargs['units']
    self.num_layers = kargs['num_layers']
    self.embedding = tf.keras.layers.Embedding(kargs['vocab_size'], self.d_model)
    self.pos_encoding = positional_encoding(kargs['sequence_length'], self.d_model)
    self.dec_layers = [DecoderLayer(**kargs) for _ in range(self.num_layers)]
    self.dropout = tf.keras.layers.Dropout(kargs['rate'])

  def call(self, x, enc_output, look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)
    x*=tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:,:seq_len,:]

    x = self.dropout(x)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1']=block1
      attention_weights[f'decoder_layer{i+1}_block2']=block2

    return x, attention_weights

class Transformer(tf.keras.Model):
  def __init__(self, **kargs):

    super(Transformer, self).__init__()
    self.enc_token_idx = kargs['end_token_idx']
    self.encoder = Encoder(**kargs)
    self.decoder = Decoder(**kargs)

    self.Dense = tf.keras.layers.Dense(kargs['vocab_size'])

  def call(self, x):
    inp, tar = x
    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)

    enc_output = self.encoder(inp, enc_padding_mask)
    dec_output, _ = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)

    out = self.Dense(dec_output)
    return out

  def inference(self, x):
    # encoder에 들어갈 값
    inp = x

    tar = tf.expand_dims([STD_INDEX],0)

    enc_padding_mask, look_ahead_mask, dec_padding_mask= create_masks(inp, tar)

    enc_output = self.encoder(inp, enc_padding_mask)
    
    predict_tokens=[]
    for i in range(kargs['sequence_length']):
      dec_output, _=self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
      output = self.Dense(dec_output)
      output = tf.argmax(output, -1).numpy()
      pred_token = output[0][-1]

      if pred_token == self.enc_token_idx:
        break
      predict_tokens.append(pred_token)
      tar = tf.expand_dims([STD_INDEX]+predict_tokens, 0)
      _, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)

    return predict_tokens
    
      











# if __name__ =="__main__":
  
  # x=tf.random.uniform(shape=(11823,25))
  # # y = tf.random.uniform(shape=(11823,25))
  # enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(x, y)

  # # x=tf.keras.layers.Embedding(data_configs['vocab_size'], kargs['d_model'])(x)
  # # print("enc_padding_mask",enc_padding_mask.shape)
  # # print("look_ahead_mask",look_ahead_mask.shape)
  # # print("dec_padding_mask",dec_padding_mask.shape)
  # # # scaled_dot_product_attention(x,x,x)
  # # # print(create_look_ahead_mask(2))
  # # mattention = MultiheadAttention(**kargs)
  # # # print(mattention.split_heads(x, 1).shape)
  # # mattention.call(x,x,x, enc_padding_mask )
  # # # encoder = EncoderLayer(**kargs)
  # # # print(encoder.call(x))
  # print(data_configs['char2idx'])

