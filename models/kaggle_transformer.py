import tensorflow as tf
import numpy as np
from models.specnorm import SpectralNormalization
from keras.layers import Attention

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

@tf.function
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

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

@tf.function  
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # dk = k.get_shape().as_list()[-1]
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads,spec_norm=False):
        super(MultiHeadAttention, self).__init__()
        self.kernel_init = tf.keras.initializers.Orthogonal()
        self.num_heads = num_heads
        self.d_model = d_model
        if spec_norm:
            self.spec_norm = SpectralNormalization
        else:
            self.spec_norm = lambda x: x

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = self.spec_norm(tf.keras.layers.Dense(d_model,kernel_initializer=self.kernel_init))
        self.wk = self.spec_norm(tf.keras.layers.Dense(d_model,kernel_initializer=self.kernel_init))
        self.wv = self.spec_norm(tf.keras.layers.Dense(d_model,kernel_initializer=self.kernel_init))

        self.dense = self.spec_norm(tf.keras.layers.Dense(d_model,kernel_initializer=self.kernel_init))
        # self.attn_layer = Attention(use_scale=True)
        
    @tf.function   
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention= scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention = self.attn_layer([q,v,k],[mask,None])
        # attention_weights = None

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
    
def point_wise_feed_forward_network(d_model, dff,spec_norm=False):
    if spec_norm:
        wrapper = SpectralNormalization
    else:
        wrapper = lambda x: x
    kernel_init = tf.keras.initializers.Orthogonal()
    return tf.keras.Sequential([
      wrapper(tf.keras.layers.Dense(dff, activation='relu',kernel_initializer=kernel_init)),  # (batch_size, seq_len, dff)
      wrapper(tf.keras.layers.Dense(d_model,kernel_initializer=kernel_init))  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1,spec_norm=False):
        super(EncoderLayer, self).__init__()
        self.spec_norm = spec_norm
        self.mha = MultiHeadAttention(d_model, num_heads,self.spec_norm)
        self.ffn = point_wise_feed_forward_network(d_model, dff,self.spec_norm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
         
    def call(self, x, training, mask):

        attn_output = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1= self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3
    
from keras.layers import TimeDistributed
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1,spec_norm=False):
        super(Encoder, self).__init__()
        
        if spec_norm:
            self.spec_norm = SpectralNormalization
        else:
            self.spec_norm = lambda x: x


        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = TimeDistributed(self.spec_norm(tf.keras.layers.Dense(d_model,kernel_initializer = tf.keras.initializers.Orthogonal())))
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate,spec_norm) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
    def build(self, input_shape):
        self.n_timesteps = input_shape.as_list()[1]
        self.pos_encoding = positional_encoding(self.n_timesteps, self.d_model)
           
    def transform(self,x):
        seq_len = x.get_shape().as_list()[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]
        return x

    def call(self, x, training, mask):
        
        # seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x = x * np.sqrt(self.d_model)
        # x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transform(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = TimeDistributed(tf.keras.layers.Dense(d_model,kernel_initializer = tf.keras.initializers.Orthogonal()))

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def build(self, input_shape):
        self.n_timesteps = input_shape.as_list()[1]
        self.pos_encoding = positional_encoding(self.n_timesteps, self.d_model)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            # attention_weights[f'decoder_layer{i+1}_block1'] = block1
            # attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x
    
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    # return tf.cast(pos_encoding, dtype=tf.float32)
    return pos_encoding

class Transformer(tf.keras.Model):
    def __init__(self,n_target_features, num_layers=4, d_model=128, num_heads=8, dff=512, rate=0.1, name='xformer'):
        super().__init__(name=name)
        
        self.n_target_features = n_target_features
        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate)

        # self.final_layer = TimeDistributed(tf.keras.layers.Dense(self.n_target_features))
        self.final_layer = tf.keras.layers.Dense(self.n_target_features)

    def call(self, inp, tar, training):

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        
        enc_output = self.tokenizer(inp, training, None)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, None)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output