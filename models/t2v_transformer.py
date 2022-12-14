import tensorflow as tf
from keras.layers import Layer
from keras import backend as K

# class T2V(Layer):
    
#     def __init__(self, n_timesteps, sample_features, d_model, **kwargs):
#         super(T2V, self).__init__(**kwargs)
#         self.in_time_features = 1
#         self.linear_kernel = 1
#         self.period_kernel = d_model - sample_features - self.linear_kernel + 1
#         # self.n_timesteps = n_timesteps
#         self.sample_features = sample_features
    
#     def build(self, input_shape):
#         self.W = self.add_weight(name='W',
#                                 shape=(self.in_time_features, self.period_kernel),
#                                 initializer='random_uniform',
#                                 trainable=True)

#         self.P = self.add_weight(name='P',
#                                 shape=(self.in_time_features, self.period_kernel),
#                                 initializer='random_uniform',
#                                 trainable=True)

#         self.w = self.add_weight(name='w',
#                                 shape=(self.in_time_features, 1),
#                                 initializer='random_uniform',
#                                 trainable=True)

#         self.p = self.add_weight(name='p',
#                                 shape=(self.in_time_features, 1),
#                                 initializer='random_uniform',
#                                 trainable=True)
        
#     def call(self, x):
#         # print(self.W.shape, self.P.shape, self.w.shape, self.p.shape)
#         time_vector = tf.expand_dims(x[:,:,-1], axis=-1)
#         # print(time_vector.shape)
#         linear = tf.math.multiply(time_vector, self.w) + self.p
#         sin_period = tf.math.sin(tf.math.multiply(time_vector, self.W) + self.P)
#         output = K.concatenate([x[:,:,:-1], linear, sin_period], -1)
#         # print(output.shape)
#         return output

class T2V(Layer):

  def __init__(self, n_sample_features, n_time_features, n_timesteps, d_model, trainable=True, **kwargs):
    super(T2V, self).__init__(trainable, **kwargs)

    # self.linear_kernel = 1
    self.n_time_features = n_time_features
    self.n_timesteps = n_timesteps
    self.time_kernel = d_model - n_sample_features + n_time_features
    assert(self.time_kernel > 0)
    self.period_activation = tf.math.sin
    # self.linear_layer = tf.keras.layers.Dense(self.linear_kernel)
    # self.period_layer = tf.keras.layers.Dense(self.period_kernel)
    # self.flatten = tf.keras.layers.Flatten()
  
  def build(self, input_shape):

        self.w = self.add_weight(name='weight',
                                shape=(self.n_time_features, self.time_kernel),
                                initializer='random_uniform',
                                trainable=True)

        self.b = self.add_weight(name='bias',
                                shape=(self.n_timesteps, self.time_kernel),
                                initializer='random_uniform',
                                trainable=True)

  def call(self,x):

    x_time = x[:,:,-self.n_time_features:]
    total_time_vec = tf.linalg.matmul(x_time, self.w) + self.b
    linear_time_vec = total_time_vec[:,:,0]
    linear_time_vec = tf.expand_dims(linear_time_vec, axis=-1)
    period_time_vec = self.period_activation(total_time_vec[:,:,1:])

    output = tf.concat([x[:,:,:-self.n_time_features], linear_time_vec, period_time_vec], axis=-1)
    
    return output

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)

    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask=True
        )

    # print('Attention mask shape:', np.stack([np.tri(x.shape[-2], x.shape[-2]) for _ in range(x.shape[0])], axis=0).shape)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
    ])
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.dropout(self.seq(x))])
    x = self.layer_norm(x) 
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ff_layer = FeedForward(d_model, dff, dropout_rate=dropout_rate)

  def call(self, x, training):
    # print('Global self-attention input:', x.shape)
    x = self.self_attention(x, training=training)
    x = self.ff_layer(x, training=training)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, n_time_features,
               d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.n_time_features = n_time_features
    
    self.enc_layers = [
        EncoderLayer(d_model=self.d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
  
  def build(self, input_shape):
        # self.n_timesteps = input_shape.as_list()[1]
        sample_features = input_shape.as_list()[-1]
        n_timesteps = input_shape.as_list()[1]
        self.time_vector = T2V(sample_features, self.n_time_features, n_timesteps, self.d_model, name='t2v_enc')
  
  def call(self, x, training):
    # `x` is single window shape: (batch, seq_len)

    x = self.time_vector(x)  # Shape `(batch_size, seq_len, n_features_in)`.

    # Add dropout.
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training=training)

    return x  # Shape `(batch_size, seq_len, n_features_in)`.

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

  def call(self, x, context, training):
    # print('Causal self-attention input:', x.shape)
    x = self.causal_self_attention(x=x, training=training)
    x = self.cross_attention(x=x, context=context, training=training)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x, training=training)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, n_time_features,
              d_model, num_heads, dff, dropout_rate):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.n_time_features = n_time_features

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=self.d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context, training):
    # print("Decoder call input shapes (x, context):", x.shape, context.shape)
    # `x` is token-IDs shape (batch, target_seq_len)

    # n_timesteps = tf.shape(x)[1]
    sample_features = tf.shape(x)[-1]
    n_timesteps = tf.shape(x)[1]
    self.time_vector = T2V(sample_features, self.n_time_features, n_timesteps, self.d_model, name='t2v_dec')

    x = self.time_vector(x)  # (batch_size, target_seq_len, d_model)
  
    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context, training=training)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x

class T2V_Transformer(tf.keras.Model):
  def __init__(self, num_layers, num_heads, dff, n_time_features,
              target_features, d_model, dropout_rate=0.1, name='t2v_transformer'):
    super().__init__(name=name)
    
    self.target_features = target_features
    # self.encoder_input = tf.keras.layers.Dense(d_model-time_kernel, activation='linear')
    # self.decoder_input = tf.keras.layers.Dense(d_model-time_kernel, activation='linear')
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, n_time_features=n_time_features,
                           num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model, n_time_features=n_time_features,
                           num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_features)

  def call(self, context, x, training):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    # context, x  = inputs

    # context = self.encoder_input(context)
    context = self.encoder(context, training=training)  # (batch_size, context_len, d_model)

    # x = self.decoder_input(x)
    x = self.decoder(x, context, training=training)  # (batch_size, target_len, d_model)
    # Final linear layer output.
    output = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
    
    # try:
    # #   # Drop the keras mask, so it doesn't scale the losses/metrics.
    #     del output._keras_mask
    # except AttributeError:
    #     pass

    # Return the final output and the attention weights.
    # if self.n_steps_out==1:
    #   return tf.squeeze(output, axis=-1)
    # else:
    return output