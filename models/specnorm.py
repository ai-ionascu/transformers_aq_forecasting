from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
# import tensorflow as tf

class SpectralNormalization(layers.Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

        if not hasattr(self.layer, 'kernel'):
            raise ValueError(
                '`SpectralNormalization` must wrap a layer that'
                ' contains a `kernel` for weights')
        if hasattr(self.layer, 'recurrent_kernel'):
            print("YAYYYYYYYYY")

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(
            shape=tuple([1, self.w_shape[-1]]),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            name='sn_u',
            trainable=False,
            dtype=dtypes.float32)

        super(SpectralNormalization, self).build()

    @def_function.function
    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = K.learning_phase()
        
        if training==True:
            # Recompute weights for each forward pass
            self._compute_weights()
        
        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = array_ops.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = array_ops.identity(self.u)
        _v = math_ops.matmul(_u, array_ops.transpose(w_reshaped))
        _v = _v / math_ops.maximum(math_ops.reduce_sum(_v**2)**0.5, eps)
        _u = math_ops.matmul(_v, w_reshaped)
        _u = _u / math_ops.maximum(math_ops.reduce_sum(_u**2)**0.5, eps)

        self.u.assign(_u)
        sigma = math_ops.matmul(math_ops.matmul(_v, w_reshaped), array_ops.transpose(_u))

        self.layer.kernel.assign(self.w / sigma)

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

class SpectralNormalization_RNN(layers.Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization_RNN, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

        if not hasattr(self.layer, 'kernel'):
            raise ValueError(
                '`SpectralNormalization` must wrap a layer that'
                ' contains a `kernel` for weights')
            
        self.w = self.layer.kernel
        self.w2 = self.layer.recurrent_kernel

        self.w_shape = self.w.shape.as_list()
        self.w2_shape = self.w2.shape.as_list()

        self.u = self.add_weight(
            shape=tuple([1, self.w_shape[-1]]),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            name='sn_u',
            trainable=False,
            dtype=dtypes.float32)
        self.u2 = self.add_weight(
            shape=tuple([1, self.w2_shape[-1]]),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            name='sn_u2',
            trainable=False,
            dtype=dtypes.float32)

        super(SpectralNormalization_RNN, self).build()

    @def_function.function
    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = K.learning_phase()
        
        if training==True:
            # Recompute weights for each forward pass
            self._compute_weights()
        
        output = self.layer(inputs)
        return output

    def _compute_weights(self,it=5):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        # w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        # u = self.u

        # for _ in range(it):
        #     v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
        #     u = tf.math.l2_normalize(tf.matmul(v, w))

        # sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

        # self.layer.kernel.assign(self.w / sigma)
        # self.u.assign(u)

        # w = tf.reshape(self.w2, [-1, self.w2_shape[-1]])
        # u = self.u2

        # for _ in range(it):
        #     v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
        #     u = tf.math.l2_normalize(tf.matmul(v, w))

        # sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

        # self.layer.recurrent_kernel.assign(self.w2 / sigma)
        # self.u2.assign(u)


        w_reshaped = array_ops.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = self.u
        for i in range(it):
            _u = array_ops.identity(_u)
            _v = math_ops.matmul(_u, array_ops.transpose(w_reshaped))
            _v = _v / math_ops.maximum(math_ops.reduce_sum(_v**2)**0.5, eps)
            _u = math_ops.matmul(_v, w_reshaped)
            _u = _u / math_ops.maximum(math_ops.reduce_sum(_u**2)**0.5, eps)
        
        self.u.assign(_u)
        sigma = math_ops.matmul(math_ops.matmul(_v, w_reshaped), array_ops.transpose(_u))

        

        w2_reshaped = array_ops.reshape(self.w2, [-1, self.w2_shape[-1]])
        eps = 1e-12
        _u2 = self.u2
        for i in range(it):
            _u2 = array_ops.identity(_u2)
            _v2 = math_ops.matmul(_u2, array_ops.transpose(w2_reshaped))
            _v2 = _v2 / math_ops.maximum(math_ops.reduce_sum(_v2**2)**0.5, eps)
            _u2 = math_ops.matmul(_v2, w2_reshaped)
            _u2 = _u2 / math_ops.maximum(math_ops.reduce_sum(_u2**2)**0.5, eps)

        self.u2.assign(_u2)
        sigma = math_ops.matmul(math_ops.matmul(_v2, w2_reshaped), array_ops.transpose(_u2))

        

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())