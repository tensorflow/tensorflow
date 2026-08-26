import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import activations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn

class QuantizedDense(Layer):
    """A densely-connected layer with weight quantization.
    
    This layer acts like a standard Dense layer but simulates
    4-bit or 8-bit weight quantization using fake quantization nodes.
    """
    
    def __init__(self,
                 units,
                 bits=8,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(QuantizedDense, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)
        self.units = int(units)
        self.bits = int(bits)
        if self.bits not in [4, 8]:
            raise ValueError('Only 4-bit and 8-bit quantization are supported.')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `QuantizedDense` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        # Apply Fake Quantization to weights
        # fake_quant_with_min_max_vars requires float32 inputs.
        kernel = tf.cast(self.kernel, dtypes.float32)
        min_val = tf.math.reduce_min(kernel)
        max_val = tf.math.reduce_max(kernel)
        # Ensure min_val and max_val are not equal to avoid division by zero or NaN gradients
        max_val = tf.math.maximum(max_val, min_val + 1e-5)
        
        quant_bits = self.bits
        
        quantized_kernel = tf.quantization.fake_quant_with_min_max_vars(
            kernel, 
            min_val, 
            max_val, 
            num_bits=quant_bits,
            narrow_range=True)
        quantized_kernel = tf.cast(quantized_kernel, self.dtype)
            
        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            if isinstance(inputs, tf.SparseTensor):
                raise NotImplementedError("Sparse inputs not supported.")
            outputs = tf.matmul(a=inputs, b=quantized_kernel)
        else:
            outputs = tf.tensordot(inputs, quantized_kernel, [[rank - 1], [0]])
            
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
            
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
        
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)
        
    def get_config(self):
        config = super(QuantizedDense, self).get_config()
        config.update({
            'units': self.units,
            'bits': self.bits,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config
