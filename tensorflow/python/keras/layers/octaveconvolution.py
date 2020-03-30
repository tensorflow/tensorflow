
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.keras.layers.convolutional import UpSampling1D
from tensorflow.python.keras.layers.convolutional import UpSampling2D


# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.octconv1d','keras.layers.OctaveConv1D')
class OctaveConv1D(Layer):
    """Octave 1D convolutions which creates two convolution kernel that is convolved
    with the layer of input over a single spatial dimmension to produce
    low and high frequency output tensor


    # Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        octave: The division of the spatial dimensions by a power or multiples of 2.
        ratio_out: The ratio of filters for lower spatial resolution.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    #Usage:
        inputs = tf.keras.layers.Input(shape=(32, 3))
        conv=tf.keras.layers.OctaveConv1D(filters=16, kernel_size=3, octave=2, ratio_out=0.125)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=conv)
        input_data=tf.random.normal([1,32,3])
        lout=model(input_data)

        output_shape
        lout[0].shape,lout[1].shape (high frequency output,low frequency output)

        output=>(TensorShape([1, 32, 14]), TensorShape([1, 16, 2]))

    # References
        - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution]
          (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 ratio_out=0.5,
                 strides=1,
                 dilation_rate=1,
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
        super(OctaveConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.octave = octave
        self.ratio_out = ratio_out
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.filters_low = int(filters * self.ratio_out)
        self.filters_high = filters - self.filters_low

        self.conv_high_to_high, self.conv_low_to_high = None, None
        if self.filters_high > 0:
            self.conv_high_to_high = self._init_conv(self.filters_high, name='{}-Conv1D-HH'.format(self.name))
            self.conv_low_to_high = self._init_conv(self.filters_high, name='{}-Conv1D-LH'.format(self.name))
        self.conv_low_to_low, self.conv_high_to_low = None, None
        if self.filters_low > 0:
            self.conv_low_to_low = self._init_conv(self.filters_low, name='{}-Conv1D-HL'.format(self.name))
            self.conv_high_to_low = self._init_conv(self.filters_low, name='{}-Conv1D-LL'.format(self.name))
        self.pooling = AveragePooling1D(
            pool_size=self.octave,
            padding='valid',
            name='{}-AveragePooling1D'.format(self.name),
        )
        self.up_sampling = UpSampling1D(
            size=self.octave,
            name='{}-UpSampling1D'.format(self.name),
        )

    def _init_conv(self, filters, name):
        return Conv1D(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=name,
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None
        if input_shape_high[-1] is None:
            raise ValueError('The channel dimension of the higher spatial inputs '
                             'should be defined. Found `None`.')
        if input_shape_low is not None and input_shape_low[-1] is None:
            raise ValueError('The channel dimension of the lower spatial inputs '
                             'should be defined. Found `None`.')
        if input_shape_high[-2] is not None and input_shape_high[-2] % self.octave != 0:
            raise ValueError('The length of the higher spatial inputs should be divisible by the octave. '
                             'Found {} and {}.'.format(input_shape_high, self.octave))
        if input_shape_low is None:
            self.conv_low_to_high, self.conv_low_to_low = None, None

        if self.conv_high_to_high is not None:
            with backend.name_scope(self.conv_high_to_high.name):
                self.conv_high_to_high.build(input_shape_high)
        if self.conv_low_to_high is not None:
            with backend.name_scope(self.conv_low_to_high.name):
                self.conv_low_to_high.build(input_shape_low)
        if self.conv_high_to_low is not None:
            with backend.name_scope(self.conv_high_to_low.name):
                self.conv_high_to_low.build(input_shape_high)
        if self.conv_low_to_low is not None:
            with backend.name_scope(self.conv_low_to_low.name):
                self.conv_low_to_low.build(input_shape_low)
        super(OctaveConv1D, self).build(input_shape)

    @property
    def trainable_weights(self):
        weights = []
        if self.conv_high_to_high is not None:
            weights += self.conv_high_to_high.trainable_weights
        if self.conv_low_to_high is not None:
            weights += self.conv_low_to_high.trainable_weights
        if self.conv_high_to_low is not None:
            weights += self.conv_high_to_low.trainable_weights
        if self.conv_low_to_low is not None:
            weights += self.conv_low_to_low.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        if self.conv_high_to_high is not None:
            weights += self.conv_high_to_high.non_trainable_weights
        if self.conv_low_to_high is not None:
            weights += self.conv_low_to_high.non_trainable_weights
        if self.conv_high_to_low is not None:
            weights += self.conv_high_to_low.non_trainable_weights
        if self.conv_low_to_low is not None:
            weights += self.conv_low_to_low.non_trainable_weights
        return weights

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None

        output_shape_high = None
        if self.filters_high > 0:
            output_shape_high = self.conv_high_to_high.compute_output_shape(input_shape_high)
        output_shape_low = None
        if self.filters_low > 0:
            output_shape_low = self.conv_high_to_low.compute_output_shape(
                self.pooling.compute_output_shape(input_shape_high),
            )

        if self.filters_low == 0:
            return output_shape_high
        if self.filters_high == 0:
            return output_shape_low
        return [output_shape_high, output_shape_low]

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs_high, inputs_low = inputs
        else:
            inputs_high, inputs_low = inputs, None

        outputs_high_to_high, outputs_low_to_high = 0.0, 0.0
        if self.conv_high_to_high is not None:
            outputs_high_to_high = self.conv_high_to_high(inputs_high)
        if self.conv_low_to_high is not None:
            outputs_low_to_high = self.up_sampling(self.conv_low_to_high(inputs_low))
        outputs_high = outputs_high_to_high + outputs_low_to_high

        outputs_low_to_low, outputs_high_to_low = 0.0, 0.0
        if self.conv_low_to_low is not None:
            outputs_low_to_low = self.conv_low_to_low(inputs_low)
        if self.conv_high_to_low is not None:
            outputs_high_to_low = self.conv_high_to_low(self.pooling(inputs_high))
        outputs_low = outputs_low_to_low + outputs_high_to_low

        if self.filters_low == 0:
            return outputs_high
        if self.filters_high == 0:
            return outputs_low
        return [outputs_high, outputs_low]

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'octave': self.octave,
            'ratio_out': self.ratio_out,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(OctaveConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



@keras_export('keras.layers.octconv2d','keras.layers.OctaveConv2D')
class OctaveConv2D(Layer):
    """Octave 2D convolutions which creates convolution kernel that is convolved
    with the layer of input over a single spatial dimmension to produce
    low and high frequency output tensor


    # Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        octave: The division of the spatial dimensions by a power or multiples of 2.
        ratio_out: The ratio of filters for lower spatial resolution.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
    # Arguments
        octave: The division of the spatial dimensions by a power of 2.
        ratio_out: The ratio of filters for lower spatial resolution.
    # References
        - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution]
          (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 ratio_out=0.5,
                 strides=(1, 1),
                 data_format=None,
                 dilation_rate=(1, 1),
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
        super(OctaveConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.octave = octave
        self.ratio_out = ratio_out
        self.strides = strides
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.filters_low = int(filters * self.ratio_out)
        self.filters_high = filters - self.filters_low

        self.conv_high_to_high, self.conv_low_to_high = None, None
        if self.filters_high > 0:
            self.conv_high_to_high = self._init_conv(self.filters_high, name='{}-Conv2D-HH'.format(self.name))
            self.conv_low_to_high = self._init_conv(self.filters_high, name='{}-Conv2D-LH'.format(self.name))
        self.conv_low_to_low, self.conv_high_to_low = None, None
        if self.filters_low > 0:
            self.conv_low_to_low = self._init_conv(self.filters_low, name='{}-Conv2D-HL'.format(self.name))
            self.conv_high_to_low = self._init_conv(self.filters_low, name='{}-Conv2D-LL'.format(self.name))
        self.pooling = AveragePooling2D(
            pool_size=self.octave,
            padding='valid',
            data_format=data_format,
            name='{}-AveragePooling2D'.format(self.name),
        )
        self.up_sampling = UpSampling2D(
            size=self.octave,
            data_format=data_format,
            interpolation='nearest',
            name='{}-UpSampling2D'.format(self.name),
        )

    def _init_conv(self, filters, name):
        return Conv2D(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=name,
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None
        if self.data_format == 'channels_first':
            channel_axis, rows_axis, cols_axis = 1, 2, 3
        else:
            rows_axis, cols_axis, channel_axis = 1, 2, 3
        if input_shape_high[channel_axis] is None:
            raise ValueError('The channel dimension of the higher spatial inputs '
                             'should be defined. Found `None`.')
        if input_shape_low is not None and input_shape_low[channel_axis] is None:
            raise ValueError('The channel dimension of the lower spatial inputs '
                             'should be defined. Found `None`.')
        if input_shape_high[rows_axis] is not None and input_shape_high[rows_axis] % self.octave != 0 or \
           input_shape_high[cols_axis] is not None and input_shape_high[cols_axis] % self.octave != 0:
            raise ValueError('The rows and columns of the higher spatial inputs should be divisible by the octave. '
                             'Found {} and {}.'.format(input_shape_high, self.octave))
        if input_shape_low is None:
            self.conv_low_to_high, self.conv_low_to_low = None, None

        if self.conv_high_to_high is not None:
            with backend.name_scope(self.conv_high_to_high.name):
                self.conv_high_to_high.build(input_shape_high)
        if self.conv_low_to_high is not None:
            with backend.name_scope(self.conv_low_to_high.name):
                self.conv_low_to_high.build(input_shape_low)
        if self.conv_high_to_low is not None:
            with backend.name_scope(self.conv_high_to_low.name):
                self.conv_high_to_low.build(input_shape_high)
        if self.conv_low_to_low is not None:
            with backend.name_scope(self.conv_low_to_low.name):
                self.conv_low_to_low.build(input_shape_low)
        super(OctaveConv2D, self).build(input_shape)

    @property
    def trainable_weights(self):
        weights = []
        if self.conv_high_to_high is not None:
            weights += self.conv_high_to_high.trainable_weights
        if self.conv_low_to_high is not None:
            weights += self.conv_low_to_high.trainable_weights
        if self.conv_high_to_low is not None:
            weights += self.conv_high_to_low.trainable_weights
        if self.conv_low_to_low is not None:
            weights += self.conv_low_to_low.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        if self.conv_high_to_high is not None:
            weights += self.conv_high_to_high.non_trainable_weights
        if self.conv_low_to_high is not None:
            weights += self.conv_low_to_high.non_trainable_weights
        if self.conv_high_to_low is not None:
            weights += self.conv_high_to_low.non_trainable_weights
        if self.conv_low_to_low is not None:
            weights += self.conv_low_to_low.non_trainable_weights
        return weights

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None

        output_shape_high = None
        if self.filters_high > 0:
            output_shape_high = self.conv_high_to_high.compute_output_shape(input_shape_high)
        output_shape_low = None
        if self.filters_low > 0:
            output_shape_low = self.conv_high_to_low.compute_output_shape(
                self.pooling.compute_output_shape(input_shape_high),
            )

        if self.filters_low == 0:
            return output_shape_high
        if self.filters_high == 0:
            return output_shape_low
        return [output_shape_high, output_shape_low]

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs_high, inputs_low = inputs
        else:
            inputs_high, inputs_low = inputs, None

        outputs_high_to_high, outputs_low_to_high = 0.0, 0.0
        if self.conv_high_to_high is not None:
            outputs_high_to_high = self.conv_high_to_high(inputs_high)
        if self.conv_low_to_high is not None:
            outputs_low_to_high = self.up_sampling(self.conv_low_to_high(inputs_low))
        outputs_high = outputs_high_to_high + outputs_low_to_high

        outputs_low_to_low, outputs_high_to_low = 0.0, 0.0
        if self.conv_low_to_low is not None:
            outputs_low_to_low = self.conv_low_to_low(inputs_low)
        if self.conv_high_to_low is not None:
            outputs_high_to_low = self.conv_high_to_low(self.pooling(inputs_high))
        outputs_low = outputs_low_to_low + outputs_high_to_low

        if self.filters_low == 0:
            return outputs_high
        if self.filters_high == 0:
            return outputs_low
        return [outputs_high, outputs_low]

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'octave': self.octave,
            'ratio_out': self.ratio_out,
            'strides': self.strides,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(OctaveConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.octconvdual','keras.layers.OctaveConvDual')
class OctaveConvDual(Layer):
    def __init__(self):
        super(OctaveConvDual,self).__init__()

    def call(self,layers_, builder):
        """Apply layers for outputs of octave convolution.
    #Arguments:
        :param layers: The outputs of octave convolution.
        :param builder: A function that builds the layer or just a layer.
    :return: The output tensors.

    #Usage:
        from tensorflow.keras.layers import MaxPool2D
        inputs = tf.keras.layers.Input(shape=(32, 32, 3))
        conv = tf.keras.layers.OctaveConv2D(filters=16, kernel_size=3)(inputs)

        pool = tf.keras.layers.OctaveConvDual()(conv, MaxPool2D())
        conv = tf.keras.layers.OctaveConv2D(filters=8, kernel_size=3)(pool)

        pool = tf.keras.layers.OctaveConvDual()(conv, MaxPool2D())
        conv = tf.keras.layers.OctaveConv2D(filters=4, kernel_size=3, ratio_out=0.0)(pool)

        flatten = tf.keras.layers.Flatten()(conv)
        outputs = tf.keras.layers.Dense(units=10, activation='softmax')(flatten)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        OUTPUT:
        Model: "model_1"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to
        ==================================================================================================
        input_2 (InputLayer)            [(None, 32, 32, 3)]  0
        __________________________________________________________________________________________________
        octave_conv2d_2 (OctaveConv2D)  [(None, 32, 32, 8),  448         input_2[0][0]
        __________________________________________________________________________________________________
        octave_conv_dual (OctaveConvDua [(None, 16, 16, 8),  0           octave_conv2d_2[0][0]
                                                                         octave_conv2d_2[0][1]
        __________________________________________________________________________________________________
        octave_conv2d_3 (OctaveConv2D)  [(None, 16, 16, 4),  1168        octave_conv_dual[0][0]
                                                                         octave_conv_dual[0][1]
        __________________________________________________________________________________________________
        octave_conv_dual_1 (OctaveConvD [(None, 8, 8, 4), (N 0           octave_conv2d_3[0][0]
                                                                         octave_conv2d_3[0][1]
        __________________________________________________________________________________________________
        octave_conv2d_4 (OctaveConv2D)  (None, 8, 8, 4)      296         octave_conv_dual_1[0][0]
                                                                         octave_conv_dual_1[0][1]
        __________________________________________________________________________________________________
        flatten_1 (Flatten)             (None, 256)          0           octave_conv2d_4[0][0]
        __________________________________________________________________________________________________
        dense_1 (Dense)                 (None, 10)           2570        flatten_1[0][0]
        ==================================================================================================
        Total params: 4,482
        Trainable params: 4,482
        Non-trainable params: 0
    """
        if not isinstance(layers_, (list, tuple)):
            layers_ = [layers_]
        if isinstance(builder, Layer):
            intermediates = [builder] + [copy.copy(builder) for _ in range(len(layers_) - 1)]
        else:
            intermediates = [builder() for _ in range(len(layers_))]
        """for i, names_ in enumerate(['H', 'L']):
            if i < len(intermediates):
                intermediates[i].names += '-' + names_"""
        outputs = [intermediate(layers_[i]) for i, intermediate in enumerate(intermediates)]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
