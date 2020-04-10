# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Keras octave convolution layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import abc
import warnings

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers import UpSampling1D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import UpSampling3D

from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.layers.convolutional import Conv3DTranspose

from tensorflow.python.keras.utils import conv_utils


class OctaveConv(Layer):
    """Abstract N-D octave convolution layer (private, used as implementation base)

    The octave convolutions factorize convolutional feature maps into two groups
    at different spatial frequencies and process them with different
    convolutions at their corresponding frequency, one octave apart.
    This layer creates 4 convolution layers, 2 for high frequency feature maps
    and 2 for low frequency feature maps. For each frequency, the outputs of the
    convolution layers are concatenated in order to get 2 final outputs, with the
    ratio of low frequency feature maps being `low_freq_ratio` and
    (1 - `low_freq_ratio`) for the high frequency feature maps.
    If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the
        number of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      data_format: A string.
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the
        default initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked
        as trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.

    References
      - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
         Networks with Octave Convolution]
        (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 octave=2,
                 low_freq_ratio=0.25,
                 strides=1,
                 padding='same',
                 data_format=None,
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
                 trainable=True,
                 name=None,
                 **kwargs):
        super(OctaveConv, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs
        )
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.octave = octave
        self.low_freq_ratio = low_freq_ratio
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.filters_low = int(filters * self.low_freq_ratio)
        self.filters_high = filters - self.filters_low

        self.pooling = None
        self.up_sampling = None

        self._trainable = trainable

        if self.padding != 'same':
            warnings.warn('Padding set to {} for the octave convolution layer '
                          'with name {}. '
                          'For an optimal use of octave convolutions, set '
                          'padding to same.'.format(self.padding, self.name))

        self.kernel, self.bias = [], []
        self.conv_high_to_high, self.conv_low_to_high = None, None
        self.conv_low_to_low, self.conv_high_to_low = None, None
        self.generate_convolutions()

    def __call__(self, inputs, **kwargs):
        try:
            spec = [InputSpec(ndim=self.rank + 2)] * len(inputs)
        except TypeError:
            spec = InputSpec(ndim=self.rank + 2)
        self.input_spec = spec
        outputs = super(OctaveConv, self).__call__(inputs, **kwargs)
        return outputs

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, trainable):
        """
        Overwrites the trainable setter function of keras.layers.Layer to change
        the trainable attribute of the four keras.layers.ConvND layers
        called in this OctaveConvND layer.
        Note: there could be only two ConvND layers that are called in
        a OctaveConvND layer, for instance when you handle the input
        and the output. This is the reason for the duplicated try/except blocks.
        """
        self._trainable = trainable
        try:
            self.conv_high_to_high.trainable = trainable
        except AttributeError:
            pass
        try:
            self.conv_high_to_low.trainable = trainable
        except AttributeError:
            pass
        try:
            self.conv_low_to_high.trainable = trainable
        except AttributeError:
            pass
        try:
            self.conv_low_to_low.trainable = trainable
        except AttributeError:
            pass

    @abc.abstractmethod
    def _init_conv(self, filters, name):
        pass

    def generate_convolutions(self):
        if self.filters_high > 0:
            self.conv_high_to_high = self._init_conv(self.filters_high,
                                                     name='{}-Conv{}D-HH'.format(
                                                         self.name,
                                                         self.rank))
            self.conv_low_to_high = self._init_conv(self.filters_high,
                                                    name='{}-Conv{}D-LH'.format(
                                                        self.name,
                                                        self.rank))
        if self.filters_low > 0:
            self.conv_low_to_low = self._init_conv(self.filters_low,
                                                   name='{}-Conv{}D-LL'.format(
                                                       self.name,
                                                       self.rank))
            self.conv_high_to_low = self._init_conv(self.filters_low,
                                                    name='{}-Conv{}D-HL'.format(
                                                        self.name,
                                                        self.rank))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None

        if input_shape_low is None:
            self.conv_low_to_high, self.conv_low_to_low = None, None

        if self.conv_high_to_high is not None:
            with backend.name_scope(self.conv_high_to_high.name):
                self.conv_high_to_high.build(input_shape_high)
                self.kernel.append(self.conv_high_to_high.kernel)
                self.bias.append(self.conv_high_to_high.bias)
        if self.conv_low_to_high is not None:
            with backend.name_scope(self.conv_low_to_high.name):
                self.conv_low_to_high.build(input_shape_low)
                self.kernel.append(self.conv_low_to_high.kernel)
                self.bias.append(self.conv_low_to_high.bias)
        if self.conv_high_to_low is not None:
            with backend.name_scope(self.conv_high_to_low.name):
                self.conv_high_to_low.build(input_shape_high)
                self.kernel.append(self.conv_high_to_low.kernel)
                self.bias.append(self.conv_high_to_low.bias)
        if self.conv_low_to_low is not None:
            with backend.name_scope(self.conv_low_to_low.name):
                self.conv_low_to_low.build(input_shape_low)
                self.kernel.append(self.conv_low_to_low.kernel)
                self.bias.append(self.conv_low_to_low.bias)

        self.built = True

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs_high, inputs_low = inputs
        else:
            inputs_high, inputs_low = inputs, None

        outputs_high_to_high, outputs_low_to_high = 0.0, 0.0
        if self.conv_high_to_high is not None:
            outputs_high_to_high = self.conv_high_to_high(inputs_high)
        if self.conv_low_to_high is not None:
            outputs_low_to_high = self.up_sampling(
                self.conv_low_to_high(inputs_low))
        outputs_high = outputs_high_to_high + outputs_low_to_high

        outputs_low_to_low, outputs_high_to_low = 0.0, 0.0
        if self.conv_low_to_low is not None:
            outputs_low_to_low = self.conv_low_to_low(inputs_low)
        if self.conv_high_to_low is not None:
            outputs_high_to_low = self.conv_high_to_low(
                self.pooling(inputs_high))
        outputs_low = outputs_low_to_low + outputs_high_to_low

        if self.filters_low == 0:
            return outputs_high
        if self.filters_high == 0:
            return outputs_low
        return [outputs_high, outputs_low]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None

        output_shape_high = None
        if self.filters_high > 0:
            # outputs_high is the sum of outputs_high_to_high with
            # outputs_low_to_high so we only need to compute the output shape
            # of either one of them (output_high_to_high in this case)
            output_shape_high = self.conv_high_to_high.compute_output_shape(
                input_shape_high)
        output_shape_low = None
        if self.filters_low > 0:
            # outputs_low is the sum of outputs_high_to_low with
            # outputs_low_to_low so we only need to compute the output shape
            # of either one of them (output_high_to_low in this case)
            output_shape_low = self.conv_high_to_low.compute_output_shape(
                self.pooling.compute_output_shape(input_shape_high),
            )

        if self.filters_low == 0:
            return output_shape_high
        if self.filters_high == 0:
            return output_shape_low
        return [output_shape_high, output_shape_low]

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'octave': self.octave,
            'low_freq_ratio': self.low_freq_ratio,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(OctaveConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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


class OctaveConv1D(OctaveConv):
    """1D octave convolution layer (e.g. temporal convolution).

    This layer creates 4 1D-convolution layers that produce 2 tensors of
    outputs (see the documentation of OctaveConv for more information).
    If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword
    argument `input_shape` (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes with a single
    channel, in `data_format="channels_last"`.

    Examples:

    >>> # The inputs are 128-length vectors with 10 timesteps, and the batch size
    >>> # is None.
    >>> x = Input(shape=(10,128,))
    >>> y = tf.keras.layers.octave_convolutional.OctaveConv1D(32, 3,
    ... padding='same', activation='relu',low_freq_ratio=0.25)(x)
    >>> print(len(y))
    2
    >>> print(y[0].shape, y[1].shape])
    (None, 10, 24) (None, 5, 8)

    Arguments
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.

     Input shape:
      First case,
        single input (e.g. first octave convolution layer of the
        architecture):
          3D tensor with shape:
          `(samples, channels, input_dim)` if data_format='channels_first'
          or 3D tensor with shape:
          `(samples, input_dim, channels) if data_format='channels_last'.
      Second case,
        list of two 3D tensors with shape:
          [`(samples, (1-ratio_out) * filters, input_dim_H)`,
          `(samples, ratio_out * filters, input_dim_L)`] if
          data_format='channels_first'
         or list of two 3D tensors with shape:
          [`(samples, input_dim_H, (1-ratio_out) * filters)`,
          `(samples, input_dim_H, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Output shape:
      First case,
        single output (e.g. last octave convolution layer of the
        architecture):
          3D tensor with shape:
          `(samples, channels, output_dim)` if data_format='channels_first'
          or 3D tensor with shape:
          `(samples, output_dim, channels)` if data_format='channels_last'.
      Second case,
        list of two 3D tensors with shape:
          [`(samples, (1-ratio_out) * filters, output_dim_H)`,
          `(samples, ratio_out * filters, output_dim_L)`] if
          data_format='channels_first'
        or list of two 3D tensors with shape:
          [`(samples, output_dim_H, (1-ratio_out) * filters)`,
          `(samples, output_dim_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Raises:
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References
        - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
           Networks with Octave Convolution]
          (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 low_freq_ratio=0.25,
                 strides=1,
                 padding='same',
                 data_format=None,
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
        super(OctaveConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            octave=octave,
            low_freq_ratio=low_freq_ratio,
            strides=strides,
            padding=padding,
            data_format=conv_utils.normalize_data_format(data_format),
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.pooling = AveragePooling1D(
            pool_size=self.octave,
            padding='valid',
            data_format=data_format,
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
            padding=self.padding,
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
            trainable=self.trainable,
            name=name
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None
        if len(input_shape_high) != 3:
            raise ValueError(
                'High frequency input should have rank 3; Received '
                'input shape {}'.format(str(input_shape_high)))
        if self.data_format == 'channels_first':
            channel_axis, data_axis = 1, 2
        else:
            data_axis, channel_axis = 1, 2
        if input_shape_high[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the higher spatial inputs '
                'should be defined. Found `None`.')
        if input_shape_low is not None and input_shape_low[
            channel_axis] is None:
            raise ValueError(
                'The channel dimension of the lower spatial inputs '
                'should be defined. Found `None`.')
        if input_shape_high[data_axis] is not None and \
                input_shape_high[data_axis] % self.octave != 0:
            raise ValueError(
                'The dimension with the data of the higher spatial inputs '
                'should be divisible by the octave. '
                'Found {} and {}.'.format(input_shape_high, self.octave))

        super(OctaveConv1D, self).build(input_shape)


class OctaveConv2D(OctaveConv):
    """2D octave convolution layer.

    This layer creates 4 2D-convolution layers that produce 2 tensors of
    outputs (see the documentation of OctaveConv for more information).
    If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    Examples:

    >>> # The inputs are 28x28 RGB images with `channels_last` and the batch
    >>> # size is None.
    >>> x = Input(shape=(28,28,3,))
    >>> y = tf.keras.layers.octave_convolutional.OctaveConv2D(
    ... 32, 3, activation='relu', low_freq_ratio=0.25)(x)
    >>> print(len(y))
    2
    >>> print(y.shape)
    (None, 28, 28, 24) (None, 14, 14, 8)

    Arguments
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.

    Input shape:
      First case,
        single input (e.g. first octave convolution layer of the
        architecture):
          4D tensor with shape:
          `(samples, channels, rows, cols)` if data_format='channels_first'
          or 4D tensor with shape:
          `(samples, rows, cols, channels)` if data_format='channels_last'.
      Second case,
        two inputs:
          list of two 4D tensors with shape:
          [`(samples, (1-ratio_out) * filters, rows_H, cols_H)`,
          `(samples, ratio_out * filters, rows_L, cols_L)`] if
          data_format='channels_first'
          or list of two 4D tensors with shape:
          [`(samples, rows_H, cols_H, (1-ratio_out) * filters)`,
          `(samples, rows_L, cols_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Output shape:
      First case,
        single output (e.g. last octave convolution layer of the
        architecture):
          4D tensor with shape:
          `(samples, channels, new_rows, new_cols)` if data_format='channels_first'
          or 4D tensor with shape:
          `(samples, new_rows, new_cols, channels)` if data_format='channels_last'.
      Second case,
        list of two 4D tensors with shape:
          [`(samples, (1-ratio_out) * filters, new_rows_H, new_cols_H)`,
          `(samples, ratio_out * filters, new_rows_L, new_cols_L)`] if
          data_format='channels_first'
        or list of two 4D tensors with shape:
          [`(samples, new_rows_H, new_cols_H, (1-ratio_out) * filters)`,
          `(samples, new_rows_L, new_cols_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References
      - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
        Networks with Octave Convolution]
        (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 low_freq_ratio=0.5,
                 strides=(1, 1),
                 padding='same',
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
        super(OctaveConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            octave=octave,
            low_freq_ratio=low_freq_ratio,
            strides=strides,
            padding=padding,
            data_format=conv_utils.normalize_data_format(data_format),
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

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
            padding=self.padding,
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
            trainable=self.trainable,
            name=name
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None
        if len(input_shape_high) != 4:
            raise ValueError(
                'High frequency input should have rank 4; Received '
                'input shape {}'.format(str(input_shape_high)))
        if input_shape_low is not None and len(input_shape_low) != 4:
            raise ValueError(
                'Low frequency input should have rank 4; Received '
                'input shape {}'.format(str(input_shape_low)))
        if self.data_format == 'channels_first':
            channel_axis, rows_axis, cols_axis = 1, 2, 3
        else:
            rows_axis, cols_axis, channel_axis = 1, 2, 3
        if input_shape_high[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the higher spatial inputs '
                'should be defined. Found `None`.')
        if input_shape_low is not None and input_shape_low[
            channel_axis] is None:
            raise ValueError(
                'The channel dimension of the lower spatial inputs '
                'should be defined. Found `None`.')
        if input_shape_high[rows_axis] is not None and input_shape_high[
            rows_axis] % self.octave != 0 or \
                input_shape_high[cols_axis] is not None and input_shape_high[
            cols_axis] % self.octave != 0:
            raise ValueError(
                'The rows and columns of the higher spatial inputs should be '
                'divisible by the octave. '
                'Found {} and {}.'.format(input_shape_high, self.octave))

        super(OctaveConv2D, self).build(input_shape)


class OctaveConv3D(OctaveConv):
    """3D octave convolution layer.

    This layer creates 4 3D-convolution layers that produce 2 tensors of
    outputs(see the documentation of OctaveConv for more information).
    If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.

    Examples:

    >>> # The inputs are 28x28x28 volumes with a single channel, and the
    >>> # batch size is None.
    >>> x = Input(shape=(28, 28, 28, 1,))
    >>> y = tf.keras.layers.octave_convolutional.OctaveConv3D(
    ... 32, 3, activation='relu', low_freq_ratio=0.25)(x)
    >>> print(len(y))
    2
    >>> print(y.shape)
    (None, 28, 28, 28, 24) (None, 14, 14, 14, 8)

    Arguments
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.

    Input shape:
      First case,
        single input (e.g. first octave convolution layer of the
        architecture):
          5D tensor with shape:
          `(samples, channels, depth, rows, cols)` if data_format='channels_first'
          or 5D tensor with shape:
          `(samples, depth, rows, cols, channels)` if data_format='channels_last'.
      Second case,
        two inputs:
          list of two 5D tensors with shape:
          [`(samples, (1-ratio_out) * filters, depth_H, rows_H, cols_H)`,
          `(samples, ratio_out * filters, depth_L, rows_L, cols_L)`] if
          data_format='channels_first'
          or list of two 5D tensors with shape:
          [`(samples, depth_H, rows_H, cols_H, (1-ratio_out) * filters)`,
          `(samples, depth_L, rows_L, cols_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Output shape:
      First case,
        single output (e.g. last octave convolution layer of the
        architecture):
          5D tensor with shape:
          `(samples, channels, new_depth, new_rows, new_cols)`
          if data_format='channels_first'
          or 5D tensor with shape:
          `(samples, new_depth, new_rows, new_cols, channels)`
          if data_format='channels_last'.
      Second case,
        list of two 5D tensors with shape:
          [`(samples, (1-ratio_out) * filters, new_depth_H, new_rows_H, new_cols_H)`,
          `(samples, ratio_out * filters, new_depth_L, new_rows_L, new_cols_L)`]
          if data_format='channels_first'
        or list of two 5D tensors with shape:
          [`(samples, new_depth_H, new_rows_H, new_cols_H, (1-ratio_out) * filters)`,
          `(samples, new_depth_L, new_rows_L, new_cols_L, ratio_out * filters)`]
          if data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References
      - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
         Networks with Octave Convolution]
        (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 low_freq_ratio=0.25,
                 strides=(1, 1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
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
        super(OctaveConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            octave=octave,
            low_freq_ratio=low_freq_ratio,
            strides=strides,
            padding=padding,
            data_format=conv_utils.normalize_data_format(data_format),
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.pooling = AveragePooling3D(
            pool_size=self.octave,
            padding='valid',
            data_format=data_format,
            name='{}-AveragePooling3D'.format(self.name),
        )
        self.up_sampling = UpSampling3D(
            size=self.octave,
            data_format=data_format,
            name='{}-UpSampling3D'.format(self.name),
        )

    def _init_conv(self, filters, name):
        return Conv3D(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
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
            trainable=self.trainable,
            name=name
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None
        if len(input_shape_high) != 5:
            raise ValueError(
                'High frequency input should have rank 5; Received '
                'input shape {}'.format(str(input_shape_high)))
        if input_shape_low is not None and len(input_shape_low) != 5:
            raise ValueError(
                'Low frequency input should have rank 5; Received '
                'input shape {}'.format(str(input_shape_low)))
        if self.data_format == 'channels_first':
            channel_axis, depth_axis, height_axis, width_axis = 1, 2, 3, 4
        else:
            depth_axis, height_axis, width_axis, channel_axis = 1, 2, 3, 4
        if input_shape_high[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the higher spatial inputs '
                'should be defined. Found `None`.')
        if input_shape_low is not None and \
                input_shape_low[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the lower spatial inputs '
                'should be defined. Found `None`.')
        if input_shape_high[depth_axis] is not None and \
                input_shape_high[depth_axis] % self.octave != 0 or \
                input_shape_high[height_axis] is not None and \
                input_shape_high[height_axis] % self.octave != 0 or \
                input_shape_high[width_axis] is not None and \
                input_shape_high[width_axis] % self.octave != 0:
            raise ValueError(
                'The depths, heights and widths of the higher spatial inputs '
                'should be divisible by the octave. '
                'Found {} and {}.'.format(input_shape_high, self.octave))
        super(OctaveConv3D, self).build(input_shape)


class OctaveConv2DTranspose(OctaveConv2D):
    """2D transposed octave convolution layer (sometimes called Deconvolution).

    This layer creates 4 convolution layers that produce 2 tensors of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    with a single channel,
    in `data_format="channels_last"`.

    Arguments
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      output_padding: An integer or tuple/list of 2 integers,
        specifying the amount of padding along the height and width
        of the output tensor.
        Can be a single integer to specify the same value for all
        spatial dimensions.
        The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.
        If set to `None` (default), the output shape is inferred.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
         bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.

    Input shape:
      First case,
        single input (e.g. first octave convolution layer of the
        architecture):
          4D tensor with shape:
          `(samples, channels, rows, cols)` if data_format='channels_first'
          or 4D tensor with shape:
          `(samples, rows, cols, channels)` if data_format='channels_last'.
      Second case,
        two inputs:
          list of two 4D tensors with shape:
          [`(samples, (1-ratio_out) * filters, rows_H, cols_H)`,
          `(samples, ratio_out * filters, rows_L, cols_L)`] if
          data_format='channels_first'
          or list of two 4D tensors with shape:
          [`(samples, rows_H, cols_H, (1-ratio_out) * filters)`,
          `(samples, rows_L, cols_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Output shape:
      First case,
        single output (e.g. last octave convolution layer of the
        architecture):
          4D tensor with shape:
          `(samples, channels, new_rows, new_cols)` if data_format='channels_first'
          or 4D tensor with shape:
          `(samples, new_rows, new_cols, channels)` if data_format='channels_last'.
      Second case,
        list of two 4D tensors with shape:
          [`(samples, (1-ratio_out) * filters, new_rows_H, new_cols_H)`,
          `(samples, ratio_out * filters, new_rows_L, new_cols_L)`] if
          data_format='channels_first'
        or list of two 4D tensors with shape:
          [`(samples, new_rows_H, new_cols_H, (1-ratio_out) * filters)`,
          `(samples, new_rows_L, new_cols_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References
      - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
        Networks with Octave Convolution]
        (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 low_freq_ratio=0.25,
                 strides=(1, 1),
                 padding='same',
                 output_padding=None,
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

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides)
                                     + ' must be greater than output padding ' +
                                     str(self.output_padding))

        super(OctaveConv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            octave=octave,
            low_freq_ratio=low_freq_ratio,
            strides=strides,
            padding=padding,
            data_format=conv_utils.normalize_data_format(data_format),
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        if self.filters_high > 0:
            self.conv_high_to_high._name = '{}-ConvTrans2D-HH'.format(self.name)
            self.conv_low_to_high._name = '{}-ConvTrans2D-LH'.format(self.name)
        if self.filters_low > 0:
            self.conv_low_to_low._name = '{}-ConvTrans2D-LL'.format(self.name)
            self.conv_high_to_low._name = '{}-ConvTrans2D-HL'.format(self.name)

    def _init_conv(self, filters, name):
        return Conv2DTranspose(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
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
            trainable=self.trainable,
            name=name
        )

    def get_config(self):
        config = {
            'output_padding': self.output_padding
        }
        base_config = super(OctaveConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OctaveConv3DTranspose(OctaveConv3D):
    """3D transposed octave convolution layer (sometimes called Deconvolution).

    This layer creates 4 convolution layers that produce 2 tensors of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.

    Arguments
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      output_padding: An integer or tuple/list of 2 integers,
        specifying the amount of padding along the height and width
        of the output tensor.
        Can be a single integer to specify the same value for all
        spatial dimensions.
        The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.
        If set to `None` (default), the output shape is inferred.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.

    Input shape:
      First case,
        single input (e.g. first octave convolution layer of the
        architecture):
          5D tensor with shape:
          `(samples, channels, depth, rows, cols)` if data_format='channels_first'
          or 5D tensor with shape:
          `(samples, depth, rows, cols, channels)` if data_format='channels_last'.
      Second case,
        two inputs:
          list of two 5D tensors with shape:
          [`(samples, (1-ratio_out) * filters, depth_H, rows_H, cols_H)`,
          `(samples, ratio_out * filters, depth_L, rows_L, cols_L)`] if
          data_format='channels_first'
          or list of two 5D tensors with shape:
          [`(samples, depth_H, rows_H, cols_H, (1-ratio_out) * filters)`,
          `(samples, depth_L, rows_L, cols_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Output shape:
      First case,
        single output (e.g. last octave convolution layer of the
        architecture):
          5D tensor with shape:
          `(samples, channels, new_depth, new_rows, new_cols)`
          if data_format='channels_first'
          or 5D tensor with shape:
          `(samples, new_depth, new_rows, new_cols, channels)`
          if data_format='channels_last'.
      Second case,
        list of two 5D tensors with shape:
          [`(samples, (1-ratio_out) * filters, new_depth_H, new_rows_H, new_cols_H)`,
          `(samples, ratio_out * filters, new_depth_L, new_rows_L, new_cols_L)`]
          if data_format='channels_first'
        or list of two 5D tensors with shape:
          [`(samples, new_depth_H, new_rows_H, new_cols_H, (1-ratio_out) * filters)`,
          `(samples, new_depth_L, new_rows_L, new_cols_L, ratio_out * filters)`]
          if data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References
      - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
         Networks with Octave Convolution]
        (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 low_freq_ratio=0.25,
                 strides=(1, 1, 1),
                 padding='same',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1, 1),
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
        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides)
                                     + ' must be greater than output padding ' +
                                     str(self.output_padding))

        super(OctaveConv3DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            octave=octave,
            low_freq_ratio=low_freq_ratio,
            strides=strides,
            padding=padding,
            data_format=conv_utils.normalize_data_format(data_format),
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        if self.filters_high > 0:
            self.conv_high_to_high._name = '{}-ConvTrans2D-HH'.format(self.name)
            self.conv_low_to_high._name = '{}-ConvTrans2D-LH'.format(self.name)
        if self.filters_low > 0:
            self.conv_low_to_low._name = '{}-ConvTrans2D-LL'.format(self.name)
            self.conv_high_to_low._name = '{}-ConvTrans2D-HL'.format(self.name)

    def _init_conv(self, filters, name):
        return Conv3DTranspose(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
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
            trainable=self.trainable,
            name=name
        )

    def get_config(self):
        config = {
            'output_padding': self.output_padding
        }
        base_config = super(OctaveConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OctaveConvAdd(Layer):
    """
    Layer that applies core keras layers to the high frequency and low frequency
    outputs of Octave Convolution layers

    Examples:

     >>> # The inputs are 28x28 grayscale images with `channels_last` and the batch
     >>> # size is None.
     >>> l1_input = Input(shape=(28,28, 1))
     >>> l2 = OctaveConv2D(32, (5, 5), activation='relu', low_freq_ratio=0.25)
     ... (l1_input)
     >>> l3 = OctaveConvAdd()(l2, builder=MaxPooling2D(pool_size=(2, 2)))

    Arguments:
      inputs: list/tuple of input tensors.
      **kwargs: should contain a `builder` argument which is a keras layer
        (e.g. tensorflow.python.keras.layers.Dropout)

    Returns:
      outputs: list of output tensors.

    Raises:
      ValueError: if **kwargs doesn't have a valid `builder` argument.
    """

    def __init__(self):
        super(OctaveConvAdd, self).__init__()

    def call(self, inputs, **kwargs):
        builder = kwargs.get('builder', None)
        if builder is None:
            raise ValueError('A builder layer must be passed as argument.')
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if len(inputs) > 2:
            raise ValueError('The maximum number of inputs is 2, {} were '
                             'given'.format(str(len(inputs))))
        if isinstance(builder, Layer):
            intermediates = [builder] + [copy.copy(builder) for _ in
                                         range(len(inputs) - 1)]
        else:
            intermediates = [builder() for _ in range(len(inputs))]
        for i, name in enumerate(['H', 'L']):
            if i < len(intermediates):
                intermediates[i]._name += '-' + name
        outputs = [intermediate(inputs[i]) for i, intermediate in
                   enumerate(intermediates)]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
