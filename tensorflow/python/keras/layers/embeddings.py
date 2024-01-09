# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Embedding layer."""
# pylint: disable=g-classes-have-attributes

from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops


class Embedding(Layer):
  """Turns positive integers (indexes) into dense vectors of fixed size.

  e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

  This layer can only be used as the first layer in a model.

  Example:

  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
  >>> # The model will take as input an integer matrix of size (batch,
  >>> # input_length), and the largest integer (i.e. word index) in the input
  >>> # should be no larger than 999 (vocabulary size).
  >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
  >>> # dimension.
  >>> input_array = np.random.randint(1000, size=(32, 10))
  >>> model.compile('rmsprop', 'mse')
  >>> output_array = model.predict(input_array)
  >>> print(output_array.shape)
  (32, 10, 64)

  Args:
    input_dim: Integer. Size of the vocabulary,
      i.e. maximum integer index + 1.
    output_dim: Integer. Dimension of the dense embedding.
    embeddings_initializer: Initializer for the `embeddings`
      matrix (see `keras.initializers`).
    embeddings_regularizer: Regularizer function applied to
      the `embeddings` matrix (see `keras.regularizers`).
    embeddings_constraint: Constraint function applied to
      the `embeddings` matrix (see `keras.constraints`).
    mask_zero: Boolean, whether or not the input value 0 is a special "padding"
      value that should be masked out.
      This is useful when using recurrent layers
      which may take variable length input.
      If this is `True`, then all subsequent layers
      in the model need to support masking or an exception will be raised.
      If mask_zero is set to True, as a consequence, index 0 cannot be
      used in the vocabulary (input_dim should equal size of
      vocabulary + 1).
    input_length: Length of input sequences, when it is constant.
      This argument is required if you are going to connect
      `Flatten` then `Dense` layers upstream
      (without it, the shape of the dense outputs cannot be computed).

  Input shape:
    2D tensor with shape: `(batch_size, input_length)`.

  Output shape:
    3D tensor with shape: `(batch_size, input_length, output_dim)`.

  **Note on variable placement:**
  By default, if a GPU is available, the embedding matrix will be placed on
  the GPU. This achieves the best performance, but it might cause issues:

  - You may be using an optimizer that does not support sparse GPU kernels.
  In this case you will see an error upon training your model.
  - Your embedding matrix may be too large to fit on your GPU. In this case
  you will see an Out Of Memory (OOM) error.

  In such cases, you should place the embedding matrix on the CPU memory.
  You can do so with a device scope, as such:

  ```python
  with tf.device('cpu:0'):
    embedding_layer = Embedding(...)
    embedding_layer.build()
  ```

  The pre-built `embedding_layer` instance can then be added to a `Sequential`
  model (e.g. `model.add(embedding_layer)`), called in a Functional model
  (e.g. `x = embedding_layer(x)`), or used in a subclassed model.
  """

  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               **kwargs):
    if 'input_shape' not in kwargs:
      if input_length:
        kwargs['input_shape'] = (input_length,)
      else:
        kwargs['input_shape'] = (None,)
    if input_dim <= 0 or output_dim <= 0:
      raise ValueError('Both `input_dim` and `output_dim` should be positive, '
                       'found input_dim {} and output_dim {}'.format(
                           input_dim, output_dim))
    if (not base_layer_utils.v2_dtype_behavior_enabled() and
        'dtype' not in kwargs):
      # In TF1, the dtype defaults to the input dtype which is typically int32,
      # so explicitly set it to floatx
      kwargs['dtype'] = backend.floatx()
    # We set autocast to False, as we do not want to cast floating- point inputs
    # to self.dtype. In call(), we cast to int32, and casting to self.dtype
    # before casting to int32 might cause the int32 values to be different due
    # to a loss of precision.
    kwargs['autocast'] = False
    super(Embedding, self).__init__(**kwargs)

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embeddings_initializer = initializers.get(embeddings_initializer)
    self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.embeddings_constraint = constraints.get(embeddings_constraint)
    self.mask_zero = mask_zero
    self.supports_masking = mask_zero
    self.input_length = input_length

  @tf_utils.shape_type_conversion
  def build(self, input_shape=None):
    self.embeddings = self.add_weight(
        shape=(self.input_dim, self.output_dim),
        initializer=self.embeddings_initializer,
        name='embeddings',
        regularizer=self.embeddings_regularizer,
        constraint=self.embeddings_constraint,
        experimental_autocast=False)
    self.built = True

  def compute_mask(self, inputs, mask=None):
    if not self.mask_zero:
      return None
    return math_ops.not_equal(inputs, 0)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.input_length is None:
      return input_shape + (self.output_dim,)
    else:
      # input_length can be tuple if input is 3D or higher
      if isinstance(self.input_length, (list, tuple)):
        in_lens = list(self.input_length)
      else:
        in_lens = [self.input_length]
      if len(in_lens) != len(input_shape) - 1:
        raise ValueError('"input_length" is %s, '
                         'but received input has shape %s' % (str(
                             self.input_length), str(input_shape)))
      else:
        for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
          if s1 is not None and s2 is not None and s1 != s2:
            raise ValueError('"input_length" is %s, '
                             'but received input has shape %s' % (str(
                                 self.input_length), str(input_shape)))
          elif s1 is None:
            in_lens[i] = s2
      return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

  def call(self, inputs):
    dtype = backend.dtype(inputs)
    if dtype != 'int32' and dtype != 'int64':
      inputs = math_ops.cast(inputs, 'int32')
    out = embedding_ops.embedding_lookup_v2(self.embeddings, inputs)
    if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
      # Instead of casting the variable as in most layers, cast the output, as
      # this is mathematically equivalent but is faster.
      out = math_ops.cast(out, self._dtype_policy.compute_dtype)
    return out

  def get_config(self):
    config = {
        'input_dim': self.input_dim,
        'output_dim': self.output_dim,
        'embeddings_initializer':
            initializers.serialize(self.embeddings_initializer),
        'embeddings_regularizer':
            regularizers.serialize(self.embeddings_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'embeddings_constraint':
            constraints.serialize(self.embeddings_constraint),
        'mask_zero': self.mask_zero,
        'input_length': self.input_length
    }
    base_config = super(Embedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
