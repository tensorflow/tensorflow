# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Module implementing for RNN wrappers for TF v2."""

# Note that all the APIs under this module are exported as tf.nn.*. This is due
# to the fact that those APIs were from tf.nn.rnn_cell_impl. They are ported
# here to avoid the cyclic dependency issue for serialization. These APIs will
# probably be deprecated and removed in future since similar API is available in
# existing Keras RNN API.


from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util.tf_export import tf_export


class _RNNCellWrapperV2(recurrent.AbstractRNNCell):
  """Base class for cells wrappers V2 compatibility.

  This class along with `rnn_cell_impl._RNNCellWrapperV1` allows to define
  wrappers that are compatible with V1 and V2, and defines helper methods for
  this purpose.
  """

  def __init__(self, cell, *args, **kwargs):
    super(_RNNCellWrapperV2, self).__init__(*args, **kwargs)
    self.cell = cell
    cell_call_spec = tf_inspect.getfullargspec(cell.call)
    self._expects_training_arg = ("training" in cell_call_spec.args) or (
        cell_call_spec.varkw is not None
    )

  def call(self, inputs, state, **kwargs):
    """Runs the RNN cell step computation.

    When `call` is being used, we assume that the wrapper object has been built,
    and therefore the wrapped cells has been built via its `build` method and
    its `call` method can be used directly.

    This allows to use the wrapped cell and the non-wrapped cell equivalently
    when using `call` and `build`.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      **kwargs: Additional arguments passed to the wrapped cell's `call`.

    Returns:
      A pair containing:

      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """
    return self._call_wrapped_cell(
        inputs, state, cell_call_fn=self.cell.call, **kwargs)

  def build(self, inputs_shape):
    """Builds the wrapped cell."""
    self.cell.build(inputs_shape)
    self.built = True

  def get_config(self):
    config = {
        "cell": {
            "class_name": self.cell.__class__.__name__,
            "config": self.cell.get_config()
        },
    }
    base_config = super(_RNNCellWrapperV2, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    from tensorflow.python.keras.layers.serialization import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cell = deserialize_layer(config.pop("cell"), custom_objects=custom_objects)
    return cls(cell, **config)


@tf_export("nn.RNNCellDropoutWrapper", v1=[])
class DropoutWrapper(rnn_cell_wrapper_impl.DropoutWrapperBase,
                     _RNNCellWrapperV2):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super(DropoutWrapper, self).__init__(*args, **kwargs)
    if isinstance(self.cell, recurrent.LSTMCell):
      raise ValueError("keras LSTM cell does not work with DropoutWrapper. "
                       "Please use LSTMCell(dropout=x, recurrent_dropout=y) "
                       "instead.")

  __init__.__doc__ = rnn_cell_wrapper_impl.DropoutWrapperBase.__init__.__doc__


@tf_export("nn.RNNCellResidualWrapper", v1=[])
class ResidualWrapper(rnn_cell_wrapper_impl.ResidualWrapperBase,
                      _RNNCellWrapperV2):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super(ResidualWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = rnn_cell_wrapper_impl.ResidualWrapperBase.__init__.__doc__


@tf_export("nn.RNNCellDeviceWrapper", v1=[])
class DeviceWrapper(rnn_cell_wrapper_impl.DeviceWrapperBase,
                    _RNNCellWrapperV2):
  """Operator that ensures an RNNCell runs on a particular device."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super(DeviceWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = rnn_cell_wrapper_impl.DeviceWrapperBase.__init__.__doc__
