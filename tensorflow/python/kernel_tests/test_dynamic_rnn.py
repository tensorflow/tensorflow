from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class DummyMultiDimensionalLSTM(tf.nn.rnn_cell.RNNCell):
  """LSTM Cell generating (output, new_state) = (input + 1, state + 1).

  The input to this cell may have an arbitrary number of dimensions that follow
  the preceding 'Time' and 'Batch' dimensions.
  """

  def __init__(self, dims):
    """Initialize the Multi-dimensional LSTM cell.

    Args:
      dims: tuple that contains the dimensions of the output of the cell,
      without including 'Time' or 'Batch' dimensions.
    """
    if not isinstance(dims, tuple):
      raise TypeError("The dimensions passed to DummyMultiDimensionalLSTM"
                      "should be a tuple of ints.")
    self._dims = dims
    self._output_size = tf.TensorShape(self._dims)
    self._state_size = (tf.TensorShape(self._dims), tf.TensorShape(self._dims))

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, input_, state, scope=None):
    h, c = state
    return (input_ + 1, (h + 1, c + 1))


class DynamicRNNInputND(tf.test.TestCase):
  """
  Check that dynamic_rnn works for tensors of rank>3
  when time_major=False.
  Both, when the rank is known at the graph construction
  time as well as when it is not.

  Note: the following tests only work if the shape-checks in rnn.py/dynamic_rnn and
  rnn.py/_dynamic_rnn_loop are disabled.
  """
  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testCore(self,input_shape,units=3,use_placeholder=False,time_major=False):
    with self.test_session(graph=tf.Graph()) as sess:
      cell = DummyMultiDimensionalLSTM(input_shape[2:])

      inputs_v = np.random.randn(*input_shape).astype(np.float32)
      if use_placeholder:
        inputs = tf.placeholder(tf.float32)
      else:
        inputs = tf.constant(inputs_v,dtype=tf.float32)
      outputs, state = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32,
                                          time_major=time_major)
      tf.initialize_all_variables().run()
      if use_placeholder:
        outputs_v = sess.run(outputs,feed_dict={inputs:inputs_v})
      else:
        outputs_v = outputs.eval()
      self.assertAllClose(outputs_v,inputs_v+1)

  def _testLoop(self,input_shape):
    for use_placeholder in [True,False]:
      for time_major in [True,False]:
        self._testCore(input_shape,use_placeholder=use_placeholder,
                       time_major=time_major)

  def test_main(self):
    # test on 3D inputs:
    input_shape = (3,4,5)
    self._testLoop(input_shape)
    # test on 4D inputs:
    input_shape = (3,4,5,6)
    self._testLoop(input_shape)
    # test on 5D inputs:
    input_shape = (3,4,5,6,7)
    self._testLoop(input_shape)


if __name__ == "__main__":
  tf.test.main()
