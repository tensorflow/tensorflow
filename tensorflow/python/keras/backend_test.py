from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import test_combinations
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class BinaryCrossentropyEagerGraphConsistencyTest(test_combinations.TestCase):

  def test_binary_crossentropy_eager_graph_consistency(self):
    x = ops.convert_to_tensor([[1449.6967]])
    y = ops.convert_to_tensor([[0.0]])

    eager_loss = keras.backend.binary_crossentropy(y, math_ops.sigmoid(x), from_logits=False)

    @def_function.function
    def graph_loss_fn():
      return keras.backend.binary_crossentropy(y, math_ops.sigmoid(x), from_logits=False)

    graph_loss = graph_loss_fn()
    self.assertAllClose(eager_loss, graph_loss)

if __name__ == '__main__':
  test.main()
