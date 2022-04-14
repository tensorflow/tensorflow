import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables


class MyModel(tf.Module):
  def __init__(self):
    self.var1 = variables.Variable(
        np.array([[[13.]]], dtype=np.float32), name='var1')
    self.var2 = variables.Variable(
        np.array([[[37.]]], dtype=np.float32), name='var2')

  @tf.function
  def __call__(self, input1, input2):
    mul1 = input1 * self.var1
    mul2 = input2 * self.var2
    add = mul1 + mul2
    sub = add - 45.
    return array_ops.identity(sub, name="output")


def GenerateModelWithReadVariableOp(tf_saved_model_dir):
  """Generate a model with ReadVariableOp nodes."""
  my_model = MyModel()
  cfunc = my_model.__call__.get_concrete_function(
      tf.TensorSpec([None, 1, 1], tf.float32),
      tf.TensorSpec([None, 1, 1], tf.float32))
  tf.saved_model.save(my_model, tf_saved_model_dir, signatures=cfunc)


if __name__ == "__main__":
  GenerateModelWithReadVariableOp(
      tf_saved_model_dir="tf_readvariableop_saved_model")
