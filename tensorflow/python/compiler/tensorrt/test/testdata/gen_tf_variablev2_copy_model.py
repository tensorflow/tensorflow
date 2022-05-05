import numpy as np
import tensorflow as tf
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

ops.disable_eager_execution()
variable_scope.disable_resource_variables()

def GenerateModelWithVariableV2_1Var2Engines(tf_saved_model_dir):
  """Generate a model to test 1 VariableV2 node used in 2 engines."""

  def SimpleModel():
    """Define model with a TF graph."""

    def GraphFn():
      input1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 32, 32, 32, 3], name="input1")
      var1 = variable_scope.get_variable(
          'var1', shape=[3, 3, 3, 3, 3],
          initializer=tf.constant_initializer(np.reshape(
              np.r_[:243].astype(np.float32), (3, 3, 3, 3, 3))),
          dtype=tf.float32)
      conv0 = nn.conv3d(
          input=input1,
          filter=var1,
          strides=[1, 1, 1, 1, 1],
          padding="SAME",
          name="conv0")
      bias0 = constant_op.constant([1., 2., 3.], name="bias0")
      added0 = nn.bias_add(conv0, bias0, name="bias_add0")
      relu0 = nn.relu(added0, "relu0")
      incompatible0 = math_ops.erfc(relu0, name="incompatible0")
      conv1 = nn.conv3d(
          input=incompatible0,
          filter=var1,
          strides=[1, 1, 1, 1, 1],
          padding="SAME",
          name="conv0")
      bias1 = constant_op.constant([4., 5., 6.], name="bias1")
      added1 = nn.bias_add(conv1, bias1, name="bias_add1")
      relu1 = nn.relu(added1, "relu1")
      out = array_ops.identity(relu1, name="output")
      return g, input1, out

    g = ops.Graph()
    with g.as_default():
      return GraphFn()

  g, input1, out = SimpleModel()
  signature_def = signature_def_utils.build_signature_def(
      inputs={
          "input1": utils.build_tensor_info(input1)
      },
      outputs={"output": utils.build_tensor_info(out)},
      method_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
  saved_model_builder = builder.SavedModelBuilder(tf_saved_model_dir)
  with session.Session(graph=g) as sess:
    variables.global_variables_initializer().run()
    saved_model_builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        })
  saved_model_builder.save()

if __name__ == "__main__":
  GenerateModelWithVariableV2_1Var2Engines(
      tf_saved_model_dir="tf_variablev2_copy_model")
