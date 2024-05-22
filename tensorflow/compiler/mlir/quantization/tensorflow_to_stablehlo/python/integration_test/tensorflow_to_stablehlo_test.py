# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import tempfile
from mlir import ir
from mlir.dialects import stablehlo
import tensorflow as tf
from tensorflow.compiler.mlir.quantization.tensorflow_to_stablehlo.python import pywrap_tensorflow_to_stablehlo as tensorflow_to_stablehlo
from tensorflow.python.platform import test


def build_savedmodel(tempdir) -> str:

  class AddOneModel(tf.keras.Model):

    def call(self, x):
      return x + 1

  model = AddOneModel()

  x_train = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
  y_train = tf.constant([2, 3, 4, 5, 6], dtype=tf.float32)

  model.compile(optimizer='sgd', loss='mse')
  model.fit(x_train, y_train, epochs=1)

  path = tempdir + '/add_one_model'
  model.save(path)
  return path


class TensorflowToStableHLOTest(test.TestCase):

  def test_saved_model_to_stablehlo(self):
    with tempfile.TemporaryDirectory() as tempdir:
      path = build_savedmodel(tempdir)
      module_bytecode = tensorflow_to_stablehlo.savedmodel_to_stablehlo(
          input_path=path, input_arg_shapes_str='4'
      )
      with ir.Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = ir.Module.parse(module_bytecode)
        self.assertIn('stablehlo.add %arg0, %cst : tensor<4xf32>', str(module))

  def test_tf_mlir_to_stablehlo(self):
    assembly = """
      module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
        func.func @main(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
          %cst = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
          %0 = "tf.Add"(%arg0, %cst): (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
        }
      }
    """
    module_bytecode = tensorflow_to_stablehlo.tensorflow_module_to_stablehlo(
        module=assembly,
        input_arg_shapes_str='4',
    )
    with ir.Context() as ctx:
      stablehlo.register_dialect(ctx)
      module = ir.Module.parse(module_bytecode)
      self.assertIn('stablehlo.add %arg0, %cst : tensor<4xf32>', str(module))


if __name__ == '__main__':
  test.main()
