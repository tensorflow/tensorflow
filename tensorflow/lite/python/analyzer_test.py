# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for analyzer package."""

from tensorflow.lite.python import analyzer
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class ConvertTest(test_util.TensorFlowTestCase):

  def testTxt(self):
    model_path = resource_loader.get_path_to_datafile(
        "testdata/permute_float.tflite")
    txt = analyzer.ModelAnalyzer.analyze(model_path, "txt")
    self.assertIn("Subgraph#0 main(T#0) -> [T#2]", txt)
    self.assertIn("Op#0 FULLY_CONNECTED(T#0, T#1, T#-1) -> [T#2]", txt)

  def testHtml(self):
    model_path = resource_loader.get_path_to_datafile(
        "testdata/permute_float.tflite")
    html = analyzer.ModelAnalyzer.analyze(model_path, "html")
    self.assertIn("<html>\n<head>", html)
    self.assertIn("FULLY_CONNECTED (0)", html)

  def testMlir(self):
    model_path = resource_loader.get_path_to_datafile(
        "testdata/permute_float.tflite")
    mlir = analyzer.ModelAnalyzer.analyze(model_path, "mlir")
    self.assertIn(
        "func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> attributes "
        '{tf.entry_function = {inputs = "input", outputs = "output"}}', mlir)
    self.assertIn(
        '%1 = "tfl.fully_connected"(%arg0, %0, %cst) {fused_activation_function'
        ' = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : '
        "(tensor<1x4xf32>, tensor<4x4xf32>, none) -> tensor<1x4xf32>", mlir)

  def testMlirHugeConst(self):
    model_path = resource_loader.get_path_to_datafile(
        "../testdata/conv_huge_im2col.bin")
    mlir = analyzer.ModelAnalyzer.analyze(model_path, "mlir")
    self.assertIn(
        '%1 = "tfl.pseudo_const"() {value = opaque<"_", "0xDEADBEEF"> : '
        "tensor<3x3x3x8xf32>} : () -> tensor<3x3x3x8xf32>", mlir)


if __name__ == "__main__":
  test.main()
