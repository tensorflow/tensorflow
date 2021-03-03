# Lint as: python2, python3
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
"""Tests for util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow.lite.python import util
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# TODO(nupurgarg): Add test for Grappler and frozen graph related functions.
class UtilTest(test_util.TensorFlowTestCase):

  def testConvertEnumToDtype(self):
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(0), dtypes.float32)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(1), dtypes.float16)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(2), dtypes.int32)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(3), dtypes.uint8)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(4), dtypes.int64)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(5), dtypes.string)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(6), dtypes.bool)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(7), dtypes.int16)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(8), dtypes.complex64)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(9), dtypes.int8)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(10), dtypes.float64)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(11), dtypes.complex128)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(16), dtypes.uint32)
    with self.assertRaises(ValueError) as error:
      util._convert_tflite_enum_type_to_tf_type(20)
    self.assertEqual(
        "Unsupported enum 20. The valid map of enum to tf types is : "
        "{0: tf.float32, 1: tf.float16, 2: tf.int32, 3: tf.uint8, 4: tf.int64, "
        "5: tf.string, 6: tf.bool, 7: tf.int16, 8: tf.complex64, 9: tf.int8, "
        "10: tf.float64, 11: tf.complex128, 16: tf.uint32}",
        str(error.exception))

  def testTensorName(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[4])
      out_tensors = array_ops.split(
          value=in_tensor, num_or_size_splits=[1, 1, 1, 1], axis=0)

    expect_names = ["split", "split:1", "split:2", "split:3"]
    for i in range(len(expect_names)):
      got_name = util.get_tensor_name(out_tensors[i])
      self.assertEqual(got_name, expect_names[i])

  def testUint32PassThrough(self):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,), dtype=tf.uint32),
        tf.keras.layers.Reshape(target_shape=(2, 2))
    ])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    self.assertEqual(input_details["dtype"], np.uint32)
    self.assertEqual(output_details["dtype"], np.uint32)

    in_array = np.array([[1, 1, 1, 1]], dtype="uint32") * ((1 << 32) - 1)
    expected_out = np.reshape(in_array, (2, 2))

    interpreter.set_tensor(input_details["index"], in_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details["index"])[0]
    self.assertAllEqual(expected_out, output_data)

  @test_util.enable_control_flow_v2
  def testRemoveLowerUsingSwitchMerge(self):
    with ops.Graph().as_default():
      i = array_ops.placeholder(dtype=dtypes.int32, shape=())
      c = lambda i: math_ops.less(i, 10)
      b = lambda i: math_ops.add(i, 1)
      control_flow_ops.while_loop(c, b, [i])
      sess = session.Session()

    new_graph_def = convert_to_constants.disable_lower_using_switch_merge(
        sess.graph_def)
    lower_using_switch_merge_is_removed = False
    for node in new_graph_def.node:
      if node.op == "While" or node.op == "StatelessWhile":
        if not node.attr["_lower_using_switch_merge"].b:
          lower_using_switch_merge_is_removed = True
    self.assertTrue(lower_using_switch_merge_is_removed)

  def testConvertBytes(self):
    source, header = util.convert_bytes_to_c_source(
        b"\x00\x01\x02\x23", "foo", 16, use_tensorflow_license=False)
    self.assertTrue(
        source.find("const unsigned char foo[] DATA_ALIGN_ATTRIBUTE = {"))
    self.assertTrue(source.find("""    0x00, 0x01,
    0x02, 0x23,"""))
    self.assertNotEqual(-1, source.find("const int foo_len = 4;"))
    self.assertEqual(-1, source.find("/* Copyright"))
    self.assertEqual(-1, source.find("#include " ""))
    self.assertNotEqual(-1, header.find("extern const unsigned char foo[];"))
    self.assertNotEqual(-1, header.find("extern const int foo_len;"))
    self.assertEqual(-1, header.find("/* Copyright"))

    source, header = util.convert_bytes_to_c_source(
        b"\xff\xfe\xfd\xfc",
        "bar",
        80,
        include_guard="MY_GUARD",
        include_path="my/guard.h",
        use_tensorflow_license=True)
    self.assertNotEqual(
        -1, source.find("const unsigned char bar[] DATA_ALIGN_ATTRIBUTE = {"))
    self.assertNotEqual(-1, source.find("""    0xff, 0xfe, 0xfd, 0xfc,"""))
    self.assertNotEqual(-1, source.find("/* Copyright"))
    self.assertNotEqual(-1, source.find("#include \"my/guard.h\""))
    self.assertNotEqual(-1, header.find("#ifndef MY_GUARD"))
    self.assertNotEqual(-1, header.find("#define MY_GUARD"))
    self.assertNotEqual(-1, header.find("/* Copyright"))


class TensorFunctionsTest(test_util.TensorFlowTestCase):

  def testGetTensorsValid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1, 16, 16, 3])
      _ = in_tensor + in_tensor
      sess = session.Session()

    tensors = util.get_tensors_from_tensor_names(sess.graph, ["Placeholder"])
    self.assertEqual("Placeholder:0", tensors[0].name)

  def testGetTensorsInvalid(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1, 16, 16, 3])
      _ = in_tensor + in_tensor
      sess = session.Session()

    with self.assertRaises(ValueError) as error:
      util.get_tensors_from_tensor_names(sess.graph, ["invalid-input"])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

  def testSetTensorShapeValid(self):
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    util.set_tensor_shapes([tensor], {"Placeholder": [5, 3, 5]})
    self.assertAllEqual([5, 3, 5], tensor.shape)

  def testSetTensorShapeNoneValid(self):
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32)

    util.set_tensor_shapes([tensor], {"Placeholder": [1, 3, 5]})
    self.assertAllEqual([1, 3, 5], tensor.shape)

  def testSetTensorShapeArrayInvalid(self):
    # Tests set_tensor_shape where the tensor name passed in doesn't exist.
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    with self.assertRaises(ValueError) as error:
      util.set_tensor_shapes([tensor], {"invalid-input": [5, 3, 5]})
    self.assertEqual(
        "Invalid tensor 'invalid-input' found in tensor shapes map.",
        str(error.exception))
    self.assertAllEqual([None, 3, 5], tensor.shape)

  def testSetTensorShapeDimensionInvalid(self):
    # Tests set_tensor_shape where the shape passed in is incompatible.
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    with self.assertRaises(ValueError) as error:
      util.set_tensor_shapes([tensor], {"Placeholder": [1, 5, 5]})
    self.assertIn("The shape of tensor 'Placeholder' cannot be changed",
                  str(error.exception))
    self.assertAllEqual([None, 3, 5], tensor.shape)

  def testSetTensorShapeEmpty(self):
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    util.set_tensor_shapes([tensor], {})
    self.assertAllEqual([None, 3, 5], tensor.shape)


def _generate_integer_tflite_model(quantization_type=dtypes.int8):
  """Define an integer post-training quantized tflite model."""
  # Define a pseudo MNIST dataset (as downloading the dataset on-the-fly causes
  # network connection failures)
  n = 10  # Number of samples
  images = np.random.randint(low=0, high=255, size=[n, 28, 28], dtype=np.uint8)
  labels = np.random.randint(low=0, high=9, size=(n,), dtype=np.uint8)

  # Normalize the input image so that each pixel value is between 0 to 1.
  images = images / 255.0

  # Define TF model
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)
  ])

  # Train
  model.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"])

  model.fit(
      images,
      labels,
      epochs=1,
      validation_split=0.1,
  )

  # Convert TF Model to an Integer Quantized TFLite Model
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = {tf.lite.Optimize.DEFAULT}
  def representative_dataset_gen():
    for _ in range(2):
      yield [
          np.random.uniform(low=0, high=1, size=(1, 28, 28)).astype(
              np.float32)
      ]
  converter.representative_dataset = representative_dataset_gen
  if quantization_type == dtypes.int8:
    converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
  else:
    converter.target_spec.supported_ops = {
        tf.lite.OpsSet
        .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    }
  tflite_model = converter.convert()

  return tflite_model


def _test_param_modify_integer_model_io_type():
  """Function to generate parameterized inputs for testing."""
  params = []
  str_template = "_{}{}{}{}"
  map_model_type = {
      "PostTraining": True,
      # "DuringTraining": False,
  }
  map_quantize_type_to_io_types = {
      tf.int8: {tf.float32, tf.int8, tf.uint8},
      tf.int16: {tf.float32, tf.int16}
  }
  for k1, v1 in map_model_type.items():
    for qtype, v2 in map_quantize_type_to_io_types.items():
      qstr = "_IntegerQuantize{}".format(qtype.name.capitalize())
      for itype in v2:
        istr = "_Input{}".format(itype.name.capitalize())
        for otype in v2:
          ostr = "_Output{}".format(otype.name.capitalize())
          params.append((str_template.format(k1, qstr, istr, ostr),
                         v1, qtype, itype, otype))
  return params


# TODO(b/161174063):  Merge tests for integer input/output type
class UtilModifyIntegerQuantizedModelIOTypeTest(
    test_util.TensorFlowTestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(UtilModifyIntegerQuantizedModelIOTypeTest, cls).setUpClass()
    cls.post_train_int8_model = _generate_integer_tflite_model()
    cls.post_train_int16_model = _generate_integer_tflite_model(
        quantization_type=dtypes.int16)

  @parameterized.named_parameters(_test_param_modify_integer_model_io_type())
  def test(self, is_post_train, quantization_type, in_tftype, out_tftype):
    """Modify the float input/output type of an integer quantized model."""

    def _run_tflite_inference(model, in_tftype, out_tftype):
      """Run inference on a model with a specific input/output type."""
      # Load TFLite model and allocate tensors.
      interpreter = tf.lite.Interpreter(model_content=model)
      interpreter.allocate_tensors()
      input_details = interpreter.get_input_details()[0]
      output_details = interpreter.get_output_details()[0]

      # Validate TFLite model input and output types
      self.assertEqual(input_details["dtype"], in_tftype.as_numpy_dtype)
      self.assertEqual(output_details["dtype"], out_tftype.as_numpy_dtype)

      # Define Input
      np.random.seed(0)
      input_data = np.random.uniform(low=0, high=1, size=(1, 28, 28))
      input_data = input_data.astype(np.float32)
      if input_details["dtype"] != np.float32:
        # quantize float to int
        scale, zero_point = input_details["quantization"]
        input_data = input_data / scale + zero_point
        input_data = input_data.astype(input_details["dtype"])

      # Run Inference
      interpreter.set_tensor(input_details["index"], input_data)
      interpreter.invoke()

      # Get output
      output_data = interpreter.get_tensor(output_details["index"])[0]
      if output_details["dtype"] != np.float32:
        # dequantize int to float
        scale, zero_point = output_details["quantization"]
        output_data = output_data.astype(np.float32)
        output_data = (output_data - zero_point) * scale

      return output_data

    if is_post_train and quantization_type == tf.int8:
      model = self.__class__.post_train_int8_model
    elif is_post_train and quantization_type == tf.int16:
      model = self.__class__.post_train_int16_model
    else:
      model = None
    # Run model inference with float input output type
    output_data = _run_tflite_inference(model, tf.float32, tf.float32)
    # Modify the model io types to the target input/output types.
    model_io = util.modify_model_io_type(model, in_tftype, out_tftype)
    # Run model inference with modified integer input output type
    output_io_data = _run_tflite_inference(model_io, in_tftype, out_tftype)
    # Validate that both the outputs are the same
    self.assertAllClose(output_data, output_io_data, atol=1.0)

    # Modify the model with the target input/output types should be a no op.
    model_io = util.modify_model_io_type(model_io, in_tftype, out_tftype)
    # Run model inference with modified integer input output type
    output_io_data = _run_tflite_inference(model_io, in_tftype, out_tftype)
    # Validate that both the outputs are the same
    self.assertAllClose(output_data, output_io_data, atol=1.0)


if __name__ == "__main__":
  test.main()
