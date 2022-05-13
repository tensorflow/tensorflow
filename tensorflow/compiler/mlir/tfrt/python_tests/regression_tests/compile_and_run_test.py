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
"""Tests for Tensorflow -> jitrt compilation."""

import os
import time

from absl import flags
from mlir import ir
import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.compiler.mlir.tfrt.jit.python_binding import tfrt_fallback
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

_COMPARE_WITH_TENSORFLOW = flags.DEFINE_boolean(
    'compare_with_tensorflow', None,
    'Whether the results should be compared to Tensorflow')
_INPUT_DATA_SEED = flags.DEFINE_integer(
    'input_data_seed', None, 'The random seed to be used for initializing.')
_ONE_SHOT_BUFFERIZE = flags.DEFINE_boolean(
    'one_shot_bufferize', None,
    'Whether one-shot bufferization should be enabled')
_TEST_FILE_NAME = flags.DEFINE_string(
    'test_file_name', None,
    'The filename of the file containing the MLIR IR that should be tested')
_VECTORIZE = flags.DEFINE_boolean('vectorize', None,
                                  'Whether vectorization should be enabled')

jitrt = tf_jitrt.TfJitRtExecutor()


_STATIC_TYPE_ATTRIBUTE_NAME = 'python_test_attrs.static_type'
_SHAPE_VALUE_ATTRIBUTE_NAME = 'python_test_attrs.shape_value'
_ARG_ATTRIBUTES_NAME = 'arg_attrs'


class CompileAndRunTest(test.TestCase):

  @staticmethod
  def mlir_type_to_np_type(mlir_type: ir.Type):
    if ir.IntegerType.isinstance(mlir_type):
      mlir_type = ir.IntegerType(mlir_type)
      if mlir_type.width == 1:
        return bool
      if mlir_type.width == 8:
        if mlir_type.is_unsigned:
          return np.uint8
        return np.int8
      if mlir_type.width == 16:
        if mlir_type.is_unsigned:
          return np.uint16
        return np.int16
      if mlir_type.width == 32:
        if mlir_type.is_unsigned:
          return np.uint32
        return np.int32
      if mlir_type.width == 64:
        if mlir_type.is_unsigned:
          return np.uint64
        return np.int64
    if ir.F16Type.isinstance(mlir_type):
      return np.float16
    if ir.F32Type.isinstance(mlir_type):
      return np.float32
    if ir.F64Type.isinstance(mlir_type):
      return np.float64
    raise Exception(f'unknown scalar type: {mlir_type}')

  def test_compile_and_run(self):
    filename = _TEST_FILE_NAME.value
    if not os.path.isabs(filename):
      filename = os.path.join(resource_loader.get_data_files_path(), filename)
    with gfile.GFile(filename, mode='r') as f:
      mlir_function = f.read()
      arg_attrs = []
      with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(mlir_function)
        func = module.body.operations[0]
        function_name = ir.StringAttr(func.attributes['sym_name']).value
        # If the function has arguments, we expect argument attributes.
        if func.regions[0].blocks[0].arguments:
          self.assertIn(_ARG_ATTRIBUTES_NAME, func.attributes)
          arg_attrs = ir.ArrayAttr(func.attributes[_ARG_ATTRIBUTES_NAME])
      logging.info(f'processing {filename}')
      start = time.perf_counter()
      compiled = jitrt.compile(
          mlir_function,
          function_name,
          tf_jitrt.Specialization.ENABLED,
          vectorize=_VECTORIZE.value,
          one_shot_bufferize=_ONE_SHOT_BUFFERIZE.value)
      end = time.perf_counter()
      logging.info(f'compiled {filename} in {end-start:0.4f} seconds')
      np.random.seed(_INPUT_DATA_SEED.value)
      args = []
      for arg_attr in arg_attrs:
        attr_dict = ir.DictAttr(arg_attr)
        if _SHAPE_VALUE_ATTRIBUTE_NAME in attr_dict:
          shape_value_attr = ir.DenseIntElementsAttr(
              attr_dict[_SHAPE_VALUE_ATTRIBUTE_NAME])
          shape_value = np.array(list(shape_value_attr)).astype(np.int32)
          args.append(shape_value)
        elif _STATIC_TYPE_ATTRIBUTE_NAME in attr_dict:
          static_type = ir.TypeAttr(
              attr_dict[_STATIC_TYPE_ATTRIBUTE_NAME]).value
          shaped_type = ir.ShapedType(static_type)
          np_element_type = CompileAndRunTest.mlir_type_to_np_type(
              shaped_type.element_type)
          arg = np.random.uniform(
              -10000.0, 10000.0, size=shaped_type.shape).astype(np_element_type)
          args.append(arg)
      self.assertEqual(len(args), len(arg_attrs))
      start = time.perf_counter()
      result = jitrt.execute(compiled, args)
      end = time.perf_counter()
      logging.info(f'executed {filename} in {end-start:0.4f} seconds')
      if _COMPARE_WITH_TENSORFLOW.value:
        start = time.perf_counter()
        expected = tfrt_fallback.run_tfrt_fallback(mlir_function, function_name,
                                                   args)
        end = time.perf_counter()
        logging.info(
            f'executed {filename} via tfrt fallback in {end-start:0.4f} seconds'
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
  flags.mark_flag_as_required('compare_with_tensorflow')
  flags.mark_flag_as_required('input_data_seed')
  flags.mark_flag_as_required('one_shot_bufferize')
  flags.mark_flag_as_required('test_file_name')
  flags.mark_flag_as_required('vectorize')
  test.main()
