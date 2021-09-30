# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SavedModelCLI tool."""
import contextlib
import os
import pickle
import platform
import shutil
import sys

from absl.testing import parameterized
import numpy as np
from six import StringIO

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import save
from tensorflow.python.tools import saved_model_cli
from tensorflow.python.training.tracking import tracking

SAVED_MODEL_PATH = ('cc/saved_model/testdata/half_plus_two/00000123')


@contextlib.contextmanager
def captured_output():
  new_out, new_err = StringIO(), StringIO()
  old_out, old_err = sys.stdout, sys.stderr
  try:
    sys.stdout, sys.stderr = new_out, new_err
    yield sys.stdout, sys.stderr
  finally:
    sys.stdout, sys.stderr = old_out, old_err


class SavedModelCLITestCase(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SavedModelCLITestCase, self).setUp()
    if platform.system() == 'Windows':
      self.skipTest('Skipping failing tests on Windows.')

  def testShowCommandAll(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args(['show', '--dir', base_path, '--all'])
    with captured_output() as (out, err):
      saved_model_cli.show(args)
    output = out.getvalue().strip()
    # pylint: disable=line-too-long
    exp_out = """MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/classify

signature_def['classify_x_to_y']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_example:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/classify

signature_def['regress_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/regress

signature_def['regress_x_to_y']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_example:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/regress

signature_def['regress_x_to_y2']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_example:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y2:0
  Method name is: tensorflow/serving/regress

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/predict"""
    # pylint: enable=line-too-long
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowAllWithFunctions(self):

    class DummyModel(tracking.AutoTrackable):
      """Model with callable polymorphic functions specified."""

      @def_function.function
      def func1(self, a, b, c):
        if c:
          return a + b
        else:
          return a * b

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32)
      ])
      def func2(self, x):
        return x + 2

      @def_function.function
      def __call__(self, y, c=7):
        return y + 2 * c

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    # Call with specific values to create new polymorphic function traces.
    dummy_model.func1(constant_op.constant(5), constant_op.constant(9), True)
    dummy_model(constant_op.constant(5))
    save.save(dummy_model, saved_model_dir)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args(['show', '--dir', saved_model_dir, '--all'])
    with captured_output() as (out, err):
      saved_model_cli.show(args)
    output = out.getvalue().strip()
    exp_out = """MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (2, 2)
        name: serving_default_x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: (2, 2)
        name: PartitionedCall:0
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          y: TensorSpec(shape=(), dtype=tf.int32, name='y')
        Argument #2
          DType: int
          Value: 7

  Function Name: 'func1'
    Option #1
      Callable with:
        Argument #1
          a: TensorSpec(shape=(), dtype=tf.int32, name='a')
        Argument #2
          b: TensorSpec(shape=(), dtype=tf.int32, name='b')
        Argument #3
          DType: bool
          Value: True

  Function Name: 'func2'
    Option #1
      Callable with:
        Argument #1
          x: TensorSpec(shape=(2, 2), dtype=tf.float32, name='x')
""".strip()  # pylint: enable=line-too-long
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowAllWithPureConcreteFunction(self):

    class DummyModel(tracking.AutoTrackable):
      """Model with a callable concrete function."""

      def __init__(self):
        function = def_function.function(
            self.multiply,
            input_signature=[
                tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
                tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32)
            ])
        self.pure_concrete_function = function.get_concrete_function()
        super(DummyModel, self).__init__()

      def multiply(self, a, b):
        return a * b

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    save.save(dummy_model, saved_model_dir)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args(['show', '--dir', saved_model_dir, '--all'])
    with captured_output() as (out, err):
      saved_model_cli.show(args)
    output = out.getvalue().strip()
    exp_out = """MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['a'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_a:0
    inputs['b'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_b:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: PartitionedCall:0
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: 'pure_concrete_function'
    Option #1
      Callable with:
        Argument #1
          a: TensorSpec(shape=(), dtype=tf.float32, name='a')
        Argument #2
          b: TensorSpec(shape=(), dtype=tf.float32, name='b')
""".strip()  # pylint: enable=line-too-long
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandTags(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args(['show', '--dir', base_path])
    with captured_output() as (out, err):
      saved_model_cli.show(args)
    output = out.getvalue().strip()
    exp_out = 'The given SavedModel contains the following tag-sets:\n\'serve\''
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandSignature(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args(
        ['show', '--dir', base_path, '--tag_set', 'serve'])
    with captured_output() as (out, err):
      saved_model_cli.show(args)
    output = out.getvalue().strip()
    exp_header = ('The given SavedModel MetaGraphDef contains SignatureDefs '
                  'with the following keys:')
    exp_start = 'SignatureDef key: '
    exp_keys = [
        '"classify_x2_to_y3"', '"classify_x_to_y"', '"regress_x2_to_y3"',
        '"regress_x_to_y"', '"regress_x_to_y2"', '"serving_default"'
    ]
    # Order of signatures does not matter
    self.assertMultiLineEqual(
        output,
        '\n'.join([exp_header] + [exp_start + exp_key for exp_key in exp_keys]))
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandErrorNoTagSet(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args(
        ['show', '--dir', base_path, '--tag_set', 'badtagset'])
    with self.assertRaises(RuntimeError):
      saved_model_cli.show(args)

  def testShowCommandInputsOutputs(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args([
        'show', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'serving_default'
    ])
    with captured_output() as (out, err):
      saved_model_cli.show(args)
    output = out.getvalue().strip()
    expected_output = (
        'The given SavedModel SignatureDef contains the following input(s):\n'
        '  inputs[\'x\'] tensor_info:\n'
        '      dtype: DT_FLOAT\n      shape: (-1, 1)\n      name: x:0\n'
        'The given SavedModel SignatureDef contains the following output(s):\n'
        '  outputs[\'y\'] tensor_info:\n'
        '      dtype: DT_FLOAT\n      shape: (-1, 1)\n      name: y:0\n'
        'Method name is: tensorflow/serving/predict')
    self.assertEqual(output, expected_output)
    self.assertEqual(err.getvalue().strip(), '')

  def testPrintREFTypeTensor(self):
    ref_tensor_info = meta_graph_pb2.TensorInfo()
    ref_tensor_info.dtype = types_pb2.DT_FLOAT_REF
    with captured_output() as (out, err):
      saved_model_cli._print_tensor_info(ref_tensor_info)
    self.assertTrue('DT_FLOAT_REF' in out.getvalue().strip())
    self.assertEqual(err.getvalue().strip(), '')

  def testInputPreProcessFormats(self):
    input_str = 'input1=/path/file.txt[ab3];input2=file2'
    input_expr_str = 'input3=np.zeros([2,2]);input4=[4,5]'
    input_dict = saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_expr_dict = saved_model_cli.preprocess_input_exprs_arg_string(
        input_expr_str, safe=False)
    self.assertTrue(input_dict['input1'] == ('/path/file.txt', 'ab3'))
    self.assertTrue(input_dict['input2'] == ('file2', None))
    print(input_expr_dict['input3'])
    self.assertAllClose(input_expr_dict['input3'], np.zeros([2, 2]))
    self.assertAllClose(input_expr_dict['input4'], [4, 5])
    self.assertTrue(len(input_dict) == 2)
    self.assertTrue(len(input_expr_dict) == 2)

  def testInputPreProcessExamplesWithStrAndBytes(self):
    input_examples_str = 'inputs=[{"text":["foo"], "bytes":[b"bar"]}]'
    input_dict = saved_model_cli.preprocess_input_examples_arg_string(
        input_examples_str)
    feature = example_pb2.Example.FromString(input_dict['inputs'][0])
    self.assertProtoEquals(
        """
          features {
            feature {
              key: "bytes"
              value {
                bytes_list {
                  value: "bar"
                }
              }
            }
            feature {
              key: "text"
              value {
                bytes_list {
                  value: "foo"
                }
              }
            }
          }
    """, feature)

  def testInputPreprocessExampleWithCodeInjection(self):
    input_examples_str = 'inputs=os.system("echo hacked")'
    with self.assertRaisesRegex(RuntimeError, 'not a valid python literal.'):
      saved_model_cli.preprocess_input_examples_arg_string(input_examples_str)

  def testInputPreProcessFileNames(self):
    input_str = (r'inputx=C:\Program Files\data.npz[v:0];'
                 r'input:0=c:\PROGRA~1\data.npy')
    input_dict = saved_model_cli.preprocess_inputs_arg_string(input_str)
    self.assertTrue(input_dict['inputx'] == (r'C:\Program Files\data.npz',
                                             'v:0'))
    self.assertTrue(input_dict['input:0'] == (r'c:\PROGRA~1\data.npy', None))

  def testInputPreProcessErrorBadFormat(self):
    input_str = 'inputx=file[[v1]v2'
    with self.assertRaises(RuntimeError):
      saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_str = 'inputx:file'
    with self.assertRaises(RuntimeError):
      saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_str = 'inputx:np.zeros((5))'
    with self.assertRaisesRegex(RuntimeError, 'format is incorrect'):
      saved_model_cli.preprocess_input_exprs_arg_string(input_str, safe=False)

  def testInputParserNPY(self):
    x0 = np.array([[1], [2]])
    x1 = np.array(range(6)).reshape(2, 3)
    input0_path = os.path.join(test.get_temp_dir(), 'input0.npy')
    input1_path = os.path.join(test.get_temp_dir(), 'input1.npy')
    np.save(input0_path, x0)
    np.save(input1_path, x1)
    input_str = 'x0=' + input0_path + '[x0];x1=' + input1_path
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, '', '')
    self.assertTrue(np.all(feed_dict['x0'] == x0))
    self.assertTrue(np.all(feed_dict['x1'] == x1))

  def testInputParserNPZ(self):
    x0 = np.array([[1], [2]])
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0)
    input_str = 'x=' + input_path + '[a];y=' + input_path
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, '', '')
    self.assertTrue(np.all(feed_dict['x'] == x0))
    self.assertTrue(np.all(feed_dict['y'] == x0))

  def testInputParserPickle(self):
    pkl0 = {'a': 5, 'b': np.array(range(4))}
    pkl1 = np.array([1])
    pkl2 = np.array([[1], [3]])
    input_path0 = os.path.join(test.get_temp_dir(), 'pickle0.pkl')
    input_path1 = os.path.join(test.get_temp_dir(), 'pickle1.pkl')
    input_path2 = os.path.join(test.get_temp_dir(), 'pickle2.pkl')
    with open(input_path0, 'wb') as f:
      pickle.dump(pkl0, f)
    with open(input_path1, 'wb') as f:
      pickle.dump(pkl1, f)
    with open(input_path2, 'wb') as f:
      pickle.dump(pkl2, f)
    input_str = 'x=' + input_path0 + '[b];y=' + input_path1 + '[c];'
    input_str += 'z=' + input_path2
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, '', '')
    self.assertTrue(np.all(feed_dict['x'] == pkl0['b']))
    self.assertTrue(np.all(feed_dict['y'] == pkl1))
    self.assertTrue(np.all(feed_dict['z'] == pkl2))

  def testInputParserPythonExpression(self):
    x1 = np.ones([2, 10])
    x2 = np.array([[1], [2], [3]])
    x3 = np.mgrid[0:5, 0:5]
    x4 = [[3], [4]]
    input_expr_str = ('x1=np.ones([2,10]);x2=np.array([[1],[2],[3]]);'
                      'x3=np.mgrid[0:5,0:5];x4=[[3],[4]]')
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        '', input_expr_str, '')
    self.assertTrue(np.all(feed_dict['x1'] == x1))
    self.assertTrue(np.all(feed_dict['x2'] == x2))
    self.assertTrue(np.all(feed_dict['x3'] == x3))
    self.assertTrue(np.all(feed_dict['x4'] == x4))

  def testInputParserBoth(self):
    x0 = np.array([[1], [2]])
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0)
    x1 = np.ones([2, 10])
    input_str = 'x0=' + input_path + '[a]'
    input_expr_str = 'x1=np.ones([2,10])'
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, input_expr_str, '')
    self.assertTrue(np.all(feed_dict['x0'] == x0))
    self.assertTrue(np.all(feed_dict['x1'] == x1))

  def testInputParserBothDuplicate(self):
    x0 = np.array([[1], [2]])
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0)
    x1 = np.ones([2, 10])
    input_str = 'x0=' + input_path + '[a]'
    input_expr_str = 'x0=np.ones([2,10])'
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(
        input_str, input_expr_str, '')
    self.assertTrue(np.all(feed_dict['x0'] == x1))

  def testInputParserErrorNoName(self):
    x0 = np.array([[1], [2]])
    x1 = np.array(range(5))
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0, b=x1)
    input_str = 'x=' + input_path
    with self.assertRaises(RuntimeError):
      saved_model_cli.load_inputs_from_input_arg_string(input_str, '', '')

  def testInputParserErrorWrongName(self):
    x0 = np.array([[1], [2]])
    x1 = np.array(range(5))
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0, b=x1)
    input_str = 'x=' + input_path + '[c]'
    with self.assertRaises(RuntimeError):
      saved_model_cli.load_inputs_from_input_arg_string(input_str, '', '')

  def testRunCommandInputExamples(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'regress_x_to_y', '--input_examples',
        'inputs=[{"x":[8.0],"x2":[5.0]}, {"x":[4.0],"x2":[3.0]}]', '--outdir',
        output_dir
    ])
    saved_model_cli.run(args)
    y_actual = np.load(os.path.join(output_dir, 'outputs.npy'))
    y_expected = np.array([[6.0], [4.0]])
    self.assertAllEqual(y_expected, y_actual)

  def testRunCommandExistingOutdir(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    input_path = os.path.join(test.get_temp_dir(), 'testRunCommand_inputs.npz')
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'outputs.npy')
    if os.path.exists(output_file):
      os.remove(output_file)
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'regress_x2_to_y3', '--inputs', 'inputs=' + input_path + '[x0]',
        '--outdir',
        test.get_temp_dir()
    ])
    saved_model_cli.run(args)
    y_actual = np.load(output_file)
    y_expected = np.array([[3.5], [4.0]])
    self.assertAllClose(y_expected, y_actual)

  def testRunCommandNewOutdir(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandNewOutdir_inputs.npz')
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)
    np.savez(input_path, x0=x, x1=x_notused)
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'serving_default', '--inputs', 'x=' + input_path + '[x0]', '--outdir',
        output_dir
    ])
    saved_model_cli.run(args)
    y_actual = np.load(os.path.join(output_dir, 'y.npy'))
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  def testRunCommandOutOverwrite(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandOutOverwrite_inputs.npz')
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'y.npy')
    open(output_file, 'a').close()
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'serving_default', '--inputs', 'x=' + input_path + '[x0]', '--outdir',
        test.get_temp_dir(), '--overwrite'
    ])
    saved_model_cli.run(args)
    y_actual = np.load(output_file)
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  def testRunCommandInvalidInputKeyError(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'regress_x2_to_y3', '--input_exprs', 'x2=np.ones((3,1))'
    ])
    with self.assertRaises(ValueError):
      saved_model_cli.run(args)

  def testRunCommandInvalidSignature(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'INVALID_SIGNATURE', '--input_exprs', 'x2=np.ones((3,1))'
    ])
    with self.assertRaisesRegex(ValueError,
                                'Could not find signature "INVALID_SIGNATURE"'):
      saved_model_cli.run(args)

  def testRunCommandInputExamplesNotListError(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'regress_x_to_y', '--input_examples', 'inputs={"x":8.0,"x2":5.0}',
        '--outdir', output_dir
    ])
    with self.assertRaisesRegex(ValueError, 'must be a list'):
      saved_model_cli.run(args)

  def testRunCommandInputExamplesFeatureValueNotListError(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'regress_x_to_y', '--input_examples', 'inputs=[{"x":8.0,"x2":5.0}]',
        '--outdir', output_dir
    ])
    with self.assertRaisesRegex(ValueError, 'feature value must be a list'):
      saved_model_cli.run(args)

  def testRunCommandInputExamplesFeatureBadType(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'regress_x_to_y', '--input_examples', 'inputs=[{"x":[[1],[2]]}]',
        '--outdir', output_dir
    ])
    with self.assertRaisesRegex(ValueError, 'is not supported'):
      saved_model_cli.run(args)

  def testRunCommandOutputFileExistError(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandOutOverwrite_inputs.npz')
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'y.npy')
    open(output_file, 'a').close()
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'serving_default', '--inputs', 'x=' + input_path + '[x0]', '--outdir',
        test.get_temp_dir()
    ])
    with self.assertRaises(RuntimeError):
      saved_model_cli.run(args)

  def testRunCommandInputNotGivenError(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'serving_default'
    ])
    with self.assertRaises(AttributeError):
      saved_model_cli.run(args)

  def testRunCommandWithDebuggerEnabled(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandNewOutdir_inputs.npz')
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)
    np.savez(input_path, x0=x, x1=x_notused)
    args = self.parser.parse_args([
        'run', '--dir', base_path, '--tag_set', 'serve', '--signature_def',
        'serving_default', '--inputs', 'x=' + input_path + '[x0]', '--outdir',
        output_dir, '--tf_debug'
    ])

    def fake_wrapper_session(sess):
      return sess

    with test.mock.patch.object(
        local_cli_wrapper,
        'LocalCLIDebugWrapperSession',
        side_effect=fake_wrapper_session,
        autospec=True) as fake:
      saved_model_cli.run(args)
      fake.assert_called_with(test.mock.ANY)

    y_actual = np.load(os.path.join(output_dir, 'y.npy'))
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  def testScanCommand(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    args = self.parser.parse_args(['scan', '--dir', base_path])
    with captured_output() as (out, _):
      saved_model_cli.scan(args)
    output = out.getvalue().strip()
    self.assertTrue('does not contain denylisted ops' in output)

  def testScanCommandFoundDenylistedOp(self):
    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    args = self.parser.parse_args(
        ['scan', '--dir', base_path, '--tag_set', 'serve'])
    op_denylist = saved_model_cli._OP_DENYLIST
    saved_model_cli._OP_DENYLIST = set(['VariableV2'])
    with captured_output() as (out, _):
      saved_model_cli.scan(args)
    saved_model_cli._OP_DENYLIST = op_denylist
    output = out.getvalue().strip()
    self.assertTrue('\'VariableV2\'' in output)

  def testAOTCompileCPUWrongSignatureDefKey(self):
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')

    self.parser = saved_model_cli.create_parser()
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir')
    args = self.parser.parse_args([
        'aot_compile_cpu', '--dir', base_path, '--tag_set', 'serve',
        '--output_prefix', output_dir, '--cpp_class', 'Compiled',
        '--signature_def_key', 'MISSING'
    ])
    with self.assertRaisesRegex(ValueError, 'Unable to find signature_def'):
      saved_model_cli.aot_compile_cpu(args)

  class AOTCompileDummyModel(tracking.AutoTrackable):
    """Model compatible with XLA compilation."""

    def __init__(self):
      self.var = variables.Variable(1.0, name='my_var')
      self.write_var = variables.Variable(1.0, name='write_var')

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32),
        # Test unused inputs.
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
    ])
    def func2(self, x, y):
      del y
      return {'res': x + self.var}

    @def_function.function(input_signature=[
        # Test large inputs.
        tensor_spec.TensorSpec(shape=(2048, 16), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
    ])
    def func3(self, x, y):
      del y
      return {'res': x + self.var}

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
    ])
    def func_write(self, x, y):
      del y
      self.write_var.assign(x + self.var)
      return {'res': self.write_var}

  @parameterized.named_parameters(
      ('VariablesToFeedNone', '', 'func2', None),
      ('VariablesToFeedNoneTargetAarch64Linux', '', 'func2',
       'aarch64-none-linux-gnu'),
      ('VariablesToFeedNoneTargetAarch64Android', '', 'func2',
       'aarch64-none-android'),
      ('VariablesToFeedAll', 'all', 'func2', None),
      ('VariablesToFeedMyVar', 'my_var', 'func2', None),
      ('VariablesToFeedNoneLargeConstant', '', 'func3', None),
      ('WriteToWriteVar', 'all', 'func_write', None),
  )
  def testAOTCompileCPUFreezesAndCompiles(
      self, variables_to_feed, func, target_triple):
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = self.AOTCompileDummyModel()
    func = getattr(dummy_model, func)
    with self.cached_session():
      self.evaluate(dummy_model.var.initializer)
      self.evaluate(dummy_model.write_var.initializer)
      save.save(dummy_model, saved_model_dir, signatures={'func': func})

    self.parser = saved_model_cli.create_parser()
    output_prefix = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir/out')
    args = [  # Use the default seving signature_key.
        'aot_compile_cpu', '--dir', saved_model_dir, '--tag_set', 'serve',
        '--signature_def_key', 'func', '--output_prefix', output_prefix,
        '--variables_to_feed', variables_to_feed, '--cpp_class', 'Generated'
    ]
    if target_triple:
      args.extend(['--target_triple', target_triple])
    args = self.parser.parse_args(args)
    with test.mock.patch.object(logging, 'warn') as captured_warn:
      saved_model_cli.aot_compile_cpu(args)
    self.assertRegex(
        str(captured_warn.call_args),
        'Signature input key \'y\'.*has been pruned while freezing the graph.')
    self.assertTrue(file_io.file_exists('{}.o'.format(output_prefix)))
    self.assertTrue(file_io.file_exists('{}.h'.format(output_prefix)))
    self.assertTrue(file_io.file_exists('{}_metadata.o'.format(output_prefix)))
    self.assertTrue(
        file_io.file_exists('{}_makefile.inc'.format(output_prefix)))
    header_contents = file_io.read_file_to_string('{}.h'.format(output_prefix))
    self.assertIn('class Generated', header_contents)
    self.assertIn('arg_feed_x_data', header_contents)
    self.assertIn('result_fetch_res_data', header_contents)
    # arg_y got filtered out as it's not used by the output.
    self.assertNotIn('arg_feed_y_data', header_contents)
    if variables_to_feed:
      # Read-only-variables' setters preserve constness.
      self.assertIn('set_var_param_my_var_data(const float', header_contents)
      self.assertNotIn('set_var_param_my_var_data(float', header_contents)
    if func == dummy_model.func_write:
      # Writeable variables setters do not preserve constness.
      self.assertIn('set_var_param_write_var_data(float', header_contents)
      self.assertNotIn('set_var_param_write_var_data(const float',
                       header_contents)

    makefile_contents = file_io.read_file_to_string(
        '{}_makefile.inc'.format(output_prefix))
    self.assertIn('-D_GLIBCXX_USE_CXX11_ABI=', makefile_contents)


if __name__ == '__main__':
  test.main()
