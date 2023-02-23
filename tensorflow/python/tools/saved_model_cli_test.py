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
import io
import os
import pickle
import platform
import shutil
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import save
from tensorflow.python.tools import saved_model_cli
from tensorflow.python.trackable import autotrackable

SAVED_MODEL_PATH = ('cc/saved_model/testdata/half_plus_two/00000123')


@contextlib.contextmanager
def captured_output():
  new_out, new_err = io.StringIO(), io.StringIO()
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

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowCommandAll(self, get_ops_mock):
    # Mocking _get_ops_in_metagraph because it returns a nondeterministically
    # ordered set of ops.
    get_ops_mock.return_value = {'Op1', 'Op2', 'Op3'}
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

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
  Method name is: tensorflow/serving/predict
The MetaGraph with tag set ['serve'] contains the following ops:"""
    # pylint: enable=line-too-long

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', base_path, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    get_ops_mock.assert_called_once()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce useful error msg if the comparison fails
    self.assertIn(exp_out, output)
    self.assertIn('Op1', output)
    self.assertIn('Op2', output)
    self.assertIn('Op3', output)
    self.assertEqual(err.getvalue().strip(), '')

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowAllWithFunctions(self, get_ops_mock):

    class DummyModel(autotrackable.AutoTrackable):
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

    # Mocking _get_ops_in_metagraph because it returns a nondeterministically
    # ordered set of ops.
    get_ops_mock.return_value = {'Op1'}
    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    # Call with specific values to create new polymorphic function traces.
    dummy_model.func1(constant_op.constant(5), constant_op.constant(9), True)
    dummy_model(constant_op.constant(5))
    with self.cached_session():
      save.save(dummy_model, saved_model_dir)

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
The MetaGraph with tag set ['serve'] contains the following ops: {'Op1'}

Concrete Functions:
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

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', saved_model_dir, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  @test.mock.patch.object(saved_model_cli, '_get_ops_in_metagraph')
  def testShowAllWithPureConcreteFunction(self, get_ops_mock):

    class DummyModel(autotrackable.AutoTrackable):
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

    # Mocking _get_ops_in_metagraph because it returns a nondeterministically
    # ordered set of ops.
    get_ops_mock.return_value = {'Op1'}
    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    with self.cached_session():
      save.save(dummy_model, saved_model_dir)

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
The MetaGraph with tag set ['serve'] contains the following ops: {'Op1'}

Concrete Functions:
  Function Name: 'pure_concrete_function'
    Option #1
      Callable with:
        Argument #1
          a: TensorSpec(shape=(), dtype=tf.float32, name='a')
        Argument #2
          b: TensorSpec(shape=(), dtype=tf.float32, name='b')
""".strip()  # pylint: enable=line-too-long

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', saved_model_dir, '--all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandTags(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    exp_out = 'The given SavedModel contains the following tag-sets:\n\'serve\''

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(['saved_model_cli', 'show', '--dir', base_path])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandSignature(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    exp_header = ('The given SavedModel MetaGraphDef contains SignatureDefs '
                  'with the following keys:')
    exp_start = 'SignatureDef key: '
    exp_keys = [
        '"classify_x2_to_y3"', '"classify_x_to_y"', '"regress_x2_to_y3"',
        '"regress_x_to_y"', '"regress_x_to_y2"', '"serving_default"'
    ]

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS(
        ['saved_model_cli', 'show', '--dir', base_path, '--tag_set', 'serve'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    # Order of signatures does not matter
    self.assertMultiLineEqual(
        output,
        '\n'.join([exp_header] +
                  [exp_start + exp_key for exp_key in exp_keys]))
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandErrorNoTagSet(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path,
        '--tag_set', 'badtagset'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(RuntimeError):
      saved_model_cli.show()

  def testShowCommandInputsOutputs(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    expected_output = (
        'The given SavedModel SignatureDef contains the following input(s):\n'
        '  inputs[\'x\'] tensor_info:\n'
        '      dtype: DT_FLOAT\n      shape: (-1, 1)\n      name: x:0\n'
        'The given SavedModel SignatureDef contains the following output(s):\n'
        '  outputs[\'y\'] tensor_info:\n'
        '      dtype: DT_FLOAT\n      shape: (-1, 1)\n      name: y:0\n'
        'Method name is: tensorflow/serving/predict')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default'
    ])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertEqual(output, expected_output)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandListOps(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path, '--tag_set', 'serve',
        '--list_ops'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.assertIn(
        'The MetaGraph with tag set [\'serve\'] contains the following ops:',
        output)
    self.assertIn('\'VariableV2\'', output)
    self.assertIn('\'Add\'', output)
    self.assertIn('\'RestoreV2\'', output)
    self.assertIn('\'ShardedFilename\'', output)
    self.assertIn('\'Placeholder\'', output)
    self.assertIn('\'Mul\'', output)
    self.assertIn('\'Pack\'', output)
    self.assertIn('\'Reshape\'', output)
    self.assertIn('\'SaveV2\'', output)
    self.assertIn('\'Const\'', output)
    self.assertIn('\'Identity\'', output)
    self.assertIn('\'Assign\'', output)
    self.assertIn('\'ParseExample\'', output)
    self.assertIn('\'StringJoin\'', output)
    self.assertIn('\'MergeV2Checkpoints\'', output)
    self.assertIn('\'NoOp\'', output)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandListOpsNoTags(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    exp_out = ('--list_ops must be paired with a tag-set or with --all.\n'
               'The given SavedModel contains the following tag-sets:\n'
               '\'serve\'').strip()

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'show', '--dir', base_path, '--list_ops'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, err):
      saved_model_cli.show()
    output = out.getvalue().strip()
    self.maxDiff = None  # Produce a useful error msg if the comparison fails
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testPrintREFTypeTensor(self):
    ref_tensor_info = meta_graph_pb2.TensorInfo(
        dtype=types_pb2.DT_FLOAT_REF)
    with captured_output() as (out, err):
      saved_model_cli._print_tensor_info(ref_tensor_info)
    self.assertIn('DT_FLOAT_REF', out.getvalue().strip())
    self.assertEqual(err.getvalue().strip(), '')

  def testInputPreProcessFormats(self):
    input_str = 'input1=/path/file.txt[ab3];input2=file2'
    input_expr_str = 'input3=np.zeros([2,2]);input4=[4,5]'
    input_dict = saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_expr_dict = saved_model_cli.preprocess_input_exprs_arg_string(
        input_expr_str, safe=False)
    self.assertEqual(input_dict['input1'], ('/path/file.txt', 'ab3'))
    self.assertEqual(input_dict['input2'], ('file2', None))
    print(input_expr_dict['input3'])
    self.assertAllClose(input_expr_dict['input3'], np.zeros([2, 2]))
    self.assertAllClose(input_expr_dict['input4'], [4, 5])
    self.assertLen(input_dict, 2)
    self.assertLen(input_expr_dict, 2)

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
    self.assertEqual(input_dict['inputx'], (r'C:\Program Files\data.npz',
                                            'v:0'))
    self.assertEqual(input_dict['input:0'], (r'c:\PROGRA~1\data.npy', None))

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

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamples(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(),
                              'input_examples' + ('tfrt' if use_tfrt else ''))

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y', '--input_examples',
        'inputs=[{"x":[8.0],"x2":[5.0]}, {"x":[4.0],"x2":[3.0]}]',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(os.path.join(output_dir, 'outputs.npy'))
    y_expected = np.array([[6.0], [4.0]])
    self.assertAllEqual(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandLongInputExamples(self, use_tfrt):

    class DummyModel(autotrackable.AutoTrackable):
      """Model with callable polymorphic functions specified."""

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=None, dtype=dtypes.string),
      ])
      def func(self, inputs):
        ex = parsing_ops.parse_example(serialized=inputs, features={
            'variable0': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable1': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable2': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable3': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable4': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable5': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable6': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable7': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable8': parsing_config.FixedLenFeature(
                (), dtypes.float32),
            'variable9': parsing_config.FixedLenFeature(
                (), dtypes.float32),
        })
        return {'outputs': sum(ex.values())}

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = DummyModel()
    func = getattr(dummy_model, 'func')

    with self.cached_session():
      save.save(dummy_model, saved_model_dir, signatures={'func': func})

    output_dir = os.path.join(
        test.get_temp_dir(),
        'long_input_examples' + ('tfrt' if use_tfrt else ''))

    saved_model_cli.flags.FLAGS.unparse_flags()
    input_examples = (
        'inputs=[{"variable0":[0.0],"variable1":[1.0],"variable2":[2.0],'
        '"variable3":[3.0],"variable4":[4.0],"variable5":[5.0],'
        '"variable6":[6.0],"variable7":[7.0],"variable8":[8.0],'
        '"variable9":[9.0]}, {"variable0":[10.0],"variable1":[1.0],'
        '"variable2":[2.0],"variable3":[3.0],"variable4":[4.0],'
        '"variable5":[5.0],"variable6":[6.0],"variable7":[7.0],'
        '"variable8":[8.0],"variable9":[9.0]}]')
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', saved_model_dir, '--tag_set', 'serve',
        '--signature_def', 'func', '--input_examples', input_examples,
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(os.path.join(output_dir, 'outputs.npy'))
    y_expected = np.array([45.0, 55.0])
    self.assertAllEqual(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandExistingOutdir(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(), 'testRunCommand_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'outputs.npy')
    if os.path.exists(output_file):
      os.remove(output_file)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x2_to_y3', '--inputs',
        'inputs=' + input_path + '[x0]', '--outdir', test.get_temp_dir()
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(output_file)
    y_expected = np.array([[3.5], [4.0]])
    self.assertAllClose(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandNewOutdir(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandNewOutdir_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(os.path.join(output_dir, 'y.npy'))
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandOutOverwrite(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandOutOverwrite_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'y.npy')
    open(output_file, 'a').close()

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', test.get_temp_dir(),
        '--overwrite'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    saved_model_cli.run()
    y_actual = np.load(output_file)
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInvalidInputKeyError(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x2_to_y3',
        '--input_exprs', 'x2=[1,2,3]'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(ValueError):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInvalidSignature(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'INVALID_SIGNATURE',
        '--input_exprs', 'x2=[1,2,3]'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError,
                                'Could not find signature '
                                '"INVALID_SIGNATURE"'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamplesNotListError(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y',
        '--input_examples', 'inputs={"x":8.0,"x2":5.0}',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'must be a list'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamplesFeatureValueNotListError(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y',
        '--input_examples', 'inputs=[{"x":8.0,"x2":5.0}]',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'feature value must be a list'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputExamplesFeatureBadType(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'regress_x_to_y',
        '--input_examples', 'inputs=[{"x":[[1],[2]]}]',
        '--outdir', output_dir
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'is not supported'):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandOutputFileExistError(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandOutOverwrite_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_file = os.path.join(test.get_temp_dir(), 'y.npy')
    open(output_file, 'a').close()

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', test.get_temp_dir()
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(RuntimeError):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandInputNotGivenError(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaises(AttributeError):
      saved_model_cli.run()

  @parameterized.named_parameters(('non_tfrt', False))
  def testRunCommandWithDebuggerEnabled(self, use_tfrt):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    input_path = os.path.join(test.get_temp_dir(),
                              'testRunCommandNewOutdir_inputs.npz')
    x = np.array([[1], [2]])
    x_notused = np.zeros((6, 3))
    np.savez(input_path, x0=x, x1=x_notused)
    output_dir = os.path.join(test.get_temp_dir(), 'new_dir')
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'run', '--dir', base_path, '--tag_set', 'serve',
        '--signature_def', 'serving_default', '--inputs', 'x=' +
        input_path + '[x0]', '--outdir', output_dir, '--tf_debug'
        ] + (['--use_tfrt'] if use_tfrt else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()

    def fake_wrapper_session(sess):
      return sess

    with test.mock.patch.object(
        local_cli_wrapper,
        'LocalCLIDebugWrapperSession',
        side_effect=fake_wrapper_session,
        autospec=True) as fake:
      saved_model_cli.run()
      fake.assert_called_with(test.mock.ANY)

    y_actual = np.load(os.path.join(output_dir, 'y.npy'))
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)

  def testScanCommand(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli', 'scan', '--dir', base_path])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, _):
      saved_model_cli.scan()
    output = out.getvalue().strip()
    self.assertIn(('MetaGraph with tag set [\'serve\'] does not contain the '
                   'default denylisted ops: {\''), output)
    self.assertIn('\'ReadFile\'', output)
    self.assertIn('\'WriteFile\'', output)
    self.assertIn('\'PrintV2\'', output)

  def testScanCommandFoundCustomDenylistedOp(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'scan', '--dir', base_path, '--tag_set', 'serve', '--op_denylist',
        'VariableV2,Assign,Relu6'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with captured_output() as (out, _):
      saved_model_cli.scan()
    output = out.getvalue().strip()
    self.assertIn(('MetaGraph with tag set [\'serve\'] contains the following'
                   ' denylisted ops:'), output)
    self.assertTrue(('{\'VariableV2\', \'Assign\'}' in output) or
                    ('{\'Assign\', \'VariableV2\'}' in output))

  def testAOTCompileCPUWrongSignatureDefKey(self):
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')

    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    output_dir = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',
        'aot_compile_cpu', '--dir', base_path, '--tag_set', 'serve',
        '--output_prefix', output_dir, '--cpp_class', 'Compiled',
        '--signature_def_key', 'MISSING'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with self.assertRaisesRegex(ValueError, 'Unable to find signature_def'):
      saved_model_cli.aot_compile_cpu()

  class AOTCompileDummyModel(autotrackable.AutoTrackable):
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
  def testAOTCompileCPUFreezesAndCompiles(self, variables_to_feed, func,
                                          target_triple):
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = self.AOTCompileDummyModel()
    func = getattr(dummy_model, func)
    with self.cached_session():
      self.evaluate(dummy_model.var.initializer)
      self.evaluate(dummy_model.write_var.initializer)
      save.save(dummy_model, saved_model_dir, signatures={'func': func})

    output_prefix = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir/out')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',  # Use the default serving signature_key.
        'aot_compile_cpu', '--dir', saved_model_dir, '--tag_set', 'serve',
        '--signature_def_key', 'func', '--output_prefix', output_prefix,
        '--variables_to_feed', variables_to_feed,
        '--cpp_class', 'Generated'
        ] + (['--target_triple', target_triple] if target_triple else []))
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with test.mock.patch.object(logging, 'warn') as captured_warn:
      saved_model_cli.aot_compile_cpu()
    self.assertRegex(
        str(captured_warn.call_args),
        'Signature input key \'y\'.*has been pruned while freezing the '
        'graph.')
    self.assertTrue(file_io.file_exists('{}.o'.format(output_prefix)))
    self.assertTrue(file_io.file_exists('{}.h'.format(output_prefix)))
    self.assertTrue(file_io.file_exists(
        '{}_metadata.o'.format(output_prefix)))
    self.assertTrue(
        file_io.file_exists('{}_makefile.inc'.format(output_prefix)))
    header_contents = file_io.read_file_to_string(
        '{}.h'.format(output_prefix))
    self.assertIn('class Generated', header_contents)
    self.assertIn('arg_feed_x_data', header_contents)
    self.assertIn('result_fetch_res_data', header_contents)
    # arg_y got filtered out as it's not used by the output.
    self.assertNotIn('arg_feed_y_data', header_contents)
    if variables_to_feed:
      # Read-only-variables' setters preserve constness.
      self.assertIn('set_var_param_my_var_data(const float', header_contents)
      self.assertNotIn('set_var_param_my_var_data(float', header_contents)
    if func == dummy_model.func_write:  # pylint: disable=comparison-with-callable
      # Writeable variables setters do not preserve constness.
      self.assertIn('set_var_param_write_var_data(float', header_contents)
      self.assertNotIn('set_var_param_write_var_data(const float',
                       header_contents)

    makefile_contents = file_io.read_file_to_string(
        '{}_makefile.inc'.format(output_prefix))
    self.assertIn('-D_GLIBCXX_USE_CXX11_ABI=', makefile_contents)

  def testFreezeModel(self):
    if not test.is_built_with_xla():
      self.skipTest('Skipping test because XLA is not compiled in.')

    saved_model_dir = os.path.join(test.get_temp_dir(), 'dummy_model')
    dummy_model = self.AOTCompileDummyModel()
    func = getattr(dummy_model, 'func2')
    with self.cached_session():
      self.evaluate(dummy_model.var.initializer)
      self.evaluate(dummy_model.write_var.initializer)
      save.save(dummy_model, saved_model_dir, signatures={'func': func})

    output_prefix = os.path.join(test.get_temp_dir(), 'aot_compile_cpu_dir/out')

    saved_model_cli.flags.FLAGS.unparse_flags()
    saved_model_cli.flags.FLAGS([
        'saved_model_cli',  # Use the default seving signature_key.
        'freeze_model', '--dir', saved_model_dir, '--tag_set', 'serve',
        '--signature_def_key', 'func', '--output_prefix', output_prefix,
        '--variables_to_feed', 'all'])
    parser = saved_model_cli.create_parser()
    parser.parse_args()
    with test.mock.patch.object(logging, 'warn'):
      saved_model_cli.freeze_model()
    self.assertTrue(
        file_io.file_exists(os.path.join(output_prefix, 'frozen_graph.pb')))
    self.assertTrue(
        file_io.file_exists(os.path.join(output_prefix, 'config.pbtxt')))


if __name__ == '__main__':
  test.main()
