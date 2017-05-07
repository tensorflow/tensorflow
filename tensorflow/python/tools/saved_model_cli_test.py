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
"""Tests for SavedModelCLI tool.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import pickle
import shutil
import sys

import numpy as np
from six import StringIO

from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.platform import test
from tensorflow.python.tools import saved_model_cli

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


class SavedModelCLITestCase(test.TestCase):

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
    self.assertMultiLineEqual(output, exp_out)
    self.assertEqual(err.getvalue().strip(), '')

  def testShowCommandTags(self):
    base_path = test.test_src_dir_path(SAVED_MODEL_PATH)
    self.parser = saved_model_cli.create_parser()
    args = self.parser.parse_args(['show', '--dir', base_path])
    with captured_output() as (out, err):
      saved_model_cli.show(args)
    output = out.getvalue().strip()
    exp_out = 'The given SavedModel contains the following tag-sets:\nserve'
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
        'inputs[\'x\'] tensor_info:\n'
        '    dtype: DT_FLOAT\n    shape: (-1, 1)\n    name: x:0\n'
        'The given SavedModel SignatureDef contains the following output(s):\n'
        'outputs[\'y\'] tensor_info:\n'
        '    dtype: DT_FLOAT\n    shape: (-1, 1)\n    name: y:0\n'
        'Method name is: tensorflow/serving/predict')
    self.assertEqual(output, expected_output)
    self.assertEqual(err.getvalue().strip(), '')

  def testInputPreProcessFormats(self):
    input_str = 'input1=/path/file.txt[ab3];input2=file2'
    input_expr_str = 'input3=np.zeros([2,2]);input4=[4,5]'
    input_dict = saved_model_cli.preprocess_inputs_arg_string(input_str)
    input_expr_dict = saved_model_cli.preprocess_input_exprs_arg_string(
        input_expr_str)
    self.assertTrue(input_dict['input1'] == ('/path/file.txt', 'ab3'))
    self.assertTrue(input_dict['input2'] == ('file2', None))
    self.assertTrue(input_expr_dict['input3'] == 'np.zeros([2,2])')
    self.assertTrue(input_expr_dict['input4'] == '[4,5]')
    self.assertTrue(len(input_dict) == 2)
    self.assertTrue(len(input_expr_dict) == 2)

  def testInputPreProcessFileNames(self):
    input_str = (r'inputx=C:\Program Files\data.npz[v:0];'
                 r'input:0=c:\PROGRA~1\data.npy')
    input_dict = saved_model_cli.preprocess_inputs_arg_string(input_str)
    print(input_dict)
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
    with self.assertRaises(RuntimeError):
      saved_model_cli.preprocess_input_exprs_arg_string(input_str)

  def testInputParserNPY(self):
    x0 = np.array([[1], [2]])
    x1 = np.array(range(6)).reshape(2, 3)
    input0_path = os.path.join(test.get_temp_dir(), 'input0.npy')
    input1_path = os.path.join(test.get_temp_dir(), 'input1.npy')
    np.save(input0_path, x0)
    np.save(input1_path, x1)
    input_str = 'x0=' + input0_path + '[x0];x1=' + input1_path
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(input_str, '')
    self.assertTrue(np.all(feed_dict['x0'] == x0))
    self.assertTrue(np.all(feed_dict['x1'] == x1))

  def testInputParserNPZ(self):
    x0 = np.array([[1], [2]])
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0)
    input_str = 'x=' + input_path + '[a];y=' + input_path
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(input_str, '')
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
    feed_dict = saved_model_cli.load_inputs_from_input_arg_string(input_str, '')
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
        '', input_expr_str)
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
        input_str, input_expr_str)
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
        input_str, input_expr_str)
    self.assertTrue(np.all(feed_dict['x0'] == x1))

  def testInputParserErrorNoName(self):
    x0 = np.array([[1], [2]])
    x1 = np.array(range(5))
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0, b=x1)
    input_str = 'x=' + input_path
    with self.assertRaises(RuntimeError):
      saved_model_cli.load_inputs_from_input_arg_string(input_str, '')

  def testInputParserErrorWrongName(self):
    x0 = np.array([[1], [2]])
    x1 = np.array(range(5))
    input_path = os.path.join(test.get_temp_dir(), 'input.npz')
    np.savez(input_path, a=x0, b=x1)
    input_str = 'x=' + input_path + '[c]'
    with self.assertRaises(RuntimeError):
      saved_model_cli.load_inputs_from_input_arg_string(input_str, '')

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

    with test.mock.patch.object(local_cli_wrapper,
                                'LocalCLIDebugWrapperSession',
                                side_effect=fake_wrapper_session,
                                autospec=True) as fake:
      saved_model_cli.run(args)
      fake.assert_called_with(test.mock.ANY)

    y_actual = np.load(os.path.join(output_dir, 'y.npy'))
    y_expected = np.array([[2.5], [3.0]])
    self.assertAllClose(y_expected, y_actual)


if __name__ == '__main__':
  test.main()
