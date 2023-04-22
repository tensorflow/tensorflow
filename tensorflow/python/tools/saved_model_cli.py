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
"""Command-line interface to inspect and execute a graph in a SavedModel.

For detailed usages and examples, please refer to:
https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys

from absl import app  # pylint: disable=unused-import
import numpy as np
import six

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc


_XLA_DEBUG_OPTIONS_URL = (
    'https://github.com/tensorflow/tensorflow/blob/master/'
    'tensorflow/compiler/xla/debug_options_flags.cc')


# Set of ops to denylist.
_OP_DENYLIST = set(['WriteFile', 'ReadFile', 'PrintV2'])


def _show_tag_sets(saved_model_dir):
  """Prints the tag-sets stored in SavedModel directory.

  Prints all the tag-sets for MetaGraphs stored in SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
  tag_sets = saved_model_utils.get_saved_model_tag_sets(saved_model_dir)
  print('The given SavedModel contains the following tag-sets:')
  for tag_set in sorted(tag_sets):
    print('%r' % ', '.join(sorted(tag_set)))


def _show_signature_def_map_keys(saved_model_dir, tag_set):
  """Prints the keys for each SignatureDef in the SignatureDef map.

  Prints the list of SignatureDef keys from the SignatureDef map specified by
  the given tag-set and SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef to get SignatureDef map from,
        in string format, separated by ','. For tag-set contains multiple tags,
        all tags must be passed in.
  """
  signature_def_map = get_signature_def_map(saved_model_dir, tag_set)
  print('The given SavedModel MetaGraphDef contains SignatureDefs with the '
        'following keys:')
  for signature_def_key in sorted(signature_def_map.keys()):
    print('SignatureDef key: \"%s\"' % signature_def_key)


def _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                signature_def_key):
  """Gets TensorInfo for all inputs of the SignatureDef.

  Returns a dictionary that maps each input key to its TensorInfo for the given
  signature_def_key in the meta_graph_def

  Args:
    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDef map to
        look up SignatureDef key.
    signature_def_key: A SignatureDef key string.

  Returns:
    A dictionary that maps input tensor keys to TensorInfos.
  """
  return meta_graph_def.signature_def[signature_def_key].inputs


def _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                 signature_def_key):
  """Gets TensorInfos for all outputs of the SignatureDef.

  Returns a dictionary that maps each output key to its TensorInfo for the given
  signature_def_key in the meta_graph_def.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDefmap to
    look up signature_def_key.
    signature_def_key: A SignatureDef key string.

  Returns:
    A dictionary that maps output tensor keys to TensorInfos.
  """
  return meta_graph_def.signature_def[signature_def_key].outputs


def _show_inputs_outputs(saved_model_dir, tag_set, signature_def_key, indent=0):
  """Prints input and output TensorInfos.

  Prints the details of input and output TensorInfos for the SignatureDef mapped
  by the given signature_def_key.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef, in string format, separated by
        ','. For tag-set contains multiple tags, all tags must be passed in.
    signature_def_key: A SignatureDef key string.
    indent: How far (in increments of 2 spaces) to indent each line of output.
  """
  meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,
                                                        tag_set)
  inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)
  outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)

  indent_str = '  ' * indent
  def in_print(s):
    print(indent_str + s)

  in_print('The given SavedModel SignatureDef contains the following input(s):')
  for input_key, input_tensor in sorted(inputs_tensor_info.items()):
    in_print('  inputs[\'%s\'] tensor_info:' % input_key)
    _print_tensor_info(input_tensor, indent+1)

  in_print('The given SavedModel SignatureDef contains the following '
           'output(s):')
  for output_key, output_tensor in sorted(outputs_tensor_info.items()):
    in_print('  outputs[\'%s\'] tensor_info:' % output_key)
    _print_tensor_info(output_tensor, indent+1)

  in_print('Method name is: %s' %
           meta_graph_def.signature_def[signature_def_key].method_name)


def _show_defined_functions(saved_model_dir):
  """Prints the callable concrete and polymorphic functions of the Saved Model.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
  meta_graphs = saved_model_utils.read_saved_model(saved_model_dir).meta_graphs
  has_object_graph_def = False

  for meta_graph_def in meta_graphs:
    has_object_graph_def |= meta_graph_def.HasField('object_graph_def')
  if not has_object_graph_def:
    return
  with ops_lib.Graph().as_default():
    trackable_object = load.load(saved_model_dir)

  print('\nDefined Functions:', end='')
  functions = (
      save._AugmentedGraphView(trackable_object)  # pylint: disable=protected-access
      .list_functions(trackable_object))
  functions = sorted(functions.items(), key=lambda x: x[0])
  for name, function in functions:
    print('\n  Function Name: \'%s\'' % name)
    concrete_functions = []
    if isinstance(function, defun.ConcreteFunction):
      concrete_functions.append(function)
    if isinstance(function, def_function.Function):
      concrete_functions.extend(
          function._list_all_concrete_functions_for_serialization())  # pylint: disable=protected-access
    concrete_functions = sorted(concrete_functions, key=lambda x: x.name)
    for index, concrete_function in enumerate(concrete_functions, 1):
      args, kwargs = None, None
      if concrete_function.structured_input_signature:
        args, kwargs = concrete_function.structured_input_signature
      elif concrete_function._arg_keywords:  # pylint: disable=protected-access
        # For pure ConcreteFunctions we might have nothing better than
        # _arg_keywords.
        args = concrete_function._arg_keywords  # pylint: disable=protected-access
      if args:
        print('    Option #%d' % index)
        print('      Callable with:')
        _print_args(args, indent=4)
      if kwargs:
        _print_args(kwargs, 'Named Argument', indent=4)


def _print_args(arguments, argument_type='Argument', indent=0):
  """Formats and prints the argument of the concrete functions defined in the model.

  Args:
    arguments: Arguments to format print.
    argument_type: Type of arguments.
    indent: How far (in increments of 2 spaces) to indent each line of
     output.
  """
  indent_str = '  ' * indent

  def _maybe_add_quotes(value):
    is_quotes = '\'' * isinstance(value, str)
    return is_quotes + str(value) + is_quotes

  def in_print(s, end='\n'):
    print(indent_str + s, end=end)

  for index, element in enumerate(arguments, 1):
    if indent == 4:
      in_print('%s #%d' % (argument_type, index))
    if isinstance(element, six.string_types):
      in_print('  %s' % element)
    elif isinstance(element, tensor_spec.TensorSpec):
      print((indent + 1) * '  ' + '%s: %s' % (element.name, repr(element)))
    elif (isinstance(element, collections_abc.Iterable) and
          not isinstance(element, dict)):
      in_print('  DType: %s' % type(element).__name__)
      in_print('  Value: [', end='')
      for value in element:
        print('%s' % _maybe_add_quotes(value), end=', ')
      print('\b\b]')
    elif isinstance(element, dict):
      in_print('  DType: %s' % type(element).__name__)
      in_print('  Value: {', end='')
      for (key, value) in element.items():
        print('\'%s\': %s' % (str(key), _maybe_add_quotes(value)), end=', ')
      print('\b\b}')
    else:
      in_print('  DType: %s' % type(element).__name__)
      in_print('  Value: %s' % str(element))


def _print_tensor_info(tensor_info, indent=0):
  """Prints details of the given tensor_info.

  Args:
    tensor_info: TensorInfo object to be printed.
    indent: How far (in increments of 2 spaces) to indent each line output
  """
  indent_str = '  ' * indent
  def in_print(s):
    print(indent_str + s)

  in_print('    dtype: ' +
           {value: key
            for (key, value) in types_pb2.DataType.items()}[tensor_info.dtype])
  # Display shape as tuple.
  if tensor_info.tensor_shape.unknown_rank:
    shape = 'unknown_rank'
  else:
    dims = [str(dim.size) for dim in tensor_info.tensor_shape.dim]
    shape = ', '.join(dims)
    shape = '(' + shape + ')'
  in_print('    shape: ' + shape)
  in_print('    name: ' + tensor_info.name)


def _show_all(saved_model_dir):
  """Prints tag-set, SignatureDef and Inputs/Outputs information in SavedModel.

  Prints all tag-set, SignatureDef and Inputs/Outputs information stored in
  SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
  tag_sets = saved_model_utils.get_saved_model_tag_sets(saved_model_dir)
  for tag_set in sorted(tag_sets):
    print("\nMetaGraphDef with tag-set: '%s' "
          "contains the following SignatureDefs:" % ', '.join(tag_set))

    tag_set = ','.join(tag_set)
    signature_def_map = get_signature_def_map(saved_model_dir, tag_set)
    for signature_def_key in sorted(signature_def_map.keys()):
      print('\nsignature_def[\'' + signature_def_key + '\']:')
      _show_inputs_outputs(saved_model_dir, tag_set, signature_def_key,
                           indent=1)
  _show_defined_functions(saved_model_dir)


def get_meta_graph_def(saved_model_dir, tag_set):
  """DEPRECATED: Use saved_model_utils.get_meta_graph_def instead.

  Gets MetaGraphDef from SavedModel. Returns the MetaGraphDef for the given
  tag-set and SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect or execute.
    tag_set: Group of tag(s) of the MetaGraphDef to load, in string format,
        separated by ','. For tag-set contains multiple tags, all tags must be
        passed in.

  Raises:
    RuntimeError: An error when the given tag-set does not exist in the
        SavedModel.

  Returns:
    A MetaGraphDef corresponding to the tag-set.
  """
  return saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)


def get_signature_def_map(saved_model_dir, tag_set):
  """Gets SignatureDef map from a MetaGraphDef in a SavedModel.

  Returns the SignatureDef map for the given tag-set in the SavedModel
  directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect or execute.
    tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in
        string format, separated by ','. For tag-set contains multiple tags, all
        tags must be passed in.

  Returns:
    A SignatureDef map that maps from string keys to SignatureDefs.
  """
  meta_graph = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
  return meta_graph.signature_def


def scan_meta_graph_def(meta_graph_def):
  """Scans meta_graph_def and reports if there are ops on denylist.

  Print ops if they are on black list, or print success if no denylisted ops
  found.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
  """
  all_ops_set = set(
      meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))
  denylisted_ops = _OP_DENYLIST & all_ops_set
  if denylisted_ops:
    # TODO(yifeif): print more warnings
    print(
        'MetaGraph with tag set %s contains the following denylisted ops:' %
        meta_graph_def.meta_info_def.tags, denylisted_ops)
  else:
    print('MetaGraph with tag set %s does not contain denylisted ops.' %
          meta_graph_def.meta_info_def.tags)


def run_saved_model_with_feed_dict(saved_model_dir, tag_set, signature_def_key,
                                   input_tensor_key_feed_dict, outdir,
                                   overwrite_flag, worker=None, init_tpu=False,
                                   tf_debug=False):
  """Runs SavedModel and fetch all outputs.

  Runs the input dictionary through the MetaGraphDef within a SavedModel
  specified by the given tag_set and SignatureDef. Also save the outputs to file
  if outdir is not None.

  Args:
    saved_model_dir: Directory containing the SavedModel to execute.
    tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in
        string format, separated by ','. For tag-set contains multiple tags, all
        tags must be passed in.
    signature_def_key: A SignatureDef key string.
    input_tensor_key_feed_dict: A dictionary maps input keys to numpy ndarrays.
    outdir: A directory to save the outputs to. If the directory doesn't exist,
        it will be created.
    overwrite_flag: A boolean flag to allow overwrite output file if file with
        the same name exists.
    worker: If provided, the session will be run on the worker.  Valid worker
        specification is a bns or gRPC path.
    init_tpu: If true, the TPU system will be initialized after the session
        is created.
    tf_debug: A boolean flag to use TensorFlow Debugger (TFDBG) to observe the
        intermediate Tensor values and runtime GraphDefs while running the
        SavedModel.

  Raises:
    ValueError: When any of the input tensor keys is not valid.
    RuntimeError: An error when output file already exists and overwrite is not
    enabled.
  """
  # Get a list of output tensor names.
  meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,
                                                        tag_set)

  # Re-create feed_dict based on input tensor name instead of key as session.run
  # uses tensor name.
  inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)

  # Check if input tensor keys are valid.
  for input_key_name in input_tensor_key_feed_dict.keys():
    if input_key_name not in inputs_tensor_info:
      raise ValueError(
          '"%s" is not a valid input key. Please choose from %s, or use '
          '--show option.' %
          (input_key_name, '"' + '", "'.join(inputs_tensor_info.keys()) + '"'))

  inputs_feed_dict = {
      inputs_tensor_info[key].name: tensor
      for key, tensor in input_tensor_key_feed_dict.items()
  }
  # Get outputs
  outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)
  # Sort to preserve order because we need to go from value to key later.
  output_tensor_keys_sorted = sorted(outputs_tensor_info.keys())
  output_tensor_names_sorted = [
      outputs_tensor_info[tensor_key].name
      for tensor_key in output_tensor_keys_sorted
  ]

  with session.Session(worker, graph=ops_lib.Graph()) as sess:
    if init_tpu:
      print('Initializing TPU System ...')
      # This is needed for freshly started worker, or if the job
      # restarts after a preemption.
      sess.run(tpu.initialize_system())

    loader.load(sess, tag_set.split(','), saved_model_dir)

    if tf_debug:
      sess = local_cli_wrapper.LocalCLIDebugWrapperSession(sess)

    outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict)

    for i, output in enumerate(outputs):
      output_tensor_key = output_tensor_keys_sorted[i]
      print('Result for output key %s:\n%s' % (output_tensor_key, output))

      # Only save if outdir is specified.
      if outdir:
        # Create directory if outdir does not exist
        if not os.path.isdir(outdir):
          os.makedirs(outdir)
        output_full_path = os.path.join(outdir, output_tensor_key + '.npy')

        # If overwrite not enabled and file already exist, error out
        if not overwrite_flag and os.path.exists(output_full_path):
          raise RuntimeError(
              'Output file %s already exists. Add \"--overwrite\" to overwrite'
              ' the existing output files.' % output_full_path)

        np.save(output_full_path, output)
        print('Output %s is saved to %s' % (output_tensor_key,
                                            output_full_path))


def preprocess_inputs_arg_string(inputs_str):
  """Parses input arg into dictionary that maps input to file/variable tuple.

  Parses input string in the format of, for example,
  "input1=filename1[variable_name1],input2=filename2" into a
  dictionary looks like
  {'input_key1': (filename1, variable_name1),
   'input_key2': (file2, None)}
  , which maps input keys to a tuple of file name and variable name(None if
  empty).

  Args:
    inputs_str: A string that specified where to load inputs. Inputs are
    separated by semicolons.
        * For each input key:
            '<input_key>=<filename>' or
            '<input_key>=<filename>[<variable_name>]'
        * The optional 'variable_name' key will be set to None if not specified.

  Returns:
    A dictionary that maps input keys to a tuple of file name and variable name.

  Raises:
    RuntimeError: An error when the given input string is in a bad format.
  """
  input_dict = {}
  inputs_raw = inputs_str.split(';')
  for input_raw in filter(bool, inputs_raw):  # skip empty strings
    # Format of input=filename[variable_name]'
    match = re.match(r'([^=]+)=([^\[\]]+)\[([^\[\]]+)\]$', input_raw)

    if match:
      input_dict[match.group(1)] = match.group(2), match.group(3)
    else:
      # Format of input=filename'
      match = re.match(r'([^=]+)=([^\[\]]+)$', input_raw)
      if match:
        input_dict[match.group(1)] = match.group(2), None
      else:
        raise RuntimeError(
            '--inputs "%s" format is incorrect. Please follow'
            '"<input_key>=<filename>", or'
            '"<input_key>=<filename>[<variable_name>]"' % input_raw)

  return input_dict


def preprocess_input_exprs_arg_string(input_exprs_str):
  """Parses input arg into dictionary that maps input key to python expression.

  Parses input string in the format of 'input_key=<python expression>' into a
  dictionary that maps each input_key to its python expression.

  Args:
    input_exprs_str: A string that specifies python expression for input keys.
    Each input is separated by semicolon. For each input key:
        'input_key=<python expression>'

  Returns:
    A dictionary that maps input keys to their values.

  Raises:
    RuntimeError: An error when the given input string is in a bad format.
  """
  input_dict = {}

  for input_raw in filter(bool, input_exprs_str.split(';')):
    if '=' not in input_exprs_str:
      raise RuntimeError('--input_exprs "%s" format is incorrect. Please follow'
                         '"<input_key>=<python expression>"' % input_exprs_str)
    input_key, expr = input_raw.split('=', 1)
    # ast.literal_eval does not work with numpy expressions
    input_dict[input_key] = eval(expr)  # pylint: disable=eval-used
  return input_dict


def preprocess_input_examples_arg_string(input_examples_str):
  """Parses input into dict that maps input keys to lists of tf.Example.

  Parses input string in the format of 'input_key1=[{feature_name:
  feature_list}];input_key2=[{feature_name:feature_list}];' into a dictionary
  that maps each input_key to its list of serialized tf.Example.

  Args:
    input_examples_str: A string that specifies a list of dictionaries of
    feature_names and their feature_lists for each input.
    Each input is separated by semicolon. For each input key:
      'input=[{feature_name1: feature_list1, feature_name2:feature_list2}]'
      items in feature_list can be the type of float, int, long or str.

  Returns:
    A dictionary that maps input keys to lists of serialized tf.Example.

  Raises:
    ValueError: An error when the given tf.Example is not a list.
  """
  input_dict = preprocess_input_exprs_arg_string(input_examples_str)
  for input_key, example_list in input_dict.items():
    if not isinstance(example_list, list):
      raise ValueError(
          'tf.Example input must be a list of dictionaries, but "%s" is %s' %
          (example_list, type(example_list)))
    input_dict[input_key] = [
        _create_example_string(example) for example in example_list
    ]
  return input_dict


def _create_example_string(example_dict):
  """Create a serialized tf.example from feature dictionary."""
  example = example_pb2.Example()
  for feature_name, feature_list in example_dict.items():
    if not isinstance(feature_list, list):
      raise ValueError('feature value must be a list, but %s: "%s" is %s' %
                       (feature_name, feature_list, type(feature_list)))
    if isinstance(feature_list[0], float):
      example.features.feature[feature_name].float_list.value.extend(
          feature_list)
    elif isinstance(feature_list[0], str):
      example.features.feature[feature_name].bytes_list.value.extend(
          [f.encode('utf8') for f in feature_list])
    elif isinstance(feature_list[0], bytes):
      example.features.feature[feature_name].bytes_list.value.extend(
          feature_list)
    elif isinstance(feature_list[0], six.integer_types):
      example.features.feature[feature_name].int64_list.value.extend(
          feature_list)
    else:
      raise ValueError(
          'Type %s for value %s is not supported for tf.train.Feature.' %
          (type(feature_list[0]), feature_list[0]))
  return example.SerializeToString()


def load_inputs_from_input_arg_string(inputs_str, input_exprs_str,
                                      input_examples_str):
  """Parses input arg strings and create inputs feed_dict.

  Parses '--inputs' string for inputs to be loaded from file, and parses
  '--input_exprs' string for inputs to be evaluated from python expression.
  '--input_examples' string for inputs to be created from tf.example feature
  dictionary list.

  Args:
    inputs_str: A string that specified where to load inputs. Each input is
        separated by semicolon.
        * For each input key:
            '<input_key>=<filename>' or
            '<input_key>=<filename>[<variable_name>]'
        * The optional 'variable_name' key will be set to None if not specified.
        * File specified by 'filename' will be loaded using numpy.load. Inputs
            can be loaded from only .npy, .npz or pickle files.
        * The "[variable_name]" key is optional depending on the input file type
            as descripted in more details below.
        When loading from a npy file, which always contains a numpy ndarray, the
        content will be directly assigned to the specified input tensor. If a
        variable_name is specified, it will be ignored and a warning will be
        issued.
        When loading from a npz zip file, user can specify which variable within
        the zip file to load for the input tensor inside the square brackets. If
        nothing is specified, this function will check that only one file is
        included in the zip and load it for the specified input tensor.
        When loading from a pickle file, if no variable_name is specified in the
        square brackets, whatever that is inside the pickle file will be passed
        to the specified input tensor, else SavedModel CLI will assume a
        dictionary is stored in the pickle file and the value corresponding to
        the variable_name will be used.
    input_exprs_str: A string that specifies python expressions for inputs.
        * In the format of: '<input_key>=<python expression>'.
        * numpy module is available as np.
    input_examples_str: A string that specifies tf.Example with dictionary.
        * In the format of: '<input_key>=<[{feature:value list}]>'

  Returns:
    A dictionary that maps input tensor keys to numpy ndarrays.

  Raises:
    RuntimeError: An error when a key is specified, but the input file contains
        multiple numpy ndarrays, none of which matches the given key.
    RuntimeError: An error when no key is specified, but the input file contains
        more than one numpy ndarrays.
  """
  tensor_key_feed_dict = {}

  inputs = preprocess_inputs_arg_string(inputs_str)
  input_exprs = preprocess_input_exprs_arg_string(input_exprs_str)
  input_examples = preprocess_input_examples_arg_string(input_examples_str)

  for input_tensor_key, (filename, variable_name) in inputs.items():
    data = np.load(file_io.FileIO(filename, mode='rb'), allow_pickle=True)  # pylint: disable=unexpected-keyword-arg

    # When a variable_name key is specified for the input file
    if variable_name:
      # if file contains a single ndarray, ignore the input name
      if isinstance(data, np.ndarray):
        logging.warn(
            'Input file %s contains a single ndarray. Name key \"%s\" ignored.'
            % (filename, variable_name))
        tensor_key_feed_dict[input_tensor_key] = data
      else:
        if variable_name in data:
          tensor_key_feed_dict[input_tensor_key] = data[variable_name]
        else:
          raise RuntimeError(
              'Input file %s does not contain variable with name \"%s\".' %
              (filename, variable_name))
    # When no key is specified for the input file.
    else:
      # Check if npz file only contains a single numpy ndarray.
      if isinstance(data, np.lib.npyio.NpzFile):
        variable_name_list = data.files
        if len(variable_name_list) != 1:
          raise RuntimeError(
              'Input file %s contains more than one ndarrays. Please specify '
              'the name of ndarray to use.' % filename)
        tensor_key_feed_dict[input_tensor_key] = data[variable_name_list[0]]
      else:
        tensor_key_feed_dict[input_tensor_key] = data

  # When input is a python expression:
  for input_tensor_key, py_expr_evaluated in input_exprs.items():
    if input_tensor_key in tensor_key_feed_dict:
      logging.warn(
          'input_key %s has been specified with both --inputs and --input_exprs'
          ' options. Value in --input_exprs will be used.' % input_tensor_key)
    tensor_key_feed_dict[input_tensor_key] = py_expr_evaluated

  # When input is a tf.Example:
  for input_tensor_key, example in input_examples.items():
    if input_tensor_key in tensor_key_feed_dict:
      logging.warn(
          'input_key %s has been specified in multiple options. Value in '
          '--input_examples will be used.' % input_tensor_key)
    tensor_key_feed_dict[input_tensor_key] = example
  return tensor_key_feed_dict


def show(args):
  """Function triggered by show command.

  Args:
    args: A namespace parsed from command line.
  """
  # If all tag is specified, display all information.
  if args.all:
    _show_all(args.dir)
  else:
    # If no tag is specified, display all tag_set, if no signature_def key is
    # specified, display all SignatureDef keys, else show input output tensor
    # information corresponding to the given SignatureDef key
    if args.tag_set is None:
      _show_tag_sets(args.dir)
    else:
      if args.signature_def is None:
        _show_signature_def_map_keys(args.dir, args.tag_set)
      else:
        _show_inputs_outputs(args.dir, args.tag_set, args.signature_def)


def run(args):
  """Function triggered by run command.

  Args:
    args: A namespace parsed from command line.

  Raises:
    AttributeError: An error when neither --inputs nor --input_exprs is passed
    to run command.
  """
  if not args.inputs and not args.input_exprs and not args.input_examples:
    raise AttributeError(
        'At least one of --inputs, --input_exprs or --input_examples must be '
        'required')
  tensor_key_feed_dict = load_inputs_from_input_arg_string(
      args.inputs, args.input_exprs, args.input_examples)
  run_saved_model_with_feed_dict(args.dir, args.tag_set, args.signature_def,
                                 tensor_key_feed_dict, args.outdir,
                                 args.overwrite, worker=args.worker,
                                 init_tpu=args.init_tpu, tf_debug=args.tf_debug)


def scan(args):
  """Function triggered by scan command.

  Args:
    args: A namespace parsed from command line.
  """
  if args.tag_set:
    scan_meta_graph_def(
        saved_model_utils.get_meta_graph_def(args.dir, args.tag_set))
  else:
    saved_model = saved_model_utils.read_saved_model(args.dir)
    for meta_graph_def in saved_model.meta_graphs:
      scan_meta_graph_def(meta_graph_def)


def convert_with_tensorrt(args):
  """Function triggered by 'convert tensorrt' command.

  Args:
    args: A namespace parsed from command line.
  """
  # Import here instead of at top, because this will crash if TensorRT is
  # not installed
  from tensorflow.python.compiler.tensorrt import trt_convert as trt  # pylint: disable=g-import-not-at-top

  if not args.convert_tf1_model:
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=args.max_workspace_size_bytes,
        precision_mode=args.precision_mode,
        minimum_segment_size=args.minimum_segment_size)
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=args.dir,
        input_saved_model_tags=args.tag_set.split(','),
        conversion_params=params)
    try:
      converter.convert()
    except Exception as e:
      raise RuntimeError(
          '{}. Try passing "--convert_tf1_model=True".'.format(e))
    converter.save(output_saved_model_dir=args.output_dir)
  else:
    trt.create_inference_graph(
        None,
        None,
        max_batch_size=1,
        max_workspace_size_bytes=args.max_workspace_size_bytes,
        precision_mode=args.precision_mode,
        minimum_segment_size=args.minimum_segment_size,
        is_dynamic_op=True,
        input_saved_model_dir=args.dir,
        input_saved_model_tags=args.tag_set.split(','),
        output_saved_model_dir=args.output_dir)


def aot_compile_cpu(args):
  """Function triggered by aot_compile_cpu command.

  Args:
    args: A namespace parsed from command line.
  """
  checkpoint_path = (
      args.checkpoint_path
      or os.path.join(args.dir, 'variables/variables'))
  if not args.variables_to_feed:
    variables_to_feed = []
  elif args.variables_to_feed.lower() == 'all':
    variables_to_feed = None  # We will identify them after.
  else:
    variables_to_feed = args.variables_to_feed.split(',')

  saved_model_aot_compile.aot_compile_cpu_meta_graph_def(
      checkpoint_path=checkpoint_path,
      meta_graph_def=saved_model_utils.get_meta_graph_def(
          args.dir, args.tag_set),
      signature_def_key=args.signature_def_key,
      variables_to_feed=variables_to_feed,
      output_prefix=args.output_prefix,
      target_triple=args.target_triple,
      target_cpu=args.target_cpu,
      cpp_class=args.cpp_class,
      multithreading=args.multithreading.lower() not in ('f', 'false', '0'))


def add_show_subparser(subparsers):
  """Add parser for `show`."""
  show_msg = (
      'Usage examples:\n'
      'To show all tag-sets in a SavedModel:\n'
      '$saved_model_cli show --dir /tmp/saved_model\n\n'
      'To show all available SignatureDef keys in a '
      'MetaGraphDef specified by its tag-set:\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve\n\n'
      'For a MetaGraphDef with multiple tags in the tag-set, all tags must be '
      'passed in, separated by \';\':\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve,gpu\n\n'
      'To show all inputs and outputs TensorInfo for a specific'
      ' SignatureDef specified by the SignatureDef key in a'
      ' MetaGraph.\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve'
      ' --signature_def serving_default\n\n'
      'To show all available information in the SavedModel:\n'
      '$saved_model_cli show --dir /tmp/saved_model --all')
  parser_show = subparsers.add_parser(
      'show',
      description=show_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_show.add_argument(
      '--dir',
      type=str,
      required=True,
      help='directory containing the SavedModel to inspect')
  parser_show.add_argument(
      '--all',
      action='store_true',
      help='if set, will output all information in given SavedModel')
  parser_show.add_argument(
      '--tag_set',
      type=str,
      default=None,
      help='tag-set of graph in SavedModel to show, separated by \',\'')
  parser_show.add_argument(
      '--signature_def',
      type=str,
      default=None,
      metavar='SIGNATURE_DEF_KEY',
      help='key of SignatureDef to display input(s) and output(s) for')
  parser_show.set_defaults(func=show)


def add_run_subparser(subparsers):
  """Add parser for `run`."""
  run_msg = ('Usage example:\n'
             'To run input tensors from files through a MetaGraphDef and save'
             ' the output tensors to files:\n'
             '$saved_model_cli show --dir /tmp/saved_model --tag_set serve \\\n'
             '   --signature_def serving_default \\\n'
             '   --inputs input1_key=/tmp/124.npz[x],input2_key=/tmp/123.npy '
             '\\\n'
             '   --input_exprs \'input3_key=np.ones(2)\' \\\n'
             '   --input_examples '
             '\'input4_key=[{"id":[26],"weights":[0.5, 0.5]}]\' \\\n'
             '   --outdir=/out\n\n'
             'For more information about input file format, please see:\n'
             'https://www.tensorflow.org/guide/saved_model_cli\n')
  parser_run = subparsers.add_parser(
      'run', description=run_msg, formatter_class=argparse.RawTextHelpFormatter)
  parser_run.add_argument(
      '--dir',
      type=str,
      required=True,
      help='directory containing the SavedModel to execute')
  parser_run.add_argument(
      '--tag_set',
      type=str,
      required=True,
      help='tag-set of graph in SavedModel to load, separated by \',\'')
  parser_run.add_argument(
      '--signature_def',
      type=str,
      required=True,
      metavar='SIGNATURE_DEF_KEY',
      help='key of SignatureDef to run')
  msg = ('Loading inputs from files, in the format of \'<input_key>=<filename>,'
         ' or \'<input_key>=<filename>[<variable_name>]\', separated by \';\'.'
         ' The file format can only be from .npy, .npz or pickle.')
  parser_run.add_argument('--inputs', type=str, default='', help=msg)
  msg = ('Specifying inputs by python expressions, in the format of'
         ' "<input_key>=\'<python expression>\'", separated by \';\'. '
         'numpy module is available as \'np\'. '
         'Will override duplicate input keys from --inputs option.')
  parser_run.add_argument('--input_exprs', type=str, default='', help=msg)
  msg = (
      'Specifying tf.Example inputs as list of dictionaries. For example: '
      '<input_key>=[{feature0:value_list,feature1:value_list}]. Use ";" to '
      'separate input keys. Will override duplicate input keys from --inputs '
      'and --input_exprs option.')
  parser_run.add_argument('--input_examples', type=str, default='', help=msg)
  parser_run.add_argument(
      '--outdir',
      type=str,
      default=None,
      help='if specified, output tensor(s) will be saved to given directory')
  parser_run.add_argument(
      '--overwrite',
      action='store_true',
      help='if set, output file will be overwritten if it already exists.')
  parser_run.add_argument(
      '--tf_debug',
      action='store_true',
      help='if set, will use TensorFlow Debugger (tfdbg) to watch the '
           'intermediate Tensors and runtime GraphDefs while running the '
           'SavedModel.')
  parser_run.add_argument(
      '--worker',
      type=str,
      default=None,
      help='if specified, a Session will be run on the worker. '
           'Valid worker specification is a bns or gRPC path.')
  parser_run.add_argument(
      '--init_tpu',
      action='store_true',
      default=None,
      help='if specified, tpu.initialize_system will be called on the Session. '
           'This option should be only used if the worker is a TPU job.')
  parser_run.set_defaults(func=run)


def add_scan_subparser(subparsers):
  """Add parser for `scan`."""
  scan_msg = ('Usage example:\n'
              'To scan for denylisted ops in SavedModel:\n'
              '$saved_model_cli scan --dir /tmp/saved_model\n'
              'To scan a specific MetaGraph, pass in --tag_set\n')
  parser_scan = subparsers.add_parser(
      'scan',
      description=scan_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_scan.add_argument(
      '--dir',
      type=str,
      required=True,
      help='directory containing the SavedModel to execute')
  parser_scan.add_argument(
      '--tag_set',
      type=str,
      help='tag-set of graph in SavedModel to scan, separated by \',\'')
  parser_scan.set_defaults(func=scan)


def add_convert_subparser(subparsers):
  """Add parser for `convert`."""
  convert_msg = ('Usage example:\n'
                 'To convert the SavedModel to one that have TensorRT ops:\n'
                 '$saved_model_cli convert \\\n'
                 '   --dir /tmp/saved_model \\\n'
                 '   --tag_set serve \\\n'
                 '   --output_dir /tmp/saved_model_trt \\\n'
                 '   tensorrt \n')
  parser_convert = subparsers.add_parser(
      'convert',
      description=convert_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_convert.add_argument(
      '--dir',
      type=str,
      required=True,
      help='directory containing the SavedModel to convert')
  parser_convert.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='output directory for the converted SavedModel')
  parser_convert.add_argument(
      '--tag_set',
      type=str,
      required=True,
      help='tag-set of graph in SavedModel to convert, separated by \',\'')
  convert_subparsers = parser_convert.add_subparsers(
      title='conversion methods',
      description='valid conversion methods',
      help='the conversion to run with the SavedModel')
  parser_convert_with_tensorrt = convert_subparsers.add_parser(
      'tensorrt',
      description='Convert the SavedModel with Tensorflow-TensorRT integration',
      formatter_class=argparse.RawTextHelpFormatter)
  parser_convert_with_tensorrt.add_argument(
      '--max_workspace_size_bytes',
      type=int,
      default=2 << 20,
      help=('the maximum GPU temporary memory which the TRT engine can use at '
            'execution time'))
  parser_convert_with_tensorrt.add_argument(
      '--precision_mode',
      type=str,
      default='FP32',
      help='one of FP32, FP16 and INT8')
  parser_convert_with_tensorrt.add_argument(
      '--minimum_segment_size',
      type=int,
      default=3,
      help=('the minimum number of nodes required for a subgraph to be replaced'
            'in a TensorRT node'))
  parser_convert_with_tensorrt.add_argument(
      '--convert_tf1_model',
      type=bool,
      default=False,
      help='support TRT conversion for TF1 models')
  parser_convert_with_tensorrt.set_defaults(func=convert_with_tensorrt)


def add_aot_compile_cpu_subparser(subparsers):
  """Add parser for `aot_compile_cpu`."""
  compile_msg = '\n'.join(
      ['Usage example:',
       'To compile a SavedModel signature via (CPU) XLA AOT:',
       '$saved_model_cli aot_compile_cpu \\',
       '   --dir /tmp/saved_model \\',
       '   --tag_set serve \\',
       '   --output_dir /tmp/saved_model_xla_aot',
       '', '',
       'Note: Additional XLA compilation options are available by setting the ',
       'XLA_FLAGS environment variable.  See the XLA debug options flags for ',
       'all the options: ',
       '  {}'.format(_XLA_DEBUG_OPTIONS_URL),
       '',
       'For example, to disable XLA fast math when compiling:',
       '',
       'XLA_FLAGS="--xla_cpu_enable_fast_math=false" $saved_model_cli '
       'aot_compile_cpu ...',
       '',
       'Some possibly useful flags:',
       '  --xla_cpu_enable_fast_math=false',
       '  --xla_force_host_platform_device_count=<num threads>',
       '    (useful in conjunction with disabling multi threading)'
      ])

  parser_compile = subparsers.add_parser(
      'aot_compile_cpu',
      description=compile_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_compile.add_argument(
      '--dir',
      type=str,
      required=True,
      help='directory containing the SavedModel to convert')
  parser_compile.add_argument(
      '--output_prefix',
      type=str,
      required=True,
      help=('output directory + filename prefix for the resulting header(s) '
            'and object file(s)'))
  parser_compile.add_argument(
      '--tag_set',
      type=str,
      required=True,
      help='tag-set of graph in SavedModel to convert, separated by \',\'')
  parser_compile.add_argument(
      '--signature_def_key',
      type=str,
      default=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
      help=('signature_def key to use.  '
            'default: DEFAULT_SERVING_SIGNATURE_DEF_KEY'))
  parser_compile.add_argument(
      '--target_triple',
      type=str,
      default='x86_64-pc-linux',
      help=('Target triple for LLVM during AOT compilation.  Examples: '
            'x86_64-none-darwin, x86_64-apple-ios, arm64-none-ios, '
            'armv7-none-android.  More examples are available in tfcompile.bzl '
            'in the tensorflow codebase.'))
  parser_compile.add_argument(
      '--target_cpu',
      type=str,
      default='',
      help=('Target cpu name for LLVM during AOT compilation.  Examples: '
            'x86_64, skylake, haswell, westmere, <empty> (unknown).  For '
            'a complete list of options, run (for x86 targets): '
            '`llc -march=x86 -mcpu=help`'))
  parser_compile.add_argument(
      '--checkpoint_path',
      type=str,
      default=None,
      help='Custom checkpoint to use (default: use the SavedModel variables)')
  parser_compile.add_argument(
      '--cpp_class',
      type=str,
      required=True,
      help=('The name of the generated C++ class, wrapping the generated '
            'function.  The syntax of this flag is '
            '[[<optional_namespace>::],...]<class_name>.  This mirrors the '
            'C++ syntax for referring to a class, where multiple namespaces '
            'may precede the class name, separated by double-colons.  '
            'The class will be generated in the given namespace(s), or if no '
            'namespaces are given, within the global namespace.'))
  parser_compile.add_argument(
      '--variables_to_feed',
      type=str,
      default='',
      help=('The names of variables that will be fed into the network.  '
            'Options are: empty (default; all variables are frozen, none may '
            'be fed), \'all\' (all variables may be fed), or a '
            'comma-delimited list of names of variables that may be fed.  In '
            'the last case, the non-fed variables will be frozen in the graph.'
            '**NOTE** Any variables passed to `variables_to_feed` *must be set '
            'by the user*.  These variables will NOT be frozen and their '
            'values will be uninitialized in the compiled object '
            '(this applies to all input arguments from the signature as '
            'well).'))
  parser_compile.add_argument(
      '--multithreading',
      type=str,
      default='False',
      help=('Enable multithreading in the compiled computation.  '
            'Note that if using this option, the resulting object files '
            'may have external dependencies on multithreading libraries '
            'like nsync.'))

  parser_compile.set_defaults(func=aot_compile_cpu)


def create_parser():
  """Creates a parser that parse the command line arguments.

  Returns:
    A namespace parsed from command line arguments.
  """
  parser = argparse.ArgumentParser(
      description='saved_model_cli: Command-line interface for SavedModel')
  parser.add_argument('-v', '--version', action='version', version='0.1.0')

  subparsers = parser.add_subparsers(
      title='commands', description='valid commands', help='additional help')

  # show command
  add_show_subparser(subparsers)

  # run command
  add_run_subparser(subparsers)

  # scan command
  add_scan_subparser(subparsers)

  # tensorrt convert command
  add_convert_subparser(subparsers)

  # aot_compile_cpu command
  add_aot_compile_cpu_subparser(subparsers)

  return parser


def main():
  logging.set_verbosity(logging.INFO)
  parser = create_parser()
  args = parser.parse_args()
  if not hasattr(args, 'func'):
    parser.error('too few arguments')
  args.func(args)


if __name__ == '__main__':
  sys.exit(main())
