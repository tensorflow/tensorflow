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

import argparse

import ast
import os
import re

from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
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


# Custom SavedModel CLI flags
_SMCLI_DIR = flags.DEFINE_string(
    name='dir', default=None, help='Directory containing the SavedModel.')

_SMCLI_ALL = flags.DEFINE_bool(
    name='all', default=False,
    help='If set, outputs all available information in the given SavedModel.')

_SMCLI_TAG_SET = flags.DEFINE_string(
    name='tag_set', default=None,
    help='Comma-separated set of tags that identify variant graphs in the '
    'SavedModel.')

_SMCLI_SIGNATURE_DEF = flags.DEFINE_string(
    name='signature_def', default=None,
    help='Specifies a SignatureDef (by key) within the SavedModel to display '
    'input(s) and output(s) for.')

_SMCLI_LIST_OPS = flags.DEFINE_bool(
    name='list_ops', default=False,
    help='If set, will output ops used by a MetaGraphDef specified by tag_set.')

_SMCLI_INPUTS = flags.DEFINE_string(
    name='inputs', default='',
    help='Specifies input data files to pass to numpy.load(). Format should be '
    '\'<input_key>=<filename>\' or \'<input_key>=<filename>[<variable_name>]\','
    ' separated by \';\'. File formats are limited to .npy, .npz, or pickle.')

_SMCLI_INPUT_EXPRS = flags.DEFINE_string(
    name='input_exprs', default='',
    help='Specifies Python literal expressions or numpy functions. Format '
    'should be "<input_key>=\'<python_expression>\'", separated by \';\'. Numpy'
    ' can be accessed with \'np\'. Note that expressions are passed to '
    'literal_eval(), making this flag susceptible to code injection. Overrides '
    'duplicate input keys provided with the --inputs flag.')

_SMCLI_INPUT_EXAMPLES = flags.DEFINE_string(
    name='input_examples', default='',
    help='Specifies tf.train.Example objects as inputs. Format should be '
    '\'<input_key>=[{{feature0:value_list,feature1:value_list}}]\', where input'
    ' keys are separated by \';\'. Overrides duplicate input keys provided with'
    ' the --inputs and --input_exprs flags.')

_SMCLI_OUTDIR = flags.DEFINE_string(
    name='outdir', default=None,
    help='If specified, writes CLI output to the given directory.')

_SMCLI_OVERWRITE = flags.DEFINE_bool(
    name='overwrite', default=False,
    help='If set, overwrites output file if it already exists.')

_SMCLI_TF_DEBUG = flags.DEFINE_bool(
    name='tf_debug', default=False,
    help='If set, uses the Tensorflow Debugger (tfdbg) to watch intermediate '
    'Tensors and runtime GraphDefs while running the SavedModel.')

_SMCLI_WORKER = flags.DEFINE_string(
    name='worker', default=None,
    help='If specified, runs the session on the given worker (bns or gRPC '
    'path).')

_SMCLI_INIT_TPU = flags.DEFINE_bool(
    name='init_tpu', default=False,
    help='If set, calls tpu.initialize_system() on the session. '
                  'Should only be set if the specified worker is a TPU job.')

_SMCLI_USE_TFRT = flags.DEFINE_bool(
    name='use_tfrt', default=False,
    help='If set, runs a TFRT session, instead of a TF1 session.')

_SMCLI_OP_DENYLIST = flags.DEFINE_string(
    name='op_denylist', default=None,
    help='If specified, detects and reports the given ops. List of ops should '
    'be comma-separated. If not specified, the default list of ops is '
    '[WriteFile, ReadFile, PrintV2]. To specify an empty list, pass in the '
    'empty string.')

_SMCLI_OUTPUT_DIR = flags.DEFINE_string(
    name='output_dir', default=None,
    help='Output directory for the SavedModel.')

_SMCLI_MAX_WORKSPACE_SIZE_BYTES = flags.DEFINE_integer(
    name='max_workspace_size_bytes', default=2 << 20,
    help='The maximum temporary GPU memory which the TensorRT engine can use at'
    ' execution time.')

_SMCLI_PRECISION_MODE = flags.DEFINE_enum(
    name='precision_mode', default='FP32', enum_values=['FP32', 'FP16', 'INT8'],
    help='TensorRT data precision. One of FP32, FP16, or INT8.')

_SMCLI_MINIMUM_SEGMENT_SIZE = flags.DEFINE_integer(
    name='minimum_segment_size', default=3,
    help='The minimum number of nodes required for a subgraph to be replaced in'
    ' a TensorRT node.')

_SMCLI_CONVERT_TF1_MODEL = flags.DEFINE_bool(
    name='convert_tf1_model', default=False,
    help='Support TensorRT conversion for TF1 models.')

_SMCLI_OUTPUT_PREFIX = flags.DEFINE_string(
    name='output_prefix', default=None,
    help='Output directory + filename prefix for the resulting header(s) and '
    'object file(s).')

_SMCLI_SIGNATURE_DEF_KEY = flags.DEFINE_string(
    name='signature_def_key',
    default=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    help='SavedModel SignatureDef key to use.')

_SMCLI_CHECKPOINT_PATH = flags.DEFINE_string(
    name='checkpoint_path', default=None,
    help='Custom checkpoint to use. Uses SavedModel variables by default.')

_SMCLI_VARIABLES_TO_FEED = flags.DEFINE_string(
    name='variables_to_feed', default='',
    help='Names of the variables that will be fed into the SavedModel graph. '
    'Pass in \'\' to feed no variables, \'all\' to feed all variables, or a '
    'comma-separated list of variable names. Variables not fed will be frozen. '
    '*NOTE* Variables passed here must be set *by the user*. These variables '
    'will NOT be frozen, and their values will be uninitialized in the compiled'
    ' object.')

_SMCLI_TARGET_TRIPLE = flags.DEFINE_string(
    name='target_triple', default='',
    help='Triple identifying a target variation, containing information such as'
    ' processor architecture, vendor, operating system, and environment. '
    'Defaults to \'x86_64-pc-linux\'.')

_SMCLI_TARGET_CPU = flags.DEFINE_string(
    name='target_cpu', default='',
    help='Target CPU name for LLVM during AOT compilation. Examples include '
    '\'x86_64\', \'skylake\', \'haswell\', \'westmere\', \'\' (unknown).')

_SMCLI_CPP_CLASS = flags.DEFINE_string(
    name='cpp_class', default=None,
    help='The name of the generated C++ class, wrapping the generated function.'
    ' Format should be [[<optional_namespace>::],...]<class_name>, i.e. the '
    'same syntax as C++ for specifying a class. This class will be generated in'
    ' the given namespace(s), or, if none are specified, the global namespace.')

_SMCLI_MULTITHREADING = flags.DEFINE_string(
    name='multithreading', default='False',
    help='Enable multithreading in the compiled computation. Note that with '
    'this flag enabled, the resulting object files may have external '
    'dependencies on multithreading libraries, such as \'nsync\'.')

command_required_flags = {
    'show': ['dir'],
    'run': ['dir', 'tag_set', 'signature_def'],
    'scan': ['dir'],
    'convert': ['dir', 'output_dir', 'tag_set'],
    'freeze_model': ['dir', 'output_prefix', 'tag_set'],
    'aot_compile_cpu': ['cpp_class'],
}


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


def _get_ops_in_metagraph(meta_graph_def):
  """Returns a set of the ops in the MetaGraph.

  Returns the set of all the ops used in the MetaGraphDef indicated by the
  tag_set stored in SavedModel directory.

  Args:
    meta_graph_def: MetaGraphDef to list the ops of.

  Returns:
    A set of ops.
  """
  return set(meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))


def _show_ops_in_metagraph(saved_model_dir, tag_set):
  """Prints the ops in the MetaGraph.

  Prints all the ops used in the MetaGraphDef indicated by the tag_set stored in
  SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef in string format, separated by
      ','. For tag-set contains multiple tags, all tags must be passed in.
  """
  meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,
                                                        tag_set)
  all_ops_set = _get_ops_in_metagraph(meta_graph_def)
  print(
      'The MetaGraph with tag set %s contains the following ops:' %
      meta_graph_def.meta_info_def.tags, all_ops_set)


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

  Raises:
    ValueError if `signature_def_key` is not found in the MetaGraphDef.
  """
  if signature_def_key not in meta_graph_def.signature_def:
    raise ValueError(
        f'Could not find signature "{signature_def_key}". Please choose from: '
        f'{", ".join(meta_graph_def.signature_def.keys())}')
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

  print('\nConcrete Functions:', end='')
  children = list(
      save._AugmentedGraphView(trackable_object)  # pylint: disable=protected-access
      .list_children(trackable_object))
  children = sorted(children, key=lambda x: x.name)
  for name, child in children:
    concrete_functions = []
    if isinstance(child, defun.ConcreteFunction):
      concrete_functions.append(child)
    elif isinstance(child, def_function.Function):
      concrete_functions.extend(
          child._list_all_concrete_functions_for_serialization())  # pylint: disable=protected-access
    else:
      continue
    print('\n  Function Name: \'%s\'' % name)
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
    if isinstance(element, str):
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
  """Prints tag-set, ops, SignatureDef, and Inputs/Outputs of SavedModel.

  Prints all tag-set, ops, SignatureDef and Inputs/Outputs information stored in
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
    _show_ops_in_metagraph(saved_model_dir, tag_set)
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


def _get_op_denylist_set(op_denylist):
  # Note: Discard empty ops so that "" can mean the empty denylist set.
  set_of_denylisted_ops = set([op for op in op_denylist.split(',') if op])
  return set_of_denylisted_ops


def scan_meta_graph_def(meta_graph_def, op_denylist):
  """Scans meta_graph_def and reports if there are ops on denylist.

  Print ops if they are on denylist, or print success if no denylisted ops
  found.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    op_denylist: set of ops to scan for.
  """
  ops_in_metagraph = set(
      meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))
  denylisted_ops = op_denylist & ops_in_metagraph
  if denylisted_ops:
    # TODO(yifeif): print more warnings
    print(
        'MetaGraph with tag set %s contains the following denylisted ops:' %
        meta_graph_def.meta_info_def.tags, denylisted_ops)
  else:
    print(
        'MetaGraph with tag set %s does not contain the default denylisted ops:'
        % meta_graph_def.meta_info_def.tags, op_denylist)


def run_saved_model_with_feed_dict(saved_model_dir,
                                   tag_set,
                                   signature_def_key,
                                   input_tensor_key_feed_dict,
                                   outdir,
                                   overwrite_flag,
                                   worker=None,
                                   init_tpu=False,
                                   use_tfrt=False,
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
    use_tfrt: If true, TFRT session will be used.
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

  config = None
  if use_tfrt:
    logging.info('Using TFRT session.')
    config = config_pb2.ConfigProto(
        experimental=config_pb2.ConfigProto.Experimental(use_tfrt=True))
  with session.Session(worker, graph=ops_lib.Graph(), config=config) as sess:
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


def preprocess_input_exprs_arg_string(input_exprs_str, safe=True):
  """Parses input arg into dictionary that maps input key to python expression.

  Parses input string in the format of 'input_key=<python expression>' into a
  dictionary that maps each input_key to its python expression.

  Args:
    input_exprs_str: A string that specifies python expression for input keys.
      Each input is separated by semicolon. For each input key:
        'input_key=<python expression>'
    safe: Whether to evaluate the python expression as literals or allow
      arbitrary calls (e.g. numpy usage).

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
    if safe:
      try:
        input_dict[input_key] = ast.literal_eval(expr)
      except Exception as exc:
        raise RuntimeError(
            f'Expression "{expr}" is not a valid python literal.') from exc
    else:
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
    elif isinstance(feature_list[0], int):
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


def show():
  """Function triggered by show command."""
  # If all tag is specified, display all information.
  if _SMCLI_ALL.value:
    _show_all(_SMCLI_DIR.value)
  else:
    # If no tag is specified, display all tag_sets.
    # If a tag set is specified:
    # # If list_ops is set, display all ops in the specified MetaGraphDef.
    # # If no signature_def key is specified, display all SignatureDef keys.
    # # If a signature_def is specified, show its corresponding input output
    # # tensor information.
    if _SMCLI_TAG_SET.value is None:
      if _SMCLI_LIST_OPS.value:
        print('--list_ops must be paired with a tag-set or with --all.')
      _show_tag_sets(_SMCLI_DIR.value)
    else:
      if _SMCLI_LIST_OPS.value:
        _show_ops_in_metagraph(_SMCLI_DIR.value, _SMCLI_TAG_SET.value)
      if _SMCLI_SIGNATURE_DEF.value is None:
        _show_signature_def_map_keys(_SMCLI_DIR.value, _SMCLI_TAG_SET.value)
      else:
        _show_inputs_outputs(
            _SMCLI_DIR.value, _SMCLI_TAG_SET.value, _SMCLI_SIGNATURE_DEF.value)


def run():
  """Function triggered by run command.

  Raises:
    AttributeError: An error when neither --inputs nor --input_exprs is passed
    to run command.
  """
  if not _SMCLI_INPUTS.value and not _SMCLI_INPUT_EXPRS.value and not _SMCLI_INPUT_EXAMPLES.value:
    raise AttributeError(
        'At least one of --inputs, --input_exprs or --input_examples must be '
        'required')
  tensor_key_feed_dict = load_inputs_from_input_arg_string(
      _SMCLI_INPUTS.value,
      _SMCLI_INPUT_EXPRS.value,
      _SMCLI_INPUT_EXAMPLES.value)
  run_saved_model_with_feed_dict(
      _SMCLI_DIR.value,
      _SMCLI_TAG_SET.value,
      _SMCLI_SIGNATURE_DEF.value,
      tensor_key_feed_dict,
      _SMCLI_OUTDIR.value,
      _SMCLI_OVERWRITE.value,
      worker=_SMCLI_WORKER.value,
      init_tpu=_SMCLI_INIT_TPU.value,
      use_tfrt=_SMCLI_USE_TFRT.value,
      tf_debug=_SMCLI_TF_DEBUG.value)


def scan():
  """Function triggered by scan command."""
  if _SMCLI_TAG_SET.value and _SMCLI_OP_DENYLIST.value:
    scan_meta_graph_def(
        saved_model_utils.get_meta_graph_def(
            _SMCLI_DIR.value, _SMCLI_TAG_SET.value),
        _get_op_denylist_set(_SMCLI_OP_DENYLIST.value))
  elif _SMCLI_TAG_SET.value:
    scan_meta_graph_def(
        saved_model_utils.get_meta_graph_def(
            _SMCLI_DIR.value, _SMCLI_TAG_SET.value),
        _OP_DENYLIST)
  else:
    saved_model = saved_model_utils.read_saved_model(_SMCLI_DIR.value)
    if _SMCLI_OP_DENYLIST.value:
      for meta_graph_def in saved_model.meta_graphs:
        scan_meta_graph_def(meta_graph_def,
                            _get_op_denylist_set(_SMCLI_OP_DENYLIST.value))
    else:
      for meta_graph_def in saved_model.meta_graphs:
        scan_meta_graph_def(meta_graph_def, _OP_DENYLIST)


def convert_with_tensorrt():
  """Function triggered by 'convert tensorrt' command."""
  # Import here instead of at top, because this will crash if TensorRT is
  # not installed
  from tensorflow.python.compiler.tensorrt import trt_convert as trt  # pylint: disable=g-import-not-at-top

  if not _SMCLI_CONVERT_TF1_MODEL.value:
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value,
        precision_mode=_SMCLI_PRECISION_MODE.value,
        minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value)
    try:
      converter = trt.TrtGraphConverterV2(
          input_saved_model_dir=_SMCLI_DIR.value,
          input_saved_model_tags=_SMCLI_TAG_SET.value.split(','),
          **params._asdict())
      converter.convert()
    except Exception as exc:
      raise RuntimeError(
          '{}. Try passing "--convert_tf1_model=True".'.format(exc)) from exc
    converter.save(output_saved_model_dir=_SMCLI_OUTPUT_DIR.value)
  else:
    trt.create_inference_graph(
        None,
        None,
        max_batch_size=1,
        max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value,
        precision_mode=_SMCLI_PRECISION_MODE.value,
        minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value,
        is_dynamic_op=True,
        input_saved_model_dir=_SMCLI_DIR.value,
        input_saved_model_tags=_SMCLI_TAG_SET.value.split(','),
        output_saved_model_dir=_SMCLI_OUTPUT_DIR.value)


def freeze_model():
  """Function triggered by freeze_model command."""
  checkpoint_path = (
      _SMCLI_CHECKPOINT_PATH.value
      or os.path.join(_SMCLI_DIR.value, 'variables/variables'))
  if not _SMCLI_VARIABLES_TO_FEED.value:
    variables_to_feed = []
  elif _SMCLI_VARIABLES_TO_FEED.value.lower() == 'all':
    variables_to_feed = None  # We will identify them after.
  else:
    variables_to_feed = _SMCLI_VARIABLES_TO_FEED.value.split(',')

  saved_model_aot_compile.freeze_model(
      checkpoint_path=checkpoint_path,
      meta_graph_def=saved_model_utils.get_meta_graph_def(
          _SMCLI_DIR.value, _SMCLI_TAG_SET.value),
      signature_def_key=_SMCLI_SIGNATURE_DEF_KEY.value,
      variables_to_feed=variables_to_feed,
      output_prefix=_SMCLI_OUTPUT_PREFIX.value)


def aot_compile_cpu():
  """Function triggered by aot_compile_cpu command."""
  checkpoint_path = (
      _SMCLI_CHECKPOINT_PATH.value
      or os.path.join(_SMCLI_DIR.value, 'variables/variables'))
  if not _SMCLI_VARIABLES_TO_FEED.value:
    variables_to_feed = []
  elif _SMCLI_VARIABLES_TO_FEED.value.lower() == 'all':
    variables_to_feed = None  # We will identify them after.
  else:
    variables_to_feed = _SMCLI_VARIABLES_TO_FEED.value.split(',')

  saved_model_aot_compile.aot_compile_cpu_meta_graph_def(
      checkpoint_path=checkpoint_path,
      meta_graph_def=saved_model_utils.get_meta_graph_def(
          _SMCLI_DIR.value, _SMCLI_TAG_SET.value),
      signature_def_key=_SMCLI_SIGNATURE_DEF_KEY.value,
      variables_to_feed=variables_to_feed,
      output_prefix=_SMCLI_OUTPUT_PREFIX.value,
      target_triple=_SMCLI_TARGET_TRIPLE.value,
      target_cpu=_SMCLI_TARGET_CPU.value,
      cpp_class=_SMCLI_CPP_CLASS.value,
      multithreading=(
          _SMCLI_MULTITHREADING.value.lower() not in ('f', 'false', '0')))


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
      'To show all ops in a MetaGraph.\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve'
      ' --list_ops\n\n'
      'To show all available information in the SavedModel:\n'
      '$saved_model_cli show --dir /tmp/saved_model --all')
  parser_show = subparsers.add_parser(
      'show',
      description=show_msg,
      formatter_class=argparse.RawTextHelpFormatter)
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
  parser_run.set_defaults(func=run)


def add_scan_subparser(subparsers):
  """Add parser for `scan`."""
  scan_msg = ('Usage example:\n'
              'To scan for default denylisted ops in SavedModel:\n'
              '$saved_model_cli scan --dir /tmp/saved_model\n'
              'To scan for a specific set of ops in SavedModel:\n'
              '$saved_model_cli scan --dir /tmp/saved_model --op_denylist '
              'OpName,OpName,OpName\n'
              'To scan a specific MetaGraph, pass in --tag_set\n')
  parser_scan = subparsers.add_parser(
      'scan',
      description=scan_msg,
      formatter_class=argparse.RawTextHelpFormatter)
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
  convert_subparsers = parser_convert.add_subparsers(
      title='conversion methods',
      description='valid conversion methods',
      help='the conversion to run with the SavedModel')
  parser_convert_with_tensorrt = convert_subparsers.add_parser(
      'tensorrt',
      description='Convert the SavedModel with Tensorflow-TensorRT integration',
      formatter_class=argparse.RawTextHelpFormatter)
  parser_convert_with_tensorrt.set_defaults(func=convert_with_tensorrt)


def add_freeze_model_subparser(subparsers):
  """Add parser for `freeze_model`."""
  compile_msg = '\n'.join(
      ['Usage example:',
       'To freeze a SavedModel in preparation for tfcompile:',
       '$saved_model_cli freeze_model \\',
       '   --dir /tmp/saved_model \\',
       '   --tag_set serve \\',
       '   --output_prefix /tmp/saved_model_xla_aot',
      ])

  parser_compile = subparsers.add_parser(
      'freeze_model',
      description=compile_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_compile.set_defaults(func=freeze_model)


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
       'XLA_FLAGS="--xla_cpu_enable_fast_math=false" $saved_model_cli ',
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

  parser_compile.set_defaults(func=aot_compile_cpu)


def create_parser():
  """Creates a parser that parse the command line arguments.

  Returns:
    A namespace parsed from command line arguments.
  """
  parser = argparse_flags.ArgumentParser(
      description='saved_model_cli: Command-line interface for SavedModel',
      conflict_handler='resolve')
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

  # freeze_model command
  add_freeze_model_subparser(subparsers)
  return parser


def main():
  logging.set_verbosity(logging.INFO)

  def smcli_main(argv):
    parser = create_parser()
    if len(argv) < 2:
      parser.error('Too few arguments.')
    flags.mark_flags_as_required(command_required_flags[argv[1]])
    args = parser.parse_args()
    args.func()

  app.run(smcli_main)


if __name__ == '__main__':
  main()
