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

If TensorFlow is installed on your system through pip, the 'saved_model_cli'
binary can be invoked directly from command line.

At a high level, SavedModel CLI allows users to both inspect and execute
computations on a MetaGraphDef in a SavedModel. These are done through `show`
and `run` commands. Following is the usage of the two commands. SavedModel
CLI will also display these information with -h option.

'show' command usage: saved_model_cli show [-h] --dir DIR [--tag_set TAG_SET]
                          [--signature_def SIGNATURE_DEF_KEY]
Examples:
To show all available tag-sets in the SavedModel:
  $saved_model_cli show --dir /tmp/saved_model

To show all available SignatureDef keys in a MetaGraphDef specified by its
tag-set:
  $saved_model_cli show --dir /tmp/saved_model --tag_set serve
For a MetaGraphDef with multiple tags in the tag-set, all tags must be passed
in, separated by ',':
  $saved_model_cli show --dir /tmp/saved_model --tag_set serve,gpu

To show all inputs and outputs TensorInfo for a specific SignatureDef specified
by the SignatureDef key in a MetaGraphDef:
  $saved_model_cli show --dir /tmp/saved_model --tag_set serve
  --signature_def serving_default
Example output:
  The given SavedModel SignatureDef contains the following input(s):
  inputs['input0'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
  inputs['input1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
  The given SavedModel SignatureDef contains the following output(s):
  outputs['output'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
  Method name is: tensorflow/serving/regress

To show all available information in the SavedModel:
  $saved_model_cli show --dir /tmp/saved_model --all

'run' command usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET
                         --signature_def SIGNATURE_DEF_KEY --inputs INPUTS
                         [--outdir OUTDIR] [--overwrite]
Examples:
To run input tensors from files through a MetaGraphDef and save the output
tensors to files:
  $saved_model_cli run --dir /tmp/saved_model --tag_set serve
  --signature_def serving_default --inputs x:0=/tmp/124.npz,x2=/tmp/123.npy
  --outdir /tmp/out

To observe the intermediate Tensor values in the runtime graph, use the
--tf_debug flag, e.g.:
  $saved_model_cli run --dir /tmp/saved_model --tag_set serve
  --signature_def serving_default --inputs x:0=/tmp/124.npz,x2=/tmp/123.npy
  --outdir /tmp/out --tf_debug

To build this tool from source, run:
  $bazel build tensorflow/python/tools:saved_model_cli

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import warnings

import numpy as np

from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.platform import app
from tensorflow.python.saved_model import loader


def _show_tag_sets(saved_model_dir):
  """Prints the tag-sets stored in SavedModel directory.

  Prints all the tag-sets for MetaGraphs stored in SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
  tag_sets = reader.get_saved_model_tag_sets(saved_model_dir)
  print('The given SavedModel contains the following tag-sets:')
  for tag_set in sorted(tag_sets):
    print(', '.join(sorted(tag_set)))


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
  return signature_def_utils.get_signature_def_by_key(meta_graph_def,
                                                      signature_def_key).inputs


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
  return signature_def_utils.get_signature_def_by_key(meta_graph_def,
                                                      signature_def_key).outputs


def _show_inputs_outputs(saved_model_dir, tag_set, signature_def_key):
  """Prints input and output TensorInfos.

  Prints the details of input and output TensorInfos for the SignatureDef mapped
  by the given signature_def_key.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef, in string format, separated by
        ','. For tag-set contains multiple tags, all tags must be passed in.
    signature_def_key: A SignatureDef key string.
  """
  meta_graph_def = get_meta_graph_def(saved_model_dir, tag_set)
  inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)
  outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)

  print('The given SavedModel SignatureDef contains the following input(s):')
  for input_key, input_tensor in sorted(inputs_tensor_info.items()):
    print('inputs[\'%s\'] tensor_info:' % input_key)
    _print_tensor_info(input_tensor)

  print('The given SavedModel SignatureDef contains the following output(s):')
  for output_key, output_tensor in sorted(outputs_tensor_info.items()):
    print('outputs[\'%s\'] tensor_info:' % output_key)
    _print_tensor_info(output_tensor)

  print('Method name is: %s' %
        meta_graph_def.signature_def[signature_def_key].method_name)


def _print_tensor_info(tensor_info):
  """Prints details of the given tensor_info.

  Args:
    tensor_info: TensorInfo object to be printed.
  """
  print('    dtype: ' + types_pb2.DataType.keys()[tensor_info.dtype])
  # Display shape as tuple.
  if tensor_info.tensor_shape.unknown_rank:
    shape = 'unknown_rank'
  else:
    dims = [str(dim.size) for dim in tensor_info.tensor_shape.dim]
    shape = ', '.join(dims)
    shape = '(' + shape + ')'
  print('    shape: ' + shape)
  print('    name: ' + tensor_info.name)


def _show_all(saved_model_dir):
  """Prints tag-set, SignatureDef and Inputs/Outputs information in SavedModel.

  Prints all tag-set, SignatureDef and Inputs/Outputs information stored in
  SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
  tag_sets = reader.get_saved_model_tag_sets(saved_model_dir)
  for tag_set in sorted(tag_sets):
    tag_set = ', '.join(tag_set)
    print('\nMetaGraphDef with tag-set: \'' + tag_set +
          '\' contains the following SignatureDefs:')

    signature_def_map = get_signature_def_map(saved_model_dir, tag_set)
    for signature_def_key in sorted(signature_def_map.keys()):
      print('\nsignature_def[\'' + signature_def_key + '\']:')
      _show_inputs_outputs(saved_model_dir, tag_set, signature_def_key)


def get_meta_graph_def(saved_model_dir, tag_set):
  """Gets MetaGraphDef from SavedModel.

  Returns the MetaGraphDef for the given tag-set and SavedModel directory.

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
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set(tag_set.split(','))
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def

  raise RuntimeError('MetaGraphDef associated with tag-set ' + tag_set +
                     ' could not be found in SavedModel')


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
  meta_graph = get_meta_graph_def(saved_model_dir, tag_set)
  return meta_graph.signature_def


def run_saved_model_with_feed_dict(saved_model_dir, tag_set, signature_def_key,
                                   input_tensor_key_feed_dict, outdir,
                                   overwrite_flag, tf_debug=False):
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
    tf_debug: A boolean flag to use TensorFlow Debugger (TFDBG) to observe the
        intermediate Tensor values and runtime GraphDefs while running the
        SavedModel.

  Raises:
    RuntimeError: An error when output file already exists and overwrite is not
    enabled.
  """
  # Get a list of output tensor names.
  meta_graph_def = get_meta_graph_def(saved_model_dir, tag_set)

  # Re-create feed_dict based on input tensor name instead of key as session.run
  # uses tensor name.
  inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)
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

  with session.Session(graph=ops_lib.Graph()) as sess:
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


def preprocess_input_arg_string(inputs_str):
  """Parses input arg into dictionary that maps input to file/variable tuple.

  Parses input string in the format of, for example,
  "input1=filename1[variable_name1],input2=filename2" into a
  dictionary looks like
  {'input_key1': (filename1, variable_name1),
   'input_key2': (file2, None)}
  , which maps input keys to a tuple of file name and varaible name(None if
  empty).

  Args:
    inputs_str: A string that specified where to load inputs. Each input is
        separated by comma.
        * If the command line arg for inputs is quoted and contains
            whitespace(s), all whitespaces will be ignored.
        * For each input key:
            'input=filename<[variable_name]>'
        * The "[variable_name]" key is optional. Will be set to None if not
            specified.

  Returns:
    A dictionary that maps input keys to a tuple of file name and varaible name.

  Raises:
    RuntimeError: An error when the given input is in a bad format.
  """
  input_dict = {}
  inputs_raw = inputs_str.split(',')
  for input_raw in filter(bool, inputs_raw):  # skip empty strings
    # Remove quotes and whitespaces
    input_raw = input_raw.replace('"', '').replace('\'', '').replace(' ', '')

    # Format of input=filename[variable_name]'
    match = re.match(r'^([\w\-]+)=([\w\-.\/]+)\[([\w\-]+)\]$', input_raw)
    if match:
      input_dict[match.group(1)] = (match.group(2), match.group(3))
    else:
      # Format of input=filename'
      match = re.match(r'^([\w\-]+)=([\w\-.\/]+)$', input_raw)
      if match:
        input_dict[match.group(1)] = (match.group(2), None)
      else:
        raise RuntimeError(
            'Input \"%s\" format is incorrect. Please follow \"--inputs '
            'input_key=file_name[variable_name]\" or input_key=file_name' %
            input_raw)

  return input_dict


def load_inputs_from_input_arg_string(inputs_str):
  """Parses input arg string and load inputs into a dictionary.

  Parses input string in the format of, for example,
  "input1=filename1[variable_name1],input2=filename2" into a
  dictionary looks like
  {'input1:0': ndarray_saved_as_variable_name1_in_filename1 ,
   'input2:0': ndarray_saved_in_filename2}
  , which maps input keys to a numpy ndarray loaded from file. See Args section
  for more details on inputs format.

  Args:
    inputs_str: A string that specified where to load inputs. Each input is
        separated by comma.
        * If the command line arg for inputs is quoted and contains
            whitespace(s), all whitespaces will be ignored.
        * For each input key:
            'input=filename[variable_name]'
        * File specified by 'filename' will be loaded using numpy.load. Inputs
            can be loaded from only .npy, .npz or pickle files.
        * The "[variable_name]" key is optional depending on the input file type
            as descripted in more details below.
        When loading from a npy file, which always contains a numpy ndarray, the
        content will be directly assigned to the specified input tensor. If a
        varaible_name is specified, it will be ignored and a warning will be
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

  Returns:
    A dictionary that maps input tensor keys to a numpy ndarray loaded from
    file.

  Raises:
    RuntimeError: An error when a key is specified, but the input file contains
        multiple numpy ndarrays, none of which matches the given key.
    RuntimeError: An error when no key is specified, but the input file contains
        more than one numpy ndarrays.
  """
  tensor_key_feed_dict = {}

  for input_tensor_key, (
      filename,
      variable_name) in preprocess_input_arg_string(inputs_str).items():
    # When a variable_name key is specified for the input file
    if variable_name:
      data = np.load(filename)

      # if file contains a single ndarray, ignore the input name
      if isinstance(data, np.ndarray):
        warnings.warn(
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
      data = np.load(filename)
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
    # If no tag is specified, display all tag_set, if no signaure_def key is
    # specified, display all SignatureDef keys, else show input output tensor
    # infomation corresponding to the given SignatureDef key
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
  """
  tensor_key_feed_dict = load_inputs_from_input_arg_string(args.inputs)
  run_saved_model_with_feed_dict(args.dir, args.tag_set, args.signature_def,
                                 tensor_key_feed_dict, args.outdir,
                                 args.overwrite, tf_debug=args.tf_debug)


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
  show_msg = (
      'Usage examples:\n'
      'To show all tag-sets in a SavedModel:\n'
      '$saved_model_cli show --dir /tmp/saved_model\n'
      'To show all available SignatureDef keys in a '
      'MetaGraphDef specified by its tag-set:\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve\n'
      'For a MetaGraphDef with multiple tags in the tag-set, all tags must be '
      'passed in, separated by \',\':\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve,gpu\n\n'
      'To show all inputs and outputs TensorInfo for a specific'
      ' SignatureDef specified by the SignatureDef key in a'
      ' MetaGraph.\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve'
      '--signature_def serving_default\n\n'
      'To show all available information in the SavedModel\n:'
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
      help='if set, will output all infomation in given SavedModel')
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

  # run command
  run_msg = ('Usage example:\n'
             'To run input tensors from files through a MetaGraphDef and save'
             ' the output tensors to files:\n'
             '$saved_model_cli show --dir /tmp/saved_model --tag_set serve'
             '--signature_def serving_default '
             '--inputs input1_key=/tmp/124.npz[x],input2_key=/tmp/123.npy'
             '--outdir=/out\n\n'
             'For more information about input file format, please see:\n')
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
  msg = ('inputs in the format of \'input_key=filename[variable_name]\', '
         'separated by \',\'. Inputs can only be loaded from .npy, .npz or '
         'pickle files. Please use input keys instead of input names.')
  parser_run.add_argument('--inputs', type=str, required=True, help=msg)
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
  parser_run.set_defaults(func=run)

  return parser


def main():
  parser = create_parser()
  args = parser.parse_args()
  args.func(args)


if __name__ == '__main__':
  sys.exit(main())
