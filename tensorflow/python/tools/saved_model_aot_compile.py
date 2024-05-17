# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Helper utilities for AOT compilation."""

import collections
import copy
import os
import re
import shlex
from typing import List, Tuple

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib

try:
  from tensorflow.python import _pywrap_tfcompile  # pylint: disable=g-import-not-at-top
except ImportError as e:
  _pywrap_tfcompile_import_error = ImportError(
      'Unable to import _pywrap_tfcompile; you must build TensorFlow '
      'with XLA.  You may need to build tensorflow with flag '
      '--define=with_xla_support=true.  Original error: {}'.format(str(e)))
else:
  _pywrap_tfcompile_import_error = None


_READ_ONLY_VARIABLE_OPS = (
    'ReadVariableOp',
    'IsVariableInitializedOp',
    'ResourceGather',
    'ResourceGatherNd',
    'VariableShape',
)

_PASS_THROUGH_VARIABLE_OPS = ('Identity', 'IdentityN')


def _shlex_quote(s):
  return shlex.quote(s)


def _sysconfig_module():
  """Load tf.sysconfig if available and working (i.e., inside a pip package)."""
  try:
    _ = sysconfig_lib.get_include()
  except (ImportError, ValueError):
    # ValueError may come from saved_model_cli_test trying to enable
    # eager mode twice.
    return None
  return sysconfig_lib


def _parse_tensor_name(name):
  """Convert a tensor name like 'tensor:0' into a tuple ('tensor', 0)."""
  if ':' in name and not name.endswith(':'):
    node_name = name[:name.rfind(':')]
    output_slot = int(name[name.rfind(':') + 1:])
    return node_name, output_slot
  else:
    return name, None


_XLA_MAKEFILE_TEMPLATE = """
INC = -I{tensorflow_includes}
LIB = -L{compiled_dir}
CXXFLAGS = {cxx_flags}
"""


def _xla_makefile_string(output_prefix):
  """Returns a Makefile string with variables for using XLA binary object files.

  Attempts to identify the right include header paths when run from either
  an installed TensorFlow pip package, or from bazel run.

  Args:
    output_prefix: A string containing the output prefix for the XLA AOT
      compiled header + object files.

  Returns:
    A string containing a filled out `_XLA_MAKEFILE_TEMPLATE`.
  """
  sysconfig = _sysconfig_module()
  output_dir, _ = os.path.split(output_prefix)
  if sysconfig:
    tensorflow_includes = _shlex_quote(sysconfig.get_include())
  else:
    # Try hard to find the real source directory if this is a local bazel run.
    if os.path.islink(__file__):
      this_file = __file__
      while os.path.islink(this_file):
        this_file = os.readlink(this_file)
      base = os.path.realpath(
          os.path.join(os.path.dirname(this_file), *([os.path.pardir] * 3)))
    else:
      try:
        base = test.test_src_dir_path('')
      except KeyError:  # Can't find TEST_SRCDIR in environment path.
        base = os.path.realpath(
            os.path.join(os.path.dirname(__file__), *([os.path.pardir] * 3)))
    expected_header = os.path.join(
        base, 'tensorflow', 'compiler', 'tf2xla', 'xla_compiled_cpu_function.h')
    if not os.path.exists(expected_header):
      logging.error(
          'Could not find includes path.  Missing file: {}'
          .format(expected_header))
    tensorflow_includes = base

  return _XLA_MAKEFILE_TEMPLATE.format(
      tensorflow_includes=tensorflow_includes,
      compiled_dir=_shlex_quote(output_dir),
      cxx_flags='-D_GLIBCXX_USE_CXX11_ABI={}'.format(
          versions.CXX11_ABI_FLAG))


def _get_variable_nodes_from_graph_def(graph_def):
  """Get the list of Variable nodes from `graph_def`.

  Args:
    graph_def: An instance of `GraphDef`.  This GraphDef *must*
      have already been optimized by Grappler.  In particular, function
      inlining must have already happened.

  Returns:
    A dict mapping string names of variables to tuples `(node_def, modified)`,
    where `node_def` is the `NodeDef` corresponding to variable, and `modified`
    is a python bool describing whether the variable is modified during runtime.
  """
  variables = [n for n in graph_def.node if n.op == 'VarHandleOp']
  variable_name_map = dict((n.name, n) for n in variables)
  child_map = collections.defaultdict(lambda: [])
  for n in graph_def.node:
    for inp in n.input:
      if not inp.startswith('^'):
        child_map[inp].append(n)
  variables = {}
  for (v_name, v_node) in variable_name_map.items():
    queue = list(child_map[v_name])
    processed = set([])
    while queue:
      n_current = queue.pop()
      if n_current.name in processed:
        continue
      processed.add(n_current.name)
      if n_current.op in _PASS_THROUGH_VARIABLE_OPS:
        children = child_map.get(n_current.name, [])
        queue.extend(children)
      elif n_current.op not in _READ_ONLY_VARIABLE_OPS:
        variables[v_name] = (v_node, True)
        queue = []
    if v_name not in variables:
      variables[v_name] = (v_node, False)

  return variables


def _prune_removed_feed_nodes(signature_def, graph_def):
  """Identify the inputs in the signature no longer in graph_def, prune them.

  Args:
    signature_def: A `SignatureDef` instance.
    graph_def: A `GraphDef` instance.

  Returns:
    A new pruned `SignatureDef`.
  """
  node_names = set([n.name for n in graph_def.node])
  new_signature_def = meta_graph_pb2.SignatureDef()
  new_signature_def.CopyFrom(signature_def)
  for (k, v) in signature_def.inputs.items():
    tensor_name, _ = _parse_tensor_name(v.name)
    if tensor_name not in node_names:
      logging.warn(
          'Signature input key \'{}\', tensor name \'{}\', has been pruned '
          'while freezing the graph.  Removing it from the compiled signatures.'
          .format(k, tensor_name))
      del new_signature_def.inputs[k]
  return new_signature_def


def freeze_model(checkpoint_path: str,
                 meta_graph_def: meta_graph_pb2.MetaGraphDef,
                 output_prefix: str, signature_def_key: str,
                 variables_to_feed: List[str]) -> Tuple[str, str]:
  """Freeze a `MetaGraphDef` in preparation for tfcompile`.

  The graph is always optimized with grappler, and optionally (by default)
  variables are frozen as constants, before compilation happens.

  Args:
    checkpoint_path: Python string.  Path to checkpoints/variables.
    meta_graph_def: Instance of `MetaGraphDef`.
    output_prefix: Python string.  Path prefix for outputs.
    signature_def_key: String, the signature_def to use in the SavedModel.
    variables_to_feed: A list of strings, the variables that will be fed by the
      user; these won't be frozen.  If `None`, then we will extract all the
      variables in the graph and mark them as to-feed.  The default behavior is
      an empty tuple: all variables must be frozen.
  Returns:
    a pair containing the path to the frozen model and the path to the config.
  Raises:
    RuntimeError: If tensorflow was not built with XLA.
    ImportError: If tensorflow was built with XLA but there was another
      issue importing the tfcompile python wrapper.
    ValueError: If `meta_graph_def.signature_def[signature_def_key]` is
      missing or has empty outputs.
  """
  signature_def_map = meta_graph_def.signature_def
  if signature_def_key not in signature_def_map:
    raise ValueError(
        f"Unable to find signature_def_key '{signature_def_key}' in signature "
        'def map of `meta_graph_def`. Available keys: '
        f'{list(signature_def_map.keys())}')
  signature_def = signature_def_map[signature_def_key]
  if not signature_def.outputs:
    raise ValueError(
        f'Signature key {signature_def_key} must have outputs, but saw none:\n'
        f'{str(signature_def)}')

  file_io.recursive_create_dir(output_prefix)
  if logging.get_verbosity() >= logging.INFO:
    original_graph_def_location = os.path.join(output_prefix,
                                               'original_graph.pb')
    with file_io.FileIO(original_graph_def_location, 'wb') as graph_writer:
      graph_writer.write(meta_graph_def.graph_def.SerializeToString())

  # This updates graph_def in place.
  _replace_input_placeholders_with_default_values(
      meta_graph_def.graph_def, signature_def)

  graph_def = _optimize_graph(meta_graph_def, signature_def)

  all_variables = _get_variable_nodes_from_graph_def(graph_def)
  if variables_to_feed is None:
    variable_nodes_to_feed = list(all_variables.values())
  else:
    not_in_graph = set(variables_to_feed).difference(list(all_variables))
    if not_in_graph:
      raise ValueError('Asked to feed variables that were not found in graph: '
                       f'{not_in_graph}. Variables contained in the graph: '
                       f'{list(all_variables)}')
    variable_nodes_to_feed = [
        all_variables[name] for name in variables_to_feed
    ]

  if logging.get_verbosity() >= logging.INFO:
    prefrozen_graph_def_location = os.path.join(output_prefix,
                                                'prefrozen_graph.pb')
    with file_io.FileIO(prefrozen_graph_def_location, 'wb') as graph_writer:
      graph_writer.write(graph_def.SerializeToString())

  # Load the Variables so that we can freeze the graph.
  with session.Session(graph=ops_lib.Graph()) as sess:
    restorer = saver_lib.import_meta_graph(meta_graph_def, clear_devices=True)
    if restorer is not None:
      restorer.restore(sess, checkpoint_path)
    graph_def.CopyFrom(
        convert_to_constants.convert_variables_to_constants(
            sess,
            graph_def,
            output_node_names=[
                _parse_tensor_name(n.name)[0]
                for n in signature_def.outputs.values()
            ],
            variable_names_blacklist=[
                n.name for n, _ in variable_nodes_to_feed
            ],
        ))

  signature_def = _prune_removed_feed_nodes(signature_def, graph_def)

  frozen_graph_def_location = os.path.join(output_prefix, 'frozen_graph.pb')
  config_pbtxt_location = os.path.join(output_prefix, 'config.pbtxt')
  logging.info('Writing graph def to: {}'.format(frozen_graph_def_location))
  with file_io.FileIO(frozen_graph_def_location, 'wb') as graph_writer:
    graph_writer.write(graph_def.SerializeToString())
  config = _signature_to_tf2xla_config(
      signature_def, variable_nodes_to_feed=variable_nodes_to_feed)
  logging.info('Writing config_pbtxt to: {}'.format(config_pbtxt_location))
  with file_io.FileIO(config_pbtxt_location, mode='w') as config_writer:
    config_writer.write(str(config))
  return frozen_graph_def_location, config_pbtxt_location


def aot_compile_cpu_meta_graph_def(checkpoint_path,
                                   meta_graph_def,
                                   output_prefix,
                                   signature_def_key,
                                   cpp_class,
                                   target_triple,
                                   target_cpu,
                                   variables_to_feed=(),
                                   multithreading=False):
  """Compile a `MetaGraphDef` to header+object files in `output_prefix`.

  Use XLA AOT (`tfcompile`) to convert the given meta graph and
  signature into a header + object files.  Also create an include makefile
  that helps identify the appropriate necessary include and library paths
  to incorporate these files into your C++ program.

  Freezing a graph entails restoring the checkpoint and replacing any inputs and
  variables with constants. If values are feed, those are used, else inputs are
  replaced with default all-zero constants. Finally, the graph is pruned and
  then optimized with grappler.

  If the `freeze_graph` is `True`, all variables are embedded as constants
  into the graph and binary objects.  If it is `False`, then the variable
  values become inputs and outputs of the compiled class and the C++
  caller must set these values manually.

  Args:
    checkpoint_path: Python string.  Path to checkpoints/variables.
    meta_graph_def: Instance of `MetaGraphDef`.
    output_prefix: Python string.  Path prefix for outputs.
    signature_def_key: String, the signature_def to use in the SavedModel.
    cpp_class: String, Name of output C++ class.
    target_triple: String, LLVM target triple.
    target_cpu: String, LLVM target cpu name.
    variables_to_feed: A list of strings, the variables that will be fed by the
      user; these won't be frozen.  If `None`, then we will extract all the
      variables in the graph and mark them as to-feed.  The default behavior is
      an empty tuple: all variables must be frozen.
    multithreading: Whether to enable multithreading in the compiled
      computation.  Note that if using this option, the resulting object files
      may have external dependencies on multithreading libraries like nsync.

  Raises:
    RuntimeError: If tensorflow was not built with XLA.
    ImportError: If tensorflow was built with XLA but there was another
      issue importing the tfcompile python wrapper.
    ValueError: If `meta_graph_def.signature_def[signature_def_key]` is
      missing or has empty outputs.
  """
  if _pywrap_tfcompile_import_error:
    raise _pywrap_tfcompile_import_error  # pylint: disable=raising-bad-type

  else:
    # TODO(ebrevdo): Pipe DebugOptions through tfcompile::Main and pywrap
    # so that we can set these directly instead of relying on env vars.
    xla_flags = os.environ.get('XLA_FLAGS')
    if not xla_flags:
      xla_flags = '--xla_cpu_multi_thread_eigen={}'.format(
          'true' if multithreading else 'false')
    else:
      xla_flags += ' --xla_cpu_multi_thread_eigen={}'.format(
          'true' if multithreading else 'false')
    os.environ['XLA_FLAGS'] = xla_flags

  temp_dir = test.get_temp_dir()
  file_io.recursive_create_dir(temp_dir)
  frozen_graph_def_location, config_pbtxt_location = freeze_model(
      checkpoint_path=checkpoint_path,
      meta_graph_def=meta_graph_def,
      output_prefix=temp_dir,
      signature_def_key=signature_def_key,
      variables_to_feed=variables_to_feed)
  output_dir = os.path.dirname(output_prefix)
  file_io.recursive_create_dir(output_dir)

  entry_point = re.sub(
      '[^0-9a-zA-Z]+', '_',
      '__xla_' + output_prefix + '__' + cpp_class)

  logging.info('Generating XLA AOT artifacts in: {}'.format(output_dir))

  makefile_inc_location = '{}_makefile.inc'.format(output_prefix)
  with file_io.FileIO(makefile_inc_location, mode='w') as makefile_writer:
    makefile_writer.write(_xla_makefile_string(output_prefix))

  output_prefix = _shlex_quote(output_prefix)

  _pywrap_tfcompile.Compile(
      graph=frozen_graph_def_location,
      config=config_pbtxt_location,
      cpp_class=cpp_class,
      target_triple=target_triple,
      target_cpu=target_cpu,
      entry_point=entry_point,
      out_function_object='{}.o'.format(output_prefix),
      out_header='{}.h'.format(output_prefix),
      out_metadata_object='{}_metadata.o'.format(output_prefix),
      gen_name_to_index=True,
      # ProgramShape isn't uniquefied by entry_point.
      gen_program_shape=False)


def _optimize_graph(meta_graph_def, signature_def):
  """Optimize `meta_graph_def` using grappler.  Returns a `GraphDef`."""
  # We need to add a collection called 'train_op' so that grappler
  # knows what the outputs are.
  new_meta_graph_def = copy.deepcopy(meta_graph_def)
  fetch_collection = meta_graph_pb2.CollectionDef()
  for tensor_info in (
      list(signature_def.inputs.values()) +
      list(signature_def.outputs.values())):
    fetch_collection.node_list.value.append(tensor_info.name)

  new_meta_graph_def.collection_def['train_op'].CopyFrom(fetch_collection)
  # We freeze the graph, so consider all variables to be readonly.
  new_meta_graph_def.ClearField('saver_def')
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  rewrite_options.min_graph_nodes = -1  # do not skip small graphs
  return tf_optimizer.OptimizeGraph(config, new_meta_graph_def)


def _replace_input_placeholders_with_default_values(graph_def, signature_def):
  """Replace graphdef's `tf.placeholder` input ops with all-zero constants."""
  name_to_node_map = dict((n.name, n) for n in graph_def.node)
  processed_nodes = set([])
  for name, input_ in signature_def.inputs.items():
    tensor_name, _ = _parse_tensor_name(input_.name)
    if tensor_name in processed_nodes:
      continue
    processed_nodes.add(tensor_name)
    if tensor_name not in name_to_node_map:
      raise RuntimeError(
          f"Unable to find input signature tensor '{tensor_name}' in optimized "
          f'GraphDef. Graph nodes are: {list(name_to_node_map.keys())}')
    node = name_to_node_map[tensor_name]
    if node.op not in ('Placeholder', 'PlaceholderV2'):
      logging.info(
          'Tried to convert SavedModel input node \'{}\' from a placeholder, '
          'but it doesn\'t look like a placeholder: {}'.format(tensor_name,
                                                               node))
      continue
    shape = tensor_shape.TensorShape(input_.tensor_shape)
    if not shape.is_fully_defined():
      raise ValueError(
          f"Expected fully defined input shape for signature_def '{name}', "
          f"tensor name: '{tensor_name}'; but shape is: {shape}.")
    temp_graph = ops_lib.Graph()
    with temp_graph.as_default():
      const = array_ops.zeros(
          shape, dtype=input_.dtype, name=tensor_name)
    node.CopyFrom(const.op.node_def)
    # Sometimes zeros() also creates additional nodes
    for op in temp_graph.get_operations():
      if op.name == const.op.name:  # We just inserted this one.
        continue
      graph_def.node.append(op.node_def)
      name_to_node_map[op.node_def.name] = op.node_def


def _signature_to_tf2xla_config(signature_def, variable_nodes_to_feed):
  """Convert `signature_def` to tf2xla config.  Returns a `tf2xla.Config` proto.

  Args:
    signature_def: Instance of `SignatureDef`.
    variable_nodes_to_feed: List of tuples of form `(node_def, modified)`
      corresponding to VarHandleOp, and a boolean `modified` that describes
      whether the variable was modified during execution.

  Returns:
    An instance of `tf2xla.Config` proto.

  Raises:
    RuntimeError: If TensorFlow was not compiled with XLA.
  """
  from tensorflow.compiler.tf2xla import tf2xla_pb2  # pylint: disable=g-import-not-at-top

  config = tf2xla_pb2.Config()
  tensor_id = tf2xla_pb2.TensorId

  for name, input_ in signature_def.inputs.items():
    name = name.replace('/', '_')
    name = 'feed_{}'.format(name)
    (node_name, output_index) = _parse_tensor_name(input_.name)
    output_index = int(output_index)
    config.feed.append(
        tf2xla_pb2.Feed(
            id=tensor_id(node_name=node_name, output_index=output_index),
            name=name,
            type=input_.dtype,
            shape=input_.tensor_shape))
  for name, output_ in signature_def.outputs.items():
    name = name.replace('/', '_')
    name = 'fetch_{}'.format(name)
    (node_name, output_index) = _parse_tensor_name(output_.name)
    output_index = int(output_index)
    config.fetch.append(
        tf2xla_pb2.Fetch(
            id=tensor_id(node_name=node_name, output_index=output_index),
            name=name,
            type=output_.dtype,
            shape=output_.tensor_shape))
  for (node, modified) in variable_nodes_to_feed:
    name = node.name.replace('/', '_')
    name = 'param_{}'.format(name)
    config.variable.append(
        tf2xla_pb2.Variable(
            node_name=node.name,
            name=name,
            type=node.attr['dtype'].type,
            shape=node.attr['shape'].shape,
            readonly=not modified))

  return config
