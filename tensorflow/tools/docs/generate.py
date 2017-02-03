# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Generate docs for the TensorFlow Python API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inspect
import os

import six
import tensorflow as tf

from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse
from tensorflow.tools.docs import doc_generator_visitor
from tensorflow.tools.docs import parser


def write_docs(output_dir, base_dir, duplicate_of, duplicates, index, tree):
  """Write previously extracted docs to disk.

  Write a docs page for each symbol in `index` to a tree of docs at
  `output_dir`.

  Symbols with multiple aliases will have only one page written about them,
  which is referenced for all aliases. `duplicate_of` and `duplicates` are used
  to determine which docs pages to write.

  Args:
    output_dir: Directory to write documentation markdown files to. Will be
      created if it doesn't exist.
    base_dir: Base directory of the code being documented. This prefix is
      stripped from all file paths that are part of the documentation.
    duplicate_of: A `dict` mapping fully qualified names to "master" names. This
      is used to resolve "@{symbol}" references to the "master" name.
    duplicates: A `dict` mapping fully qualified names to a set of all
      aliases of this name. This is used to automatically generate a list of all
      aliases for each name.
    index: A `dict` mapping fully qualified names to the corresponding Python
      objects. Used to produce docs for child objects, and to check the validity
      of "@{symbol}" references.
    tree: A `dict` mapping a fully qualified name to the names of all its
      members. Used to populate the members section of a class or module page.
  """
  # Make output_dir.
  try:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  except OSError as e:
    print('Creating output dir "%s" failed: %s' % (output_dir, e))
    raise

  # Parse and write Markdown pages, resolving cross-links (@{symbol}).
  for full_name, py_object in six.iteritems(index):

    if full_name in duplicate_of:
      print('Not writing docs for %s, duplicate of %s.' % (
          full_name, duplicate_of[full_name]))
      continue

    # Methods and some routines are documented only as part of their class.
    if not (inspect.ismodule(py_object) or
            inspect.isclass(py_object) or
            inspect.isfunction(py_object)):
      print('Not writing docs for %s, not a class, module, or function.' % (
          full_name))
      continue

    print('Writing docs for %s (%r).' % (full_name, py_object))

    # Generate docs for `py_object`, resolving references.
    markdown = parser.generate_markdown(full_name, py_object,
                                        duplicate_of=duplicate_of,
                                        duplicates=duplicates,
                                        index=index,
                                        tree=tree,
                                        base_dir=base_dir)

    # TODO(deannarubin): use _tree to generate sidebar information.

    path = os.path.join(output_dir, parser.documentation_path(full_name))
    directory = os.path.dirname(path)
    try:
      if not os.path.exists(directory):
        os.makedirs(directory)
      with open(path, 'w') as f:
        f.write(markdown)
    except OSError as e:
      print('Cannot write documentation for %s to %s: %s' % (full_name,
                                                             directory, e))
      raise
    # TODO(deannarubin): write sidebar file?

  # Write a global index containing all full names with links.
  with open(os.path.join(output_dir, 'full_index.md'), 'w') as f:
    f.write(parser.generate_global_index('TensorFlow', 'tensorflow',
                                         index, duplicate_of))


def extract():
  """Extract docs from tf namespace and write them to disk."""
  visitor = doc_generator_visitor.DocGeneratorVisitor()
  api_visitor = public_api.PublicAPIVisitor(visitor)

  # Access something in contrib so tf.contrib is properly loaded (it's hidden
  # behind lazy loading)
  _ = tf.contrib.__name__

  # Exclude some libaries in contrib from the documentation altogether.
  # TODO(wicke): Shrink this list.
  api_visitor.do_not_descend_map.update({
      'contrib': [
          'compiler',
          'factorization',
          'grid_rnn',
          'labeled_tensor',
          'ndlstm',
          'quantization',
          'session_bundle',
          'slim',
          'solvers',
          'specs',
          'tensor_forest',
          'tensorboard',
          'testing',
          'tfprof',
          'training',
      ],
      'contrib.bayesflow': [
          'entropy', 'monte_carlo',
          'special_math', 'stochastic_gradient_estimators',
          'stochastic_graph', 'stochastic_tensor',
          'stochastic_variables', 'variational_inference'
      ],
      'contrib.distributions': ['bijector'],
      'contrib.graph_editor': [
          'edit',
          'match',
          'reroute',
          'subgraph',
          'transform',
          'select',
          'util'
      ],
      'contrib.layers': [
          'feature_column',
          'summaries'
      ],
      'contrib.learn': [
          'datasets',
          'head',
          'graph_actions',
          'io',
          'models',
          'monitors',
          'ops',
          'preprocessing',
          'utils',
      ],
      'contrib.util': ['loader'],
  })

  traverse.traverse(tf, api_visitor)

  return visitor


def write(output_dir, base_dir, visitor):
  """Write documentation for an index in a `DocGeneratorVisitor` to disk.

  This function will create `output_dir` if it doesn't exist, and write
  the documentation contained in `visitor`.

  Args:
    output_dir: The directory to write documentation to. Must not exist.
    base_dir: The base dir of the library `visitor` has traversed. This is used
      to compute relative paths for file references.
    visitor: A `DocGeneratorVisitor` that has traversed a library located at
      `base_dir`.
  """
  duplicate_of, duplicates = visitor.find_duplicates()
  write_docs(output_dir, os.path.abspath(base_dir),
             duplicate_of, duplicates, visitor.index, visitor.tree)


if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument(
      '--output_dir',
      type=str,
      default=None,
      required=True,
      help='Directory to write docs to. Must not exist.'
  )

  # This doc generator works on the TensorFlow codebase. Since this script lives
  # at tensorflow/tools/docs, we can compute the base directory (three levels
  # up), which is valid unless we're trying to apply this to a different code
  # base, or are moving the script around.
  script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
  default_base_dir = os.path.join(script_dir, '..', '..', '..')

  argument_parser.add_argument(
      '--base_dir',
      type=str,
      default=default_base_dir,
      help=('Base directory to to strip from file names referenced in docs. '
            'Defaults to three directories up from the location of this file.')
  )

  flags, _ = argument_parser.parse_known_args()

  if os.path.exists(flags.output_dir):
    raise RuntimeError('output_dir %s exists.\n'
                       'Cowardly refusing to wipe it, please do that yourself.'
                       % flags.output_dir)

  write(flags.output_dir, flags.base_dir, extract())
