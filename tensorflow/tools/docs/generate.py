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
import sys

import six
import tensorflow as tf

from tensorflow.python import debug as tf_debug

from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse
from tensorflow.tools.docs import doc_generator_visitor
from tensorflow.tools.docs import parser
from tensorflow.tools.docs import py_guide_parser


def write_docs(output_dir, base_dir, duplicate_of, duplicates, index, tree,
               reverse_index, doc_index, guide_index):
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
    reverse_index: A `dict` mapping object ids to fully qualified names.
    doc_index: A `dict` mapping a doc key to a DocInfo.
    guide_index: A `dict` mapping symbol name strings to GuideRef.
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
      continue

    # Methods and some routines are documented only as part of their class.
    if not (inspect.ismodule(py_object) or
            inspect.isclass(py_object) or
            inspect.isfunction(py_object)):
      continue

    print('Writing docs for %s (%r).' % (full_name, py_object))

    # Generate docs for `py_object`, resolving references.
    markdown = parser.generate_markdown(full_name, py_object,
                                        duplicate_of=duplicate_of,
                                        duplicates=duplicates,
                                        index=index,
                                        tree=tree,
                                        reverse_index=reverse_index,
                                        doc_index=doc_index,
                                        guide_index=guide_index,
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
  with open(os.path.join(output_dir, 'index.md'), 'w') as f:
    f.write(parser.generate_global_index('TensorFlow', index, duplicate_of))


def extract():
  """Extract docs from tf namespace and write them to disk."""
  visitor = doc_generator_visitor.DocGeneratorVisitor('tf')
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
      ],
      'contrib.bayesflow': [
          'entropy', 'monte_carlo',
          'special_math', 'stochastic_gradient_estimators',
          'stochastic_graph', 'stochastic_tensor',
          'stochastic_variables', 'variational_inference'
      ],
      'contrib.distributions': ['bijector'],
      'contrib.ffmpeg': ['ffmpeg_ops'],
      'contrib.graph_editor': [
          'edit',
          'match',
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

  # tf_debug is not imported with tf, it's a separate module altogether
  visitor.set_root_name('tfdbg')
  traverse.traverse(tf_debug, api_visitor)

  return visitor


class GetMarkdownTitle(py_guide_parser.PyGuideParser):
  """Extract the title from a .md file."""

  def __init__(self):
    self.title = None
    py_guide_parser.PyGuideParser.__init__(self)

  def process_title(self, _, title):
    if self.title is None:  # only use the first title
      self.title = title


class DocInfo(object):
  """A simple struct for holding a doc's url and title."""

  def __init__(self, url, title):
    self.url = url
    self.title = title


def build_doc_index(src_dir):
  """Build an index from a keyword designating a doc to DocInfo objects."""
  doc_index = {}
  for dirpath, _, filenames in os.walk(src_dir):
    suffix = os.path.relpath(path=dirpath, start=src_dir)
    for base_name in filenames:
      if not base_name.endswith('.md'): continue
      title_parser = GetMarkdownTitle()
      title_parser.process(os.path.join(dirpath, base_name))
      key_parts = os.path.join(suffix, base_name[:-3]).split('/')
      if key_parts[-1] == 'index':
        key_parts = key_parts[:-1]
      doc_info = DocInfo(os.path.join(suffix, base_name), title_parser.title)
      doc_index[key_parts[-1]] = doc_info
      if len(key_parts) > 1:
        doc_index['/'.join(key_parts[-2:])] = doc_info

  return doc_index


class GuideRef(object):

  def __init__(self, base_name, title, section_title, section_tag):
    self.url = 'api_guides/python/' + (
        ('%s#%s' % (base_name, section_tag)) if section_tag else base_name)
    self.link_text = (('%s > %s' % (title, section_title))
                      if section_title else title)

  def make_md_link(self, url_prefix):
    return '[%s](%s%s)' % (self.link_text, url_prefix, self.url)


class GenerateGuideIndex(py_guide_parser.PyGuideParser):
  """Turn guide files into an index from symbol name to a list of GuideRefs."""

  def __init__(self):
    self.index = {}
    py_guide_parser.PyGuideParser.__init__(self)

  def process(self, full_path, base_name):
    """Index a file, reading from `full_path`, with `base_name` as the link."""
    self.full_path = full_path
    self.base_name = base_name
    self.title = None
    self.section_title = None
    self.section_tag = None
    py_guide_parser.PyGuideParser.process(self, full_path)

  def process_title(self, _, title):
    if self.title is None:  # only use the first title
      self.title = title

  def process_section(self, _, section_title, tag):
    self.section_title = section_title
    self.section_tag = tag

  def process_line(self, _, line):
    """Index @{symbol} references as in the current file & section."""
    for match in parser.SYMBOL_REFERENCE_RE.finditer(line):
      val = self.index.get(match.group(1), [])
      val.append(GuideRef(
          self.base_name, self.title, self.section_title, self.section_tag))
      self.index[match.group(1)] = val


def build_guide_index(guide_src_dir):
  """Return dict: symbol name -> GuideRef from the files in `guide_src_dir`."""
  index_generator = GenerateGuideIndex()
  for full_path, base_name in py_guide_parser.md_files_in_dir(guide_src_dir):
    index_generator.process(full_path, base_name)
  return index_generator.index


def write(output_dir, base_dir, doc_index, guide_index, visitor):
  """Write documentation for an index in a `DocGeneratorVisitor` to disk.

  This function will create `output_dir` if it doesn't exist, and write
  the documentation contained in `visitor`.

  Args:
    output_dir: The directory to write documentation to. Must not exist.
    base_dir: The base dir of the library `visitor` has traversed. This is used
      to compute relative paths for file references.
    doc_index: A `dict` mapping a doc key to a DocInfo.
    guide_index: A `dict` mapping symbol name strings to GuideRef.
    visitor: A `DocGeneratorVisitor` that has traversed a library located at
      `base_dir`.
  """
  write_docs(output_dir, os.path.abspath(base_dir),
             visitor.duplicate_of, visitor.duplicates,
             visitor.index, visitor.tree, visitor.reverse_index,
             doc_index, guide_index)


class UpdateTags(py_guide_parser.PyGuideParser):
  """Rewrites a Python guide so that each section has an explicit tag."""

  def process_section(self, line_number, section_title, tag):
    self.replace_line(line_number, '<h2 id="%s">%s</h2>' % (tag, section_title))


def other_docs(src_dir, output_dir, visitor, doc_index):
  """Convert all the files in `src_dir` and write results to `output_dir`."""
  header = '<!-- DO NOT EDIT! Automatically generated file. -->\n'

  # Iterate through all the source files and process them.
  tag_updater = UpdateTags()
  for dirpath, _, filenames in os.walk(src_dir):
    # How to get from `dirpath` to api_docs/python/
    relative_path_to_root = os.path.relpath(
        path=os.path.join(src_dir, 'api_docs/python'), start=dirpath)

    # Make the directory under output_dir.
    new_dir = os.path.join(output_dir,
                           os.path.relpath(path=dirpath, start=src_dir))
    try:
      if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    except OSError as e:
      print('Creating output dir "%s" failed: %s' % (new_dir, e))
      raise

    for base_name in filenames:
      full_in_path = os.path.join(dirpath, base_name)
      suffix = os.path.relpath(path=full_in_path, start=src_dir)
      full_out_path = os.path.join(output_dir, suffix)
      if not base_name.endswith('.md'):
        print('Copying non-md file %s...' % suffix)
        open(full_out_path, 'w').write(open(full_in_path).read())
        continue
      if dirpath.endswith('/api_guides/python'):
        print('Processing Python guide %s...' % base_name)
        md_string = tag_updater.process(full_in_path)
      else:
        print('Processing doc %s...' % suffix)
        md_string = open(full_in_path).read()

      output = parser.replace_references(
          md_string, relative_path_to_root, visitor.duplicate_of,
          doc_index=doc_index, index=visitor.index)
      with open(full_out_path, 'w') as f:
        f.write(header + output)

  print('Done.')


def _main(src_dir, output_dir, base_dir):
  doc_index = build_doc_index(src_dir)
  visitor = extract()
  write(os.path.join(output_dir, 'api_docs/python'), base_dir,
        doc_index,
        build_guide_index(os.path.join(src_dir, 'api_guides/python')),
        visitor)
  other_docs(src_dir, output_dir, visitor, doc_index)


if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument(
      '--output_dir',
      type=str,
      default=None,
      required=True,
      help='Directory to write docs to.'
  )

  argument_parser.add_argument(
      '--src_dir',
      type=str,
      default=None,
      required=True,
      help='Directory with the source docs.'
  )

  # This doc generator works on the TensorFlow codebase. Since this script lives
  # at tensorflow/tools/docs, and all code is defined somewhere inside
  # tensorflow/, we can compute the base directory (two levels up), which is
  # valid unless we're trying to apply this to a different code base, or are
  # moving the script around.
  script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
  default_base_dir = os.path.join(script_dir, '..', '..')

  argument_parser.add_argument(
      '--base_dir',
      type=str,
      default=default_base_dir,
      help=('Base directory to to strip from file names referenced in docs. '
            'Defaults to two directories up from the location of this file.')
  )

  flags, _ = argument_parser.parse_known_args()
  _main(flags.src_dir, flags.output_dir, flags.base_dir)
  if parser.all_errors:
    print('Errors during processing:' + '\n  '.join(parser.all_errors))
    sys.exit(1)
