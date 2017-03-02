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
"""Turn Python docstrings into Markdown for TensorFlow documentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import functools
import inspect
import os
import re

import codegen
import six


# A regular expression capturing a python indentifier.
IDENTIFIER_RE = '[a-zA-Z_][a-zA-Z0-9_]*'

# Log of all reported errors
all_errors = []


def log_error(s):
  all_errors.append(s)
  print('ERROR:', s)


def documentation_path(full_name):
  """Returns the file path for the documentation for the given API symbol.

  Given the fully qualified name of a library symbol, compute the path to which
  to write the documentation for that symbol (relative to a base directory).
  Documentation files are organized into directories that mirror the python
  module/class structure.

  Args:
    full_name: Fully qualified name of a library symbol.

  Returns:
    The file path to which to write the documentation for `full_name`.
  """
  dirs = full_name.split('.')
  return os.path.join(*dirs) + '.md'


def _get_raw_docstring(py_object):
  """Get the docs for a given python object.

  Args:
    py_object: A python object to retrieve the docs for (class, function/method,
      or module).

  Returns:
    The docstring, or the empty string if no docstring was found.
  """
  # For object instances, inspect.getdoc does give us the docstring of their
  # type, which is not what we want. Only return the docstring if it is useful.
  if (inspect.isclass(py_object) or inspect.ismethod(py_object) or
      inspect.isfunction(py_object) or inspect.ismodule(py_object) or
      isinstance(py_object, property)):
    return inspect.getdoc(py_object) or ''
  else:
    return ''


# A regular expression for capturing a @{symbol} reference.
SYMBOL_REFERENCE_RE = re.compile(r'@\{([^}]+)\}')


class ReferenceResolver(object):
  """Class for replacing @{...} references with Markdown links.

  Args:
    duplicate_of: A map from duplicate names to preferred names of API
      symbols.
    doc_index: A `dict` mapping symbol name strings to objects with `url`
      and `title` fields. Used to resolve @{$doc} references in docstrings.
    index: A map from all full names to python objects.
  """

  def __init__(self, duplicate_of, doc_index, index):
    self._duplicate_of = duplicate_of
    self._doc_index = doc_index
    self._index = index

  def replace_references(self, string, relative_path_to_root):
    """Replace "@{symbol}" references with links to symbol's documentation page.

    This functions finds all occurrences of "@{symbol}" in `string`
    and replaces them with markdown links to the documentation page
    for "symbol".

    `relative_path_to_root` is the relative path from the document
    that contains the "@{symbol}" reference to the root of the API
    documentation that is linked to. If the containing page is part of
    the same API docset, `relative_path_to_root` can be set to
    `os.path.dirname(documentation_path(name))`, where `name` is the
    python name of the object whose documentation page the reference
    lives on.

    Args:
      string: A string in which "@{symbol}" references should be replaced.
      relative_path_to_root: The relative path from the containing document to
        the root of the API documentation that is being linked to.

    Returns:
      `string`, with "@{symbol}" references replaced by Markdown links.
    """
    return re.sub(SYMBOL_REFERENCE_RE,
                  lambda match: self._one_ref(match.group(1),  # pylint: disable=g-long-lambda
                                              relative_path_to_root),
                  string)

  def python_link(self, link_text, ref_full_name, relative_path_to_root,
                  code_ref=True):
    """Resolve a "@{python symbol}" reference to a Markdown link.

    This will pick the canonical location for duplicate symbols.  The
    input to this function should already be stripped of the '@' and
    '{}'.  This function returns a Markdown link. If `code_ref` is
    true, it is assumed that this is a code reference, so the link
    text will be rendered as code (using backticks).
    `link_text` should refer to a library symbol, starting with 'tf.'.

    Args:
      link_text: The text of the Markdown link.
      ref_full_name: The fully qualified name of the symbol to link to.
      relative_path_to_root: The relative path from the location of the current
        document to the root of the API documentation.
      code_ref: If true (the default), put `link_text` in `...`.

    Returns:
      A markdown link to the documentation page of `ref_full_name`.
    """
    link = self._reference_to_link(ref_full_name, relative_path_to_root)
    if code_ref:
      return '[`%s`](%s)' % (link_text, link)
    else:
      return '[%s](%s)' % (link_text, link)

  def py_master_name(self, full_name):
    """Return the master name for a Python symbol name."""
    return self._duplicate_of.get(full_name, full_name)

  def py_name_to_object(self, full_name):
    """Return the Python object for a Python symbol name."""
    return self._index[full_name]

  def _reference_to_link(self, ref_full_name, relative_path_to_root):
    """Resolve a "@{python symbol}" reference to a relative path.

    The input to this function should already be stripped of the '@'
    and '{}', and its output is only the link, not the full Markdown.

    If `ref_full_name` is the name of a class member, method, or property, the
    link will point to the page of the containing class, and it will include the
    method name as an anchor. For example, `tf.module.MyClass.my_method` will be
    translated into a link to
    `os.join.path(relative_path_to_root, 'tf/module/MyClass.md#my_method')`.

    Args:
      ref_full_name: The fully qualified name of the symbol to link to.
      relative_path_to_root: The relative path from the location of the current
        document to the root of the API documentation.

    Returns:
      A relative path that links from the documentation page of `from_full_name`
      to the documentation page of `ref_full_name`.

    Raises:
      RuntimeError: If `ref_full_name` is not in `self._index`.
    """
    master_name = self._duplicate_of.get(ref_full_name, ref_full_name)

    # Check whether this link exists
    if master_name not in self._index:
      # TODO(josh11b): Make error reporting more uniform.
      print('ERROR: Cannot make link to %s (original: %s): Not in index.' %
            (master_name, ref_full_name))
      return 'BROKEN_LINK'

    # If this is a member of a class, link to the class page with an anchor.
    ref_path = None
    py_object = self._index[master_name]
    if not (inspect.isclass(py_object) or inspect.ismodule(py_object)):
      idents = master_name.split('.')
      if len(idents) > 1:
        class_name = '.'.join(idents[:-1])
        assert class_name in self._index
        if inspect.isclass(self._index[class_name]):
          ref_path = documentation_path(class_name) + '#%s' % idents[-1]

    if not ref_path:
      ref_path = documentation_path(master_name)

    return os.path.join(relative_path_to_root, ref_path)

  def _one_ref(self, string, relative_path_to_root):
    """Return a link for a single "@{symbol}" reference."""
    # Look for link text after $.
    dollar = string.rfind('$')
    if dollar > 0:  # Ignore $ in first character
      link_text = string[dollar + 1:]
      string = string[:dollar]
      manual_link_text = True
    else:
      link_text = string
      manual_link_text = False

    # Handle different types of references.
    if string.startswith('$'):  # Doc reference
      string = string[1:]  # remove leading $

      # If string has a #, split that part into `hash_tag`
      hash_pos = string.find('#')
      if hash_pos > -1:
        hash_tag = string[hash_pos:]
        string = string[:hash_pos]
      else:
        hash_tag = ''

      if string in self._doc_index:
        if not manual_link_text: link_text = self._doc_index[string].title
        url = os.path.normpath(os.path.join(
            relative_path_to_root, '../..', self._doc_index[string].url))
        return '[%s](%s%s)' % (link_text, url, hash_tag)
      log_error('Handle doc reference "@{$%s}"' % string)
      return 'TODO:%s' % string

    # TODO(josh11b): The list of Python prefixes should be passed in.
    elif string.startswith('tf.') or string.startswith('tfdbg.'):
      # Python symbol
      return self.python_link(link_text, string, relative_path_to_root,
                              code_ref=not manual_link_text)

    elif string.startswith('tensorflow::'):
      # C++ symbol
      # TODO(josh11b): Fix this hard-coding of paths.
      if string == 'tensorflow::ClientSession':
        ret = 'class/tensorflow/client-session.md'
      elif string == 'tensorflow::Scope':
        ret = 'class/tensorflow/scope.md'
      elif string == 'tensorflow::Status':
        ret = 'class/tensorflow/status.md'
      elif string == 'tensorflow::Tensor':
        ret = 'class/tensorflow/tensor.md'
      elif string == 'tensorflow::ops::Const':
        ret = 'namespace/tensorflow/ops.md#const'
      else:
        log_error('Handle C++ reference "@{%s}"' % string)
        return 'TODO_C++:%s' % string
      # relative_path_to_root gets you to api_docs/python, we go from there
      # to api_docs/cc, and then add ret.
      cc_relative_path = os.path.normpath(os.path.join(
          relative_path_to_root, '../cc', ret))
      return '[`%s`](%s)' % (link_text, cc_relative_path)

    # Error!
    log_error('Did not understand "@{%s}"' % string)
    return 'ERROR:%s' % string


def _md_brief_docstring(py_object, relative_path_to_root, reference_resolver):
  """Get the brief docstring from an object and make it into nice Markdown.

  For links within the same set of docs, the `relative_path_to_root` for a
  docstring on the page for `full_name` can be set to

  ```python
  relative_path_to_root = os.path.relpath(
    path='.', start=os.path.dirname(documentation_path(full_name)) or '.')
  ```

  Args:
    py_object: A python object to retrieve the docs for (class, function/method,
      or module).
    relative_path_to_root: The relative path from the location of the current
      document to the root of the Python API documentation. This is used to
      compute links for "@{symbol}" references.
    reference_resolver: An instance of ReferenceResolver.

  Returns:
    The docstring, or the empty string if no docstring was found.
  """
  # TODO(wicke): If this is a partial, use the .func docstring and add a note.
  docstring = _get_raw_docstring(py_object).split('\n')[0]
  docstring = reference_resolver.replace_references(
      docstring, relative_path_to_root)
  return docstring


# TODO(aselle): Collect these into a big list for all modules and functions
# and make a rosetta stone page.
def _handle_compatibility(doc):
  """Parse and remove compatibility blocks from the main docstring.

  Args:
    doc: The docstring that contains compatibility notes"

  Returns:
    a tuple of the modified doc string and a hash that maps from compatibility
    note type to the text of the note.
  """
  compatibility_notes = {}
  match_compatibility = re.compile(r'[ \t]*@compatibility\((\w+)\)\s*\n'
                                   r'((?:[^@\n]*\n)+)'
                                   r'\s*@end_compatibility')
  for f in match_compatibility.finditer(doc):
    compatibility_notes[f.group(1)] = f.group(2)
  return match_compatibility.subn(r'', doc)[0], compatibility_notes


def _md_docstring(py_object, relative_path_to_root, reference_resolver):
  """Get the docstring from an object and make it into nice Markdown.

  For links within the same set of docs, the `relative_path_to_root` for a
  docstring on the page for `full_name` can be set to

  ```python
  relative_path_to_root = os.path.relpath(
    path='.', start=os.path.dirname(documentation_path(full_name)) or '.')
  ```

  Args:
    py_object: A python object to retrieve the docs for (class, function/method,
      or module).
    relative_path_to_root: The relative path from the location of the current
      document to the root of the Python API documentation. This is used to
      compute links for "@{symbol}" references.
    reference_resolver: An instance of ReferenceResolver.

  Returns:
    The docstring, or the empty string if no docstring was found.
  """
  # TODO(wicke): If this is a partial, use the .func docstring and add a note.
  raw_docstring = _get_raw_docstring(py_object)
  raw_docstring = reference_resolver.replace_references(
      raw_docstring, relative_path_to_root)
  raw_docstring, compatibility = _handle_compatibility(raw_docstring)
  raw_lines = raw_docstring.split('\n')

  # Define regular expressions used during parsing below.
  symbol_list_item_re = re.compile(r'^  (%s): ' % IDENTIFIER_RE)
  section_re = re.compile(r'^(\w+):\s*$')
  atat_re = re.compile(r'^\s*@@[a-zA-Z_.0-9]+\s*$')

  # Translate docstring line by line.
  in_special_section = False
  lines = []

  def is_section_start(i):
    # Previous line is empty, line i is "Word:", and next line is indented.
    return (i > 0  and i < len(raw_lines) and not raw_lines[i-1].strip() and
            re.match(section_re, raw_lines[i]) and
            len(raw_lines) > i+1 and raw_lines[i+1].startswith('  '))
  for i, line in enumerate(raw_lines):
    if re.match(atat_re, line):
      continue

    if not in_special_section and is_section_start(i):
      in_special_section = True
      lines.append('#### ' + section_re.sub(r'\1:', line))
      lines.append('')
      continue

    # If the next line starts a new section, this one ends. Add an extra line.
    if in_special_section and is_section_start(i+1):
      in_special_section = False
      lines.append('')

    if in_special_section:
      # Translate symbols in 'Args:', 'Parameters:', 'Raises:', etc. sections.
      lines.append(symbol_list_item_re.sub(r'* <b>`\1`</b>: ', line))
    else:
      lines.append(line)
  docstring = '\n'.join(lines)
  sorted_keys = compatibility.keys()
  sorted_keys.sort()
  for key in sorted_keys:
    value = compatibility[key]
    docstring += ('\n\n#### %s compatibility\n%s\n' % (key, value))
  # TODO(deannarubin): Improve formatting for devsite
  return docstring


def _get_arg_spec(func):
  """Extracts signature information from a function or functools.partial object.

  For functions, uses `inspect.getargspec`. For `functools.partial` objects,
  corrects the signature of the underlying function to take into account the
  removed arguments.

  Args:
    func: A function whose signature to extract.

  Returns:
    An `ArgSpec` namedtuple `(args, varargs, keywords, defaults)`, as returned
    by `inspect.getargspec`.
  """
  # getargspec does not work for functools.partial objects directly.
  if isinstance(func, functools.partial):
    argspec = inspect.getargspec(func.func)
    # Remove the args from the original function that have been used up.
    first_default_arg = (
        len(argspec.args or []) - len(argspec.defaults or []))
    partial_args = len(func.args)
    argspec_args = []

    if argspec.args:
      argspec_args = list(argspec.args[partial_args:])

    argspec_defaults = list(argspec.defaults or ())
    if argspec.defaults and partial_args > first_default_arg:
      argspec_defaults = list(argspec.defaults[partial_args-first_default_arg:])

    first_default_arg = max(0, first_default_arg - partial_args)
    for kwarg in (func.keywords or []):
      if kwarg in (argspec.args or []):
        i = argspec_args.index(kwarg)
        argspec_args.pop(i)
        if i >= first_default_arg:
          argspec_defaults.pop(i-first_default_arg)
        else:
          first_default_arg -= 1
    return inspect.ArgSpec(args=argspec_args,
                           varargs=argspec.varargs,
                           keywords=argspec.keywords,
                           defaults=tuple(argspec_defaults))
  else:  # Regular function or method, getargspec will work fine.
    return inspect.getargspec(func)


def _remove_first_line_indent(string):
  indent = len(re.match(r'^\s*', string).group(0))
  return '\n'.join([line[indent:] for line in string.split('\n')])


def _generate_signature(func, reverse_index):
  """Given a function, returns a string representing its args.

  This function produces a string representing the arguments to a python
  function, including surrounding parentheses. It uses inspect.getargspec, which
  does not generalize well to Python 3.x, which is more flexible in how *args
  and **kwargs are handled. This is not a problem in TF, since we have to remain
  compatible to Python 2.7 anyway.

  This function uses `__name__` for callables if it is available. This can lead
  to poor results for functools.partial and other callable objects.

  The returned string is Python code, so if it is included in a Markdown
  document, it should be typeset as code (using backticks), or escaped.

  Args:
    func: A function, method, or functools.partial to extract the signature for.
    reverse_index: A map from object ids to canonical full names to use.

  Returns:
    A string representing the signature of `func` as python code.
  """

  # This produces poor signatures for decorated functions.
  # TODO(wicke): We need to use something like the decorator module to fix it.

  args_list = []

  argspec = _get_arg_spec(func)
  first_arg_with_default = (
      len(argspec.args or []) - len(argspec.defaults or []))

  # Python documentation skips `self` when printing method signatures.
  # Note we cannot test for ismethod here since unbound methods do not register
  # as methods (in Python 3).
  first_arg = 1 if 'self' in argspec.args[:1] else 0

  # Add all args without defaults.
  for arg in argspec.args[first_arg:first_arg_with_default]:
    args_list.append(arg)

  # Add all args with defaults.
  if argspec.defaults:
    source = _remove_first_line_indent(inspect.getsource(func))
    func_ast = ast.parse(source)
    ast_defaults = func_ast.body[0].args.defaults

    for arg, default, ast_default in zip(
        argspec.args[first_arg_with_default:], argspec.defaults, ast_defaults):
      if id(default) in reverse_index:
        default_text = reverse_index[id(default)]
      else:
        default_text = codegen.to_source(ast_default)
        if default_text != repr(default):
          # This may be an internal name. If so, handle the ones we know about.
          # TODO(wicke): This should be replaced with a lookup in the index.
          # TODO(wicke): (replace first ident with tf., check if in index)
          internal_names = {
              'ops.GraphKeys': 'tf.GraphKeys',
              '_ops.GraphKeys': 'tf.GraphKeys',
              'init_ops.zeros_initializer': 'tf.zeros_initializer',
              'init_ops.ones_initializer': 'tf.ones_initializer',
              'saver_pb2.SaverDef': 'tf.SaverDef',
          }
          full_name_re = '^%s(.%s)+' % (IDENTIFIER_RE, IDENTIFIER_RE)
          match = re.match(full_name_re, default_text)
          if match:
            lookup_text = default_text
            for internal_name, public_name in six.iteritems(internal_names):
              if match.group(0).startswith(internal_name):
                lookup_text = public_name + default_text[len(internal_name):]
                break
            if default_text is lookup_text:
              print('WARNING: Using default arg, failed lookup: %s, repr: %r' %
                    (default_text, default))
            else:
              default_text = lookup_text

      args_list.append('%s=%s' % (arg, default_text))

  # Add *args and *kwargs.
  if argspec.varargs:
    args_list.append('*' + argspec.varargs)
  if argspec.keywords:
    args_list.append('**' + argspec.keywords)

  return '(%s)' % ', '.join(args_list)


def _get_guides_markdown(duplicate_names, guide_index, relative_path):
  all_guides = []
  for name in duplicate_names:
    all_guides.extend(guide_index.get(name, []))
  if not all_guides: return ''
  prefix = '../' * (relative_path.count('/') + 3)
  links = sorted(set([guide_ref.make_md_link(prefix)
                      for guide_ref in all_guides]))
  return 'See the guide%s: %s\n\n' % (
      's' if len(links) > 1 else '', ', '.join(links))


def _generate_markdown_for_function(full_name, duplicate_names,
                                    function, reference_resolver,
                                    reverse_index, guide_index):
  """Generate Markdown docs for a function or method.

  This function creates a documentation page for a function. It uses the
  function name (incl. signature) as the title, followed by a list of duplicate
  names (if there are any), and the Markdown formatted docstring of the
  function.

  Args:
    full_name: The preferred name of the function. Used in the title. Must not
      be present in `duplicate_of` (master names never are).
    duplicate_names: A sorted list of alternative names (incl. `full_name`).
    function: The python object referenced by `full_name`.
    reference_resolver: An instance of ReferenceResolver.
    reverse_index: A map from object ids in the index to full names.
    guide_index: A `dict` mapping symbol name strings to objects with a
      `make_md_link()` method.

  Returns:
    A string that can be written to a documentation file for this function.
  """
  # TODO(wicke): Make sure this works for partials.
  relative_path = os.path.relpath(
      path='.', start=os.path.dirname(documentation_path(full_name)) or '.')
  docstring = _md_docstring(function, relative_path, reference_resolver)
  signature = _generate_signature(function, reverse_index)
  guides = _get_guides_markdown(duplicate_names, guide_index, relative_path)

  if duplicate_names:
    aliases = '\n'.join(['### `%s`' % (name + signature)
                         for name in duplicate_names])
    aliases += '\n\n'
  else:
    aliases = ''

  return '# %s%s\n\n%s%s%s' % (
      full_name, signature, aliases, guides, docstring)


def _generate_markdown_for_class(full_name, duplicate_names, py_class,
                                 reference_resolver, tree, reverse_index,
                                 guide_index):
  """Generate Markdown docs for a class.

  This function creates a documentation page for a class. It uses the
  class name as the title, followed by a list of duplicate
  names (if there are any), the Markdown formatted docstring of the
  class, a list of links to all child class docs, a list of all properties
  including their docstrings, a list of all methods incl. their docstrings, and
  a list of all class member names (public fields).

  Args:
    full_name: The preferred name of the class. Used in the title. Must not
      be present in `duplicate_of` (master names never are).
    duplicate_names: A sorted list of alternative names (incl. `full_name`).
    py_class: The python object referenced by `full_name`.
    reference_resolver: An instance of ReferenceResolver.
    tree: A map from full names to the names of all documentable child objects.
    reverse_index: A map from object ids in the index to full names.
    guide_index: A `dict` mapping symbol name strings to objects with a
      `make_md_link()` method.

  Returns:
    A string that can be written to a documentation file for this class.
  """
  relative_path = os.path.relpath(
      path='.', start=os.path.dirname(documentation_path(full_name)) or '.')
  docstring = _md_docstring(py_class, relative_path, reference_resolver)
  guides = _get_guides_markdown(duplicate_names, guide_index, relative_path)
  if duplicate_names:
    aliases = '\n'.join(['### `class %s`' % name for name in duplicate_names])
    aliases += '\n\n'
  else:
    aliases = ''

  docs = '# %s\n\n%s%s%s\n\n' % (full_name, aliases, guides, docstring)

  field_names = []
  properties = []
  methods = []
  class_links = []
  for member in tree[full_name]:
    child_name = '.'.join([full_name, member])
    child = reference_resolver.py_name_to_object(child_name)

    if isinstance(child, property):
      properties.append((member, child))
    elif inspect.isclass(child):
      class_links.append(reference_resolver.python_link(
          'class ' + member, child_name, relative_path))
    elif inspect.ismethod(child) or inspect.isfunction(child):
      methods.append((member, child))
    else:
      # TODO(wicke): We may want to also remember the object itself.
      field_names.append(member)

  if class_links:
    docs += '## Child Classes\n'
    docs += '\n\n'.join(sorted(class_links))
    docs += '\n\n'

  if properties:
    docs += '## Properties\n\n'
    for property_name, prop in sorted(properties, key=lambda x: x[0]):
      docs += '<h3 id="%s"><code>%s</code></h3>\n\n%s\n\n' % (
          property_name, property_name,
          _md_docstring(prop, relative_path, reference_resolver))
    docs += '\n\n'

  if methods:
    docs += '## Methods\n\n'
    for method_name, method in sorted(methods, key=lambda x: x[0]):
      method_signature = method_name + _generate_signature(method,
                                                           reverse_index)
      docs += '<h3 id="%s"><code>%s</code></h3>\n\n%s\n\n' % (
          method_name, method_signature, _md_docstring(
              method, relative_path, reference_resolver))
    docs += '\n\n'

  if field_names:
    docs += '## Class Members\n\n'
    # TODO(wicke): Document the value of the members, at least for basic types.
    docs += '\n\n'.join(
        ['<h3 id="%s"><code>%s</code></h3>' % (field, field)
         for field in sorted(field_names)])

    docs += '\n\n'

  return docs


def _generate_markdown_for_module(full_name, duplicate_names, module,
                                  reference_resolver, tree):
  """Generate Markdown docs for a module.

  This function creates a documentation page for a module. It uses the
  module name as the title, followed by a list of duplicate
  names (if there are any), the Markdown formatted docstring of the
  class, and a list of links to all members of this module.

  Args:
    full_name: The preferred name of the module. Used in the title. Must not
      be present in `duplicate_of` (master names never are).
    duplicate_names: A sorted list of alternative names (incl. `full_name`).
    module: The python object referenced by `full_name`.
    reference_resolver: An instance of ReferenceResolver.
    tree: A map from full names to the names of all documentable child objects.

  Returns:
    A string that can be written to a documentation file for this module.
  """
  relative_path = os.path.relpath(
      path='.', start=os.path.dirname(documentation_path(full_name)) or '.')
  docstring = _md_docstring(module, relative_path, reference_resolver)
  if duplicate_names:
    aliases = '\n'.join(['### Module `%s`' % name for name in duplicate_names])
    aliases += '\n\n'
  else:
    aliases = ''

  member_names = tree.get(full_name, [])

  # Make links to all members.
  member_links = []
  for name in member_names:
    member_full_name = full_name + '.' + name if full_name else name
    member = reference_resolver.py_name_to_object(member_full_name)

    suffix = ''
    if inspect.isclass(member):
      link_text = 'class ' + name
    elif inspect.isfunction(member):
      link_text = name + '(...)'
    elif inspect.ismodule(member):
      link_text = name
      suffix = ' module'
    else:
      member_links.append('Constant ' + name)
      continue

    brief_docstring = _md_brief_docstring(
        member, relative_path, reference_resolver)
    if brief_docstring:
      suffix = '%s: %s' % (suffix, brief_docstring)

    member_links.append(reference_resolver.python_link(
        link_text, member_full_name, relative_path) + suffix)

  # TODO(deannarubin): Make this list into a table.

  return '# Module: %s\n\n%s%s\n\n## Members\n\n%s' % (
      full_name, aliases, docstring, '\n\n'.join(member_links))


_CODE_URL_PREFIX = (
    'https://www.tensorflow.org/code/tensorflow/')


def generate_markdown(full_name, py_object, duplicates, reference_resolver,
                      tree, reverse_index, guide_index, base_dir):
  """Generate Markdown docs for a given object that's part of the TF API.

  This function uses _md_docstring to obtain the docs pertaining to
  `object`.

  This function resolves '@{symbol}' references in the docstrings into links to
  the appropriate location. It also adds a list of alternative names for the
  symbol automatically.

  It assumes that the docs for each object live in a file given by
  `documentation_path`, and that relative links to files within the
  documentation are resolvable.

  The output is Markdown that can be written to file and published.

  Args:
    full_name: The fully qualified name of the symbol to be
      documented.
    py_object: The Python object to be documented. Its documentation is sourced
      from `py_object`'s docstring.
    duplicates: A `dict` mapping fully qualified names to a set of all
      aliases of this name. This is used to automatically generate a list of all
      aliases for each name.
    reference_resolver: An instance of ReferenceResolver.
    tree: A `dict` mapping a fully qualified name to the names of all its
      members. Used to populate the members section of a class or module page.
    reverse_index: A `dict` mapping objects in the index to full names.
    guide_index: A `dict` mapping symbol name strings to objects with a
      `make_md_link()` method.
    base_dir: A base path that is stripped from file locations written to the
      docs.

  Returns:
    A string containing the Markdown docs for `py_object`.

  Raises:
    RuntimeError: If an object is encountered for which we don't know how
      to make docs.
  """

  # Which other aliases exist for the object referenced by full_name?
  master_name = reference_resolver.py_master_name(full_name)
  duplicate_names = duplicates.get(master_name, [full_name])

  # TODO(wicke): Once other pieces are ready, enable this also for partials.
  if (inspect.ismethod(py_object) or inspect.isfunction(py_object) or
      # Some methods in classes from extensions come in as routines.
      inspect.isroutine(py_object)):
    markdown = _generate_markdown_for_function(
        master_name, duplicate_names, py_object, reference_resolver,
        reverse_index, guide_index)
  elif inspect.isclass(py_object):
    markdown = _generate_markdown_for_class(
        master_name, duplicate_names, py_object, reference_resolver, tree,
        reverse_index, guide_index)
  elif inspect.ismodule(py_object):
    markdown = _generate_markdown_for_module(
        master_name, duplicate_names, py_object, reference_resolver, tree)
  else:
    raise RuntimeError('Cannot make docs for object %s: %r' % (full_name,
                                                               py_object))

  # Every page gets a note on the bottom about where this object is defined
  # TODO(wicke): If py_object is decorated, get the decorated object instead.
  # TODO(wicke): Only use decorators that support this in TF.

  try:
    path = os.path.relpath(path=inspect.getfile(py_object), start=base_dir)

    # In case this is compiled, point to the original
    if path.endswith('.pyc'):
      path = path[:-1]

    # TODO(wicke): If this is a generated file, link to the source instead.
    # TODO(wicke): Move all generated files to a generated/ directory.
    # TODO(wicke): And make their source file predictable from the file name.
    # Never include links outside this code base.
    if not path.startswith('..'):
      if re.match('.*/gen_[^/]*.py$', path):
        # Generated file, don't incude a futile link.
        markdown += '\n\nDefined in `tensorflow/%s`.\n\n' % path
      else:
        markdown += '\n\nDefined in [`tensorflow/%s`](%s%s).\n\n' % (
            path, _CODE_URL_PREFIX, path)
  except TypeError:  # getfile throws TypeError if py_object is a builtin.
    markdown += '\n\nThis is an alias for a Python built-in.'

  return markdown


def generate_global_index(library_name, index, reference_resolver):
  """Given a dict of full names to python objects, generate an index page.

  The index page generated contains a list of links for all symbols in `index`
  that have their own documentation page.

  Args:
    library_name: The name for the documented library to use in the title.
    index: A dict mapping full names to python objects.
    reference_resolver: An instance of ReferenceResolver.

  Returns:
    A string containing an index page as Markdown.
  """
  symbol_links = []
  for full_name, py_object in six.iteritems(index):
    if (inspect.ismodule(py_object) or inspect.isfunction(py_object) or
        inspect.isclass(py_object)):
      # In Python 3, unbound methods are functions, so eliminate those.
      if inspect.isfunction(py_object):
        if full_name.count('.') == 0:
          parent_name = ''
        else:
          parent_name = full_name[:full_name.rfind('.')]
        if parent_name in index and inspect.isclass(index[parent_name]):
          # Skip methods (=functions with class parents).
          continue
      symbol_links.append((
          full_name, reference_resolver.python_link(full_name, full_name, '.')))

  lines = ['# All symbols in %s' % library_name, '']
  for _, link in sorted(symbol_links, key=lambda x: x[0]):
    lines.append('*  %s' % link)

  # TODO(deannarubin): Make this list into a table and add the brief docstring.
  # (use _md_brief_docstring)

  return '\n'.join(lines)
