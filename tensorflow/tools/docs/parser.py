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
import collections
import functools
import inspect
import os
import re

import codegen
import six

from google.protobuf.message import Message as ProtoMessage

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
    py_module_names: A list of string names of Python modules.
  """

  def __init__(self, duplicate_of, doc_index, index, py_module_names):
    self._duplicate_of = duplicate_of
    self._doc_index = doc_index
    self._index = index
    self._py_module_names = py_module_names

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
    link = self.reference_to_url(ref_full_name, relative_path_to_root)
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

  def reference_to_url(self, ref_full_name, relative_path_to_root):
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
      return self._doc_link(
          string, link_text, manual_link_text, relative_path_to_root)

    elif string.startswith('tensorflow::'):
      # C++ symbol
      return self._cc_link(
          string, link_text, manual_link_text, relative_path_to_root)

    else:
      is_python = False
      for py_module_name in self._py_module_names:
        if string == py_module_name or string.startswith(py_module_name + '.'):
          is_python = True
          break
      if is_python:  # Python symbol
        return self.python_link(link_text, string, relative_path_to_root,
                                code_ref=not manual_link_text)

    # Error!
    log_error('Did not understand "@{%s}"' % string)
    return 'ERROR:%s' % string

  def _doc_link(self, string, link_text, manual_link_text,
                relative_path_to_root):
    """Generate a link for a @{$...} reference."""
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
    return self._doc_missing(string, hash_tag, link_text, manual_link_text,
                             relative_path_to_root)

  def _doc_missing(self, string, unused_hash_tag, link_text,
                   unused_manual_link_text, unused_relative_path_to_root):
    """Generate an error for unrecognized @{$...} references."""
    log_error('Handle doc reference "@{$%s}"' % string)
    return link_text

  def _cc_link(self, string, link_text, unused_manual_link_text,
               relative_path_to_root):
    """Generate a link for a @{tensorflow::...} reference."""
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


def _gen_pairs(items):
  """Given an list of items [a,b,a,b...], generate pairs [(a,b),(a,b)...].

  Args:
    items: A list of items (length must be even)

  Yields:
    The original items, in pairs
  """
  assert len(items) % 2 == 0
  items = iter(items)
  while True:
    yield next(items), next(items)


class _FunctionDetail(
    collections.namedtuple('_FunctionDetail', ['keyword', 'header', 'items'])):
  """A simple class to contain function details.

  Composed of a "keyword", a possibly empty "header" string, and a possibly
  empty
  list of key-value pair "items".
  """
  __slots__ = []

  def __str__(self):
    """Return the original string that represents the function detail."""
    parts = [self.keyword + ':\n']
    parts.append(self.header)
    for key, value in self.items:
      parts.append('  ' + key + ':')
      parts.append(value)

    return ''.join(parts)


def _parse_function_details(docstring):
  r"""Given a docstring, split off the header and parse the function details.

  For example the docstring of tf.nn.relu:

  '''Computes rectified linear: `max(features, 0)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`,
      `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`,
      `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  '''

  This is parsed, and returned as:

  ```
  ('Computes rectified linear: `max(features, 0)`.\n\n', [
      _FunctionDetail(
          keyword='Args',
          header='',
          items=[
              ('features', ' A `Tensor`. Must be ...'),
              ('name', ' A name for the operation (optional).\n\n')]),
      _FunctionDetail(
          keyword='Returns',
          header='  A `Tensor`. Has the same type as `features`.',
          items=[])
  ])
  ```

  Args:
    docstring: The docstring to parse

  Returns:
    A (header, function_details) pair, where header is a string and
    function_details is a (possibly empty) list of `_FunctionDetail` objects.
  """

  detail_keywords = '|'.join([
      'Args', 'Arguments', 'Fields', 'Returns', 'Yields', 'Raises', 'Attributes'
  ])
  # TODO(markdaoust): Check if the leading blank line is used consistently.
  tag_re = re.compile('(?<=\n\n)(' + detail_keywords + '):\n', re.MULTILINE)
  parts = tag_re.split(docstring)

  # The first part is the main docstring
  docstring = parts[0]

  # Everything else alternates keyword-content
  pairs = list(_gen_pairs(parts[1:]))

  function_details = []
  item_re = re.compile(r'^  (\w+):', re.MULTILINE)

  for keyword, content in pairs:
    content = item_re.split(content)
    header = content[0]
    items = list(_gen_pairs(content[1:]))

    function_details.append(_FunctionDetail(keyword, header, items))

  return docstring, function_details


_DocstringInfo = collections.namedtuple('_DocstringInfo', [
    'brief', 'docstring', 'function_details', 'compatibility'
])


def _parse_md_docstring(py_object, relative_path_to_root, reference_resolver):
  """Parse the object's docstring and return a `_DocstringInfo`.

  This function clears @@'s from the docstring, and replaces @{} references
  with markdown links.

  For links within the same set of docs, the `relative_path_to_root` for a
  docstring on the page for `full_name` can be set to:

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
    A _DocstringInfo object, all fields will be empty if no docstring was found.
  """
  # TODO(wicke): If this is a partial, use the .func docstring and add a note.
  raw_docstring = _get_raw_docstring(py_object)

  raw_docstring = reference_resolver.replace_references(
      raw_docstring, relative_path_to_root)

  atat_re = re.compile(r' *@@[a-zA-Z_.0-9]+ *$')
  raw_docstring = '\n'.join(
      line for line in raw_docstring.split('\n') if not atat_re.match(line))

  docstring, compatibility = _handle_compatibility(raw_docstring)
  docstring, function_details = _parse_function_details(docstring)

  return _DocstringInfo(
      docstring.split('\n')[0], docstring, function_details, compatibility)


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
  """Given a function, returns a list of strings representing its args.

  This function produces a list of strings representing the arguments to a
  python function. It uses inspect.getargspec, which
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
    A list of strings representing the argument signature of `func` as python
    code.
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
    try:
      source = _remove_first_line_indent(inspect.getsource(func))
      func_ast = ast.parse(source)
      ast_defaults = func_ast.body[0].args.defaults
    except IOError:  # If this is a builtin, getsource fails with IOError
      # If we cannot get the source, assume the AST would be equal to the repr
      # of the defaults.
      ast_defaults = [None] * len(argspec.defaults)

    for arg, default, ast_default in zip(
        argspec.args[first_arg_with_default:], argspec.defaults, ast_defaults):
      if id(default) in reverse_index:
        default_text = reverse_index[id(default)]
      elif ast_default is not None:
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
              'saver_pb2.SaverDef': 'tf.train.SaverDef',
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
      else:
        default_text = repr(default)

      args_list.append('%s=%s' % (arg, default_text))

  # Add *args and *kwargs.
  if argspec.varargs:
    args_list.append('*' + argspec.varargs)
  if argspec.keywords:
    args_list.append('**' + argspec.keywords)

  return args_list


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


def _get_defining_class(py_class, name):
  for cls in inspect.getmro(py_class):
    if name in cls.__dict__:
      return cls
  return None


class _LinkInfo(
    collections.namedtuple(
        '_LinkInfo', ['short_name', 'full_name', 'obj', 'doc', 'url'])):

  __slots__ = []

  def is_link(self):
    return True


class _OtherMemberInfo(
    collections.namedtuple('_OtherMemberInfo',
                           ['short_name', 'full_name', 'obj', 'doc'])):

  __slots__ = []

  def is_link(self):
    return False


_PropertyInfo = collections.namedtuple(
    '_PropertyInfo', ['short_name', 'full_name', 'obj', 'doc'])

_MethodInfo = collections.namedtuple(
    '_MethodInfo', ['short_name', 'full_name', 'obj', 'doc', 'signature'])


class _FunctionPageInfo(object):
  """Collects docs For a function Page."""

  def __init__(self, full_name):
    self._full_name = full_name
    self._defined_in = None
    self._aliases = None
    self._doc = None
    self._guides = None

    self._signature = None

  def for_function(self):
    return True

  def for_class(self):
    return False

  def for_module(self):
    return False

  @property
  def full_name(self):
    return self._full_name

  @property
  def short_name(self):
    return self._full_name.split('.')[-1]

  @property
  def defined_in(self):
    return self._defined_in

  def set_defined_in(self, defined_in):
    assert self.defined_in is None
    self._defined_in = defined_in

  @property
  def aliases(self):
    return self._aliases

  def set_aliases(self, aliases):
    assert self.aliases is None
    self._aliases = aliases

  @property
  def doc(self):
    return self._doc

  def set_doc(self, doc):
    assert self.doc is None
    self._doc = doc

  @property
  def guides(self):
    return self._guides

  def set_guides(self, guides):
    assert self.guides is None
    self._guides = guides

  @property
  def signature(self):
    return self._signature

  def set_signature(self, function, reverse_index):
    """Attach the function's signature.

    Args:
      function: The python function being documented.
      reverse_index: A map from object ids in the index to full names.
    """

    assert self.signature is None
    self._signature = _generate_signature(function, reverse_index)


class _ClassPageInfo(object):
  """Collects docs for a class page."""

  def __init__(self, full_name):
    self._full_name = full_name
    self._defined_in = None
    self._aliases = None
    self._doc = None
    self._guides = None

    self._properties = []
    self._methods = []
    self._classes = []
    self._other_members = []

  def for_function(self):
    return False

  def for_class(self):
    return True

  def for_module(self):
    return False

  @property
  def full_name(self):
    return self._full_name

  @property
  def short_name(self):
    return self._full_name.split('.')[-1]

  @property
  def defined_in(self):
    return self._defined_in

  def set_defined_in(self, defined_in):
    assert self.defined_in is None
    self._defined_in = defined_in

  @property
  def aliases(self):
    return self._aliases

  def set_aliases(self, aliases):
    assert self.aliases is None
    self._aliases = aliases

  @property
  def doc(self):
    return self._doc

  def set_doc(self, doc):
    assert self.doc is None
    self._doc = doc

  @property
  def guides(self):
    return self._guides

  def set_guides(self, guides):
    assert self.guides is None
    self._guides = guides

  @property
  def properties(self):
    return self._properties

  def _add_property(self, short_name, full_name, obj, doc):
    property_info = _PropertyInfo(short_name, full_name, obj, doc)
    self._properties.append(property_info)

  @property
  def methods(self):
    return self._methods

  def _add_method(self, short_name, full_name, obj, doc, signature):
    method_info = _MethodInfo(short_name, full_name, obj, doc, signature)
    self._methods.append(method_info)

  @property
  def classes(self):
    return self._classes

  def _add_class(self, short_name, full_name, obj, doc, url):
    page_info = _LinkInfo(short_name, full_name, obj, doc, url)

    self._classes.append(page_info)

  @property
  def other_members(self):
    return self._other_members

  def _add_other_member(self, short_name, full_name, obj, doc):
    other_member_info = _OtherMemberInfo(short_name, full_name, obj, doc)
    self._other_members.append(other_member_info)

  def collect_docs_for_class(self, py_class,
                             reference_resolver, tree, reverse_index):
    """Collect information necessary specifically for a class's doc page.

    Mainly, this is details about information about the class's members.

    Args:
      py_class: The class object to collect docs for.
      reference_resolver: An instance of ReferenceResolver.
      tree: A map from full names to the names of all documentable child
        objects.
      reverse_index: A map from object ids in the index to full names.
    """
    doc_path = documentation_path(self.full_name)
    relative_path = os.path.relpath(
        path='.', start=os.path.dirname(doc_path) or '.')

    for short_name in tree[self.full_name]:
      # Remove builtin members that we never want to document.
      if short_name in ['__class__', '__base__', '__weakref__', '__doc__',
                        '__module__', '__dict__', '__abstractmethods__',
                        '__slots__', '__getnewargs__']:
        continue

      child_name = '.'.join([self.full_name, short_name])
      child = reference_resolver.py_name_to_object(child_name)

      # Don't document anything that is defined in object or by protobuf.
      defining_class = _get_defining_class(py_class, short_name)
      if (defining_class is object or
          defining_class is type or defining_class is tuple or
          defining_class is BaseException or defining_class is Exception or
          # The following condition excludes most protobuf-defined symbols.
          defining_class and defining_class.__name__ in ['CMessage', 'Message',
                                                         'MessageMeta']):
        continue
      # TODO(markdaoust): Add a note in child docs showing the defining class.

      child_doc = _parse_md_docstring(child, relative_path, reference_resolver)

      if isinstance(child, property):
        self._add_property(short_name, child_name, child, child_doc)

      elif inspect.isclass(child):
        if defining_class is None:
          continue
        url = reference_resolver.reference_to_url(
            child_name, relative_path)
        self._add_class(short_name, child_name, child, child_doc, url)

      elif (inspect.ismethod(child) or inspect.isfunction(child) or
            inspect.isroutine(child)):
        if defining_class is None:
          continue

        # Omit methods defined by namedtuple.
        original_method = defining_class.__dict__[short_name]
        if (hasattr(original_method, '__module__') and
            (original_method.__module__ or '').startswith('namedtuple')):
          continue

        # Some methods are often overridden without documentation. Because it's
        # obvious what they do, don't include them in the docs if there's no
        # docstring.
        if not child_doc.brief.strip() and short_name in [
            '__str__', '__repr__', '__hash__', '__del__', '__copy__']:
          print('Skipping %s, defined in %s, no docstring.' % (child_name,
                                                               defining_class))
          continue

        try:
          child_signature = _generate_signature(child, reverse_index)
        except TypeError:
          # If this is a (dynamically created) slot wrapper, inspect will
          # raise typeerror when trying to get to the code. Ignore such
          # functions.
          continue

        self._add_method(short_name, child_name, child, child_doc,
                         child_signature)
      else:
        # Exclude members defined by protobuf that are useless
        if issubclass(py_class, ProtoMessage):
          if (short_name.endswith('_FIELD_NUMBER') or
              short_name in ['__slots__', 'DESCRIPTOR']):
            continue

        # TODO(wicke): We may want to also remember the object itself.
        self._add_other_member(short_name, child_name, child, child_doc)


class _ModulePageInfo(object):
  """Collects docs for a module page."""

  def __init__(self, full_name):
    self._full_name = full_name
    self._defined_in = None
    self._aliases = None
    self._doc = None
    self._guides = None

    self._members = []

  def for_function(self):
    return False

  def for_class(self):
    return False

  def for_module(self):
    return True

  @property
  def full_name(self):
    return self._full_name

  @property
  def short_name(self):
    return self._full_name.split('.')[-1]

  @property
  def defined_in(self):
    return self._defined_in

  def set_defined_in(self, defined_in):
    assert self.defined_in is None
    self._defined_in = defined_in

  @property
  def aliases(self):
    return self._aliases

  def set_aliases(self, aliases):
    assert self.aliases is None
    self._aliases = aliases

  @property
  def doc(self):
    return self._doc

  def set_doc(self, doc):
    assert self.doc is None
    self._doc = doc

  @property
  def guides(self):
    return self._guides

  def set_guides(self, guides):
    assert self.guides is None
    self._guides = guides

  @property
  def members(self):
    return self._members

  def _add_member(self, short_name, full_name, obj, doc, url=None):
    if url is None:
      member = _OtherMemberInfo(short_name, full_name, obj, doc)
    else:
      member = _LinkInfo(short_name, full_name, obj, doc, url)

    self._members.append(member)

  def collect_docs_for_module(self, reference_resolver, tree):
    """Collect information necessary specifically for a module's doc page.

    Mainly this is information about the members of the module.

    Args:
      reference_resolver: An instance of ReferenceResolver.
      tree: A map from full names to the names of all documentable child
        objects.
    """
    relative_path = os.path.relpath(
        path='.',
        start=os.path.dirname(documentation_path(self.full_name)) or '.')

    member_names = tree.get(self.full_name, [])
    for name in member_names:

      if name in ['__builtins__', '__doc__', '__file__', '__name__', '__path__',
                  '__package__']:
        continue

      member_full_name = self.full_name + '.' + name if self.full_name else name
      member = reference_resolver.py_name_to_object(member_full_name)

      member_doc = _parse_md_docstring(member, relative_path,
                                       reference_resolver)

      if (not inspect.isclass(member) and not inspect.isfunction(member) and
          not inspect.ismodule(member)):
        self._add_member(name, member_full_name, member, member_doc)
        continue

      url = reference_resolver.reference_to_url(
          member_full_name, relative_path)

      self._add_member(name, member_full_name, member, member_doc, url)


class ParserConfig(object):

  def __init__(self, reference_resolver, duplicates, tree, reverse_index,
               guide_index, base_dir):
    """Object with the common config for docs_for_object() calls.

    Args:
      reference_resolver: An instance of ReferenceResolver.
      duplicates: A `dict` mapping fully qualified names to a set of all
        aliases of this name. This is used to automatically generate a list of
        all aliases for each name.
      tree: A `dict` mapping a fully qualified name to the names of all its
        members. Used to populate the members section of a class or module page.
      reverse_index: A `dict` mapping objects in the index to full names.
      guide_index: A `dict` mapping symbol name strings to objects with a
        `make_md_link()` method.
      base_dir: A base path that is stripped from file locations written to the
        docs.
    """
    self.reference_resolver = reference_resolver
    self.duplicates = duplicates
    self.tree = tree
    self.reverse_index = reverse_index
    self.guide_index = guide_index
    self.base_dir = base_dir
    self.defined_in_prefix = 'tensorflow/'
    self.code_url_prefix = (
        'https://www.tensorflow.org/code/tensorflow/')  # pylint: disable=line-too-long


def docs_for_object(full_name, py_object, parser_config):
  """Return a PageInfo object describing a given object from the TF API.

  This function uses _parse_md_docstring to parse the docs pertaining to
  `object`.

  This function resolves '@{symbol}' references in the docstrings into links to
  the appropriate location. It also adds a list of alternative names for the
  symbol automatically.

  It assumes that the docs for each object live in a file given by
  `documentation_path`, and that relative links to files within the
  documentation are resolvable.

  Args:
    full_name: The fully qualified name of the symbol to be
      documented.
    py_object: The Python object to be documented. Its documentation is sourced
      from `py_object`'s docstring.
    parser_config: A ParserConfig object.

  Returns:
    Either a `_FunctionPageInfo`, `_ClassPageInfo`, or a `_ModulePageInfo`
    depending on the type of the python object being documented.

  Raises:
    RuntimeError: If an object is encountered for which we don't know how
      to make docs.
  """

  # Which other aliases exist for the object referenced by full_name?
  master_name = parser_config.reference_resolver.py_master_name(full_name)
  duplicate_names = parser_config.duplicates.get(master_name, [full_name])

  # TODO(wicke): Once other pieces are ready, enable this also for partials.
  if (inspect.ismethod(py_object) or inspect.isfunction(py_object) or
      # Some methods in classes from extensions come in as routines.
      inspect.isroutine(py_object)):
    page_info = _FunctionPageInfo(master_name)
    page_info.set_signature(py_object, parser_config.reverse_index)

  elif inspect.isclass(py_object):
    page_info = _ClassPageInfo(master_name)
    page_info.collect_docs_for_class(py_object,
                                     parser_config.reference_resolver,
                                     parser_config.tree,
                                     parser_config.reverse_index)

  elif inspect.ismodule(py_object):
    page_info = _ModulePageInfo(master_name)
    page_info.collect_docs_for_module(parser_config.reference_resolver,
                                      parser_config.tree)

  else:
    raise RuntimeError('Cannot make docs for object %s: %r' % (full_name,
                                                               py_object))

  relative_path = os.path.relpath(
      path='.', start=os.path.dirname(documentation_path(full_name)) or '.')

  page_info.set_doc(_parse_md_docstring(
      py_object, relative_path, parser_config.reference_resolver))

  page_info.set_aliases(duplicate_names)

  page_info.set_guides(_get_guides_markdown(
      duplicate_names, parser_config.guide_index, relative_path))

  page_info.set_defined_in(_get_defined_in(py_object, parser_config))

  return page_info


class _PythonBuiltin(object):
  """This class indicated that the object in question is a python builtin.

  This can be used for the `defined_in` slot of the `PageInfo` objects.
  """

  def is_builtin(self):
    return True

  def is_python_file(self):
    return False

  def is_generated_file(self):
    return False

  def __str__(self):
    return 'This is an alias for a Python built-in.\n\n'


class _PythonFile(object):
  """This class indicates that the object is defined in a regular python file.

  This can be used for the `defined_in` slot of the `PageInfo` obejcts.
  """

  def __init__(self, path, parser_config):
    self.path = path
    self.path_prefix = parser_config.defined_in_prefix
    self.code_url_prefix = parser_config.code_url_prefix

  def is_builtin(self):
    return False

  def is_python_file(self):
    return True

  def is_generated_file(self):
    return False

  def __str__(self):
    return 'Defined in [`{prefix}{path}`]({code_prefix}{path}).\n\n'.format(
        path=self.path, prefix=self.path_prefix,
        code_prefix=self.code_url_prefix)


class _ProtoFile(object):
  """This class indicates that the object is defined in a .proto file.

  This can be used for the `defined_in` slot of the `PageInfo` objects.
  """

  def __init__(self, path, parser_config):
    self.path = path
    self.path_prefix = parser_config.defined_in_prefix
    self.code_url_prefix = parser_config.code_url_prefix

  def is_builtin(self):
    return False

  def is_python_file(self):
    return False

  def is_generated_file(self):
    return False

  def __str__(self):
    return 'Defined in [`{prefix}{path}`]({code_prefix}{path}).\n\n'.format(
        path=self.path, prefix=self.path_prefix,
        code_prefix=self.code_url_prefix)


class _GeneratedFile(object):
  """This class indicates that the object is defined in a generated python file.

  Generated files should not be linked to directly.

  This can be used for the `defined_in` slot of the `PageInfo` objects.
  """

  def __init__(self, path, parser_config):
    self.path = path
    self.path_prefix = parser_config.defined_in_prefix

  def is_builtin(self):
    return False

  def is_python_file(self):
    return False

  def is_generated_file(self):
    return True

  def __str__(self):
    return 'Defined in `%s%s`.\n\n' % (self.path_prefix, self.path)


def _get_defined_in(py_object, parser_config):
  """Returns a description of where the passed in python object was defined.

  Arguments:
    py_object: The Python object.
    parser_config: A ParserConfig object.

  Returns:
    Either a `_PythonBuiltin`, `_PythonFile`, or a `_GeneratedFile`
  """
  # Every page gets a note about where this object is defined
  # TODO(wicke): If py_object is decorated, get the decorated object instead.
  # TODO(wicke): Only use decorators that support this in TF.

  try:
    path = os.path.relpath(path=inspect.getfile(py_object),
                           start=parser_config.base_dir)
  except TypeError:  # getfile throws TypeError if py_object is a builtin.
    return _PythonBuiltin()

  # TODO(wicke): If this is a generated file, link to the source instead.
  # TODO(wicke): Move all generated files to a generated/ directory.
  # TODO(wicke): And make their source file predictable from the file name.

  # In case this is compiled, point to the original
  if path.endswith('.pyc'):
    path = path[:-1]

  # Never include links outside this code base.
  if path.startswith('..'):
    return None

  if re.match(r'.*/gen_[^/]*\.py$', path):
    return _GeneratedFile(path, parser_config)
  elif re.match(r'.*_pb2\.py$', path):
    # The _pb2.py files all appear right next to their defining .proto file.
    return _ProtoFile(path[:-7] + '.proto', parser_config)
  else:
    return _PythonFile(path, parser_config)


# TODO(markdaoust): This should just parse, pretty_docs should generate the md.
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
  return '\n'.join(lines)
