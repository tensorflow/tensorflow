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

import functools
import inspect
import os
import re

import six

# A regular expression capturing a python indentifier.
IDENTIFIER_RE = '[a-zA-Z_][a-zA-Z0-9_]*'


def documentation_path(full_name):
  """Returns the file path for the documentation for the given API symbol.

  Given the fully qualified name of a library symbol, compute the path to which
  to write the documentation for that symbol (relative to a base directory).
  Documentation files are organized into directories that mirror the python
  module/class structure. The path for the top-level module (whose full name is
  '') is 'index.md'.

  Args:
    full_name: Fully qualified name of a library symbol.

  Returns:
    The file path to which to write the documentation for `full_name`.
  """
  # The main page is special, since it has no name in here.
  if not full_name:
    dirs = ['index']
  else:
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


def _get_brief_docstring(py_object):
  """Gets the one line docstring of a python object."""
  return _get_raw_docstring(py_object).split('\n')[0]


def _reference_to_link(ref_full_name, relative_path_to_root, duplicate_of):
  """Resolve a "@{symbol}" reference to a relative path, respecting duplicates.

  The input to this function should already be stripped of the '@' and '{}', and
  its output is only the link, not the full Markdown.

  Args:
    ref_full_name: The fully qualified name of the symbol to link to.
    relative_path_to_root: The relative path from the location of the current
      document to the root of the API documentation.
    duplicate_of: A map from duplicate full names to master names.

  Returns:
    A relative path that links from the documentation page of `from_full_name`
    to the documentation page of `ref_full_name`.
  """
  master_name = duplicate_of.get(ref_full_name, ref_full_name)
  ref_path = documentation_path(master_name)
  return os.path.join(relative_path_to_root, ref_path)


def _markdown_link(link_text, ref_full_name, relative_path_to_root,
                   duplicate_of):
  """Resolve a "@{symbol}" reference to a Markdown link, respecting duplicates.

  The input to this function should already be stripped of the '@' and '{}'.
  This function returns a Markdown link. It is assumed that this is a code
  reference, so the link text will always be rendered as code (using backticks).

  `link_text` should refer to a library symbol. You can either refer to it with
  or without the `tf.` prefix.

  Args:
    link_text: The text of the Markdown link.
    ref_full_name: The fully qualified name of the symbol to link to
      (may optionally include 'tf.').
    relative_path_to_root: The relative path from the location of the current
      document to the root of the API documentation.
    duplicate_of: A map from duplicate full names to master names.

  Returns:
    A markdown link from the documentation page of `from_full_name`
    to the documentation page of `ref_full_name`.
  """
  if ref_full_name.startswith('tf.'):
    ref_full_name = ref_full_name[3:]

  return '[`%s`](%s)' % (
      link_text,
      _reference_to_link(ref_full_name, relative_path_to_root, duplicate_of))


def replace_references(string, relative_path_to_root, duplicate_of):
  """Replace "@{symbol}" references with links to symbol's documentation page.

  This functions finds all occurrences of "@{symbol}" in `string` and replaces
  them with markdown links to the documentation page for "symbol".

  `relative_path_to_root` is the relative path from the document that contains
  the "@{symbol}" reference to the root of the API documentation that is linked
  to. If the containing page is part of the same API docset,
  `relative_path_to_root` can be set to
  `os.path.dirname(documentation_path(name))`, where `name` is the python name
  of the object whose documentation page the reference lives on.

  Args:
    string: A string in which "@{symbol}" references should be replaced.
    relative_path_to_root: The relative path from the contianing document to the
      root of the API documentation that is being linked to.
    duplicate_of: A map from duplicate names to preferred names of API symbols.

  Returns:
    `string`, with "@{symbol}" references replaced by Markdown links.
  """
  full_name_re = '%s(.%s)*' % (IDENTIFIER_RE, IDENTIFIER_RE)
  symbol_reference_re = re.compile(r'@\{(' + full_name_re + r')\}')
  match = symbol_reference_re.search(string)
  while match:
    symbol_name = match.group(1)
    link_text = _markdown_link(symbol_name, symbol_name,
                               relative_path_to_root, duplicate_of)

    # Remove only the '@symbol' part of the match, and replace with the link.
    string = string[:match.start()] + link_text + string[match.end():]
    match = symbol_reference_re.search(string,
                                       pos=match.start() + len(link_text))
  return string


def _md_docstring(py_object, relative_path_to_root, duplicate_of):
  """Get the docstring from an object and make it into nice Markdown.

  For links within the same set of docs, the `relative_path_to_root` for a
  docstring on the page for `full_name` can be set to

  ```python
  relative_path_to_root = os.path.relpath(
    os.path.dirname(documentation_path(full_name)) or '.', '.')
  ```

  Args:
    py_object: A python object to retrieve the docs for (class, function/method,
      or module).
    relative_path_to_root: The relative path from the location of the current
      document to the root of the API documentation. This is used to compute
      links for "@symbol" references.
    duplicate_of: A map from duplicate symbol names to master names. Used to
      resolve "@symbol" references.

  Returns:
    The docstring, or the empty string if no docstring was found.
  """
  # TODO(wicke): If this is a partial, use the .func docstring and add a note.
  raw_docstring = _get_raw_docstring(py_object)
  raw_lines = raw_docstring.split('\n')

  # Define regular expressions used during parsing below.
  symbol_list_item_re = re.compile(r'^  (%s): ' % IDENTIFIER_RE)
  section_re = re.compile(r'^(\w+):\s*$')

  # Translate docstring line by line.
  in_special_section = False
  lines = []

  def is_section_start(i):
    # Previous line is empty, line i is "Word:", and next line is indented.
    return (i > 0 and not raw_lines[i-1].strip() and
            re.match(section_re, raw_lines[i]) and
            len(raw_lines) > i+1 and raw_lines[i+1].startswith('  '))

  for i, line in enumerate(raw_lines):
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

  # TODO(deannarubin): Improve formatting for devsite
  # TODO(deannarubin): Interpret @compatibility and other formatting notes.

  return replace_references(docstring, relative_path_to_root, duplicate_of)


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


def _generate_signature(func):
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
    func: A function of method to extract the signature for (anything
      `inspect.getargspec` will accept).

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
    for arg, default in zip(
        argspec.args[first_arg_with_default:], argspec.defaults):
      # Some callables don't have __name__, fall back to including their repr.
      # TODO(wicke): This could be improved at least for common cases.
      if callable(default) and hasattr(default, '__name__'):
        args_list.append('%s=%s' % (arg, default.__name__))
      else:
        args_list.append('%s=%r' % (arg, default))

  # Add *args and *kwargs.
  if argspec.varargs:
    args_list.append('*' + argspec.varargs)
  if argspec.keywords:
    args_list.append('**' + argspec.keywords)

  return '(%s)' % ', '.join(args_list)


def _generate_markdown_for_function(full_name, duplicate_names,
                                    function, duplicate_of):
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
    duplicate_of: A map of duplicate full names to master names. Used to resolve
      @{symbol} references in the docstring.

  Returns:
    A string that can be written to a documentation file for this function.
  """
  # TODO(wicke): Make sure this works for partials.
  relative_path = os.path.relpath(
      os.path.dirname(documentation_path(full_name)) or '.', '.')
  docstring = _md_docstring(function, relative_path, duplicate_of)
  signature = _generate_signature(function)

  if duplicate_names:
    aliases = '\n'.join(['### `%s`' % (name + signature)
                         for name in duplicate_names])
    aliases += '\n\n'
  else:
    aliases = ''

  return '#`%s%s`\n\n%s%s' % (full_name, signature, aliases, docstring)


def _generate_markdown_for_class(full_name, duplicate_names, py_class,
                                 duplicate_of, index, tree):
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
    duplicate_of: A map of duplicate full names to master names. Used to resolve
      @{symbol} references in the docstrings.
    index: A map from full names to python object references.
    tree: A map from full names to the names of all documentable child objects.

  Returns:
    A string that can be written to a documentation file for this class.
  """
  relative_path = os.path.relpath(
      os.path.dirname(documentation_path(full_name)) or '.', '.')
  docstring = _md_docstring(py_class, relative_path, duplicate_of)
  if duplicate_names:
    aliases = '\n'.join(['### `class %s`' % name for name in duplicate_names])
    aliases += '\n\n'
  else:
    aliases = ''

  docs = '# `%s`\n\n%s%s\n\n' % (full_name, aliases, docstring)

  field_names = []
  properties = []
  methods = []
  class_links = []
  for member in tree[full_name]:
    child_name = '.'.join([full_name, member])
    child = index[child_name]

    if isinstance(child, property):
      properties.append((member, child))
    elif inspect.isclass(child):
      class_links.append(_markdown_link('class ' + member, child_name,
                                        relative_path, duplicate_of))
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
      docs += '### `%s`\n\n%s\n\n' % (
          property_name, _md_docstring(prop, relative_path, duplicate_of))
    docs += '\n\n'

  if methods:
    docs += '## Methods\n\n'
    for method_name, method in sorted(methods, key=lambda x: x[0]):
      method_signature = method_name + _generate_signature(method)
      docs += '### `%s`\n\n%s\n\n' % (method_signature,
                                      _md_docstring(method, relative_path,
                                                    duplicate_of))
    docs += '\n\n'

  if field_names:
    docs += '## Class Members\n\n'
    # TODO(wicke): Document the value of the members, at least for basic types.
    docs += '\n\n'.join(sorted(field_names))
    docs += '\n\n'

  return docs


def _generate_markdown_for_module(full_name, duplicate_names, module,
                                  duplicate_of, index, tree):
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
    duplicate_of: A map of duplicate full names to master names. Used to resolve
      @{symbol} references in the docstrings.
    index: A map from full names to python object references.
    tree: A map from full names to the names of all documentable child objects.

  Returns:
    A string that can be written to a documentation file for this module.
  """
  relative_path = os.path.relpath(
      os.path.dirname(documentation_path(full_name)) or '.', '.')
  docstring = _md_docstring(module, relative_path, duplicate_of)
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
    member = index[member_full_name]

    if inspect.isclass(member):
      link_text = 'class ' + name
    elif inspect.isfunction(member):
      link_text = name + _generate_signature(member)
    else:
      link_text = name

    member_links.append(_markdown_link(link_text, member_full_name,
                                       relative_path, duplicate_of))

  # TODO(deannarubin): Make this list into a table and add the brief docstring.
  # (use _get_brief_docstring)

  return '# Module `%s`\n\n%s%s\n\n## Members\n\n%s' % (
      full_name, aliases, docstring, '\n\n'.join(member_links))


def generate_markdown(full_name, py_object,
                      duplicate_of, duplicates,
                      index, tree, base_dir):
  """Generate Markdown docs for a given object that's part of the TF API.

  This function uses _md_docstring to obtain the docs pertaining to
  `object`.

  This function resolves '@symbol' references in the docstrings into links to
  the appropriate location. It also adds a list of alternative names for the
  symbol automatically.

  It assumes that the docs for each object live in a file given by
  `documentation_path`, and that relative links to files within the
  documentation are resolvable.

  The output is Markdown that can be written to file and published.

  Args:
    full_name: The fully qualified name (excl. "tf.") of the symbol to be
      documented.
    py_object: The Python object to be documented. Its documentation is sourced
      from `py_object`'s docstring.
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
    base_dir: A base path that is stripped from file locations written to the
      docs.

  Returns:
    A string containing the Markdown docs for `py_object`.

  Raises:
    RuntimeError: If an object is encountered for which we don't know how
      to make docs.
  """

  # Which other aliases exist for the object referenced by full_name?
  master_name = duplicate_of.get(full_name, full_name)
  duplicate_names = duplicates.get(master_name, [full_name])

  # TODO(wicke): Once other pieces are ready, enable this also for partials.
  if (inspect.ismethod(py_object) or inspect.isfunction(py_object) or
      # Some methods in classes from extensions come in as routines.
      inspect.isroutine(py_object)):
    markdown = _generate_markdown_for_function(master_name, duplicate_names,
                                               py_object, duplicate_of)
  elif inspect.isclass(py_object):
    markdown = _generate_markdown_for_class(master_name, duplicate_names,
                                            py_object, duplicate_of,
                                            index, tree)
  elif inspect.ismodule(py_object):
    markdown = _generate_markdown_for_module(master_name, duplicate_names,
                                             py_object, duplicate_of,
                                             index, tree)
  else:
    raise RuntimeError('Cannot make docs for object %s: %r' % (full_name,
                                                               py_object))

  # Every page gets a note on the bottom about where this object is defined
  # TODO(wicke): If py_object is decorated, get the decorated object instead.
  # TODO(wicke): Only use decorators that support this in TF.

  try:
    path = os.path.relpath(inspect.getfile(py_object), base_dir)

    # TODO(wicke): If this is a generated file, point to the source instead.

    # Never include links outside this code base.
    if not path.startswith('..'):
      # TODO(wicke): Make this a link to github.
      markdown += '\n\ndefined in %s\n\n' % path
  except TypeError:  # getfile throws TypeError if py_object is a builtin.
    markdown += '\n\nthis is an alias for a Python built-in.'

  return markdown


def generate_global_index(library_name, root_name, index, duplicate_of):
  """Given a dict of full names to python objects, generate an index page.

  The index page generated contains a list of links for all symbols in `index`
  that have their own documentation page.

  Args:
    library_name: The name for the documented library to use in the title.
    root_name: The name to use for the root module.
    index: A dict mapping full names to python objects.
    duplicate_of: A map of duplicate names to preferred names.

  Returns:
    A string containing an index page as Markdown.
  """
  symbol_links = []
  for full_name, py_object in six.iteritems(index):
    index_name = full_name or root_name
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
      
      symbol_links.append((index_name,
                           _markdown_link(index_name, full_name,
                                          '.', duplicate_of)))

  lines = ['# All symbols in %s' % library_name, '']
  for _, link in sorted(symbol_links, key=lambda x: x[0]):
    lines.append('*  %s' % link)

  # TODO(deannarubin): Make this list into a table and add the brief docstring.
  # (use _get_brief_docstring)

  return '\n'.join(lines)
