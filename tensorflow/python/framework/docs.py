# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Updates generated docs from Python doc comments.

Updates the documentation files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import re


_arg_re = re.compile(" *([*]{0,2}[a-zA-Z][a-zA-Z0-9_]*):")
_section_re = re.compile("([A-Z][a-zA-Z ]*):$")
_always_drop_symbol_re = re.compile("_[_a-zA-Z0-9]")
_anchor_re = re.compile(r"^[\w.]+$")
_member_mark = "@@"
_indiv_dir = "functions_and_classes"


class Document(object):
  """Base class for an automatically generated document."""

  def write_markdown_to_file(self, f):
    """Writes a Markdown-formatted version of this document to file `f`.

    Args:
      f: The output file.
    """
    raise NotImplementedError("Document.WriteToFile")


class Index(Document):
  """An automatically generated index for a collection of documents."""

  def __init__(self, module_to_name, members, filename_to_library_map,
               path_prefix):
    """Creates a new Index.

    Args:
      module_to_name: Dictionary mapping modules to short names.
      members: Dictionary mapping member name to (fullname, member).
      filename_to_library_map: A list of (filename, Library) pairs. The order
        corresponds to the order in which the libraries appear in the index.
      path_prefix: Prefix to add to links in the index.
    """
    self._module_to_name = module_to_name
    self._members = members
    self._filename_to_library_map = filename_to_library_map
    self._path_prefix = path_prefix

  def write_markdown_to_file(self, f):
    """Writes this index to file `f`.

    The output is formatted as an unordered list. Each list element
    contains the title of the library, followed by a list of symbols
    in that library hyperlinked to the corresponding anchor in that
    library.

    Args:
      f: The output file.
    """
    print("<!-- This file is machine generated: DO NOT EDIT! -->", file=f)
    print("", file=f)
    print("# TensorFlow Python reference documentation", file=f)
    print("", file=f)
    fullname_f = lambda name: self._members[name][0]
    anchor_f = lambda name: _get_anchor(self._module_to_name, fullname_f(name))

    for filename, library in self._filename_to_library_map:
      sorted_names = sorted(library.mentioned, key=lambda x: (str.lower(x), x))
      member_names = [n for n in sorted_names if n in self._members]
      # TODO(wicke): This is a hack that should be removed as soon as the
      # website code allows it.
      full_filename = self._path_prefix + filename
      links = ["[`%s`](%s#%s)" % (name, full_filename, anchor_f(name))
               for name in member_names]
      if links:
        print("* **[%s](%s)**:" % (library.title, full_filename), file=f)
        for link in links:
          print("  * %s" % link, file=f)
        print("", file=f)


def collect_members(module_to_name, exclude=()):
  """Collect all symbols from a list of modules.

  Args:
    module_to_name: Dictionary mapping modules to short names.
    exclude: Set of fully qualified names to exclude.

  Returns:
    Dictionary mapping name to (fullname, member) pairs.

  Raises:
    RuntimeError: if we can not resolve a name collision.
  """
  members = {}
  for module, module_name in module_to_name.items():
    all_names = getattr(module, "__all__", None)
    for name, member in inspect.getmembers(module):
      if ((inspect.isfunction(member) or inspect.isclass(member)) and
          not _always_drop_symbol_re.match(name) and
          (all_names is None or name in all_names)):
        fullname = "%s.%s" % (module_name, name)
        if fullname in exclude:
          continue
        if name in members:
          other_fullname, other_member = members[name]
          if member is not other_member:
            raise RuntimeError("Short name collision between %s and %s" %
                               (fullname, other_fullname))
          if len(fullname) == len(other_fullname):
            raise RuntimeError("Can't decide whether to use %s or %s for %s: "
                               "both full names have length %d" %
                               (fullname, other_fullname, name, len(fullname)))
          if len(fullname) > len(other_fullname):
            continue  # Use the shorter full name
        members[name] = fullname, member
  return members


def _get_anchor(module_to_name, fullname):
  """Turn a full member name into an anchor.

  Args:
    module_to_name: Dictionary mapping modules to short names.
    fullname: Fully qualified name of symbol.

  Returns:
    HTML anchor string.  The longest module name prefix of fullname is
    removed to make the anchor.

  Raises:
    ValueError: If fullname uses characters invalid in an anchor.
  """
  if not _anchor_re.match(fullname):
    raise ValueError("'%s' is not a valid anchor" % fullname)
  anchor = fullname
  for module_name in module_to_name.values():
    if fullname.startswith(module_name + "."):
      rest = fullname[len(module_name)+1:]
      # Use this prefix iff it is longer than any found before
      if len(anchor) > len(rest):
        anchor = rest
  return anchor


class Library(Document):
  """An automatically generated document for a set of functions and classes."""

  def __init__(self,
               title,
               module,
               module_to_name,
               members,
               documented,
               exclude_symbols=(),
               prefix=None):
    """Creates a new Library.

    Args:
      title: A human-readable title for the library.
      module: Module to pull high level docstring from (for table of contents,
        list of Ops to document, etc.).
      module_to_name: Dictionary mapping modules to short names.
      members: Dictionary mapping member name to (fullname, member).
      documented: Set of documented names to update.
      exclude_symbols: A list of specific symbols to exclude.
      prefix: A string to include at the beginning of the page.
    """
    self._title = title
    self._module = module
    self._module_to_name = module_to_name
    self._members = dict(members)  # Copy since we mutate it below
    self._exclude_symbols = frozenset(exclude_symbols)
    documented.update(exclude_symbols)
    self._documented = documented
    self._mentioned = set()
    self._prefix = prefix or ""

  @property
  def title(self):
    """The human-readable title for this library."""
    return self._title

  @property
  def mentioned(self):
    """Set of names mentioned in this library."""
    return self._mentioned

  @property
  def exclude_symbols(self):
    """Set of excluded symbols."""
    return self._exclude_symbols

  def _should_include_member(self, name):
    """Returns True if this member should be included in the document."""
    # Always exclude symbols matching _always_drop_symbol_re.
    if _always_drop_symbol_re.match(name):
      return False
    # Finally, exclude any specifically-excluded symbols.
    if name in self._exclude_symbols:
      return False
    return True

  def get_imported_modules(self, module):
    """Returns the list of modules imported from `module`."""
    for name, member in inspect.getmembers(module):
      if inspect.ismodule(member):
        yield name, member

  def get_class_members(self, cls_name, cls):
    """Returns the list of class members to document in `cls`.

    This function filters the class member to ONLY return those
    defined by the class.  It drops the inherited ones.

    Args:
      cls_name: Qualified name of `cls`.
      cls: An inspect object of type 'class'.

    Yields:
      name, member tuples.
    """
    for name, member in inspect.getmembers(cls):
      # Only show methods and properties presently.  In Python 3,
      # methods register as isfunction.
      is_method = inspect.ismethod(member) or inspect.isfunction(member)
      if not (is_method or isinstance(member, property)):
        continue
      if ((is_method and member.__name__ == "__init__")
          or self._should_include_member(name)):
        yield name, ("%s.%s" % (cls_name, name), member)

  def set_functions_and_classes_dir(self, dirname):
    """Sets the name of the directory for function and class markdown files.

    Args:
      dirname: string. The name of the directory in which to store function
        and class markdown files.
    """
    self.functions_and_classes_dir = dirname

  def _generate_signature_for_function(self, func):
    """Given a function, returns a string representing its args."""
    args_list = []
    argspec = inspect.getargspec(func)
    first_arg_with_default = (
        len(argspec.args or []) - len(argspec.defaults or []))
    for arg in argspec.args[:first_arg_with_default]:
      if arg == "self":
        # Python documentation typically skips `self` when printing method
        # signatures.
        continue
      args_list.append(arg)

    # TODO(mrry): This is a workaround for documenting signature of
    # functions that have the @contextlib.contextmanager decorator.
    # We should do something better.
    if argspec.varargs == "args" and argspec.keywords == "kwds":
      original_func = func.__closure__[0].cell_contents
      return self._generate_signature_for_function(original_func)

    if argspec.defaults:
      for arg, default in zip(
          argspec.args[first_arg_with_default:], argspec.defaults):
        if callable(default):
          args_list.append("%s=%s" % (arg, default.__name__))
        else:
          args_list.append("%s=%r" % (arg, default))
    if argspec.varargs:
      args_list.append("*" + argspec.varargs)
    if argspec.keywords:
      args_list.append("**" + argspec.keywords)
    return "(" + ", ".join(args_list) + ")"

  def _remove_docstring_indent(self, docstring):
    """Remove indenting.

    We follow Python's convention and remove the minimum indent of the lines
    after the first, see:
    https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation
    preserving relative indentation.

    Args:
      docstring: A docstring.

    Returns:
      A list of strings, one per line, with the minimum indent stripped.
    """
    docstring = docstring or ""
    lines = docstring.strip().split("\n")

    min_indent = len(docstring)
    for l in lines[1:]:
      l = l.rstrip()
      if l:
        i = 0
        while i < len(l) and l[i] == " ":
          i += 1
        if i < min_indent: min_indent = i
    for i in range(1, len(lines)):
      l = lines[i].rstrip()
      if len(l) >= min_indent:
        l = l[min_indent:]
      lines[i] = l
    return lines

  def _print_formatted_docstring(self, docstring, f):
    """Formats the given `docstring` as Markdown and prints it to `f`."""
    lines = self._remove_docstring_indent(docstring)

    # Output the lines, identifying "Args" and other section blocks.
    i = 0

    def _at_start_of_section():
      """Returns the header if lines[i] is at start of a docstring section."""
      l = lines[i]
      match = _section_re.match(l)
      if match and i + 1 < len(
          lines) and lines[i + 1].startswith(" "):
        return match.group(1)
      else:
        return None

    while i < len(lines):
      l = lines[i]

      section_header = _at_start_of_section()
      if section_header:
        if i == 0 or lines[i-1]:
          print("", file=f)
        # Use at least H4 to keep these out of the TOC.
        print("##### " + section_header + ":", file=f)
        print("", file=f)
        i += 1
        outputting_list = False
        while i < len(lines):
          l = lines[i]
          # A new section header terminates the section.
          if _at_start_of_section():
            break
          match = _arg_re.match(l)
          if match:
            if not outputting_list:
              # We need to start a list. In Markdown, a blank line needs to
              # precede a list.
              print("", file=f)
              outputting_list = True
            suffix = l[len(match.group()):].lstrip()
            print("*  <b>`" + match.group(1) + "`</b>: " + suffix, file=f)
          else:
            # For lines that don't start with _arg_re, continue the list if it
            # has enough indentation.
            outputting_list &= l.startswith("   ")
            print(l, file=f)
          i += 1
      else:
        print(l, file=f)
        i += 1

  def _print_function(self, f, prefix, fullname, func):
    """Prints the given function to `f`."""
    heading = prefix + " `" + fullname
    if not isinstance(func, property):
      heading += self._generate_signature_for_function(func)
    heading += "` {#%s}" % _get_anchor(self._module_to_name, fullname)
    print(heading, file=f)
    print("", file=f)
    self._print_formatted_docstring(inspect.getdoc(func), f)
    print("", file=f)

  def _write_member_markdown_to_file(self, f, prefix, name, member):
    """Print `member` to `f`."""
    if (inspect.isfunction(member) or inspect.ismethod(member) or
        isinstance(member, property)):
      print("- - -", file=f)
      print("", file=f)
      self._print_function(f, prefix, name, member)
      print("", file=f)

      # Write an individual file for each function.
      if inspect.isfunction(member):
        indivf = open(
            os.path.join(self.functions_and_classes_dir, name + ".md"), "w+")
        self._print_function(indivf, prefix, name, member)
    elif inspect.isclass(member):
      print("- - -", file=f)
      print("", file=f)
      print("%s `class %s` {#%s}" % (prefix, name,
                                     _get_anchor(self._module_to_name, name)),
            file=f)
      print("", file=f)
      self._write_class_markdown_to_file(f, name, member)
      print("", file=f)

      # Write an individual file for each class.
      indivf = open(
          os.path.join(self.functions_and_classes_dir, name + ".md"), "w+")
      self._write_class_markdown_to_file(indivf, name, member)
    else:
      raise RuntimeError("Member %s has unknown type %s" % (name, type(member)))

  def _write_docstring_markdown_to_file(self, f, prefix, docstring, members,
                                        imports):
    for l in self._remove_docstring_indent(docstring):
      if l.startswith(_member_mark):
        name = l[len(_member_mark):].strip(" \t")
        if name in members:
          self._documented.add(name)
          self._mentioned.add(name)
          self._write_member_markdown_to_file(f, prefix, *members[name])
          del members[name]
        elif name in imports:
          self._write_module_markdown_to_file(f, imports[name])
        else:
          raise ValueError("%s: unknown member `%s`, markdown=`%s`." % (
              self._title, name, l))
      else:
        print(l, file=f)

  def _write_class_markdown_to_file(self, f, name, cls):
    """Write the class doc to `f`.

    Args:
      f: File to write to.
      name: name to use.
      cls: class object.
    """
    # Build the list of class methods to document.
    methods = dict(self.get_class_members(name, cls))
    # Used later to check if any methods were called out in the class
    # docstring.
    num_methods = len(methods)
    try:
      self._write_docstring_markdown_to_file(f, "####", inspect.getdoc(cls),
                                             methods, {})
    except ValueError as e:
      raise ValueError(str(e) + " in class `%s`" % cls.__name__)

    # If some methods were not described, describe them now if they are
    # defined by the class itself (not inherited).  If NO methods were
    # described, describe all methods.
    #
    # TODO(touts): when all methods have been categorized make it an error
    # if some methods are not categorized.
    any_method_called_out = (len(methods) != num_methods)
    if any_method_called_out:
      other_methods = {n: m for n, m in methods.items() if n in cls.__dict__}
      if other_methods:
        print("\n#### Other Methods", file=f)
    else:
      other_methods = methods
    for name in sorted(other_methods):
      self._write_member_markdown_to_file(f, "####", *other_methods[name])

  def _write_module_markdown_to_file(self, f, module):
    imports = dict(self.get_imported_modules(module))
    self._write_docstring_markdown_to_file(f, "###", inspect.getdoc(module),
                                           self._members, imports)

  def write_markdown_to_file(self, f):
    """Prints this library to file `f`.

    Args:
      f: File to write to.

    Returns:
      Dictionary of documented members.
    """
    print("<!-- This file is machine generated: DO NOT EDIT! -->", file=f)
    print("", file=f)
    # TODO(touts): Do not insert these.  Let the doc writer put them in
    # the module docstring explicitly.
    print("#", self._title, file=f)
    if self._prefix:
      print(self._prefix, file=f)
    print("[TOC]", file=f)
    print("", file=f)
    if self._module is not None:
      self._write_module_markdown_to_file(f, self._module)

  def write_other_members(self, f, catch_all=False):
    """Writes the leftover members to `f`.

    Args:
      f: File to write to.
      catch_all: If true, document all missing symbols from any module.
        Otherwise, document missing symbols from just this module.
    """
    if catch_all:
      names = self._members.items()
    else:
      names = inspect.getmembers(self._module)
      all_names = getattr(self._module, "__all__", None)
      if all_names is not None:
        names = [(n, m) for n, m in names if n in all_names]
    leftovers = []
    for name, _ in names:
      if name in self._members and name not in self._documented:
        leftovers.append(name)
    if leftovers:
      print("%s: undocumented members: %d" % (self._title, len(leftovers)))
      print("\n## Other Functions and Classes", file=f)
      for name in sorted(leftovers):
        print("  %s" % name)
        self._documented.add(name)
        self._mentioned.add(name)
        self._write_member_markdown_to_file(f, "###", *self._members[name])

  def assert_no_leftovers(self):
    """Generate an error if there are leftover members."""
    leftovers = []
    for name in self._members:
      if name in self._members and name not in self._documented:
        leftovers.append(name)
    if leftovers:
      raise RuntimeError("%s: undocumented members: %s" %
                         (self._title, ", ".join(leftovers)))


def write_libraries(output_dir, libraries):
  """Write a list of libraries to disk.

  Args:
    output_dir: Output directory.
    libraries: List of (filename, library) pairs.
  """
  files = [open(os.path.join(output_dir, k), "w") for k, _ in libraries]

  # Set the directory in which to save individual class and function md files,
  # creating it if it doesn't exist.
  indiv_dir = os.path.join(output_dir, _indiv_dir)
  if not os.path.exists(indiv_dir):
    os.makedirs(indiv_dir)

  # Document mentioned symbols for all libraries
  for f, (_, v) in zip(files, libraries):
    v.set_functions_and_classes_dir(indiv_dir)
    v.write_markdown_to_file(f)
  # Document symbols that no library mentioned.  We do this after writing
  # out all libraries so that earlier libraries know what later libraries
  # documented.
  for f, (_, v) in zip(files, libraries):
    v.write_other_members(f)
    f.close()
