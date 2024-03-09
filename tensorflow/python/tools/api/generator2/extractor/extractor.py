# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Extracts API information for a set of Python sources."""

import ast
from collections.abc import Sequence
import re
from typing import Any, Optional, Union, cast

from absl import flags
from absl import logging

from tensorflow.python.tools.api.generator2.shared import exported_api

_OUTPUT = flags.DEFINE_string('output', '', 'File to output contents to.')
_DECORATOR = flags.DEFINE_string(
    'decorator',
    '',
    'Full path to Python decorator function used for exporting API.',
)
_API_NAME = flags.DEFINE_string(
    'api_name',
    '',
    'Prefix for all exported symbols and docstrings.',
)

_DOCSTRING_PATTERN: re.Pattern[str] = re.compile(
    r'\s*API\s+docstring:\s*([\w.]+)\s*'
)


class BadExportError(Exception):
  """Exception for bad exports."""


class Parser(ast.NodeVisitor):
  """Parser for Python source files that extracts TF API exports."""

  _exports: exported_api.ExportedApi
  _decorator_package: str
  _decorator_symbol: str
  _api_name: str
  _current_file: Optional[str] = None
  _current_file_decorators: set[str]

  def __init__(
      self,
      exports: exported_api.ExportedApi,
      decorator: str,
      api_name: str,
  ):
    self._exports = exports
    self._decorator_package, self._decorator_symbol = decorator.rsplit('.', 1)
    self._api_name = api_name

  def process_file(self, filename: str) -> None:
    """Finds exported APIs in filename."""
    try:
      with open(filename, mode='r', encoding='utf-8') as f:
        contents = f.read()
    except Exception as e:  # pylint: disable=broad-exception-caught
      # log and ignore exceptions from read
      logging.exception('Error reading %s: %s', filename, e)
    else:
      self.process(filename, contents)

  def process(self, filename: str, contents: str) -> None:
    """Finds exported APIs in contents."""
    self._current_file_decorators = set()
    self._current_file = filename
    try:
      parsed = ast.parse(contents, filename=filename)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # logging errors when parsing file
      logging.exception('Error parsing %s: %s', filename, e)
    else:
      self.visit(parsed)
    finally:
      self._current_file = None
      self._current_file_decorators = set()

  def visit_Module(self, node: ast.Module) -> None:  # pylint: disable=invalid-name
    for stmt in node.body:
      self._process_stmt(stmt)

  def _process_stmt(self, node: ast.stmt) -> None:
    """Process top-level statement for exported apis."""
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
      self._process_def(node)
    elif isinstance(node, ast.Assign):
      self._process_assign(node)
    elif isinstance(node, ast.Expr):
      self._process_expr(node)
    else:
      self.visit(node)

  def visit_Import(self, node: ast.Import) -> None:  # pylint: disable=invalid-name
    """Identifies imports of decorator."""
    for name in node.names:
      if name.name == self._decorator_package:
        if name.asname:
          # import <package> as <name>
          self._current_file_decorators.add(
              name.asname + '.' + self._decorator_symbol
          )
        else:
          # import <package>
          _, module = self._decorator_package.rsplit('.', 1)
          self._current_file_decorators.add(
              module + '.' + self._decorator_symbol
          )
    self.generic_visit(node)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # pylint: disable=invalid-name
    """Identifies imports of decorator."""
    if node.module == self._decorator_package:
      for name in node.names:
        if name.name == self._decorator_symbol:
          if name.asname:
            # from <package> import <symbol> as <name>
            self._current_file_decorators.add(name.asname)
          else:
            # from <package> import <symbol>
            self._current_file_decorators.add(name.name)
    else:
      parent, module = self._decorator_package.rsplit('.', 1)
      if node.module == parent:
        for name in node.names:
          if name.name == module:
            if name.asname:
              # from <parent> import <module> as <name>
              self._current_file_decorators.add(
                  name.asname + '.' + self._decorator_symbol
              )
            else:
              # from <parent> import <module>
              self._current_file_decorators.add(
                  name.name + '.' + self._decorator_symbol
              )
    self.generic_visit(node)

  def _process_def(self, node: Union[ast.ClassDef, ast.FunctionDef]) -> None:
    """Process top-level [Class|Function]Def for potential symbol export."""
    # @tf_export(...)
    # [class|def] <id>:
    for decorator in node.decorator_list:
      if self._is_export_call(decorator):
        self._add_exported_symbol(cast(ast.Call, decorator), node.name)
      else:
        self.visit(decorator)

    if isinstance(node, ast.ClassDef):
      for base in node.bases:
        self.visit(base)
      for kw in node.keywords:
        self.visit(kw)
    elif isinstance(node, ast.FunctionDef):
      self.visit(node.args)
      if node.returns:
        self.visit(node.returns)

    for stmt in node.body:
      self.visit(stmt)

  def _process_assign(self, node: ast.Assign) -> None:
    """Process top-level assign for potential symbol export."""
    if isinstance(node.value, ast.Call) and self._is_export_call(
        node.value.func
    ):
      # id = tf_export(...)(...)
      if len(node.targets) != 1:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' assigned to a single value: {ast.dump(node)}'
        )
      symbol = self._name(node.targets[0])
      if not symbol:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' assigned to a single value: {ast.dump(node)}'
        )
      self._add_exported_symbol(node.value.func, symbol)
    else:
      self.visit(node)

  def _process_expr(self, node: ast.Expr) -> None:
    """Process top-level expression for potential symbol export."""
    if isinstance(node.value, ast.Call):
      self._process_call(node.value)
    elif isinstance(node.value, ast.Constant):
      self._process_constant(node.value)
    else:
      self.visit(node)

  def _process_call(self, node: ast.Call) -> None:
    """Process top-level call for potential symbol export."""
    func = node.func
    if self._is_export_call(func):
      func = cast(ast.Call, func)
      # tf_export(...)(id)
      if len(node.args) != 1 or node.keywords:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' called with a single value: {ast.dump(node)}'
        )
      symbol = self._name(self._unwrap_simple_call(node.args[0]))
      if not symbol:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export must be'
            f' called with a single value: {ast.dump(node)}'
        )
      self._add_exported_symbol(func, symbol)
    elif (
        isinstance(func, ast.Attribute)
        and func.attr == 'export_constant'
        and self._is_export_call(func.value)
    ):
      # tf_export(...).export_constant(__name__, id)
      if (
          len(node.args) != 2
          or node.keywords
          or self._name(node.args[0]) != '__name__'
      ):
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export_constant must be'
            f' called with __name__, <id>: {ast.dump(node)}'
        )
      self._add_exported_symbol(func.value, self._literal_value(node.args[1]))
    else:
      self.visit(node)

  def _process_constant(self, node: ast.Constant) -> None:
    """Process top-level constant for a potential API docstring export."""
    if isinstance(node.value, str):
      docstring, modules = self._extract_docstring(node.value)
      if modules:
        self._exports.add_doc(
            exported_api.ExportedDoc.create(
                file_name=self._current_file,
                line_no=node.lineno,
                modules=modules,
                docstring=docstring,
            )
        )
      else:
        self.visit(node)

  def _extract_docstring(self, value: str) -> tuple[str, Sequence[str]]:
    """Extract docstring and list of modules that it should be applied to."""
    docstring = ''
    modules = []
    for line in value.splitlines():
      match = _DOCSTRING_PATTERN.match(line)
      if match:
        module = match.group(1).strip()
        # API docstring: <module>
        if module == self._api_name or module.startswith(self._api_name + '.'):
          modules.append(module)
      else:
        docstring += line + '\n'
    return (docstring.strip(), modules)

  def visit_Call(self, node: ast.Call) -> None:  # pylint: disable=invalid-name
    if self._is_export_call(node):
      raise BadExportError(
          f'{self._current_file}:{node.lineno} export must be'
          f' used at top level of file: {ast.dump(node)}'
      )
    self.generic_visit(node)

  def visit_Constant(self, node: ast.Constant) -> None:
    if isinstance(node.value, str):
      _, modules = self._extract_docstring(node.value)
      if modules:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} API docstrings must be'
            f' at top level of file: {ast.dump(node)}'
        )
    self.generic_visit(node)

  def _is_export_call(self, node: ast.expr) -> bool:  # TypeGuard[ast.Call]
    return (
        isinstance(node, ast.Call)
        and self._name(node.func) in self._current_file_decorators
    )

  def _name(self, node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
      return node.id
    if isinstance(node, ast.Attribute):
      parent = self._name(node.value)
      if parent:
        return f'{parent}.{node.attr}'

  def _unwrap_simple_call(self, node: ast.expr) -> ast.expr:
    """Unwraps a function call that takes a single unnamed parameter."""
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
      return self._unwrap_simple_call(node.args[0])
    return node

  def _literal_value(self, node: ast.expr) -> Any:
    try:
      return ast.literal_eval(node)
    except Exception as e:
      raise BadExportError(
          f'{self._current_file}:{node.lineno} all arguments to'
          f' export must be literal values: {ast.dump(node)}'
      ) from e

  def _add_exported_symbol(self, node: ast.Call, symbol_name: str) -> None:
    """Adds an exported symbol represented by the given call."""
    if symbol_name.find('.') != -1:
      raise BadExportError(
          f'{self._current_file}:{node.lineno} export called with symbol'
          f' {symbol_name} not defined in current file: {ast.dump(node)}'
      )
    v2_apis = tuple(
        f'{self._api_name}.{self._literal_value(arg)}' for arg in node.args
    )
    v1_apis = v2_apis
    for kw in node.keywords:
      if kw.arg == 'v1':
        v1_apis = tuple(
            f'{self._api_name}.{v}' for v in self._literal_value(kw.value)
        )
      elif kw.arg == 'allow_multiple_exports':
        # no-op kept for backward comapatibility of `tf-keras` with TF 2.13
        pass
      else:
        raise BadExportError(
            f'{self._current_file}:{node.lineno} export called'
            f' with unknown argument {kw.arg}: {ast.dump(node)}'
        )
    self._exports.add_symbol(
        exported_api.ExportedSymbol.create(
            file_name=self._current_file,
            line_no=node.lineno,
            symbol_name=symbol_name,
            v2_apis=v2_apis,
            v1_apis=v1_apis,
        )
    )


def main(argv: Sequence[str]) -> None:
  exporter = exported_api.ExportedApi()
  p = Parser(exporter, _DECORATOR.value, _API_NAME.value)
  for arg in argv[1:]:
    p.process_file(arg)

  exporter.write(_OUTPUT.value)
