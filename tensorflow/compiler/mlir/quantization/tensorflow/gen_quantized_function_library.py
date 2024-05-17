# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Generates the quantized function library contained header file."""

import ast
import re
import string
from typing import Sequence

from absl import app
from absl import flags

# TODO(b/263048929): Create a test for gen_quantized_function_library.

_OUTPUT_FILE = flags.DEFINE_string('output_file', None, 'output file location')
_SRCS = flags.DEFINE_string('src', None, 'source file locations')
_NAMESPACE = flags.DEFINE_string('namespace', 'mlir::quant',
                                 'namespace in the generated file')

flags.mark_flags_as_required(['output_file', 'src'])


def _substitute_for_loop_template(module: str) -> str:
  """Substitutes the for loop templates in the given module."""
  compiled_regex = re.compile(
      r'^\s*for\s(.*?)\sin\s(\[.*?\])\s\{(.*?)\}\s//\send\sfor\n',
      re.MULTILINE | re.DOTALL)
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    try:
      arg_name = func_match.group(1)
      arg_values = ast.literal_eval(func_match.group(2))
      loop_template = string.Template(func_match.group(3))
    except Exception as e:  # pylint: disable=broad-except
      raise ValueError('The loop template is in wrong format') from e

    replacement_text = ''
    for arg_value in arg_values:
      arg_dict = {arg_name: arg_value}
      replacement_text += '\\n'
      replacement_text += _substitute_parameterization_template(
          loop_template.safe_substitute(arg_dict))
    module = re.sub(compiled_regex, replacement_text, module, count=1)

  return module


def _substitute_parameterization_template(module: str) -> str:
  """Substitutes all the function templates in the given module."""
  compiled_regex = re.compile(
      r'^\s*parameters(\[.*?\])\n?(^\s*(?:func\.)+func.*?\{.*?(?:func\.)+return.*?\}\n)',
      re.MULTILINE | re.DOTALL)
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    try:
      value_list = ast.literal_eval(func_match.group(1))
      # Escapes template $-based substitutions for attributes containing $.
      # $$ is replaced with a single $.
      func_template = string.Template(
          func_match.group(2).replace('tfdtype$DT_', 'tfdtype$$DT_'))
    except Exception as e:  # pylint: disable=broad-except
      raise ValueError('The function template is in wrong format') from e

    replacement_text = ''
    for value_dict in value_list:
      for key, value in value_dict.items():
        # Replace single quote to double quote since single quote around a
        # string are not valid in the MLIR representation.
        value_dict[key] = str(value).replace("'", '"')
      replacement_text += '\\n'
      replacement_text += func_template.substitute(value_dict)
    module = re.sub(compiled_regex, replacement_text, module, count=1)
  return module


def _format_snake_case_op_name(s):
  """Formats the op name to snake case."""
  s = s.replace('2D', '2d').replace('3D', '3d')
  snake_case = ''.join(['_' + i.lower() if i.isupper() else i for i in s
                       ]).lstrip('_')
  return snake_case.replace('mat_mul', 'matmul').replace('bias_add', 'bias')


def _substitute_impl_function_name_template(module: str) -> str:
  """Generates the op-specific implementation function name."""
  compiled_regex = re.compile(r'GenerateImplFunctionName\(([\w\s]+)\)')
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    text = func_match.group(1)
    function_name = 'internal_{}_fn'.format(_format_snake_case_op_name(text))
    module = re.sub(compiled_regex, function_name, module, count=1)
  return module


def _substitute_quantized_function_name_template(module: str) -> str:
  """Generates the quantized function name."""
  compiled_regex = re.compile(
      r'GenerateQuantizedFunctionName(\([\w\s\'\"\[\],]+\))')
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    # Make sure the string ends with ",)" so the parsed value is a tuple.
    argument_string = func_match.group(1)
    if not argument_string.endswith(',)'):
      argument_string = argument_string[:-1] + ',)'
    arguments = ast.literal_eval(argument_string)

    if len(arguments) < 1 or len(arguments) > 2:
      raise ValueError(
          'Wrong number of arguments to GenerateQuantizedFunctionName')

    quantized_ops = arguments[0]
    if not quantized_ops:
      raise ValueError('The quantized_ops list must not be empty')

    # Add op names to the function name.
    function_name = 'quantized_{}'.format(
        _format_snake_case_op_name(quantized_ops[0]))
    if len(quantized_ops) > 1:
      function_name += '_with_{}'.format(
          _format_snake_case_op_name(quantized_ops[1]))
    if len(quantized_ops) > 1:
      for quantized_op in quantized_ops[2:]:
        function_name += '_and_{}'.format(
            _format_snake_case_op_name(quantized_op))

    # Add suffix based on output type.
    suffix = '_fn'
    if len(arguments) > 1 and arguments[1] == 'f32':
      suffix = '_float_output_fn'
    function_name += suffix

    module = re.sub(compiled_regex, function_name, module, count=1)
  return module


def main(_: Sequence[str]) -> None:
  namespaces = _NAMESPACE.value.split('::')
  src_files = _SRCS.value.split(' ')
  file_prefix = 'quantized_function_library'
  module_prefix = 'kQuantizedFunctionLibraryInMLIR'

  modules = []

  for src_file in src_files:
    with open(src_file, 'r') as f:
      content = f.read()

      # Skip the copyright in the source file.
      module_match = re.search(r'(^module\s\{)(.*)(^\})', content,
                               re.MULTILINE | re.DOTALL)
      if module_match is None:
        raise ValueError("Couldn't find module in the function library")
      module = module_match.group()

      # Substitute all the function templates.
      out = re.split(file_prefix, src_file)
      if len(out) != 2:
        raise ValueError('The file name must start with {}'.format(file_prefix))
      tag = out[1][:-5]  # the last five values = ".mlir"
      module = _substitute_for_loop_template(module)
      module = _substitute_parameterization_template(module)
      module = _substitute_quantized_function_name_template(module)
      module = _substitute_impl_function_name_template(module)
      modules.append((tag, module))

  with open(_OUTPUT_FILE.value, 'w') as f:
    f.write("""/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_
""")

    for namespace in namespaces:
      f.write('namespace {0} {{\n'.format(namespace))

    for tag, module in modules:
      f.write('constexpr char {0}[] ='.format(module_prefix + tag.upper()))

      for line in module.splitlines():
        f.write('\n  "')
        f.write(line.rstrip().replace('"', r'\"'))
        f.write('\\n"')

      f.write(';\n')

    for namespace in reversed(namespaces):
      f.write('}}  // namespace {0}\n'.format(namespace))

    f.write(
        '#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_'
    )


if __name__ == '__main__':
  app.run(main)
