#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 The MLIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for updating SPIR-V dialect by scraping information from SPIR-V
# HTML and JSON specs from the Internet.
#
# For example, to define the enum attribute for SPIR-V memory model:
#
# ./gen_spirv_dialect.py --base_td_path /path/to/SPIRVBase.td \
#                        --new-enum MemoryModel
#
# The 'operand_kinds' dict of spirv.core.grammar.json contains all supported
# SPIR-V enum classes.

import re
import requests
import textwrap

SPIRV_HTML_SPEC_URL = 'https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html'
SPIRV_JSON_SPEC_URL = 'https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/spirv.core.grammar.json'

AUTOGEN_OP_DEF_SEPARATOR = '\n// -----\n\n'
AUTOGEN_ENUM_SECTION_MARKER = 'enum section. Generated from SPIR-V spec; DO NOT MODIFY!'
AUTOGEN_OPCODE_SECTION_MARKER = (
    'opcode section. Generated from SPIR-V spec; DO NOT MODIFY!')


def get_spirv_doc_from_html_spec():
  """Extracts instruction documentation from SPIR-V HTML spec.

  Returns:
    - A dict mapping from instruction opcode to documentation.
  """
  response = requests.get(SPIRV_HTML_SPEC_URL)
  spec = response.content

  from bs4 import BeautifulSoup
  spirv = BeautifulSoup(spec, 'html.parser')

  section_anchor = spirv.find('h3', {'id': '_a_id_instructions_a_instructions'})

  doc = {}

  for section in section_anchor.parent.find_all('div', {'class': 'sect3'}):
    for table in section.find_all('table'):
      inst_html = table.tbody.tr.td.p
      opname = inst_html.a['id']
      # Ignore the first line, which is just the opname.
      doc[opname] = inst_html.text.split('\n', 1)[1].strip()

  return doc


def get_spirv_grammar_from_json_spec():
  """Extracts operand kind and instruction grammar from SPIR-V JSON spec.

  Returns:
    - A list containing all operand kinds' grammar
    - A list containing all instructions' grammar
  """
  response = requests.get(SPIRV_JSON_SPEC_URL)
  spec = response.content

  import json
  spirv = json.loads(spec)

  return spirv['operand_kinds'], spirv['instructions']


def split_list_into_sublists(items, offset):
  """Split the list of items into multiple sublists.

  This is to make sure the string composed from each sublist won't exceed
  80 characters.

  Arguments:
    - items: a list of strings
    - offset: the offset in calculating each sublist's length
  """
  chuncks = []
  chunk = []
  chunk_len = 0

  for item in items:
    chunk_len += len(item) + 2
    if chunk_len > 80:
      chuncks.append(chunk)
      chunk = []
      chunk_len = len(item) + 2
    chunk.append(item)

  if len(chunk) != 0:
    chuncks.append(chunk)

  return chuncks


def gen_operand_kind_enum_attr(operand_kind):
  """Generates the TableGen I32EnumAttr definition for the given operand kind.

  Returns:
    - The operand kind's name
    - A string containing the TableGen I32EnumAttr definition
  """
  if 'enumerants' not in operand_kind:
    return '', ''

  kind_name = operand_kind['kind']
  kind_acronym = ''.join([c for c in kind_name if c >= 'A' and c <= 'Z'])
  kind_cases = [(case['enumerant'], case['value'])
                for case in operand_kind['enumerants']]
  max_len = max([len(symbol) for (symbol, _) in kind_cases])

  # Generate the definition for each enum case
  fmt_str = 'def SPV_{acronym}_{symbol} {colon:>{offset}} '\
            'I32EnumAttrCase<"{symbol}", {value}>;'
  case_defs = [
      fmt_str.format(
          acronym=kind_acronym,
          symbol=case[0],
          value=case[1],
          colon=':',
          offset=(max_len + 1 - len(case[0]))) for case in kind_cases
  ]
  case_defs = '\n'.join(case_defs)

  # Generate the list of enum case names
  fmt_str = 'SPV_{acronym}_{symbol}';
  case_names = [fmt_str.format(acronym=kind_acronym,symbol=case[0])
                for case in kind_cases]

  # Split them into sublists and concatenate into multiple lines
  case_names = split_list_into_sublists(case_names, 6)
  case_names = ['{:6}'.format('') + ', '.join(sublist)
                for sublist in case_names]
  case_names = ',\n'.join(case_names)

  # Generate the enum attribute definition
  enum_attr = 'def SPV_{name}Attr :\n    '\
      'I32EnumAttr<"{name}", "valid SPIR-V {name}", [\n{cases}\n    ]> {{\n'\
      '  let returnType = "::mlir::spirv::{name}";\n'\
      '  let convertFromStorage = '\
            '"static_cast<::mlir::spirv::{name}>($_self.getInt())";\n'\
      '  let cppNamespace = "::mlir::spirv";\n}}'.format(
          name=kind_name, cases=case_names)
  return kind_name, case_defs + '\n\n' + enum_attr


def gen_opcode(instructions):
  """ Generates the TableGen definition to map opname to opcode

  Returns:
    - A string containing the TableGen SPV_OpCode definition
  """

  max_len = max([len(inst['opname']) for inst in instructions])
  def_fmt_str = 'def SPV_OC_{name} {colon:>{offset}} '\
            'I32EnumAttrCase<"{name}", {value}>;'
  opcode_defs = [
      def_fmt_str.format(
          name=inst['opname'],
          value=inst['opcode'],
          colon=':',
          offset=(max_len + 1 - len(inst['opname']))) for inst in instructions
  ]
  opcode_str = '\n'.join(opcode_defs)

  decl_fmt_str = 'SPV_OC_{name}'
  opcode_list = [
      decl_fmt_str.format(name=inst['opname']) for inst in instructions
  ]
  opcode_list = split_list_into_sublists(opcode_list, 6)
  opcode_list = [
      '{:6}'.format('') + ', '.join(sublist) for sublist in opcode_list
  ]
  opcode_list = ',\n'.join(opcode_list)
  enum_attr = 'def SPV_OpcodeAttr :\n'\
              '    I32EnumAttr<"{name}", "valid SPIR-V instructions", [\n'\
              '{lst}\n'\
              '      ]> {{\n'\
              '    let returnType = "::mlir::spirv::{name}";\n'\
              '    let convertFromStorage = '\
              '"static_cast<::mlir::spirv::{name}>($_self.getInt())";\n'\
              '    let cppNamespace = "::mlir::spirv";\n}}'.format(
                  name='Opcode', lst=opcode_list)
  return opcode_str + '\n\n' + enum_attr


def update_td_opcodes(path, instructions, filter_list):
  """Updates SPIRBase.td with new generated opcode cases.

  Arguments:
    - path: the path to SPIRBase.td
    - instructions: a list containing all SPIR-V instructions' grammar
    - filter_list: a list containing new opnames to add
  """

  with open(path, 'r') as f:
    content = f.read()

  content = content.split(AUTOGEN_OPCODE_SECTION_MARKER)
  assert len(content) == 3

  # Extend opcode list with existing list
  existing_opcodes = [k[11:] for k in re.findall('def SPV_OC_\w+', content[1])]
  filter_list.extend(existing_opcodes)
  filter_list = list(set(filter_list))

  # Generate the opcode for all instructions in SPIR-V
  filter_instrs = list(
      filter(lambda inst: (inst['opname'] in filter_list), instructions))
  # Sort instruction based on opcode
  filter_instrs.sort(key=lambda inst: inst['opcode'])
  opcode = gen_opcode(filter_instrs)

  # Substitute the opcode
  content = content[0] + AUTOGEN_OPCODE_SECTION_MARKER + '\n\n' + \
        opcode + '\n\n// End ' + AUTOGEN_OPCODE_SECTION_MARKER \
        + content[2]

  with open(path, 'w') as f:
    f.write(content)


def update_td_enum_attrs(path, operand_kinds, filter_list):
  """Updates SPIRBase.td with new generated enum definitions.

  Arguments:
    - path: the path to SPIRBase.td
    - operand_kinds: a list containing all operand kinds' grammar
    - filter_list: a list containing new enums to add
  """
  with open(path, 'r') as f:
    content = f.read()

  content = content.split(AUTOGEN_ENUM_SECTION_MARKER)
  assert len(content) == 3

  # Extend filter list with existing enum definitions
  existing_kinds = [
      k[8:-4] for k in re.findall('def SPV_\w+Attr', content[1])]
  filter_list.extend(existing_kinds)

  # Generate definitions for all enums in filter list
  defs = [gen_operand_kind_enum_attr(kind)
          for kind in operand_kinds if kind['kind'] in filter_list]
  # Sort alphabetically according to enum name
  defs.sort(key=lambda enum : enum[0])
  # Only keep the definitions from now on
  defs = [enum[1] for enum in defs]

  # Substitute the old section
  content = content[0] + AUTOGEN_ENUM_SECTION_MARKER + '\n\n' + \
      '\n\n'.join(defs) + "\n\n// End " + AUTOGEN_ENUM_SECTION_MARKER  \
      + content[2];

  with open(path, 'w') as f:
    f.write(content)


def snake_casify(name):
  """Turns the given name to follow snake_case convension."""
  name = re.sub('\W+', '', name).split()
  name = [s.lower() for s in name]
  return '_'.join(name)


def map_spec_operand_to_ods_argument(operand):
  """Maps a operand in SPIR-V JSON spec to an op argument in ODS.

  Arguments:
    - A dict containing the operand's kind, quantifier, and name

  Returns:
    - A string containing both the type and name for the argument
  """
  kind = operand['kind']
  quantifier = operand.get('quantifier', '')

  # These instruction "operands" are for encoding the results; they should
  # not be handled here.
  assert kind != 'IdResultType', 'unexpected to handle "IdResultType" kind'
  assert kind != 'IdResult', 'unexpected to handle "IdResult" kind'

  if kind == 'IdRef':
    if quantifier == '':
      arg_type = 'SPV_Type'
    elif quantifier == '?':
      arg_type = 'SPV_Optional<SPV_Type>'
    else:
      arg_type = 'Variadic<SPV_Type>'
  elif kind == 'IdMemorySemantics' or kind == 'IdScope':
    # TODO(antiagainst): Need to further constrain 'IdMemorySemantics'
    # and 'IdScope' given that they should be gernated from OpConstant.
    assert quantifier == '', ('unexpected to have optional/variadic memory '
                              'semantics or scope <id>')
    arg_type = 'I32'
  elif kind == 'LiteralInteger':
    if quantifier == '':
      arg_type = 'I32Attr'
    elif quantifier == '?':
      arg_type = 'OptionalAttr<I32Attr>'
    else:
      arg_type = 'OptionalAttr<I32ArrayAttr>'
  elif kind == 'LiteralString' or \
      kind == 'LiteralContextDependentNumber' or \
      kind == 'LiteralExtInstInteger' or \
      kind == 'LiteralSpecConstantOpInteger' or \
      kind == 'PairLiteralIntegerIdRef' or \
      kind == 'PairIdRefLiteralInteger' or \
      kind == 'PairIdRefIdRef':
    assert False, '"{}" kind unimplemented'.format(kind)
  else:
    # The rest are all enum operands that we represent with op attributes.
    assert quantifier != '*', 'unexpected to have variadic enum attribute'
    arg_type = 'SPV_{}Attr'.format(kind)
    if quantifier == '?':
      arg_type = 'OptionalAttr<{}>'.format(arg_type)

  name = operand.get('name', '')
  name = snake_casify(name) if name else kind.lower()

  return '{}:${}'.format(arg_type, name)


def get_op_definition(instruction, doc, existing_info):
  """Generates the TableGen op definition for the given SPIR-V instruction.

  Arguments:
    - instruction: the instruction's SPIR-V JSON grammar
    - doc: the instruction's SPIR-V HTML doc
    - existing_info: a dict containing potential manually specified sections for
      this instruction

  Returns:
    - A string containing the TableGen op definition
  """
  fmt_str = 'def SPV_{opname}Op : SPV_Op<"{opname}", [{traits}]> {{\n'\
            '  let summary = {summary};\n\n'\
            '  let description = [{{\n'\
            '{description}\n\n'\
            '    ### Custom assembly form\n'\
            '{assembly}'\
            '}}];\n\n'\
            '  let arguments = (ins{args});\n\n'\
            '  let results = (outs{results});\n'\
            '{extras}'\
            '}}\n'

  opname = instruction['opname'][2:]

  summary, description = doc.split('\n', 1)
  wrapper = textwrap.TextWrapper(
      width=76, initial_indent='    ', subsequent_indent='    ')

  # Format summary. If the summary can fit in the same line, we print it out
  # as a "-quoted string; otherwise, wrap the lines using "[{...}]".
  summary = summary.strip();
  if len(summary) + len('  let summary = "";') <= 80:
    summary = '"{}"'.format(summary)
  else:
    summary = '[{{\n{}\n  }}]'.format(wrapper.fill(summary))

  # Wrap description
  description = description.split('\n')
  description = [wrapper.fill(line) for line in description if line]
  description = '\n\n'.join(description)

  operands = instruction.get('operands', [])

  # Set op's result
  results = ''
  if len(operands) > 0 and operands[0]['kind'] == 'IdResultType':
    results = '\n    SPV_Type:$result\n  '
    operands = operands[1:]
  if 'results' in existing_info:
    results = existing_info['results']

  # Ignore the operand standing for the result <id>
  if len(operands) > 0 and operands[0]['kind'] == 'IdResult':
    operands = operands[1:]

  # Set op' argument
  arguments = existing_info.get('arguments', None)
  if arguments is None:
    arguments = [map_spec_operand_to_ods_argument(o) for o in operands]
    arguments = '\n    '.join(arguments)
    if arguments:
      # Prepend and append whitespace for formatting
      arguments = '\n    {}\n  '.format(arguments)

  assembly = existing_info.get('assembly', None)
  if assembly is None:
    assembly = '    ``` {.ebnf}\n'\
               '    [TODO]\n'\
               '    ```\n\n'\
               '    For example:\n\n'\
               '    ```\n'\
               '    [TODO]\n'\
               '    ```\n  '

  return fmt_str.format(
      opname=opname,
      traits=existing_info.get('traits', ''),
      summary=summary,
      description=description,
      assembly=assembly,
      args=arguments,
      results=results,
      extras=existing_info.get('extras', ''))


def extract_td_op_info(op_def):
  """Extracts potentially manually specified sections in op's definition.

  Arguments: - A string containing the op's TableGen definition
    - doc: the instruction's SPIR-V HTML doc

  Returns:
    - A dict containing potential manually specified sections
  """
  # Get opname
  opname = [o[8:-2] for o in re.findall('def SPV_\w+Op', op_def)]
  assert len(opname) == 1, 'more than one ops in the same section!'
  opname = opname[0]

  # Get traits
  op_tmpl_params = op_def.split('<', 1)[1].split('>', 1)[0].split(', ', 1)
  if len(op_tmpl_params) == 1:
    traits = ''
  else:
    traits = op_tmpl_params[1].strip('[]')

  # Get custom assembly form
  rest = op_def.split('### Custom assembly form\n')
  assert len(rest) == 2, \
          '{}: cannot find "### Custom assembly form"'.format(opname)
  rest = rest[1].split('  let arguments = (ins')
  assert len(rest) == 2, '{}: cannot find arguments'.format(opname)
  assembly = rest[0].rstrip('}];\n')

  # Get arguments
  rest = rest[1].split('  let results = (outs')
  assert len(rest) == 2, '{}: cannot find results'.format(opname)
  args = rest[0].rstrip(');\n')

  # Get results
  rest = rest[1].split(');', 1)
  assert len(rest) == 2, \
          '{}: cannot find ");" ending results'.format(opname)
  results = rest[0]

  extras = rest[1].strip(' }\n')
  if extras:
    extras = '\n  {}\n'.format(extras)

  return {
      # Prefix with 'Op' to make it consistent with SPIR-V spec
      'opname': 'Op{}'.format(opname),
      'traits': traits,
      'assembly': assembly,
      'arguments': args,
      'results': results,
      'extras': extras
  }


def update_td_op_definitions(path, instructions, docs, filter_list):
  """Updates SPIRVOps.td with newly generated op definition.

  Arguments:
    - path: path to SPIRVOps.td
    - instructions: SPIR-V JSON grammar for all instructions
    - docs: SPIR-V HTML doc for all instructions
    - filter_list: a list containing new opnames to include

  Returns:
    - A string containing all the TableGen op definitions
  """
  with open(path, 'r') as f:
    content = f.read()

  # Split the file into chuncks, each containing one op.
  ops = content.split(AUTOGEN_OP_DEF_SEPARATOR)
  header = ops[0]
  footer = ops[-1]
  ops = ops[1:-1]

  # For each existing op, extract the manually-written sections out to retain
  # them when re-generating the ops. Also append the existing ops to filter
  # list.
  op_info_dict = {}
  for op in ops:
    info_dict = extract_td_op_info(op)
    opname = info_dict['opname']
    op_info_dict[opname] = info_dict
    filter_list.append(opname)
  filter_list = sorted(list(set(filter_list)))

  op_defs = []
  for opname in filter_list:
    # Find the grammar spec for this op
    instruction = next(
        inst for inst in instructions if inst['opname'] == opname)
    op_defs.append(
        get_op_definition(instruction, docs[opname],
                          op_info_dict.get(opname, {})))

  # Substitute the old op definitions
  op_defs = [header] + op_defs + [footer]
  content = AUTOGEN_OP_DEF_SEPARATOR.join(op_defs)

  with open(path, 'w') as f:
    f.write(content)


if __name__ == '__main__':
  import argparse

  cli_parser = argparse.ArgumentParser(
      description='Update SPIR-V dialect definitions using SPIR-V spec')

  cli_parser.add_argument(
      '--base-td-path',
      dest='base_td_path',
      type=str,
      default=None,
      help='Path to SPIRVBase.td')
  cli_parser.add_argument(
      '--op-td-path',
      dest='op_td_path',
      type=str,
      default=None,
      help='Path to SPIRVOps.td')

  cli_parser.add_argument(
      '--new-enum',
      dest='new_enum',
      type=str,
      default=None,
      help='SPIR-V enum to be added to SPIRVBase.td')
  cli_parser.add_argument(
      '--new-opcodes',
      dest='new_opcodes',
      type=str,
      default=None,
      nargs='*',
      help='update SPIR-V opcodes in SPIRVBase.td')
  cli_parser.add_argument(
      '--new-inst',
      dest='new_inst',
      type=str,
      default=None,
      help='SPIR-V instruction to be added to SPIRVOps.td')

  args = cli_parser.parse_args()

  operand_kinds, instructions = get_spirv_grammar_from_json_spec()

  # Define new enum attr
  if args.new_enum is not None:
    assert args.base_td_path is not None
    filter_list = [args.new_enum] if args.new_enum else []
    update_td_enum_attrs(args.base_td_path, operand_kinds, filter_list)

  # Define new opcode
  if args.new_opcodes is not None:
    assert args.base_td_path is not None
    update_td_opcodes(args.base_td_path, instructions, args.new_opcodes)

  # Define new op
  if args.new_inst is not None:
    assert args.op_td_path is not None
    filter_list = [args.new_inst] if args.new_inst else []
    docs = get_spirv_doc_from_html_spec()
    update_td_op_definitions(args.op_td_path, instructions, docs, filter_list)
    print('Done. Note that this script just generates a template; ', end='')
    print('please read the spec and update traits, arguments, and ', end='')
    print('results accordingly.')
