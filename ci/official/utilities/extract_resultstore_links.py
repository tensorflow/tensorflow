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
# ==============================================================================

"""Extracts ResultStore links from a log containing Bazel invocations.

The links and the invocations' status can then be printed out, or output in the
form of JUnit-based XML.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
from typing import Dict, Union
import xml.etree.ElementTree as ElemTree


# Use typing.Dict/Union so this runtime alias remains importable on Python <3.10
ResultDictType = Dict[str, Dict[str, Union[str, int]]]

RESULT_STORE_LINK_RE = re.compile(
    r'^INFO: Streaming build results to: (https://[\w./\-]+)'
)
FAILED_BUILD_PREFIXES = (
    'FAILED: Build did NOT complete successfully',
    'ERROR: Build did NOT complete successfully',
)
BUILD_STATUS_LINE = 'INFO: Build'
BUILD_PASSED_LINE = 'INFO: Build completed successfully'
TESTS_FAILED_RE = re.compile(r'^INFO: Build completed, \d+ tests? FAILED')
BAZEL_COMMAND_RE = re.compile(
    r'(^| )(?P<command>bazel (.*? )?(?P<type>test|build) .+)'
)


class InvokeStatus:
  """Status values for parsed Bazel ResultStore invocations."""

  tests_failed = 'tests_failed'
  build_failed = 'build_failed'
  superseded_by_retry = 'superseded_by_retry'
  passed = 'passed'


def parse_args() -> argparse.Namespace:
  """Parses the commandline args."""
  parser = argparse.ArgumentParser(
      description=(
          'Extracts ResultStore links from a build log.\n'
          'These can be then printed out, and/or output into a '
          'JUnit-based XML file inside a specified directory.'
      )
  )

  parser.add_argument('build_log', help='Path to a build log.')
  parser.add_argument(
      '--xml-out-path',
      required=False,
      help=(
          'Path to which to output the JUnit-based XML with ResultStore links.'
      ),
  )
  parser.add_argument(
      '--print',
      action='store_true',
      dest='print',
      default=False,
      help=(
          'Whether to print out a short summary with the '
          'found ResultStore links (if any).'
      ),
  )
  parser.add_argument(
      '-v',
      '--verbose',
      action='store_true',
      dest='verbose',
      default=False,
      help='Prints out lines helpful for debugging.',
  )
  parsed_args = parser.parse_args()
  if not parsed_args.print and not parsed_args.xml_out_path:
    raise TypeError('`--print` or `--xml-out-path` must be specified')

  return parsed_args


def _mark_superseded_retries(result_store_links: ResultDictType) -> None:
  """Marks build failures superseded when a later identical retry passes."""
  invocations = list(result_store_links.items())
  for idx, (_, lines) in enumerate(invocations):
    if lines.get('status') != InvokeStatus.build_failed:
      continue
    command = lines.get('command')
    if not command:
      continue
    for later_url, later_lines in invocations[idx + 1 :]:
      if (
          later_lines.get('command') == command
          and later_lines.get('status') == InvokeStatus.passed
          and later_lines.get('completed_successfully') == 1
      ):
        lines['status'] = InvokeStatus.superseded_by_retry
        lines['superseded_by'] = later_url
        break


def parse_log(file_path: str, verbose: bool = False) -> ResultDictType:
  """Finds ResultStore links, and tries to determine their status."""
  with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    log_lines = f.read().splitlines()

  result_store_links: ResultDictType = {}
  current_url: str | None = None
  for i, line in enumerate(log_lines):
    result_store_line_match = re.search(RESULT_STORE_LINK_RE, line)
    if not result_store_line_match:
      continue

    url = result_store_line_match.group(1)
    url_lines = result_store_links.setdefault(url, {})
    # Each bazel RBE invocation should produce two
    # 'Streaming build results to: ...' lines, one at the start, and one at the
    # end of the invocation.
    # If there's a failure message, it will be found in-between these two.

    if not current_url:
      url_lines['start'] = i
    elif current_url == url:
      url_lines['end'] = i
    else:
      result_store_links[current_url]['next_url'] = i
      url_lines['start'] = i
    current_url = url

  previous_end_line: int | None = None
  for url, lines in result_store_links.items():
    lines['status'] = InvokeStatus.passed  # default to passed
    start_line = int(lines['start'])
    end_line = int(lines.get('end', lines.get('next_url', len(log_lines)))) - 1
    k = end_line
    while k > start_line:
      backtrack_line = log_lines[k]
      build_failed = backtrack_line.startswith(FAILED_BUILD_PREFIXES)
      if build_failed or not backtrack_line.startswith(BUILD_STATUS_LINE):
        tests_failed = False
      else:
        tests_failed = bool(re.search(TESTS_FAILED_RE, backtrack_line))
      if build_failed or tests_failed:
        log_fragment = '\n'.join(log_lines[max(k - 20, 0) : end_line + 1])
        lines['log_fragment'] = log_fragment
        lines['status'] = (
            InvokeStatus.build_failed
            if build_failed
            else InvokeStatus.tests_failed
        )
        if verbose:
          print(
              f'Found failed invocation: {url.rsplit("/")[-1]}\n'
              'Log fragment:\n'
              f'```\n{log_fragment}\n```\n'
              f'{"=" * 140}'
          )
        break
      if backtrack_line.startswith(BUILD_PASSED_LINE):
        lines['completed_successfully'] = 1
        break
      k -= 1

    # A low-effort attempt to find the bazel command that triggered the
    # invocation.
    bazel_comm_min_line_i = (
        previous_end_line + 1 if previous_end_line is not None else 0
    )
    cmd_line_i = start_line - 1
    while cmd_line_i >= bazel_comm_min_line_i:
      backtrack_line = log_lines[cmd_line_i]
      # Don't attempt to parse multi-line commands broken up by backslashes
      if 'bazel ' in backtrack_line and not backtrack_line.endswith('\\'):
        bazel_line = BAZEL_COMMAND_RE.search(backtrack_line)
        if bazel_line:
          lines['command'] = bazel_line.group('command')
          lines['command_type'] = bazel_line.group('type')
          break
      cmd_line_i -= 1
    previous_end_line = int(lines.get('end', start_line))

  _mark_superseded_retries(result_store_links)
  return result_store_links


def indent_xml(elem: ElemTree.Element, level: int = 0) -> None:
  """Indents and newlines the XML for better output."""
  indent_str = '\n' + level * '  '
  if len(elem):  # pylint: disable=g-explicit-length-test  # `if elem` not valid
    if not elem.text or not elem.text.strip():
      elem.text = indent_str + '  '
    if not elem.tail or not elem.tail.strip():
      elem.tail = indent_str
    last_child = elem
    for child in elem:
      indent_xml(child, level + 1)
      last_child = child
    if not last_child.tail or not last_child.tail.strip():
      last_child.tail = indent_str
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = indent_str


def create_xml_file(
    result_store_dict: ResultDictType, output_path: str, verbose: bool = False
) -> None:
  """Creates a JUnit-based XML file, with each invocation as a testcase."""
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  failure_count = 0
  error_count = 0

  attrib = {
      'name': 'Bazel Invocations',
      'time': '0.0',
      'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }
  testsuites = ElemTree.Element('testsuites')
  testsuite = ElemTree.SubElement(testsuites, 'testsuite')
  for url, invocation_results in result_store_dict.items():
    invocation_id = url.rsplit('/')[-1]
    if verbose:
      print(f'Creating testcase for invocation {invocation_id}')
    status = invocation_results['status']
    command = invocation_results.get('command')
    command_type = invocation_results.get('command_type')

    case_attrib = attrib.copy()
    if command_type:
      command_type_title = str(command_type).title()
      case_name = f'{command_type_title} invocation {invocation_id}'
    else:
      case_name = f' Invocation {invocation_id}'
    case_attrib.update(
        {'name': case_name, 'status': 'run', 'result': 'completed'}
    )

    testcase = ElemTree.SubElement(testsuite, 'testcase', attrib=case_attrib)
    if status in (InvokeStatus.tests_failed, InvokeStatus.build_failed):
      if status == InvokeStatus.tests_failed:
        failure_count += 1
        elem_name = 'failure'
      else:
        error_count += 1
        elem_name = 'error'
      if command:
        failure_msg = f'\nThe command was:\n\n{command}\n\n'
      else:
        failure_msg = (
            "\nCouldn't parse a bazel command "
            'matching the invocation, inside the log. '
            'Please look for it in the build log.\n\n'
        )
      failure_msg += (
          'See the ResultStore link for a detailed view of failed targets:\n'
          f'{url}\n\n'
      )
      failure_msg += (
          "Here's a fragment of the log containing the failure:\n\n"
          '[ ... TRUNCATED ... ]\n\n'
          f'{invocation_results["log_fragment"]}\n'
          '\n[ ... TRUNCATED ... ]\n'
      )
      failure = ElemTree.SubElement(
          testcase,
          elem_name,
          message=f'Bazel invocation {invocation_id} failed.',
      )
      failure.text = failure_msg
    elif status == InvokeStatus.superseded_by_retry:
      properties = ElemTree.SubElement(testcase, 'properties')
      superseded_by = str(invocation_results.get('superseded_by', ''))
      retry_msg = (
          'Invocation failed initially, but a subsequent retry succeeded.\n'
          f'Failed attempt: {url}\n'
          f'Passing retry: {superseded_by}'
      )
      ElemTree.SubElement(
          properties, 'property', name='description', value=retry_msg
      )
      ElemTree.SubElement(
          properties, 'property', name='superseded_by', value=superseded_by
      )
      if command:
        ElemTree.SubElement(
            properties, 'property', name='bazel_command', value=str(command)
        )
      log_fragment = invocation_results.get('log_fragment')
      if log_fragment:
        system_out = ElemTree.SubElement(testcase, 'system-out')
        system_out.text = str(log_fragment)
    else:
      properties = ElemTree.SubElement(testcase, 'properties')
      success_msg = f'Build completed successfully.\nSee {url} for details.'
      ElemTree.SubElement(
          properties, 'property', name='description', value=success_msg
      )
      if command:
        ElemTree.SubElement(
            properties, 'property', name='bazel_command', value=str(command)
        )

  suite_specific = {
      'tests': str(len(result_store_dict)),
      'errors': str(error_count),
      'failures': str(failure_count),
  }
  suite_attrib = attrib.copy()
  suite_attrib.update(suite_specific)
  testsuites.attrib = suite_attrib
  testsuite.attrib = suite_attrib
  indent_xml(testsuites)

  tree = ElemTree.ElementTree(testsuites)
  file_path = os.path.join(output_path)
  with open(file_path, 'wb') as f:
    f.write(b'<?xml version="1.0"?>\n')
    tree.write(f)
    if verbose:
      print(f'\nWrote XML with Bazel invocation results to {file_path}')


def print_invocation_results(result_store_dict: ResultDictType) -> None:
  """Prints out a short summary of the found ResultStore links (if any)."""
  print()
  if not result_store_dict:
    print('Found no ResultStore links for Bazel build/test invocations.')
  else:
    print(
        f'Found {len(result_store_dict)} ResultStore link(s) for '
        'Bazel invocations.\n'
        'ResultStore contains individual representations of each target '
        'that were run/built during the invocation.\n'
        'These results are generally easier to read than looking through '
        'the entire build log:\n'
    )
  for i, (url, invocation_results) in enumerate(
      result_store_dict.items(), start=1
  ):
    line_str = f'Invocation #{i} ({invocation_results["status"]}):\n'
    command = invocation_results.get('command')
    if command:
      line_str += str(command)
    else:
      line_str += (
          "Couldn't parse the bazel command, check inside the build log instead"
      )
    line_str += f'\n{url}\n'
    superseded_by = invocation_results.get('superseded_by')
    if superseded_by:
      line_str += f'Superseded by passing retry: {superseded_by}\n'
    print(line_str)


def main() -> None:
  args = parse_args()
  verbose = args.verbose
  build_log_path = os.path.expandvars(args.build_log)
  links = parse_log(build_log_path, verbose=verbose)

  if args.xml_out_path:
    output_path = os.path.expandvars(args.xml_out_path)
    create_xml_file(links, output_path, verbose=verbose)
  if args.print:
    print_invocation_results(links)


if __name__ == '__main__':
  main()
