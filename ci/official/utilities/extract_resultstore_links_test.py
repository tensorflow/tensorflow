# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for extract_resultstore_links.

Run manually from the TensorFlow repository root:
  python3 ci/official/utilities/extract_resultstore_links_test.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElemTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_resultstore_links as erl  # pylint: disable=g-import-not-at-top

_URL_1 = (
    'https://source.cloud.google.com/results/invocations/'
    '11111111-1111-1111-1111-111111111111'
)
_URL_2 = (
    'https://source.cloud.google.com/results/invocations/'
    '22222222-2222-2222-2222-222222222222'
)
_URL_3 = (
    'https://source.cloud.google.com/results/invocations/'
    '33333333-3333-3333-3333-333333333333'
)


class ExtractResultstoreLinksTest(unittest.TestCase):
  """Tests ResultStore log parsing and JUnit XML retry handling."""

  def _parse_and_write_xml(
      self, log_content: str
  ) -> tuple[erl.ResultDictType, ElemTree.Element]:
    with tempfile.TemporaryDirectory() as tmpdir:
      log_path = os.path.join(tmpdir, 'script.log')
      xml_path = os.path.join(tmpdir, 'out', 'sponge_log.xml')
      with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(log_content)
      parsed = erl.parse_log(log_path)
      erl.create_xml_file(parsed, xml_path)
      tree = ElemTree.parse(xml_path)
      return parsed, tree.getroot()

  def test_error_build_line_marked_build_failed(self) -> None:
    log_content = '\n'.join([
        '+ docker exec tf bazel --output_user_root=C:/x test //tensorflow/...',
        f'INFO: Streaming build results to: {_URL_1}',
        'lld-link: error: undefined symbol: main',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_1}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(parsed[_URL_1]['status'], erl.InvokeStatus.build_failed)
    self.assertEqual(
        parsed[_URL_1]['command'],
        'bazel --output_user_root=C:/x test //tensorflow/...',
    )
    self.assertEqual(xml_root.attrib['tests'], '1')
    self.assertEqual(xml_root.attrib['errors'], '1')
    self.assertEqual(xml_root.attrib['failures'], '0')
    self.assertEqual(len(xml_root.findall('.//error')), 1)

  def test_failed_build_superseded_by_passing_retry(self) -> None:
    cmd = (
        '+ docker exec tf bazel --output_user_root=C:/x test'
        ' --config=windows_x86_cpu_2022_pycpp_test'
    )
    log_content = '\n'.join([
        cmd,
        f'INFO: Streaming build results to: {_URL_1}',
        'lld-link: error: undefined symbol: main',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_1}',
        (
            '==TFCI==: Bazel command failed with exit code 1 (attempt 1/3);'
            ' retrying against warm Bazel cache...'
        ),
        cmd,
        f'INFO: Streaming build results to: {_URL_2}',
        'INFO: Build completed successfully, 6500 total actions',
        f'INFO: Streaming build results to: {_URL_2}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(
        parsed[_URL_1]['status'], erl.InvokeStatus.superseded_by_retry
    )
    self.assertEqual(parsed[_URL_1]['superseded_by'], _URL_2)
    self.assertEqual(parsed[_URL_2]['status'], erl.InvokeStatus.passed)
    self.assertEqual(xml_root.attrib['tests'], '2')
    self.assertEqual(xml_root.attrib['errors'], '0')
    self.assertEqual(xml_root.attrib['failures'], '0')
    self.assertEqual(len(xml_root.findall('.//error')), 0)
    self.assertEqual(len(xml_root.findall('.//failure')), 0)
    system_outs = xml_root.findall('.//system-out')
    self.assertEqual(len(system_outs), 1)
    self.assertIn('undefined symbol: main', system_outs[0].text or '')
    xml_bytes = ElemTree.tostring(xml_root, encoding='unicode')
    self.assertIn(_URL_1, xml_bytes)
    self.assertIn(_URL_2, xml_bytes)

  def test_two_failed_attempts_superseded_by_third_attempt_and_printed(
      self,
  ) -> None:
    cmd = (
        '+ docker exec tf bazel --output_user_root=C:/x build'
        ' --config=windows_x86_cpu_2022 //tensorflow/tools/pip_package:wheel'
    )
    log_content = '\n'.join([
        cmd,
        f'INFO: Streaming build results to: {_URL_1}',
        'lld-link: error: undefined symbol: first_flake',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_1}',
        cmd,
        f'INFO: Streaming build results to: {_URL_2}',
        'lld-link: error: undefined symbol: second_flake',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_2}',
        cmd,
        f'INFO: Streaming build results to: {_URL_3}',
        'INFO: Build completed successfully, 6500 total actions',
        f'INFO: Streaming build results to: {_URL_3}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(
        parsed[_URL_1]['status'], erl.InvokeStatus.superseded_by_retry
    )
    self.assertEqual(parsed[_URL_1]['superseded_by'], _URL_3)
    self.assertEqual(
        parsed[_URL_2]['status'], erl.InvokeStatus.superseded_by_retry
    )
    self.assertEqual(parsed[_URL_2]['superseded_by'], _URL_3)
    self.assertEqual(parsed[_URL_3]['status'], erl.InvokeStatus.passed)
    self.assertEqual(xml_root.attrib['tests'], '3')
    self.assertEqual(xml_root.attrib['errors'], '0')
    self.assertEqual(xml_root.attrib['failures'], '0')
    self.assertEqual(len(xml_root.findall('.//system-out')), 2)

    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
      erl.print_invocation_results(parsed)
    printed = stdout_buf.getvalue()
    self.assertEqual(printed.count(f'Superseded by passing retry: {_URL_3}'), 2)

  def test_incomplete_retry_does_not_supersede_earlier_failure(self) -> None:
    cmd = '+ bazel build //tensorflow/core/...'
    log_content = '\n'.join([
        cmd,
        f'INFO: Streaming build results to: {_URL_1}',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_1}',
        cmd,
        f'INFO: Streaming build results to: {_URL_2}',
        'FATAL: bazel crashed due to an internal error.',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(parsed[_URL_1]['status'], erl.InvokeStatus.build_failed)
    self.assertEqual(xml_root.attrib['errors'], '1')

  def test_failed_tests_not_superseded_by_later_pass(self) -> None:
    cmd = '+ bazel test --config=windows_x86_cpu_2022 //tensorflow/core/...'
    log_content = '\n'.join([
        cmd,
        f'INFO: Streaming build results to: {_URL_1}',
        'INFO: Build completed, 1 test FAILED, 500 total actions',
        f'INFO: Streaming build results to: {_URL_1}',
        cmd,
        f'INFO: Streaming build results to: {_URL_2}',
        'INFO: Build completed successfully, 500 total actions',
        f'INFO: Streaming build results to: {_URL_2}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(parsed[_URL_1]['status'], erl.InvokeStatus.tests_failed)
    self.assertEqual(xml_root.attrib['errors'], '0')
    self.assertEqual(xml_root.attrib['failures'], '1')

  def test_exhausted_retries_still_report_errors(self) -> None:
    cmd = '+ bazel build --config=windows_x86_cpu_2022 //tensorflow/core/...'
    log_content = '\n'.join([
        cmd,
        f'INFO: Streaming build results to: {_URL_1}',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_1}',
        cmd,
        f'INFO: Streaming build results to: {_URL_2}',
        'FAILED: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_2}',
        cmd,
        f'INFO: Streaming build results to: {_URL_3}',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_3}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(parsed[_URL_1]['status'], erl.InvokeStatus.build_failed)
    self.assertEqual(parsed[_URL_2]['status'], erl.InvokeStatus.build_failed)
    self.assertEqual(parsed[_URL_3]['status'], erl.InvokeStatus.build_failed)
    self.assertEqual(xml_root.attrib['tests'], '3')
    self.assertEqual(xml_root.attrib['errors'], '3')
    self.assertEqual(xml_root.attrib['failures'], '0')

  def test_different_bazel_command_does_not_supersede_earlier_failure(
      self,
  ) -> None:
    log_content = '\n'.join([
        (
            '+ bazel build //tensorflow/tools/pip_package:wheel'
            ' --repo_env=WHEEL_NAME=tensorflow'
        ),
        f'INFO: Streaming build results to: {_URL_1}',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_1}',
        (
            '+ bazel build //tensorflow/tools/pip_package:wheel'
            ' --repo_env=WHEEL_NAME=tensorflow_cpu'
        ),
        f'INFO: Streaming build results to: {_URL_2}',
        'INFO: Build completed successfully, 100 total actions',
        f'INFO: Streaming build results to: {_URL_2}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(parsed[_URL_1]['status'], erl.InvokeStatus.build_failed)
    self.assertEqual(parsed[_URL_2]['status'], erl.InvokeStatus.passed)
    self.assertEqual(xml_root.attrib['errors'], '1')

  def test_earlier_pass_does_not_supersede_later_failure(self) -> None:
    cmd = '+ bazel test //tensorflow/core/...'
    log_content = '\n'.join([
        cmd,
        f'INFO: Streaming build results to: {_URL_1}',
        'INFO: Build completed successfully, 100 total actions',
        f'INFO: Streaming build results to: {_URL_1}',
        cmd,
        f'INFO: Streaming build results to: {_URL_2}',
        'INFO: Build completed, 2 tests FAILED, 100 total actions',
        f'INFO: Streaming build results to: {_URL_2}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(parsed[_URL_1]['status'], erl.InvokeStatus.passed)
    self.assertEqual(parsed[_URL_2]['status'], erl.InvokeStatus.tests_failed)
    self.assertEqual(xml_root.attrib['failures'], '1')

  def test_unparsed_command_failure_is_not_superseded(self) -> None:
    log_content = '\n'.join([
        f'INFO: Streaming build results to: {_URL_1}',
        'ERROR: Build did NOT complete successfully',
        f'INFO: Streaming build results to: {_URL_1}',
        f'INFO: Streaming build results to: {_URL_2}',
        'INFO: Build completed successfully, 100 total actions',
        f'INFO: Streaming build results to: {_URL_2}',
    ])
    parsed, xml_root = self._parse_and_write_xml(log_content)

    self.assertEqual(parsed[_URL_1]['status'], erl.InvokeStatus.build_failed)
    self.assertEqual(xml_root.attrib['errors'], '1')


if __name__ == '__main__':
  unittest.main()
