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
"""Tests for windows_lib_diagnostics."""

import io
import os
from unittest import mock

from tensorflow.python.platform import test
from tensorflow.python.platform import windows_lib_diagnostics


class _FakeStdout(io.StringIO):

  def __init__(self, encoding):
    super().__init__()
    self._encoding = encoding

  @property
  def encoding(self):
    return self._encoding


class WindowsLibDiagnosticsTest(test.TestCase):

  def testSafePrintWithAsciiConsole(self):
    out = _FakeStdout("ascii")
    with mock.patch("sys.stdout", out):
      windows_lib_diagnostics._safe_print("C:\\Users\\Jos\u00e9\\x.dll")
    self.assertIn("Jos\\xe9", out.getvalue())

  def testSafePrintWithNoneEncoding(self):
    out = _FakeStdout(None)
    with mock.patch("sys.stdout", out):
      windows_lib_diagnostics._safe_print("caf\u00e9")
    self.assertIn("caf\\xe9", out.getvalue())

  def testSafePrintNeverRaises(self):
    with mock.patch("builtins.print", side_effect=ValueError("boom")):
      windows_lib_diagnostics._safe_print("text")

  def testUnsetPathInEnviron(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      windows_lib_diagnostics.run_diagnosis("nonexistent.dll")

  def testRunDiagnosisSwallowsUnexpectedErrors(self):
    out = _FakeStdout("utf-8")
    with mock.patch("sys.stdout", out), mock.patch.object(
        windows_lib_diagnostics.os.path, "abspath",
        side_effect=RuntimeError("unexpected")):
      windows_lib_diagnostics.run_diagnosis(None)
    self.assertIn("Diagnostic failed: unexpected", out.getvalue())


if __name__ == "__main__":
  test.main()
