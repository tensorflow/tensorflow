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

import builtins
import io
import os
import sys
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
    with mock.patch.object(sys, "stdout", out):
      windows_lib_diagnostics._safe_print("C:\\Users\\Jos\u00e9\\x.dll")
    self.assertIn("Jos\\xe9", out.getvalue())

  def testSafePrintWithNoneEncoding(self):
    out = _FakeStdout(None)
    with mock.patch.object(sys, "stdout", out):
      windows_lib_diagnostics._safe_print("caf\u00e9")
    self.assertIn("caf\\xe9", out.getvalue())

  def testSafePrintNeverRaises(self):
    with mock.patch.object(builtins, "print", side_effect=ValueError("boom")):
      windows_lib_diagnostics._safe_print("text")

  def testUnsetPathInEnviron(self):
    out = _FakeStdout("utf-8")
    fake_ctypes = mock.MagicMock()
    fake_ctypes.WinDLL.side_effect = OSError("mock load failure")
    with mock.patch.object(sys, "stdout", out), \
        mock.patch.dict(os.environ, {}, clear=True), \
        mock.patch.object(windows_lib_diagnostics, "ctypes", fake_ctypes), \
        mock.patch.object(
            windows_lib_diagnostics.os.path, "exists",
            side_effect=lambda p: p == "main.dll"), \
        mock.patch.object(
            windows_lib_diagnostics, "get_dll_dependencies",
            return_value=["dep.dll"]):
      # Calls `diagnose_dll_load` directly (rather than `run_diagnosis`) so
      # that this actually reaches the `os.environ.get("PATH", "")` lookup,
      # which `run_diagnosis` only does on Windows with an existing path.
      # With PATH unset, this must not raise a KeyError while searching for
      # `dep.dll`'s location.
      windows_lib_diagnostics.diagnose_dll_load("main.dll")
    self.assertIn("Failed to load dep.dll", out.getvalue())

  def testRunDiagnosisSwallowsUnexpectedErrors(self):
    out = _FakeStdout("utf-8")
    with mock.patch.object(sys, "stdout", out), mock.patch.object(
        windows_lib_diagnostics.os.path, "abspath",
        side_effect=RuntimeError("unexpected")):
      windows_lib_diagnostics.run_diagnosis(None)
    self.assertIn("Diagnostic failed: unexpected", out.getvalue())


if __name__ == "__main__":
  test.main()
