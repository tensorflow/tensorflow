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
"""Tests for the TensorFlow build integrity checks."""

import os
import sys
from unittest import mock

from tensorflow.python.platform import self_check
from tensorflow.python.platform import test


class ConflictingInstallsTest(test.TestCase):

  def _with_installed(self, installed):
    return mock.patch.object(
        self_check,
        "_installed_tensorflow_distributions",
        return_value=installed,
    )

  def testSingleDistributionPasses(self):
    with self._with_installed({"tensorflow": "2.21.0"}):
      self_check._check_conflicting_installs()

  def testNoDistributionsPasses(self):
    # Metadata can be absent entirely, e.g. when running from a source tree.
    with self._with_installed({}):
      self_check._check_conflicting_installs()

  def testMatchingVersionsPass(self):
    # On Windows `tensorflow` is a meta-package requiring `tensorflow-intel`
    # pinned to the same version, so this combination is expected.
    with self._with_installed(
        {"tensorflow": "2.21.0", "tensorflow-intel": "2.21.0"}
    ):
      self_check._check_conflicting_installs()

  def testMismatchedVersionsRaise(self):
    with self._with_installed(
        {"tensorflow": "2.21.0", "tensorflow-cpu": "2.20.0"}
    ):
      with self.assertRaisesRegex(ImportError, "conflicting TensorFlow"):
        self_check._check_conflicting_installs()

  def testErrorNamesEachDistributionAndVersion(self):
    with self._with_installed(
        {"tensorflow": "2.21.0", "tensorflow-cpu": "2.20.0"}
    ):
      with self.assertRaisesRegex(
          ImportError, "tensorflow 2.21.0, tensorflow-cpu 2.20.0"
      ):
        self_check._check_conflicting_installs()

  def testNightlyMismatchRaises(self):
    with self._with_installed(
        {"tf-nightly": "2.22.0.dev1", "tf-nightly-cpu": "2.22.0.dev2"}
    ):
      with self.assertRaisesRegex(ImportError, "conflicting TensorFlow"):
        self_check._check_conflicting_installs()

  def testMissingMetadataIsIgnored(self):
    import importlib.metadata  # pylint: disable=g-import-not-at-top

    with mock.patch.object(
        importlib.metadata,
        "version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
      self.assertEqual(self_check._installed_tensorflow_distributions(), {})

  def testMalformedMetadataDoesNotMaskRealError(self):
    import importlib.metadata  # pylint: disable=g-import-not-at-top

    with mock.patch.object(
        importlib.metadata, "version", side_effect=ValueError("bad metadata")
    ):
      self.assertEqual(self_check._installed_tensorflow_distributions(), {})


class PythonBitnessTest(test.TestCase):

  def testSixtyFourBitPasses(self):
    with mock.patch.object(self_check, "_python_bitness", return_value=64):
      self_check._check_python_bitness()

  def testThirtyTwoBitRaises(self):
    with mock.patch.object(self_check, "_python_bitness", return_value=32):
      with self.assertRaisesRegex(ImportError, "64-bit Python"):
        self_check._check_python_bitness()

  def testReportsActualBitnessInMessage(self):
    with mock.patch.object(self_check, "_python_bitness", return_value=32):
      with self.assertRaisesRegex(ImportError, "this interpreter is 32-bit"):
        self_check._check_python_bitness()

  def testRealInterpreterIsSixtyFourBit(self):
    # The test binary itself must be 64-bit, so the unmocked check passes.
    self.assertEqual(self_check._python_bitness(), 64)
    self_check._check_python_bitness()


class PreloadCheckTest(test.TestCase):
  """Verifies the new checks are actually reached from `preload_check`."""

  def testWindowsRunsNewChecks(self):
    with mock.patch.object(self_check, "_is_windows", return_value=True):
      with mock.patch.object(self_check, "_check_python_bitness") as bitness:
        with mock.patch.object(
            self_check, "_check_conflicting_installs"
        ) as conflicts:
          self_check.preload_check()
    bitness.assert_called_once()
    conflicts.assert_called_once()

  def testNonWindowsSkipsNewChecks(self):
    # The non-Windows branch loads a native CPU feature guard extension, which
    # is unrelated to what this test covers; stub it so the test exercises only
    # the branch selection.
    guard = mock.MagicMock()
    with mock.patch.dict(
        sys.modules,
        {"tensorflow.python.platform._pywrap_cpu_feature_guard": guard},
    ):
      with mock.patch.object(self_check, "_is_windows", return_value=False):
        with mock.patch.object(self_check, "_check_python_bitness") as bitness:
          with mock.patch.object(
              self_check, "_check_conflicting_installs"
          ) as conflicts:
            self_check.preload_check()
    bitness.assert_not_called()
    conflicts.assert_not_called()

  def testConflictingInstallsReportedBeforeDllScan(self):
    # A conflicting install must surface its own error rather than the
    # generic missing-DLL message, which is what made this hard to diagnose.
    with mock.patch.object(self_check, "_is_windows", return_value=True):
      with mock.patch.object(
          self_check,
          "_installed_tensorflow_distributions",
          return_value={"tensorflow": "2.21.0", "tensorflow-cpu": "2.20.0"},
      ):
        with mock.patch.dict(
            self_check.build_info.build_info,
            {self_check.MSVCP_DLL_NAMES: "msvcp140.dll"},
        ):
          with mock.patch.dict(os.environ, {"PATH": ""}):
            with self.assertRaisesRegex(ImportError, "conflicting TensorFlow"):
              self_check.preload_check()


if __name__ == "__main__":
  test.main()
