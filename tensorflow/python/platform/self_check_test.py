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
    with mock.patch.object(self_check.struct, "calcsize", return_value=8):
      self_check._check_python_bitness()

  def testThirtyTwoBitRaises(self):
    with mock.patch.object(self_check.struct, "calcsize", return_value=4):
      with self.assertRaisesRegex(ImportError, "64-bit Python"):
        self_check._check_python_bitness()


if __name__ == "__main__":
  test.main()
