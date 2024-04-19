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

"""Import API packages test.

This is a Python test that verifies whether API v2 packages can be imported
from the current build or not.

It uses the `_api/v2/api_packages.txt` file from the local wheel file.
The `_api/v2/api_packages.txt` file is created during the process of generating
TensorFlow API v2 init files and is stored in the wheel file after the build.

See README.md file for "how to run" instruction.
"""

import logging
import unittest
import pkg_resources

logging.basicConfig(level=logging.INFO)


class ImportApiPackagesTest(unittest.TestCase):
  """ImportApiPackagesTest class. See description at the top of the file."""

  def setUp(self):
    def _get_api_packages_v2():
      api_packages_path = pkg_resources.resource_filename(
          "tensorflow", "_api/v2/api_packages.txt"
      )

      logging.info("Load api packages file: %s", api_packages_path)
      with open(api_packages_path) as file:
        return set(file.read().splitlines())

    super().setUp()
    self.api_packages_v2 = _get_api_packages_v2()
    # Some paths in the api_packages_path file cannot be directly imported.
    # These paths may point to attributes or sub-modules within a module's
    # namespace, but they don't correspond to an actual file
    # or directory on the filesystem.
    self.packages_for_skip = [
        "tensorflow.distribute.cluster_resolver",
        "tensorflow.lite.experimental.authoring",
        "tensorflow.distribute.experimental.coordinator",
        "tensorflow.summary.experimental",
        "tensorflow.distribute.coordinator",
        "tensorflow.distribute.experimental.partitioners",
    ]

  def test_import_runtime(self):
    """Try to import all packages from api packages file one by one."""
    version = "v2"
    failed_packages = []

    logging.info("Try to import packages at runtime...")
    for package_name in self.api_packages_v2:
      # Convert package name to the short version:
      # tensorflow._api.v2.distribute.experimental to
      # tensorflow.distribute.experimental
      short_package_name = package_name.replace(f"_api.{version}.", "")
      # skip non-importable paths
      if short_package_name not in self.packages_for_skip:
        try:
          __import__(short_package_name)
        except ImportError:
          logging.exception("error importing %s", short_package_name)
          failed_packages.append(package_name)

    if failed_packages:
      self.fail(
          "Failed to import"
          f" {len(failed_packages)}/{len(self.api_packages_v2)} packages"
          f" {version}:\n{failed_packages}"
      )
    logging.info("Import of packages was successful.")


if __name__ == "__main__":
  unittest.main()
