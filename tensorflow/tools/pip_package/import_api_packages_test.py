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

This Python test verifies whether the API v2 packages can be imported from the
current build. It utilizes the `_api/v2/api_packages.txt` list of packages from
the local wheel file specified in the `requirements_lock_<python_version>.txt`.

Packages are imported one by one in alphabetical order during runtime.

The test doesn't identify package's order-dependent issues; for instance,
importing "tf.foo" followed by "tf.bar" won't reveal that "tf.bar" depends on
"tf.foo" being imported first.

The `_api/v2/api_packages.txt` file is generated during the TensorFlow API v2
init files creation process and is subsequently stored in the wheel file after
the build. It also contains a few paths that cannot be directly imported. These
paths point to attributes or sub-modules within a module's namespace, but they
don't correspond to an actual file or directory on the filesystem. The list of
such paths is stored in the packages_for_skip variable and will be skipped
during the test.
"""

import logging
import os
import unittest

try:
  import importlib.resources as pkg_resources  # pylint: disable=g-import-not-at-top
except ImportError:
  import importlib_resources as pkg_resources  # pylint: disable=g-import-not-at-top

logging.basicConfig(level=logging.INFO)


class ImportApiPackagesTest(unittest.TestCase):
  """ImportApiPackagesTest class. See description at the top of the file."""

  def setUp(self):
    def _get_api_packages_v2():
      api_packages_path = os.path.join("_api", "v2", "api_packages.txt")
      logging.info("Load api packages file: %s", api_packages_path)
      return set(
          pkg_resources.files("tensorflow")
          .joinpath(api_packages_path)
          .read_text()
          .splitlines()
      )

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
