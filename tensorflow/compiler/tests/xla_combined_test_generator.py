# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
r"""Generates a combined test file for a list of test files.

Given a list of python files, this script will generate a new test file that
imports all classes from all those files and then calls test.main(). The
tensorflow.python.platform.test will automatically find all the classes that
have been imported and run them as tests.

The way this script finds the classes from the files is unable to determine the
original package and so the caller of this script must provide the package name
as a flag.

Typical usage:
  generate_xla_test --test_files=file1.py,file2.py  \
    --package=package.name

Assuming each of the test files contains a test class with a similar name, this
usage will generate the following on stdout:
  from tensorflow.python.platform import test

  from package.name.file1 import TestClass1
  from package.name.file2 import TestClass2

  if __name__ == "__main__":
    test.main()
"""

import importlib.util
import inspect
import os

from absl import app
from absl import flags

_TEST_FILES = flags.DEFINE_list(
    "test_files", None, "list of python files containing xla tests"
)
_PACKAGE = flags.DEFINE_string(
    "package", None, "package name for test to import"
)


def find_classes_in_path(file_name):
  """Finds all classes within a given file."""

  class_names = []

  if os.path.isfile(file_name) and file_name.endswith(".py"):
    # Load the module
    spec = importlib.util.spec_from_file_location("module_name", file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find classes in the module
    for name, obj in inspect.getmembers(module):
      if inspect.isclass(obj):
        class_names.append(name)

  return class_names


def main(unused_argv):
  print("from tensorflow.python.platform import test")
  for test_file in _TEST_FILES.value:
    class_names = find_classes_in_path(test_file)
    for class_name in class_names:
      print(
          "from"
          f" {_PACKAGE.value}.{os.path.basename(test_file).replace('.py', '')} import"
          f" {class_name}"
      )
  print("""
if __name__ == "__main__":
  test.main()
""")


if __name__ == "__main__":
  app.run(main)
