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
"""lazy loader tests."""

# pylint: disable=unused-import
import doctest
import inspect
import types

from tensorflow.python.platform import test
from tensorflow.python.util import lazy_loader


class LazyLoaderTest(test.TestCase):

  def testDocTestDoesNotLoad(self):
    module = types.ModuleType("mytestmodule")
    module.foo = lazy_loader.LazyLoader("foo", module.__dict__, "os.path")

    self.assertIsInstance(module.foo, lazy_loader.LazyLoader)

    finder = doctest.DocTestFinder()
    finder.find(module)

    self.assertIsInstance(module.foo, lazy_loader.LazyLoader)


if __name__ == "__main__":
  test.main()
