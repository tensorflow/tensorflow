# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest
from tensorflow.python.platform import resource_loader


class ResourceLoaderTest(googletest.TestCase):

  def test_exception(self):
    with self.assertRaises(IOError):
      resource_loader.load_resource("/fake/file/path/dne")

  def test_exists(self):
    contents = resource_loader.load_resource(
        "python/platform/resource_loader.py")
    self.assertIn(b"tensorflow", contents)


if __name__ == "__main__":
  googletest.main()
