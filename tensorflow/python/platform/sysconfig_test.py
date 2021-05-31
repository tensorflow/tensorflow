# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import re

from tensorflow.python.platform import googletest
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import test


class SysconfigTest(googletest.TestCase):

  def test_get_build_info_works(self):
    build_info = sysconfig.get_build_info()
    self.assertIsInstance(build_info, dict)

  def test_rocm_cuda_info_matches(self):
    build_info = sysconfig.get_build_info()
    self.assertEqual(build_info["is_rocm_build"], test.is_built_with_rocm())
    self.assertEqual(build_info["is_cuda_build"], test.is_built_with_cuda())

  def test_compile_flags(self):
    # Must contain an include directory, and define _GLIBCXX_USE_CXX11_ABI,
    # EIGEN_MAX_ALIGN_BYTES
    compile_flags = sysconfig.get_compile_flags()

    def list_contains(items, regex_str):
      regex = re.compile(regex_str)
      return any(regex.match(item) for item in items)

    self.assertTrue(list_contains(compile_flags, ".*/include"))
    self.assertTrue(list_contains(compile_flags, ".*_GLIBCXX_USE_CXX11_ABI.*"))
    self.assertTrue(list_contains(compile_flags, ".*EIGEN_MAX_ALIGN_BYTES.*"))


if __name__ == "__main__":
  googletest.main()
