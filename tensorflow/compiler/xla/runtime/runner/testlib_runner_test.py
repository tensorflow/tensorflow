# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for testlib_runner."""

from tensorflow.compiler.xla.runtime.runner import runner
from tensorflow.python.platform import googletest


class TestlibRunnerTest(googletest.TestCase):

  def testAdd(self):
    module = """
      func.func @add(%arg0: i32) -> i32 {
        %0 = arith.constant 42 : i32
        %1 = arith.addi %arg0, %0 : i32
        return %1 : i32
      }"""

    r = runner.Runner('testlib_runner')
    [res] = r.execute(module, 'add', [42])
    self.assertEqual(res, 84)

if __name__ == '__main__':
  googletest.main()
