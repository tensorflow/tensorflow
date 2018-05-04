# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.platform import test
from tensorflow.python.training import checkpointable


class InterfaceTests(test.TestCase):

  def testMultipleAssignment(self):
    root = checkpointable.Checkpointable()
    root.leaf = checkpointable.Checkpointable()
    root.leaf = root.leaf
    duplicate_name_dep = checkpointable.Checkpointable()
    with self.assertRaises(ValueError):
      root._track_checkpointable(duplicate_name_dep, name="leaf")
    # No error; we're overriding __setattr__, so we can't really stop people
    # from doing this while maintaining backward compatibility.
    root.leaf = duplicate_name_dep
    root._track_checkpointable(duplicate_name_dep, name="leaf", overwrite=True)


if __name__ == "__main__":
  test.main()
