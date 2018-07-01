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
from tensorflow.python.training.checkpointable import tracking


class InterfaceTests(test.TestCase):

  def testMultipleAssignment(self):
    root = tracking.Checkpointable()
    root.leaf = tracking.Checkpointable()
    root.leaf = root.leaf
    duplicate_name_dep = tracking.Checkpointable()
    with self.assertRaises(ValueError):
      root._track_checkpointable(duplicate_name_dep, name="leaf")
    # No error; we're overriding __setattr__, so we can't really stop people
    # from doing this while maintaining backward compatibility.
    root.leaf = duplicate_name_dep
    root._track_checkpointable(duplicate_name_dep, name="leaf", overwrite=True)

  def testNoDependency(self):
    root = tracking.Checkpointable()
    hasdep = tracking.Checkpointable()
    root.hasdep = hasdep
    nodep = tracking.Checkpointable()
    root.nodep = tracking.NoDependency(nodep)
    self.assertEqual(1, len(root._checkpoint_dependencies))
    self.assertIs(root._checkpoint_dependencies[0].ref, root.hasdep)
    self.assertIs(root.hasdep, hasdep)
    self.assertIs(root.nodep, nodep)

if __name__ == "__main__":
  test.main()
