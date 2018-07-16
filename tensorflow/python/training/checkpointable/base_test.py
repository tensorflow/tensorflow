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
from tensorflow.python.training.checkpointable import base


class InterfaceTests(test.TestCase):

  def testOverwrite(self):
    root = base.CheckpointableBase()
    leaf = base.CheckpointableBase()
    root._track_checkpointable(leaf, name="leaf")
    (current_name, current_dependency), = root._checkpoint_dependencies
    self.assertIs(leaf, current_dependency)
    self.assertEqual("leaf", current_name)
    duplicate_name_dep = base.CheckpointableBase()
    with self.assertRaises(ValueError):
      root._track_checkpointable(duplicate_name_dep, name="leaf")
    root._track_checkpointable(duplicate_name_dep, name="leaf", overwrite=True)
    (current_name, current_dependency), = root._checkpoint_dependencies
    self.assertIs(duplicate_name_dep, current_dependency)
    self.assertEqual("leaf", current_name)

if __name__ == "__main__":
  test.main()
