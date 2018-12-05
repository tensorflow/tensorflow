# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpointable object SavedModel loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.training.checkpointable import tracking


class LoadTest(test.TestCase):

  def test_structure_import(self):
    root = tracking.Checkpointable()
    root.f = def_function.function(
        lambda x: 2. * x,
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    root.dep_one = tracking.Checkpointable()
    root.dep_two = tracking.Checkpointable()
    root.dep_two.dep = tracking.Checkpointable()
    root.dep_three = root.dep_two.dep
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    imported = load.load(save_dir)
    self.assertIs(imported.dep_three, imported.dep_two.dep)
    self.assertIsNot(imported.dep_one, imported.dep_two)


if __name__ == "__main__":
  test.main()
