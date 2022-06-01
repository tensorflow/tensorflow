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
import os

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest


class InterfaceTests(test.TestCase):

  def testMultipleAssignment(self):
    root = autotrackable.AutoTrackable()
    root.leaf = autotrackable.AutoTrackable()
    root.leaf = root.leaf
    duplicate_name_dep = autotrackable.AutoTrackable()
    with self.assertRaisesRegex(ValueError, "already declared"):
      root._track_trackable(duplicate_name_dep, name="leaf")
    # No error; we're overriding __setattr__, so we can't really stop people
    # from doing this while maintaining backward compatibility.
    root.leaf = duplicate_name_dep
    root._track_trackable(duplicate_name_dep, name="leaf", overwrite=True)
    self.assertIs(duplicate_name_dep, root._lookup_dependency("leaf"))
    self.assertIs(duplicate_name_dep, root._trackable_children()["leaf"])

  def testRemoveDependency(self):
    root = autotrackable.AutoTrackable()
    root.a = autotrackable.AutoTrackable()
    self.assertEqual(1, len(root._trackable_children()))
    self.assertEqual(1, len(root._unconditional_checkpoint_dependencies))
    self.assertIs(root.a, root._trackable_children()["a"])
    del root.a
    self.assertFalse(hasattr(root, "a"))
    self.assertEqual(0, len(root._trackable_children()))
    self.assertEqual(0, len(root._unconditional_checkpoint_dependencies))
    root.a = autotrackable.AutoTrackable()
    self.assertEqual(1, len(root._trackable_children()))
    self.assertEqual(1, len(root._unconditional_checkpoint_dependencies))
    self.assertIs(root.a, root._trackable_children()["a"])

  def testListBasic(self):
    a = autotrackable.AutoTrackable()
    b = autotrackable.AutoTrackable()
    a.l = [b]
    c = autotrackable.AutoTrackable()
    a.l.append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertIn("l", a._trackable_children())
    direct_a_dep = a._trackable_children()["l"]
    self.assertIn(b, direct_a_dep)
    self.assertIn(c, direct_a_dep)

  @test_util.run_in_graph_and_eager_modes
  def testMutationDirtiesList(self):
    a = autotrackable.AutoTrackable()
    b = autotrackable.AutoTrackable()
    a.l = [b]
    c = autotrackable.AutoTrackable()
    a.l.insert(0, c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegex(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testOutOfBandEditDirtiesList(self):
    a = autotrackable.AutoTrackable()
    b = autotrackable.AutoTrackable()
    held_reference = [b]
    a.l = held_reference
    c = autotrackable.AutoTrackable()
    held_reference.append(c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegex(ValueError, "The wrapped list was modified"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testNestedLists(self):
    a = autotrackable.AutoTrackable()
    a.l = []
    b = autotrackable.AutoTrackable()
    a.l.append([b])
    c = autotrackable.AutoTrackable()
    a.l[0].append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    a.l[0].append(1)
    d = autotrackable.AutoTrackable()
    a.l[0].append(d)
    a_deps = util.list_objects(a)
    self.assertIn(d, a_deps)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertNotIn(1, a_deps)
    e = autotrackable.AutoTrackable()
    f = autotrackable.AutoTrackable()
    a.l1 = [[], [e]]
    a.l1[0].append(f)
    a_deps = util.list_objects(a)
    self.assertIn(e, a_deps)
    self.assertIn(f, a_deps)
    checkpoint = util.Checkpoint(a=a)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    a.l[0].append(data_structures.NoDependency([]))
    a.l[0][-1].append(5)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    # Dirtying the inner list means the root object is unsaveable.
    a.l[0][1] = 2
    with self.assertRaisesRegex(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testAssertions(self):
    a = autotrackable.AutoTrackable()
    a.l = {"k": [np.zeros([2, 2])]}
    self.assertAllEqual(nest.flatten({"k": [np.zeros([2, 2])]}),
                        nest.flatten(a.l))
    self.assertAllClose({"k": [np.zeros([2, 2])]}, a.l)
    nest.map_structure(self.assertAllClose, a.l, {"k": [np.zeros([2, 2])]})
    a.tensors = {"k": [array_ops.ones([2, 2]), array_ops.zeros([3, 3])]}
    self.assertAllClose({"k": [np.ones([2, 2]), np.zeros([3, 3])]},
                        self.evaluate(a.tensors))


if __name__ == "__main__":
  test.main()
