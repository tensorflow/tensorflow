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

import os

import numpy
import six

from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training.checkpointable import base
from tensorflow.python.training.checkpointable import data_structures
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.training.checkpointable import util
from tensorflow.python.util import nest


class InterfaceTests(test.TestCase):

  def testMultipleAssignment(self):
    root = tracking.Checkpointable()
    root.leaf = tracking.Checkpointable()
    root.leaf = root.leaf
    duplicate_name_dep = tracking.Checkpointable()
    with self.assertRaisesRegexp(ValueError, "already declared"):
      root._track_checkpointable(duplicate_name_dep, name="leaf")
    # No error; we're overriding __setattr__, so we can't really stop people
    # from doing this while maintaining backward compatibility.
    root.leaf = duplicate_name_dep
    root._track_checkpointable(duplicate_name_dep, name="leaf", overwrite=True)
    self.assertIs(duplicate_name_dep, root._lookup_dependency("leaf"))
    (_, dep_object), = root._checkpoint_dependencies
    self.assertIs(duplicate_name_dep, dep_object)

  def testNoDependency(self):
    root = tracking.Checkpointable()
    hasdep = tracking.Checkpointable()
    root.hasdep = hasdep
    nodep = tracking.Checkpointable()
    root.nodep = data_structures.NoDependency(nodep)
    self.assertEqual(1, len(root._checkpoint_dependencies))
    self.assertIs(root._checkpoint_dependencies[0].ref, root.hasdep)
    self.assertIs(root.hasdep, hasdep)
    self.assertIs(root.nodep, nodep)

    class NoDependencyModel(training.Model):

      @base.no_automatic_dependency_tracking
      def __init__(self):
        super(NoDependencyModel, self).__init__()
        self.a = []
        self.b = tracking.Checkpointable()

    nodeps = NoDependencyModel()
    self.assertEqual([nodeps], util.list_objects(nodeps))

  def testListBasic(self):
    a = tracking.Checkpointable()
    b = tracking.Checkpointable()
    a.l = [b]
    c = tracking.Checkpointable()
    a.l.append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    direct_a_dep, = a._checkpoint_dependencies
    self.assertEqual("l", direct_a_dep.name)
    self.assertIn(b, direct_a_dep.ref)
    self.assertIn(c, direct_a_dep.ref)

  @test_util.run_in_graph_and_eager_modes
  def testMutationDirtiesList(self):
    a = tracking.Checkpointable()
    b = tracking.Checkpointable()
    a.l = [b]
    c = tracking.Checkpointable()
    a.l.insert(0, c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegexp(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testOutOfBandEditDirtiesList(self):
    a = tracking.Checkpointable()
    b = tracking.Checkpointable()
    held_reference = [b]
    a.l = held_reference
    c = tracking.Checkpointable()
    held_reference.append(c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegexp(ValueError, "The wrapped list was modified"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testNestedLists(self):
    a = tracking.Checkpointable()
    a.l = []
    b = tracking.Checkpointable()
    a.l.append([b])
    c = tracking.Checkpointable()
    a.l[0].append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    a.l[0].append(1)
    d = tracking.Checkpointable()
    a.l[0].append(d)
    a_deps = util.list_objects(a)
    self.assertIn(d, a_deps)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertNotIn(1, a_deps)
    e = tracking.Checkpointable()
    f = tracking.Checkpointable()
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
    with self.assertRaisesRegexp(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testDictionariesBasic(self):
    a = training.Model()
    b = training.Model()
    a.attribute = {"b": b}
    c = training.Model()
    a.attribute["c"] = []
    a.attribute["c"].append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertIs(b, a.attribute["b"])
    six.assertCountEqual(
        self,
        ["b", "c"],
        [dep.name for dep in a.attribute._checkpoint_dependencies])
    self.assertEqual([b, c], a.layers)
    self.assertEqual([b, c], a.attribute.layers)
    self.assertEqual([c], a.attribute["c"].layers)
    checkpoint = util.Checkpoint(a=a)
    save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    with self.test_session():
      checkpoint.restore(save_path).assert_consumed().initialize_or_restore()

  @test_util.run_in_graph_and_eager_modes
  def testNoDepList(self):
    a = training.Model()
    a.l1 = data_structures.NoDependency([])
    a.l1.insert(1, 0)
    self.assertTrue(isinstance(a.l1, list))
    checkpoint = util.Checkpoint(a=a)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    a.l2 = []
    a.l2.insert(1, 0)
    with self.assertRaisesRegexp(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testAssertions(self):
    a = tracking.Checkpointable()
    a.l = {"k": [numpy.zeros([2, 2])]}
    self.assertAllEqual(nest.flatten({"k": [numpy.zeros([2, 2])]}),
                        nest.flatten(a.l))
    self.assertAllClose({"k": [numpy.zeros([2, 2])]}, a.l)
    nest.map_structure(self.assertAllClose, a.l, {"k": [numpy.zeros([2, 2])]})
    a.tensors = {"k": [array_ops.ones([2, 2]), array_ops.zeros([3, 3])]}
    self.assertAllClose({"k": [numpy.ones([2, 2]), numpy.zeros([3, 3])]},
                        self.evaluate(a.tensors))

if __name__ == "__main__":
  test.main()
