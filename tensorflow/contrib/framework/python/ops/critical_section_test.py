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
"""critical section tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import critical_section_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test
# TODO(ebrevdo): Re-enable once CriticalSection is in core.
# from tensorflow.python.training import saver as saver_lib


class CriticalSectionTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testCreateCriticalSection(self):
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn(a, b):
      c = v.read_value()
      with ops.control_dependencies([c]):
        nv = v.assign_add(a * b)
        with ops.control_dependencies([nv]):
          return array_ops.identity(c)

    num_concurrent = 100
    r = [cs.execute(fn, 1.0, 2.0) for _ in range(num_concurrent)]
    self.evaluate(v.initializer)
    r_value = self.evaluate(r)
    self.assertAllClose([2.0 * i for i in range(num_concurrent)],
                        sorted(r_value))

  @test_util.run_in_graph_and_eager_modes()
  def testCriticalSectionWithControlFlow(self):
    for outer_cond in [False, True]:
      for inner_cond in [False, True]:
        cs = critical_section_ops.CriticalSection(shared_name="cs")
        v = resource_variable_ops.ResourceVariable(0.0, name="v")
        num_concurrent = 100

        # pylint: disable=cell-var-from-loop
        def fn(a, b):
          c = v.read_value()
          def true_fn():
            with ops.control_dependencies([c]):
              nv = v.assign_add(a * b)
              with ops.control_dependencies([nv]):
                return array_ops.identity(c)
          return control_flow_ops.cond(
              array_ops.identity(inner_cond), true_fn, lambda: c)

        def execute():
          return cs.execute(fn, 1.0, 2.0)

        r = [
            control_flow_ops.cond(array_ops.identity(outer_cond),
                                  execute,
                                  v.read_value)
            for _ in range(num_concurrent)
        ]
        # pylint: enable=cell-var-from-loop

        self.evaluate(v.initializer)
        r_value = self.evaluate(r)
        if inner_cond and outer_cond:
          self.assertAllClose([2.0 * i for i in range(num_concurrent)],
                              sorted(r_value))
        else:
          self.assertAllClose([0] * num_concurrent, r_value)

  def testCriticalSectionInParallelDoesntDeadlockOnError(self):
    # No eager mode execution of this test because eager does not
    # run fn() in parallel, which is where the deadlock could
    # potentially occur (in graph mode).
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn(i):
      error = control_flow_ops.Assert((i % 2) == 1, ["Error"])
      with ops.control_dependencies([error]):
        return v.read_value()
    num_concurrent = 2
    r = [cs.execute(fn, i) for i in range(num_concurrent)]
    self.evaluate(v.initializer)
    for _ in range(100):
      with self.assertRaisesOpError("Error"):
        self.evaluate(r)

  @test_util.run_in_graph_and_eager_modes()
  def testCreateCriticalSectionFnReturnsOp(self):
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn_return_op(a, b):
      c = v.read_value()
      with ops.control_dependencies([c]):
        nv = v.assign_add(a * b)
        with ops.control_dependencies([nv]):
          return control_flow_ops.no_op()

    num_concurrent = 100
    r = [cs.execute(fn_return_op, 1.0, 2.0) for _ in range(num_concurrent)]
    self.evaluate(v.initializer)
    self.evaluate(r)
    final_v = self.evaluate(v)
    self.assertAllClose(2.0 * num_concurrent, final_v)

  def testCollection(self):
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    self.assertIn(
        cs, ops.get_collection(critical_section_ops.CRITICAL_SECTIONS))
    execute = cs.execute(lambda x: x + 1, 1.0, name="my_execute")
    execute_op = [
        x for x in execute.graph.get_operations()
        if "my_execute" in x.name and "MutexLock" in x.type
    ][0]
    self.assertIn(
        execute_op,
        [signature.op for signature in
         ops.get_collection(critical_section_ops.CRITICAL_SECTION_EXECUTIONS)])

  def testRecursiveCriticalSectionAccessIsIllegal(self):
    # This does not work properly in eager mode.  Eager users will
    # just hit a deadlock if they do this.  But at least it'll be easier
    # to debug.
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    def fn(x):
      return cs.execute(lambda x: x+1, x)
    with self.assertRaisesRegexp(
        ValueError,
        r"attempts to access the CriticalSection in which it would be running"):
      cs.execute(fn, 1.0)

  def testMultipleCSExecutionsRequestSameResource(self):
    cs0 = critical_section_ops.CriticalSection()
    cs1 = critical_section_ops.CriticalSection()
    v = resource_variable_ops.ResourceVariable(0.0, name="v")
    cs0.execute(lambda: v + 1)
    # It's OK for the same CriticalSection to access this resource.
    cs0.execute(lambda: v - 1)
    # It's *not* OK for a different CriticalSection to access it by
    # default.
    with self.assertRaisesRegexp(
        ValueError, "requested exclusive resource access"):
      cs1.execute(lambda: v + 1)
    # It's not even OK if the second call doesn't request exclusive access.
    with self.assertRaisesRegexp(
        ValueError, "requested exclusive resource access"):
      cs1.execute(lambda: v + 1, exclusive_resource_access=False)

    v2 = resource_variable_ops.ResourceVariable(0.0, name="v2")
    cs0.execute(lambda: v2 + 1, exclusive_resource_access=False)
    # It's OK if neither requests exclusive resource access.
    cs1.execute(lambda: v2 + 1, exclusive_resource_access=False)

    # It's not OK if the second request requires exlusive resource
    # access.
    with self.assertRaisesRegexp(
        ValueError, "requested exclusive resource access"):
      cs1.execute(lambda: v2 + 1)

  # TODO(ebrevdo): Re-enable once CriticalSection is in core.
  #
  # def testCriticalSectionAndExecuteOpSaverRoundTrip(self):
  #   cs = critical_section_ops.CriticalSection()
  #   r = cs.execute(lambda x: x + 1, 1.0)
  #   graph = ops.get_default_graph()
  #   meta_graph = saver_lib.export_meta_graph(
  #       graph=graph, collection_list=graph.get_all_collection_keys())
  #   graph_copy = ops.Graph()
  #   with graph_copy.as_default():
  #     _ = saver_lib.import_meta_graph(meta_graph, import_scope="imported")
  #     restored_cs = ops.get_collection(critical_section_ops.CRITICAL_SECTIONS)
  #     restored_exec = ops.get_collection(
  #         critical_section_ops.CRITICAL_SECTION_EXECUTIONS)
  #     self.assertEqual(1, len(restored_cs))
  #     self.assertEqual(1, len(restored_exec))
  #     self.assertEqual(restored_cs[0].name, "imported/%s" % cs.name)
  #     self.assertEqual(restored_exec[0].op.name, "imported/%s" % r.op.name)

  # def testToProto(self):
  #   cs = critical_section_ops.CriticalSection(shared_name="cs")
  #   proto = cs.to_proto()
  #   self.assertEqual(proto.critical_section_name, cs._handle.name)
  #   cs_copy = critical_section_ops.CriticalSection.from_proto(proto)
  #   self.assertEqual(cs_copy._handle, cs._handle)


if __name__ == "__main__":
  test.main()
