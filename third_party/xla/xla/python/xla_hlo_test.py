# Copyright 2026 The OpenXLA Authors
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
"""Tests for HLO Module in Python XLA client."""

import functools
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from xla.python import _hlo
from xla.python import xla_client

ops = xla_client.ops


def TestFactory(xla_backend, cloud_tpu=False, pathways=False):
  tests = []

  class ComputationTest(parameterized.TestCase):

    def setUp(self):
      super().setUp()
      self.backend = xla_backend()
      self.executable_devices = xla_client.DeviceList(
          tuple(self.backend.local_devices()[:1])
      )

  class TestHloModule(ComputationTest):

    def setUp(self):
      super(TestHloModule, self).setUp()
      self.backend = xla_backend()
      self.executable_devices = xla_client.DeviceList(
          tuple(self.backend.local_devices()[:1])
      )

    def ExampleComputation(self):
      hlo_string = R"""
HloModule acomputation

ENTRY %main {
  %p0 = f32[] parameter(0)
  %p1 = f32[4]{0} parameter(1)
  %p0_broadcast = f32[4]{0} broadcast(%p0), dimensions={}
  %x = f32[4]{0} multiply(%p0_broadcast, %p1)
  ROOT %add = f32[4]{0} add(%x, %x)
}
"""
      return _hlo.hlo_module_from_text(hlo_string)

    def AsyncComputation(self):
      hlo_string = R"""
HloModule async_module

%async_wrapped {
  %p = f32[4]{0} parameter(0)
  ROOT %add = f32[4]{0} add(%p, %p)
}

ENTRY %main {
  %p0 = f32[4]{0} parameter(0)
  %p1 = f32[4,4]{1,0} parameter(1)

  %async-start = ((f32[4]{0}), f32[4]{0}, u32[]) async-start(%p0), calls=%async_wrapped
  %async-done = f32[4]{0} async-done(((f32[4]{0}), f32[4]{0}, u32[]) %async-start), calls=%async_wrapped

  %dot = f32[4]{0} dot(%p0, %p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}

  ROOT %tuple = (f32[4]{0}, f32[4]{0}) tuple(%async-done, %dot)
}
"""
      return _hlo.hlo_module_from_text(hlo_string)

    def ScheduledAsyncComputation(self, dot_position="after"):
      dot_op = (
          "%dot = f32[4]{0} dot(%p0, %p1), lhs_contracting_dims={0},"
          " rhs_contracting_dims={0}"
      )
      start_op = (
          "%async-start = ((f32[4]{0}), f32[4]{0}, u32[]) async-start(%p0),"
          " calls=%async_wrapped"
      )
      done_op = (
          "%async-done = f32[4]{0} async-done(((f32[4]{0}), f32[4]{0}, u32[])"
          " %async-start), calls=%async_wrapped"
      )

      if dot_position == "before":
        hlo_ops = [dot_op, start_op, done_op]
      elif dot_position == "overlapped":
        hlo_ops = [start_op, dot_op, done_op]
      elif dot_position == "after":
        hlo_ops = [start_op, done_op, dot_op]
      else:
        raise ValueError(f"Invalid dot_position: {dot_position}")

      ops_str = "\n  ".join(hlo_ops)

      hlo_string = (
          """
HloModule async_module, is_scheduled=true

%async_wrapped {
  %p = f32[4]{0} parameter(0)
  ROOT %add = f32[4]{0} add(%p, %p)
}

ENTRY %main {
  %p0 = f32[4]{0} parameter(0)
  %p1 = f32[4,4]{1,0} parameter(1)

"""
          + ops_str
          + """

  ROOT %tuple = (f32[4]{0}, f32[4]{0}) tuple(%async-done, %dot)
}
"""
      )
      return _hlo.hlo_module_from_text(hlo_string)

    def AsyncDone(self, async_start):
      """Returns the async-done instruction for the given async-start."""
      users = async_start.users()
      self.assertLen(users, 1)
      user = users[0]
      self.assertEqual(user.opcode, _hlo.HloOpcode.kAsyncDone)
      return user

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloModuleComputations(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)
      names = [c.name for c in computations]
      self.assertTrue(any(n.startswith("main") for n in names))

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloModuleMakeNonfusionComputations(self):
      hlo_string = R"""
HloModule fusion_module

%fused_computation {
  %p0 = f32[4]{0} parameter(0)
  ROOT %add = f32[4]{0} add(%p0, %p0)
}

ENTRY %main {
  %p0 = f32[4]{0} parameter(0)
  ROOT %fusion = f32[4]{0} fusion(%p0), kind=kLoop, calls=%fused_computation
}
"""
      module = _hlo.hlo_module_from_text(hlo_string)
      self.assertIsNotNone(module)

      computations = module.computations()
      self.assertLen(computations, 2)

      nonfusion_computations = module.make_nonfusion_computations()
      self.assertLen(nonfusion_computations, 1)
      self.assertTrue(nonfusion_computations[0].name.startswith("main"))

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloComputationInstructions(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)

      for comp in computations:
        instructions = comp.instructions()
        self.assertIsInstance(instructions, list)
        self.assertNotEmpty(instructions)
        for inst in instructions:
          self.assertIsInstance(inst, _hlo.HloInstruction)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloComputationReplaceInstruction(self):
      hlo_string = R"""
HloModule copy_module

ENTRY %main {
  %p0 = f32[4]{0} parameter(0)
  %copy = f32[4]{0} copy(%p0)
  ROOT %add = f32[4]{0} add(%copy, %copy)
}
"""
      module = _hlo.hlo_module_from_text(hlo_string)
      comp = module.computations()[0]

      instructions = comp.instructions()
      p0 = [i for i in instructions if i.opcode == _hlo.HloOpcode.kParameter][0]
      copy = [i for i in instructions if i.opcode == _hlo.HloOpcode.kCopy][0]
      add = [i for i in instructions if i.opcode == _hlo.HloOpcode.kAdd][0]

      comp.replace_instruction(copy, p0)

      operands = add.operands()
      self.assertEqual(operands[0], p0)
      self.assertEqual(operands[1], p0)

      # Verify copy is removed
      instructions = comp.instructions()
      opcodes = [i.opcode for i in instructions]
      self.assertNotIn(_hlo.HloOpcode.kCopy, opcodes)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloInstructionOpcode(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)

      for comp in computations:
        instructions = comp.instructions()
        self.assertNotEmpty(instructions)
        opcodes = [i.opcode for i in instructions]
        self.assertNotEmpty(opcodes)
        for opcode in opcodes:
          self.assertIsInstance(opcode, _hlo.HloOpcode)

        # Check if we can find expected opcodes in the main computation.
        if comp.name.startswith("main"):
          opcode_names = [o.name for o in opcodes]
          self.assertIn("kParameter", opcode_names)
          # We expect either kMultiply and kAdd (no fusion) OR kFusion.
          has_multiply_and_add = (
              "kMultiply" in opcode_names and "kAdd" in opcode_names
          )
          has_fusion = "kFusion" in opcode_names
          self.assertTrue(has_multiply_and_add or has_fusion)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloInstructionUsers(self):
      module = self.ExampleComputation()
      logging.info("HLO module: %s", module.to_string())
      computations = module.computations()
      self.assertNotEmpty(computations)

      for comp in computations:
        if comp.name.startswith("main"):
          instructions = comp.instructions()
          params = [
              i for i in instructions if i.opcode == _hlo.HloOpcode.kParameter
          ]
          self.assertNotEmpty(params)

          for param in params:
            users = param.users()
            self.assertNotEmpty(users)
            for user in users:
              self.assertIsInstance(user, _hlo.HloInstruction)
              logging.info("User of %s: %s", param.name, user.name)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloInstructionOperands(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)

      for comp in computations:
        instructions = comp.instructions()
        self.assertNotEmpty(instructions)
        for inst in instructions:
          operands = inst.operands()
          self.assertIsInstance(operands, list)
          for operand in operands:
            self.assertIsInstance(operand, _hlo.HloInstruction)
            logging.info("Operand of %s: %s", inst.name, operand.name)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloInstructionName(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)
      for comp in computations:
        for inst in comp.instructions():
          self.assertIsInstance(inst.name, str)
          self.assertNotEmpty(inst.name)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloInstructionToString(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)
      for comp in computations:
        for inst in comp.instructions():
          self.assertIsInstance(inst.to_string(), str)
          self.assertNotEmpty(inst.to_string())

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloInstructionHash(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)
      for comp in computations:
        for inst in comp.instructions():
          self.assertIsInstance(hash(inst), int)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloInstructionEq(self):
      module = self.ExampleComputation()
      computations = module.computations()
      self.assertNotEmpty(computations)
      for comp in computations:
        if comp.name.startswith("main"):
          instructions = comp.instructions()
          add_ops = [i for i in instructions if i.opcode == _hlo.HloOpcode.kAdd]
          if add_ops:
            add_op = add_ops[0]
            operands = add_op.operands()
            self.assertLen(operands, 2)
            self.assertEqual(operands[0], operands[1])
            self.assertNotEqual(operands[0], add_op)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testAsyncWrappedRootOnAsyncDone(self):
      module = self.AsyncComputation()
      main_comp = [
          c for c in module.computations() if c.name.startswith("main")
      ][0]
      instructions = main_comp.instructions()
      start_inst = [
          i for i in instructions if i.opcode == _hlo.HloOpcode.kAsyncStart
      ][0]
      done_inst = self.AsyncDone(start_inst)

      start_root = start_inst.async_wrapped_root()
      self.assertIsNotNone(start_root)

      done_root = done_inst.async_wrapped_root()
      self.assertIsNotNone(done_root)
      self.assertEqual(start_root, done_root)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloScheduleToString(self):
      module = self.ScheduledAsyncComputation()
      schedule = module.schedule()
      self.assertIsNotNone(schedule)
      schedule_str = schedule.to_string()
      self.assertIsInstance(schedule_str, str)
      self.assertNotEmpty(schedule_str)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloScheduleSequence(self):
      module = self.ScheduledAsyncComputation()
      schedule = module.schedule()
      self.assertIsNotNone(schedule)
      for comp in module.computations():
        if comp.name.startswith("main"):
          seq = schedule.sequence(comp)
          self.assertIsInstance(seq, list)
          self.assertNotEmpty(seq)
          for inst in seq:
            self.assertIsInstance(inst, _hlo.HloInstruction)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloScheduleSetSequence(self):
      module = self.ScheduledAsyncComputation()
      schedule = module.schedule()
      self.assertIsNotNone(schedule)
      for comp in module.computations():
        if comp.name.startswith("main"):
          seq = schedule.sequence(comp)
          schedule.set_sequence(comp, seq)
          new_seq = schedule.sequence(comp)
          self.assertEqual(seq, new_seq)

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloScheduleUpdate(self):
      module = self.ScheduledAsyncComputation()
      schedule = module.schedule()
      self.assertIsNotNone(schedule)
      schedule.update()

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloScheduleVerify(self):
      module = self.ScheduledAsyncComputation()
      schedule = module.schedule()
      self.assertIsNotNone(schedule)
      schedule.verify()

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testScheduleHloComputation(self):
      module = self.AsyncComputation()
      self.assertIsNone(module.schedule())

      schedule = module.create_empty_schedule()
      self.assertIsNotNone(schedule)

      # Gather computations to schedule.
      to_schedule = []
      for comp in module.computations():
        if comp.name.startswith("main"):
          to_schedule.append(comp)

      # Schedule computations.
      for comp in to_schedule:
        new_seq = []
        worklist = []
        visited = set()

        # Initialize worklist with parameters.
        for inst in comp.instructions():
          if inst.opcode == _hlo.HloOpcode.kParameter:
            worklist.append(inst)
            visited.add(inst)

        # Process worklist until empty, visiting graph nodes in post-order.
        while worklist:
          # Apply scheduling policy here based instructions in worklist.
          # For this test, we just pop the first instruction and add it to the
          # schedule.
          inst = worklist.pop(0)
          new_seq.append(inst)

          # Add any users of the scheduled instruction, if they are now ready.
          for user in inst.users():
            if user not in visited and all(
                [op in visited for op in user.operands()]
            ):
              worklist.append(user)
              visited.add(user)

        # Update the schedule for the computation.
        schedule.set_sequence(comp, new_seq)

      schedule.update()
      schedule.verify()
      module.set_schedule(schedule)
      self.assertIsNotNone(module.schedule())

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    @parameterized.parameters("before", "overlapped", "after")
    def testHloScheduleUpdateAsyncOverlap(self, dot_position):
      module = self.ScheduledAsyncComputation(dot_position)
      schedule = module.schedule()
      logging.info("ENTRY schedule: %s", schedule.to_string())
      self.assertIsNotNone(schedule)

      for comp in module.computations():
        if comp.name.startswith("main"):
          seq = schedule.sequence(comp)
          async_starts = []
          dot_ops = []
          async_dones = {}
          for i, inst in enumerate(seq):
            self.assertIsInstance(inst, _hlo.HloInstruction)
            if inst.opcode == _hlo.HloOpcode.kAsyncStart:
              async_starts.append((i, inst))
            elif inst.opcode == _hlo.HloOpcode.kDot:
              dot_ops.append((i, inst))
            elif inst.opcode == _hlo.HloOpcode.kAsyncDone:
              async_dones[inst] = i
          self.assertLen(async_starts, 1)
          self.assertLen(dot_ops, 1)

          # Get matched instructions and their associated schedule positions.
          start_idx, start_inst = async_starts[0]
          dot_idx, dot_inst = dot_ops[0]
          done_inst = self.AsyncDone(start_inst)
          done_idx = async_dones[done_inst]

          # Create mapping to desired partial order in new schedule.
          special_indices = sorted([start_idx, dot_idx, done_idx])
          new_order = [start_inst, dot_inst, done_inst]
          new_mapping = dict(zip(special_indices, new_order))

          new_seq = []
          for j, inst in enumerate(seq):
            if j in new_mapping:
              new_seq.append(new_mapping[j])
            else:
              new_seq.append(inst)

          # Verify the new relative order
          self.assertLess(new_seq.index(start_inst), new_seq.index(dot_inst))
          self.assertLess(new_seq.index(dot_inst), new_seq.index(done_inst))

          # Apply the new sequence to the schedule, update and verify.
          schedule.set_sequence(comp, new_seq)
          schedule.update()
          schedule.verify()
          module.set_schedule(schedule)

          # Verify that the updated schedule has the desired partial order.
          updated_schedule = module.schedule()
          updated_seq = updated_schedule.sequence(comp)
          self.assertEqual(len(updated_seq), len(new_seq))
          self.assertEqual(updated_seq[new_seq.index(start_inst)], start_inst)
          self.assertEqual(updated_seq[new_seq.index(dot_inst)], dot_inst)
          logging.info("EXIT schedule: %s", updated_schedule.to_string())

    @unittest.skipIf(cloud_tpu or pathways, "not implemented")
    def testHloModuleWith4AsyncPairs(self):
      hlo_string = R"""
HloModule async_4_pairs

%async_wrapped.1 {
  %p = f32[4]{0} parameter(0)
  ROOT %all-gather = f32[8]{0} all-gather(%p), replica_groups={}, dimensions={0}
}

%async_wrapped.2 {
  %p = f32[8]{0} parameter(0)
  ROOT %add = f32[8]{0} add(%p, %p)
}

%async_wrapped.3 {
  %p = f32[8]{0} parameter(0)
  ROOT %all-gather = f32[16]{0} all-gather(%p), replica_groups={}, dimensions={0}
}

%async_wrapped.4 {
  %p = f32[16]{0} parameter(0)
  ROOT %add = f32[16]{0} add(%p, %p)
}

ENTRY %main {
  %p0 = f32[4]{0} parameter(0)
  %p1 = f32[8]{0} parameter(1)
  %p2 = f32[8]{0} parameter(2)
  %p3 = f32[16]{0} parameter(3)

  %async-start.1 = ((f32[4]{0}), f32[8]{0}, u32[]) async-start(%p0), calls=%async_wrapped.1
  %async-done.1 = f32[8]{0} async-done(((f32[4]{0}), f32[8]{0}, u32[]) %async-start.1), calls=%async_wrapped.1

  %async-start.2 = ((f32[8]{0}), f32[8]{0}, u32[]) async-start(%p1), calls=%async_wrapped.2
  %async-done.2 = f32[8]{0} async-done(((f32[8]{0}), f32[8]{0}, u32[]) %async-start.2), calls=%async_wrapped.2

  %async-start.3 = ((f32[8]{0}), f32[16]{0}, u32[]) async-start(%p2), calls=%async_wrapped.3
  %async-done.3 = f32[16]{0} async-done(((f32[8]{0}), f32[16]{0}, u32[]) %async-start.3), calls=%async_wrapped.3

  %async-start.4 = ((f32[16]{0}), f32[16]{0}, u32[]) async-start(%p3), calls=%async_wrapped.4
  %async-done.4 = f32[16]{0} async-done(((f32[16]{0}), f32[16]{0}, u32[]) %async-start.4), calls=%async_wrapped.4

  ROOT %tuple = (f32[8]{0}, f32[8]{0}, f32[16]{0}, f32[16]{0}) tuple(%async-done.1, %async-done.2, %async-done.3, %async-done.4)
}
"""
      module = _hlo.hlo_module_from_text(hlo_string)
      self.assertIsNotNone(module)
      logging.info("ENTRY module: %s", module.to_string())

      computations = module.computations()
      self.assertNotEmpty(computations)

      main_comp = [c for c in computations if c.name.startswith("main")][0]
      instructions = main_comp.instructions()

      comm_starts = []
      compute_starts = []
      for i in instructions:
        if i.opcode == _hlo.HloOpcode.kAsyncStart:
          wrapped_root = i.async_wrapped_root()
          if wrapped_root.opcode == _hlo.HloOpcode.kAllGather:
            comm_starts.append(i)
          else:
            compute_starts.append(i)

      self.assertLen(comm_starts, 2)
      self.assertLen(compute_starts, 2)

      # Desired schedule:
      desired_order = [
          comm_starts[0],  # Comm1-Start
          compute_starts[0],  # Compute1-Start
          self.AsyncDone(comm_starts[0]),  # Comm1-Done
          comm_starts[1],  # Comm2-Start
          compute_starts[1],  # Compute2-Start
          self.AsyncDone(compute_starts[0]),  # Compute1-Done
          self.AsyncDone(comm_starts[1]),  # Comm2-Done
          self.AsyncDone(compute_starts[1]),  # Compute2-Done
      ]

      parameters = [
          i for i in instructions if i.opcode == _hlo.HloOpcode.kParameter
      ]
      root_tuple = [
          i for i in instructions if i.opcode == _hlo.HloOpcode.kTuple
      ][0]

      new_seq = parameters + desired_order + [root_tuple]

      schedule = module.create_empty_schedule()
      schedule.set_sequence(main_comp, new_seq)
      schedule.update()
      schedule.verify()
      module.set_schedule(schedule)

      # Verify that the updated schedule has the desired sequence.
      updated_schedule = module.schedule()
      self.assertIsNotNone(updated_schedule)
      updated_seq = updated_schedule.sequence(main_comp)
      self.assertEqual(updated_seq, new_seq)
      logging.info("EXIT module: %s", module.to_string())

  tests.append(TestHloModule)
  return tests


def InstantiateTests(globals_dict, backend_fn, test_prefix="", **kw):
  backend_fn = functools.lru_cache(maxsize=None)(backend_fn)
  for klass in TestFactory(backend_fn, **kw):
    test = type(test_prefix + klass.__name__, (klass,), {})
    test.__qualname__ = test.__name__
    globals_dict[test.__name__] = test


if __name__ == "__main__":
  InstantiateTests(
      globals(), functools.partial(xla_client.make_cpu_client, num_devices=2)
  )
  absltest.main()
