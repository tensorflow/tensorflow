/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/buffer_assignment.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/computation_tracker.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_scheduling.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

namespace {

// DFS visitor that collects the instructions referenced by a computation
// without descending into nested computations, i.e., only from the operands.
class InstructionListVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit InstructionListVisitor(const HloInstruction* root) : root_(root) {}

  Status DefaultAction(HloInstruction* hlo) override {
    // For each instruction, just push it on the list after walking the
    // operands.
    instructions_.push_back(hlo);
    VLOG(0) << "List instruction " << hlo->ToString();
    return Status::OK();
  }

  std::vector<const HloInstruction*> GetInstructions() { return instructions_; }

 private:
  // The instruction root of the computation.
  const HloInstruction* root_;

  // The full set of instructions found (may be duplicates, e.g., kParameter).
  std::vector<const HloInstruction*> instructions_;

  TF_DISALLOW_COPY_AND_ASSIGN(InstructionListVisitor);
};

const std::vector<const HloInstruction*> GetInstructions(HloInstruction* root) {
  InstructionListVisitor main_list(root);
  TF_CHECK_OK(root->Accept(&main_list));
  return main_list.GetInstructions();
}

class BufferAssignmentTest : public HloTestBase {
 protected:
  BufferAssignmentTest() : computation_tracker_() {}
  ~BufferAssignmentTest() override {}

  std::unique_ptr<BufferAssignment> RunBufferAssignment(HloModule* module,
                                                        int64 alignment = 1) {
    return BufferAssigner::Run(
               module, xla::MakeUnique<DependencyHloOrdering>(module),
               backend().compiler()->BufferSizeBytesFunction(),
               [alignment](LogicalBuffer::Color) { return alignment; })
        .ConsumeValueOrDie();
  }

  std::unique_ptr<BufferAssignment> RunColoredBufferAssignment(
      HloModule* module, BufferLiveness::Colorer colorer, int64 alignment = 1) {
    return BufferAssigner::Run(
               module, xla::MakeUnique<DependencyHloOrdering>(module),
               backend().compiler()->BufferSizeBytesFunction(),
               [alignment](LogicalBuffer::Color) { return alignment; }, false,
               std::move(colorer))
        .ConsumeValueOrDie();
  }

  // Builds an x+1.0 computation to use in a Map.
  std::unique_ptr<HloComputation> BuildMapComputationPlus1(const string& name) {
    auto builder = HloComputation::Builder(name);
    auto param =
        builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
    auto value = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
    builder.AddInstruction(
        HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, param, value));
    return builder.Build();
  }

  // Builds a simple compare-to-limit (x < 4) computation for a While.
  //
  // condition:
  //   const4[s32] -----------------------------------\
  //                                                   \
  //   param[(s32,f32[4])] --- get-tuple-element[0] --- less-than
  //
  std::unique_ptr<HloComputation> BuildWhileConditionComputation(
      const string& name) {
    auto builder = HloComputation::Builder(name);
    auto const4 = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int>(4)));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v4_, "x"));
    auto index = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const4->shape(), param, 0));
    builder.AddInstruction(
        HloInstruction::CreateBinary(r0f32_, HloOpcode::kLt, index, const4));
    return builder.Build();
  }

  // Builds a simple body computation for a While.
  //
  // body:
  //   constv[f32[4]] --------------------------------------\
  //                                                         \
  //                           /--- get-tuple-elementv[1] --- addv ---\
  //   param[(s32,f32[4])] ---|                                    tuple
  //                           \--- get-tuple-elementc[0] --- addc ---/
  //                                                         /
  //   const1[s32] -----------------------------------------/
  //
  std::unique_ptr<HloComputation> BuildWhileBodyComputation(
      const string& name) {
    auto builder = HloComputation::Builder(name);
    auto const1 = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int>(1)));
    auto constv = builder.AddInstruction(HloInstruction::CreateConstant(
        Literal::CreateR1<float>({1.1f, 2.2f, 3.3f, 4.4f})));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v4_, "x"));
    auto indexc = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const1->shape(), param, 0));
    auto addc = builder.AddInstruction(HloInstruction::CreateBinary(
        indexc->shape(), HloOpcode::kAdd, indexc, const1));
    auto indexv = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(constv->shape(), param, 1));
    auto addv = builder.AddInstruction(HloInstruction::CreateBinary(
        constv->shape(), HloOpcode::kAdd, indexv, constv));
    builder.AddInstruction(HloInstruction::CreateTuple({addc, addv}));
    return builder.Build();
  }

  // Verifies that the given instruction hlo has a valid input buffer assigned,
  // i.e., the parameter number matches the op's.
  const BufferAllocation& GetAssignedInputAllocation(
      const BufferAssignment& buffers, HloInstruction* hlo) {
    LOG(INFO) << "Checking input: " << hlo->ToString();
    const BufferAllocation& buffer =
        *buffers.GetUniqueTopLevelSlice(hlo).ConsumeValueOrDie().allocation();
    EXPECT_EQ(hlo->parameter_number(), buffer.parameter_number());
    return buffer;
  }

  // Verifies that the given instruction hlo has a valid output buffer
  // assigned, and returns it.
  const BufferAllocation& GetAssignedOutputAllocation(
      const BufferAssignment& buffers, HloInstruction* hlo) {
    LOG(INFO) << "Checking output: " << hlo->ToString();
    const BufferAllocation& buffer = GetTopLevelAllocation(buffers, hlo);
    return buffer;
  }

  // Returns the allocation for the given instruction.
  const BufferAllocation& GetAllocation(const BufferAssignment& buffers,
                                        const HloInstruction* hlo,
                                        const ShapeIndex& index) {
    return *buffers.GetUniqueSlice(hlo, index).ConsumeValueOrDie().allocation();
  }
  const BufferAllocation& GetTopLevelAllocation(const BufferAssignment& buffers,
                                                const HloInstruction* hlo) {
    return *buffers.GetUniqueTopLevelSlice(hlo)
                .ConsumeValueOrDie()
                .allocation();
  }

  // Verifies that all instructions in the given instruction list except
  // kConstant have assigned buffers, and returns their total size. If min_index
  // and max_index are not nullptr, the minimum and maximum buffer indices in
  // the assignment are written into them.
  int64 ValidateBuffers(const std::vector<const HloInstruction*>& instructions,
                        const BufferAssignment& buffers) {
    // Verifies all instructions have buffers, and gets the index ranges.
    for (const HloInstruction* hlo : instructions) {
      if (!buffers.HasTopLevelAllocation(hlo)) {
        // If `hlo` has no assigned buffer, it is either a constant or a nested
        // parameter.
        EXPECT_TRUE(HloOpcode::kConstant == hlo->opcode() ||
                    HloOpcode::kParameter == hlo->opcode());
        continue;
      }
    }

    // Gets the total size of all buffers assigned.
    int64 total_size = 0;
    for (auto& allocation : buffers.Allocations()) {
      total_size += allocation.size();
    }
    return total_size;
  }

  // Computation tracker for nested computations.
  ComputationTracker computation_tracker_;

  // Shapes for use in the examples.
  Shape s32_ = ShapeUtil::MakeShape(xla::S32, {});
  Shape r0f32_ = ShapeUtil::MakeShape(xla::F32, {});
  Shape f32vec4_ = ShapeUtil::MakeShape(F32, {4});
  Shape f32vec10_ = ShapeUtil::MakeShape(F32, {10});
  Shape f32vec100_ = ShapeUtil::MakeShape(F32, {100});
  Shape f32a100x10_ = ShapeUtil::MakeShape(F32, {100, 10});
  Shape t_s32_f32v4_ = ShapeUtil::MakeTupleShape({s32_, f32vec4_});
  Shape t_s32_f32v10_ = ShapeUtil::MakeTupleShape({s32_, f32vec10_});
};

// Returns true if the buffers assigned to instructions in "a" are distinct
// from the buffers assigned to those in "b" (ie, intersection is empty).
static bool BuffersDistinct(const std::vector<const HloInstruction*>& a,
                            const std::vector<const HloInstruction*>& b,
                            const BufferAssignment& assignment) {
  std::set<BufferAllocation::Slice> a_slices;
  for (const HloInstruction* instruction : a) {
    if (assignment.HasTopLevelAllocation(instruction)) {
      a_slices.insert(
          assignment.GetUniqueTopLevelSlice(instruction).ConsumeValueOrDie());
    }
  }

  for (const HloInstruction* instruction : b) {
    if (assignment.HasTopLevelAllocation(instruction)) {
      if (a_slices.count(assignment.GetUniqueTopLevelSlice(instruction)
                             .ConsumeValueOrDie())) {
        return false;
      }
    }
  }
  return true;
}

// Tests a computation consisting of a single scalar constant node.
TEST_F(BufferAssignmentTest, ScalarConstant) {
  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto buffers = RunBufferAssignment(module.get());
  // Check that the constant does not have a buffer assigned.
  EXPECT_FALSE(buffers->HasTopLevelAllocation(const0));
}

TEST_F(BufferAssignmentTest, BufferForConst) {
  // Addition of two vector constants: checks that internal constant nodes have
  // no buffers assigned, and their consumer has a buffer.
  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({1.1f, 2.2f, 3.3f, 4.4f})));
  auto const1 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({4.1f, 4.2f, 4.3f, 4.4f})));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec4_, HloOpcode::kAdd, const0, const1));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto buffers = RunBufferAssignment(module.get());
  // The two constant nodes have no buffers assigned.
  EXPECT_FALSE(buffers->HasTopLevelAllocation(const0));
  EXPECT_FALSE(buffers->HasTopLevelAllocation(const1));
  // The add node has an output buffer.
  GetAssignedOutputAllocation(*buffers, add);
}

TEST_F(BufferAssignmentTest, HasAllocationAt) {
  // Create a tuple with non-const and const elements and check that
  // HasAllocationAt works correctly.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, "param0"));
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int>(1)));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, param0));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({negate, param0, constant}));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto buffers = RunBufferAssignment(module.get());
  // Make sure that HasAllocationAt() agrees with what HasTopLevelAllocation()
  // reports for the instruction directly.
  EXPECT_EQ(buffers->HasTopLevelAllocation(tuple),
            buffers->HasAllocationAt(tuple, /*index=*/{}));
  EXPECT_EQ(buffers->HasTopLevelAllocation(negate),
            buffers->HasAllocationAt(tuple, /*index=*/{0}));
  EXPECT_EQ(buffers->HasTopLevelAllocation(param0),
            buffers->HasAllocationAt(tuple, /*index=*/{1}));
  EXPECT_EQ(buffers->HasTopLevelAllocation(constant),
            buffers->HasAllocationAt(tuple, /*index=*/{2}));
}

TEST_F(BufferAssignmentTest, BufferForOutputConst) {
  // This computation copies a constant to output.
  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({1.1f, 2.2f, 3.3f, 4.4f})));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(const0->shape(), HloOpcode::kCopy, const0));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto buffers = RunBufferAssignment(module.get());
  // The copy node now has an output buffer.
  GetAssignedOutputAllocation(*buffers, copy);
}

TEST_F(BufferAssignmentTest, Basic) {
  // paramscalar ------- (mul) -- (add) -- (sub)
  //                     /        /        /
  // param0[100] -------/        /        /
  //                            /        /
  // param1[100] --------------/--------/
  auto builder = HloComputation::Builder(TestName());
  auto paramscalar =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, ""));
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec100_, ""));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec100_, ""));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec100_, HloOpcode::kMultiply, paramscalar, param0));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec100_, HloOpcode::kAdd, mul, param1));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec100_, HloOpcode::kSubtract, add, param1));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto buffers = RunBufferAssignment(module.get());

  // Distinct input buffers were assigned for parameters.
  BufferAllocation paramscalar_buffer =
      GetAssignedInputAllocation(*buffers, paramscalar);
  BufferAllocation param0_buffer = GetAssignedInputAllocation(*buffers, param0);
  BufferAllocation param1_buffer = GetAssignedInputAllocation(*buffers, param1);
  EXPECT_NE(paramscalar_buffer.index(), param0_buffer.index());
  EXPECT_NE(paramscalar_buffer.index(), param1_buffer.index());
  EXPECT_NE(param0_buffer.index(), param1_buffer.index());

  // The mul node has a valid buffer assigned, doesn't share with input.
  const BufferAllocation& mul_buffer = GetTopLevelAllocation(*buffers, mul);
  EXPECT_NE(mul_buffer.index(), param0_buffer.index());

  // The add node can reuse the mul node's buffer.
  const BufferAllocation& add_buffer = GetTopLevelAllocation(*buffers, add);
  EXPECT_EQ(add_buffer.index(), mul_buffer.index());

  // The sub node has a valid output buffer assigned.
  GetAssignedOutputAllocation(*buffers, sub);
}

TEST_F(BufferAssignmentTest, BasicUniquelyColored) {
  // paramscalar ------- (mul) -- (add) -- (sub)
  //                     /        /        /
  // param0[100] -------/        /        /
  //                            /        /
  // param1[100] --------------/--------/
  // The output of each op is colored with a different color, so we can not
  // share anything.
  auto builder = HloComputation::Builder(TestName());
  auto paramscalar =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, ""));
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec100_, ""));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec100_, ""));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec100_, HloOpcode::kMultiply, paramscalar, param0));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec100_, HloOpcode::kAdd, mul, param1));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec100_, HloOpcode::kSubtract, add, param1));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto colorer = [](const BufferLiveness& buffer_liveness) {
    int color = 0;

    for (LogicalBuffer::Id id = 0;
         id < buffer_liveness.points_to_analysis().num_logical_buffers();
         id++) {
      auto& buffer = buffer_liveness.points_to_analysis().logical_buffer(id);
      buffer.set_color(LogicalBuffer::Color(color++));
    }
    return Status::OK();
  };

  auto buffers = RunColoredBufferAssignment(module.get(), colorer);

  // Distinct input buffers were assigned for parameters.
  BufferAllocation paramscalar_buffer =
      GetAssignedInputAllocation(*buffers, paramscalar);
  BufferAllocation param0_buffer = GetAssignedInputAllocation(*buffers, param0);
  BufferAllocation param1_buffer = GetAssignedInputAllocation(*buffers, param1);
  EXPECT_NE(paramscalar_buffer.index(), param0_buffer.index());
  EXPECT_NE(paramscalar_buffer.index(), param1_buffer.index());
  EXPECT_NE(param0_buffer.index(), param1_buffer.index());

  // The mul node has a valid buffer assigned, doesn't share with input.
  const BufferAllocation& mul_buffer = GetTopLevelAllocation(*buffers, mul);
  EXPECT_NE(mul_buffer.index(), param0_buffer.index());

  // The add node can not reuse the mul node's buffer due to coloring.
  const BufferAllocation& add_buffer = GetTopLevelAllocation(*buffers, add);
  EXPECT_NE(add_buffer.index(), mul_buffer.index());

  // The sub node has a valid output buffer assigned.
  GetAssignedOutputAllocation(*buffers, sub);
}

TEST_F(BufferAssignmentTest, BasicPartiallyColored) {
  // paramscalar ------- (mul) -- (add) -- (sub)
  //                     /        /        /
  // param0[100] -------/        /        /
  //                            /        /
  // param1[100] --------------/--------/
  // The output of the mul and the add have the color 1, and the other buffers
  // have the color 0, which allows the mul and add to share buffers.
  auto builder = HloComputation::Builder(TestName());
  auto paramscalar =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, ""));
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec100_, ""));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec100_, ""));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec100_, HloOpcode::kMultiply, paramscalar, param0));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec100_, HloOpcode::kAdd, mul, param1));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec100_, HloOpcode::kSubtract, add, param1));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto colorer = [](const BufferLiveness& buffer_liveness) {
    for (LogicalBuffer::Id id = 0;
         id < buffer_liveness.points_to_analysis().num_logical_buffers();
         id++) {
      auto& buffer = buffer_liveness.points_to_analysis().logical_buffer(id);
      const auto& aliases =
          buffer_liveness.points_to_analysis().GetBufferAliases(buffer);
      for (const auto& alias : aliases) {
        if (alias.instruction()->opcode() == HloOpcode::kAdd ||
            alias.instruction()->opcode() == HloOpcode::kMultiply) {
          buffer.set_color(LogicalBuffer::Color(1));
        }
      }
      if (!buffer.has_color()) {
        buffer.set_color(LogicalBuffer::Color(0));
      }
    }
    return Status::OK();
  };

  auto buffers = RunColoredBufferAssignment(module.get(), colorer);

  // Distinct input buffers were assigned for parameters.
  BufferAllocation paramscalar_buffer =
      GetAssignedInputAllocation(*buffers, paramscalar);
  BufferAllocation param0_buffer = GetAssignedInputAllocation(*buffers, param0);
  BufferAllocation param1_buffer = GetAssignedInputAllocation(*buffers, param1);
  EXPECT_NE(paramscalar_buffer.index(), param0_buffer.index());
  EXPECT_NE(paramscalar_buffer.index(), param1_buffer.index());
  EXPECT_NE(param0_buffer.index(), param1_buffer.index());

  // The mul node has a valid buffer assigned, doesn't share with input.
  const BufferAllocation& mul_buffer = GetTopLevelAllocation(*buffers, mul);
  EXPECT_NE(mul_buffer.index(), param0_buffer.index());

  // The add node can reuse the mul node's buffer.
  const BufferAllocation& add_buffer = GetTopLevelAllocation(*buffers, add);
  EXPECT_EQ(add_buffer.index(), mul_buffer.index());

  // The sub node has a valid output buffer assigned.
  GetAssignedOutputAllocation(*buffers, sub);
}

TEST_F(BufferAssignmentTest, MultipleUsersForNode) {
  // This is similar to the Basic test, with the difference that (sub) is
  // another user of (mul)'s result, so (mul)'s buffer cannot be reused for
  // (add)'s output.
  //
  // paramscalar -------\     /-----------\
  //                     \   /             \
  // param0[100] ------- (mul) -- (add) -- (sub)
  //                              /
  // param1[100] ----------------/
  //
  auto builder = HloComputation::Builder(TestName());
  auto paramscalar =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, ""));
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32vec100_, ""));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32vec100_, ""));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      f32vec100_, HloOpcode::kMultiply, paramscalar, param0));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec100_, HloOpcode::kAdd, mul, param1));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(f32vec100_, HloOpcode::kSubtract, add, mul));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto buffers = RunBufferAssignment(module.get());

  // Input buffers were assigned for parameters.
  BufferAllocation paramscalar_buffer =
      GetAssignedInputAllocation(*buffers, paramscalar);
  BufferAllocation param0_buffer = GetAssignedInputAllocation(*buffers, param0);
  BufferAllocation param1_index = GetAssignedInputAllocation(*buffers, param1);
  EXPECT_NE(paramscalar_buffer.index(), param0_buffer.index());
  EXPECT_NE(paramscalar_buffer.index(), param1_index.index());
  EXPECT_NE(param0_buffer.index(), param1_index.index());

  // The mul node had a buffer allocated.
  const BufferAllocation& mul_buffer = GetTopLevelAllocation(*buffers, mul);

  // Now the add node can't reuse the mul node's buffer.
  const BufferAllocation& add_buffer = GetTopLevelAllocation(*buffers, add);
  EXPECT_NE(add_buffer.index(), mul_buffer.index());

  // Log size information for inspection.
  const std::vector<const HloInstruction*> level0 = GetInstructions(sub);
  int64 size0 = ValidateBuffers(level0, *buffers);
  LOG(INFO) << "LogicalBuffer count " << buffers->Allocations().size()
            << " for " << level0.size() << " instructions; "
            << "total buffer size " << size0;
}

TEST_F(BufferAssignmentTest, TrivialMap) {
  // This tests a trivial x+1 map as the only operation.
  //
  // param0[100x10] ---> (map x+1)
  //
  // Builds the map function.
  auto module = CreateNewModule();
  auto map_computation =
      module->AddEmbeddedComputation(BuildMapComputationPlus1("f32+1"));
  auto inner_last = map_computation->root_instruction();

  // Creates the main kernel and verifies instruction counts.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32a100x10_, ""));
  auto map = builder.AddInstruction(
      HloInstruction::CreateMap(f32a100x10_, {param0}, map_computation));
  module->AddEntryComputation(builder.Build());

  const std::vector<const HloInstruction*> level0 = GetInstructions(map);
  EXPECT_EQ(2, level0.size()) << "Invalid main kernel size";
  const std::vector<const HloInstruction*> level1 = GetInstructions(inner_last);
  EXPECT_EQ(3, level1.size()) << "Invalid nested add+1 size";

  // Assigns buffers and fetches sizes.
  auto buffers = RunBufferAssignment(module.get());
  int64 size0 = ValidateBuffers(level0, *buffers);
  int64 size1 = ValidateBuffers(level1, *buffers);

  // Both algorithms assign the map's buffer before processing the embedded
  // computation, so we can verify that the buffers aren't shared between them
  // by checking:
  EXPECT_TRUE(BuffersDistinct(level0, level1, *buffers))
      << "Reuse between main kernel and embedded mapping.";

  // An input buffer was assigned for the parameter.
  BufferAllocation param0_buffer = GetAssignedInputAllocation(*buffers, param0);

  // An output buffer was assigned for the map.
  BufferAllocation map_buffer = GetAssignedOutputAllocation(*buffers, map);
  EXPECT_NE(param0_buffer.index(), map_buffer.index());

  // The final computation node of the map is an add of an f32 parm and a
  // constant.
  EXPECT_EQ(HloOpcode::kAdd, inner_last->opcode());
  const BufferAllocation& inner_add_buffer =
      GetTopLevelAllocation(*buffers, inner_last);
  EXPECT_NE(inner_add_buffer.index(), map_buffer.index());

  // Log size information for inspection.
  LOG(INFO) << "LogicalBuffer count " << buffers->Allocations().size()
            << " for " << level0.size() + level1.size() << " instructions; "
            << "total buffer size " << size0 + size1;
}

TEST_F(BufferAssignmentTest, CannotReuseInputBufferOfReduce) {
  // Make sure that the input buffer of a reduce cannot be reused for its
  // output.  (Reuse is not safe in the general case, as it reshapes and some
  // out-of-order reductions could overwrite an element before a use.)
  //
  // param0[100] --- (exp1) --- (exp2) --- (reduce x+1) --- (exp3)
  auto module = CreateNewModule();
  auto reduce_computation =
      module->AddEmbeddedComputation(BuildMapComputationPlus1("f32+1"));

  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32a100x10_, ""));
  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32a100x10_, HloOpcode::kExp, param0));
  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32a100x10_, HloOpcode::kExp, exp1));
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      /*shape=*/f32vec10_,
      /*operand=*/exp2,
      /*init_value=*/const0,
      /*dimensions_to_reduce=*/{0}, reduce_computation));
  auto exp3 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec10_, HloOpcode::kExp, reduce));

  module->AddEntryComputation(builder.Build());

  auto buffers = RunBufferAssignment(module.get());
  const std::vector<const HloInstruction*> instrs = GetInstructions(exp3);
  ValidateBuffers(instrs, *buffers);

  const BufferAllocation& exp1_buffer = GetTopLevelAllocation(*buffers, exp1);
  const BufferAllocation& exp2_buffer = GetTopLevelAllocation(*buffers, exp2);
  const BufferAllocation& reduce_buffer =
      GetTopLevelAllocation(*buffers, reduce);

  // The buffer of exp1 is trivially reusable for exp2 - this is just for sanity
  // checking.
  EXPECT_EQ(exp1_buffer.index(), exp2_buffer.index());

  // The buffer of exp2 cannot be used for reduce, even though it's the only
  // operand.
  EXPECT_NE(exp2_buffer.index(), reduce_buffer.index());
}

TEST_F(BufferAssignmentTest, ExampleWhile) {
  // This tests a While loop example from the ir_semantics document.
  //
  // condition (s32,f32[4]) -> bool -- see BuildWhileConditionComputation.
  // body: (s32,f32[4]) -> (s32,f32[4]) -- see BuildWhileBodyComputation.
  //
  // const3[s32] -------\
  // const4[f32[4]] --- tuple --- while[condition, body]
  //
  // Builds the nested condition and body.
  auto module = CreateNewModule();
  auto condition_computation =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("if<4"));
  auto body_computation =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("add-update"));

  // Creates the main kernel and verifies instruction counts.
  auto builder = HloComputation::Builder(TestName());
  auto const3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int>(0)));
  auto const4 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR1<float>({1.1f, 2.2f, 3.3f, 4.4f})));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({const3, const4}));
  auto while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      t_s32_f32v4_, condition_computation, body_computation, tuple));
  module->AddEntryComputation(builder.Build());

  const std::vector<const HloInstruction*> level0 = GetInstructions(while_op);
  EXPECT_EQ(4, level0.size()) << "Invalid while kernel size";
  const std::vector<const HloInstruction*> levelc =
      GetInstructions(condition_computation->root_instruction());
  EXPECT_EQ(4, levelc.size()) << "Invalid nested condition size";
  const std::vector<const HloInstruction*> levelb =
      GetInstructions(body_computation->root_instruction());
  EXPECT_EQ(8, levelb.size()) << "Invalid nested body size";

  // Assigns buffers and fetches sizes.
  auto buffers = RunBufferAssignment(module.get());
  int64 size0 = ValidateBuffers(level0, *buffers);
  int64 sizec = ValidateBuffers(levelc, *buffers);
  int64 sizeb = ValidateBuffers(levelb, *buffers);

  // BufferAssignment will assign a single allocation for the following
  // instructions: while, while.cond.param, while.body.param, while.body.result.
  EXPECT_FALSE(BuffersDistinct(level0, levelc, *buffers))
      << "Should be reuse between main kernel and embedded condition.";
  EXPECT_FALSE(BuffersDistinct(levelb, levelc, *buffers))
      << "Should be reuse between embedded condition and body.";
  // Expect buffer reuse between main kernel and body computation.
  EXPECT_FALSE(BuffersDistinct(level0, levelb, *buffers))
      << "Should be reuse between main kernel and embedded body.";

  // The final computation node of the while body is a tuple of s32 and
  // f32[4] adds.
  HloInstruction* body_root = body_computation->root_instruction();
  EXPECT_EQ(HloOpcode::kTuple, body_root->opcode());

  // Check that buffer for each subshape of 'while_op' shares allocation with
  // corresponding buffer from while body computation at same index.
  ShapeUtil::ForEachSubshape(
      while_op->shape(),
      [this, &buffers, while_op, body_root](const Shape& /*subshape*/,
                                            const ShapeIndex& index) {
        auto while_op_allocation = GetAllocation(*buffers, while_op, index);
        auto body_root_allocation = GetAllocation(*buffers, body_root, index);
        EXPECT_EQ(while_op_allocation.index(), body_root_allocation.index());
      });

  // Log size information for inspection.
  LOG(INFO) << "LogicalBuffer count " << buffers->Allocations().size()
            << " for " << level0.size() + levelc.size() + levelb.size()
            << " instructions; total buffer size " << size0 + sizec + sizeb;
}

TEST_F(BufferAssignmentTest, UnaryOpReuseChain) {
  // param0[100] ---> (exp) ---> (tanh) ---> (exp) ---> (neg)
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, ""));
  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kExp, param0));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kTanh, exp1));
  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kExp, tanh));
  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, exp2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // tanh and exp2 can reuse exp1's buffer
  EXPECT_TRUE(assignment->HasTopLevelAllocation(exp1));
  auto& buffer_for_exp1 = GetTopLevelAllocation(*assignment, exp1);
  EXPECT_EQ(buffer_for_exp1, GetTopLevelAllocation(*assignment, tanh));
  EXPECT_EQ(buffer_for_exp1, GetTopLevelAllocation(*assignment, exp2));
  EXPECT_EQ(buffer_for_exp1, GetTopLevelAllocation(*assignment, neg));
}

TEST_F(BufferAssignmentTest, ReuseNonOperandBuffer) {
  // This computation is a chain of operations which decreases in buffer size
  // (via slice) then increases in size (via broadcast):
  //
  // param ---> (negate) ---> (slice) ---> (broadcast)
  //
  // The negate should share a buffer with broadcast.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, "param0"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, param0));
  auto slice = builder.AddInstruction(
      HloInstruction::CreateSlice(f32vec10_, negate, {0}, {10}, {1}));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32a100x10_, slice, {1}));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // negate and broadcast should share a buffer.
  EXPECT_TRUE(assignment->HasTopLevelAllocation(broadcast));
  auto& buffer_for_bcast = GetTopLevelAllocation(*assignment, broadcast);
  EXPECT_EQ(buffer_for_bcast, GetTopLevelAllocation(*assignment, negate));

  // Slice should have its own buffer.
  EXPECT_NE(buffer_for_bcast, GetTopLevelAllocation(*assignment, slice));
}

TEST_F(BufferAssignmentTest, NoReuseLiveBuffer) {
  // This computation is identical to that in ReuseNonOperandBuffer, but the
  // negate value is live until the end of the computation (due to it being an
  // operand of the output tuple) preventing reuse.
  //
  // param ---> (negate) ---> (slice) ---> (broadcast)-> (tuple)
  //                  \-----------------------------------/
  //
  // The negate should not share a buffer with broadcast.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, "param0"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, param0));
  auto slice = builder.AddInstruction(
      HloInstruction::CreateSlice(f32vec10_, negate, {0}, {10}, {1}));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32a100x10_, slice, {1}));
  builder.AddInstruction(HloInstruction::CreateTuple({negate, broadcast}));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // The instructions should not share buffers.
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, negate));
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, slice));
  EXPECT_NE(GetTopLevelAllocation(*assignment, negate),
            GetTopLevelAllocation(*assignment, slice));
}

TEST_F(BufferAssignmentTest, NoReuseAliasedBuffer) {
  // This computation is identical to that in ReuseNonOperandBuffer, but the
  // negate value is placed into a tuple which lives to the end of the
  // computation. This extends the live range of negate's buffer preventing
  // reuse due to buffer aliasing.
  //
  // param ---> (negate) ---> (tuple) -> (slice) ---> (broadcast)-> (tuple)
  //                              \-----------------------------------/
  //
  // The negate should not share a buffer with broadcast.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, "param0"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, param0));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({negate}));
  auto tuple_element = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32vec100_, tuple, 0));
  auto slice = builder.AddInstruction(
      HloInstruction::CreateSlice(f32vec10_, tuple_element, {0}, {10}, {1}));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32a100x10_, slice, {1}));
  builder.AddInstruction(HloInstruction::CreateTuple({tuple, broadcast}));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // The instructions should not share buffers.
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, negate));
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, slice));
  EXPECT_NE(GetTopLevelAllocation(*assignment, negate),
            GetTopLevelAllocation(*assignment, slice));
}

TEST_F(BufferAssignmentTest, DoNotReuseOversizedOutputBuffer) {
  // This computation is very similar to ReuseNonOperandBuffer except the
  // broadcast has a smaller output than the negate. This should block reuse of
  // negate's buffer by broadcast because the output buffer(s) of a computation
  // should be exactly sized for the value.
  //
  // param ---> (negate) ---> (slice) ---> (broadcast)
  //
  // Neither negate nor slice may share a buffer with broadcast.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, "param0"));
  // Negate output is 100 elements.
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, param0));
  // Slice output is 10 elements.
  auto slice = builder.AddInstruction(
      HloInstruction::CreateSlice(f32vec10_, negate, {0}, {10}, {1}));
  // Broadcast output is 40 elements.
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {10, 4}), slice, {0}));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // The broadcast output buffer cannot be shared.
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, negate));
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, slice));
}

TEST_F(BufferAssignmentTest, ReuseOutputBufferIfExactlySized) {
  // This is identical to DoNotReuseOversizedOutputBuffer except the broadcast
  // output is exactly the same size as the negate (rather than being
  // smaller). This enables reuse of negate's buffer by the broadcast because
  // the output buffer will be sized exactly to its value.
  //
  // param ---> (negate) ---> (slice) ---> (broadcast)
  //
  // The negate should *not* share a buffer with broadcast.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, "param0"));
  // Negate output is 100 elements.
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, param0));
  auto slice = builder.AddInstruction(
      HloInstruction::CreateSlice(f32vec10_, negate, {0}, {10}, {1}));
  // Broadcast output is 40 elements.
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {10, 10}), slice, {0}));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // negate and broadcast should share a buffer.
  EXPECT_TRUE(assignment->HasTopLevelAllocation(broadcast));
  auto& buffer_for_bcast = GetTopLevelAllocation(*assignment, broadcast);
  EXPECT_EQ(buffer_for_bcast, GetTopLevelAllocation(*assignment, negate));

  // Slice should have its own buffer.
  EXPECT_NE(buffer_for_bcast, GetTopLevelAllocation(*assignment, slice));
}

TEST_F(BufferAssignmentTest, DoNotReuseOversizedOutputBufferInTuple) {
  // This computation is very similar to ReuseNonOperandBuffer except the
  // broadcast has a smaller output than the negate, and the broadcast is
  // contained in the computation output as a tuple element. This should block
  // reuse of the negate's buffer by the broadcast because the output buffer(s)
  // of a computation should be exactly sized for the value. This includes those
  // buffers aliased in the output (eg, contained as tuple elements).
  //
  // param ---> (negate) ---> (slice) ---> (broadcast) --> (tuple)
  //
  // Neither negate nor slice may share a buffer with broadcast.
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32vec100_, "param0"));
  // Negate output is 100 elements.
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32vec100_, HloOpcode::kNegate, param0));
  // Slice output is 10 elements.
  auto slice = builder.AddInstruction(
      HloInstruction::CreateSlice(f32vec10_, negate, {0}, {10}, {1}));
  // Broadcast output is 40 elements.
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {10, 4}), slice, {0}));
  builder.AddInstruction(HloInstruction::CreateTuple({broadcast}));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // The broadcast output buffer cannot be shared.
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, negate));
  EXPECT_NE(GetTopLevelAllocation(*assignment, broadcast),
            GetTopLevelAllocation(*assignment, slice));
}

TEST_F(BufferAssignmentTest, EmbeddedComputationBuffers) {
  // Verify that buffers for embedded computations are properly marked as
  // thread-local and that embedded parameters are not marked as
  // is_entry_computation_parameter.
  auto module = CreateNewModule();
  auto vec_shape = ShapeUtil::MakeShape(F32, {42});
  auto scalar_shape = ShapeUtil::MakeShape(F32, {});

  // Create a scalar computation to use in a map.
  auto map_builder = HloComputation::Builder(TestName() + "_map");
  auto map_param = map_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "map_param"));
  auto map_root = map_builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, map_param));
  auto map_computation = module->AddEmbeddedComputation(map_builder.Build());

  // Create a vector computation to use in a kCall.
  auto call_builder = HloComputation::Builder(TestName() + "_call");
  auto call_param = call_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec_shape, "vec_param"));
  auto call_root = call_builder.AddInstruction(
      HloInstruction::CreateUnary(vec_shape, HloOpcode::kExp, call_param));
  auto call_computation = module->AddEmbeddedComputation(call_builder.Build());

  // Create entry computation which kCalls call_computation and then calls map
  // with map_computation on the result.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec_shape, "param"));
  auto call = builder.AddInstruction(
      HloInstruction::CreateCall(vec_shape, {param}, call_computation));
  auto map = builder.AddInstruction(
      HloInstruction::CreateMap(vec_shape, {call}, map_computation));
  module->AddEntryComputation(builder.Build());

  auto assignment = RunBufferAssignment(module.get());

  // Allocations for the map computation should be thread-local and not
  // live-out.
  auto& map_param_alloc = GetTopLevelAllocation(*assignment, map_param);
  EXPECT_FALSE(map_param_alloc.is_entry_computation_parameter());
  EXPECT_FALSE(map_param_alloc.maybe_live_out());
  EXPECT_TRUE(map_param_alloc.is_thread_local());

  auto& map_root_alloc = GetTopLevelAllocation(*assignment, map_root);
  EXPECT_FALSE(map_root_alloc.is_entry_computation_parameter());
  EXPECT_FALSE(map_root_alloc.maybe_live_out());
  EXPECT_TRUE(map_root_alloc.is_thread_local());

  // Allocations for the call computation should not be thread-local.
  auto& call_param_alloc = GetTopLevelAllocation(*assignment, call_param);
  EXPECT_FALSE(call_param_alloc.is_entry_computation_parameter());
  EXPECT_FALSE(call_param_alloc.maybe_live_out());
  EXPECT_FALSE(call_param_alloc.is_thread_local());

  auto& call_root_alloc = GetTopLevelAllocation(*assignment, call_root);
  EXPECT_FALSE(call_root_alloc.is_entry_computation_parameter());
  EXPECT_FALSE(call_root_alloc.is_thread_local());

  // Entry computation allocations can be marked liveout and
  // is_entry_computation_parameter.
  auto& param_alloc = GetTopLevelAllocation(*assignment, param);
  EXPECT_TRUE(param_alloc.is_entry_computation_parameter());
  EXPECT_FALSE(param_alloc.maybe_live_out());
  EXPECT_FALSE(param_alloc.is_thread_local());

  auto& map_alloc = GetTopLevelAllocation(*assignment, map);
  EXPECT_FALSE(map_alloc.is_entry_computation_parameter());
  EXPECT_TRUE(map_alloc.maybe_live_out());
  EXPECT_FALSE(map_alloc.is_thread_local());
}

TEST_F(BufferAssignmentTest, TupleParameterAsOutput) {
  // Test a computation that returns a tuple parameter.
  auto builder = HloComputation::Builder(TestName());
  auto tuple_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(PRED, {1, 2, 3, 4}),
                                 ShapeUtil::MakeShape(F32, {}),
                                 ShapeUtil::MakeShape(S32, {42})}),
      "param0"));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // There should be four allocations: one for vector of pointers, and one for
  // each tuple element.
  EXPECT_EQ(4, assignment->Allocations().size());

  // Verify each buffer allocation is marked as an entry computation parameter
  // and is liveout.
  ShapeUtil::ForEachSubshape(
      tuple_param->shape(),
      [this, &assignment, tuple_param](const Shape& /*subshape*/,
                                       const ShapeIndex& index) {
        auto allocation = GetAllocation(*assignment, tuple_param, index);
        EXPECT_TRUE(allocation.is_entry_computation_parameter());
        EXPECT_EQ(0, allocation.parameter_number());
        EXPECT_TRUE(allocation.maybe_live_out());
      });
}

TEST_F(BufferAssignmentTest, ElementOfNestedTupleParameterAsOutput) {
  // Test a computation which returns a GetElementTuple of a nested tuple
  // parameter.
  auto builder = HloComputation::Builder(TestName());
  auto tuple_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(PRED, {1, 2, 3, 4}),
           ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {42}),
                                      ShapeUtil::MakeShape(S32, {101})})}),
      "param0"));
  auto tuple_element =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetSubshape(tuple_param->shape(), {1}), tuple_param, 1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // Only some of the elements of the input param are liveout.
  EXPECT_FALSE(
      GetAllocation(*assignment, tuple_param, /*index=*/{}).maybe_live_out());
  // Tuple element at index={1} is live out because GetTupleElement({1})
  // forwards a pointer to this allocation (instead of defining its own buffer).
  EXPECT_TRUE(
      GetAllocation(*assignment, tuple_param, /*index=*/{1}).maybe_live_out());
  EXPECT_TRUE(GetAllocation(*assignment, tuple_param, /*index=*/{1, 0})
                  .maybe_live_out());
  EXPECT_TRUE(GetAllocation(*assignment, tuple_param, /*index=*/{1, 1})
                  .maybe_live_out());

  // The GetTupleElement output is liveout.
  EXPECT_TRUE(
      GetTopLevelAllocation(*assignment, tuple_element).maybe_live_out());

  // Verify that the GetTupleElement allocations of its elements match the
  // corresponding tuple parameter allocations because they alias.
  EXPECT_EQ(GetAllocation(*assignment, tuple_param, /*index=*/{1, 0}),
            GetAllocation(*assignment, tuple_element, /*index=*/{0}));
  EXPECT_EQ(GetAllocation(*assignment, tuple_param, /*index=*/{1, 1}),
            GetAllocation(*assignment, tuple_element, /*index=*/{1}));

  // GetTupleElement forwards a pointer to its underlying buffer, so verify
  // that it has the same allocation than the corresponding parameter element.
  EXPECT_EQ(GetAllocation(*assignment, tuple_param, /*index=*/{1}),
            GetTopLevelAllocation(*assignment, tuple_element));
}

// TODO(b/32248867): Enable when buffer assignment gives allocations to
// constants.
TEST_F(BufferAssignmentTest, DISABLED_TupleConstantAsOutput) {
  // Test that a tuple constant which is forwarded to the computation output
  // is properly handled.
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(Literal::MakeTuple(
      {Literal::CreateR0<int64>(0).get(), Literal::CreateR0<int64>(1).get()})));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  EXPECT_EQ(3, assignment->Allocations().size());
}

TEST_F(BufferAssignmentTest, TupleCustomCallAsOutput) {
  // Test a computation which returns a tuple custom call value.
  auto builder = HloComputation::Builder(TestName());
  auto custom_call = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(PRED, {1, 2, 3, 4}),
                                 ShapeUtil::MakeShape(S32, {101})}),
      /*operands=*/{}, /*custom_call_target=*/"foo_function"));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  EXPECT_EQ(3, assignment->Allocations().size());
  EXPECT_TRUE(
      GetAllocation(*assignment, custom_call, /*index=*/{}).maybe_live_out());
  EXPECT_TRUE(
      GetAllocation(*assignment, custom_call, /*index=*/{0}).maybe_live_out());
  EXPECT_TRUE(
      GetAllocation(*assignment, custom_call, /*index=*/{1}).maybe_live_out());
}

TEST_F(BufferAssignmentTest, TupleCallAsOutput) {
  // Test a computation which returns a tuple call value.
  auto module = CreateNewModule();
  auto elem_shape = f32vec4_;
  auto tuple_shape = ShapeUtil::MakeTupleShape({elem_shape});

  auto sub_builder = HloComputation::Builder(TestName() + "_sub");
  auto sub_param = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, elem_shape, "sub_param"));
  auto sub_tuple =
      sub_builder.AddInstruction(HloInstruction::CreateTuple({sub_param}));
  auto sub_computation = module->AddEmbeddedComputation(sub_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, elem_shape, "param"));
  auto call = builder.AddInstruction(
      HloInstruction::CreateCall(tuple_shape, {param}, sub_computation));
  module->AddEntryComputation(builder.Build());

  auto assignment = RunBufferAssignment(module.get());

  EXPECT_EQ(3, assignment->Allocations().size());
  // Buffers for call are colocated with the sub-computation.
  EXPECT_EQ(GetAllocation(*assignment, call, /*index=*/{}),
            GetAllocation(*assignment, sub_tuple, /*index=*/{}));
  EXPECT_EQ(GetAllocation(*assignment, call, /*index=*/{0}),
            GetAllocation(*assignment, sub_param, /*index=*/{}));
  // The parameter isn't aliased with anything.
  EXPECT_NE(GetTopLevelAllocation(*assignment, param),
            GetTopLevelAllocation(*assignment, sub_tuple));
  EXPECT_NE(GetTopLevelAllocation(*assignment, param),
            GetTopLevelAllocation(*assignment, sub_param));
}

TEST_F(BufferAssignmentTest, TupleChainedCallAsOutput) {
  // Test a chain of calls with tuple output. The chain looks like:
  // A: call(B, tuple(param))
  // B: call(C, param)
  // C: call(D, param)
  // D: param
  auto module = CreateNewModule();
  auto elem_shape = f32vec4_;
  auto tuple_shape = ShapeUtil::MakeTupleShape({elem_shape});

  auto d_builder = HloComputation::Builder(TestName() + "_d");
  auto d_param = d_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "d_param"));
  auto d_computation = d_builder.Build();

  auto c_builder = HloComputation::Builder(TestName() + "_c");
  auto c_param = c_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "c_param"));
  auto c_call = c_builder.AddInstruction(
      HloInstruction::CreateCall(tuple_shape, {c_param}, d_computation.get()));
  auto c_computation = c_builder.Build();

  auto b_builder = HloComputation::Builder(TestName() + "_b");
  auto b_param = b_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "b_param"));
  auto b_call = b_builder.AddInstruction(
      HloInstruction::CreateCall(tuple_shape, {b_param}, c_computation.get()));
  auto b_computation = b_builder.Build();

  auto a_builder = HloComputation::Builder(TestName());
  auto a_param = a_builder.AddInstruction(
      HloInstruction::CreateParameter(0, elem_shape, "param"));
  auto a_tuple =
      a_builder.AddInstruction(HloInstruction::CreateTuple({a_param}));
  auto a_call = a_builder.AddInstruction(
      HloInstruction::CreateCall(tuple_shape, {a_tuple}, b_computation.get()));
  auto a_computation = a_builder.Build();

  // Add the computations in an order that doesn't match the dependency
  // post-order, to shake out more possible bugs.
  module->AddEmbeddedComputation(std::move(d_computation));
  module->AddEmbeddedComputation(std::move(c_computation));
  module->AddEntryComputation(std::move(a_computation));
  module->AddEmbeddedComputation(std::move(b_computation));

  auto assignment = RunBufferAssignment(module.get());

  // Buffers for call are colocated with the sub-computations.
  EXPECT_EQ(GetAllocation(*assignment, a_call, /*index=*/{}),
            GetAllocation(*assignment, b_call, /*index=*/{}));
  EXPECT_EQ(GetAllocation(*assignment, b_call, /*index=*/{}),
            GetAllocation(*assignment, c_call, /*index=*/{}));
  EXPECT_EQ(GetAllocation(*assignment, c_call, /*index=*/{}),
            GetAllocation(*assignment, d_param, /*index=*/{}));
  EXPECT_EQ(GetAllocation(*assignment, a_call, /*index=*/{0}),
            GetAllocation(*assignment, b_call, /*index=*/{0}));
  EXPECT_EQ(GetAllocation(*assignment, b_call, /*index=*/{0}),
            GetAllocation(*assignment, c_call, /*index=*/{0}));
  EXPECT_EQ(GetAllocation(*assignment, c_call, /*index=*/{0}),
            GetAllocation(*assignment, d_param, /*index=*/{0}));
  // The parameters aren't aliased with anything.
  EXPECT_TRUE(BuffersDistinct({a_param}, {b_param}, *assignment));
  EXPECT_TRUE(BuffersDistinct({a_param}, {c_param}, *assignment));
  EXPECT_TRUE(BuffersDistinct({a_param}, {d_param}, *assignment));
  EXPECT_TRUE(BuffersDistinct({b_param}, {c_param}, *assignment));
  EXPECT_TRUE(BuffersDistinct({b_param}, {d_param}, *assignment));
  EXPECT_TRUE(BuffersDistinct({c_param}, {d_param}, *assignment));
}

TEST_F(BufferAssignmentTest, BitcastAsOutput) {
  // Test a computation which returns a bitcast value.
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {42}), "param"));
  auto bitcast = builder.AddInstruction(
      HloInstruction::CreateUnary(param->shape(), HloOpcode::kBitcast, param));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // Bitcast should get the same allocation as the param.
  EXPECT_EQ(1, assignment->Allocations().size());
  EXPECT_EQ(GetTopLevelAllocation(*assignment, param),
            GetTopLevelAllocation(*assignment, bitcast));
}

TEST_F(BufferAssignmentTest, AmbiguousBufferAsOutput) {
  // Test a computation with an output that has an ambiguous points-to set.
  // This is constructed using a select among tuple shapes.
  auto builder = HloComputation::Builder(TestName());
  auto tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(PRED, {1, 2, 3, 4})});

  auto tuple_param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param0"));
  auto tuple_param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, tuple_shape, "param1"));
  auto pred_param = builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(PRED, {}), "param1"));
  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple_shape, HloOpcode::kSelect, pred_param, tuple_param0, tuple_param1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // Select shallow copies one of its operands so it defines its own top-level
  // buffer and receives its own allocation.
  auto select_alloc = GetTopLevelAllocation(*assignment, select);
  EXPECT_EQ(1, select_alloc.assigned_buffers().size());
  EXPECT_EQ(select,
            select_alloc.assigned_buffers().begin()->first->instruction());

  // The buffer for the tuple element of the select is forwarded from one its
  // operands which cannot be determined statically. Therefore its slices
  // should include the slices of both of the elements in the parameters.
  auto element_slices = assignment->GetAllSlices(select, /*index=*/{0});
  EXPECT_EQ(2, element_slices.size());
  EXPECT_THAT(element_slices,
              ::testing::UnorderedElementsAre(
                  assignment->GetUniqueSlice(tuple_param0, /*index=*/{0})
                      .ConsumeValueOrDie(),
                  assignment->GetUniqueSlice(tuple_param1, /*index=*/{0})
                      .ConsumeValueOrDie()));
}

// TODO(b/34669761): Remove this test when buffers are allowed to share
// allocations.
TEST_F(BufferAssignmentTest, TupleBufferNotReused) {
  // Test a computation that returns a tuple parameter.
  auto builder = HloComputation::Builder(TestName());
  auto scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param0"));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({param}));
  auto tuple_element = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
  auto copy = builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape, HloOpcode::kCopy, tuple_element));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get());

  // There should be no buffer reuse. The copy should not reuse the tuple
  // buffer.
  EXPECT_EQ(3, assignment->Allocations().size());
  EXPECT_NE(GetTopLevelAllocation(*assignment, tuple),
            GetTopLevelAllocation(*assignment, copy));
}

TEST_F(BufferAssignmentTest, OneTempAllocation) {
  // Test a computation that requires multiple temp buffers, and ensure they
  // are combined into a single allocation.
  auto builder = HloComputation::Builder(TestName());
  Shape shape_2x3 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape_2x4 = ShapeUtil::MakeShape(F32, {2, 4});
  Shape shape_3x4 = ShapeUtil::MakeShape(F32, {3, 4});
  Shape shape_4x4 = ShapeUtil::MakeShape(F32, {4, 4});
  Shape shape_5x4 = ShapeUtil::MakeShape(F32, {5, 4});

  // There should be separate temp buffers for dot_ab and dot_bc.
  auto param_a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape_2x3, "param_a"));
  auto param_b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape_3x4, "param_b"));
  auto param_c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape_4x4, "param_c"));
  auto dot_ab = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_2x4, HloOpcode::kDot, param_a, param_b));
  auto dot_bc = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_3x4, HloOpcode::kDot, param_b, param_c));
  builder.AddInstruction(
      HloInstruction::CreateConcatenate(shape_5x4, {dot_ab, dot_bc}, 1));

  // Run buffer assignment with alignment=1.
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto assignment = RunBufferAssignment(module.get(), /*alignment=*/1);

  // There are 5 allocations: 3 parameters, 1 output, and 1 temp.
  EXPECT_EQ(5, assignment->Allocations().size());

  // Ensure the temp buffers for dot_ab and dot_bc share a single allocation,
  // and each occupies different slices of that allocation.
  BufferAllocation::Slice slice_ab =
      assignment->GetUniqueTopLevelSlice(dot_ab).ConsumeValueOrDie();
  BufferAllocation::Slice slice_bc =
      assignment->GetUniqueTopLevelSlice(dot_bc).ConsumeValueOrDie();
  EXPECT_EQ(slice_ab.allocation(), slice_bc.allocation());
  EXPECT_NE(slice_ab, slice_bc);
  EXPECT_EQ(32, slice_ab.size());
  EXPECT_EQ(48, slice_bc.size());
  EXPECT_EQ(80, slice_ab.allocation()->size());
  EXPECT_EQ(80, slice_bc.allocation()->size());

  // Re-run buffer assignment with alignment=64.
  assignment = RunBufferAssignment(module.get(), /*alignment=*/64);
  EXPECT_EQ(5, assignment->Allocations().size());
  slice_ab = assignment->GetUniqueTopLevelSlice(dot_ab).ConsumeValueOrDie();
  slice_bc = assignment->GetUniqueTopLevelSlice(dot_bc).ConsumeValueOrDie();
  EXPECT_EQ(slice_ab.allocation(), slice_bc.allocation());
  EXPECT_NE(slice_ab, slice_bc);
  EXPECT_EQ(32, slice_ab.size());
  EXPECT_EQ(48, slice_bc.size());
  // Ensure the offsets and allocation size account for the alignment, without
  // assuming which buffer gets assigned first.
  if (slice_ab.offset() == 0) {
    EXPECT_EQ(64, slice_bc.offset());
    EXPECT_EQ(64 + 48, slice_ab.allocation()->size());
    EXPECT_EQ(64 + 48, slice_bc.allocation()->size());
  } else {
    EXPECT_EQ(64, slice_ab.offset());
    EXPECT_EQ(0, slice_bc.offset());
    EXPECT_EQ(64 + 32, slice_ab.allocation()->size());
    EXPECT_EQ(64 + 32, slice_bc.allocation()->size());
  }
}

class WhileBufferAssignmentTest : public HloTestBase {
 protected:
  std::unique_ptr<HloComputation> BuildWhileConditionComputation(
      const string& name) {
    auto builder = HloComputation::Builder(name);
    builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    auto zero = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int>(0)));
    auto ten = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int>(10)));
    builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, zero, ten));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> BuildWhileBodyComputation(
      const string& name) {
    auto builder = HloComputation::Builder(name);
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    auto input = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 0));
    auto weights = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));
    auto output = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, input, weights));
    builder.AddInstruction(
        HloInstruction::CreateTuple({input, weights, output}));
    return builder.Build();
  }

  std::unique_ptr<BufferAssignment> RunBufferAssignment(HloModule* module,
                                                        int64 alignment = 1) {
    auto sequence =
        CreateMemoryMinimizingSequence(*module, ByteSizeOf).ConsumeValueOrDie();
    return BufferAssigner::Run(
               module, xla::MakeUnique<SequentialHloOrdering>(module, sequence),
               ByteSizeOf,
               [alignment](LogicalBuffer::Color) { return alignment; })
        .ConsumeValueOrDie();
  }

  static int64 ByteSizeOf(const LogicalBuffer& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
  }

  Shape data_shape_ = ShapeUtil::MakeShape(F32, {4});
  Shape loop_state_shape_ =
      ShapeUtil::MakeTupleShape({data_shape_, data_shape_, data_shape_});
};

static void RunCopyInsertion(HloModule* module) {
  CopyInsertion copy_insertion;
  EXPECT_IS_OK(copy_insertion.Run(module).status());
}

TEST_F(WhileBufferAssignmentTest, TwoForwardWhileLoops) {
  auto module = xla::MakeUnique<HloModule>(TestName());
  auto builder = HloComputation::Builder("entry");

  auto input0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape_, "input0"));
  auto weights0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "weights0"));
  auto weights1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, data_shape_, "weights1"));

  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0)));
  auto output0 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));
  auto output1 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));

  auto cond0 =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body0 =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));

  auto tuple0 = builder.AddInstruction(
      HloInstruction::CreateTuple({input0, weights0, output0}));
  auto while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond0, body0, tuple0));

  auto cond1 =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body1 =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));
  auto input1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape_, while0, 2));
  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({input1, weights1, output1}));
  auto while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond1, body1, tuple1));

  module->AddEntryComputation(builder.Build());
  RunCopyInsertion(module.get());
  auto assignment = RunBufferAssignment(module.get());

  // Verify 'input0' and read-only use while0{0} alias.
  EXPECT_EQ(assignment->GetUniqueSlice(input0, {}).ConsumeValueOrDie(),
            assignment->GetUniqueSlice(while0, {0}).ConsumeValueOrDie());
  // Verify 'weights0' and read-only use while0{1} alias.
  EXPECT_EQ(assignment->GetUniqueSlice(weights0, {}).ConsumeValueOrDie(),
            assignment->GetUniqueSlice(while0, {1}).ConsumeValueOrDie());
  // Verify 'while0{2}' and read-only use while1{0} alias.
  EXPECT_EQ(assignment->GetUniqueSlice(while0, {2}).ConsumeValueOrDie(),
            assignment->GetUniqueSlice(while1, {0}).ConsumeValueOrDie());
  // Verify 'weights1' and read-only use while1{1} alias.
  EXPECT_EQ(assignment->GetUniqueSlice(weights1, {}).ConsumeValueOrDie(),
            assignment->GetUniqueSlice(while1, {1}).ConsumeValueOrDie());
}

TEST_F(WhileBufferAssignmentTest, OneForwardBackwardWhileLoopSet) {
  auto module = xla::MakeUnique<HloModule>(TestName());
  auto builder = HloComputation::Builder("entry");

  auto input0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape_, "input0"));
  auto weights0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "weights0"));

  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0)));
  auto output0 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));
  auto output1 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));

  auto cond0 =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body0 =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));

  auto tuple0 = builder.AddInstruction(
      HloInstruction::CreateTuple({input0, weights0, output0}));
  auto while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond0, body0, tuple0));

  auto cond1 =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body1 =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));

  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({input0, weights0, output1}));
  auto while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond1, body1, tuple1));

  module->AddEntryComputation(builder.Build());
  RunCopyInsertion(module.get());
  auto assignment = RunBufferAssignment(module.get());

  // while0 and while1 buffers should be completely aligned.
  EXPECT_EQ(assignment->GetUniqueSlice(while0, {0}).ConsumeValueOrDie(),
            assignment->GetUniqueSlice(while1, {0}).ConsumeValueOrDie());
  EXPECT_EQ(assignment->GetUniqueSlice(while0, {1}).ConsumeValueOrDie(),
            assignment->GetUniqueSlice(while1, {1}).ConsumeValueOrDie());
  EXPECT_EQ(assignment->GetUniqueSlice(while0, {2}).ConsumeValueOrDie(),
            assignment->GetUniqueSlice(while1, {2}).ConsumeValueOrDie());
}

TEST_F(BufferAssignmentTest, TwoCalls) {
  auto module = xla::MakeUnique<HloModule>(TestName());
  Shape r0f32 = ShapeUtil::MakeShape(xla::F32, {});
  HloComputation* sub_computation;
  {
    auto builder = HloComputation::Builder(TestName() + "_sub_comp");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, r0f32, "param"));
    auto constant1 = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
    auto add = builder.AddInstruction(
        HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param, constant1));
    sub_computation = module->AddEmbeddedComputation(builder.Build(add));
  }
  auto builder = HloComputation::Builder(TestName());
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.0)));
  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(3.0)));
  auto call1 = builder.AddInstruction(
      HloInstruction::CreateCall(r0f32, {constant2}, sub_computation));
  auto call2 = builder.AddInstruction(
      HloInstruction::CreateCall(r0f32, {constant3}, sub_computation));
  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, call1, constant2));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, call2, add1));
  module->AddEntryComputation(builder.Build(add2));

  {
    FlattenCallGraph flatten;
    TF_ASSERT_OK_AND_ASSIGN(bool result, flatten.Run(module.get()));
    EXPECT_TRUE(result);
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  }

  RunCopyInsertion(module.get());
  auto assignment = RunBufferAssignment(module.get());

  EXPECT_TRUE(BuffersDistinct({call1}, {call2}, *assignment));
}

static bool IsPostOrderTraversal(
    const std::vector<const HloInstruction*>& sequence) {
  tensorflow::gtl::FlatSet<const HloInstruction*> seen_so_far;
  auto has_not_been_seen_yet = [&](const HloInstruction* instruction) {
    return seen_so_far.count(instruction) == 0;
  };

  for (auto instruction : sequence) {
    if (std::any_of(instruction->operands().begin(),
                    instruction->operands().end(), has_not_been_seen_yet) ||
        std::any_of(instruction->control_predecessors().begin(),
                    instruction->control_predecessors().end(),
                    has_not_been_seen_yet)) {
      return false;  // Not a post order.
    }
    if (!seen_so_far.insert(instruction).second) {
      return false;  // Not a "traversal".
    }
  }

  return true;
}

TEST_F(WhileBufferAssignmentTest, WhileLoopsInterferingResultRange) {
  auto module = xla::MakeUnique<HloModule>(TestName());
  auto builder = HloComputation::Builder(TestName());

  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0)));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));

  auto input0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape_, "input0"));
  auto weights0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "weights0"));
  auto output0 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));

  auto input1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, data_shape_, "input1"));
  auto weights1 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, data_shape_, "weights1"));
  auto output1 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, one, {1}));

  auto cond =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body = module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));

  auto tuple0 = builder.AddInstruction(
      HloInstruction::CreateTuple({input0, weights0, output0}));
  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({input1, weights1, output1}));

  auto while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond, body, tuple0));
  auto while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond, body, tuple1));

  auto root_add = builder.AddInstruction(HloInstruction::CreateBinary(
      while0->shape(), HloOpcode::kAdd, while0, while1));
  module->AddEntryComputation(builder.Build());

  RunCopyInsertion(module.get());

  {
    FlattenCallGraph flatten;
    TF_ASSERT_OK_AND_ASSIGN(bool result, flatten.Run(module.get()));
    EXPECT_TRUE(result);
  }

  auto sequence =
      CreateMemoryMinimizingSequence(*module, ByteSizeOf).ConsumeValueOrDie();

  // To trigger b/38494731, we want a specific Hlo sequence for the
  // root computation, so we overwrite that entry with a manually
  // crafted sequence.
  std::vector<const HloInstruction*> sequence_for_buffer_assigment = {
      input1,   weights1, one,     output1, tuple1, while1,  input0,
      weights0, zero,     output0, tuple0,  while0, root_add};

  // If this ASSERT_TRUE fails, we constructed a bogus sequence above
  // and this test itself is buggy.
  ASSERT_TRUE(IsPostOrderTraversal(sequence_for_buffer_assigment));

  sequence[module->entry_computation()] =
      std::move(sequence_for_buffer_assigment);

  auto assignment =
      BufferAssigner::Run(
          module.get(),
          xla::MakeUnique<SequentialHloOrdering>(module.get(), sequence),
          ByteSizeOf,
          [](LogicalBuffer::Color) { return 1; })
      .ConsumeValueOrDie();

  EXPECT_TRUE(BuffersDistinct({while0}, {while1}, *assignment));
}

// Test buffer assignment for while nodes with multiple uses.
// TODO(b/37245345): Fix buffer assignment for this case.
TEST_F(WhileBufferAssignmentTest, DISABLED_TwoWhiles) {
  auto module = xla::MakeUnique<HloModule>(TestName());
  auto builder = HloComputation::Builder(TestName());

  auto input0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape_, "input0"));
  auto weights0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "weights0"));

  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0)));
  auto output0 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));

  auto cond0 =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body0 =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));

  auto tuple0 = builder.AddInstruction(
      HloInstruction::CreateTuple({input0, weights0, output0}));
  auto while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond0, body0, tuple0));
  auto while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond0, body0, while0));

  auto get0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape_, while0, 2));
  auto get1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape_, while1, 2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(data_shape_, HloOpcode::kAdd, get0, get1));
  module->AddEntryComputation(builder.Build());

  RunCopyInsertion(module.get());

  {
    FlattenCallGraph flatten;
    TF_ASSERT_OK_AND_ASSIGN(bool result, flatten.Run(module.get()));
    EXPECT_TRUE(result);
  }

  auto assignment = RunBufferAssignment(module.get());

  EXPECT_TRUE(BuffersDistinct({while0}, {while1}, *assignment));
}

TEST_F(WhileBufferAssignmentTest, WhilesDontShareEntryParamIfLiveOut) {
  auto module = xla::MakeUnique<HloModule>(TestName());
  auto builder = HloComputation::Builder("entry");

  auto input0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, data_shape_, "input0"));
  auto weights0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "weights0"));

  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0)));
  auto output0 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));
  auto output1 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape_, zero, {1}));

  auto cond0 =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body0 =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));

  auto tuple0 = builder.AddInstruction(
      HloInstruction::CreateTuple({input0, weights0, output0}));
  auto while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond0, body0, tuple0));

  // Get output of 'while0' and feed as input to 'while1'.
  auto while0_out = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape_, while0, 2));

  auto cond1 =
      module->AddEmbeddedComputation(BuildWhileConditionComputation("cond"));
  auto body1 =
      module->AddEmbeddedComputation(BuildWhileBodyComputation("body"));

  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({while0_out, weights0, output1}));
  auto while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape_, cond1, body1, tuple1));

  // Get output of 'while1' so that it is live out of computation.
  auto while1_out = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape_, while1, 2));

  module->AddEntryComputation(builder.Build());
  RunCopyInsertion(module.get());
  auto assignment = RunBufferAssignment(module.get());
  // Get BufferAllocation for root instruction.
  auto* root_alloc = assignment->GetUniqueTopLevelSlice(while1_out)
                         .ConsumeValueOrDie()
                         .allocation();
  // Test that root instruction allocation is live out.
  EXPECT_TRUE(root_alloc->maybe_live_out());
  // Test that root instruction allocation is not an entry parameter.
  EXPECT_FALSE(root_alloc->is_entry_computation_parameter());
}

}  // namespace
}  // namespace xla
