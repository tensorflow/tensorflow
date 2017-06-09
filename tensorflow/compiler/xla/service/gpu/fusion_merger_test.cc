/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"

#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class FusionMergerTest : public HloTestBase {
 protected:
  FusionMergerTest() : module_(CreateNewModule()) {}

  // Builds the following computation:
  //
  //                 Param
  //               /   |   \
  //              /    |    \
  //  OnesVec  GTE(0) GTE(1) GTE(2)
  //       \   /         \   /
  //        Add           Add  OnesVec
  //         \           /  \  /
  //           \      Add   Mul  OnesVec
  //            \      |     |  /
  //             \    Mul    Add
  //              \    |    /
  //               \   |   /
  //                 Tuple
  //
  HloComputation* BuildComputation0() {
    auto builder = HloComputation::Builder(TestName() + ".Computation0");
    // Create param instruction to access computation state.
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape3_, "param"));

    // Create GetTupleElement instructions for each tuple element.
    auto gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, param, 0));
    auto gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, param, 1));
    auto gte2 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, param, 2));

    // Create const vector of ones to be used in element-wise computations.
    auto one_vec = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f, 1.f})));

    // Create simple fusable computation for tuple element 0 (wont get merged).
    auto out0 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, one_vec, gte0));

    // Create fusable computation which is dependent on second and third tuple
    // elements (will initially be fused on its own).
    auto add1 = builder.AddInstruction(
        HloInstruction::CreateBinary(data_shape_, HloOpcode::kAdd, gte1, gte2));

    // Create two sub-computations, both of which are users of 'add1'.

    // First sub-computation: out1 = Mul(Add(add1, one_vec), one_vec)
    auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, add1, one_vec));
    auto out1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, add2, one_vec));

    // Second sub-computation: out2 = Add(Mul(add1, one_vec), one_vec)
    auto mul0 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, add1, one_vec));
    auto out2 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, mul0, one_vec));

    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({out0, out1, out2}));
    return module_->AddEntryComputation(builder.Build());
  }

  // Builds the following computation:
  //
  //                 Param
  //               /      \
  //            GTE(0)   GTE(1)
  //            | | \   /
  //            | |  Mul
  //             \  \ |
  //              \  Mul
  //               \ |
  //      OnesVec   Mul  OnesVec
  //             \  /  \ /
  //     OnesVec  Add  Mul  OnesVec
  //            \  |    |  /
  //             Mul    Add
  //               \    /
  //                \  /
  //                Tuple
  //
  HloComputation* BuildComputation1() {
    auto builder = HloComputation::Builder(TestName() + ".Computation1");
    Shape tuple_shape2_ = ShapeUtil::MakeTupleShape({data_shape_, data_shape_});
    // Create param instruction to access computation state.
    auto state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape2_, "state"));

    // Create shared sub-computation (will initially be fused on its own).
    auto gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, state, 0));
    auto gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, state, 2));
    // Calculate the flops we need to generate for this shared computation
    // to exceed the threshold flops_to_bytes_ratio.
    // Note that bytes transferred is multiplied by 3 because there are two
    // operands and one output of size 'data_shape_'.
    const int64 flops_needed = FusionMerger::GetThresholdFlopsToBytesRatio() *
                               ShapeUtil::ByteSizeOf(data_shape_) * 3;
    const int64 vec_elements = ShapeUtil::ElementsIn(data_shape_);
    const int64 iters = (flops_needed + vec_elements - 1) / vec_elements;

    auto mul0 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, gte0, gte1));
    for (int i = 0; i < iters; ++i) {
      mul0 = builder.AddInstruction(HloInstruction::CreateBinary(
          data_shape_, HloOpcode::kMultiply, gte0, mul0));
    }

    // Create two sub-computations, both of which are users of 'mul0'.
    auto one_vec = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f, 1.f})));

    // First sub-computation: out0 = Mul(Add(mul0, one_vec), one_vec)
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, mul0, one_vec));
    auto out0 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, add0, one_vec));

    // Second sub-computation: out1 = Add(Mul(mul0, one_vec), one_vec)
    auto mul1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, mul0, one_vec));
    auto out1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, mul1, one_vec));

    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({out0, out1}));
    return module_->AddEntryComputation(builder.Build());
  }

  // Builds the following computation:
  //
  //                Param
  //             /   |   |  \
  //            /    |   |   \
  //           /     |   |    \
  //      GTE(0) GTE(1) GTE(2) GTE(3)
  //           \   /    /     /
  //            Add    /     /
  //              \   /     /
  //               Add     /
  //                 \    /
  //                  \  /
  //         OnesVec   Add  OnesVec
  //                \  /  \ /
  //        OnesVec  Add  Mul OnesVec
  //              \  |    |  /
  //               Mul    Add
  //                 \    /
  //                  \  /
  //                  Tuple
  //
  HloComputation* BuildComputation2(bool add_extra_input) {
    auto builder = HloComputation::Builder(TestName() + ".Computation2");
    Shape state_shape = add_extra_input ? tuple_shape4_ : tuple_shape3_;
    // Create param instruction to access computation state.
    auto state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, state_shape, "state"));

    // Create GetTupleElement instructions for each tuple element.
    auto gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, state, 0));
    auto gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, state, 1));
    auto gte2 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, state, 2));

    // Create shared fusable computation that reduces its operands.
    auto reduce0 = builder.AddInstruction(
        HloInstruction::CreateBinary(data_shape_, HloOpcode::kAdd, gte0, gte1));
    auto reduce_out = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, reduce0, gte2));
    if (add_extra_input) {
      auto gte3 = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(data_shape_, state, 3));
      reduce_out = builder.AddInstruction(HloInstruction::CreateBinary(
          data_shape_, HloOpcode::kAdd, reduce_out, gte3));
    }

    // Create two fusable sub-computations which are dependent on shared
    // computation 'reduce_out'.
    auto one_vec = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f, 1.f})));

    // First sub-computation: out0 = Mul(Add(reduce_out, one_vec), one_vec)
    auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, reduce_out, one_vec));
    auto out0 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, add2, one_vec));

    // Second sub-computation: out1 = Add(Mul(reduce_out, one_vec), one_vec)
    auto mul0 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kMultiply, reduce_out, one_vec));
    auto out1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, mul0, one_vec));

    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({out0, out1}));
    return module_->AddEntryComputation(builder.Build());
  }

  Shape data_shape_ = ShapeUtil::MakeShape(F32, {4});
  Shape tuple_shape2_ = ShapeUtil::MakeTupleShape({data_shape_, data_shape_});
  Shape tuple_shape3_ =
      ShapeUtil::MakeTupleShape({data_shape_, data_shape_, data_shape_});
  Shape tuple_shape4_ = ShapeUtil::MakeTupleShape(
      {data_shape_, data_shape_, data_shape_, data_shape_});

  std::unique_ptr<HloModule> module_;
};

// Tests that we can merge a fusion instruction that is below threshold.
//
// Original computation:
//
//                 Param
//                /  |  \
//               /   |   \
//  OnesVec  GTE(0) GTE(1) GTE(2)
//       \   /         \   /
//        Add           Add  OnesVec
//         \           /  \  /
//           \      Add   Mul  OnesVec
//            \      |     |  /
//             \    Mul    Add
//              \    |    /
//               \   |   /
//                 Tuple
//
// Computation after fusion passes:
//
//                  Param
//                 /     \
//            Fusion3    Fusion2
//               |       /     \
//                \ Fusion0  Fusion1
//                 \    |   /
//                  \   |  /
//                   Tuple
//
// Computation after fusion merger pass (Fusion2 is merged into Fusion0 and
// Fusion1):
//                   Param
//                 /   |   \
//          Fusion3 Fusion0 Fusion1
//                 \   |   /
//                   Tuple
//
TEST_F(FusionMergerTest, MergeSharedFusionInstruction) {
  auto computation = BuildComputation0();
  // Run standard fusion passes.
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/false)
                  .Run(module_.get())
                  .ValueOrDie());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module_.get())
                   .ValueOrDie());
  // Run fusion merger pass, which should merge the shared fusion instruction
  // into its two users.
  EXPECT_TRUE(FusionMerger().Run(module_.get()).ValueOrDie());

  auto* root = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kTuple, root->opcode());
  // Check operand 0 (not merged). Should have 4 instructions.
  auto* operand0 = root->operand(0);
  EXPECT_EQ(HloOpcode::kFusion, operand0->opcode());
  EXPECT_EQ(4, operand0->fused_instructions().size());
  // Check operand 1 (should have merged in its operand fusion instruction).
  auto* operand1 = root->operand(1);
  EXPECT_EQ(HloOpcode::kFusion, operand1->opcode());
  EXPECT_EQ(7, operand1->fused_instructions().size());
  // Check operand 2 (should have merged in its operand fusion instruction).
  auto* operand2 = root->operand(2);
  EXPECT_EQ(HloOpcode::kFusion, operand2->opcode());
  EXPECT_EQ(7, operand2->fused_instructions().size());
}

// Tests that we do not merge a fusion instruction that above flops to bytes
// threshold.
//
// Original computation:
//
//                 Param
//                /     \
//            GTE(0)   GTE(1)
//            | | \   /
//            | |  Mul
//             \  \ |
//              \  Mul
//               \ |
//      OnesVec   Mul  OnesVec
//             \  /  \ /
//     OnesVec  Add  Mul  OnesVec
//            \  |    |  /
//             Mul    Add
//               \    /
//                \  /
//                Tuple
//
// Computation after fusion passes and fusion merger pass (Fusion2 is not
// merged because it exceeds the threshold flops to bytes ratio).
//
//                 Param
//                   |
//                Fusion2
//                /     \
//           Fusion0  Fusion1
//                \    /
//                 Tuple
//
TEST_F(FusionMergerTest, FlopsToBytesRatioThresholdExceeded) {
  BuildComputation1();
  // Run standard fusion passes.
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/false)
                  .Run(module_.get())
                  .ValueOrDie());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module_.get())
                   .ValueOrDie());
  // Run fusion merger pass, which should detect that the flops/bytes of the
  // shared fusion instruction exceeds the threshold ratio, and therefore
  // cannot be merged with other fusion instructions.
  EXPECT_FALSE(FusionMerger().Run(module_.get()).ValueOrDie());
}

// Tests that threshold for bytes transferred if merged is exceeded.
//
// Original computation:
//
//                Param
//             /   |   |  \
//            /    |   |   \
//           /     |   |    \
//      GTE(0) GTE(1) GTE(2) GTE(3)
//           \   /    /     /
//            Add    /     /
//              \   /     /
//               Add     /
//                 \    /
//                  \  /
//         OnesVec   Add  OnesVec
//                \  /  \ /
//        OnesVec  Add  Mul OnesVec
//              \  |    |  /
//               Mul    Add
//                 \    /
//                  \  /
//                  Tuple
//
// Computation after fusion passes and fusion merger pass. Fusion2 is not
// merged because it exceeds the threshold bytes transferred. This is because
// the bytes read by Fusion2 (when replicated if the instruction is merged
// into Fusion0 and Fusion1) would exceed the bytes transferred threshold.
//
//                 Param
//                   |
//                Fusion2
//                /     \
//           Fusion0  Fusion1
//                \    /
//                 Tuple
//
TEST_F(FusionMergerTest, BytesTransferredThresholdExeceeded) {
  BuildComputation2(/*add_extra_input=*/true);
  // Run standard fusion passes.
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/false)
                  .Run(module_.get())
                  .ValueOrDie());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module_.get())
                   .ValueOrDie());
  // Run fusion merger pass, which should detect that the net bytes transferred
  // (if merged) would increase.
  EXPECT_FALSE(FusionMerger().Run(module_.get()).ValueOrDie());
}

// Tests that threshold for bytes transferred if merged is not exceeded.
//
// Original computation:
//
//               Param
//             /   |  \
//            /    |   \
//           /     |    \
//      GTE(0) GTE(1) GTE(2)
//           \   /    /
//            Add    /
//              \   /
//     OnesVec   Add  OnesVec
//            \  /  \ /
//   OnesVec  Add   Mul OnesVec
//              \  /   \  /
//               Mul    Add
//                 \    /
//                  \  /
//                  Tuple
//
// Computation after fusion passes:
//
//                 Param
//                   |
//                Fusion2
//                /     \
//           Fusion0  Fusion1
//                \    /
//                 Tuple
//
// Computation after fusion merger pass (Fusion2 is merged into Fusion0 and
// Fusion1, because bytes read from Param by Fusion2 is reduced for this test
// which makes the merge operation into its operand below the bytes
// transferred threshold.
//
//                   Param
//                   /  \
//             Fusion0  Fusion1
//                   \    /
//                   Tuple
//
TEST_F(FusionMergerTest, BytesTransferredThresholdNotExeceeded) {
  BuildComputation2(/*add_extra_input=*/false);
  // Run standard fusion passes.
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/false)
                  .Run(module_.get())
                  .ValueOrDie());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module_.get())
                   .ValueOrDie());
  // Run fusion merger pass, which should detect that the net bytes transferred
  // (if merged) would not increase.
  EXPECT_TRUE(FusionMerger().Run(module_.get()).ValueOrDie());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char** argv) {
  return xla::ParseDebugOptionsFlagsAndRunTests(argc, argv);
}
