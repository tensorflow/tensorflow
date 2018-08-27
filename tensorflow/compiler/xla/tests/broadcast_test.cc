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

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class BroadcastTest : public HloTestBase {};

XLA_TEST_F(BroadcastTest, BroadcastScalarToScalar) {
  // Test degenerate case of broadcasting a scalar into a scalar.
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {}), input, {}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(LiteralTestUtil::Near(*LiteralUtil::CreateR0<float>(42.0),
                                    *result, error_spec_));
}

XLA_TEST_F(BroadcastTest, BroadcastScalarTo2D) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 2}), input, {}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(LiteralTestUtil::Near(
      *LiteralUtil::CreateR2<float>({{42.0, 42.0}, {42.0, 42.0}}), *result,
      error_spec_));
}

XLA_TEST_F(BroadcastTest, BroadcastVectorTo2D) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));

  // Broadcast vector in both dimension 0 and dimension 1. Join them in a tuple
  // to enable testing of the results.
  auto element1 = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2}), input, {0}));
  auto element2 = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 3}), input, {1}));
  builder.AddInstruction(HloInstruction::CreateTuple({element1, element2}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(LiteralTestUtil::Near(
      *LiteralUtil::CreateR2<float>({{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}}),
      LiteralSlice(*result, {0}), error_spec_));

  EXPECT_TRUE(LiteralTestUtil::Near(
      *LiteralUtil::CreateR2<float>({{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}),
      LiteralSlice(*result, {1}), error_spec_));
}

XLA_TEST_F(BroadcastTest, Broadcast2DTo2D) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 2}), input, {0, 1}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(LiteralTestUtil::Near(
      *LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}), *result,
      error_spec_));
}

XLA_TEST_F(BroadcastTest, Broadcast2DTo2DTranspose) {
  // Degenerately broadcasting a shape into a shape of the same rank reorders
  // the dimensions, ie transpose.
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 2}), input, {1, 0}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(LiteralTestUtil::Near(
      *LiteralUtil::CreateR2<float>({{1.0, 3.0}, {2.0, 4.0}}), *result,
      error_spec_));
}

XLA_TEST_F(BroadcastTest, Broadcast2DTo3D) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 3, 2}), input, {0, 2}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(LiteralTestUtil::Near(
      *LiteralUtil::CreateR3<float>({{{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}},
                                     {{3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}}}),
      *result, error_spec_));
}

TEST_F(BroadcastTest, Broadcast_R1_2_To_R4_2x2x3x3) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1.0, 2.0})));

  // Broadcast vector in dimension 1.
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 2, 3, 3}), input, {1}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  Array4D<float> expected(2, 2, 3, 3);
  Array2D<float> pz({{1, 2}, {1, 2}});
  expected.FillWithPZ(pz);

  EXPECT_TRUE(
      LiteralTestUtil::Near(*LiteralUtil::CreateR4FromArray4D<float>(expected),
                            *result, error_spec_));
}

TEST_F(BroadcastTest, Broadcast_R1_1025_To_R4_3x3x3x1025) {
  auto builder = HloComputation::Builder(TestName());
  std::vector<float> input_data(1025);
  int64 r1_size = input_data.size();
  std::iota(input_data.begin(), input_data.end(), 0.0f);
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(input_data)));

  // Broadcast vector in dimension 3.
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 3, 3, r1_size}), input, {3}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  Array4D<float> expected(3, 3, 3, 1025);
  Array2D<float> yx(3, r1_size);
  for (int64 y = 0; y < 3; ++y) {
    for (int64 x = 0; x < r1_size; ++x) {
      yx(y, x) = input_data[x];
    }
  }
  expected.FillWithYX(yx);

  EXPECT_TRUE(
      LiteralTestUtil::Near(*LiteralUtil::CreateR4FromArray4D<float>(expected),
                            *result, error_spec_));
}

XLA_TEST_F(BroadcastTest, Broadcast_R1_64_To_R4_32x64x7x7) {
  auto builder = HloComputation::Builder(TestName());
  Array4D<float> r4_array(32, 64, 7, 7);
  r4_array.Fill(42.0);
  std::vector<float> r1_array(64, 42.0);

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(r1_array)));

  // Broadcast vector in dimension 1.
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {32, 64, 7, 7}), input, {1}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(LiteralTestUtil::Near(*LiteralUtil::CreateR4FromArray4D(r4_array),
                                    *result, error_spec_));
}

TEST_F(BroadcastTest, Broadcast_R0_to_R4_64x64x3x3) {
  auto builder = HloComputation::Builder(TestName());
  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {64, 64, 3, 3}), input, {}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  LOG(INFO) << hlo_module->ToString();
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  Array4D<float> expected(64, 64, 3, 3);
  expected.Fill(1.0f);

  EXPECT_TRUE(
      LiteralTestUtil::Near(*LiteralUtil::CreateR4FromArray4D<float>(expected),
                            *result, error_spec_));
}

TEST_F(BroadcastTest, Broadcast_R2_2x2_To_R4_3x3x2x2) {
  auto builder = HloComputation::Builder(TestName());
  Array2D<float> to_broadcast({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<float>(to_broadcast)));

  // Broadcast vector in dimensions 2 and 3.
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 3, 2, 2}), input, {2, 3}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  Array4D<float> expected(3, 3, 2, 2);
  expected.FillWithYX(to_broadcast);

  EXPECT_TRUE(
      LiteralTestUtil::Near(*LiteralUtil::CreateR4FromArray4D<float>(expected),
                            *result, error_spec_));
}

TEST_F(BroadcastTest, Broadcast_R3_2x3x4_to_R4_2x3x4x5) {
  auto builder = HloComputation::Builder(TestName());
  Array3D<float> input_vals(2, 3, 4);
  input_vals.FillRandom(1.0);

  Array4D<float> expected(2, 3, 4, 5);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int m = 0; m < 5; ++m) {
          expected(i, j, k, m) = input_vals(i, j, k);
        }
      }
    }
  }
  auto input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR3FromArray3D<float>(input_vals)));

  // Broadcast vector in dimensions 2 and 3.
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 3, 4, 5}), input, {0, 1, 2}));

  // Create HLO module, compile, and execute.
  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});

  EXPECT_TRUE(
      LiteralTestUtil::Near(*LiteralUtil::CreateR4FromArray4D<float>(expected),
                            *result, error_spec_));
}

}  // namespace
}  // namespace xla
