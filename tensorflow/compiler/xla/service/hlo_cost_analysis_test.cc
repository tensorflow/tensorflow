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

#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/service/computation_tracker.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/service/user_computation.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"

namespace xla {
namespace {

constexpr int64 kPointerSize = 8;

int64 ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

// This test suite tests the HLO cost analysis by first building a computation
// using the client computation builder and running the HloCostAnalysis that
// returns the number of floating point and transcendental operations in the
// graph. We test both individual HLO operations as well as a mixed graph.
class HloCostAnalysisTest : public ::testing::Test {
 protected:
  HloCostAnalysisTest()
      : client_(ClientLibrary::LocalClientOrDie()),
        // Accessing service instance is required for the unit tests to enable
        // whitebox accesses to the user computation built from the client,
        // as shown in the BuildHloGraph functions below.
        service_(static_cast<Service*>(ClientLibrary::GetXlaService(
            static_cast<LocalClient*>(client_)->platform()))),
        computation_tracker_(service_->computation_tracker()) {
    // Create a computation for a unary user function: x => exp(x + 0.5)
    {
      ComputationBuilder builder(client_, "add_and_exp");
      auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
      auto half = builder.ConstantR0<float>(0.5);
      builder.Exp(builder.Add(x, half));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_and_exp_ = computation_status.ConsumeValueOrDie();
    }

    // Create a computation for a binary user function: (x, y) => x + y
    {
      ComputationBuilder builder(client_, "add");
      auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
      builder.Add(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_ = computation_status.ConsumeValueOrDie();
    }

    // Create a computation for a sigmoid function: x => 1 / (1 + exp(-x))
    {
      ComputationBuilder builder(client_, "sigmoid");
      auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
      auto one = builder.ConstantR0<float>(1.0);
      builder.Div(one, builder.Add(one, builder.Exp(builder.Neg(x))));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      sigmoid_ = computation_status.ConsumeValueOrDie();
    }

    // Create a computation for a binary max function: (x, y) => max (x, y)
    {
      ComputationBuilder builder(client_, "max");
      auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
      builder.Max(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      max_ = computation_status.ConsumeValueOrDie();
    }

    // Create a computation for a binary GT function: (x, y) => x > y
    {
      ComputationBuilder builder(client_, "gt");
      auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
      builder.Gt(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      gt_ = computation_status.ConsumeValueOrDie();
    }
  }

  // Build HLO graph from the given builder and return the HLO module.
  std::unique_ptr<HloModule> BuildHloGraph(ComputationBuilder* builder) {
    auto computation_status = builder->Build();
    TF_CHECK_OK(computation_status.status());
    auto computation = computation_status.ConsumeValueOrDie();
    auto user_computation_status =
        computation_tracker_.Resolve(computation.handle());
    TF_CHECK_OK(user_computation_status.status());
    auto user_computation = user_computation_status.ConsumeValueOrDie();
    VersionedComputationHandle versioned_handle =
        user_computation->GetVersionedHandle();
    return std::move(
        computation_tracker_.BuildHloModule(versioned_handle, HloModuleConfig())
            .ValueOrDie());
  }

  Client* client_;
  Service* service_;
  const ComputationTracker& computation_tracker_;

  // User computations used for higher order operations (e.g., Map, Reduce).
  Computation add_;
  Computation add_and_exp_;
  Computation sigmoid_;
  Computation max_;
  Computation gt_;
};

TEST_F(HloCostAnalysisTest, MatrixMultiply) {
  ComputationBuilder builder(client_, "matrix_multiply");
  auto lhs = builder.Parameter(0, ShapeUtil::MakeShape(F32, {10, 5}), "lhs");
  auto rhs = builder.Parameter(1, ShapeUtil::MakeShape(F32, {5, 30}), "rhs");
  auto result = builder.Dot(lhs, rhs);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Check the number of computations returned from the analysis (1500 FMAs).
  EXPECT_EQ(analysis.flop_count(), 2 * 10 * 30 * 5);

  EXPECT_EQ(analysis.transcendental_count(), 0);

  // Bytes accessed is sum of inputs and output.
  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (10 * 5 + 5 * 30 + 10 * 30));
}

TEST_F(HloCostAnalysisTest, Map) {
  ComputationBuilder builder(client_, "map");
  auto input = builder.Parameter(0, ShapeUtil::MakeShape(F32, {10}), "in");
  auto result = builder.Map({input}, add_and_exp_, {0});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // add contributes to 10 flops and exp contributes to 10 transcendental ops.
  EXPECT_EQ(analysis.flop_count(), 10);
  EXPECT_EQ(analysis.transcendental_count(), 10);
  EXPECT_EQ(analysis.bytes_accessed(), 80);
}

TEST_F(HloCostAnalysisTest, Convolution) {
  ComputationBuilder builder(client_, "convolution");
  auto input = builder.Parameter(
      0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/10,
                                 /*x_dim=*/20}),
      "input");
  auto kernel = builder.Parameter(
      1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/3,
                                 /*x_dim=*/3}),
      "kernel");
  auto result = builder.Conv(input, kernel, {1, 1}, Padding::kValid);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Output shape is [1x1x8x18] and each output element requires (3x3)
  // FMAs and one FMA is 2 flops.
  EXPECT_EQ(analysis.flop_count(), 8 * 18 * 2 * 3 * 3);

  // Bytes accessed is sum of inputs and output.
  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (10 * 20 + 3 * 3 + 8 * 18));
}

TEST_F(HloCostAnalysisTest, Reduce) {
  ComputationBuilder builder(client_, "reduce");
  auto input =
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {10, 20}), "input");
  auto result =
      builder.Reduce(input, builder.ConstantR0<float>(0.0f), add_, {1});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Subtracting the output size from the input size gives the number of
  // reduction operations performed.
  EXPECT_EQ(analysis.flop_count(), 10 * 20 - 10);
}

TEST_F(HloCostAnalysisTest, ReduceWindow) {
  ComputationBuilder builder(client_, "reduce_window");
  auto input =
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {10, 20}), "input");
  auto result = builder.ReduceWindow(input, builder.ConstantR0<float>(0), add_,
                                     {4, 5}, {4, 5}, Padding::kValid);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Each of [2x4] output elements are generated from reducing [4x5] elements.
  EXPECT_EQ(analysis.flop_count(), 2 * 4 * (4 * 5 - 1));
}

TEST_F(HloCostAnalysisTest, SelectAndScatter) {
  ComputationBuilder builder(client_, "select_and_scatter");
  auto operand =
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {10, 20}), "input");
  auto source =
      builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 4}), "source");
  auto result =
      builder.SelectAndScatter(operand, gt_, {4, 5}, {4, 5}, Padding::kValid,
                               source, builder.ConstantR0<float>(0), add_);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Each of [2x4] source elements computes its destination from reducing [4x5]
  // elements followed by the scatter computation.
  EXPECT_EQ(analysis.flop_count(), 2 * 4 * (4 * 5 - 1 + 1));
}

TEST_F(HloCostAnalysisTest, Broadcast) {
  ComputationBuilder b(client_, "broadcast");
  b.Broadcast(b.ConstantR0<float>(42), {10, 7});
  auto hlo_module = BuildHloGraph(&b);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));
  EXPECT_EQ(analysis.flop_count(), 0);
}

// Calculates the computation cost of a graph with more than one HLO node.
TEST_F(HloCostAnalysisTest, FullyConnectedForward) {
  ComputationBuilder builder(client_, "fully_connected_forward");
  auto input =
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {10, 5}), "input");
  auto weight =
      builder.Parameter(1, ShapeUtil::MakeShape(F32, {5, 20}), "weight");
  auto bias = builder.Parameter(2, ShapeUtil::MakeShape(F32, {20}), "bias");
  // sigmoid(input * weight + bias)
  auto result = builder.Map(
      {builder.Add(builder.Dot(input, weight), bias, {1})}, sigmoid_, {0, 1});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // 1000 FMAs from matrix multiplication, 200 flops from bias addition,
  // 600 flops from sigmoid, and 200 transcendental ops from sigmoid.
  EXPECT_EQ(analysis.flop_count(), 2 * 1000 + 200 + 3 * 200);
  EXPECT_EQ(analysis.transcendental_count(), 200);
}

TEST_F(HloCostAnalysisTest, MatmulAndConvolutionCanBeTheSameComputation) {
  HloCostAnalysis conv_analysis(ShapeSize);
  {
    ComputationBuilder builder(client_, "conv_looking_matmul");
    auto lhs = builder.Parameter(0, ShapeUtil::MakeShape(F32, {64, 64, 1, 1}),
                                 "input");
    auto rhs = builder.Parameter(1, ShapeUtil::MakeShape(F32, {64, 64, 1, 1}),
                                 "weights");
    builder.Conv(lhs, rhs, {1, 1}, Padding::kSame);
    auto hlo_module = BuildHloGraph(&builder);
    ASSERT_IS_OK(hlo_module->entry_computation()->root_instruction()->Accept(
        &conv_analysis));
  }

  HloCostAnalysis matmul_analysis(ShapeSize);
  {
    ComputationBuilder builder(client_, "matmul");
    auto lhs =
        builder.Parameter(0, ShapeUtil::MakeShape(F32, {64, 64}), "input");
    auto rhs =
        builder.Parameter(1, ShapeUtil::MakeShape(F32, {64, 64}), "weights");
    builder.Dot(lhs, rhs);
    auto hlo_module = BuildHloGraph(&builder);
    ASSERT_IS_OK(hlo_module->entry_computation()->root_instruction()->Accept(
        &matmul_analysis));
  }

  EXPECT_EQ(conv_analysis.flop_count(), matmul_analysis.flop_count());
}

using FusionCostAnalysis = HloTestBase;

TEST_F(FusionCostAnalysis, LoopFusion) {
  // Do this 4 times with different per-second rates to test the computation of
  // bottleneck time on fusion nodes.
  for (int i = 0; i < 4; ++i) {
    Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});

    // Fuse all instructions in complicated expression:
    //
    //   add = Add(C1, C2)
    //   clamp = Clamp(C2, add, add)
    //   exp = Exp(add)
    //   mul = Mul(exp, C3)
    //   sub = Sub(mul, clamp)
    //   tuple = Tuple({sub, sub, mul, C1})
    HloComputation::Builder builder(TestName());
    auto c1 = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR2F32Linspace(
            /*from=*/0.0f, /*to=*/1.0f, /*rows=*/2, /*cols=*/2)));
    auto c2 = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR2F32Linspace(
            /*from=*/1.0f, /*to=*/2.0f, /*rows=*/2, /*cols=*/2)));
    auto c3 = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR2F32Linspace(
            /*from=*/2.0f, /*to=*/3.0f, /*rows=*/2, /*cols=*/2)));
    auto add = builder.AddInstruction(
        HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, c1, c2));
    auto clamp = builder.AddInstruction(
        HloInstruction::CreateTernary(r2f32, HloOpcode::kClamp, c2, add, add));
    auto exp = builder.AddInstruction(
        HloInstruction::CreateUnary(r2f32, HloOpcode::kExp, add));
    auto mul = builder.AddInstruction(
        HloInstruction::CreateBinary(r2f32, HloOpcode::kMultiply, exp, c3));
    auto sub = builder.AddInstruction(
        HloInstruction::CreateBinary(r2f32, HloOpcode::kSubtract, mul, clamp));
    auto tuple = HloInstruction::CreateTuple({sub, sub, mul, c1});

    HloModule module(TestName());
    auto* computation = module.AddEntryComputation(builder.Build());
    auto* fusion = computation->CreateFusionInstruction(
        {sub, mul, exp, clamp, add}, HloInstruction::FusionKind::kLoop);

    // The time given these rates at i == 0 is exactly even among the properties
    // at 1.0 seconds. For other values, one of the rates is slower so that it
    // becomes the bottleneck.
    HloCostAnalysis fusion_analysis(ShapeSize);
    fusion_analysis.set_flops_per_second(16 * (i == 1 ? 1 / 2.0 : 1.0));
    fusion_analysis.set_transcendentals_per_second(4 *
                                                   (i == 2 ? 1 / 4.0 : 1.0));
    fusion_analysis.set_bytes_per_second(64 * (i == 3 ? 1 / 8.0 : 1.0));
    ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

    EXPECT_EQ(fusion_analysis.flop_count(), 16);
    EXPECT_EQ(fusion_analysis.transcendental_count(), 4);
    constexpr int64 bytes_accessed = sizeof(float) * 4 * 2 * 2;
    static_assert(bytes_accessed == 64, "");
    EXPECT_EQ(fusion_analysis.bytes_accessed(), bytes_accessed);

    EXPECT_EQ(fusion_analysis.optimal_seconds(), 1 << i);
  }
}

TEST_F(FusionCostAnalysis, NoLayout) {
  Shape shape_with_layout = ShapeUtil::MakeShape(F32, {2, 3, 4, 5});
  // Instructions within a fused op may have no layout.
  Shape shape_without_layout = shape_with_layout;
  shape_without_layout.clear_layout();

  HloComputation::Builder builder(TestName());
  auto c1 = builder.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR4FromArray4D(Array4D<float>(2, 3, 4, 5))));
  auto c2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({1, 2, 3})));

  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(shape_without_layout, c2, {1}));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_with_layout, HloOpcode::kAdd, c1, broadcast));

  HloModule module(TestName());
  auto* computation = module.AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {add, broadcast}, HloInstruction::FusionKind::kLoop);

  HloCostAnalysis fusion_analysis(ShapeSize);
  ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

  EXPECT_EQ(fusion_analysis.flop_count(), 120);
  EXPECT_EQ(fusion_analysis.transcendental_count(), 0);
}

TEST_F(HloCostAnalysisTest, TupleCost) {
  HloCostAnalysis analysis(ShapeSize);
  {
    ComputationBuilder builder(client_, "matmul");
    auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {123}), "x");
    auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {42}), "y");
    auto tuple = builder.Tuple({x, y});
    auto hlo_module = BuildHloGraph(&builder);

    ASSERT_IS_OK(
        hlo_module->entry_computation()->root_instruction()->Accept(&analysis));
  }

  EXPECT_EQ(analysis.flop_count(), 0);
  EXPECT_EQ(analysis.transcendental_count(), 0);
  EXPECT_EQ(analysis.bytes_accessed(), kPointerSize * 2);
}

TEST_F(HloCostAnalysisTest, BaseDilatedConvolution) {
  ComputationBuilder builder(client_, "BaseDilatedConvolution");
  auto input = builder.Parameter(
      0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/10,
                                 /*x_dim=*/20}),
      "input");
  auto kernel = builder.Parameter(
      1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/3,
                                 /*x_dim=*/3}),
      "kernel");

  auto result = builder.ConvGeneralDilated(
      input, kernel, /*window_strides=*/{1, 1}, /*padding=*/{{1, 1}, {1, 1}},
      /*lhs_dilation=*/{3, 5}, /*rhs_dilation=*/{7, 11},
      ComputationBuilder::CreateDefaultConvDimensionNumbers(2));

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 1472);
}

}  // namespace
}  // namespace xla
