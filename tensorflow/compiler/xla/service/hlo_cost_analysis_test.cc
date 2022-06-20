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
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

constexpr int64_t kPointerSize = 8;

int64_t ShapeSize(const Shape& shape) {
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
            static_cast<LocalClient*>(client_)->platform()))) {
    // Create a computation for a unary user function: x => exp(x + 0.5)
    {
      XlaBuilder builder("add_and_exp");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto half = ConstantR0<float>(&builder, 0.5);
      Exp(Add(x, half));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_and_exp_ = std::move(computation_status).value();
    }

    // Create a computation for a binary user function: (x, y) => x + y
    {
      XlaBuilder builder("add");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
      Add(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_ = std::move(computation_status).value();
    }

    // Create a computation for a sigmoid function: x => 1 / (1 + exp(-x))
    {
      XlaBuilder builder("sigmoid");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto one = ConstantR0<float>(&builder, 1.0);
      Div(one, Add(one, Exp(Neg(x))));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      sigmoid_ = std::move(computation_status).value();
    }

    // Create a computation for a binary max function: (x, y) => max (x, y)
    {
      XlaBuilder builder("max");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
      Max(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      max_ = std::move(computation_status).value();
    }

    // Create a computation for a binary GT function: (x, y) => x > y
    {
      XlaBuilder builder("gt");
      auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x");
      auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {}), "y");
      Gt(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      gt_ = std::move(computation_status).value();
    }
  }

  // Build HLO graph from the given builder and return the HLO module.
  std::unique_ptr<HloModule> BuildHloGraph(XlaBuilder* builder) {
    auto computation_status = builder->Build();
    TF_CHECK_OK(computation_status.status());
    auto computation = std::move(computation_status).value();
    auto config = HloModule::CreateModuleConfigFromProto(computation.proto(),
                                                         DebugOptions())
                      .value();
    return HloModule::CreateFromProto(computation.proto(), config).value();
  }

  Client* client_;
  Service* service_;

  // User computations used for higher order operations (e.g., Map, Reduce).
  XlaComputation add_;
  XlaComputation add_and_exp_;
  XlaComputation sigmoid_;
  XlaComputation max_;
  XlaComputation gt_;
};

using HloCostAnalysisHloTest = HloTestBase;

TEST_F(HloCostAnalysisTest, MatrixMultiply) {
  XlaBuilder builder("matrix_multiply");
  auto lhs = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 5}), "lhs");
  auto rhs = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {5, 30}), "rhs");
  Dot(lhs, rhs);

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

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 5);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 5 * 30);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10 * 30);
}

TEST_F(HloCostAnalysisTest, DotGeneral) {
  XlaBuilder builder("matrix_multiply");
  auto lhs =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 5, 5}), "lhs");
  auto rhs =
      Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {5, 5, 30}), "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(0);
  dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs, rhs, dnums);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Check the number of computations returned from the analysis (1500 FMAs).
  EXPECT_EQ(analysis.flop_count(), 2 * 10 * 30 * 5 * 5);

  EXPECT_EQ(analysis.transcendental_count(), 0);

  // Bytes accessed is sum of inputs and output.
  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (10 * 5 * 5 + 5 * 5 * 30 + 10 * 30));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0),
            sizeof(float) * 10 * 5 * 5);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1),
            sizeof(float) * 5 * 5 * 30);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10 * 30);
}

TEST_F(HloCostAnalysisTest, DotGeneral2) {
  XlaBuilder builder("matrix_multiply");
  auto lhs =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 5, 5}), "lhs");
  auto rhs =
      Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {5, 5, 30}), "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(2);
  dnums.add_rhs_contracting_dimensions(0);
  dnums.add_rhs_batch_dimensions(1);
  DotGeneral(lhs, rhs, dnums);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Check the number of computations returned from the analysis (1500 FMAs).
  EXPECT_EQ(analysis.flop_count(), 2 * 10 * 30 * 5 * 5);

  EXPECT_EQ(analysis.transcendental_count(), 0);

  // Bytes accessed is sum of inputs and output.
  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (10 * 5 * 5 + 5 * 5 * 30 + 5 * 10 * 30));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0),
            sizeof(float) * 10 * 5 * 5);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1),
            sizeof(float) * 5 * 5 * 30);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 5 * 10 * 30);
}

TEST_F(HloCostAnalysisTest, DotGeneral3) {
  XlaBuilder builder("matrix_multiply");
  auto lhs = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 5}), "lhs");
  auto rhs = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {5, 30}), "rhs");
  DotDimensionNumbers dnums;
  DotGeneral(lhs, rhs, dnums);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Check the number of computations returned from the analysis (1500 FMAs).
  EXPECT_EQ(analysis.flop_count(), 2 * 10 * 30 * 5 * 5);

  EXPECT_EQ(analysis.transcendental_count(), 0);

  // Bytes accessed is sum of inputs and output.
  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (10 * 5 + 5 * 30 + 5 * 5 * 10 * 30));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 5);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 5 * 30);
  EXPECT_EQ(analysis.output_bytes_accessed(*root),
            sizeof(float) * 5 * 5 * 10 * 30);
}

TEST_F(HloCostAnalysisTest, Map) {
  XlaBuilder builder("map");
  auto input = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10}), "in");
  Map(&builder, {input}, add_and_exp_, {0});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // add contributes to 10 flops and exp contributes to 10 transcendental ops.
  EXPECT_EQ(analysis.flop_count(), 10);
  EXPECT_EQ(analysis.transcendental_count(), 10);
  EXPECT_EQ(analysis.bytes_accessed(), 80);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10);
}

TEST_F(HloCostAnalysisTest, Convolution) {
  XlaBuilder builder("convolution");
  auto input = Parameter(
      &builder, 0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/10,
                                 /*x_dim=*/20}),
      "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/3,
                                 /*x_dim=*/3}),
      "kernel");
  Conv(input, kernel, {1, 1}, Padding::kValid);

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

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 20);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 3 * 3);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 8 * 18);
}

TEST_F(HloCostAnalysisTest, ConvolutionSame) {
  XlaBuilder builder("convolution_same");
  const int iw = 3;
  const int ih = 3;
  const int kw = 3;
  const int kh = 3;
  const int ow = iw;
  const int oh = ih;
  const int sx = 1;
  const int sy = 1;
  auto input = Parameter(
      &builder, 0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/ih,
                                 /*x_dim=*/iw}),
      "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kh,
                                 /*x_dim=*/kw}),
      "kernel");
  Conv(input, kernel, {sx, sy}, Padding::kSame);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Output shape is [1x1x3x3] and each output element requires (3x3)
  // FMAs and one FMA is 2 flops.
  // NOTE: This formula only works for the hard-coded dimensions for now.
  EXPECT_EQ(analysis.flop_count(), 2 * (4 + 6 + 4 + 6 + 9 + 6 + 4 + 6 + 4));

  // Bytes accessed is sum of inputs and output.
  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (iw * ih + kw * kh + ow * oh));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * iw * ih);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * kw * kh);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * ow * oh);
}

TEST_F(HloCostAnalysisTest, ConvolutionExtreme) {
  XlaBuilder builder("convolution");
  constexpr int64_t kLarge = 512 * 1024;
  auto input = Parameter(
      &builder, 0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kLarge}),
      "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kLarge}),
      "kernel");
  ConvGeneralDilated(input, kernel, {kLarge - 1}, {{0, 0}}, {kLarge}, {1},
                     XlaBuilder::CreateDefaultConvDimensionNumbers(1));

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 2 * kLarge);
}

TEST_F(HloCostAnalysisTest, ConvolutionExtreme2) {
  XlaBuilder builder("convolution");
  constexpr int64_t kLarge = 512 * 1024;
  auto input = Parameter(
      &builder, 0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/1}),
      "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kLarge}),
      "kernel");
  ConvGeneralDilated(input, kernel, {1}, {{kLarge - 1, kLarge - 1}}, {1}, {1},
                     XlaBuilder::CreateDefaultConvDimensionNumbers(1));

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 2 * kLarge);
}

TEST_F(HloCostAnalysisTest, ConvolutionWithFeatureGroup) {
  XlaBuilder builder("convolution");
  auto input = Parameter(
      &builder, 0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/120, /*y_dim=*/10,
                                 /*x_dim=*/20}),
      "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/120, /*z_dim=*/1, /*y_dim=*/3,
                                 /*x_dim=*/3}),
      "kernel");
  Conv(input, kernel, {1, 1}, Padding::kValid, /*feature_group_count=*/120);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Output shape is [1x120x8x18] and each output element requires (3x3)
  // FMAs and one FMA is 2 flops.
  EXPECT_EQ(analysis.flop_count(), 120 * 8 * 18 * 2 * 3 * 3);

  // Bytes accessed is sum of inputs and output.
  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (120 * 10 * 20 + 120 * 3 * 3 + 120 * 8 * 18));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0),
            sizeof(float) * 120 * 10 * 20);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1),
            sizeof(float) * 120 * 3 * 3);
  EXPECT_EQ(analysis.output_bytes_accessed(*root),
            sizeof(float) * 120 * 8 * 18);
}

TEST_F(HloCostAnalysisTest, Reduce) {
  XlaBuilder builder("reduce");
  auto input =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 20}), "input");
  Reduce(input, ConstantR0<float>(&builder, 0.0f), add_, {1});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Subtracting the output size from the input size gives the number of
  // reduction operations performed.
  EXPECT_EQ(analysis.flop_count(), 10 * 20 - 10);

  EXPECT_EQ(analysis.bytes_accessed(), sizeof(float) * (10 * 20 + 1 + 10));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 20);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10);
}

TEST_F(HloCostAnalysisTest, ReduceWindow) {
  XlaBuilder builder("reduce_window");
  auto input =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 20}), "input");
  ReduceWindow(input, ConstantR0<float>(&builder, 0), add_, {4, 5}, {4, 5},
               Padding::kValid);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Each of [2x4] output elements are generated from reducing [4x5] elements.
  EXPECT_EQ(analysis.flop_count(), 2 * 4 * (4 * 5 - 1));

  EXPECT_EQ(analysis.bytes_accessed(), sizeof(float) * (10 * 20 + 1 + 2 * 4));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 20);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 2 * 4);
}

TEST_F(HloCostAnalysisTest, ReduceWindowVariadic) {
  XlaBuilder builder("reduce_window_variadic");
  auto elem_shape = ShapeUtil::MakeShape(F32, {});
  auto p2 = Parameter(&builder, 0, elem_shape, "x0");
  auto p3 = Parameter(&builder, 1, elem_shape, "x1");
  auto p4 = Parameter(&builder, 2, elem_shape, "y0");
  auto p5 = Parameter(&builder, 3, elem_shape, "y1");
  absl::InlinedVector<XlaOp, 2> compute_vec = {Min(p2, p4), Min(p3, p5)};
  Tuple(&builder, compute_vec);
  TF_ASSERT_OK_AND_ASSIGN(auto compute_tuple, builder.Build());
  auto input1 =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 20}), "input1");
  auto input2 =
      Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {10, 20}), "input2");
  auto init = ConstantR0<float>(&builder, 0);
  ReduceWindow({input1, input2}, {init, init}, compute_tuple, {4, 5}, {4, 5},
               Padding::kValid);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Each of [2x4] output elements are generated from reducing [4x5] elements.
  EXPECT_EQ(analysis.flop_count(), 2 * 4 * 2 * (4 * 5 - 1));

  EXPECT_EQ(analysis.bytes_accessed(), sizeof(float) * (10 * 20 * 2 + 2 * 3));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 10 * 20);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 20);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 4);
}

TEST_F(HloCostAnalysisTest, SelectAndScatter) {
  XlaBuilder builder("select_and_scatter");
  auto operand =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 20}), "input");
  auto source =
      Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {2, 4}), "source");
  SelectAndScatter(operand, gt_, {4, 5}, {4, 5}, Padding::kValid, source,
                   ConstantR0<float>(&builder, 0), add_);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Each of [2x4] source elements computes its destination from reducing [4x5]
  // elements followed by the scatter computation.
  EXPECT_EQ(analysis.flop_count(), 2 * 4 * (4 * 5 - 1 + 1));

  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (10 * 20 + 2 * 4 + 1 + 10 * 20));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 20);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 2 * 4);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 2), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10 * 20);
}

TEST_F(HloCostAnalysisTest, Broadcast) {
  XlaBuilder b("broadcast");
  Broadcast(ConstantR0<float>(&b, 42), {10, 7});
  auto hlo_module = BuildHloGraph(&b);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));
  EXPECT_EQ(analysis.flop_count(), 0);

  EXPECT_EQ(analysis.bytes_accessed(), sizeof(float) * (1 + 10 * 7));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10 * 7);
}

// Calculates the computation cost of a graph with more than one HLO node.
TEST_F(HloCostAnalysisTest, FullyConnectedForward) {
  XlaBuilder builder("fully_connected_forward");
  auto input =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {10, 5}), "input");
  auto weight =
      Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {5, 20}), "weight");
  auto bias = Parameter(&builder, 2, ShapeUtil::MakeShape(F32, {20}), "bias");
  // sigmoid(input * weight + bias)
  Map(&builder, {Add(Dot(input, weight), bias, {1})}, sigmoid_, {0, 1});

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
    XlaBuilder builder("conv_looking_matmul");
    auto lhs = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {64, 64, 1, 1}),
                         "input");
    auto rhs = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {64, 64, 1, 1}),
                         "weights");
    Conv(lhs, rhs, {1, 1}, Padding::kSame);
    auto hlo_module = BuildHloGraph(&builder);
    ASSERT_IS_OK(hlo_module->entry_computation()->root_instruction()->Accept(
        &conv_analysis));
  }

  HloCostAnalysis matmul_analysis(ShapeSize);
  {
    XlaBuilder builder("matmul");
    auto lhs =
        Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {64, 64}), "input");
    auto rhs =
        Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {64, 64}), "weights");
    Dot(lhs, rhs);
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
        HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
            /*from=*/0.0f, /*to=*/1.0f, /*rows=*/2, /*cols=*/2)));
    auto c2 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
            /*from=*/1.0f, /*to=*/2.0f, /*rows=*/2, /*cols=*/2)));
    auto c3 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
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

    auto module = CreateNewVerifiedModule();
    auto* computation = module->AddEntryComputation(builder.Build());
    auto* fusion = computation->CreateFusionInstruction(
        {sub, mul, exp, clamp, add}, HloInstruction::FusionKind::kLoop);

    // The time given these rates at i == 0 is exactly even among the properties
    // at 1.0 seconds. For other values, one of the rates is slower so that it
    // becomes the bottleneck.
    HloCostAnalysis::Options options{ShapeSize};
    options.set_flops_per_second(16 * (i == 1 ? 1 / 2.0 : 1.0));
    options.set_transcendentals_per_second(4 * (i == 2 ? 1 / 4.0 : 1.0));
    options.set_bytes_per_second(64 * (i == 3 ? 1 / 8.0 : 1.0));
    HloCostAnalysis fusion_analysis(options);
    ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

    EXPECT_EQ(fusion_analysis.flop_count(), 16);
    EXPECT_EQ(fusion_analysis.transcendental_count(), 4);
    constexpr int64_t bytes_accessed = sizeof(float) * 4 * 2 * 2;
    static_assert(bytes_accessed == 64, "");
    EXPECT_EQ(fusion_analysis.bytes_accessed(), bytes_accessed);

    EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0),
              sizeof(float) * 2 * 2);
    EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 1),
              sizeof(float) * 2 * 2);
    EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 2),
              sizeof(float) * 2 * 2);
    EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion),
              sizeof(float) * 2 * 2);

    EXPECT_EQ(fusion_analysis.optimal_seconds(), 1 << i);
  }
}

TEST_F(FusionCostAnalysis, LoopFusionTupleOutput) {
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});

  // Same as above but the fusion outputs a tuple.
  HloComputation::Builder builder(TestName());
  auto c1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/0.0f, /*to=*/1.0f, /*rows=*/2, /*cols=*/2)));
  auto c2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/1.0f, /*to=*/2.0f, /*rows=*/2, /*cols=*/2)));
  auto c3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/2.0f, /*to=*/3.0f, /*rows=*/2, /*cols=*/2)));
  auto tuple1 = builder.AddInstruction(HloInstruction::CreateTuple({c1, c2}));
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
  auto tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({sub, sub, mul, tuple1}));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {tuple2, sub, mul, exp, clamp, add}, HloInstruction::FusionKind::kLoop);

  HloCostAnalysis fusion_analysis(ShapeSize);
  ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

  EXPECT_EQ(fusion_analysis.flop_count(), 16);
  EXPECT_EQ(fusion_analysis.transcendental_count(), 4);
  EXPECT_EQ(fusion_analysis.bytes_accessed(*fusion),
            sizeof(float) * (5 + 5) * 2 * 2);

  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0),
            sizeof(float) * 2 * 2 * 2);
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 1),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 2),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 3),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion),
            sizeof(float) * 5 * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion, {0}),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion, {1}),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion, {2}),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion, {3}),
            sizeof(float) * 2 * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion, {3, 0}),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion, {3, 1}),
            sizeof(float) * 2 * 2);
}

TEST_F(FusionCostAnalysis, NoLayout) {
  Shape shape_with_layout = ShapeUtil::MakeShape(F32, {2, 3, 4, 5});
  // Instructions within a fused op may have no layout.
  Shape shape_without_layout = shape_with_layout;
  shape_without_layout.clear_layout();

  HloComputation::Builder builder(TestName());
  auto c1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR4FromArray4D(Array4D<float>(2, 3, 4, 5))));
  auto c2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1, 2, 3})));

  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(shape_without_layout, c2, {1}));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_with_layout, HloOpcode::kAdd, c1, broadcast));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {add, broadcast}, HloInstruction::FusionKind::kLoop);

  HloCostAnalysis fusion_analysis(ShapeSize);
  ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

  EXPECT_EQ(fusion_analysis.flop_count(), 120);
  EXPECT_EQ(fusion_analysis.transcendental_count(), 0);

  EXPECT_EQ(fusion_analysis.bytes_accessed(),
            sizeof(float) * (2 * 3 * 4 * 5 + 3 + 2 * 3 * 4 * 5));

  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0),
            sizeof(float) * 2 * 3 * 4 * 5);
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 1),
            sizeof(float) * 3);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion),
            sizeof(float) * 2 * 3 * 4 * 5);
}

TEST_F(FusionCostAnalysis, NonTupleWithTupleParamBytesAccessed) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

fused_computation {
  param = (f32[3,2]{1,0}, f32[3,2]{1,0}) parameter(0)
  gte0 = f32[3,2]{1,0} get-tuple-element(param), index=0
  gte1 = f32[3,2]{1,0} get-tuple-element(param), index=1
  ROOT add = f32[3,2]{1,0} add(gte0, gte1)
}

ENTRY entry {
  param0 = f32[3,2]{1,0} parameter(0)
  param1 = f32[3,2]{1,0} parameter(1)
  tuple = (f32[3,2]{1,0}, f32[3,2]{1,0}) tuple(param0, param1)
  ROOT fusion = f32[3,2]{1,0} fusion(tuple), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* fusion = module->entry_computation()->root_instruction();

  HloCostAnalysis fusion_analysis(ShapeSize);
  ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

  EXPECT_EQ(fusion_analysis.bytes_accessed(*fusion), sizeof(float) * 3 * 2 * 3);
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0),
            sizeof(float) * 3 * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion),
            sizeof(float) * 3 * 2);
}

TEST_F(FusionCostAnalysis, TupleBytesAccessed) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

fused_computation {
  param = (f32[2,2]{1,0}, f32[2,2]{1,0}) parameter(0)
  gte0 = f32[2,2]{1,0} get-tuple-element(param), index=0
  gte1 = f32[2,2]{1,0} get-tuple-element(param), index=1
  add = f32[2,2]{1,0} add(gte0, gte1)
  mul = f32[2,2]{1,0} multiply(gte0, gte1)
  ROOT root = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(add, mul)
}

ENTRY entry {
  param0 = f32[2,2]{1,0} parameter(0)
  param1 = f32[2,2]{1,0} parameter(1)
  tuple = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(param0, param1)
  ROOT fusion = (f32[2,2]{1,0}, f32[2,2]{1,0}) fusion(tuple), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* fusion = module->entry_computation()->root_instruction();

  HloCostAnalysis fusion_analysis(ShapeSize);
  ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

  EXPECT_EQ(fusion_analysis.bytes_accessed(*fusion), sizeof(float) * 2 * 2 * 4);
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0),
            sizeof(float) * 2 * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion),
            sizeof(float) * 2 * 2 * 2);
}

TEST_F(FusionCostAnalysis, InfeedOutfeed) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  after-all = token[] after-all()
  infeed = ((f32[2,3]{1,0}), token[]) infeed(after-all)
  gte0 = (f32[2,3]{1,0}) get-tuple-element(infeed), index=0
  gte1 = f32[2,3]{1,0} get-tuple-element(gte0), index=0
  add = f32[2,3]{1,0} add(gte1, gte1)
  tuple = (f32[2,3]{1,0}) tuple(add)
  tok = token[] get-tuple-element(infeed), index=1
  ROOT outfeed = token[] outfeed(tuple, tok)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* infeed =
      module->entry_computation()->GetInstructionWithName("infeed");
  HloInstruction* outfeed =
      module->entry_computation()->GetInstructionWithName("outfeed");

  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(infeed->Accept(&analysis));
  ASSERT_IS_OK(outfeed->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(*infeed), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*infeed, 0), 0);
  EXPECT_EQ(analysis.output_bytes_accessed(*infeed), sizeof(float) * 2 * 3);

  EXPECT_EQ(analysis.bytes_accessed(*outfeed), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*outfeed, 0),
            sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.output_bytes_accessed(*outfeed), 0);
}

TEST_F(FusionCostAnalysis, AllReduceTupleBytesAccessed) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

sum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[2,2]{1,0} parameter(0)
  param1 = f32[2,2]{1,0} parameter(1)
  ROOT all-reduce = (f32[2,2]{1,0}, f32[2,2]{1,0}) all-reduce(param0, param1), replica_groups={{0,1}}, to_apply=sum
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* all_reduce = module->entry_computation()->root_instruction();

  HloCostAnalysis all_reduce_analysis(ShapeSize);
  ASSERT_IS_OK(all_reduce->Accept(&all_reduce_analysis));

  EXPECT_EQ(all_reduce_analysis.bytes_accessed(*all_reduce),
            sizeof(float) * 2 * 2 * 4);
  EXPECT_EQ(all_reduce_analysis.operand_bytes_accessed(*all_reduce, 0),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(all_reduce_analysis.operand_bytes_accessed(*all_reduce, 1),
            sizeof(float) * 2 * 2);
  EXPECT_EQ(all_reduce_analysis.output_bytes_accessed(*all_reduce),
            sizeof(float) * 2 * 2 * 2);
}

TEST_F(HloCostAnalysisTest, TupleCost) {
  HloCostAnalysis analysis(ShapeSize);

  XlaBuilder builder("tuple");
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {123}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {42}), "y");
  Tuple(&builder, {x, y});
  auto hlo_module = BuildHloGraph(&builder);

  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 0);
  EXPECT_EQ(analysis.transcendental_count(), 0);
  EXPECT_EQ(analysis.bytes_accessed(), kPointerSize * 2);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), 0);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), 0);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), kPointerSize * 2);
}

using DomainCostAnalysis = HloTestBase;
TEST_F(DomainCostAnalysis, DomainCost) {
  HloCostAnalysis analysis(ShapeSize);

  HloComputation::Builder builder("domain");
  auto x = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {123}), "x"));
  auto y = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {42}), "y"));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({x, y}));
  auto domain = builder.AddInstruction(
      HloInstruction::CreateDomain(tuple->shape(), tuple, nullptr, nullptr));

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  EXPECT_EQ(hlo_module->entry_computation()->root_instruction(), domain);
  ASSERT_IS_OK(domain->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(*domain), 0);
  EXPECT_EQ(analysis.transcendental_count(*domain), 0);
  EXPECT_EQ(analysis.bytes_accessed(*domain), 0);
}

TEST_F(HloCostAnalysisTest, BaseDilatedConvolution) {
  XlaBuilder builder("BaseDilatedConvolution");
  auto input = Parameter(
      &builder, 0,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/10,
                                 /*x_dim=*/20}),
      "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/3,
                                 /*x_dim=*/3}),
      "kernel");

  ConvGeneralDilated(input, kernel, /*window_strides=*/{1, 1},
                     /*padding=*/{{1, 1}, {1, 1}},
                     /*lhs_dilation=*/{3, 5}, /*rhs_dilation=*/{7, 11},
                     XlaBuilder::CreateDefaultConvDimensionNumbers(2));

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 1472);
}

TEST_F(HloCostAnalysisTest, Slice) {
  // Test the analysis on a slice.
  XlaBuilder builder("slice");
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2}), "x");
  Slice(x, {0}, {1}, {1});
  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 8);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float));
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float));
}

TEST_F(HloCostAnalysisTest, DynamicSlice) {
  // Test the analysis on a slice.
  XlaBuilder builder("dynamic-slice");
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2}), "x");
  DynamicSlice(x, absl::Span<const XlaOp>({ConstantR0<int32_t>(&builder, 1)}),
               {1});
  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 8 + 4);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float));
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t));
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float));
}

TEST_F(HloCostAnalysisTest, DynamicUpdateSlice) {
  // Test the analysis on a slice.
  XlaBuilder builder("dynamic-update-slice");
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2}), "x");
  DynamicUpdateSlice(
      x, ConstantR1<float>(&builder, {1.0}),
      absl::Span<const XlaOp>({ConstantR0<int32_t>(&builder, 1)}));
  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 8 + 4);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();

  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), 0);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float));
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 2), sizeof(int32_t));
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float));
}

TEST_F(HloCostAnalysisTest, Gather) {
  // Test the analysis on a gather.
  XlaBuilder builder("gather");
  Shape operand_shape = ShapeUtil::MakeShape(S32, {3, 3});
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});

  auto operand = Parameter(&builder, 0, operand_shape, "operand");
  auto indices = Parameter(&builder, 1, indices_shape, "indices");
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_collapsed_slice_dims(0);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  Gather(operand, indices, dim_numbers, {1, 3});

  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 56);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t) * 2);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 2 * 3);
}

TEST_F(HloCostAnalysisTest, Scatter) {
  // Test the analysis on a scatter.
  XlaBuilder builder("scatter");
  Shape operand_shape = ShapeUtil::MakeShape(F32, {3, 3});
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});
  Shape values_shape = ShapeUtil::MakeShape(F32, {2, 3});

  auto operand = Parameter(&builder, 0, operand_shape, "operand");
  auto indices = Parameter(&builder, 1, indices_shape, "indices");
  auto values = Parameter(&builder, 2, values_shape, "values");
  ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(1);
  dim_numbers.add_update_window_dims(1);
  dim_numbers.add_inserted_window_dims(0);
  dim_numbers.add_scatter_dims_to_operand_dims(0);
  Scatter(operand, indices, values, add_, dim_numbers);

  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 4 * (2 + 3 * (2 * 3)));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t) * 2);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 2), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 2 * 3);
}

TEST_F(HloCostAnalysisTest, MultioutputScatter) {
  // Test the analysis on a scatter.
  XlaBuilder builder("scatter");
  Shape operand0_shape = ShapeUtil::MakeShape(F32, {3, 3});
  Shape operand1_shape = ShapeUtil::MakeShape(S32, {3, 3});
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});
  Shape values0_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape values1_shape = ShapeUtil::MakeShape(S32, {2, 3});

  auto operand0 = Parameter(&builder, 0, operand0_shape, "operand0");
  auto operand1 = Parameter(&builder, 1, operand1_shape, "operand1");
  auto indices = Parameter(&builder, 2, indices_shape, "indices");
  auto values0 = Parameter(&builder, 3, values0_shape, "values0");
  auto values1 = Parameter(&builder, 4, values1_shape, "values1");
  ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(1);
  dim_numbers.add_update_window_dims(1);
  dim_numbers.add_inserted_window_dims(0);
  dim_numbers.add_scatter_dims_to_operand_dims(0);
  auto add = [] {
    XlaBuilder builder("add");
    auto x0 = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x0");
    auto x1 = Parameter(&builder, 1, ShapeUtil::MakeShape(S32, {}), "x1");
    auto y0 = Parameter(&builder, 2, ShapeUtil::MakeShape(F32, {}), "y0");
    auto y1 = Parameter(&builder, 3, ShapeUtil::MakeShape(S32, {}), "y1");
    Tuple(&builder, {Add(x0, y0), Add(x1, y1)});
    auto computation_status = builder.Build();
    TF_CHECK_OK(computation_status.status());
    return std::move(computation_status).ValueOrDie();
  }();
  Scatter({operand0, operand1}, indices, {values0, values1}, add, dim_numbers);

  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis(ShapeSize);
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 4 * (2 + 2 * 3 * (2 * 3)));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 2), sizeof(int32_t) * 2);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 3), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 4), sizeof(int32_t) * 2 * 3);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), 2 * sizeof(float) * 2 * 3);
}

}  // namespace
}  // namespace xla
