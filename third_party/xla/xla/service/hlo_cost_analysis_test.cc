/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_cost_analysis.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array4d.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/service.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

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
      auto x = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "x");
      auto half = ConstantR0<float>(&builder, 0.5);
      Exp(Add(x, half));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_and_exp_ = std::move(computation_status).value();
    }

    // Create a computation for a binary user function: (x, y) => x + y
    {
      XlaBuilder builder("add");
      auto x = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "x");
      auto y = Parameter(&builder, 1,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "y");
      Add(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      add_ = std::move(computation_status).value();
    }

    // Create a computation for a sigmoid function: x => 1 / (1 + exp(-x))
    {
      XlaBuilder builder("sigmoid");
      auto x = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "x");
      auto one = ConstantR0<float>(&builder, 1.0);
      Div(one, Add(one, Exp(Neg(x))));
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      sigmoid_ = std::move(computation_status).value();
    }

    // Create a computation for a binary max function: (x, y) => max (x, y)
    {
      XlaBuilder builder("max");
      auto x = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "x");
      auto y = Parameter(&builder, 1,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "y");
      Max(x, y);
      auto computation_status = builder.Build();
      TF_CHECK_OK(computation_status.status());
      max_ = std::move(computation_status).value();
    }

    // Create a computation for a binary GT function: (x, y) => x > y
    {
      XlaBuilder builder("gt");
      auto x = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "x");
      auto y = Parameter(&builder, 1,
                         ShapeUtil::MakeValidatedShape(F32, {}).value(), "y");
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

TEST_F(HloCostAnalysisTest, MatrixMultiply) {
  XlaBuilder builder("matrix_multiply");
  auto lhs = Parameter(
      &builder, 0, ShapeUtil::MakeValidatedShape(F32, {10, 5}).value(), "lhs");
  auto rhs = Parameter(
      &builder, 1, ShapeUtil::MakeValidatedShape(F32, {5, 30}).value(), "rhs");
  Dot(lhs, rhs);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
      Parameter(&builder, 0,
                ShapeUtil::MakeValidatedShape(F32, {10, 5, 5}).value(), "lhs");
  auto rhs =
      Parameter(&builder, 1,
                ShapeUtil::MakeValidatedShape(F32, {5, 5, 30}).value(), "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(0);
  dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs, rhs, dnums);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
      Parameter(&builder, 0,
                ShapeUtil::MakeValidatedShape(F32, {10, 5, 5}).value(), "lhs");
  auto rhs =
      Parameter(&builder, 1,
                ShapeUtil::MakeValidatedShape(F32, {5, 5, 30}).value(), "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(2);
  dnums.add_rhs_contracting_dimensions(0);
  dnums.add_rhs_batch_dimensions(1);
  DotGeneral(lhs, rhs, dnums);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
  auto lhs = Parameter(
      &builder, 0, ShapeUtil::MakeValidatedShape(F32, {10, 5}).value(), "lhs");
  auto rhs = Parameter(
      &builder, 1, ShapeUtil::MakeValidatedShape(F32, {5, 30}).value(), "rhs");
  DotDimensionNumbers dnums;
  DotGeneral(lhs, rhs, dnums);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
  auto input = Parameter(
      &builder, 0, ShapeUtil::MakeValidatedShape(F32, {10}).value(), "in");
  Map(&builder, {input}, add_and_exp_, {0});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
  auto input = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(
                             F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/10,
                                   /*x_dim=*/20})
                             .value(),
                         "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeValidatedShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/3,
                                          /*x_dim=*/3})
          .value(),
      "kernel");
  Conv(input, kernel, {1, 1}, Padding::kValid);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
  auto input = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(
                             F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/ih,
                                   /*x_dim=*/iw})
                             .value(),
                         "input");
  auto kernel = Parameter(&builder, 1,
                          ShapeUtil::MakeValidatedShape(
                              F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kh,
                                    /*x_dim=*/kw})
                              .value(),
                          "kernel");
  Conv(input, kernel, {sx, sy}, Padding::kSame);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // Output shape is [1x1x3x3] with the following flops required for each
  // element:
  //    4 6 4
  //    6 9 6
  //    4 6 4
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
  auto input = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(
                             F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kLarge})
                             .value(),
                         "input");
  auto kernel = Parameter(&builder, 1,
                          ShapeUtil::MakeValidatedShape(
                              F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kLarge})
                              .value(),
                          "kernel");
  ConvGeneralDilated(input, kernel, {kLarge - 1}, {{0, 0}}, {kLarge}, {1},
                     XlaBuilder::CreateDefaultConvDimensionNumbers(1));

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 2 * kLarge);
}

TEST_F(HloCostAnalysisTest, ConvolutionExtreme2) {
  XlaBuilder builder("convolution");
  constexpr int64_t kLarge = 512 * 1024;
  auto input = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(
                             F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/1})
                             .value(),
                         "input");
  auto kernel = Parameter(&builder, 1,
                          ShapeUtil::MakeValidatedShape(
                              F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/kLarge})
                              .value(),
                          "kernel");
  ConvGeneralDilated(input, kernel, {1}, {{kLarge - 1, kLarge - 1}}, {1}, {1},
                     XlaBuilder::CreateDefaultConvDimensionNumbers(1));

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 2 * kLarge);
}

TEST_F(HloCostAnalysisTest, ConvolutionWithFeatureGroup) {
  XlaBuilder builder("convolution");
  auto input = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(
                             F32, {/*p_dim=*/1, /*z_dim=*/120, /*y_dim=*/10,
                                   /*x_dim=*/20})
                             .value(),
                         "input");
  auto kernel = Parameter(&builder, 1,
                          ShapeUtil::MakeValidatedShape(
                              F32, {/*p_dim=*/120, /*z_dim=*/1, /*y_dim=*/3,
                                    /*x_dim=*/3})
                              .value(),
                          "kernel");
  Conv(input, kernel, {1, 1}, Padding::kValid, /*feature_group_count=*/120);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
      Parameter(&builder, 0,
                ShapeUtil::MakeValidatedShape(F32, {10, 20}).value(), "input");
  Reduce(input, ConstantR0<float>(&builder, 0.0f), add_, {1});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
      Parameter(&builder, 0,
                ShapeUtil::MakeValidatedShape(F32, {10, 20}).value(), "input");
  ReduceWindow(input, ConstantR0<float>(&builder, 0), add_, {4, 5}, {4, 5},
               Padding::kValid);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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

TEST_F(HloCostAnalysisTest, ReduceWindowWithOverlaps) {
  XlaBuilder builder("reduce_window");
  auto input = Parameter(
      &builder, 0, ShapeUtil::MakeValidatedShape(F32, {8, 8}).value(), "input");
  ReduceWindow(input, ConstantR0<float>(&builder, 0), add_, {4, 5}, {2, 1},
               Padding::kValid);

  auto hlo_module = BuildHloGraph(&builder);
  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  int n_output_elements = 3 * 4;

  // Run HLO cost analysis.
  HloCostAnalysis analysis;
  ASSERT_IS_OK(root->Accept(&analysis));

  // Each of the output elements are generated from reducing [4x5] elements.
  EXPECT_EQ(analysis.flop_count(), n_output_elements * (4 * 5 - 1));

  EXPECT_EQ(analysis.bytes_accessed(),
            sizeof(float) * (8 * 8 + 1 + n_output_elements));

  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 8 * 8);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root),
            sizeof(float) * n_output_elements);
}

TEST_F(HloCostAnalysisTest, ReduceWindowSingleDimReduceBroadcast) {
  absl::string_view hlo_text = R"(
 HloModule fusion.50

region_0.868 {
  Arg_1.870 = f32[] parameter(1)
  Arg_0.869 = f32[] parameter(0)
  ROOT maximum.871 = f32[] maximum(Arg_0.869, Arg_1.870)
}

ENTRY fusion.50 {
  constant.367 = f32[] constant(-inf)
  param0 = f32[2,3,1024,1024]{2,3,1,0} parameter(0)
  ROOT reduce-window.159 = f32[2,3,1024,1024]{2,3,1,0} reduce-window(param0, constant.367), window={size=1x1x1x2047 pad=0_0x0_0x0_0x1023_1023}, to_apply=region_0.868
}
)";
  auto hlo_module = ParseAndReturnUnverifiedModule(hlo_text).value();
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));
  EXPECT_EQ(analysis.flop_count(), (2 * 3 * 1024) + (1024 - 1));
  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0),
            sizeof(float) * 2 * 3 * 1024 * 1024);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root),
            sizeof(float) * 2 * 3 * 1024 * 1024);
}

TEST_F(HloCostAnalysisTest, ReduceWindowVariadic) {
  XlaBuilder builder("reduce_window_variadic");
  auto elem_shape = ShapeUtil::MakeValidatedShape(F32, {}).value();
  auto p2 = Parameter(&builder, 0, elem_shape, "x0");
  auto p3 = Parameter(&builder, 1, elem_shape, "x1");
  auto p4 = Parameter(&builder, 2, elem_shape, "y0");
  auto p5 = Parameter(&builder, 3, elem_shape, "y1");
  absl::InlinedVector<XlaOp, 2> compute_vec = {Min(p2, p4), Min(p3, p5)};
  Tuple(&builder, compute_vec);
  TF_ASSERT_OK_AND_ASSIGN(auto compute_tuple, builder.Build());
  auto input1 =
      Parameter(&builder, 0,
                ShapeUtil::MakeValidatedShape(F32, {10, 20}).value(), "input1");
  auto input2 =
      Parameter(&builder, 1,
                ShapeUtil::MakeValidatedShape(F32, {10, 20}).value(), "input2");
  auto init = ConstantR0<float>(&builder, 0);
  ReduceWindow({input1, input2}, {init, init}, compute_tuple, {4, 5}, {4, 5},
               Padding::kValid);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
      Parameter(&builder, 0,
                ShapeUtil::MakeValidatedShape(F32, {10, 20}).value(), "input");
  auto source =
      Parameter(&builder, 1, ShapeUtil::MakeValidatedShape(F32, {2, 4}).value(),
                "source");
  SelectAndScatter(operand, gt_, {4, 5}, {4, 5}, Padding::kValid, source,
                   ConstantR0<float>(&builder, 0), add_);

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
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
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));
  EXPECT_EQ(analysis.flop_count(), 0);

  EXPECT_EQ(analysis.bytes_accessed(), sizeof(float) * (1 + 10 * 7));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 1);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10 * 7);
}

TEST_F(HloCostAnalysisTest, BroadcastCountMultipleInputAccesses) {
  XlaBuilder b("broadcast");
  Broadcast(ConstantR0<float>(&b, 42), {10, 7});
  auto hlo_module = BuildHloGraph(&b);
  HloCostAnalysis analysis(
      HloCostAnalysis::Options{.count_multiple_input_accesses = true});
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));
  EXPECT_EQ(analysis.flop_count(), 0);

  EXPECT_EQ(analysis.bytes_accessed(), sizeof(float) * (1 + 10 * 7));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 10 * 7);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 10 * 7);
}

// Calculates the computation cost of a graph with more than one HLO node.
TEST_F(HloCostAnalysisTest, FullyConnectedForward) {
  XlaBuilder builder("fully_connected_forward");
  auto input =
      Parameter(&builder, 0,
                ShapeUtil::MakeValidatedShape(F32, {10, 5}).value(), "input");
  auto weight =
      Parameter(&builder, 1,
                ShapeUtil::MakeValidatedShape(F32, {5, 20}).value(), "weight");
  auto bias = Parameter(
      &builder, 2, ShapeUtil::MakeValidatedShape(F32, {20}).value(), "bias");
  // sigmoid(input * weight + bias)
  Map(&builder, {Add(Dot(input, weight), bias, {1})}, sigmoid_, {0, 1});

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  // 1000 FMAs from matrix multiplication, 200 flops from bias addition,
  // 600 flops from sigmoid, and 200 transcendental ops from sigmoid.
  EXPECT_EQ(analysis.flop_count(), 2 * 1000 + 200 + 3 * 200);
  EXPECT_EQ(analysis.transcendental_count(), 200);
}

TEST_F(HloCostAnalysisTest, MatmulAndConvolutionCanBeTheSameComputation) {
  HloCostAnalysis conv_analysis;
  {
    XlaBuilder builder("conv_looking_matmul");
    auto lhs = Parameter(
        &builder, 0, ShapeUtil::MakeValidatedShape(F32, {64, 64, 1, 1}).value(),
        "input");
    auto rhs = Parameter(
        &builder, 1, ShapeUtil::MakeValidatedShape(F32, {64, 64, 1, 1}).value(),
        "weights");
    Conv(lhs, rhs, {1, 1}, Padding::kSame);
    auto hlo_module = BuildHloGraph(&builder);
    ASSERT_IS_OK(hlo_module->entry_computation()->root_instruction()->Accept(
        &conv_analysis));
  }

  HloCostAnalysis matmul_analysis;
  {
    XlaBuilder builder("matmul");
    auto lhs = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(F32, {64, 64}).value(),
                         "input");
    auto rhs = Parameter(&builder, 1,
                         ShapeUtil::MakeValidatedShape(F32, {64, 64}).value(),
                         "weights");
    Dot(lhs, rhs);
    auto hlo_module = BuildHloGraph(&builder);
    ASSERT_IS_OK(hlo_module->entry_computation()->root_instruction()->Accept(
        &matmul_analysis));
  }

  EXPECT_EQ(conv_analysis.flop_count(), matmul_analysis.flop_count());
}

// No instruction can finish faster than the clock cycle
TEST_F(HloCostAnalysisTest, LatencyBoundedOptimalTime) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY Entry {
    param0 = f32[1,1] parameter(0)
    param1 = f32[1,1] parameter(1)
    ROOT add = f32[1,1] add(param0, param1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  const HloInstruction* add = module->entry_computation()->root_instruction();
  HloCostAnalysis::Options options;
  const float clock_cycle_seconds = 10.0f;
  options.set_flops_per_second(1024);
  options.set_bytes_per_second(1024);
  options.set_transcendentals_per_second(1024);
  options.set_flops_min_latency_second(clock_cycle_seconds);
  HloCostAnalysis cost_analysis(options);
  ASSERT_IS_OK(add->Accept(&cost_analysis));
  EXPECT_EQ(cost_analysis.optimal_seconds(), clock_cycle_seconds);
}

using FusionCostAnalysis = HloHardwareIndependentTestBase;

TEST_F(FusionCostAnalysis, LoopFusionDynUpdateSlice) {
  // Test for b/234935631.
  // DynamicUpdateSlice within a loop fusion needs to respect operand-output
  // aliasing.
  const char* hlo_fusion_module_str = R"(
  HloModule module

  _.1 {
    tmp_0 = bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)} parameter(0)
    tmp_1 = bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)} parameter(2)
    tmp_2 = s32[]{:T(128)} parameter(1)
    tmp_3 = s32[]{:T(128)} constant(0)
    tmp_4 = bf16[1,32,256,1152]{3,2,1,0:T(8,128)(2,1)S(3)} dynamic-slice(tmp_1, tmp_2, tmp_3, tmp_3, tmp_3), dynamic_slice_sizes={1,32,256,1152}
    tmp_11 = bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)} dynamic-update-slice(tmp_0, tmp_4, tmp_2, tmp_3, tmp_3, tmp_3)
    ROOT tmp_20 = (bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)}) tuple(tmp_11)
  }

  ENTRY _ {
    _0 = bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)} parameter(0)
    _1 = s32[]{:T(128)} parameter(1)
    _4 = bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)} parameter(2)
    ROOT _ = (bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)}) fusion(_0, _1, _4), kind=kLoop, calls=_.1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_fusion_module_str));
  HloCostAnalysis fusion_analysis;

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

  const char* hlo_dus_module_str = R"(
  HloModule module

  ENTRY _ {
    _0 = bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)} parameter(0)
    _1 = s32[]{:T(128)} parameter(1)
    _2 = bf16[1,32,256,1152]{3,2,1,0:T(8,128)(2,1)} parameter(2)
    ROOT _ = bf16[50,32,256,1152]{3,2,1,0:T(8,128)(2,1)} dynamic-update-slice(_0, _2, _1, _1, _1, _1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto dus_module,
                          ParseAndReturnVerifiedModule(hlo_dus_module_str));
  HloCostAnalysis dus_analysis;
  auto dus = dus_module->entry_computation()->root_instruction();
  ASSERT_IS_OK(dus->Accept(&dus_analysis));
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0), 0);
  EXPECT_EQ(fusion_analysis.bytes_accessed(), dus_analysis.bytes_accessed());
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0),
            dus_analysis.operand_bytes_accessed(*dus, 0));
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 1),
            dus_analysis.operand_bytes_accessed(*dus, 2));
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 2),
            dus_analysis.operand_bytes_accessed(*dus, 1));
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion),
            dus_analysis.output_bytes_accessed(*dus));
}

TEST_F(FusionCostAnalysis, LoopFusion) {
  // Do this 4 times with different per-second rates to test the computation of
  // bottleneck time on fusion nodes.
  for (int i = 0; i < 4; ++i) {
    Shape r2f32 = ShapeUtil::MakeValidatedShape(F32, {2, 2}).value();

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
    HloCostAnalysis::Options options;
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

TEST_F(FusionCostAnalysis, NestedCopyFusion) {
  absl::string_view nested_fusion_text = R"(
HloModule temp, is_scheduled=true

copy_fusion.1291.clone {
  input.1291 = s8[2,6144,2,256]{3,1,0,2:T(32,128)(4,1)S(1)} parameter(0)
  ROOT copy.74276 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} copy(input.1291)
}

fused_computation.4150.clone {
  param_0.185389 = s8[2,6144,2,256]{3,1,0,2:T(32,128)(4,1)} parameter(0)
  fusion.103344 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} fusion(param_0.185389), kind=kLoop, calls=copy_fusion.1291.clone
  constant.230138 = s32[]{:T(128)} constant(0)
  param_1.219146 = s32[]{:T(128)S(6)} parameter(1)
  ROOT dynamic-slice.40526 = s8[2,384,2,256]{3,1,0,2:T(8,128)(4,1)} dynamic-slice(fusion.103344, constant.230138, param_1.219146, constant.230138, constant.230138), dynamic_slice_sizes={2,384,2,256}
}

ENTRY temp {
  param_2.123719 = s8[2,6144,2,256]{3,1,0,2:T(32,128)(4,1)} parameter(0)
  param_3.66279 = s32[]{:T(128)S(6)} parameter(1)
  ROOT fusion.85943 = s8[2,384,2,256]{3,1,0,2:T(8,128)(4,1)} fusion(param_2.123719, param_3.66279), kind=kLoop, calls=fused_computation.4150.clone
}
)";
  absl::string_view fusion_text = R"(
HloModule temp, is_scheduled=true

fused_computation.4150.clone {
  param_0.185389 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} parameter(0)
  constant.230138 = s32[]{:T(128)} constant(0)
  param_1.219146 = s32[]{:T(128)S(6)} parameter(1)
  ROOT dynamic-slice.40526 = s8[2,384,2,256]{3,1,0,2:T(8,128)(4,1)} dynamic-slice(param_0.185389, constant.230138, param_1.219146, constant.230138, constant.230138), dynamic_slice_sizes={2,384,2,256}
}

ENTRY temp {
  param_2.123719 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} parameter(0)
  param_3.66279 = s32[]{:T(128)S(6)} parameter(1)
  ROOT fusion.85943 = s8[2,384,2,256]{3,1,0,2:T(8,128)(4,1)} fusion(param_2.123719, param_3.66279), kind=kLoop, calls=fused_computation.4150.clone
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto nested_fusion_module,
                          ParseAndReturnVerifiedModule(nested_fusion_text));
  HloCostAnalysis nested_analysis;
  auto* nested_root =
      nested_fusion_module->entry_computation()->root_instruction();
  ASSERT_IS_OK(nested_root->Accept(&nested_analysis));
  TF_ASSERT_OK_AND_ASSIGN(auto fusion_module,
                          ParseAndReturnVerifiedModule(fusion_text));
  HloCostAnalysis fusion_analysis;
  auto* fusion_root = fusion_module->entry_computation()->root_instruction();
  ASSERT_IS_OK(fusion_root->Accept(&fusion_analysis));
  // The nested fusion should only access the bytes size amount of the parameter
  // based on the size of the consuming dynamic slice.
  EXPECT_EQ(nested_analysis.bytes_accessed(*nested_root),
            fusion_analysis.bytes_accessed(*fusion_root));
}

TEST_F(FusionCostAnalysis, NestedCopyFusionDUS) {
  absl::string_view nested_fusion_text = R"(
HloModule temp, is_scheduled=true

copy_fusion.1291.clone {
  input.1291 = s8[2,6144,2,256]{3,1,0,2:T(32,128)(4,1)} parameter(0)
  ROOT copy.74276 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} copy(input.1291)
}

fused_computation.4150.clone {
  param_0.185389 = s8[2,6144,2,256]{3,1,0,2:T(32,128)(4,1)} parameter(0)
  fusion.103344 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} fusion(param_0.185389), kind=kLoop, calls=copy_fusion.1291.clone
  param_1.185389 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} parameter(2)
  constant.230138 = s32[]{:T(128)} constant(0)
  param_1.219146 = s32[]{:T(128)S(6)} parameter(1)
  param_3.229 = pred[]{:T(512)} constant(false)
  broadcast.11499 = pred[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} broadcast(param_3.229), dimensions={}
  dynamic-slice.11241 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} dynamic-slice(fusion.103344, constant.230138, constant.230138, param_1.219146, constant.230138), dynamic_slice_sizes={2,6144,1,256}
  select.9063 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} select(broadcast.11499, param_1.185389, dynamic-slice.11241)
  ROOT dynamic-update-slice.40526 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} dynamic-update-slice(fusion.103344, select.9063, constant.230138, constant.230138, param_1.219146, constant.230138)
}

ENTRY temp {
  param_2.123719 = s8[2,6144,2,256]{3,1,0,2:T(32,128)(4,1)} parameter(0)
  param_3.66279 = s32[]{:T(128)S(6)} parameter(1)
  param_1.123719 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} parameter(2)
  ROOT fusion.85943 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} fusion(param_2.123719, param_3.66279, param_1.123719), kind=kLoop, calls=fused_computation.4150.clone
}
)";
  absl::string_view fusion_text = R"(
HloModule temp, is_scheduled=true

fused_computation.4150.clone {
  param_0.185389 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} parameter(0)
  param_1.185389 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} parameter(2)
  constant.230138 = s32[]{:T(128)} constant(0)
  param_1.219146 = s32[]{:T(128)S(6)} parameter(1)
  param_3.229 = pred[]{:T(512)} constant(false)
  broadcast.11499 = pred[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} broadcast(param_3.229), dimensions={}
  dynamic-slice.11241 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} dynamic-slice(param_0.185389, constant.230138, constant.230138, param_1.219146, constant.230138), dynamic_slice_sizes={2,6144,1,256}
  select.9063 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} select(broadcast.11499, param_1.185389, dynamic-slice.11241)
  ROOT dynamic-update-slice.40526 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} dynamic-update-slice(param_0.185389, select.9063, constant.230138, constant.230138, param_1.219146, constant.230138)
}

ENTRY temp {
  param_2.123719 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} parameter(0)
  param_3.66279 = s32[]{:T(128)S(6)} parameter(1)
  param_1.123719 = s8[2,6144,1,256]{3,1,0,2:T(8,128)(4,1)} parameter(2)
  ROOT fusion.85943 = s8[2,6144,2,256]{3,1,0,2:T(8,128)(4,1)} fusion(param_2.123719, param_3.66279, param_1.123719), kind=kLoop, calls=fused_computation.4150.clone
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto nested_fusion_module,
                          ParseAndReturnVerifiedModule(nested_fusion_text));
  HloCostAnalysis nested_analysis;
  auto* nested_root =
      nested_fusion_module->entry_computation()->root_instruction();
  ASSERT_IS_OK(nested_root->Accept(&nested_analysis));
  TF_ASSERT_OK_AND_ASSIGN(auto fusion_module,
                          ParseAndReturnVerifiedModule(fusion_text));
  HloCostAnalysis fusion_analysis;
  auto* fusion_root = fusion_module->entry_computation()->root_instruction();
  ASSERT_IS_OK(fusion_root->Accept(&fusion_analysis));
  // The nested fusion should only access the bytes size amount of the parameter
  // based on the size of the consuming dynamic slice.
  EXPECT_EQ(nested_analysis.bytes_accessed(*nested_root),
            fusion_analysis.bytes_accessed(*fusion_root));
}

TEST_F(FusionCostAnalysis, NestedFusionFeedsMultipleUsers) {
  absl::string_view hlo_text = R"(
HloModule temp, is_scheduled=true

fused_computation.1 {
  tmp_0 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} parameter(0)
  tmp_1 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} fusion(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_0), kind=kLoop, calls=
  {
    tmp_0 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} parameter(0)
    ROOT tmp_4 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} add(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_0, bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_0)
  }
  tmp_2 = bf16[]{:T(256)} constant(0)
  tmp_3 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} reduce-window(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_1, bf16[]{:T(256)} tmp_2), window={size=1x1x1x1023 pad=0_0x0_0x0_0x511_511}, to_apply=
  {
    tmp_0 = bf16[]{:T(256)} parameter(0)
    tmp_1 = bf16[]{:T(256)} parameter(1)
    ROOT tmp_2 = bf16[]{:T(256)} add(bf16[]{:T(256)} tmp_0, bf16[]{:T(256)} tmp_1)
  }
  ROOT tmp_4 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} divide(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_1, bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_3)
}

ENTRY temp {
  tmp_0 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} parameter(0)
  ROOT result = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} fusion(tmp_0), kind=kLoop, calls=fused_computation.1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto fusion_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  HloCostAnalysis fusion_analysis;
  auto* fusion_root = fusion_module->entry_computation()->root_instruction();
  ASSERT_IS_OK(fusion_root->Accept(&fusion_analysis));
  EXPECT_EQ(1073741824, fusion_analysis.bytes_accessed(*fusion_root));
}

TEST_F(FusionCostAnalysis, ParamFeedsNestedFusionAndTrivialUser) {
  absl::string_view hlo_text = R"(
HloModule temp, is_scheduled=true

fused_computation.1 {
  tmp_0 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} parameter(0)
  tmp_1 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} fusion(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_0), kind=kLoop, calls=
  {
    tmp_0 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} parameter(0)
    ROOT tmp_4 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} add(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_0, bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_0)
  }
  tmp_2 = bf16[]{:T(256)} constant(0)
  tmp_3 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} reduce-window(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_1, bf16[]{:T(256)} tmp_2), window={size=1x1x1x1023 pad=0_0x0_0x0_0x511_511}, to_apply=
  {
    tmp_0 = bf16[]{:T(256)} parameter(0)
    tmp_1 = bf16[]{:T(256)} parameter(1)
    ROOT tmp_2 = bf16[]{:T(256)} add(bf16[]{:T(256)} tmp_0, bf16[]{:T(256)} tmp_1)
  }
  ROOT tmp_4 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} divide(bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_0, bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} tmp_3)
}

ENTRY temp {
  tmp_0 = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} parameter(0)
  ROOT result = bf16[64,16,512,512]{2,3,1,0:T(8,128)(2,1)} fusion(tmp_0), kind=kLoop, calls=fused_computation.1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto fusion_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  HloCostAnalysis fusion_analysis;
  auto* fusion_root = fusion_module->entry_computation()->root_instruction();
  ASSERT_IS_OK(fusion_root->Accept(&fusion_analysis));
  EXPECT_EQ(1610612736, fusion_analysis.bytes_accessed(*fusion_root));
}

TEST_F(FusionCostAnalysis, LoopFusionTupleOutput) {
  Shape r2f32 = ShapeUtil::MakeValidatedShape(F32, {2, 2}).value();

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

  HloCostAnalysis fusion_analysis;
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
  Shape shape_with_layout =
      ShapeUtil::MakeValidatedShape(F32, {2, 3, 4, 5}).value();
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

  HloCostAnalysis fusion_analysis;
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

  HloCostAnalysis fusion_analysis;
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

  HloCostAnalysis fusion_analysis;
  ASSERT_IS_OK(fusion->Accept(&fusion_analysis));

  EXPECT_EQ(fusion_analysis.bytes_accessed(*fusion), sizeof(float) * 2 * 2 * 4);
  EXPECT_EQ(fusion_analysis.operand_bytes_accessed(*fusion, 0),
            sizeof(float) * 2 * 2 * 2);
  EXPECT_EQ(fusion_analysis.output_bytes_accessed(*fusion),
            sizeof(float) * 2 * 2 * 2);
}

TEST_F(FusionCostAnalysis, IgnoreUnusedParameterShape) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = (s8[3], s8[100]) parameter(0)
  gte0 = s8[3] get-tuple-element(p0), index=0
  c1 = s8[3] constant(0)
  a1 = s8[3] add(gte0, c1)
  ROOT r1 = s8[3] add(a1, c1)
}

ENTRY e {
  param0 = (s8[3], s8[100]) parameter(0)
  ROOT r0 = s8[3] fusion(param0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* root = module->entry_computation()->root_instruction();

  HloCostAnalysis analysis;
  ASSERT_IS_OK(root->Accept(&analysis));

  EXPECT_EQ(analysis.output_bytes_accessed(*root), 3);
  // 2-element tuple (pointers) + its 3-element shape #0
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0),
            2 * HloCostAnalysis::kDefaultPointerSize + 3);
  // Same as above + non-scalar constant c1 + output.
  EXPECT_EQ(analysis.bytes_accessed(*root),
            2 * HloCostAnalysis::kDefaultPointerSize + 3 + 3 + 3);
  EXPECT_EQ(analysis.bytes_accessed(),
            2 * HloCostAnalysis::kDefaultPointerSize + 3 + 3 + 3);
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

  HloCostAnalysis analysis;
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

  HloCostAnalysis all_reduce_analysis;
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
  HloCostAnalysis analysis;

  XlaBuilder builder("tuple");
  auto x = Parameter(&builder, 0,
                     ShapeUtil::MakeValidatedShape(F32, {123}).value(), "x");
  auto y = Parameter(&builder, 1,
                     ShapeUtil::MakeValidatedShape(F32, {42}).value(), "y");
  Tuple(&builder, {x, y});
  auto hlo_module = BuildHloGraph(&builder);

  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 0);
  EXPECT_EQ(analysis.transcendental_count(), 0);
  EXPECT_EQ(analysis.bytes_accessed(),
            HloCostAnalysis::kDefaultPointerSize * 2);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), 0);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), 0);
  EXPECT_EQ(analysis.output_bytes_accessed(*root),
            HloCostAnalysis::kDefaultPointerSize * 2);
}

using DomainCostAnalysis = HloHardwareIndependentTestBase;
TEST_F(DomainCostAnalysis, DomainCost) {
  HloCostAnalysis analysis;

  HloComputation::Builder builder("domain");
  auto x = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeValidatedShape(F32, {123}).value(), "x"));
  auto y = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeValidatedShape(F32, {42}).value(), "y"));
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
  auto input = Parameter(&builder, 0,
                         ShapeUtil::MakeValidatedShape(
                             F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/10,
                                   /*x_dim=*/20})
                             .value(),
                         "input");
  auto kernel = Parameter(
      &builder, 1,
      ShapeUtil::MakeValidatedShape(F32, {/*p_dim=*/1, /*z_dim=*/1, /*y_dim=*/3,
                                          /*x_dim=*/3})
          .value(),
      "kernel");

  ConvGeneralDilated(input, kernel, /*window_strides=*/{1, 1},
                     /*padding=*/{{1, 1}, {1, 1}},
                     /*lhs_dilation=*/{3, 5}, /*rhs_dilation=*/{7, 11},
                     XlaBuilder::CreateDefaultConvDimensionNumbers(2));

  // Run HLO cost analysis.
  auto hlo_module = BuildHloGraph(&builder);
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.flop_count(), 1472);
}

TEST_F(HloCostAnalysisTest, Slice) {
  // Test the analysis on a slice.
  XlaBuilder builder("slice");
  auto x = Parameter(&builder, 0,
                     ShapeUtil::MakeValidatedShape(F32, {2}).value(), "x");
  Slice(x, {0}, {1}, {1});
  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis;
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
  auto x = Parameter(&builder, 0,
                     ShapeUtil::MakeValidatedShape(F32, {2}).value(), "x");
  DynamicSlice(x, absl::Span<const XlaOp>({ConstantR0<int32_t>(&builder, 1)}),
               {1});
  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis;
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
  auto x = Parameter(&builder, 0,
                     ShapeUtil::MakeValidatedShape(F32, {2}).value(), "x");
  DynamicUpdateSlice(
      x, ConstantR1<float>(&builder, {1.0}),
      absl::Span<const XlaOp>({ConstantR0<int32_t>(&builder, 1)}));
  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis;
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
  Shape operand_shape = ShapeUtil::MakeValidatedShape(S32, {3, 3}).value();
  Shape indices_shape = ShapeUtil::MakeValidatedShape(S32, {2}).value();

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
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 56);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t) * 2);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 2 * 3);
}

TEST_F(HloCostAnalysisTest, GatherBatchingDims) {
  // Test the analysis on a gather.
  XlaBuilder builder("gather");
  Shape operand_shape = ShapeUtil::MakeValidatedShape(S32, {5, 3, 3}).value();
  Shape indices_shape = ShapeUtil::MakeValidatedShape(S32, {5}).value();

  auto operand = Parameter(&builder, 0, operand_shape, "operand");
  auto indices = Parameter(&builder, 1, indices_shape, "indices");
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_collapsed_slice_dims(1);
  dim_numbers.add_operand_batching_dims(0);
  dim_numbers.add_start_indices_batching_dims(0);
  dim_numbers.add_start_index_map(1);
  dim_numbers.set_index_vector_dim(1);
  Gather(operand, indices, dim_numbers, {1, 1, 3});

  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 140);

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 5 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t) * 5);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 5 * 3);
}

TEST_F(HloCostAnalysisTest, Scatter) {
  // Test the analysis on a scatter.
  XlaBuilder builder("scatter");
  Shape operand_shape = ShapeUtil::MakeValidatedShape(F32, {3, 3}).value();
  Shape indices_shape = ShapeUtil::MakeValidatedShape(S32, {2}).value();
  Shape values_shape = ShapeUtil::MakeValidatedShape(F32, {2, 3}).value();

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
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 4 * (2 + 3 * (2 * 3)));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t) * 2);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 2), sizeof(float) * 2 * 3);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 2 * 3);
}

TEST_F(HloCostAnalysisTest, ScatterBatchingDims) {
  // Test the analysis on a scatter.
  XlaBuilder builder("scatter");
  Shape operand_shape = ShapeUtil::MakeValidatedShape(F32, {5, 3, 3}).value();
  Shape indices_shape = ShapeUtil::MakeValidatedShape(S32, {5}).value();
  Shape values_shape = ShapeUtil::MakeValidatedShape(F32, {5, 3}).value();

  auto operand = Parameter(&builder, 0, operand_shape, "operand");
  auto indices = Parameter(&builder, 1, indices_shape, "indices");
  auto values = Parameter(&builder, 2, values_shape, "values");
  ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(1);
  dim_numbers.add_update_window_dims(1);
  dim_numbers.add_inserted_window_dims(1);
  dim_numbers.add_input_batching_dims(0);
  dim_numbers.add_scatter_indices_batching_dims(0);
  dim_numbers.add_scatter_dims_to_operand_dims(1);
  Scatter(operand, indices, values, add_, dim_numbers);

  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis;
  ASSERT_IS_OK(
      hlo_module->entry_computation()->root_instruction()->Accept(&analysis));

  EXPECT_EQ(analysis.bytes_accessed(), 4 * (5 + 3 * (5 * 3)));

  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), sizeof(float) * 5 * 3);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 1), sizeof(int32_t) * 5);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 2), sizeof(float) * 5 * 3);
  EXPECT_EQ(analysis.output_bytes_accessed(*root), sizeof(float) * 5 * 3);
}

TEST_F(HloCostAnalysisTest, MultioutputScatter) {
  // Test the analysis on a scatter.
  XlaBuilder builder("scatter");
  Shape operand0_shape = ShapeUtil::MakeValidatedShape(F32, {3, 3}).value();
  Shape operand1_shape = ShapeUtil::MakeValidatedShape(S32, {3, 3}).value();
  Shape indices_shape = ShapeUtil::MakeValidatedShape(S32, {2}).value();
  Shape values0_shape = ShapeUtil::MakeValidatedShape(F32, {2, 3}).value();
  Shape values1_shape = ShapeUtil::MakeValidatedShape(S32, {2, 3}).value();

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
    auto x0 = Parameter(&builder, 0,
                        ShapeUtil::MakeValidatedShape(F32, {}).value(), "x0");
    auto x1 = Parameter(&builder, 1,
                        ShapeUtil::MakeValidatedShape(S32, {}).value(), "x1");
    auto y0 = Parameter(&builder, 2,
                        ShapeUtil::MakeValidatedShape(F32, {}).value(), "y0");
    auto y1 = Parameter(&builder, 3,
                        ShapeUtil::MakeValidatedShape(S32, {}).value(), "y1");
    Tuple(&builder, {Add(x0, y0), Add(x1, y1)});
    auto computation_status = builder.Build();
    TF_CHECK_OK(computation_status.status());
    return std::move(computation_status).value();
  }();
  Scatter({operand0, operand1}, indices, {values0, values1}, add, dim_numbers);

  auto hlo_module = BuildHloGraph(&builder);

  // Run HLO cost analysis.
  HloCostAnalysis analysis;
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

TEST_F(FusionCostAnalysis, Broadcast) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = s8[] parameter(0)
  c1 = s8[] constant(0)
  a1 = s8[] add(p0, c1)
  b1 = s8[10000] broadcast(a1), dimensions={}
  b2 = s8[10000] broadcast(c1), dimensions={}
  ROOT r1 = s8[10000] add(b1, b2)
}

ENTRY e {
  param0 = s8[] parameter(0)
  ROOT r0 = s8[10000] fusion(param0), kind=kInput, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* root = module->entry_computation()->root_instruction();

  HloCostAnalysis analysis;
  ASSERT_IS_OK(root->Accept(&analysis));

  EXPECT_EQ(analysis.output_bytes_accessed(*root), 10000);
  EXPECT_EQ(analysis.operand_bytes_accessed(*root, 0), 1);
  EXPECT_EQ(analysis.bytes_accessed(*root), 10000 + 1);
  EXPECT_EQ(analysis.bytes_accessed(), 10000 + 1);
}

TEST_F(FusionCostAnalysis, RevisitModifiedFusion) {
  Shape r2f32 = ShapeUtil::MakeValidatedShape(F32, {2, 2}).value();
  HloComputation::Builder builder(TestName());
  HloInstruction* c1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/0.0f, /*to=*/1.0f, /*rows=*/2, /*cols=*/2)));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, c1, c1));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kMultiply, add, add));
  HloInstruction* neg = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kNegate, mul));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {neg, mul, add}, HloInstruction::FusionKind::kLoop);

  HloCostAnalysis::Options options;
  HloCostAnalysis analysis(options);
  ASSERT_IS_OK(fusion->Accept(&analysis));

  constexpr int64_t bytes_accessed = sizeof(float) * 2 * 2 * 2;
  static_assert(bytes_accessed == 32, "");

  EXPECT_EQ(analysis.flop_count(), 4 * 3);
  EXPECT_EQ(analysis.transcendental_count(), 0);
  EXPECT_EQ(analysis.bytes_accessed(), bytes_accessed);
  EXPECT_EQ(analysis.operand_bytes_accessed(*fusion, 0), sizeof(float) * 2 * 2);
  EXPECT_EQ(analysis.output_bytes_accessed(*fusion), sizeof(float) * 2 * 2);

  // Revisit the root (fusion) instruction and expect no changes.

  ASSERT_IS_OK(analysis.RevisitInstruction(fusion));

  EXPECT_EQ(analysis.flop_count(), 4 * 3);
  EXPECT_EQ(analysis.transcendental_count(), 0);
  EXPECT_EQ(analysis.bytes_accessed(), bytes_accessed);
  EXPECT_EQ(analysis.operand_bytes_accessed(*fusion, 0), sizeof(float) * 2 * 2);
  EXPECT_EQ(analysis.output_bytes_accessed(*fusion), sizeof(float) * 2 * 2);

  // Now modify the fusion and verify that the partially updated analysis is
  // correct.

  HloComputation* fused_computation = fusion->fused_instructions_computation();
  HloInstruction* to_replace = fused_computation->root_instruction();

  // Replace negate(multiply(...)) with exp(multiply(...)) at the fusion root.
  HloInstruction* exp =
      fused_computation->AddInstruction(HloInstruction::CreateUnary(
          r2f32, HloOpcode::kExp, to_replace->mutable_operand(0)));
  ASSERT_IS_OK(fused_computation->ReplaceInstruction(to_replace, exp));
  ASSERT_IS_OK(module->Verify());

  ASSERT_IS_OK(analysis.RevisitInstruction(fusion));

  // One floating point instruction (kNegate) removed.
  EXPECT_EQ(analysis.flop_count(), 4 * 2);
  // One transcendental instruction (kExp) added.
  EXPECT_EQ(analysis.transcendental_count(), 4);
  // The rest remains unchanged.
  EXPECT_EQ(analysis.bytes_accessed(), bytes_accessed);
  EXPECT_EQ(analysis.operand_bytes_accessed(*fusion, 0), sizeof(float) * 2 * 2);
  EXPECT_EQ(analysis.output_bytes_accessed(*fusion), sizeof(float) * 2 * 2);
}

TEST_F(FusionCostAnalysis, RevisitAlteredFusion) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  fp0 = s8[10] parameter(0)
  ROOT fr = s8[1] slice(fp0), slice={[0:1]}
}

ENTRY e {
  p0 = s8[10] parameter(0)
  ROOT r = s8[1] fusion(p0), kind=kLoop, calls=f
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* root = module->entry_computation()->root_instruction();

  HloCostAnalysis modified_analysis;
  ASSERT_IS_OK(root->Accept(&modified_analysis));
  HloInstruction* fusion_root =
      root->called_computations()[0]->root_instruction();
  EXPECT_FLOAT_EQ(modified_analysis.operand_utilization(*fusion_root, 0), 0.1);

  // Modify fusion root, revisit the fusion with the analysis and expect
  // updated values. Compare against a complete fresh analysis.
  fusion_root->mutable_slice_limits()->at(0) = 2;
  fusion_root->mutable_shape()->set_dimensions(0, 2);
  root->mutable_shape()->set_dimensions(0, 2);
  module->mutable_config().SetDefaultComputationLayout(
      module->entry_computation()->ComputeProgramShape());
  ASSERT_IS_OK(modified_analysis.RevisitInstruction(root));

  HloCostAnalysis unmodified_analysis;
  ASSERT_IS_OK(root->Accept(&unmodified_analysis));

  EXPECT_FLOAT_EQ(modified_analysis.operand_utilization(*fusion_root, 0), 0.2);
  EXPECT_FLOAT_EQ(modified_analysis.operand_utilization(*fusion_root, 0),
                  unmodified_analysis.operand_utilization(*fusion_root, 0));
}

TEST_F(FusionCostAnalysis, RevisitWithSharedComputation) {
  absl::string_view hlo_string = R"(
HloModule m

add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT r = f32[] add(arg_0.1, arg_1.1)
}

ENTRY e {
  p0 = f32[127,125] parameter(0)
  p1 = f32[127,125] parameter(1)
  constant_zero = f32[] constant(0)
  r0 = f32[127] reduce(p0, constant_zero), dimensions={1}, to_apply=add_computation
  r1 = f32[127] reduce(p0, constant_zero), dimensions={1}, to_apply=add_computation
  ROOT _ = f32[127] add(r0, r1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* root = module->entry_computation()->root_instruction();
  HloCostAnalysis analysis;

  // add_computation is shared by two reductions - r0 and r1.
  // Removing/revisiting one of them should not affect the other one.
  HloInstruction* add_root =
      root->operand(1)->called_computations()[0]->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis));
  EXPECT_EQ(analysis.operand_utilization(*add_root, 0), 1);
  ASSERT_IS_OK(analysis.RemoveInstruction(root->mutable_operand(0)));
  EXPECT_EQ(analysis.operand_utilization(*add_root, 0), 1);
  ASSERT_IS_OK(analysis.RevisitInstruction(root->mutable_operand(0)));
  EXPECT_EQ(analysis.operand_utilization(*add_root, 0), 1);
}

using Properties = HloCostAnalysis::Properties;
constexpr auto kFlopsKey = HloCostAnalysis::kFlopsKey;
constexpr auto kTranscendentalsKey = HloCostAnalysis::kTranscendentalsKey;
constexpr auto kBytesAccessedKey = HloCostAnalysis::kBytesAccessedKey;
constexpr auto kOptimalSecondsKey = HloCostAnalysis::kOptimalSecondsKey;
constexpr auto kUtilizationKey = HloCostAnalysis::kUtilizationKey;
constexpr auto kReserved0Key = HloCostAnalysis::kReserved0Key;

TEST(HloCostAnalysisProperties, ZeroWhenInitialized) {
  Properties p;
  EXPECT_EQ(0, p[kFlopsKey]);
  EXPECT_EQ(0, p[kTranscendentalsKey]);
  EXPECT_EQ(0, p[kBytesAccessedKey]);
  EXPECT_EQ(0, p[kOptimalSecondsKey]);
  EXPECT_EQ(0, p[kUtilizationKey]);
  EXPECT_EQ(0, p[kReserved0Key]);

  EXPECT_EQ(0, p.operand_utilization(0, {}));
  EXPECT_EQ(0, p.operand_utilization(1, {}));
  EXPECT_EQ(0, p.operand_utilization(2, {}));
  EXPECT_EQ(0, p.operand_utilization(0, {0}));
  EXPECT_EQ(0, p.operand_utilization(2, {0}));
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandUtilizationKey(0, {})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandUtilizationKey(1, {})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandUtilizationKey(2, {})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandUtilizationKey(0, {0})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandUtilizationKey(2, {0})]);

  EXPECT_EQ(0, p.operand_bytes_accessed(0, {}));
  EXPECT_EQ(0, p.operand_bytes_accessed(1, {}));
  EXPECT_EQ(0, p.operand_bytes_accessed(2, {}));
  EXPECT_EQ(0, p.operand_bytes_accessed(0, {0}));
  EXPECT_EQ(0, p.operand_bytes_accessed(2, {0}));
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandBytesAccessedKey(0, {})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandBytesAccessedKey(1, {})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandBytesAccessedKey(2, {})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandBytesAccessedKey(0, {0})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOperandBytesAccessedKey(2, {0})]);

  EXPECT_EQ(0, p.output_bytes_accessed({}));
  EXPECT_EQ(0, p.output_bytes_accessed({0}));
  EXPECT_EQ(0, p[HloCostAnalysis::GetOutputBytesAccessedKey({})]);
  EXPECT_EQ(0, p[HloCostAnalysis::GetOutputBytesAccessedKey({0})]);

  EXPECT_EQ(0, p["foobar"]);

  std::vector<std::pair<std::string, float>> vals;
  Properties().ForEach([&](absl::string_view key, float val) {
    vals.push_back({std::string(key), val});
  });
  EXPECT_THAT(vals, ::testing::IsEmpty());
}

TEST(HloCostAnalysisProperties, SetValues) {
  Properties p;

  p[kFlopsKey] = 1;
  p[kTranscendentalsKey] = 2;
  p[kBytesAccessedKey] = 3;
  p[kOptimalSecondsKey] = 4;
  p[kUtilizationKey] = 5;
  p[kReserved0Key] = 6;
  EXPECT_EQ(1, p[kFlopsKey]);
  EXPECT_EQ(2, p[kTranscendentalsKey]);
  EXPECT_EQ(3, p[kBytesAccessedKey]);
  EXPECT_EQ(4, p[kOptimalSecondsKey]);
  EXPECT_EQ(5, p[kUtilizationKey]);
  EXPECT_EQ(6, p[kReserved0Key]);

  p.set_operand_utilization(0, {}, 10);
  p.set_operand_utilization(1, {}, 11);
  p.set_operand_utilization(2, {}, 12);
  p.set_operand_utilization(0, {0}, 13);
  p.set_operand_utilization(2, {0}, 14);
  EXPECT_EQ(10, p.operand_utilization(0, {}));
  EXPECT_EQ(11, p.operand_utilization(1, {}));
  EXPECT_EQ(12, p.operand_utilization(2, {}));
  EXPECT_EQ(13, p.operand_utilization(0, {0}));
  EXPECT_EQ(14, p.operand_utilization(2, {0}));
  EXPECT_EQ(10, p[HloCostAnalysis::GetOperandUtilizationKey(0, {})]);
  EXPECT_EQ(11, p[HloCostAnalysis::GetOperandUtilizationKey(1, {})]);
  EXPECT_EQ(12, p[HloCostAnalysis::GetOperandUtilizationKey(2, {})]);
  EXPECT_EQ(13, p[HloCostAnalysis::GetOperandUtilizationKey(0, {0})]);
  EXPECT_EQ(14, p[HloCostAnalysis::GetOperandUtilizationKey(2, {0})]);

  p.set_operand_bytes_accessed(0, {}, 20);
  p.set_operand_bytes_accessed(1, {}, 21);
  p.set_operand_bytes_accessed(2, {}, 22);
  p.set_operand_bytes_accessed(0, {0}, 23);
  p.set_operand_bytes_accessed(2, {0}, 24);
  EXPECT_EQ(20, p.operand_bytes_accessed(0, {}));
  EXPECT_EQ(21, p.operand_bytes_accessed(1, {}));
  EXPECT_EQ(22, p.operand_bytes_accessed(2, {}));
  EXPECT_EQ(23, p.operand_bytes_accessed(0, {0}));
  EXPECT_EQ(24, p.operand_bytes_accessed(2, {0}));
  EXPECT_EQ(20, p[HloCostAnalysis::GetOperandBytesAccessedKey(0, {})]);
  EXPECT_EQ(21, p[HloCostAnalysis::GetOperandBytesAccessedKey(1, {})]);
  EXPECT_EQ(22, p[HloCostAnalysis::GetOperandBytesAccessedKey(2, {})]);
  EXPECT_EQ(23, p[HloCostAnalysis::GetOperandBytesAccessedKey(0, {0})]);
  EXPECT_EQ(24, p[HloCostAnalysis::GetOperandBytesAccessedKey(2, {0})]);

  p.set_output_bytes_accessed({}, 30);
  p.set_output_bytes_accessed({0}, 31);
  EXPECT_EQ(30, p.output_bytes_accessed({}));
  EXPECT_EQ(31, p.output_bytes_accessed({0}));
  EXPECT_EQ(30, p[HloCostAnalysis::GetOutputBytesAccessedKey({})]);
  EXPECT_EQ(31, p[HloCostAnalysis::GetOutputBytesAccessedKey({0})]);

  p["foo"] = 100;
  EXPECT_EQ(100, p["foo"]);

  p["bar"] += 101;
  EXPECT_EQ(101, p["bar"]);
}
}  // namespace
}  // namespace xla
