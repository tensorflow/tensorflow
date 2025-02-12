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

#include "xla/service/transpose_folding.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::tsl::testing::IsOkAndHolds;

using TransposeFoldingTest = HloTestBase;

TEST_F(TransposeFoldingTest, FoldDotTranspose) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = f32[2,3]{1,0} parameter(0)
  y = f32[2,3]{1,0} parameter(1)
  transpose = f32[3,2]{1,0} transpose(y), dimensions={1,0}
  ROOT dot = f32[2,2]{1,0} dot(x, transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/1));
}

TEST_F(TransposeFoldingTest, DontFoldTransposeOfBatchDimByDefault) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = f32[2,3] parameter(0)
  y = f32[3,2] parameter(1)
  transpose = f32[2,3] transpose(y), dimensions={1,0}
  ROOT dot = f32[2] dot(x, transpose), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldTransposeOfBatchWhenPermitted) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = f32[5,2,3] parameter(0)
  y = f32[3,5,4] parameter(1)
  transpose = f32[5,3,4] transpose(y), dimensions={1,0,2}
  ROOT dot = f32[5,2,4] dot(x, transpose), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  TransposeFolding transpose_folding(
      /*dot_can_fold_transpose_operand=*/[](const HloInstruction&, int64_t) {
        return true;
      });
  EXPECT_THAT(transpose_folding.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/2, /*rhs_contracting_dim=*/0));
}

TEST_F(TransposeFoldingTest, DontFoldTransposeOfRank1Dot) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = f32[3] parameter(0)
  y = f32[3,2] parameter(1)
  transpose = f32[2,3] transpose(y), dimensions={1,0}
  ROOT dot = f32[2] dot(x, transpose), lhs_batch_dims={}, rhs_batch_dims={}, lhs_contracting_dims={0}, rhs_contracting_dims={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, DontFoldTransposeOfDotWithoutContractingDims) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = f32[3,4] parameter(0)
  y = f32[3,4,6,7] parameter(1)
  transpose = f32[3,4,7,6] transpose(y), dimensions={0,1,3,2}
  ROOT dot = f32[3,4,7,6] dot(x, transpose), lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={}, rhs_contracting_dims={}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldDotTransposeConstant) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldDotTransposeConstant

ENTRY entry_computation {
  constant = f32[2,1]{1,0} constant({ { 1 }, { 2 } })
  transpose = f32[1,2]{1,0} transpose(constant), dimensions={1,0}
  constant.1 = f32[3,2]{1,0} constant({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  transpose.1 = f32[2,3]{1,0} transpose(constant.1), dimensions={1,0}
  ROOT dot = f32[1,3]{1,0} dot(transpose, transpose.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Constant(), op::Constant(),
                      /*lhs_contracting_dim=*/0, /*rhs_contracting_dim=*/1));
}

TEST_F(TransposeFoldingTest, FuseDotWithConstantOperands) {
  auto builder = HloComputation::Builder("entry");
  // (1.0 + 2.0) * (2.0 - 3.0)
  HloInstruction* const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* const2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  HloInstruction* const3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      const1->shape(), HloOpcode::kAdd, const1, const2));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      const2->shape(), HloOpcode::kSubtract, const2, const3));
  HloInstruction* mul = builder.AddInstruction(HloInstruction::CreateBinary(
      add->shape(), HloOpcode::kMultiply, add, sub));

  auto module = CreateNewVerifiedModule("fuse_with_constant_operands");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(mul));
  HloInstruction* call = module->OutlineExpressionFromComputation(
      {add, sub, mul}, "entry", entry_computation);
  EXPECT_EQ(call, entry_computation->root_instruction());
  HloComputation* callee_computation = call->to_apply();
  // The arguments to the call should be const1, const2, and const3.
  EXPECT_THAT(call->operands(),
              ::testing::UnorderedElementsAre(const1, const2, const3));

  // The callee should contain 3 parameters and 3 binary operators.
  EXPECT_EQ(6, callee_computation->instruction_count());
}

TEST_F(TransposeFoldingTest, FoldDotTransposeInCall) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldDotTransposeInCall

callee {
  name.0 = f32[2,3]{1,0} parameter(0)
  name.1 = f32[2,3]{1,0} parameter(1)
  transpose.clone = f32[3,2]{1,0} transpose(name.0), dimensions={1,0}
  ROOT dot.clone = f32[2,2]{1,0} dot(name.1, transpose.clone), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry_computation {
  y = f32[2,3]{1,0} parameter(1)
  x = f32[2,3]{1,0} parameter(0)
  ROOT call = f32[2,2]{1,0} call(y, x), to_apply=callee
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));

  const HloComputation* callee = module->GetComputationWithName("callee");
  ASSERT_NE(callee, nullptr);
  EXPECT_THAT(callee->root_instruction(),
              op::Dot(op::Parameter(1), op::Parameter(0),
                      /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/1));
}

// Test that a two dimension swap of the kernel gets folded into convolution.
TEST_F(TransposeFoldingTest, FoldConvDimSwapTransposeRhs) {
  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {3, 2, 1, 1}),
      /*name=*/"y"));
  HloInstruction* transpose_y =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), y, {1, 0, 2, 3}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(
        transpose_y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  absl::StatusOr<Shape> conv_shape = ShapeInference::InferConvolveShape(
      x->shape(), transpose_y->shape(), /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      /*preferred_element_type=*/std::nullopt);
  EXPECT_IS_OK(conv_shape);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape.value(), x, transpose_y,
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));

  // Instructions after folding: x, y, and the convolution.
  absl::flat_hash_set<HloInstruction*> instruction_set(
      entry_computation->instructions().begin(),
      entry_computation->instructions().end());
  CHECK_EQ(1, instruction_set.erase(x)) << "x is not in entry_computation.";
  CHECK_EQ(1, instruction_set.erase(y)) << "y is not in entry_computation.";
  CHECK_EQ(1, instruction_set.size())
      << "entry_computation should contain exactly 3 instructions.";
  HloInstruction* new_conv = *instruction_set.begin();
  EXPECT_EQ(HloOpcode::kConvolution, new_conv->opcode());
  EXPECT_EQ(dnums.kernel_input_feature_dimension(),
            new_conv->convolution_dimension_numbers()
                .kernel_output_feature_dimension());
  EXPECT_EQ(dnums.kernel_output_feature_dimension(),
            new_conv->convolution_dimension_numbers()
                .kernel_input_feature_dimension());
}

// Test that a complex transpose of the kernel gets folded into convolution.
TEST_F(TransposeFoldingTest, FoldConvComplexTransposeRhs) {
  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {1, 2, 1, 3}),
      /*name=*/"y"));
  HloInstruction* transpose_y =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), y, {1, 3, 0, 2}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(
        transpose_y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  absl::StatusOr<Shape> conv_shape = ShapeInference::InferConvolveShape(
      x->shape(), transpose_y->shape(), /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      /*preferred_element_type=*/std::nullopt);
  EXPECT_IS_OK(conv_shape);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape.value(), x, transpose_y,
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));

  // Instructions after folding: x, y, and the convolution.
  absl::flat_hash_set<HloInstruction*> instruction_set(
      entry_computation->instructions().begin(),
      entry_computation->instructions().end());
  CHECK_EQ(1, instruction_set.erase(x)) << "x is not in entry_computation.";
  CHECK_EQ(1, instruction_set.erase(y)) << "y is not in entry_computation.";
  CHECK_EQ(1, instruction_set.size())
      << "entry_computation should contain exactly 3 instructions.";
  HloInstruction* new_conv = *instruction_set.begin();
  EXPECT_EQ(HloOpcode::kConvolution, new_conv->opcode());
  EXPECT_EQ(dnums.kernel_input_feature_dimension(),
            new_conv->convolution_dimension_numbers()
                .kernel_output_feature_dimension());
  EXPECT_EQ(dnums.kernel_spatial_dimensions(1),
            new_conv->convolution_dimension_numbers()
                .kernel_input_feature_dimension());
  EXPECT_EQ(
      dnums.kernel_output_feature_dimension(),
      new_conv->convolution_dimension_numbers().kernel_spatial_dimensions(0));
  EXPECT_EQ(
      dnums.kernel_spatial_dimensions(0),
      new_conv->convolution_dimension_numbers().kernel_spatial_dimensions(1));
}

// Test that a transpose of the activations gets folded into convolution.
TEST_F(TransposeFoldingTest, FoldConvTransposeLhs) {
  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {3, 2, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"y"));
  HloInstruction* transpose_x =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), x, {1, 0, 2, 3}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  absl::StatusOr<Shape> conv_shape = ShapeInference::InferConvolveShape(
      transpose_x->shape(), y->shape(), /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      /*preferred_element_type=*/std::nullopt);
  EXPECT_IS_OK(conv_shape);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape.value(), transpose_x, y,
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));

  // Instructions after folding: x, y, and the convolution.
  absl::flat_hash_set<HloInstruction*> instruction_set(
      entry_computation->instructions().begin(),
      entry_computation->instructions().end());
  EXPECT_EQ(1, instruction_set.erase(x)) << "x is not in entry_computation.";
  EXPECT_EQ(1, instruction_set.erase(y)) << "y is not in entry_computation.";
  EXPECT_EQ(1, instruction_set.size())
      << "entry_computation should contain exactly 3 instructions.";
  HloInstruction* new_conv = *instruction_set.begin();
  EXPECT_EQ(HloOpcode::kConvolution, new_conv->opcode());
  EXPECT_EQ(dnums.input_feature_dimension(),
            new_conv->convolution_dimension_numbers().input_batch_dimension());
  EXPECT_EQ(
      dnums.input_batch_dimension(),
      new_conv->convolution_dimension_numbers().input_feature_dimension());
  EXPECT_EQ(
      dnums.input_spatial_dimensions(0),
      new_conv->convolution_dimension_numbers().input_spatial_dimensions(0));
  EXPECT_EQ(
      dnums.input_spatial_dimensions(1),
      new_conv->convolution_dimension_numbers().input_spatial_dimensions(1));
  EXPECT_EQ(
      dnums.output_spatial_dimensions(0),
      new_conv->convolution_dimension_numbers().output_spatial_dimensions(0));
  EXPECT_EQ(
      dnums.output_spatial_dimensions(1),
      new_conv->convolution_dimension_numbers().output_spatial_dimensions(1));
}

// Test that a transpose of every dimension in the activations gets folded into
// convolution.
TEST_F(TransposeFoldingTest, FoldConvComplexTransposeLhs) {
  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {3, 2, 1, 1}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {2, 3, 1, 1}),
      /*name=*/"y"));
  HloInstruction* transpose_x =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3, 1, 1}), x, {1, 0, 3, 2}));
  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers();
  Window window;
  for (int i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    dim->set_stride(1);
    dim->set_size(y->shape().dimensions(dnums.kernel_spatial_dimensions(i)));
  }
  absl::StatusOr<Shape> conv_shape = ShapeInference::InferConvolveShape(
      transpose_x->shape(), y->shape(), /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      /*preferred_element_type=*/std::nullopt);
  EXPECT_IS_OK(conv_shape);
  HloInstruction* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      conv_shape.value(), transpose_x, y,
      /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule("test_module");
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build(conv));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));

  // Instructions after folding: x, y, and the convolution.
  absl::flat_hash_set<HloInstruction*> instruction_set(
      entry_computation->instructions().begin(),
      entry_computation->instructions().end());
  EXPECT_EQ(1, instruction_set.erase(x)) << "x is not in entry_computation.";
  EXPECT_EQ(1, instruction_set.erase(y)) << "y is not in entry_computation.";
  EXPECT_EQ(1, instruction_set.size())
      << "entry_computation should contain exactly 3 instructions.";
  HloInstruction* new_conv = *instruction_set.begin();
  EXPECT_EQ(HloOpcode::kConvolution, new_conv->opcode());
  EXPECT_EQ(dnums.input_feature_dimension(),
            new_conv->convolution_dimension_numbers().input_batch_dimension());
  EXPECT_EQ(
      dnums.input_batch_dimension(),
      new_conv->convolution_dimension_numbers().input_feature_dimension());
  EXPECT_EQ(
      dnums.input_spatial_dimensions(0),
      new_conv->convolution_dimension_numbers().input_spatial_dimensions(1));
  EXPECT_EQ(
      dnums.input_spatial_dimensions(1),
      new_conv->convolution_dimension_numbers().input_spatial_dimensions(0));
  EXPECT_EQ(
      dnums.output_spatial_dimensions(0),
      new_conv->convolution_dimension_numbers().output_spatial_dimensions(0));
  EXPECT_EQ(
      dnums.output_spatial_dimensions(1),
      new_conv->convolution_dimension_numbers().output_spatial_dimensions(1));
}

TEST_F(TransposeFoldingTest, FoldBatchDotTranspose) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldBatchDotTranspose

ENTRY entry_computation {
  x = f32[7,7,2,3]{3,2,1,0} parameter(0)
  y = f32[7,7,2,3]{3,2,1,0} parameter(1)
  transpose = f32[7,7,3,2]{3,2,1,0} transpose(y), dimensions={0,1,3,2}
  ROOT dot = f32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/3, /*rhs_contracting_dim=*/3));
}

TEST_F(TransposeFoldingTest, NoFoldBatchDotTransposeBatch) {
  constexpr absl::string_view kHloString = R"(
HloModule NoFoldBatchDotTransposeBatch

ENTRY entry_computation {
  x = f32[7,7,2,3]{3,2,1,0} parameter(0)
  y = f32[7,7,2,3]{3,2,1,0} parameter(1)
  transpose = f32[7,7,3,2]{3,2,1,0} transpose(y), dimensions={1,0,3,2}
  ROOT dot = f32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldBatchDotTransposeNonContiguousBatch) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldBatchDotTransposeNonContiguousBatch

ENTRY entry_computation {
  x = f32[7,2,7,3]{3,2,1,0} parameter(0)
  y = f32[7,2,7,3]{3,2,1,0} parameter(1)
  transpose = f32[7,3,7,2]{3,2,1,0} transpose(y), dimensions={0,3,2,1}
  ROOT dot = f32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={1}, lhs_batch_dims={0,2}, rhs_batch_dims={0,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/3, /*rhs_contracting_dim=*/3));
}

TEST_F(TransposeFoldingTest, NoFoldBatchDotTransposeIdentity) {
  constexpr absl::string_view kHloString = R"(
HloModule NoFoldBatchDotTransposeIdentity

ENTRY entry_computation {
  x = f32[7,7,2,3]{3,2,1,0} parameter(0)
  y = f32[7,7,3,2]{3,2,1,0} parameter(1)
  transpose = f32[7,7,3,2]{3,2,1,0} transpose(y), dimensions={0,1,2,3}
  ROOT dot = f32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldTransposeWithBackendConfig) {
  constexpr absl::string_view kHloString = R"(
HloModule FoldTransposeWithBackendConfig

ENTRY entry_computation {
  x = f32[7,2,7,3]{3,2,1,0} parameter(0)
  y = f32[7,2,7,3]{3,2,1,0} parameter(1)
  transpose = f32[7,3,7,2]{3,2,1,0} transpose(y), dimensions={0,3,2,1}
  ROOT dot = f32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={1}, lhs_batch_dims={0,2}, rhs_batch_dims={0,2}, backend_config={"force_earliest_schedule":false}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_TRUE(
      module->entry_computation()->root_instruction()->has_backend_config());
}

}  // namespace
}  // namespace xla
