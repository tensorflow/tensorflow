/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dynamic_padder.h"

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

OpDynamismSupport OpHasDynamismSupport(HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kCustomCall) {
    return OpDynamismSupport::kNoSupport;
  }
  if (hlo->custom_call_target() == "OpWithDynamicLowering") {
    return OpDynamismSupport::kRequired;
  }
  return OpDynamismSupport::kNoSupport;
}

Status CustomCallDynamicDimensionInference(
    HloInstruction* hlo, DynamicDimensionInference* inferencer) {
  if (hlo->custom_call_target() == "OpWithDynamicLowering") {
    if (hlo->shape().IsTuple()) {
      // Use the operand's dynamic size as output dynamic size.
      HloInstruction* dynamic_size =
          inferencer->GetDynamicSize(hlo->mutable_operand(0), {1}, 0);
      inferencer->SetDynamicSize(hlo, {1}, 0, dynamic_size);
    } else {
      // Use the operand's dynamic size as output dynamic size.
      HloInstruction* dynamic_size =
          inferencer->GetDynamicSize(hlo->mutable_operand(0), {}, 0);
      inferencer->SetDynamicSize(hlo, {}, 0, dynamic_size);
    }
  }

  return Status::OK();
}

class DynamicPadderTest : public HloTestBase {
 protected:
  DynamicPadderTest() : HloTestBase() { module_ = CreateNewVerifiedModule(); }

  std::unique_ptr<HloModule> GetHloModule(const string& hlo_text) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
    return module;
  }

  StatusOr<bool> RunPadder(bool slice_dynamic_output = false) {
    DynamicPadder padder(/*slice_dynamic_output=*/slice_dynamic_output,
                         CustomCallDynamicDimensionInference,
                         OpHasDynamismSupport);
    return padder.Run(module_.get());
  }

  void ExpectPadded(const HloInstruction* inst) {
    EXPECT_THAT(inst,
                op::Select(op::Lt(op::Iota(), op::Broadcast(op::Parameter())),
                           ::testing::_, op::Broadcast()));
  }

  HloComputation* GetScalarAddComputation() {
    auto embedded_builder = HloComputation::Builder("add");
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {}), "rhs"));
    embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
    return module_->AddEmbeddedComputation(embedded_builder.Build());
  }

  std::unique_ptr<HloModule> module_;
  const Shape scalar_shape_ = ShapeUtil::MakeShape(S32, {});
};

TEST_F(DynamicPadderTest, ReduceTest) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {2});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(input_shape, HloOpcode::kNegate, data_param));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, negate, init, {0, 2}, GetScalarAddComputation()));
  EXPECT_FALSE(module_->is_dynamic());
  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  TF_ASSERT_OK(RunPadder().status());

  ExpectPadded(reduce->operand(0));
  EXPECT_TRUE(module_->is_dynamic());
}

TEST_F(DynamicPadderTest, DynamicLoweringTest) {
  const string hlo_text = R"(
HloModule DynamicLowering

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[<=5] set-dimension-size(param, const),
                dimensions={0}
  custom-call.1 = s32[<=5] custom-call(param_padded),
    custom_call_target="OpWithDynamicLowering"
  custom-call.2 = s32[<=5] custom-call(custom-call.1),
    custom_call_target="OpWithDynamicLowering"
  // Negate doesn't support dynamic lowering.
  ROOT negate = s32[<=5] negate(custom-call.2)
}
)";

  module_ = GetHloModule(hlo_text);

  TF_ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  // After rewrite, we should have :
  //
  //   param
  //     |
  //  SliceToDynamic
  //     |
  //  OpWithDynamicLowering (custom_call_1)
  //     |
  //  OpWithDynamicLowering (custom_call_2)
  //     |
  //  PadToStatic
  //     |
  //   Negate
  //     |
  //   SliceToDynamic // Root require dynamic form tensor.
  auto custom_call_1 =
      module_->entry_computation()->GetInstructionWithName("custom-call.1");
  auto custom_call_2 =
      module_->entry_computation()->GetInstructionWithName("custom-call.2");
  // Test that the input to custom call
  HloInstruction* slice_to_dynamic = custom_call_1->mutable_operand(0);
  ASSERT_THAT(slice_to_dynamic->opcode(), HloOpcode::kCustomCall);
  ASSERT_THAT(slice_to_dynamic->custom_call_target(), "SliceToDynamic");
  ASSERT_EQ(custom_call_2->user_count(), 1);
  HloInstruction* pad_to_static = custom_call_2->users()[0];
  ASSERT_THAT(pad_to_static->opcode(), HloOpcode::kCustomCall);
  ASSERT_THAT(pad_to_static->custom_call_target(), "PadToStatic");
  slice_to_dynamic = module_->entry_computation()->root_instruction();
  ASSERT_THAT(slice_to_dynamic->opcode(), HloOpcode::kCustomCall);
  ASSERT_THAT(slice_to_dynamic->custom_call_target(), "SliceToDynamic");
}

TEST_F(DynamicPadderTest, DynamicLoweringTestTupleInput) {
  const string hlo_text = R"(
HloModule DynamicLowering

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[<=5] set-dimension-size(param, const),
                dimensions={0}
  // Create a tuple with static and dynamic componenet.
  tuple_arg = (s32[], s32[<=5]) tuple(const, param_padded)
  custom-call.1 = (s32[], s32[<=5]) custom-call(tuple_arg),
    custom_call_target="OpWithDynamicLowering"
  custom-call.2 = (s32[], s32[<=5]) custom-call(custom-call.1),
    custom_call_target="OpWithDynamicLowering"
  data = s32[<=5]{0} get-tuple-element(custom-call.2), index=1
  // Negate doesn't support dynamic lowering.
  ROOT negate = s32[<=5] negate(data)
}
)";

  module_ = GetHloModule(hlo_text);

  TF_ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  // After rewrite, we should have :
  //
  //   param
  //     |
  //  SliceToDynamic
  //     |
  //    Tuple
  //     |
  //  OpWithDynamicLowering (custom_call_1)
  //     |
  //  OpWithDynamicLowering (custom_call_2)
  //     |
  //   GTE
  //     |
  //  PadToStatic
  //     |
  //   Negate
  //     |
  //   SliceToDynamic // Root require dynamic form tensor.

  auto* root = module_->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::CustomCall("SliceToDynamic", op::Negate(), op::Constant()));
  HloInstruction* negate = root->mutable_operand(0);
  EXPECT_THAT(
      negate,
      op::Negate(op::GetTupleElement(op::CustomCall(
          "PadToStatic", op::GetTupleElement(op::CustomCall(
                             "OpWithDynamicLowering", ::testing::_))))));
  auto custom_call_1 =
      module_->entry_computation()->GetInstructionWithName("custom-call.1");
  EXPECT_THAT(custom_call_1,
              op::CustomCall(
                  "OpWithDynamicLowering",
                  op::Tuple(op::Constant(), op::CustomCall("SliceToDynamic"))));
}

TEST_F(DynamicPadderTest, ConvolutionTest) {
  auto builder = HloComputation::Builder(TestName());
  constexpr int xdim = 3;
  constexpr int ydim = 2;
  constexpr int zdim = 1;
  auto xy_shape = ShapeUtil::MakeShape(F32, {xdim, ydim});
  auto yz_shape = ShapeUtil::MakeShape(F32, {ydim, zdim});
  auto zx_shape = ShapeUtil::MakeShape(F32, {zdim, xdim});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers(0);

  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(1);
  dnums.set_output_feature_dimension(0);

  Window window;

  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      zx_shape, a_param, b_param, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  // Set up binding for contracting dimensions.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  TF_ASSERT_OK(RunPadder().status());

  ExpectPadded(conv->operand(0));
}

TEST_F(DynamicPadderTest, ConvolutionNoPad) {
  auto builder = HloComputation::Builder(TestName());
  constexpr int xdim = 3;
  constexpr int ydim = 2;
  constexpr int zdim = 1;
  auto xy_shape = ShapeUtil::MakeShape(F32, {xdim, ydim});
  auto yz_shape = ShapeUtil::MakeShape(F32, {ydim, zdim});
  auto zx_shape = ShapeUtil::MakeShape(F32, {zdim, xdim});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers(0);

  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(1);
  dnums.set_output_feature_dimension(0);

  Window window;

  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      zx_shape, a_param, b_param, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums,
      HloTestBase::DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding for non-contracting dimension.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{2, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));

  TF_ASSERT_OK(RunPadder().status());

  EXPECT_THAT(conv->operand(0), op::Parameter());
}

TEST_F(DynamicPadderTest, ReduceWindowNoPadForTrivialWindow) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {3, 5});

  auto input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));
  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));
  TF_ASSERT_OK_AND_ASSIGN(Window window, ParseWindow("size=2x1 pad=0_0x0_0"));
  auto output = builder.AddInstruction(HloInstruction::CreateReduceWindow(
      reduce_shape, input, init, window, GetScalarAddComputation()));

  module_->AddEntryComputation(builder.Build());

  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));

  TF_ASSERT_OK(RunPadder().status());

  EXPECT_THAT(output->operand(0), op::Parameter());
}

// Test that dynamic padder has the same result as if not padded.
class ExecutionTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> GetHloModule(const string& hlo_text) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo_text).ValueOrDie();
    return module;
  }
  Literal PadAndExecute(std::unique_ptr<HloModule> module,
                        absl::Span<Literal* const> arguments,
                        bool slice_dynamic_output = true) {
    if (!slice_dynamic_output) {
      auto new_config = module->config();
      new_config.mutable_entry_computation_layout()
          ->mutable_result_layout()
          ->ClearDynamicShape();
      module->set_config(new_config);
    }
    DynamicPadder padder(slice_dynamic_output);
    TF_CHECK_OK(padder.Run(module.get()).status());
    HloDCE dce;
    TF_CHECK_OK(dce.Run(module.get()).status());
    return ExecuteAndTransfer(std::move(module), arguments);
  }
};

XLA_TEST_F(ExecutionTest, ScatterUpdate) {
  // Test that scattering on indices=[2] is same as scattering on indices=[4]
  // and dynamic dimension = 2
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[INDICES_BOUND] parameter(1)
  updates = s32[INDICES_BOUND,3] parameter(2)
  dynamic_size = s32[] parameter(3)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1

}
)";
  const string hlo_text_not_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "2"}});
  auto module_not_padded = GetHloModule(hlo_text_not_padded);

  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  Literal dynamic_size = LiteralUtil::CreateR0<int32>(2);

  Literal not_padded =
      ExecuteAndTransfer(std::move(module_not_padded),
                         {&operand, &scatter_indices, &updates, &dynamic_size});

  // Pad input to 4.
  const string hlo_text_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "4"}});
  auto module_padded = GetHloModule(hlo_text_padded);
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_padded->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));
  TF_CHECK_OK(module_padded->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{2, {}, 0}));
  // Pad the rest of input with garbage data.
  Literal scatter_indices_padded = LiteralUtil::CreateR1<int32>({0, 2, 0, 4});
  Literal updates_padded = LiteralUtil::CreateR2<int32>(
      {{10, 20, 30}, {70, 80, 90}, {30, 22, 11}, {-1, 20, -1}});
  DynamicPadder padder;
  TF_CHECK_OK(padder.Run(module_padded.get()).status());
  Literal padded = PadAndExecute(
      std::move(module_padded),
      {&operand, &scatter_indices_padded, &updates_padded, &dynamic_size});

  EXPECT_EQ(padded, not_padded);
}

XLA_TEST_F(ExecutionTest, ScatterUpdateWindowDim) {
  const string hlo_text = R"(
HloModule ScatterUpdateWindowDim

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[1,2,3] parameter(0)
  indices = s32[1] parameter(1)
  updates = s32[2,3,1] parameter(2)
  dynamic_size = s32[] constant(1)
  operand_dynamic = s32[1, <=2, 3] set-dimension-size(operand, dynamic_size),
      dimensions={1}
  updates_dynamic = s32[<=2, 3, 1] set-dimension-size(updates, dynamic_size),
      dimensions={0}
  ROOT scatter = s32[1, <=2, 3] scatter(operand_dynamic, indices, updates_dynamic),
      to_apply=update_s32,
      update_window_dims={0, 1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1

}
)";
  auto hlo_module = GetHloModule(hlo_text);

  Literal operand = LiteralUtil::CreateR3<int32>({{{0, 0, 0}, {0, 0, 0}}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0});
  Literal updates =
      LiteralUtil::CreateR3<int32>({{{10}, {20}, {30}}, {{70}, {80}, {90}}});

  Literal padded = PadAndExecute(std::move(hlo_module),
                                 {&operand, &scatter_indices, &updates}, false);
  Literal expected =
      LiteralUtil::CreateR3<int32>({{{10, 20, 30}, {70, 80, 90}}});
  EXPECT_EQ(padded, expected);
}

XLA_TEST_F(ExecutionTest, ScatterUpdateF32) {
  // Test that scattering on indices=[2] is same as scattering on indices=[4]
  // and dynamic dimension = 2
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  ROOT rhs = f32[] parameter(1)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  dynamic_size = s32[] parameter(3)
  ROOT scatter = f32[3,3] scatter(operand, indices, updates),
      to_apply=update_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1

}
)";

  auto module_not_padded = GetHloModule(hlo_text);

  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<float>({{10.0, 20.0, 30.0}, {70.0, 80.0, 90.0}});
  // Dynamic Size is 1, pad to 2
  Literal dynamic_size = LiteralUtil::CreateR0<int32>(1);

  auto module_padded = GetHloModule(hlo_text);
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_padded->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));
  TF_CHECK_OK(module_padded->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{3, {}},
      DynamicParameterBinding::DynamicDimension{2, {}, 0}));
  DynamicPadder padder;
  TF_CHECK_OK(padder.Run(module_padded.get()).status());
  Literal not_padded =
      PadAndExecute(std::move(module_padded),
                    {&operand, &scatter_indices, &updates, &dynamic_size});
  // Although we have two indices, only the first element is updated because of
  // padding.
  EXPECT_EQ(LiteralUtil::CreateR2<float>(
                {{10.0, 20.0, 30.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}),
            not_padded);
}

XLA_TEST_F(ExecutionTest, WholeDimensionGather) {
  // Second dimension (size 2) is dynamic, assuming real size is 1 and padded to
  // 2:
  //
  // [[1, 2]
  //  [3, 4]
  //  [5, 6]]
  //
  // Gathering the second dimension out creates:
  //
  // [3, 4]
  //
  // Reducing this gives us 3 (4 is padded value so ignored)
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[3, 2, 1] parameter(0)
  size = s32[] constant(1)
  param_padded = s32[3, 2, 1] set-dimension-size(param, size), dimensions={1}
  index = s32[] constant(1)
  gather = s32[2,1]{1,0} gather(param_padded, index),
              offset_dims={0,1},
              collapsed_slice_dims={0},
              start_index_map={0},
              index_vector_dim=0,
              slice_sizes={1,2,1}
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(gather, init),
      dimensions={0, 1},
      to_apply=update_s32
}
)";
  // Slicing out entire dimension propagates the dimension
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{1}, {2}}, {{3}, {4}}, {{5}, {6}}});
  auto module = GetHloModule(hlo_text);
  DynamicPadder padder;
  TF_CHECK_OK(padder.Run(module.get()).status());
  Literal result = PadAndExecute(std::move(module), {&operand});

  // Only first element will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32>(3);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, TwoDimensionReduce) {
  // Test that reducing on operand=[2,2] is same as reducing on operand=[4,4]
  // and dynamic dimension = 2
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[INDICES_BOUND, INDICES_BOUND] parameter(0)
  dynamic_size = s32[] parameter(1)
  const = s32[] constant(0)
  ROOT reduce = s32[] reduce(param, const),
      dimensions={0, 1},
      to_apply=update_s32
}
)";
  const string hlo_text_not_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "2"}});
  auto module_not_padded = GetHloModule(hlo_text_not_padded);

  Literal operand = LiteralUtil::CreateR2<int32>({{1, 2}, {4, 5}});
  Literal dynamic_size = LiteralUtil::CreateR0<int32>(2);

  Literal not_padded = ExecuteAndTransfer(std::move(module_not_padded),
                                          {&operand, &dynamic_size});

  // Pad input to 4.
  const string hlo_text_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "4"}});
  auto module_padded = GetHloModule(hlo_text_padded);
  // Set up dynamic parameter binding.
  TF_CHECK_OK(module_padded->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 0}));
  TF_CHECK_OK(module_padded->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{1, {}},
      DynamicParameterBinding::DynamicDimension{0, {}, 1}));
  // Pad the rest of input with garbage data.
  Literal operand_padded = LiteralUtil::CreateR2<int32>(
      {{1, 2, 3, 4}, {4, 5, 6, 7}, {1, 2, 3, 4}, {4, 5, 6, 7}});
  DynamicPadder padder;
  TF_CHECK_OK(padder.Run(module_padded.get()).status());
  Literal padded =
      PadAndExecute(std::move(module_padded), {&operand_padded, &dynamic_size});

  EXPECT_EQ(padded, not_padded);
}

XLA_TEST_F(ExecutionTest, DynamicDimensionClamp) {
  const string hlo_text = R"(
HloModule TensorFlowTenaryV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[5] set-dimension-size(param, const), dimensions={0}
  clamp = s32[5] clamp(param_padded, param_padded, param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(clamp, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // Input has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR1<int32>({1, 2, 3, 4, 5});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // only first 3 elements will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32>(6);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicConcat) {
  // Concatting a list of {dynamic_operand, static_operand, dynamic_operand}.
  const string hlo_text = R"(
HloModule DynamicConcat

ENTRY main {
  param_0 = s32[3] parameter(0)
  param_1 = s32[3] parameter(1)
  param_2 = s32[3] parameter(2)
  size = s32[] constant(2)
  param_padded_0 = s32[<=3] set-dimension-size(param_0, size), dimensions={0}
  param_padded_2 = s32[<=3] set-dimension-size(param_2, size), dimensions={0}
  ROOT %concatenate = s32[9]
    concatenate(s32[<=3] param_padded_0, s32[<=3] param_1, s32[<=3] param_padded_2),
    dimensions={0}
}
)";

  // Input has upper bound of 3, dynamic dimension is 2. Using -1 as padding.
  Literal operand_0 =
      LiteralUtil::CreateR1<int32>({1, 2, -1});  // Dynamic operand.
  Literal operand_1 =
      LiteralUtil::CreateR1<int32>({3, 4, 5});  // Static operand.
  Literal operand_2 =
      LiteralUtil::CreateR1<int32>({6, 7, -1});  // Dynamic operand.
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module),
                                 {&operand_0, &operand_1, &operand_2}, false);
  result.SetDynamicSize(0, 7);
  Literal expected = LiteralUtil::CreateR1<int32>({1, 2, 3, 4, 5, 6, 7});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicDimensionReduce) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[5] set-dimension-size(param, const), dimensions={0}
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(param_padded, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // Input has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR1<int32>({1, 2, 3, 4, 5});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // only first 3 elements will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32>(6);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, InputMinorDimensionReshape) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[1, 2, 5, 1] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[1, 2, 5, 1] set-dimension-size(param, const), dimensions={2}
  reshaped = s32[10] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR4<int32>(
      {{{{1}, {2}, {3}, {4}, {5}}, {{2}, {4}, {6}, {7}, {8}}}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // Only the first 6 elements will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32>(18);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, SliceSingleElement) {
  // Slicing out a single element is supported.
  const string hlo_text = R"(
HloModule Slicing

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[5] set-dimension-size(param, const), dimensions={0}
  ROOT slice = s32[1]{0} slice(param_padded), slice={[0:1]}
}
)";

  // The dynamic dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR1<int32>({0, 1, 2, 3, 4});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  Literal expected = LiteralUtil::CreateR1<int32>({0});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, OutputMinorDimensionReshape) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[12] parameter(0)
  const = s32[] constant(8)
  param_padded = s32[12] set-dimension-size(param, const), dimensions={0}
  // Second dimension is dynamic.
  reshaped = s32[2, 3, 2] reshape(param_padded), inferred_dimension=1
  init = s32[] constant(0)
  ROOT reduce = s32[2, 2] reduce(reshaped, init),
      dimensions={1},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand =
      LiteralUtil::CreateR1<int32>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // After padding and reshape we have
  //
  // [[[0, 1],
  //   [2, 3]
  //   [P, P]]
  //  [[4, 5],
  //   [6, 7],
  //   [P, P]]]
  // Reducing on the second dimension gives us
  //  [0+2, 1+3]
  //  [4+6, 5+7]
  //
  Literal expected = LiteralUtil::CreateR2<int32>({{2, 4}, {10, 12}});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, OutputMinorDimensionReshapeWithUnchangedDimMajor) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[2, 6] parameter(0)
  const = s32[] constant(4)
  param_padded = s32[2, 6] set-dimension-size(param, const), dimensions={1}
  // Third dimension is dynamic.
  reshaped = s32[2, 2, 3] reshape(param_padded), inferred_dimension=2
  init = s32[] constant(0)
  ROOT reduce = s32[2, 2] reduce(reshaped, init),
      dimensions={2},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand =
      LiteralUtil::CreateR2<int32>({{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // After padding and reshape we have
  //
  // [[[0, 1, P],
  //   [2, 3, P]],
  //  [[6, 7, P],
  //   [8, 9, P]]]
  // Reducing on the third dimension gives us
  //  [0+1, 2+3]
  //  [6+7, 8+9]
  //
  Literal expected = LiteralUtil::CreateR2<int32>({{1, 5}, {13, 17}});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, OutputMinorDimensionReshapeWithUnchangedDimMinor) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[6, 2] parameter(0)
  const = s32[] constant(4)
  param_padded = s32[6, 2] set-dimension-size(param, const), dimensions={0}
  // Second dimension is dynamic.
  reshaped = s32[2, 3, 2] reshape(param_padded), inferred_dimension=1
  init = s32[] constant(0)
  ROOT reduce = s32[2, 2] reduce(reshaped, init),
      dimensions={1},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR2<int32>(
      {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // After padding and reshape we have
  //
  // [[[0, 1],
  //   [2, 3]
  //   [P, P]],
  //  [[4, 5],
  //   [6, 7],
  //   [P, P]]]
  // Reducing on the second dimension gives us
  //  [0+2, 1+3]
  //  [4+6, 5+7]
  //
  Literal expected = LiteralUtil::CreateR2<int32>({{2, 4}, {10, 12}});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicInputFeature) {
  const string hlo_text = R"(
HloModule DynamicInputFeature

ENTRY main {
  param = f32[1, 1, 5] parameter(0)
  const = s32[] constant(5)
  one = f32[] constant(1)
  kernel = f32[1,5,1]{2,1,0} broadcast(f32[] one), dimensions={}
  param_dynamic = f32[1,1,<=5] set-dimension-size(param, const), dimensions={2}
  ROOT conv = f32[1, 1, 1]{2,1,0} custom-call(f32[1, 1, <=5] param_dynamic, f32[1,5,1]{2,1,0} kernel),
                             window={size=1 pad=0_0},
                             dim_labels=b0f_0io->b0f,
                             padding_type=PADDING_VALID,
                             custom_call_target="DynamicConvolutionForward"
}
)";

  Literal operand = LiteralUtil::CreateR3<float>({{{1, 2, 3, 4, 5}}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  Literal expected = LiteralUtil::CreateR3<float>({{{15}}});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicDimensionReshapeUnchanged) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[1, 2, 5, 1] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[1, 2, 5, 1] set-dimension-size(param, const), dimensions={2}
  reshaped = s32[2, 5] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[2] reduce(reshaped, init),
      dimensions={1},
      to_apply=update_s32
}
)";

  // Test dynamic padder in unchanged dimension reshape.
  Literal operand = LiteralUtil::CreateR4<int32>(
      {{{{1}, {2}, {3}, {4}, {5}}, {{2}, {4}, {6}, {7}, {8}}}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  Literal expected = LiteralUtil::CreateR1<int32>({6, 12});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DegeneratedDimension) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[1, 2, 5, 1] parameter(0)
  size = s32[] constant(0)
// First dimension is dynamic.
  param_padded = s32[1, 2, 5, 1] set-dimension-size(param, size),
    dimensions={0}
  reshaped = s32[10] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR4<int32>(
      {{{{1}, {2}, {3}, {4}, {5}}, {{2}, {4}, {6}, {7}, {8}}}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  Literal expected = LiteralUtil::CreateR0<int32>(0);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, ReshapeSplitCombineSameTime) {
  // [<=4, 2, <=2]
  //       |
  //    Reshape
  //       |
  // [2, <=2, <=4]
  //
  // Split one input dynamic dim to multiple output dims while combining two
  // dimensions together.
  //
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[4, 2, 2] parameter(0)
  two = s32[] constant(2)
  one = s32[] constant(1)
  param_padded_partial = s32[<=4, 2, 2] set-dimension-size(param, two),
    dimensions={0}

  param_padded_dynamic = s32[<=4, 2, <=2] set-dimension-size(param_padded_partial,
                                                             one),
    dimensions={2}
  reshaped = s32[2, <=2, <=4] reshape(param_padded_dynamic),
    inferred_dimension=1
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0, 1, 2},
      to_apply=update_s32
}
)";

  // First and last dims are dynamic. Padded data are expressed as -1.
  Literal operand = LiteralUtil::CreateR3<int32>({{{0, -1}, {1, -1}},
                                                  {{2, -1}, {3, -1}},
                                                  {{-1, -1}, {-1, -1}},
                                                  {{-1, -1}, {-1, -1}}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // Reshaping (with correct reshape rewriting) produces:
  // [[[0, 1, -1, -1], [-1, -1, -1, -1]], [[2, 3, -1, -1], [-1, -1, -1, -1]]]
  //
  //  Dynamic padder auto pads -1 with 0.
  //
  // Reducing it produces 0 + 1 + 2 + 3 = 6

  Literal expected = LiteralUtil::CreateR0<int32>(6);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, WhileLoopStack) {
  // Push into a dynamic sized stack with iteration number:
  // init:
  // [[P, P],
  //  [P, P],
  //  [P, P],
  //  [P, P]]
  // First iteration i = 0:
  // [[0, 0],
  //  [P, P],
  //  [P, P],
  //  [P, P]]
  // Second iteration i = 1:
  // [[0, 0],
  //  [1, 1],
  //  [P, P],
  //  [P, P]]
  // Third iteration i = 2:
  // [[0, 0],
  //  [1, 1],
  //  [2, 2],
  //  [P, P]]

  const string hlo_text = R"(
HloModule module

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

body {
  stack = (s32[<=4,2]) parameter(0)
  stack_buffer = s32[<=4, 2] get-tuple-element(stack), index=0
  stack_size = s32[] get-dimension-size(stack_buffer), dimensions={0}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  // content of the stack is the stack index broadcasted.
  new_data = s32[1, 2] broadcast(s32[] stack_size), dimensions={}
  new_stack_buffer = s32[<=4, 2] dynamic-update-slice(stack_buffer, new_data, stack_size, zero)
  new_stack_size = s32[] add(stack_size, one)
  new_stack_buffer_dynamic = s32[<=4, 2]set-dimension-size(new_stack_buffer, new_stack_size), dimensions={0}
  ROOT new_stack = (s32[<=4,2]) tuple(new_stack_buffer_dynamic)
}

condition {
  stack = (s32[<=4,2]) parameter(0)
  stack_buffer = s32[<=4, 2] get-tuple-element(stack), index=0
  stack_size = s32[] get-dimension-size(stack_buffer), dimensions={0}
  three = s32[] constant(3)
  ROOT less-than = pred[] compare(s32[] stack_size, s32[] three), direction=LT
}

ENTRY entry {
  zero = s32[] constant(0)
  pad = s32[] constant(-1)
  stack_buffer_input = s32[4, 2] broadcast(s32[] pad), dimensions={}
  stack_buffer_input_dynamic = s32[<=4, 2] set-dimension-size(stack_buffer_input, zero), dimensions={0}
  input_tuple = (s32[<=4 ,2]) tuple(stack_buffer_input_dynamic)
  while = (s32[<=4, 2]) while(input_tuple), body=body, condition=condition
  stack_buffer = s32[<=4, 2] get-tuple-element(while), index=0
  ROOT reduce = s32[2] reduce(stack_buffer, zero),
    dimensions={0},
    to_apply=update_s32
}
)";

  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {});

  // Stack has three valid items in it:
  // [[0, 0],
  //  [1, 1],
  //  [2, 2],
  //  [P, P]]
  //
  // Reducing along major dimension gives us [3, 3]
  Literal expected = LiteralUtil::CreateR1<int32>({{3, 3}});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DoubleDynamicDimension) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[2, 3, 3] parameter(0)
  size = s32[] constant(2)
  param_padded_partial = s32[2, 3, 3] set-dimension-size(param, size),
    dimensions={1}
  param_padded = s32[2, 3, 3] set-dimension-size(param_padded_partial, size),
    dimensions={2}
  reshaped = s32[18] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR3<int32>(
      {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // Padded data looks like this (P is padding which is ignored).
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // Reshaping (with correct reshape rewriting) produces:
  // [0, 1, 3, 4, 0, 1, 3, 4, P, P, P, P, P, P, P, P, P, P]
  //
  // Reducing it produces 16

  Literal expected = LiteralUtil::CreateR0<int32>(16);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicReshapeDoubleDynamicDimensions) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

ENTRY main {
  param = s32[2, 3, 3] parameter(0)
  size = s32[] constant(2)
  param_padded_partial = s32[2, <=3, 3] set-dimension-size(param, size),
    dimensions={1}
  param_padded = s32[2, <=3, <=3] set-dimension-size(param_padded_partial, size),
    dimensions={2}
  result_size = s32[] constant(8)
  ROOT reshaped = s32[<=18] dynamic-reshape(param_padded, result_size)
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR3<int32>(
      {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand}, false);
  result.SetDynamicSize(0, 8);
  // Padded data looks like this (P is padding which is ignored).
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // Reshaping (with correct reshape rewriting) produces:
  // [0, 1, 3, 4, 0, 1, 3, 4]
  Literal expected = LiteralUtil::CreateR1<int32>({0, 1, 3, 4, 0, 1, 3, 4});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicReshapeOutputDoubleDynamicDimensions) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

ENTRY main {
  param = s32[18] parameter(0)
  eight = s32[] constant(8)
  param_dynamic = s32[<=18] set-dimension-size(param, eight), dimensions={0}
  two = s32[] constant(2)
  // every dimension has dynamic size two.
  ROOT reshaped = s32[2, <=3, <=3] dynamic-reshape(param_dynamic, two, two, two)
}
)";
  Literal operand = LiteralUtil::CreateR1<int32>(
      {0, 1, 3, 4, 0, 1, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});

  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand}, false);

  result.SetDynamicSize(1, 2);
  result.SetDynamicSize(2, 2);
  // Padded operand is:
  // [0, 1, 3, 4, 0, 1, 3, 4, P, P ....]
  //
  // Reshaping it should produce:
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  Literal expected =
      LiteralUtil::CreateR3<int32>({{{0, 1}, {3, 4}}, {{0, 1}, {3, 4}}});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, SetGetDimensionSize) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

ENTRY main {
  param = s32[3] parameter(0)
  size = s32[] constant(2)
  param_dynamic_size = s32[3] set-dimension-size(param, size),
    dimensions={0}
  ROOT gds = s32[] get-dimension-size(param_dynamic_size),
    dimensions={0}
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR1<int32>({1, 2, 3});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand});

  // Should return the size 2 instead of 3.
  Literal expected = LiteralUtil::CreateR0<int32>(2);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicSort) {
  const string hlo_text = R"(
HloModule TEST

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

%compare-greater-than (lhs: s32[], rhs: s32[]) -> pred[] {
  %lhs = s32[] parameter(0)
  %rhs = s32[] parameter(1)
  ROOT %compare = pred[] compare(s32[] %lhs, s32[] %rhs), direction=GT
}

ENTRY main {
  param = s32[4] parameter(0)
  size = s32[] constant(3)
  param_dynamic_size = s32[4] set-dimension-size(param, size),
    dimensions={0}
  ROOT sort = s32[4]{0} sort(s32[4]{0} %param_dynamic_size),
    dimensions={0}, is_stable=false, to_apply=%compare-greater-than
}
)";

  Literal operand = LiteralUtil::CreateR1<int32>({1, 4, 3, 2});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand},
                                 /*slice_dynamic_output=*/false);
  Literal expected = LiteralUtil::CreateR1<int32>({4, 3, 1, 2});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicPad) {
  const string hlo_text = R"(
HloModule TEST

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[4] parameter(0)
  size = s32[] constant(3)
  padding = s32[] constant(2)
  param_dynamic = s32[<=4] set-dimension-size(param, size),
    dimensions={0}
  // Pad head and tail with 2
  pad = s32[<=6] pad(param_dynamic, padding), padding=1_1

  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(pad, init),
    dimensions={0},
    to_apply=update_s32
}
)";

  Literal operand = LiteralUtil::CreateR1<int32>({1, 4, 3, 5});
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  // After padding head and tail with "2", the effective data will be [2, 1, 4,
  // 3, 2]

  Literal result = PadAndExecute(std::move(module), {&operand},
                                 /*slice_dynamic_output=*/false);
  Literal expected = LiteralUtil::CreateR0<int32>(12);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicPadInteriorPadding) {
  const string hlo_text = R"(
HloModule TEST

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[4] parameter(0)
  size = s32[] constant(3)
  padding = s32[] constant(2)
  param_dynamic = s32[<=4] set-dimension-size(param, size),
    dimensions={0}
  // Pad interior with constant 2.
  pad = s32[<=7] pad(param_dynamic, padding), padding=0_0_1

  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(pad, init),
    dimensions={0},
    to_apply=update_s32
}
)";

  // Only the first 3 elements are effective: 1, 4, 3
  Literal operand = LiteralUtil::CreateR1<int32>({1, 4, 3, 5});
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  // After interior padding with "2", the effective data will be
  // [1, 2, 4, 2, 3]
  Literal result = PadAndExecute(std::move(module), {&operand},
                                 /*slice_dynamic_output=*/false);
  Literal expected = LiteralUtil::CreateR0<int32>(12);

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicConditionalDimension) {
  const string hlo_text = R"(
HloModule module

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

true_branch {
  true_param = (s32[<=3,2]) parameter(0)
  param = s32[<=3, 2] get-tuple-element(true_param), index=0
  add = s32[<=3,2] add(param, param)
  ROOT true_tuple = (s32[<=3,2], s32[<=3,2]) tuple(add, add)
}

false_branch {
  false_param = (s32[<=3,2]) parameter(0)
  param = s32[<=3, 2] get-tuple-element(false_param), index=0
  add = s32[<=3,2] add(param, param)
  ROOT false_tuple = (s32[<=3,2], s32[<=3,2]) tuple(add, add)
}

ENTRY entry {
  param0 = s32[3,2] parameter(0)
  size = s32[] constant(2)
  branch = pred[] constant(false)
  param_dynamic = s32[<=3, 2] set-dimension-size(param0, size), dimensions={0}
  param_tuple = (s32[<=3 ,2]) tuple(param_dynamic)
  conditional = (s32[<=3, 2], s32[<=3, 2]) conditional(branch, param_tuple, param_tuple),
    true_computation=true_branch, false_computation=false_branch
  gte0 = s32[<=3,2] get-tuple-element(conditional), index=1
  init = s32[] constant(0)
  ROOT reduce = s32[2] reduce(gte0, init),
    dimensions={0},
    to_apply=update_s32
}
)";

  Literal operand = LiteralUtil::CreateR2<int32>({{0, 1}, {2, 3}, {4, 5}});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand},
                                 /*slice_dynamic_output=*/false);
  Literal expected = LiteralUtil::CreateR1<int32>({4, 8});

  EXPECT_EQ(result, expected);
}

XLA_TEST_F(ExecutionTest, DynamicTupleSort) {
  const string hlo_text = R"(
HloModule TEST

%compare-greater-than (lhs: s32[], rhs: s32[], lhs_2: s32[], lhs_2: s32[]) -> pred[] {
  %lhs = s32[] parameter(0)
  %rhs = s32[] parameter(1)
  %lhs_2 = s32[] parameter(2)
  %rhs_2 = s32[] parameter(3)
  ROOT %compare = pred[] compare(s32[] %lhs, s32[] %rhs), direction=GT
}

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[3] parameter(0)
  size = s32[] constant(2)
  param_dynamic_size = s32[3] set-dimension-size(param, size),
    dimensions={0}
  sort = (s32[3]{0}, s32[3]{0}) sort(s32[3]{0} %param_dynamic_size,
                                     s32[3]{0} %param_dynamic_size),
    dimensions={0}, is_stable=true, to_apply=%compare-greater-than
  ROOT get-tuple-element = s32[3]{0} get-tuple-element((s32[3]{0}, s32[3]{0}) %sort),
    index=0
}
)";

  Literal operand = LiteralUtil::CreateR1<int32>({0, 4, 2});
  auto module = GetHloModule(hlo_text);

  Literal result = PadAndExecute(std::move(module), {&operand},
                                 /*slice_dynamic_output=*/false);
  Literal expected = LiteralUtil::CreateR1<int32>({4, 0, 2});

  EXPECT_EQ(result, expected);
}

namespace op = xla::testing::opcode_matchers;

class HloDimensionSizeLegalizerTest : public HloTestBase {
 protected:
  HloDimensionSizeLegalizerTest() {}
};

TEST_F(HloDimensionSizeLegalizerTest, Ok) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule _
ENTRY gds {
  p = s32[3,4] parameter(0)
  size0 = s32[] get-dimension-size(p), dimensions={0}
  size1 = s32[] get-dimension-size(p), dimensions={1}
  ROOT mul = s32[] multiply(size0, size1)
})")
                    .ValueOrDie();
  DynamicPadder pass;
  EXPECT_TRUE(pass.Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Constant(), op::Constant()));
}

TEST_F(HloDimensionSizeLegalizerTest, GetSetSetDimensionSizeRewriter) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule _
ENTRY gds {
  p = s32[3,4] parameter(0)
  size0 = s32[] get-dimension-size(p), dimensions={0}
  p_copy = s32[3,4] copy(p)
  p_copy_dynamic = s32[<=3, 4] set-dimension-size(p_copy, size0), dimensions={0}
  size1 = s32[] get-dimension-size(p_copy_dynamic), dimensions={0}
  ROOT mul = s32[] multiply(size0, size1)
})")
                    .ValueOrDie();
  DynamicPadder pass;
  EXPECT_TRUE(pass.Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Constant(), op::Constant()));
}

TEST_F(HloDimensionSizeLegalizerTest, IllegalType) {
  auto module = ParseAndReturnUnverifiedModule(R"(
HloModule _
ENTRY gds {
  p = s32[3]{0} parameter(0)
  ROOT gds = s64[] get-dimension-size(p), dimensions={0}
})")
                    .ValueOrDie();
  DynamicPadder pass;
  EXPECT_FALSE(pass.Run(module.get()).ok());
}

TEST_F(HloDimensionSizeLegalizerTest, IllegalDimension) {
  auto module = ParseAndReturnUnverifiedModule(R"(
HloModule _
ENTRY gds {
  p = f32[2,5] parameter(0)
  ROOT gds = s32[] get-dimension-size(p), dimensions={2}
})")
                    .ValueOrDie();
  DynamicPadder pass;
  EXPECT_FALSE(pass.Run(module.get()).ok());
}

}  // namespace
}  // namespace xla
