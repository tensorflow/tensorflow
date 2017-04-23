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

#include "tensorflow/compiler/xla/service/user_computation.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using UserComputationTest = ::testing::Test;

TEST_F(UserComputationTest, SimpleComputation) {
  const Shape kScalarShape = ShapeUtil::MakeShape(F32, {});
  const Shape kVectorShape = ShapeUtil::MakeShape(F32, {2});

  // Build a simple three operation computatation:
  //
  //   %constant = Constant({123, 42})
  //   %param = Param(0)
  //   %outfeed = Outfeed(%constant)
  //
  // Build the computation at two different versions and check invariants.
  ComputationHandle handle;
  handle.set_handle(123);
  UserComputation computation("TheComputation", handle);

  ConstantRequest constant_request;
  *constant_request.mutable_literal() =
      *LiteralUtil::CreateR1<float>({123.0f, 42.0f});
  TF_ASSIGN_OR_ASSERT_OK(ComputationDataHandle constant_handle,
                         computation.AddConstantInstruction(constant_request));

  ParameterRequest param_request;
  *param_request.mutable_shape() = kScalarShape;
  param_request.set_parameter(0);
  param_request.set_name("param0");
  TF_ASSIGN_OR_ASSERT_OK(ComputationDataHandle param_handle,
                         computation.AddParameterInstruction(param_request));

  OutfeedRequest outfeed_request;
  *outfeed_request.mutable_operand() = constant_handle;
  outfeed_request.set_outfeed_config("abc");
  TF_ASSERT_OK(computation.AddOutfeedInstruction(outfeed_request));

  auto hlo_resolver = [](const VersionedComputationHandle& handle) {
    return nullptr;
  };
  {
    // Test the computation at the latest version. In this case, the most
    // recently added operation is an outfeed. However, the outfeed is not the
    // root because outfeeds cannot be the root of a computation.
    VersionedComputationHandle latest_version =
        computation.GetVersionedHandle();

    // Program shape should have a single scalar parameter and scalar
    // result. The outfeed instruction should not affect the program shape.
    TF_ASSIGN_OR_ASSERT_OK(
        std::shared_ptr<const ProgramShape> program_shape,
        computation.ComputeProgramShape(latest_version.version));
    ASSERT_EQ(1, program_shape->parameters_size());
    EXPECT_TRUE(
        ShapeUtil::Compatible(kScalarShape, program_shape->parameters(0)));
    EXPECT_TRUE(ShapeUtil::Compatible(kScalarShape, program_shape->result()));

    // Build the HLO computation.
    TF_ASSIGN_OR_ASSERT_OK(
        std::unique_ptr<HloComputation> hlo_computation,
        computation.BuildHloComputation(latest_version.version, hlo_resolver));
    // There should be one HloInstruction per UserComputation operation.
    EXPECT_EQ(3, hlo_computation->instruction_count());
    // The root of the instruction should be the parameter instruction (not the
    // outfeed).
    EXPECT_THAT(hlo_computation->root_instruction(), op::Parameter());
  }

  {
    // Test the computation at the version right after the parameter instruction
    // is added.
    VersionedComputationHandle version_at_param =
        computation.GetVersionedHandleAtOperation(param_handle);

    // Program shape should have a single scalar parameter, and scalar result.
    TF_ASSIGN_OR_ASSERT_OK(
        std::shared_ptr<const ProgramShape> program_shape,
        computation.ComputeProgramShape(version_at_param.version));
    ASSERT_EQ(1, program_shape->parameters_size());
    EXPECT_TRUE(
        ShapeUtil::Compatible(kScalarShape, program_shape->parameters(0)));
    EXPECT_TRUE(ShapeUtil::Compatible(kScalarShape, program_shape->result()));

    // There should be two instructions, one for the constant and one for the
    // parameter. The outfeed instruction should not be included.
    TF_ASSIGN_OR_ASSERT_OK(std::unique_ptr<HloComputation> hlo_computation,
                           computation.BuildHloComputation(
                               version_at_param.version, hlo_resolver));
    EXPECT_EQ(2, hlo_computation->instruction_count());
    EXPECT_THAT(hlo_computation->root_instruction(), op::Parameter());
  }
  {
    // Test the computation at the latest version, but lowered with
    // include_unreachable_instructions set to false.
    VersionedComputationHandle latest_version =
        computation.GetVersionedHandle();

    // Build the HLO computation.
    TF_ASSIGN_OR_ASSERT_OK(std::unique_ptr<HloComputation> hlo_computation,
                           computation.BuildHloComputation(
                               latest_version.version, hlo_resolver,
                               /*include_unreachable_instructions=*/false));
    // There is only one reachable instruction, the parameter.
    EXPECT_EQ(1, hlo_computation->instruction_count());
    // The root of the instruction should be the parameter instruction (not the
    // outfeed).
    EXPECT_THAT(hlo_computation->root_instruction(), op::Parameter());
  }
}

}  // namespace
}  // namespace xla
