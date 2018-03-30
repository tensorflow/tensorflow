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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace {
void R0F32Add2(float* out, float** in) {
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float*));
  *out = **in + 2.0f;
}

void R2F32ReduceSum(float* out, float** in) {
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float) * 4);
  float* array = in[0];
  *out = array[0] + array[1] + array[2] + array[3];
}

void Add1ToValues(float* out, float** in) {
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float) * 4);
  float* array = in[0];
  out[0] = array[0] + 1;
  out[1] = array[1] + 1;
  out[2] = array[2] + 1;
  out[3] = array[3] + 1;
}
}  // namespace

REGISTER_CUSTOM_CALL_TARGET(R0F32Add2);
REGISTER_CUSTOM_CALL_TARGET(R2F32ReduceSum);
REGISTER_CUSTOM_CALL_TARGET(Add1ToValues);

namespace xla {
namespace {

class CustomCallTest : public HloTestBase {
 protected:
  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r2f32_ = ShapeUtil::MakeShape(F32, {2, 2});
};

XLA_TEST_F(CustomCallTest, DISABLED_ON_GPU(CustomCallR0F32Add2)) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(r0f32_, {constant}, "R0F32Add2"));

  module->AddEntryComputation(builder.Build());

  std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR0Near<float>(44.0f, *result, error_spec_);
}

XLA_TEST_F(CustomCallTest, DISABLED_ON_GPU(CustomCallR2F32Reduce)) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR2FromArray2D(array)));
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(r0f32_, {constant}, "R2F32ReduceSum"));

  module->AddEntryComputation(builder.Build());

  std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR0Near<float>(10.0f, *result, error_spec_);
}

XLA_TEST_F(CustomCallTest,
           DISABLED_ON_GPU(CustomCall_UsedInOtherComputations)) {
  auto module = CreateNewModule();
  auto b = HloComputation::Builder(TestName());

  auto input = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR2FromArray2D(
          Array2D<float>{{1.0f, 2.0f}, {3.0f, 4.0f}})));
  auto incremented = b.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 2, 2}), {input}, "Add1ToValues"));
  auto incremented_again = b.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 2, 2}), {incremented}, "Add1ToValues"));

  // Concatenate the values along first dim.
  b.AddInstruction(
      HloInstruction::CreateConcatenate(ShapeUtil::MakeShape(F32, {2, 2, 2}),
                                        {incremented, incremented_again}, 0));

  module->AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = ExecuteAndTransfer(std::move(module), {});
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      Array3D<float>{{{2, 3}, {4, 5}}, {{3, 4}, {5, 6}}}, *result);
}

class CustomCallClientAPITest : public ClientLibraryTestBase {};

// When using the client API, CustomCall targets can't begin with '$' -- these
// are reserved for internal use.
XLA_TEST_F(CustomCallClientAPITest, IllegalCustomCallTarget) {
  ComputationBuilder builder(client_, TestName());
  auto call = builder.CustomCall("$illegal", /*operands=*/{},
                                 ShapeUtil::MakeShape(F32, {1}));

  StatusOr<std::unique_ptr<GlobalData>> result =
      Execute(&builder, /*arguments=*/{});
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace xla
