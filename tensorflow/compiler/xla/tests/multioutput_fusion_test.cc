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

#include <math.h>
#include <algorithm>
#include <memory>
#include <new>
#include <utility>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ::tensorflow::gtl::ArraySlice;

class MultiOutputFusionTest : public HloTestBase {
 protected:
  MultiOutputFusionTest() { error_spec_ = ErrorSpec{0.0001, 1e-2}; }

  void RunTest2D(bool manual_fusion, int64 size) {
    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = CreateNewModule();

    const Shape elem_shape0 = ShapeUtil::MakeShape(F32, {});
    const Shape elem_shape2 = ShapeUtil::MakeShape(F32, {size, size});

    auto const0 = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<float>(8.0f)));
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, elem_shape0, "0"));

    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape0, HloOpcode::kAdd, param0, const0));

    HloInstruction* broadcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(elem_shape2, add1, {}));

    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, elem_shape2, "1"));

    HloInstruction* add2 = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape2, HloOpcode::kAdd, broadcast, param1));
    HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape2, HloOpcode::kSubtract, param1, broadcast));
    HloInstruction* dot = builder.AddInstruction(
        HloInstruction::CreateBinary(elem_shape2, HloOpcode::kDot, sub, add2));
    auto computation = hlo_module->AddEntryComputation(builder.Build(dot));

    if (manual_fusion) {
      auto tuple = computation->AddInstruction(HloInstruction::CreateTuple(
          ArraySlice<HloInstruction*>({sub, add2}, 0, 2)));
      auto gte0 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape2, tuple, 0));
      auto gte1 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape2, tuple, 1));
      TF_CHECK_OK(dot->ReplaceOperandWith(0, gte0));
      TF_CHECK_OK(dot->ReplaceOperandWith(1, gte1));

      CHECK_NE(
          computation->CreateFusionInstruction(
              {tuple, sub, add2, broadcast}, HloInstruction::FusionKind::kLoop),
          nullptr);
    }

    Literal input;
    input.PopulateWithValue<float>(2.5f, {size, size});
    auto p1 = TransferToDevice(input);
    auto p0 = TransferToDevice(*Literal::CreateR0<float>(-9.0f));

    Literal expect;
    expect.PopulateWithValue<float>(size * 1.5f * 3.5f, {size, size});
    auto actual = ExecuteAndTransfer(std::move(hlo_module), {p0, p1});
    LiteralTestUtil::ExpectNear(expect, *actual, error_spec_);
  }

  void RunTest1D(bool manual_fusion, int size) {
    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = CreateNewModule();

    const Shape elem_shape_F32 = ShapeUtil::MakeShape(F32, {size});
    const Shape elem_shape_U8 = ShapeUtil::MakeShape(F64, {size});
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, elem_shape_F32, "0"));
    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, elem_shape_U8, "1"));

    HloInstruction* param0_U8 = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_U8, param0));
    HloInstruction* param1_F32 = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_F32, param1));
    HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape_F32, HloOpcode::kAdd, param0, param1_F32));
    HloInstruction* sub_U8 =
        builder.AddInstruction(HloInstruction::CreateBinary(
            elem_shape_U8, HloOpcode::kSubtract, param0_U8, param1));
    HloInstruction* sub = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_F32, sub_U8));

    HloInstruction* reshape =
        builder.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(F32, {size, 1}), add));
    HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {1}), HloOpcode::kDot, sub, reshape));
    auto computation = hlo_module->AddEntryComputation(builder.Build(dot));

    if (manual_fusion) {
      auto tuple = computation->AddInstruction(HloInstruction::CreateTuple(
          ArraySlice<HloInstruction*>({sub_U8, add}, 0, 2)));

      auto gte0 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape_U8, tuple, 0));
      auto gte1 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape_F32, tuple, 1));
      TF_CHECK_OK(sub->ReplaceOperandWith(0, gte0));
      TF_CHECK_OK(reshape->ReplaceOperandWith(0, gte1));

      CHECK_NE(computation->CreateFusionInstruction(
                   {tuple, sub_U8, add, param0_U8, param1_F32},
                   HloInstruction::FusionKind::kLoop),
               nullptr);
    }

    Literal input0, input1;
    input0.PopulateWithValue<float>(2.5f, {size});
    input1.PopulateWithValue<double>(1, {size});
    auto p0 = TransferToDevice(input0);
    auto p1 = TransferToDevice(input1);

    Literal expect = *Literal::CreateR1<float>({size * 1.5f * 3.5f});
    auto actual = ExecuteAndTransfer(std::move(hlo_module), {p0, p1});
    LiteralTestUtil::ExpectNear(expect, *actual, error_spec_);
  }
};

XLA_TEST_F(MultiOutputFusionTest, 2DNofusion) { RunTest2D(false, 5); }
XLA_TEST_F(MultiOutputFusionTest, 2DFusion) { RunTest2D(true, 5); }
XLA_TEST_F(MultiOutputFusionTest, 2DFusionSize129) { RunTest2D(true, 129); }
XLA_TEST_F(MultiOutputFusionTest, DiffentTypesNoFusion) { RunTest1D(false, 8); }
XLA_TEST_F(MultiOutputFusionTest, DiffentTypesFusion) { RunTest1D(true, 8); }

}  // namespace
}  // namespace xla
