/* Copyright 2020 The OpenXLA Authors.

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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class GpuDynamicShapeTest : public GpuPjRtCodegenTest {};

TEST_F(GpuDynamicShapeTest, DynamicShapeR2) {
  HloComputation::Builder builder(TestName());

  xla::Shape dyn_input_shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 4});
  dyn_input_shape.set_dynamic_dimension(0, true);
  HloInstruction* param_x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dyn_input_shape, "x"));

  builder.AddInstruction(HloInstruction::CreateUnary(
      dyn_input_shape, HloOpcode::kNegate, param_x));
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  EXPECT_OK(CompileAndVerifyIr(std::move(hlo_module),
                               R"(
; CHECK-LABEL: define {{.*}}_kernel void @x_padded(
; CHECK: %[[SRC_INT:.*]] = ptrtoint ptr %{{.*}} to i64
; CHECK: %[[META0:.*]] = add i64 %[[SRC_INT]], 32
; CHECK: %[[PTR0:.*]] = inttoptr i64 %[[META0]] to ptr
; CHECK: %[[DIM0:.*]] = load i32, ptr %[[PTR0]], align 4
; CHECK: %[[META1:.*]] = add i64 %[[SRC_INT]], 36
; CHECK: %[[PTR1:.*]] = inttoptr i64 %[[META1]] to ptr
; CHECK: %[[DIM1:.*]] = load i32, ptr %[[PTR1]], align 4
; CHECK: store i32 %[[DIM0]], ptr
; CHECK: store i32 %[[DIM1]], ptr
; CHECK: %[[TOTAL:.*]] = mul i32 %[[DIM0]], %[[DIM1]]
; CHECK: %[[LIN:.*]] = add i32
; CHECK: icmp ult i32 %[[LIN]], %[[TOTAL]]
; CHECK: load float, ptr
; CHECK: store float
      )",
                               /*match_optimized_ir=*/false));
}

TEST_F(GpuDynamicShapeTest, ExecutePadToStaticUnalignedF16) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticUnalignedF16
    ENTRY main {
      param = f16[<=5]{0} parameter(0)
      ROOT pad_to_static = (f16[5]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(F16, {5}, /*dynamic_dimensions=*/{true});
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<half>(
      [](auto indices) { return half(indices[0] + 1.0f); }));
  input.SetDynamicSize(0, 3);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 3);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(outputs[0].Get<half>({i}), half(i + 1.0f));
  }
}

TEST_F(GpuDynamicShapeTest, ExecutePadToStaticMultiDimNonDefaultLayout) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticMultiDim
    ENTRY main {
      param = s32[2,<=3,2]{2,0,1} parameter(0)
      ROOT pad_to_static = (s32[2,3,2]{2,0,1}, s32[], s32[], s32[])
        custom-call(param), custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShapeWithDenseLayout(S32, {2, 3, 2}, {2, 0, 1});
  dyn_shape.set_dynamic_dimension(1, true);
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<int32_t>([](auto indices) -> int32_t {
    if (indices[1] >= 2) {
      return -1;
    }
    return static_cast<int32_t>(indices[0] * 100 + indices[1] * 10 +
                                indices[2] + 1);
  }));
  input.SetDynamicSize(1, 2);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 4);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 2);
  EXPECT_EQ(outputs[2].Get<int32_t>({}), 2);
  EXPECT_EQ(outputs[3].Get<int32_t>({}), 2);
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      for (int64_t k = 0; k < 2; ++k) {
        EXPECT_EQ(outputs[0].Get<int32_t>({i, j, k}), i * 100 + j * 10 + k + 1);
      }
    }
  }
}

TEST_F(GpuDynamicShapeTest, ExecuteDynamicEntryParameterViaDynamicPadder) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule DynamicEntryParamReduce
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    ENTRY main {
      param = s32[<=3,<=4]{1,0} parameter(0)
      zero = s32[] constant(0)
      ROOT sum = s32[] reduce(param, zero), dimensions={0,1}, to_apply=add
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(S32, {3, 4}, /*dynamic_dimensions=*/{true, true});
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<int32_t>([](auto indices) -> int32_t {
    if (indices[0] >= 2 || indices[1] >= 3) {
      return 1000;
    }
    return static_cast<int32_t>(indices[0] * 3 + indices[1] + 1);
  }));
  input.SetDynamicSize(0, 2);
  input.SetDynamicSize(1, 3);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input}));
  EXPECT_EQ(result, LiteralUtil::CreateR0<int32_t>(21));
}

TEST_F(GpuDynamicShapeTest,
       ExecuteDynamicEntryParameterUnalignedF16ViaDynamicPadder) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule DynamicEntryParamUnalignedF16Reduce
    add {
      lhs = f16[] parameter(0)
      rhs = f16[] parameter(1)
      ROOT sum = f16[] add(lhs, rhs)
    }
    ENTRY main {
      param = f16[<=5]{0} parameter(0)
      zero = f16[] constant(0)
      ROOT sum = f16[] reduce(param, zero), dimensions={0}, to_apply=add
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(F16, {5}, /*dynamic_dimensions=*/{true});
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<half>([](auto indices) -> half {
    if (indices[0] >= 3) {
      return half(100.0f);
    }
    return half(indices[0] + 1.0f);
  }));
  input.SetDynamicSize(0, 3);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input}));
  EXPECT_EQ(result, LiteralUtil::CreateR0<half>(half(6.0f)));
}

TEST_F(GpuDynamicShapeTest, ExecutePadToStaticZeroDynamicSize) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticZeroDynamicSize
    ENTRY main {
      param = s32[<=4]{0} parameter(0)
      ROOT pad_to_static = (s32[4]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(S32, {4}, /*dynamic_dimensions=*/{true});
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<int32_t>([](auto) { return 42; }));
  input.SetDynamicSize(0, 0);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 0);
}

}  // namespace
}  // namespace xla::gpu
