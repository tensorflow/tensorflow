/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/pad_to_static.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class PadToStaticTest : public GpuPjRtCodegenTest {
 protected:
  PadToStaticTest() : mlir_context_(CreateMlirContext()) {
    mlir_context_->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_->loadAllAvailableDialects();
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

TEST_F(PadToStaticTest, CreatePadToStaticMlirModule) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticMlir
    ENTRY main {
      param = s32[2,<=3,2]{2,0,1} parameter(0)
      ROOT pts = (s32[2,3,2]{2,0,1}, s32[], s32[], s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(12, 1, 1));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreatePadToStaticMlirModule(
                           *mlir_context_, instr,
                           /*buffer_assignment=*/nullptr, "pts", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: func.func @pts(
    CHECK-SAME: xla.backend_kind = #xla.backend_kind<gpu>
    CHECK:   %[[D0:.*]] = xla.get_dynamic_dim_size %arg0 0
    CHECK:   %[[D1:.*]] = xla.get_dynamic_dim_size %arg0 1
    CHECK:   %[[D2:.*]] = xla.get_dynamic_dim_size %arg0 2
    CHECK:   gpu.thread_id x {xla.range = [0 : index, 11 : index]}
    CHECK:   gpu.block_id x {xla.range = [0 : index, 0 : index]}
    CHECK:   scf.if
    CHECK:     tensor.insert %[[D0]] into %arg2[]
    CHECK:     tensor.insert %[[D1]] into %arg3[]
    CHECK:     tensor.insert %[[D2]] into %arg4[]
    CHECK:   scf.if
    CHECK:     arith.remui
    CHECK:     arith.divui
    CHECK:     arith.remui
    CHECK:     arith.divui
    CHECK:     %[[VAL:.*]] = xla.linear_extract %arg0[%{{.*}}]
    CHECK:     tensor.insert %[[VAL]] into %arg1[
  )"),
              IsOkAndHolds(true));
}

TEST_F(PadToStaticTest, CreatePadToStaticMlirModuleWithLargeGrid) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticLargeGrid
    ENTRY main {
      param = s32[<=4]{0} parameter(0)
      ROOT pts = (s32[4]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(2, 3, 1), se::ThreadDim(4, 1, 1));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreatePadToStaticMlirModule(
                           *mlir_context_, instr,
                           /*buffer_assignment=*/nullptr, "pts", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: gpu.thread_id x {xla.range = [0 : index, 3 : index]}
    CHECK: gpu.block_id x {xla.range = [0 : index, 1 : index]}
    CHECK: gpu.block_id y {xla.range = [0 : index, 2 : index]}
    CHECK: arith.andi
    CHECK: scf.if
    CHECK:   tensor.insert %{{.*}} into %arg2[]
  )"),
              IsOkAndHolds(true));
}

TEST_F(PadToStaticTest, CreatePadToStaticMlirModuleZeroElements) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticZeroElems
    ENTRY main {
      param = s32[0,<=4]{1,0} parameter(0)
      ROOT pts = (s32[0,4]{1,0}, s32[], s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreatePadToStaticMlirModule(
                           *mlir_context_, instr,
                           /*buffer_assignment=*/nullptr, "pts", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: xla.get_dynamic_dim_size %arg0 0
    CHECK: xla.get_dynamic_dim_size %arg0 1
    CHECK: scf.if
    CHECK:   tensor.insert
    CHECK:   tensor.insert
    CHECK-NOT: xla.linear_extract
  )"),
              IsOkAndHolds(true));
}

TEST_F(PadToStaticTest, CreatePadToStaticMlirModuleRank0) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticRank0
    ENTRY main {
      param = s32[] parameter(0)
      ROOT pts = (s32[]) custom-call(param), custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreatePadToStaticMlirModule(
                           *mlir_context_, instr,
                           /*buffer_assignment=*/nullptr, "pts", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: func.func @pts(%arg0: tensor<i32>
    CHECK:   %[[VAL:.*]] = tensor.extract %arg0[] : tensor<i32>
    CHECK:   %[[RET:.*]] = tensor.insert %[[VAL]] into %arg1[] : tensor<i32>
    CHECK:   return %[[RET]] : tensor<i32>
  )"),
              IsOkAndHolds(true));
}

TEST_F(PadToStaticTest, RejectsNonPadToStatic) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule NotPadToStatic
    ENTRY main {
      param = s32[4]{0} parameter(0)
      ROOT cc = s32[4]{0} custom-call(param), custom_call_target="OtherTarget"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims;
  EXPECT_THAT(CreatePadToStaticMlirModule(*mlir_context_, instr,
                                          /*buffer_assignment=*/nullptr, "cc",
                                          launch_dims),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected PadToStatic custom call")));
}

TEST_F(PadToStaticTest, RejectsMismatchedTupleElementShape) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticBadTuple
    ENTRY main {
      param = s32[<=4]{0} parameter(0)
      ROOT pts = (f32[4]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(4, 1, 1));
  EXPECT_THAT(CreatePadToStaticMlirModule(*mlir_context_, instr,
                                          /*buffer_assignment=*/nullptr, "pts",
                                          launch_dims),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected tuple element 0 to match input")));
}

TEST_F(PadToStaticTest, RejectsInvalidLaunchDimensions) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticBadLaunch
    ENTRY main {
      param = s32[<=4]{0} parameter(0)
      ROOT pts = (s32[4]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  EXPECT_THAT(
      CreatePadToStaticMlirModule(
          *mlir_context_, instr, /*buffer_assignment=*/nullptr, "pts",
          LaunchDimensions(se::BlockDim(1, 1, 2), se::ThreadDim(4, 1, 1))),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("block_counts().z")));
  EXPECT_THAT(
      CreatePadToStaticMlirModule(
          *mlir_context_, instr, /*buffer_assignment=*/nullptr, "pts",
          LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(2, 1, 1))),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("launch_bound()")));
}

TEST_F(PadToStaticTest, RejectsUnsupportedSubByteS1) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticS1
    ENTRY main {
      param = s1[<=4,<=6]{1,0:E(1)} parameter(0)
      ROOT pts = (s1[4,6]{1,0:E(1)}, s32[], s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(24, 1, 1));
  EXPECT_THAT(CreatePadToStaticMlirModule(*mlir_context_, instr,
                                          /*buffer_assignment=*/nullptr, "pts",
                                          launch_dims),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("does not support sub-byte type s1")));
}

TEST_F(PadToStaticTest, VerifySubByteS4LlvmIr) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticS4Ir, is_scheduled=true
    ENTRY main {
      param = s4[<=5]{0:E(4)} parameter(0)
      ROOT pts = (s4[5]{0:E(4)}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  EXPECT_OK(CompileAndVerifyIr(kHlo,
                               R"(
    CHECK-LABEL: @pts(
    CHECK: ptrtoint ptr
    CHECK: add i64 %{{.*}}, 3
    CHECK: load i32, ptr %{{.*}}, align 1
    CHECK: store i32
    CHECK: cmpxchg ptr
  )",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/false));
}

TEST_F(PadToStaticTest, ExecuteDynamicEntryParameterViaDynamicPadder) {
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

TEST_F(PadToStaticTest, ExecutePadToStaticMultiDimNonDefaultLayout) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticMultiDim
    ENTRY main {
      param = s32[2,<=3,2]{2,0,1} parameter(0)
      ROOT pts = (s32[2,3,2]{2,0,1}, s32[], s32[], s32[]) custom-call(param),
        custom_call_target="PadToStatic"
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

TEST_F(PadToStaticTest, ExecuteSliceToDynamicAndPadToStaticUnalignedF16) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicAndPadToStaticF16
    ENTRY main {
      param = f16[5]{0} parameter(0)
      dyn_size = s32[] parameter(1)
      dyn = f16[<=5]{0} custom-call(param, dyn_size),
        custom_call_target="SliceToDynamic"
      ROOT pts = (f16[5]{0}, s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Literal input = LiteralUtil::CreateR1<half>(
      {half(1.0f), half(2.0f), half(3.0f), half(4.0f), half(5.0f)});
  Literal dyn_size = LiteralUtil::CreateR0<int32_t>(3);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&input, &dyn_size},
                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 3);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(outputs[0].Get<half>({i}), half(i + 1.0f));
  }
}

TEST_F(PadToStaticTest, ExecutePadToStaticZeroDynamicSize) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticZeroDynamicSize
    ENTRY main {
      param = s32[<=4]{0} parameter(0)
      ROOT pts = (s32[4]{0}, s32[]) custom-call(param),
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

TEST_F(PadToStaticTest, ExecutePadToStaticMultipleBlocks) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticMultipleBlocks
    ENTRY main {
      param = s32[<=300]{0} parameter(0)
      ROOT pts = (s32[300]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(S32, {300}, /*dynamic_dimensions=*/{true});
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<int32_t>([](auto indices) {
    return indices[0] >= 257 ? -1 : static_cast<int32_t>(indices[0] + 1);
  }));
  input.SetDynamicSize(0, 257);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 257);
  for (int64_t i = 0; i < 257; ++i) {
    EXPECT_EQ(outputs[0].Get<int32_t>({i}), i + 1);
  }
}

TEST_F(PadToStaticTest, ExecutePadToStaticSubByteS4) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticSubByteS4
    ENTRY main {
      param = s4[<=8]{0:E(4)} parameter(0)
      ROOT pts = (s4[8]{0:E(4)}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(S4, {8}, /*dynamic_dimensions=*/{true});
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<s4>([](auto indices) {
    return s4(static_cast<int8_t>(indices[0] >= 5 ? -1 : indices[0] - 2));
  }));
  input.SetDynamicSize(0, 5);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 5);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(outputs[0].Get<s4>({i}), s4(static_cast<int8_t>(i - 2)));
  }
}

TEST_F(PadToStaticTest, ExecuteDynamicSubByteEntryParameterViaDynamicPadder) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule DynamicSubByteEntryParamReduce
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    ENTRY main {
      param = s4[<=8]{0:E(4)} parameter(0)
      conv = s32[<=8]{0} convert(param)
      zero = s32[] constant(0)
      ROOT sum = s32[] reduce(conv, zero), dimensions={0}, to_apply=add
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(S4, {8}, /*dynamic_dimensions=*/{true});
  Literal input(dyn_shape);
  ASSERT_OK(input.Populate<s4>([](auto indices) {
    return s4(static_cast<int8_t>(indices[0] >= 5 ? 7 : indices[0] - 2));
  }));
  input.SetDynamicSize(0, 5);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input}));
  // (-2) + (-1) + 0 + 1 + 2 = 0.
  EXPECT_EQ(result, LiteralUtil::CreateR0<int32_t>(0));
}

TEST_F(PadToStaticTest, ExecutePadToStaticZeroElementArray) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticZeroElemsExec
    ENTRY main {
      param = s32[0,<=4]{1,0} parameter(0)
      ROOT pts = (s32[0,4]{1,0}, s32[], s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape dyn_shape =
      ShapeUtil::MakeShape(S32, {0, 4}, /*dynamic_dimensions=*/{false, true});
  Literal input(dyn_shape);
  input.SetDynamicSize(1, 3);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 0);
  EXPECT_EQ(outputs[2].Get<int32_t>({}), 3);
}

TEST_F(PadToStaticTest, ExecutePadToStaticRank0) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule PadToStaticRank0Exec
    ENTRY main {
      param = s32[] parameter(0)
      ROOT pts = (s32[]) custom-call(param), custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Literal input = LiteralUtil::CreateR0<int32_t>(42);
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].Get<int32_t>({}), 42);
}

}  // namespace
}  // namespace xla::gpu
