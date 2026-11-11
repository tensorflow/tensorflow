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

#include "xla/backends/gpu/codegen/slice_to_dynamic.h"

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

class SliceToDynamicTest : public GpuPjRtCodegenTest {
 protected:
  SliceToDynamicTest() : mlir_context_(CreateMlirContext()) {
    mlir_context_->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_->loadAllAvailableDialects();
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

TEST_F(SliceToDynamicTest, CreateSliceToDynamicMlirModule) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicMlir
    ENTRY main {
      data = s32[2,3,2]{2,0,1} parameter(0)
      d0 = s32[] parameter(1)
      d1 = s32[] parameter(2)
      d2 = s32[] parameter(3)
      ROOT res = s32[2,<=3,2]{2,0,1} custom-call(data, d0, d1, d2),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(12, 1, 1));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreateSliceToDynamicMlirModule(
                           *mlir_context_, instr, /*buffer_assignment=*/nullptr,
                           "res", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: func.func @res(
    CHECK-SAME: xla.backend_kind = #xla.backend_kind<gpu>
    CHECK:   %[[D0:.*]] = tensor.extract %arg1[] : tensor<i32>
    CHECK:   %[[D1:.*]] = tensor.extract %arg2[] : tensor<i32>
    CHECK:   %[[D2:.*]] = tensor.extract %arg3[] : tensor<i32>
    CHECK:   gpu.thread_id x {xla.range = [0 : index, 11 : index]}
    CHECK:   gpu.block_id x {xla.range = [0 : index, 0 : index]}
    CHECK:   scf.if
    CHECK:     xla.set_dynamic_dim_size %[[D0]] into %arg4[0]
    CHECK:     xla.set_dynamic_dim_size %[[D1]] into %{{.*}}[1]
    CHECK:     xla.set_dynamic_dim_size %[[D2]] into %{{.*}}[2]
    CHECK:   scf.if
    CHECK:     arith.remui
    CHECK:     arith.divui
    CHECK:     arith.remui
    CHECK:     arith.divui
    CHECK:     %[[VAL:.*]] = tensor.extract %arg0[%{{.*}}, %{{.*}}, %{{.*}}]
    CHECK:     xla.linear_insert %[[VAL]] into %{{.*}}[%{{.*}}]
  )"),
              IsOkAndHolds(true));
}

TEST_F(SliceToDynamicTest, CreateSliceToDynamicMlirModuleWithLargeGrid) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicLargeGrid
    ENTRY main {
      data = s32[4]{0} parameter(0)
      d0 = s32[] parameter(1)
      ROOT res = s32[<=4]{0} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(2, 3, 1), se::ThreadDim(4, 1, 1));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreateSliceToDynamicMlirModule(
                           *mlir_context_, instr, /*buffer_assignment=*/nullptr,
                           "res", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: gpu.thread_id x {xla.range = [0 : index, 3 : index]}
    CHECK: gpu.block_id x {xla.range = [0 : index, 1 : index]}
    CHECK: gpu.block_id y {xla.range = [0 : index, 2 : index]}
    CHECK: arith.andi
    CHECK: scf.if
    CHECK:   xla.set_dynamic_dim_size %{{.*}} into %arg2[0]
  )"),
              IsOkAndHolds(true));
}

TEST_F(SliceToDynamicTest, CreateSliceToDynamicMlirModuleZeroElements) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicZeroElems
    ENTRY main {
      data = s32[0,4]{1,0} parameter(0)
      d0 = s32[] parameter(1)
      d1 = s32[] parameter(2)
      ROOT res = s32[0,<=4]{1,0} custom-call(data, d0, d1),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreateSliceToDynamicMlirModule(
                           *mlir_context_, instr, /*buffer_assignment=*/nullptr,
                           "res", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: scf.if
    CHECK:   xla.set_dynamic_dim_size %{{.*}} into %arg3[0]
    CHECK:   xla.set_dynamic_dim_size %{{.*}} into %{{.*}}[1]
    CHECK-NOT: xla.linear_insert
  )"),
              IsOkAndHolds(true));
}

TEST_F(SliceToDynamicTest, CreateSliceToDynamicMlirModuleRank0) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicRank0
    ENTRY main {
      data = s32[] parameter(0)
      ROOT res = s32[] custom-call(data), custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreateSliceToDynamicMlirModule(
                           *mlir_context_, instr, /*buffer_assignment=*/nullptr,
                           "res", launch_dims));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(out, R"(
    CHECK: func.func @res(%arg0: tensor<i32>
    CHECK:   %[[VAL:.*]] = tensor.extract %arg0[] : tensor<i32>
    CHECK:   %[[RET:.*]] = tensor.insert %[[VAL]] into %arg1[] : tensor<i32>
    CHECK:   return %[[RET]] : tensor<i32>
  )"),
              IsOkAndHolds(true));
}

TEST_F(SliceToDynamicTest, RejectsNonSliceToDynamic) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule NotSliceToDynamic
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
  EXPECT_THAT(CreateSliceToDynamicMlirModule(*mlir_context_, instr,
                                             /*buffer_assignment=*/nullptr,
                                             "cc", launch_dims),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected SliceToDynamic custom call")));
}

TEST_F(SliceToDynamicTest, RejectsMismatchedInputShape) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicBadInput
    ENTRY main {
      data = f32[4]{0} parameter(0)
      d0 = s32[] parameter(1)
      ROOT res = s32[<=4]{0} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(4, 1, 1));
  EXPECT_THAT(
      CreateSliceToDynamicMlirModule(*mlir_context_, instr,
                                     /*buffer_assignment=*/nullptr, "res",
                                     launch_dims),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Expected input shape compatible with output shape")));
}

TEST_F(SliceToDynamicTest, RejectsInvalidLaunchDimensions) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicBadLaunch
    ENTRY main {
      data = s32[4]{0} parameter(0)
      d0 = s32[] parameter(1)
      ROOT res = s32[<=4]{0} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  EXPECT_THAT(
      CreateSliceToDynamicMlirModule(
          *mlir_context_, instr, /*buffer_assignment=*/nullptr, "res",
          LaunchDimensions(se::BlockDim(1, 1, 2), se::ThreadDim(4, 1, 1))),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("block_counts().z")));
  EXPECT_THAT(
      CreateSliceToDynamicMlirModule(
          *mlir_context_, instr, /*buffer_assignment=*/nullptr, "res",
          LaunchDimensions(se::BlockDim(1, 1, 1), se::ThreadDim(2, 1, 1))),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("launch_bound()")));
}

TEST_F(SliceToDynamicTest, RejectsUnsupportedSubByteS1) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicS1
    ENTRY main {
      data = s1[16]{0:E(1)} parameter(0)
      d0 = s32[] parameter(1)
      ROOT res = s1[<=16]{0:E(1)} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloCustomCallInstruction* instr = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());

  LaunchDimensions launch_dims(se::BlockDim(1, 1, 1), se::ThreadDim(16, 1, 1));
  EXPECT_THAT(CreateSliceToDynamicMlirModule(*mlir_context_, instr,
                                             /*buffer_assignment=*/nullptr,
                                             "res", launch_dims),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("does not support sub-byte type s1")));
}

TEST_F(SliceToDynamicTest, VerifySubByteS4LlvmIr) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicS4Ir, is_scheduled=true
    ENTRY main {
      data = s4[5]{0:E(4)} parameter(0)
      d0 = s32[] parameter(1)
      ROOT res = s4[<=5]{0:E(4)} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  EXPECT_OK(CompileAndVerifyIr(kHlo,
                               R"(
    CHECK-LABEL: @res(
    CHECK-NOT: store i32
    CHECK: cmpxchg ptr
    CHECK: store i16 %{{.*}}, ptr %{{.*}}, align 2
    CHECK: store i8 %{{.*}}, ptr %{{.*}}, align 1
    CHECK: cmpxchg ptr
  )",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/false));
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicMultiDimNonDefaultLayout) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicMultiDim
    ENTRY main {
      data = s32[3,3,2]{2,0,1} parameter(0)
      d0 = s32[] parameter(1)
      d1 = s32[] parameter(2)
      d2 = s32[] parameter(3)
      dyn = s32[3,<=3,<=2]{2,0,1} custom-call(data, d0, d1, d2),
        custom_call_target="SliceToDynamic"
      ROOT pts = (s32[3,3,2]{2,0,1}, s32[], s32[], s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Shape input_shape =
      ShapeUtil::MakeShapeWithDenseLayout(S32, {3, 3, 2}, {2, 0, 1});
  Literal input(input_shape);
  ASSERT_OK(input.Populate<int32_t>([](auto indices) -> int32_t {
    return static_cast<int32_t>(indices[0] * 100 + indices[1] * 10 +
                                indices[2] + 1);
  }));
  Literal d0 = LiteralUtil::CreateR0<int32_t>(3);
  Literal d1 = LiteralUtil::CreateR0<int32_t>(2);
  Literal d2 = LiteralUtil::CreateR0<int32_t>(1);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&input, &d0, &d1, &d2},
                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 4);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 3);
  EXPECT_EQ(outputs[2].Get<int32_t>({}), 2);
  EXPECT_EQ(outputs[3].Get<int32_t>({}), 1);
  for (int64_t i = 0; i < 3; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      EXPECT_EQ(outputs[0].Get<int32_t>({i, j, 0}), i * 100 + j * 10 + 1);
    }
  }
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicUnalignedF16) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicUnalignedF16
    ENTRY main {
      data = f16[5]{0} parameter(0)
      d0 = s32[] parameter(1)
      dyn = f16[<=5]{0} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
      ROOT pts = (f16[5]{0}, s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Literal input = LiteralUtil::CreateR1<half>(
      {half(1.0f), half(2.0f), half(3.0f), half(4.0f), half(5.0f)});
  Literal d0 = LiteralUtil::CreateR0<int32_t>(3);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input, &d0},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 3);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(outputs[0].Get<half>({i}), half(i + 1.0f));
  }
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicSubByteS4Unaligned) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicSubByteS4
    ENTRY main {
      data = s4[5]{0:E(4)} parameter(0)
      d0 = s32[] parameter(1)
      dyn = s4[<=5]{0:E(4)} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
      ROOT pts = (s4[5]{0:E(4)}, s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Literal input =
      LiteralUtil::CreateR1<s4>({s4(-2), s4(-1), s4(0), s4(1), s4(2)});
  Literal d0 = LiteralUtil::CreateR0<int32_t>(3);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input, &d0},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 3);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(outputs[0].Get<s4>({i}), s4(static_cast<int8_t>(i - 2)));
  }
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicZeroDynamicSize) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicZeroDynSize
    ENTRY main {
      data = s32[4]{0} parameter(0)
      d0 = s32[] parameter(1)
      dyn = s32[<=4]{0} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
      ROOT pts = (s32[4]{0}, s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Literal input = LiteralUtil::CreateR1<int32_t>({10, 20, 30, 40});
  Literal d0 = LiteralUtil::CreateR0<int32_t>(0);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input, &d0},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 0);
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicMultipleBlocks) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicMultipleBlocks
    ENTRY main {
      data = s32[300]{0} parameter(0)
      d0 = s32[] parameter(1)
      dyn = s32[<=300]{0} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
      ROOT pts = (s32[300]{0}, s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  std::vector<int32_t> vals(300);
  for (int32_t i = 0; i < 300; ++i) {
    vals[i] = i + 1;
  }
  Literal input = LiteralUtil::CreateR1<int32_t>(vals);
  Literal d0 = LiteralUtil::CreateR0<int32_t>(257);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input, &d0},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 257);
  for (int64_t i = 0; i < 257; ++i) {
    EXPECT_EQ(outputs[0].Get<int32_t>({i}), i + 1);
  }
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicMultiBlockSubByteS4) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicMultiBlockS4
    ENTRY main {
      data = s4[261]{0:E(4)} parameter(0)
      d0 = s32[] parameter(1)
      dyn = s4[<=261]{0:E(4)} custom-call(data, d0),
        custom_call_target="SliceToDynamic"
      ROOT pts = (s4[261]{0:E(4)}, s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  std::vector<s4> vals;
  vals.reserve(261);
  for (int i = 0; i < 261; ++i) {
    vals.push_back(s4(static_cast<int8_t>((i % 15) - 7)));
  }
  Literal input = LiteralUtil::CreateR1<s4>(vals);
  Literal d0 = LiteralUtil::CreateR0<int32_t>(261);

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input, &d0},
                                               /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 261);
  for (int64_t i = 0; i < 261; ++i) {
    EXPECT_EQ(outputs[0].Get<s4>({i}), s4(static_cast<int8_t>((i % 15) - 7)));
  }
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicZeroElementArray) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicZeroElemsExec
    ENTRY main {
      data = s32[0,4]{1,0} parameter(0)
      d0 = s32[] parameter(1)
      d1 = s32[] parameter(2)
      dyn = s32[0,<=4]{1,0} custom-call(data, d0, d1),
        custom_call_target="SliceToDynamic"
      ROOT pts = (s32[0,4]{1,0}, s32[], s32[]) custom-call(dyn),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Literal input(ShapeUtil::MakeShape(S32, {0, 4}));
  Literal d0 = LiteralUtil::CreateR0<int32_t>(0);
  Literal d1 = LiteralUtil::CreateR0<int32_t>(3);

  ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(std::move(module), {&input, &d0, &d1}, /*run_hlo_passes=*/false));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[1].Get<int32_t>({}), 0);
  EXPECT_EQ(outputs[2].Get<int32_t>({}), 3);
}

TEST_F(SliceToDynamicTest, ExecuteSliceToDynamicRank0) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule SliceToDynamicRank0Exec
    ENTRY main {
      data = s32[] parameter(0)
      ROOT res = s32[] custom-call(data), custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  Literal input = LiteralUtil::CreateR0<int32_t>(42);
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/false));
  EXPECT_EQ(result, LiteralUtil::CreateR0<int32_t>(42));
}

}  // namespace
}  // namespace xla::gpu
