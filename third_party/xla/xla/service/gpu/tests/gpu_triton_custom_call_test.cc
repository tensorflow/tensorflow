/* Copyright 2024 The OpenXLA Authors.

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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/service/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using ::mlir::ArrayRef;
using ::mlir::NamedAttribute;
using ::testing::HasSubstr;

namespace {

constexpr absl::string_view kMLIRText = R"(
module {
  tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
    %0 = tt.get_program_id x : i32
    %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    %2 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    %cst = arith.constant 1.000000e+00 : f32
    %3 = arith.addf %1, %cst : f32
    %4 = tt.load %arg2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
    %5 = tt.load %arg3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    tt.store %arg3, %2 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
    tt.return
  }
}
)";

constexpr absl::string_view kCallName = "add_one";

std::unique_ptr<HloInstruction> CreateTritonCustomCall(
    Shape tuple_shape, std::vector<HloInstruction*> operands,
    absl::string_view mlir_text, absl::string_view call_name,
    bool is_tma_allowed = false, int64_t global_scratch_memory_size = 0,
    int32_t grid_x = 1, int32_t grid_y = 1, int32_t grid_z = 1) {
  mlir::MLIRContext context_;
  mlir::Builder builder(&context_);

  NamedAttribute name =
      builder.getNamedAttr("name", builder.getStringAttr(call_name));
  NamedAttribute ir =
      builder.getNamedAttr("ir", builder.getStringAttr(mlir_text));
  NamedAttribute num_stages =
      builder.getNamedAttr("num_stages", builder.getI32IntegerAttr(3));
  NamedAttribute num_warps =
      builder.getNamedAttr("num_warps", builder.getI32IntegerAttr(4));
  NamedAttribute grid_x_attr =
      builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(grid_x));
  NamedAttribute grid_y_attr =
      builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(grid_y));
  NamedAttribute grid_z_attr =
      builder.getNamedAttr("grid_z", builder.getI32IntegerAttr(grid_z));
  NamedAttribute debug =
      builder.getNamedAttr("debug", builder.getBoolAttr(false));
  NamedAttribute tma_allowed_attr = builder.getNamedAttr(
      "is_tma_allowed", builder.getBoolAttr(is_tma_allowed));
  NamedAttribute scratch_size_attr = builder.getNamedAttr(
      "global_scratch_memory_size",
      builder.getI32IntegerAttr(
          static_cast<int32_t>(global_scratch_memory_size)));

  std::vector<NamedAttribute> attributes = {name,
                                            ir,
                                            num_stages,
                                            num_warps,
                                            grid_x_attr,
                                            grid_y_attr,
                                            grid_z_attr,
                                            debug,
                                            tma_allowed_attr,
                                            scratch_size_attr};
  ArrayRef<NamedAttribute> attributesRef(attributes);
  mlir::DictionaryAttr backend_config =
      mlir::DictionaryAttr::get(&context_, attributesRef);

  // Parse the backend_config into a string.
  std::string backend_config_str;
  llvm::raw_string_ostream(backend_config_str) << backend_config;

  return HloInstruction::CreateCustomCall(
      tuple_shape, operands, "__gpu$xla.gpu.triton", backend_config_str);
}

}  // namespace

class GpuIrEmitterUnnestedTest : public GpuPjRtCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return device_description().cuda_compute_capability();
  }
};

TEST_F(GpuIrEmitterUnnestedTest, EmitTritonCustomCallWithCorrectLowering) {
  if (!GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  // Tests that the lowering of a Triton custom call produces the correct LLVM
  // IR, and that the arguments do not specify noalias or alignment attributes.

  HloComputation::Builder computation_builder(TestName());

  // Create parameters and custom call in the computation builder.
  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(CreateTritonCustomCall(
      tuple_shape, {param_0, param_1}, kMLIRText, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  // Check that the compiled llvm ir matches the expected lowering of our tt ir.
  // We check that the arguments do not specify noalias or alignment attributes,
  // as this prevents recompilation based on the alignment of the input buffers.
  EXPECT_OK(CompileAndVerifyIr(std::move(module),
                               R"(
  ; CHECK: @add_one
; CHECK-SAME: dereferenceable(4) %arg0
; CHECK-SAME: dereferenceable(4) %arg1
; CHECK-SAME: dereferenceable(4) %arg2
; CHECK-SAME: dereferenceable(4) %arg3
; CHECK-DAG:  addrspacecast ptr %arg0 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg1 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg2 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg3 to ptr addrspace(1)
        )",
                               /*match_optimized_ir=*/false));
}

TEST_F(GpuIrEmitterUnnestedTest, EmitTritonCustomCallParseErrorHasEscapedIr) {
  if (!GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  // Tests that MLIR IR with invalid unicode characters is escaped correctly
  // on error.
  constexpr absl::string_view kMlirIrInvalidUnicode = "ML\xef";

  HloComputation::Builder computation_builder(TestName());

  // Create parameters and custom call in the computation builder.
  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(CreateTritonCustomCall(
      tuple_shape, {param_0, param_1}, kMlirIrInvalidUnicode, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  auto result =
      CompileToExecutable(std::move(module), /*run_optimization_passes=*/true);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("Failed to parse Triton module"));

  // Verify that the error message contains the escaped IR.
  EXPECT_THAT(result.status().message(), HasSubstr("input ir: \"ML\\xef\""));
}

TEST_F(GpuIrEmitterUnnestedTest,
       EmitTritonCustomCallWithCorrectKernelParamAttributes) {
  if (!GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  constexpr absl::string_view kMLIRTextWithTMAAttributes = R"(
    module {
      tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.nv_tma_desc = 1 : i32},
      %arg1: !tt.ptr<f32, 1> {tt.nv_tma_desc = 1 : i32},
      %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32},
      %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
        %0 = tt.get_program_id x : i32
        %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
        %cst = arith.constant 1.000000e+00 : f32
        %3 = arith.addf %1, %cst : f32
        %4 = tt.load %arg2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
        tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
        tt.return
      }
    }
    )";

  // Check that the compiled LLVM IR retains the ByVal attribute that we expect
  // to be added in the TTIR lowering when we use tt.nv_tma_desc.
  HloComputation::Builder computation_builder(TestName());

  // Create parameters and custom call in the computation builder.
  Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {32, 256});
  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "arg_0"));
  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "arg_1"));

  computation_builder.AddInstruction(CreateTritonCustomCall(
      ShapeUtil::MakeTupleShape({shape, std::move(shape)}), {param_0, param_1},
      kMLIRTextWithTMAAttributes, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  // TODO(b/412980654): for custom kernels, the alignment attribute is getting
  // dropped for compile time so we do not check for this.
  EXPECT_OK(CompileAndVerifyIr(std::move(module),
                               R"(
  ; CHECK: @add_one
  ; CHECK: byval([128 x i8])
        )",
                               /*match_optimized_ir=*/false));
}

TEST_F(GpuIrEmitterUnnestedTest, RunTritonCustomCallWithDeviceSideTMA) {
  if (!device_description().cuda_compute_capability().IsAtLeastHopper()) {
    GTEST_SKIP() << "Device-side TMA is only supported on Hopper and up.";
  }

  // A kernel that copies arg0 to arg1 using TMA.
  // We set global_scratch_memory_size > 0, so Triton will add an implicit
  // 3rd argument for the scratchpad.
  constexpr absl::string_view kTMAMLIRText = R"(
    module {
      tt.func public @tma_kernel(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) {
        %0 = tt.get_program_id x : i32
        %1 = tt.get_program_id y : i32
        %c128_i32 = arith.constant 128 : i32
        %c128_i64 = arith.constant 128 : i64
        %c1_i64 = arith.constant 1 : i64
        %c64_i32 = arith.constant 64 : i32

        %desc0 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64] : <f16>, <tensor<64x64xf16>>
        %desc1 = tt.make_tensor_descriptor %arg1, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64] : <f16>, <tensor<64x64xf16>>

        %8 = arith.muli %0, %c64_i32 : i32
        %9 = arith.muli %1, %c64_i32 : i32

        %10 = tt.descriptor_load %desc0[%8, %9] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
        tt.descriptor_store %desc1[%8, %9], %10 : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16>
        tt.return
      }
    }
  )";

  HloComputation::Builder computation_builder(TestName());

  Shape shape = xla::ShapeUtil::MakeShape(xla::F16, {128, 128});
  Shape scratch_shape = xla::ShapeUtil::MakeShape(xla::U8, {32768});
  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "arg_0"));

  // HLO operand {param_0} -> Triton arg0.
  // HLO output tuple element 0 -> Triton arg1.
  // HLO output tuple element 1 -> Triton arg2 (implicit scratchpad).
  computation_builder.AddInstruction(CreateTritonCustomCall(
      ShapeUtil::MakeTupleShape({shape, scratch_shape}), {param_0},
      kTMAMLIRText, "tma_kernel",
      /*is_tma_allowed=*/true, /*global_scratch_memory_size=*/32768,
      /*grid_x=*/2, /*grid_y=*/2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  // Initialize input data.
  Literal input_literal =
      LiteralUtil::CreateFullWithDescendingLayout<float>({128, 128}, 1.0f);
  input_literal = input_literal.Convert(F16).value();

  // Run on GPU.
  absl::StatusOr<Literal> result_status =
      Execute(std::move(module), {&input_literal});
  TF_ASSERT_OK(result_status.status());
  std::vector<Literal> results = result_status->DecomposeTuple();

  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal, results.at(0)));
}

TEST_F(GpuIrEmitterUnnestedTest, CanNotEmitTritonCustomCallOnPreAmpereGpu) {
  if (GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Running on Ampere or more recent GPU, skipping.";
  }

  HloComputation::Builder computation_builder(TestName());

  // Create parameters and custom call in the computation builder.
  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(CreateTritonCustomCall(
      tuple_shape, {param_0, param_1}, kMLIRText, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  EXPECT_THAT(
      CompileToExecutable(std::move(module), /*run_optimization_passes=*/false),
      absl_testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

TEST_F(GpuIrEmitterUnnestedTest, FailGracefullyIfTritonModuleIsNotParseable) {
  if (!GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  HloComputation::Builder computation_builder(TestName());

  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(
      CreateTritonCustomCall(tuple_shape, {param_0, param_1},
                             /*mlir_text=*/"unparseable_mlir", kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_THAT(Run(std::move(module), /*run_hlo_passes=*/false).message(),
              HasSubstr("Failed to parse Triton module"));
}

TEST_F(GpuIrEmitterUnnestedTest, FailGracefullyIfCallNameIsInvalid) {
  if (!GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  HloComputation::Builder computation_builder(TestName());

  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(
      CreateTritonCustomCall(tuple_shape, {param_0, param_1}, kMLIRText,
                             /*call_name=*/"invalid_call_name"));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_THAT(Run(std::move(module), /*run_hlo_passes=*/false).message(),
              HasSubstr("Call name not found in the Triton module"));
}

class TritonCustomCallTest : public HloPjRtGpuTestBase {};

TEST_F(TritonCustomCallTest, NoArgumentDeduplication) {
  if (se::CudaComputeCapability cc =
          device_description().cuda_compute_capability();
      !cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  // Tests that no argument deduplication is done for Triton kernels.
  //
  // Triton kernels are compiled on the first call and re-used for all the
  // following calls. So, if we are unlucky, we could end up calling the
  // compiled kernel with fewer arguments than it expects in the presence
  // of argument deduplication.
  //
  // For example,
  //
  //  * The first call is f(x, y). The arguments are distinct, no deduplication
  //    is done at compilation time and the compiled kernel expects two
  //    arguments.
  //  * The second call is f(x, x). The arguments are deduplicated and we
  //    call the previously compiled kernel with just x, causing a crash.

  HloComputation::Builder computation_builder(TestName());

  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  auto* instr_0 = computation_builder.AddInstruction(CreateTritonCustomCall(
      tuple_shape, {param_0, param_1}, kMLIRText, kCallName));
  computation_builder.AddInstruction(CreateTritonCustomCall(
      tuple_shape, {instr_0, instr_0}, kMLIRText, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
}

}  // namespace gpu
}  // namespace xla
