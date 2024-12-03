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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status_matchers.h"

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
    Shape tuple_shape, HloInstruction* param_0, HloInstruction* param_1,
    absl::string_view mlir_text, absl::string_view call_name) {
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
  NamedAttribute grid_x =
      builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(1));
  NamedAttribute grid_y =
      builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(1));
  NamedAttribute grid_z =
      builder.getNamedAttr("grid_z", builder.getI32IntegerAttr(1));
  NamedAttribute debug =
      builder.getNamedAttr("debug", builder.getBoolAttr(false));

  std::vector<NamedAttribute> attributes = {
      name, ir, num_stages, num_warps, grid_x, grid_y, grid_z, debug};
  ArrayRef<NamedAttribute> attributesRef(attributes);
  mlir::DictionaryAttr backend_config =
      mlir::DictionaryAttr::get(&context_, attributesRef);

  // Parse the backend_config into a string.
  std::string backend_config_str;
  llvm::raw_string_ostream(backend_config_str) << backend_config;

  return HloInstruction::CreateCustomCall(tuple_shape, {param_0, param_1},
                                          "__gpu$xla.gpu.triton",
                                          backend_config_str);
}

}  // namespace

class GpuIrEmitterUnnestedTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_F(GpuIrEmitterUnnestedTest,
       EmitTritonCustomCallWithCorrectLoweringAndWithoutNoaliasOrAlignment) {
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
      tuple_shape, param_0, param_1, kMLIRText, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  // Check that the compiled llvm ir matches the expected lowering of our tt ir.
  // We check that the arguments do not specify noalias or alignment attributes,
  // as this prevents recompilation based on the alignment of the input buffers.
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK: @add_one
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg0
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg1
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg2
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg3
; CHECK-DAG:  addrspacecast ptr %arg0 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg1 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg2 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg3 to ptr addrspace(1)
      )",
                     /*match_optimized_ir=*/false);
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
      tuple_shape, param_0, param_1, kMLIRText, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  EXPECT_THAT(
      CompileToExecutable(std::move(module), /*run_optimization_passes=*/false),
      tsl::testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

class TritonCustomCallTest : public HloTestBase {};

TEST_F(TritonCustomCallTest, NoArgumentDeduplication) {
  if (auto cc = backend()
                    .default_stream_executor()
                    ->GetDeviceDescription()
                    .cuda_compute_capability();
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
      tuple_shape, param_0, param_1, kMLIRText, kCallName));
  computation_builder.AddInstruction(CreateTritonCustomCall(
      tuple_shape, instr_0, instr_0, kMLIRText, kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
}

TEST_F(TritonCustomCallTest, FailGracefullyIfTritonModuleIsNotParseable) {
  if (auto cc = backend()
                    .default_stream_executor()
                    ->GetDeviceDescription()
                    .cuda_compute_capability();
      cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "Running on Ampere or more recent GPU, skipping.";
  }

  HloComputation::Builder computation_builder(TestName());

  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(
      CreateTritonCustomCall(tuple_shape, param_0, param_1,
                             /*mlir_text=*/"unparseable_mlir", kCallName));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_THAT(Run(std::move(module), /*run_hlo_passes=*/false).message(),
              HasSubstr("Failed to parse Triton module"));
}

TEST_F(TritonCustomCallTest, FailGracefullyIfCallNameIsInvalid) {
  if (auto cc = backend()
                    .default_stream_executor()
                    ->GetDeviceDescription()
                    .cuda_compute_capability();
      cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "Running on Ampere or more recent GPU, skipping.";
  }

  HloComputation::Builder computation_builder(TestName());

  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(
      CreateTritonCustomCall(tuple_shape, param_0, param_1, kMLIRText,
                             /*call_name=*/"invalid_call_name"));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_THAT(Run(std::move(module), /*run_hlo_passes=*/false).message(),
              HasSubstr("Call name not found in the Triton module"));
}

}  // namespace gpu
}  // namespace xla
