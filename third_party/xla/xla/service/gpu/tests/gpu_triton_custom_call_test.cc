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

#include <gtest/gtest.h>
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/verified_hlo_module.h"

namespace xla {
namespace gpu {

using ::mlir::ArrayRef;
using ::mlir::NamedAttribute;

using GpuIrEmitterUnnestedTest = GpuCodegenTest;

TEST_F(GpuIrEmitterUnnestedTest,
       EmitTritonCustomCallWithCorrectLoweringAndWithoutNoaliasOrAlignment) {
  // Tests that the lowering of a Triton custom call produces the correct LLVM
  // IR, and that the arguments do not specify noalias or alignment attributes.

  HloComputation::Builder computation_builder(TestName());
  mlir::MLIRContext context_;
  mlir::Builder builder(&context_);

  // Create parameters and custom call in the computation builder.
  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  // Create the backend_config for the triton custom call.
  const std::string kMLIRText = R"(
  module {
    tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
      %0 = tt.get_program_id x : i32
      %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
      %2 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
      %cst = arith.constant 1.000000e+00 : f32
      %3 = arith.addf %1, %cst : f32
      %4 = tt.load %arg2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
      tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : f32
      %5 = tt.load %arg3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
      tt.store %arg3, %2 {cache = 1 : i32, evict = 1 : i32} : f32
      tt.return
    }
  }
  )";

  NamedAttribute name =
      builder.getNamedAttr("name", builder.getStringAttr("add_one"));
  NamedAttribute ir =
      builder.getNamedAttr("ir", builder.getStringAttr(kMLIRText));
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

  computation_builder.AddInstruction(HloInstruction::CreateCustomCall(
      tuple_shape, {param_0, param_1}, "__gpu$xla.gpu.triton",
      backend_config_str));

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
; CHECK: tail call i32 asm sideeffect
; CHECK: tail call i32 asm sideeffect
; CHECK: fadd float
; CHECK-SAME: 1.000000e+00
; CHECK-DAG: tail call void asm sideeffect
; CHECK-DAG: tail call void asm sideeffect
; CHECK:    ret void
      )",
                     /*match_optimized_ir=*/false);
}

}  // namespace gpu
}  // namespace xla
