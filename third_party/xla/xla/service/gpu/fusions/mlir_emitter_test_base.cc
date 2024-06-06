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

#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"

#include <string>
#include <string_view>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

MlirEmitterTestBaseImpl::MlirEmitterTestBaseImpl() {
  // clang-format off
  mlir_context_.loadDialect<
      mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,
      mlir::complex::ComplexDialect,
      mlir::func::FuncDialect,
      mlir::gpu::GPUDialect,
      mlir::math::MathDialect,
      mlir::mhlo::MhloDialect,
      mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect,
      xla::gpu::XlaGpuDialect
  >();
  // clang-format on
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir_context_.appendDialectRegistry(registry);
  thread_id_printer_ =
      AffineMapPrinter({"th_x", "th_y", "th_z", "bl_x", "bl_y", "bl_z"}, {});
}

DebugOptions MlirEmitterTestBaseImpl::GetDebugOptionsForTest() {
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_mlir_emitter_level(4);
  return debug_options;
}

absl::StatusOr<std::string> MlirEmitterTestBaseImpl::EmitIR(
    std::string_view hlo_string) {
  TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  auto fusion_emitter = GetEmitter(analysis);

  TF_ASSIGN_OR_RETURN(auto mlir_module,
                      fusion_emitter->CreateMLIRModule(
                          mlir_context_, *Cast<HloFusionInstruction>(root),
                          "fused_computation", nullptr));

  std::string out;
  llvm::raw_string_ostream os(out);
  mlir_module->print(os);

  return out;
}

absl::Status MlirEmitterTestBaseImpl::EmitAndCheckIR(
    std::string_view hlo_string, std::string_view pattern) {
  TF_ASSIGN_OR_RETURN(auto ir, EmitIR(hlo_string));
  TF_ASSIGN_OR_RETURN(auto check_result, RunFileCheck(ir, pattern));
  return check_result ? absl::Status()
                      : absl::FailedPreconditionError("match failure");
}

}  // namespace gpu
}  // namespace xla
