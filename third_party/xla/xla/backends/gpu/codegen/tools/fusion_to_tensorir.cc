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

#include <utility>

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/tensor_ir/conversion.h"
#include "xla/backends/gpu/codegen/tensor_ir/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/tools/hlo_module_loader.h"

namespace xla::gpu::tensor_ir {
namespace {

// NOLINTNEXTLINE
llvm::cl::opt<bool> compile_flag("compile",
                                 llvm::cl::desc("Compile to CudaTile dialect."),
                                 llvm::cl::init(false));

mlir::OwningOpRef<mlir::ModuleOp> HloToTensorIRTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  context->loadAllAvailableDialects();

  auto hlo_module = xla::LoadModuleFromData(input, "hlo");
  if (!hlo_module.ok()) {
    mlir::emitError(mlir::UnknownLoc::get(context))
        << hlo_module.status().message();
    return nullptr;
  }

  const HloComputation* comp = (*hlo_module)->entry_computation();
  if (auto fusion = DynCast<HloFusionInstruction>(comp->root_instruction());
      fusion != nullptr) {
    comp = fusion->fused_instructions_computation();
  }

  if (auto decision = IsSupportedFusionComputation(*comp);
      !decision.IsAllowed()) {
    mlir::emitError(mlir::UnknownLoc::get(context)) << decision.Explain();
    return nullptr;
  }

  auto module_or = ConvertFusionComputation(*comp, context);
  if (!module_or.ok()) {
    mlir::emitError(mlir::UnknownLoc::get(context))
        << module_or.status().message();
    return nullptr;
  }
  mlir::OwningOpRef<mlir::ModuleOp> module = *std::move(module_or);

  if (compile_flag) {
    mlir::PassManager pass_manager(context);
    mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions options;
    mlir::nv_tensor_ir::buildTensorToCudaTileConversionPipeline(pass_manager,
                                                                options);
    if (llvm::failed(pass_manager.run(*module))) {
      return nullptr;
    }
  }

  return module;
}

static mlir::TranslateToMLIRRegistration hlo_to_tensorir_registration(
    "hlo-to-tensorir", "Translate HLO to TensorIR", HloToTensorIRTranslate,
    [](mlir::DialectRegistry& registry) {
      registry.insert<mlir::nv_tensor_ir::TensorIRDialect,
                      mlir::cuda_tile::CudaTileDialect,
                      mlir::arith::ArithDialect>();
    });

}  // namespace
}  // namespace xla::gpu::tensor_ir

int main(int argc, char** argv) {
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "HLO Fusion to TensorIR"));
}
