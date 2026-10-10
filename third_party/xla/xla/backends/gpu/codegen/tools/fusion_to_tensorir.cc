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

#include <cstdint>

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "absl/status/statusor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "xla/backends/gpu/codegen/tensor_ir/hlo_to_tensor_ir.h"
#include "xla/backends/gpu/codegen/tensor_ir/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tools/hlo_module_loader.h"

namespace xla::gpu::tensor_ir {
namespace {

// NOLINTNEXTLINE
llvm::cl::opt<bool> compile_flag("compile",
                                 llvm::cl::desc("Compile to CudaTile dialect."),
                                 llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<int64_t> alignment_flag(
    "alignment",
    llvm::cl::desc(
        "Alignment (in bytes) to use for the graph's arguments and result."),
    llvm::cl::init(1));

// NOLINTNEXTLINE
llvm::cl::list<int32_t> tile_shape_flag(
    "tile_shape",
    llvm::cl::desc("Tile shape to use for the CudaTile conversion pipeline "
                   "(only used with --compile)."),
    llvm::cl::CommaSeparated);

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

  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(context));
  absl::StatusOr<mlir::nv_tensor_ir::GraphOp> graph_op_or =
      ImportAndLegalizeComputation(*comp, *module);
  if (!graph_op_or.ok()) {
    mlir::emitError(mlir::UnknownLoc::get(context))
        << graph_op_or.status().message();
    return nullptr;
  }
  mlir::nv_tensor_ir::GraphOp graph_op = *graph_op_or;

  // Annotate the graph's arguments and result with the alignment attribute.
  if (alignment_flag > 1) {
    mlir::StringAttr alignment_attr_name = mlir::StringAttr::get(
        context, mlir::nv_tensor_ir::TensorIRDialect::getAlignmentAttrName());
    mlir::IntegerAttr alignment_attr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(context, 64), alignment_flag.getValue());
    for (int64_t i = 0; i < graph_op.getNumArguments(); ++i) {
      graph_op.setArgAttr(i, alignment_attr_name, alignment_attr);
    }
    for (int64_t i = 0; i < graph_op.getNumResults(); ++i) {
      graph_op.setResultAttr(i, alignment_attr_name, alignment_attr);
    }
  }

  if (compile_flag) {
    mlir::PassManager pass_manager(context);
    mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions options;
    if (!tile_shape_flag.empty()) {
      options.tileSize.assign(tile_shape_flag.begin(), tile_shape_flag.end());
      options.reductionTileSize = options.tileSize.back();
      options.tileSize.pop_back();
    }
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
