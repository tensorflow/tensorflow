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

#include "xla/backends/gpu/codegen/tensor_ir/hlo_to_tensor_ir.h"

#include <string>
#include <unordered_map>

#include "tensor_ir/Dialect/TensorIR.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/tensor_ir/layout_utils.h"
#include "xla/backends/gpu/codegen/tensor_ir/transforms/passes.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::gpu::tensor_ir {

absl::StatusOr<mlir::nv_tensor_ir::GraphOp> ImportAndLegalizeComputation(
    const HloComputation& computation, mlir::ModuleOp module) {
  if (module == nullptr) {
    return absl::InvalidArgumentError(
        "ImportAndLegalizeComputation: module is null");
  }
  mlir::MLIRContext* context = module.getContext();
  context->loadDialect<mlir::nv_tensor_ir::TensorIRDialect,
                       mlir::stablehlo::StablehloDialect,
                       mlir::func::FuncDialect, mlir::arith::ArithDialect>();

  // Collect MLIR diagnostics so that they end up in the returned status rather
  // than on stderr.
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);

  mlir::SymbolTable symbol_table(module);
  mlir::Builder builder(context);
  std::unordered_map<const HloComputation*, mlir::func::FuncOp> function_map;

  absl::StatusOr<mlir::func::FuncOp> func_op =
      HloFunctionImporter::ImportAsFunc(computation, symbol_table,
                                        &function_map, &builder,
                                        /*is_main=*/true,
                                        /*flatten_computation_args_result=*/
                                        false);
  if (!func_op.ok()) {
    return diagnostic_handler.Combine(absl::Status(
        func_op.status().code(),
        absl::StrCat("Failed to import HLO computation '", computation.name(),
                     "' as StableHLO: ", func_op.status().message())));
  }

  mlir::PassManager pass_manager(context);
  pass_manager.addPass(
      mlir::nv_tensor_ir::xla::createLegalizeStablehloToTensorIrPass());
  if (llvm::failed(pass_manager.run(module))) {
    return diagnostic_handler.Combine(absl::InternalError(
        absl::StrCat("Failed to legalize StableHLO to TensorIR for "
                     "computation '",
                     computation.name(), "': ")));
  }

  llvm::SmallVector<mlir::nv_tensor_ir::GraphOp> graph_ops(
      module.getOps<mlir::nv_tensor_ir::GraphOp>());
  if (graph_ops.size() != 1) {
    return diagnostic_handler.Combine(absl::InternalError(
        absl::StrCat("Expected exactly one nv_tensor_ir.graph after legalizing "
                     "computation '",
                     computation.name(), "', got ", graph_ops.size())));
  }

  // Surface any error diagnostic that was reported without the pass manager
  // signalling failure.
  absl::Status diagnostics = diagnostic_handler.ConsumeStatus();
  if (!diagnostics.ok()) {
    return diagnostics;
  }

  // The computation is imported as the module's main function, so both the
  // function and the graph legalized from it are named `main`. Rename the
  // graph after the fact: the name ends up as the compiled kernel's name, and
  // calling every kernel `main` makes profiles and traces unattributable.
  mlir::nv_tensor_ir::GraphOp graph_op = graph_ops.front();
  graph_op.setSymName(
      llvm_ir::SanitizeFunctionName(std::string(computation.name())));

  // Non-default HLO layouts survive as `nv_tensor_ir.stride` attributes on the
  // graph boundary. This has to happen here rather than in the emitter: the
  // layout-propagation passes read these attributes and silently assume
  // row-major when they are absent, so a graph without them tiles differently
  // from one with them.
  ABSL_RETURN_IF_ERROR(AttachLayoutStrides(graph_op, computation));
  return graph_op;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportAndLegalizeComputation(
    const HloComputation& computation, mlir::MLIRContext* context) {
  if (context == nullptr) {
    return absl::InvalidArgumentError(
        "ImportAndLegalizeComputation: context is null");
  }
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(context));
  // All the work, including diagnostic capture, happens in the overload above.
  absl::StatusOr<mlir::nv_tensor_ir::GraphOp> graph_op =
      ImportAndLegalizeComputation(computation, *module);
  if (!graph_op.ok()) {
    return graph_op.status();
  }
  return module;
}

}  // namespace xla::gpu::tensor_ir
