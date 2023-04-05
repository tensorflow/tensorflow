/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir/utils/error_util.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

namespace xla {

Status MlirToXlaComputation(mlir::ModuleOp module,
                            XlaComputation& xla_computation,
                            bool use_tuple_args, bool return_tuple) {
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(module->getContext());
  {
    mlir::PassManager pm(module->getContext());
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createLegalizeSparseChloToLinalgPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createChloLegalizeToHloPass(
            /*legalizeBroadcasts=*/true, /*expandCompositions=*/true));
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // In order to export to XLA, we must sink constants to control flow
    // regions, since XLA uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createSinkConstantsToControlFlowPass());
    if (failed(pm.run(module))) {
      VLOG(1) << "MHLO->HLO lowering passes failed.";
      module->dump();
      return FromAbslStatus(diagnostic_handler.ConsumeStatus());
    }

    VLOG(5) << "MHLO module after lowering, before HLO import ";
    if (VLOG_IS_ON(5)) {
      module->dump();
    }
  }

  HloProto proto;
  mlir::MlirToHloConversionOptions options;
  TF_RETURN_IF_ERROR(ConvertMlirHloToHlo(module, &proto, use_tuple_args,
                                         return_tuple, options));

  xla_computation = XlaComputation(std::move(*proto.mutable_hlo_module()));
  return OkStatus();
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
  context.loadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
  module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(mlir_module_str.data(), mlir_module_str.size()),
      &context);
  if (!module) {
    return FromAbslStatus(diagnostic_handler.ConsumeStatus());
  }
  if (failed(module->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    module->dump();
    return FromAbslStatus(diagnostic_handler.ConsumeStatus());
  }
  return std::move(module);
}

Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, context));
  return xla::MlirToXlaComputation(*module, xla_computation, use_tuple_args,
                                   return_tuple);
}

}  // namespace xla
