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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/import_model.h"

#include <utility>

#include "base/vlog_is_on.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/assign_op_key.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/while_to_map_fn.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/mlrt/mlir_to_bytecode.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/mlrt/attribute/attribute.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace mlrt_compiler {

StatusOr<mlrt::bc::Buffer> ConvertTfMlirToBytecode(
    const TfrtCompileOptions& options,
    const tfrt_stub::FallbackState& fallback_state, mlir::ModuleOp module,
    tfrt_stub::ModelRuntimeContext& model_context,
    mlir::OwningOpRef<mlir::ModuleOp>* module_with_op_keys) {
  mlrt::bc::Buffer bytecode_buffer;
  TF_RETURN_IF_ERROR(ConvertTfMlirToRuntimeExecutable(
      options, module,
      [&bytecode_buffer, &fallback_state, module_with_op_keys](
          mlir::PassManager& pm, mlir::ModuleOp module,
          const TfrtPipelineOptions& options) {
        mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

        if (options.enable_while_parallel_iterations) {
          pm.addPass(mlrt_compiler::CreateWhileToMapFnPass());
          // Remove unreachable private functions after mapfn conversion.
          pm.addPass(mlir::createSymbolDCEPass());
        }
        tensorflow::CreateTFExecutorToTFInvariantOptimizationPipelineHelper(
            pm, options);
        // TODO(b/283481729): Add test to cover unused constants that do not
        // cause op_key discontinuity
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
        pm.addPass(mlrt_compiler::CreateAssignOpKeyPass());
        // Run passes until (including) AssignOpKeyPass.
        if (mlir::failed(pm.run(module))) {
          return diag_handler.Combine(absl::InternalError(
              "failed to finish passes before (including) assign op keys."));
        }
        if (VLOG_IS_ON(1)) {
          tensorflow::DumpMlirOpToFile("tf_dialect_after_assign_op_key",
                                       module);
        }
        // Save the module.
        if (module_with_op_keys != nullptr) {
          *module_with_op_keys = module.clone();
        }
        // Clear passes already run.
        pm.clear();
        // Create the remaining pipeline and run.
        CreateTfToMlrtPipeline(pm, options, &fallback_state);
        if (mlir::failed(pm.run(module))) {
          return diag_handler.Combine(absl::InternalError(
              "failed to lower TF Dialect to MLRT dialect."));
        }
        // Generate bytecode.
        mlrt::AttributeEncoderRegistry registry;
        registry.Register("tf_mlrt",
                          &tensorflow::tf_mlrt::EncodeTensorflowAttribute);
        auto statusor = mlrt::EmitExecutable(registry, module);
        if (!statusor.ok()) return statusor.status();
        bytecode_buffer = std::move(*statusor);
        return OkStatus();
      },
      model_context));
  return bytecode_buffer;
}

StatusOr<mlrt::bc::Buffer> ConvertTfMlirWithOpKeysToBytecode(
    const TfrtCompileOptions& options,
    const tfrt_stub::FallbackState& fallback_state,
    mlir::ModuleOp module_with_op_keys,
    const tfrt_stub::CostRecorder& cost_recorder) {
  mlir::StatusScopedDiagnosticHandler diag_handler(
      module_with_op_keys.getContext());
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tf_dialect_with_op_keys",
                                 module_with_op_keys);
  }
  // Create the reconversion pipeline and run.
  mlir::PassManager pm(module_with_op_keys.getContext());
  const auto pipeline_options = GetTfrtPipelineOptions(options);
  CreateTfToMlrtPipeline(pm, *pipeline_options, &fallback_state,
                         &cost_recorder);
  if (mlir::failed(pm.run(module_with_op_keys))) {
    return diag_handler.Combine(
        absl::InternalError("failed to lower TF Dialect to MLRT dialect."));
  }
  // Generate bytecode.
  mlrt::AttributeEncoderRegistry registry;
  registry.Register("tf_mlrt", &tensorflow::tf_mlrt::EncodeTensorflowAttribute);
  auto statusor = mlrt::EmitExecutable(registry, module_with_op_keys);
  if (!statusor.ok()) return statusor.status();
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tfrt_dialect_from_tf_dialect_with_op_keys",
                                 module_with_op_keys);
  }
  return std::move(*statusor);
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
