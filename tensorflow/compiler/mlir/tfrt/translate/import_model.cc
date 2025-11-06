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

#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/lower_cluster_to_runtime_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_asset_sinking_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/cluster_tf.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_executor_to_graph.h"
#include "tensorflow/compiler/mlir/tfrt/backend_compiler.h"
#include "tensorflow/compiler/mlir/tfrt/function/function.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tpu_passes.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/tsl/framework/device_type.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {

namespace {

// Exports all XLA functions in the form of XlaLaunch, and their nested
// functions.
absl::StatusOr<std::vector<FunctionDef>> ExportXlaFunctions(
    mlir::ModuleOp module, std::vector<std::string>* added_xla_function_names) {
  // Find all XLA functions.
  std::vector<std::string> xla_functions;
  module.walk([&](mlir::TF::XlaLaunchOp xla_launch_op) {
    std::string func_name =
        xla_launch_op.getFunctionAttr().getRootReference().str();
    xla_functions.push_back(func_name);
    if (added_xla_function_names != nullptr) {
      added_xla_function_names->push_back(func_name);
    }
  });

  // Convert all XLA functions and their nested functions.
  std::deque<std::string> queue;
  for (const std::string& func : xla_functions) {
    queue.push_back(func);
  }

  const mlir::SymbolTable symbol_table(module);
  absl::flat_hash_set<std::string> visited;
  std::vector<FunctionDef> xla_func_defs;
  while (!queue.empty()) {
    const std::string func_name = queue.front();
    queue.pop_front();

    if (visited.contains(func_name)) continue;

    const auto func_op = symbol_table.lookup<mlir::func::FuncOp>(func_name);
    if (!func_op) {
      return absl::InternalError(
          absl::StrCat("Function ", func_name, " is not found."));
    }
    FunctionDef func_def;
    TF_RETURN_IF_ERROR(
        tensorflow::tf2xla::v2::ConvertMlirFunctionToFunctionLibraryDef(
            func_op, GraphExportConfig(), &func_def));
    xla_func_defs.push_back(func_def);

    // Visit each op in the function and find out referenced functions from the
    // attributes.
    func_op->walk([&](mlir::Operation* op) {
      for (const mlir::NamedAttribute& attr : op->getAttrs()) {
        if (const auto sym =
                mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr.getValue())) {
          mlir::Operation* func =
              mlir::SymbolTable::lookupNearestSymbolFrom(op, sym);
          if (func) {
            queue.push_back(sym.getValue().str());
          }
        }
      }
    });

    // Remove the function from the module, as it will be handled by XLA.
    // It is safe to remove the function, i.e., the function won't be invoked on
    // CPU. This is because bridge guarantees that each function has only one
    // use. We don't replace the uses of the function, because we iterate from
    // the root caller and hence its uses should have been removed.
    func_op->erase();

    visited.insert(func_name);
  }
  return xla_func_defs;
}

}  // namespace

absl::Status ConvertTfMlirToRuntimeExecutable(
    const TfrtCompileOptions& options, mlir::ModuleOp module,
    absl::FunctionRef<
        absl::Status(mlir::PassManager&, mlir::ModuleOp,
                     const tensorflow::TfrtPipelineOptions& options)>
        emit_executable,
    tfrt_stub::ModelRuntimeContext& model_context,
    tfrt_stub::FallbackState* fallback_state,
    std::vector<std::string>* added_xla_function_names) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  {
    mlir::PassManager pm(module.getContext());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tf_executor::CreateTFExecutorGraphPruningPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::CreateExecutorDialectToFunctionalConversionPass());
    if (!options.saved_model_dir.empty()) {
      pm.addPass(mlir::tf_saved_model::CreateAssetSinkingPass(
          options.saved_model_dir));
    }
    if (mlir::failed(pm.run(module))) {
      return diag_handler.Combine(absl::InternalError(
          "Failed to sinking assets into initialization graphs."));
    }
  }

  if (options.backend_compiler != nullptr) {
    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("tf_dialect_before_backend_compile", module);
    }
    TF_RETURN_IF_ERROR(
        options.backend_compiler->CompileTensorflow(model_context, module));
  } else if (options.device_target == TfrtDeviceInfraTarget::kTpurt) {
    VLOG(1) << "Running MLIR TPU bridge for tpurt";
    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("tpu_bct_conversion_before", module);
    }

    TfrtTpuCompileOptions tpu_compile_options;
    tpu_compile_options.move_resource_gather_to_host =
        options.tpu_move_resource_gather_to_host;
    tpu_compile_options.gather_table_width_threshold_bytes =
        options.tpu_gather_table_width_threshold_bytes;

    auto backward_compat_result =
        tensorflow::RunTPUBackwardCompatConversion(module, tpu_compile_options);
    if (mlir::failed(backward_compat_result)) {
      return diag_handler.Combine(
          absl::InternalError("Failed to handle legacy TPU Ops"));
    }

    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("tpu_bct_conversion_after", module);
    }

    TF_RETURN_IF_ERROR(
        tensorflow::tf2xla::v2::RunFunctionTf2xlaClusteringBridge(
            module, /*is_supported_by_replicated_brige*/ true,
            /*is_in_fallback_enabled_mode=*/VLOG_IS_ON(1)));

    TF_RETURN_IF_ERROR(
        tensorflow::tfrt_compiler::RunLowerClusterToRuntimeOpsPassPipeline(
            module, tsl::DeviceType(DEVICE_TPU_XLA_JIT)));

    TF_RETURN_IF_ERROR(
        tensorflow::tf2xla::v2::ExportFromTensorflowDialectToExecutor(module));
  } else if (options.device_target == TfrtDeviceInfraTarget::kTfFallback) {
    auto tpu_partitioned_call_fallback_compat_result =
        tensorflow::RunTPUPartitionedCallFallbackCompatConversion(module);
    if (mlir::failed(tpu_partitioned_call_fallback_compat_result)) {
      return diag_handler.Combine(absl::InternalError(
          "Failed to process TPUPartitionedCallOp for fallback execution"));
    }
  } else if (options.device_target == TfrtDeviceInfraTarget::kGpu) {
    TF_RETURN_IF_ERROR(
        tensorflow::tf2xla::v2::RunFunctionTf2xlaClusteringBridge(
            module, /*is_supported_by_replicated_brige*/ false,
            /*is_in_fallback_enabled_mode=*/false));

    TF_RETURN_IF_ERROR(
        tensorflow::tfrt_compiler::RunLowerClusterToRuntimeOpsPassPipeline(
            module, tsl::DeviceType(DEVICE_GPU_XLA_JIT)));

    TF_RETURN_IF_ERROR(
        tensorflow::tf2xla::v2::ExportFromTensorflowDialectToExecutor(module));

    if (options.serialize_mlir_module_to_aot_packages) {
      const std::string mlir_string = SerializeMlirModule(module);
      TF_RETURN_IF_ERROR(WriteStringToFile(
          tsl::Env::Default(), options.aot_mlir_module_file, mlir_string));
    }

    // GPU XLA clusters are wrapped in functions, which could be transformed by
    // bridge. Hence, the MLIR functions for XLA clusters are exported and added
    // to the function library.
    TF_RETURN_IF_ERROR(
        AddXlaFunctions(fallback_state, module, added_xla_function_names));
  }

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tf_dialect", module);
  }

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  module.getContext()->appendDialectRegistry(registry);

  // Lower MLIR TF Dialect to MLIR TFRT CoreRT dialect.
  mlir::PassManager pm(module.getContext());

  auto pipeline_options = GetTfrtPipelineOptions(options);

  TF_RETURN_IF_ERROR(
      tensorflow::CreateTFExecutorToTFPreInvariantOptimizationPipeline(
          pm, *pipeline_options));

  auto status = emit_executable(pm, module, *pipeline_options);

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tfrt_dialect", module);
  }

  return status;
}

absl::Status ConvertTfMlirToBef(
    const TfrtCompileOptions& options, mlir::ModuleOp module,
    tfrt::BefBuffer* bef_buffer, tfrt_stub::ModelRuntimeContext& model_context,
    tfrt_stub::FallbackState* fallback_state,
    std::vector<std::string>* added_xla_function_names) {
  return ConvertTfMlirToRuntimeExecutable(
      options, module,
      [bef_buffer](mlir::PassManager& pm, mlir::ModuleOp module,
                   const tensorflow::TfrtPipelineOptions& options) {
        mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
        tensorflow::CreateTFInvariantOptimizationPipelineHelper(pm, options);
        tensorflow::CreateTfToTfrtPipeline(pm, options);

        if (mlir::failed(pm.run(module))) {
          if (VLOG_IS_ON(1)) {
            tensorflow::DumpMlirOpToFile("tf_to_corert_failure", module);
          }
          return diag_handler.Combine(absl::InternalError(
              "failed to lower TF Dialect to CoreRT dialect."));
        }

        *bef_buffer =
            tfrt::ConvertMLIRToBEF(module, /*disable_optional_sections=*/true);
        if (bef_buffer->empty())
          return diag_handler.Combine(
              absl::InternalError("failed to convert MLIR to BEF."));

        bef_buffer->shrink_to_fit();
        return absl::OkStatus();
      },
      model_context, fallback_state, added_xla_function_names);
}

std::unique_ptr<tensorflow::TfrtPipelineOptions> GetTfrtPipelineOptions(
    const TfrtCompileOptions& options) {
  auto pipeline_options = std::make_unique<tensorflow::TfrtPipelineOptions>();

  pipeline_options->saved_model_dir = options.saved_model_dir;

  if (!options.default_device.empty()) {
    pipeline_options->default_device = options.default_device;
  }
  if (!options.force_data_format.empty()) {
    pipeline_options->force_data_format = options.force_data_format;
  }

  // TODO(b/187991150): Consider only decomposing read-only resource variable
  // ops.
  pipeline_options->decompose_resource_ops = options.decompose_resource_ops;
  pipeline_options->enable_optimizer = options.enable_optimizer;
  pipeline_options->target_tpurt =
      (options.device_target == TfrtDeviceInfraTarget::kTpurt);
  pipeline_options->target_gpu =
      (options.device_target == TfrtDeviceInfraTarget::kGpu);
  pipeline_options->use_gpu_compile_and_execute_op =
      options.use_gpu_compile_and_execute_op;
  pipeline_options->tpu_fuse_ops = options.tpu_fuse_ops;
  pipeline_options->use_tpu_host_allocator_for_inputs =
      options.use_tpu_host_allocator_for_inputs;
  pipeline_options->tpu_allow_unpadded_batch = options.tpu_allow_unpadded_batch;
  pipeline_options->sink_in_invariant_ops = options.sink_in_invariant_ops;
  pipeline_options->hoist_invariant_ops = options.hoist_invariant_ops;
  pipeline_options->fuse_get_resource_ops_in_hoisting =
      options.fuse_get_resource_ops_in_hoisting;
  pipeline_options->enable_while_parallel_iterations =
      options.enable_while_parallel_iterations;
  pipeline_options->cost_threshold = options.cost_threshold;
  pipeline_options->min_num_batch_threads = options.min_num_batch_threads;
  pipeline_options->min_max_enqueued_batches = options.min_max_enqueued_batches;
  pipeline_options->batch_queue_global_prioritization_num_threads =
      options.batch_queue_global_prioritization_num_threads;
  pipeline_options->batch_padding_policy = options.batch_padding_policy;
  pipeline_options->num_batch_threads =
      options.batch_options.num_batch_threads();
  pipeline_options->max_batch_size = options.batch_options.max_batch_size();
  pipeline_options->batch_timeout_micros =
      options.batch_options.batch_timeout_micros();
  pipeline_options->allowed_batch_sizes = llvm::ArrayRef<int64_t>(
      std::vector<int64_t>(options.batch_options.allowed_batch_sizes().begin(),
                           options.batch_options.allowed_batch_sizes().end()));
  pipeline_options->max_enqueued_batches =
      options.batch_options.max_enqueued_batches();
  pipeline_options->merge_inter_dependent_streams =
      options.merge_inter_dependent_streams;

  return pipeline_options;
}

absl::Status AddXlaFunctions(
    tfrt_stub::FallbackState* fallback_state, mlir::ModuleOp mlir_module,
    std::vector<std::string>* added_xla_function_names) {
  if (fallback_state != nullptr) {
    TF_ASSIGN_OR_RETURN(
        const std::vector<FunctionDef> xla_func_defs,
        ExportXlaFunctions(mlir_module, added_xla_function_names));
    for (const auto& func_def : xla_func_defs) {
      TF_RETURN_IF_ERROR(fallback_state->AddFunctionDef(func_def));
    }
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
