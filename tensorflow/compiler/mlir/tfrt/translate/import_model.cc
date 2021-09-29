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

#include "absl/strings/match.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime

namespace tensorflow {

Status ConvertFunctionToBef(
    mlir::StringRef function_name, const tensorflow::FunctionBody* fbody,
    const FunctionLibraryDefinition& flib_def,
    tfrt::ArrayRef<tfrt::string_view> devices,
    const tensorflow::TfrtFunctionCompileOptions& options,
    tfrt::BefBuffer* bef_buffer) {
  mlir::MLIRContext context;
  // FunctionDef -> TF Dialect
  auto expected_module =
      tensorflow::ConvertFunctionToMlir(fbody, flib_def, &context);

  if (!expected_module.ok())
    return tensorflow::errors::Internal(
        "Failed to convert function to mlir for function ", function_name.str(),
        ". Error: ", expected_module.status().error_message());

  auto module = expected_module.ConsumeValueOrDie();

  // Attach devices to the MLIR module.
  if (!devices.empty()) {
    mlir::Builder builder(module->getContext());
    module->getOperation()->setAttr("tf.devices",
                                    builder.getStrArrayAttr(devices));
  }

  // TF Dialect -> BEF
  return tensorflow::CompileTFMLIRToBEF(options, module.get(), bef_buffer);
}

Status ConvertTfMlirToBef(const TfrtCompileOptions& options,
                          mlir::ModuleOp module, tfrt::BefBuffer* bef_buffer) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

  if (options.tpu_target == TfrtTpuInfraTarget::kTpurt) {
    VLOG(1) << "Running MLIR TPU bridge for tpurt";
    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("tpu_bct_conversion_before", module);
    }

    TfrtTpuCompileOptions tpu_compile_options;
    tpu_compile_options.move_resource_gather_to_host =
        options.tpu_move_resource_gather_to_host;

    auto backward_compat_result =
        tensorflow::RunTPUBackwardCompatConversion(module, tpu_compile_options);
    if (mlir::failed(backward_compat_result)) {
      return diag_handler.Combine(
          tensorflow::errors::Internal("Failed to handle legacy TPU Ops"));
    }

    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("tpu_bct_conversion_after", module);
    }

    TF_RETURN_IF_ERROR(
        mlir::TFTPU::TPUBridge(module, /*enable_logging=*/VLOG_IS_ON(1)));
  }

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tf_dialect", module);
  }

  // Lower MLIR TF Dialect to MLIR TFRT CoreRT dialect.
  mlir::PassManager pm(module.getContext());

  tensorflow::TfrtPipelineOptions pass_options;
  if (!options.default_device.empty()) {
    pass_options.default_device = options.default_device;
  }
  if (!options.force_data_format.empty()) {
    pass_options.force_data_format = options.force_data_format;
  }

  // TODO(b/187991150): Consider only decomposing read-only resource variable
  // ops.
  pass_options.decompose_resource_ops = true;
  pass_options.enable_optimizer = options.enable_optimizer;
  pass_options.enable_native_ops = options.enable_native_ops;
  pass_options.target_tpurt =
      (options.tpu_target == TfrtTpuInfraTarget::kTpurt);
  pass_options.tpu_fuse_ops = options.tpu_fuse_ops;
  pass_options.use_tpu_host_allocator_for_inputs =
      options.use_tpu_host_allocator_for_inputs;
  pass_options.hoist_invariant_ops = options.hoist_invariant_ops;
  pass_options.func_use_fallback_tensor = true;
  pass_options.auto_fusion_oplist = options.auto_fusion_oplist;
  pass_options.auto_fusion_min_cluster_size =
      options.auto_fusion_min_cluster_size;
  pass_options.cost_threshold = options.cost_threshold;
  pass_options.upper_cost_threshold = options.upper_cost_threshold;
  pass_options.merge_inter_dependent_streams =
      options.merge_inter_dependent_streams;
  tensorflow::CreateTfExecutorToTfrtPipeline(pm, pass_options);

  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(tensorflow::errors::Internal(
        "failed to lower TF Dialect to CoreRT dialect."));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tfrt_dialect", module);
  }

  *bef_buffer =
      tfrt::ConvertMLIRToBEF(module, /*disable_optional_sections=*/true);
  if (bef_buffer->empty())
    return diag_handler.Combine(
        tensorflow::errors::Internal("failed to convert MLIR to BEF."));

  bef_buffer->shrink_to_fit();

  return Status::OK();
}

}  // namespace tensorflow
