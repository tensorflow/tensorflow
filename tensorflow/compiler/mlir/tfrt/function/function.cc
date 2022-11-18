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

#include "tensorflow/compiler/mlir/tfrt/function/function.h"

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime

namespace tensorflow {

Status CompileTFMLIRToBEF(const TfrtFunctionCompileOptions& options,
                          mlir::ModuleOp module, tfrt::BefBuffer* bef_buffer) {
  mlir::OpPrintingFlags print_flags;
  print_flags.elideLargeElementsAttrs();

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Input TF Executor dialect:";
    DumpMlirOpToFile("tf_to_tfrt_tf_executor_dialect", module);
  }

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

  // Lower MLIR TF Dialect to MLIR TFRT CoreRT dialect.
  mlir::PassManager pm(module.getContext());
  tensorflow::applyTensorflowAndCLOptions(pm);

  tensorflow::TfrtPipelineOptions pass_options;
  if (!options.default_device.empty()) {
    pass_options.default_device = options.default_device;
  }
  if (!options.force_data_format.empty()) {
    pass_options.force_data_format = options.force_data_format;
  }
  // TODO(tfrt-devs): Current MaxPoolingOp only supports NHWC on device type
  // CPU. Enable this layout optimization after we introduce TFRT native ops
  // for training.
  if (absl::StrContains(pass_options.default_device, "CPU")) {
    pass_options.skip_fold_transpose_in_ops = true;
  }
  pass_options.enable_optimizer = options.enable_optimizer;
  // Use TFRT TPU OpKernel for training.
  pass_options.target_tpurt = false;
  pass_options.tpu_use_core_selector = options.tpu_use_core_selector;
  pass_options.tpu_use_bundled_transfer = options.tpu_use_bundled_transfer;
  pass_options.tpu_lower_to_fallback = options.tpu_lower_to_fallback;
  pass_options.tpu_fuse_ops = options.tpu_fuse_ops;
  pass_options.tpu_transfer_result_to_host =
      options.tpu_transfer_result_to_host;
  tensorflow::CreateTfExecutorToTfrtPipeline(pm, pass_options);

  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(tensorflow::errors::Internal(
        "failed to lower TF Dialect to CoreRT dialect."));

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "TFRT dialect: ";
    DumpMlirOpToFile("tf_to_tfrt_tfrt_dialect", module);
  }

  *bef_buffer =
      tfrt::ConvertMLIRToBEF(module, /* disable_optional_sections = */ true);
  if (bef_buffer->empty())
    return diag_handler.Combine(
        tensorflow::errors::Internal("failed to convert MLIR to BEF."));

  return OkStatus();
}

}  // namespace tensorflow
