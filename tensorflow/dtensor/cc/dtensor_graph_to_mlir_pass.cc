/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/dtensor_graph_to_mlir_pass.h"

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_mlir_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {

DTensorMlirPassRunner::DTensorMlirPassRunner()
    : pass_manager_(&context_), logging_enabled_(false) {
  logging_enabled_ = dtensor::MaybeEnableLogging(&pass_manager_);
  if (logging_enabled_) pass_manager_.getContext()->enableMultithreading();

  // TODO(hinsu, hongjunchoi): Figure out a better place to explicitly enable
  // the MLIR bridge.
  // Explicitly enable MLIR bridge as DTensor introduces some ops like
  // XlaAllReduce are only supported in MLIR.
  GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge =
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;

  // Creates a pipeline that include each DTensor related passes.
  mlir::TF::StandardPipelineOptions pipeline_options;
  dtensor::CreateDTensorMLIRPass(pipeline_options, &pass_manager_);
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
DTensorMlirPassRunner::ImportGraphToMlir(
    const DeviceSet& device_set, bool is_func,
    const FunctionLibraryDefinition& flib_def, const Graph& graph,
    Fprint128 cache_key) {
  GraphDebugInfo debug_info;
  GraphImportConfig import_config;
  import_config.graph_as_function = true;
  // DTensor relies on importing with shape_inference to work properly ATM.
  // Make it explicit so that we're not affected by potential flipping of the
  // flag.
  import_config.enable_shape_inference = true;
  // Graph pruning will prune away an op (may be side effecting) if the op is
  // not reachable from a fetch/result or target/control ret. With how the entry
  // function/Graph is created, it is possible if the op has no data results. To
  // make sure this op does not get pruned away, the op is defined as a
  // target/control ret.
  import_config.control_outputs = {"eager_operation"};

  // Imports GraphDef to TF MLIR.
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_ref =
      ConvertGraphToMlir(graph, debug_info, flib_def, import_config, &context_);

  // Adds DTensor attributes to ModuleOp.
  mlir::ModuleOp module = module_ref.value().get();
  AddDevicesToOp(module, &device_set);

  // Tag the module for logging or not depending on flag.
  if (!is_func && !dtensor::LogOpByOp())
    module->setAttr(dtensor::kDoNotLog, mlir::UnitAttr::get(&context_));

  // Set the cache key for the module as an attribute. This attribute will be
  // used to rename all private functions in the module (by appending the
  // cache key) so they have unique names.
  module->setAttr(
      dtensor::kCacheKey,
      mlir::StringAttr::get(&context_, absl::StrCat("_", cache_key.low64, "_",
                                                    cache_key.high64)));

  return module_ref;
}

Status DTensorMlirPassRunner::Run(mlir::ModuleOp module) {
  // Executes and collects results from the passes.
  mlir::StatusScopedDiagnosticHandler diag_handler(&context_);

  if (logging_enabled_ && !module->hasAttr(dtensor::kDoNotLog))
    pass_manager_.getContext()->disableMultithreading();
  mlir::LogicalResult result = pass_manager_.run(module);
  (void)result;
  TF_RETURN_IF_ERROR(diag_handler.ConsumeStatus());

  if (logging_enabled_) pass_manager_.getContext()->enableMultithreading();
  return OkStatus();
}

}  // namespace tensorflow
