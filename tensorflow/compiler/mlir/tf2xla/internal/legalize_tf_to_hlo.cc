/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_to_hlo.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/set_tpu_infeed_layout.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_tf_graph.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_mlir.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/compile_only_client.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/shape.h"
#include "xla/tsl/framework/device_type.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {

using metrics::IncrementTfMlirBridgeSecondPhaseCounter;
using metrics::MlirBridgeSecondPhaseMetric;
using tpu::MlirToHloArgs;

absl::Status CheckAndIncrementCounter(absl::Status status,
                                      MlirBridgeSecondPhaseMetric metric) {
  if (!status.ok()) {
    IncrementTfMlirBridgeSecondPhaseCounter(metric);
    return status;
  }
  return absl::OkStatus();
}

absl::StatusOr<XlaCompilationResult> LegalizeTfToHlo(
    mlir::ModuleOp mlir_module_op, const tpu::TPUCompileMetadataProto& metadata,
    bool use_tuple_args, llvm::StringRef device_type,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    xla::CompileOnlyClient* client, XlaCompilationResult* compilation_result) {
  if (!mlir::SetTPUInfeedLayout(mlir_module_op))
    return absl::AbortedError("Failed to set layouts attribute");

  absl::Status mlir_compilation = internal::CompileFromMlirToXlaHlo(
      /*lower_to_xla_hlo=*/false, mlir_module_op, metadata, device_type,
      shape_determination_fns, use_tuple_args, compilation_result,
      custom_legalization_passes, arg_shapes, arg_core_mapping,
      per_core_arg_shapes);

  TF_RETURN_IF_ERROR(CheckAndIncrementCounter(
      mlir_compilation, MlirBridgeSecondPhaseMetric::kMlirCombinedMlirFailure));

  IncrementTfMlirBridgeSecondPhaseCounter(
      MlirBridgeSecondPhaseMetric::kMlirCombinedMlirSuccess);

  MlirToHloArgs mlir_to_hlo_args;
  mlir_to_hlo_args.mlir_module_op = mlir_module_op;
  absl::Status old_bridge_status = v1::CompileTensorflowGraphToHlo(
      mlir_to_hlo_args, metadata, use_tuple_args, shape_determination_fns,
      arg_shapes, tsl::DeviceType(device_type.str()), arg_core_mapping,
      per_core_arg_shapes, client, compilation_result);

  TF_RETURN_IF_ERROR(CheckAndIncrementCounter(
      old_bridge_status, MlirBridgeSecondPhaseMetric::kMlirCombinedOldFailure));

  IncrementTfMlirBridgeSecondPhaseCounter(
      MlirBridgeSecondPhaseMetric::kMlirCombinedOldSuccess);

  return *compilation_result;
}
}  // namespace

absl::StatusOr<XlaCompilationResult> LegalizeTfToHlo(
    const tpu::MlirToHloArgs& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    llvm::StringRef device_type,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    xla::CompileOnlyClient* client, XlaCompilationResult* compilation_result) {
  LOG_FIRST_N(INFO, 1) << "Compiling MLIR computation to XLA HLO using the "
                          "Combined MLIR Tf2Xla Bridge.";

  if (computation.mlir_module_op.has_value()) {
    return LegalizeTfToHlo(computation.mlir_module_op.value(), metadata,
                           use_tuple_args, device_type, shape_determination_fns,
                           arg_shapes, arg_core_mapping, per_core_arg_shapes,
                           custom_legalization_passes, client,
                           compilation_result);
  }

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  absl::Status deserialization_status = tensorflow::DeserializeMlirModule(
      computation.mlir_module, &context, &mlir_module);

  TF_RETURN_IF_ERROR(CheckAndIncrementCounter(
      deserialization_status,
      MlirBridgeSecondPhaseMetric::kMlirCombinedMlirFailure));

  return LegalizeTfToHlo(mlir_module.get(), metadata, use_tuple_args,
                         device_type, shape_determination_fns, arg_shapes,
                         arg_core_mapping, per_core_arg_shapes,
                         custom_legalization_passes, client,
                         compilation_result);
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
