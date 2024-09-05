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
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_tf_graph.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/compilation_timer.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_mlir.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/compile_only_client.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

using metrics::IncrementTfMlirBridgeSecondPhaseCounter;
using metrics::MlirBridgeSecondPhaseMetric;
using tpu::MlirToHloArgs;

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

  absl::StatusOr<std::string> mlir_compilation =
      internal::CompileFromMlirToXlaHlo(
          /*lower_to_xla_hlo=*/false, computation, metadata, device_type,
          shape_determination_fns, use_tuple_args, compilation_result,
          custom_legalization_passes, arg_shapes, arg_core_mapping,
          per_core_arg_shapes);

  if (!mlir_compilation.ok()) {
    IncrementTfMlirBridgeSecondPhaseCounter(
        MlirBridgeSecondPhaseMetric::kMlirCombinedMlirFailure);
    return mlir_compilation.status();
  }

  IncrementTfMlirBridgeSecondPhaseCounter(
      MlirBridgeSecondPhaseMetric::kMlirCombinedMlirSuccess);

  Status old_bridge_status = v1::CompileTensorflowGraphToHlo(
      MlirToHloArgs{mlir_compilation.value()}, metadata, use_tuple_args,
      shape_determination_fns, arg_shapes, arg_core_mapping,
      per_core_arg_shapes, client, compilation_result);

  if (!old_bridge_status.ok()) {
    IncrementTfMlirBridgeSecondPhaseCounter(
        MlirBridgeSecondPhaseMetric::kMlirCombinedOldFailure);
    return old_bridge_status;
  }
  IncrementTfMlirBridgeSecondPhaseCounter(
      MlirBridgeSecondPhaseMetric::kMlirCombinedOldSuccess);

  return *compilation_result;
}

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow
