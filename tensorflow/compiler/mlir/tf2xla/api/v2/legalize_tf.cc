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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/legalize_tf.h"

#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_tf_graph.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_mlir.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_to_hlo.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tsl/platform/error_logging.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

using metrics::IncrementTfMlirBridgeSecondPhaseCounter;
using metrics::MlirBridgeSecondPhaseMetric;
using tpu::FunctionToHloArgs;
using tpu::MlirToHloArgs;
using tpu::ShardingAndIndex;

// Name of component for error logging. This name is fixed and required to
// enable logging.
constexpr char kBridgeComponent[] = "TFXLABridge";

namespace {

bool ShouldFallbackToGraphCompiler(
    const std::variant<MlirToHloArgs, FunctionToHloArgs>& computation) {
  if (computation.index() == 1) return true;

  return std::get<0>(computation).rollout_state ==
         ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED;
}

Status DumpHloCompilationResult(std::string_view name,
                                XlaCompilationResult* compilation_result) {
  if (VLOG_IS_ON(2)) {
    TF_ASSIGN_OR_RETURN(
        auto hlo_module_config,
        xla::HloModule::CreateModuleConfigFromProto(
            compilation_result->computation->proto(), xla::DebugOptions()));

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::HloModule> hlo_module,
        xla::HloModule::CreateFromProto(
            compilation_result->computation->proto(), hlo_module_config));

    std::string all_computations;
    for (auto computation : hlo_module->computations()) {
      all_computations += computation->ToString() + "\n\n";
    }

    tensorflow::DumpRawStringToFile(name, all_computations);
  }

  return OkStatus();
}

}  // namespace

tsl::StatusOr<tensorflow::XlaCompilationResult> LegalizeMlirToHlo(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    llvm::StringRef device_type,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client) {
  auto compilation_result = std::make_unique<XlaCompilationResult>();

  // If there are no MLIR args, compile the given function in the library.
  if (ShouldFallbackToGraphCompiler(computation)) {
    TF_RETURN_IF_ERROR(tf2xla::v1::CompileTensorflowGraphToHlo(
        computation, metadata, use_tuple_args, shape_determination_fns,
        arg_shapes, arg_core_mapping, per_core_arg_shapes, client,
        compilation_result.get()));

    DumpHloCompilationResult("legalize_tf_fallback", compilation_result.get())
        .IgnoreError();

    return *compilation_result;
  }

  auto mlir_bridge_status = internal::LegalizeWithMlirBridge(
      std::get<0>(computation), metadata, use_tuple_args, device_type,
      shape_determination_fns, arg_shapes, arg_core_mapping,
      per_core_arg_shapes, custom_legalization_passes,
      compilation_result.get());

  if (mlir_bridge_status.ok()) {
    VLOG(1) << "Successfully compiled MLIR computation to XLA HLO using MLIR "
               "tf2xla bridge";
    IncrementTfMlirBridgeSecondPhaseCounter(
        MlirBridgeSecondPhaseMetric::kMlirWithFallbackModeSuccess);

    DumpHloCompilationResult("legalize_tf_mlir_bridge",
                             compilation_result.get())
        .IgnoreError();

    return *compilation_result;
  } else if (mlir_bridge_status.status() ==
             CompileToHloGraphAnalysisFailedError()) {
    VLOG(1) << "Filtered out MLIR computation to XLA HLO using MLIR tf2xla "
               "bridge. Falling back to Combined Bridge.";
  } else {
    VLOG(1) << "Failed to compile MLIR computation to XLA HLO using "
               "MLIR Bridge. Falling back to Combined Bridge.";
    tsl::error_logging::Log(kBridgeComponent, "TFXLA_API_V2_PHASE2_MLIR_BRIDGE",
                            mlir_bridge_status.status().ToString())
        .IgnoreError();
    IncrementTfMlirBridgeSecondPhaseCounter(
        MlirBridgeSecondPhaseMetric::kMlirWithFallbackModeFailure);
  }

  auto combined_bridge_status = internal::LegalizeTfToHlo(
      std::get<0>(computation), metadata, use_tuple_args, device_type,
      shape_determination_fns, arg_shapes, arg_core_mapping,
      per_core_arg_shapes, custom_legalization_passes, client,
      compilation_result.get());

  if (combined_bridge_status.ok()) {
    VLOG(1) << "Successfully compiled MLIR computation to XLA HLO using "
               "Combined MLIR and XlaBuilder Bridge.";

    DumpHloCompilationResult("legalize_tf_combined_bridge",
                             compilation_result.get())
        .IgnoreError();

    return *compilation_result;
  }

  VLOG(1) << "Failed to compile MLIR computation to XLA HLO using "
             "Combined MLIR and XlaBuilder Bridge. Could not generate HLO.";
  tsl::error_logging::Log(kBridgeComponent, "TFXLA_API_V2_COMBINED_BRIDGE",
                          combined_bridge_status.status().ToString())
      .IgnoreError();

  return combined_bridge_status.status();
}

};  // namespace v2
};  // namespace tf2xla
};  // namespace tensorflow
