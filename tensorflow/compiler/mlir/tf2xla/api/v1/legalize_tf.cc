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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/legalize_tf.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/variant.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/set_tpu_infeed_layout.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_tf_graph.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/legalize_tf_mlir.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/tpu_compile.h"
#include "tensorflow/tsl/platform/error_logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

using tpu::FunctionToHloArgs;
using tpu::MlirToHloArgs;
using tpu::ShardingAndIndex;

auto* mlir_second_phase_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/api/v1/phase2_compilation_status" /*metric_name*/,
    "Counts the number of graphs that were analyzed prior deciding whether "
    "the MLIR or the old bridge will be used" /* metric description */,
    "status" /* metric label */);

// The label `status` is used to count the following events:
// MLIR bridge phase 2 was executed and the graph was processed successfully
// (fallback enabled).
constexpr char kMlirWithFallbackModeSuccess[] = "kMlirWithFallbackModeSuccess";
// MLIR bridge phase 2 compilation was failure (fallback enabled).
constexpr char kMlirWithFallbackModeFailure[] = "kMlirWithFallbackModeFailure";
// Old bridge compilation was run successfully (was run because MLIR bridge
// could not process the graph).
constexpr char kOldBridgeMlirFilteredSuccess[] =
    "kOldBridgeMlirFilteredSuccess";
// Old bridge failed (was run b/c MLIR bridge could not process the graph).
constexpr char kOldBridgeMlirFilteredFailure[] =
    "kOldBridgeMlirFilteredFailure";
// Old bridge compilation was successfully run after MLIR bridge ran and failed.
constexpr char kOldBridgeWithFallbackModeSuccess[] =
    "kOldBridgeWithFallbackModeSuccess";
// Old Bridge failed in fallback (was run because MLIR bridge failed first).
constexpr char kOldBridgeWithFallbackModeFailure[] =
    "kOldBridgeWithFallbackModeFailure";
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
    TF_RETURN_IF_ERROR(tf2xla::v0::CompileTensorflowGraphToHlo(
        computation, metadata, use_tuple_args, shape_determination_fns,
        arg_shapes, arg_core_mapping, per_core_arg_shapes, client,
        compilation_result.get()));
    return *compilation_result;
  }

  auto mlir_bridge_status = internal::LegalizeWithMlirBridge(
      std::get<0>(computation), metadata, use_tuple_args, device_type,
      shape_determination_fns, arg_shapes, arg_core_mapping,
      per_core_arg_shapes, custom_legalization_passes,
      compilation_result.get());

  if (mlir_bridge_status.ok()) {
    mlir_second_phase_count->GetCell(kMlirWithFallbackModeSuccess)
        ->IncrementBy(1);
    return *compilation_result;
  }

  bool filtered_graph = false;
  if (mlir_bridge_status.status() == CompileToHloGraphAnalysisFailedError()) {
    VLOG(1) << "Filtered out MLIR computation to XLA HLO using MLIR tf2xla "
               "bridge. Falling back to old (non-MLIR) bridge.";
    filtered_graph = true;
  } else {
    mlir_second_phase_count->GetCell(kMlirWithFallbackModeFailure)
        ->IncrementBy(1);
    VLOG(1) << "Failed to compile MLIR computation to XLA HLO using MLIR "
               "tf2xla bridge. Falling back to old (non-MLIR) bridge. MLIR "
               "bridge compilation status: "
            << mlir_bridge_status.status();
  }

  Status old_bridge_status = tf2xla::v0::CompileTensorflowGraphToHlo(
      computation, metadata, use_tuple_args, shape_determination_fns,
      arg_shapes, arg_core_mapping, per_core_arg_shapes, client,
      compilation_result.get());

  // Record filter/failure stats only if the old bridge succeeds. This removes
  // noise from invalid inputs.
  if (!old_bridge_status.ok()) {
    // If the old bridge failed for this input as well. Mark the input as
    // invalid. This might be incorrect in case of old bridge bugs but that
    // should be rare.
    if (filtered_graph) {
      mlir_second_phase_count->GetCell(kOldBridgeMlirFilteredFailure)
          ->IncrementBy(1);
    } else {
      mlir_second_phase_count->GetCell(kOldBridgeWithFallbackModeFailure)
          ->IncrementBy(1);
    }
    if (!old_bridge_status.ok()) {
      tsl::error_logging::Log(kBridgeComponent, "TFXLA_API_V1_OLD_BRIDGE",
                              mlir_bridge_status.status().ToString())
          .IgnoreError();
    }
    return old_bridge_status;
  }

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

    tensorflow::DumpRawStringToFile("legalize_tf_fallback_hlo",
                                    all_computations);
  }

  if (filtered_graph) {
    mlir_second_phase_count->GetCell(kOldBridgeMlirFilteredSuccess)
        ->IncrementBy(1);
  } else {
    mlir_second_phase_count->GetCell(kOldBridgeWithFallbackModeSuccess)
        ->IncrementBy(1);
  }
  return *compilation_result;
}

};  // namespace v1
};  // namespace tf2xla
};  // namespace tensorflow
