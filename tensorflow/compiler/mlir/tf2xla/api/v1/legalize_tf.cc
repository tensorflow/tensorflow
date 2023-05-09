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
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/tpu_compile.h"
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

auto* phase2_bridge_compilation_time = tsl::monitoring::Sampler<1>::New(
    {"/tensorflow/core/tf2xla/api/v1/phase2_compilation_time",
     "The wall-clock time spent on executing graphs in milliseconds.",
     "configuration"},
    // Power of 1.5 with bucket count 45 (> 23 hours)
    {tsl::monitoring::Buckets::Exponential(1, 1.5, 45)});

// The label `status` is used to count the following events:
// MLIR bridge phase 2 was executed and the graph was processed successfully
// (fallback enabled).
constexpr char kMlirWithFallbackModeSuccess[] = "kMlirWithFallbackModeSuccess";
// MLIR bridge phase 2 compilation was failure (fallback enabled).
constexpr char kMlirWithFallbackModeFailure[] = "kMlirWithFallbackModeFailure";
// MLIR bridge phase 2 compilation was successful (manually enabled).
constexpr char kMlirModeSuccess[] = "kMlirModeSuccess";
// MLIR bridge phase 2 compilation fails (manually enabled)
constexpr char kMlirModeFailure[] = "kMlirModeFailure";
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

// Time the execution of kernels (in CPU cycles). Meant to be used as RAII.
struct CompilationTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }

  int64_t ElapsedCyclesInMilliseconds() {
    std::chrono::duration<double> duration =
        profile_utils::CpuUtils::ConvertClockCycleToTime(ElapsedCycles());

    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }
};

namespace {

bool ShouldFallbackToGraphCompiler(
    const std::variant<MlirToHloArgs, FunctionToHloArgs>& computation) {
  if (computation.index() == 1) return true;

  return std::get<0>(computation).rollout_state ==
         ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED;
}

Status CompileFromMlirToXlaHlo(
    bool enable_op_fallback,
    const std::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, llvm::StringRef device_type,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns& shape_determination_fns,
    bool use_tuple_args, XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    const std::vector<TensorShape>& arg_shapes,
    std::vector<ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  if (enable_op_fallback) {
    LOG_FIRST_N(INFO, 1)
        << "Compiling MLIR computation to XLA HLO using MLIR tf2xla bridge in "
           "the op by op fallback mode. This is Phase 2 of the TF2XLA Bridge. "
           "Old (non-MLIR) bridge may be used in case of unsupported feature "
           "or compilation failure from the MLIR bridge (full fallback mode).";
  } else {
    LOG_FIRST_N(INFO, 1)
        << "Compiling MLIR computation to XLA HLO using MLIR tf2xla bridge "
           "phase 2. Fallback to the old (non-MLIR) bridge is disabled. "
           "Op-by-op fallback is also disabled.";
  }

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  TF_RETURN_IF_ERROR(DeserializeMlirModule(std::get<0>(computation).mlir_module,
                                           &context, &mlir_module));
  if (!mlir::SetTPUInfeedLayout(mlir_module))
    return errors::Internal("Failed to set layouts attribute");

  TF_RETURN_IF_ERROR(CompileSerializedMlirToXlaHlo(
      SerializeMlirModule(mlir_module.get()), arg_shapes, device_type,
      use_tuple_args, enable_op_fallback, shape_determination_fns,
      compilation_result, custom_legalization_passes, metadata.module_name()));

  // Compute how arguments are shared across different cores.
  return tpu::GetShardingInfo(metadata, arg_shapes, shape_determination_fns,
                              arg_core_mapping, per_core_arg_shapes);
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
  XlaCompilationResult compilation_result;
  // If there are no MLIR args, compile the given function in the library.
  if (ShouldFallbackToGraphCompiler(computation)) {
    TF_RETURN_IF_ERROR(tf2xla::v0::CompileTensorflowGraphToHlo(
        computation, metadata, use_tuple_args, shape_determination_fns,
        arg_shapes, arg_core_mapping, per_core_arg_shapes, client,
        &compilation_result));
    return compilation_result;
  }

  // We could only end up here if the MLIR bridge was explicitly enabled or
  // if it was in the default/unspecified state and graph analysis in the first
  // phase has not identified unsupported features.
  // Enabling op fallback also enables whole graph fallback if op by op
  // fallback failed.
  bool enable_op_fallback =
      std::get<0>(computation).rollout_state !=
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;

  Status mlir_bridge_status = tsl::OkStatus();
  {
    CompilationTimer timer;
    std::string enabled_string = enable_op_fallback ? "enabled" : "disabled";
    const std::string kMlirBridgeFallback =
        absl::StrCat("mlir_bridge_op_fallback_", enabled_string);

    mlir_bridge_status = CompileFromMlirToXlaHlo(
        enable_op_fallback, computation, metadata, device_type,
        shape_determination_fns, use_tuple_args, &compilation_result,
        custom_legalization_passes, arg_shapes, arg_core_mapping,
        per_core_arg_shapes);

    phase2_bridge_compilation_time->GetCell(kMlirBridgeFallback)
        ->Add(timer.ElapsedCyclesInMilliseconds());
  }

  if (mlir_bridge_status.ok()) {
    if (enable_op_fallback) {
      VLOG(1) << "Successfully compiled MLIR computation to XLA HLO using MLIR "
                 "tf2xla bridge";
      mlir_second_phase_count->GetCell(kMlirWithFallbackModeSuccess)
          ->IncrementBy(1);
    } else {
      mlir_second_phase_count->GetCell(kMlirModeSuccess)->IncrementBy(1);
    }
    return compilation_result;
  } else if (!enable_op_fallback) {
    // Don't fallback to the old bridge if op-by-op fallback isn't enabled.
    mlir_second_phase_count->GetCell(kMlirModeFailure)->IncrementBy(1);
    return mlir_bridge_status;
  }

  bool filtered_graph = false;
  if (mlir_bridge_status == CompileToHloGraphAnalysisFailedError()) {
    VLOG(1) << "Filtered out MLIR computation to XLA HLO using MLIR tf2xla "
               "bridge. Falling back to old (non-MLIR) bridge.";
    filtered_graph = true;
  } else {
    mlir_second_phase_count->GetCell(kMlirWithFallbackModeFailure)
        ->IncrementBy(1);

    VLOG(1) << "Failed to compile MLIR computation to XLA HLO using MLIR "
               "tf2xla bridge. Falling back to old (non-MLIR) bridge. MLIR "
               "bridge compilation status: "
            << mlir_bridge_status;
  }

  Status old_bridge_status = tf2xla::v0::CompileTensorflowGraphToHlo(
      computation, metadata, use_tuple_args, shape_determination_fns,
      arg_shapes, arg_core_mapping, per_core_arg_shapes, client,
      &compilation_result);

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
    return old_bridge_status;
  }

  if (filtered_graph) {
    mlir_second_phase_count->GetCell(kOldBridgeMlirFilteredSuccess)
        ->IncrementBy(1);
  } else {
    mlir_second_phase_count->GetCell(kOldBridgeWithFallbackModeSuccess)
        ->IncrementBy(1);
  }
  return compilation_result;
}

};  // namespace v1
};  // namespace tf2xla
};  // namespace tensorflow
