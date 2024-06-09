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
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "llvm/ADT/ScopeExit.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_tf_graph.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/compilation_timer.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_mlir.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_to_hlo.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tsl/lib/monitoring/sampler.h"
#include "tsl/platform/error_logging.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

using tpu::FunctionToHloArgs;
using tpu::MlirToHloArgs;
using tpu::ShardingAndIndex;

auto* phase2_bridge_compilation_time = tsl::monitoring::Sampler<1>::New(
    {"/tensorflow/core/tf2xla/api/v2/phase2_compilation_time",
     "The wall-clock time spent on executing graphs in milliseconds.",
     "configuration"},
    // Power of 1.5 with bucket count 45 (> 23 hours)
    {tsl::monitoring::Buckets::Exponential(1, 1.5, 45)});

// Name of component for error logging. This name is fixed and required to
// enable logging.
constexpr char kBridgeComponent[] = "TFXLABridge";
constexpr char kFullBridge[] = "full_bridge";

namespace {

bool ShouldFallbackToGraphCompiler(
    const std::variant<MlirToHloArgs, FunctionToHloArgs>& computation) {
  if (computation.index() == 1) return true;

  return std::get<0>(computation).rollout_state ==
         ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED;
}

void DumpComputationInput(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>
        computation) {
  if (!VLOG_IS_ON(2)) {
    return;
  }

  switch (computation.index()) {
    case 0:
      VLOG(2) << "LegalizeMlirToHlo with MLIR computation input: "
              << std::get<0>(computation).mlir_module;
      break;
    case 1: {
      auto input = std::get<1>(computation);
      Graph g(input.flib_def);
      VLOG(2) << "LegalizeMlirToHlo with FLIB computation input: "
              << DumpGraphToFile(
                     absl::StrCat("legalize_mlir_hlo_computation_input_",
                                  input.function->name()),
                     g, std::get<1>(computation).flib_def);
    } break;
    default:
      VLOG(2) << "LegalizeMlirToHlo computation input: unknown";
      break;
  }
}

Status DumpHloCompilationResult(std::string_view name,
                                XlaCompilationResult* compilation_result) {
  if (!VLOG_IS_ON(2) &&
      !DEBUG_DATA_DUMPER()->ShouldDump(std::string(name), kDebugGroupMain)) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      auto hlo_module_config,
      xla::HloModule::CreateModuleConfigFromProto(
          compilation_result->computation->proto(), xla::DebugOptions()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> hlo_module,
      xla::HloModule::CreateFromProto(compilation_result->computation->proto(),
                                      hlo_module_config));

  std::string all_computations;
  for (auto computation : hlo_module->computations()) {
    all_computations += computation->ToString() + "\n\n";
  }

  tensorflow::DumpRawStringToFile(name, all_computations);

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<tensorflow::XlaCompilationResult> LegalizeMlirToHlo(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    llvm::StringRef device_type,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client) {
  CompilationTimer timer;
  auto record_time = llvm::make_scope_exit([&timer] {
    phase2_bridge_compilation_time->GetCell(kFullBridge)
        ->Add(timer.ElapsedCyclesInMilliseconds());
  });

  auto compilation_result = std::make_unique<XlaCompilationResult>();

  DumpComputationInput(computation);

  // If there are no MLIR args, compile the given function in the library.
  if (ShouldFallbackToGraphCompiler(computation)) {
    TF_RETURN_IF_ERROR(tf2xla::v1::CompileTensorflowGraphToHlo(
        computation, metadata, use_tuple_args, shape_determination_fns,
        arg_shapes, arg_core_mapping, per_core_arg_shapes, client,
        compilation_result.get()));

    DumpHloCompilationResult("legalize_tf_fallback.hlo",
                             compilation_result.get())
        .IgnoreError();
    return *compilation_result;
  }

  auto combined_bridge_status = internal::LegalizeTfToHlo(
      std::get<0>(computation), metadata, use_tuple_args, device_type,
      shape_determination_fns, arg_shapes, arg_core_mapping,
      per_core_arg_shapes, custom_legalization_passes, client,
      compilation_result.get());

  if (combined_bridge_status.ok()) {
    VLOG(1) << "Successfully compiled MLIR computation to XLA HLO using "
               "Combined MLIR and XlaBuilder Bridge.";

    DumpHloCompilationResult("legalize_tf_combined_bridge.hlo",
                             compilation_result.get())
        .IgnoreError();
    return *compilation_result;
  }

  return combined_bridge_status.status();
}

};  // namespace v2
};  // namespace tf2xla
};  // namespace tensorflow
