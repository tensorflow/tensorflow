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

#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_tf_graph.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
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
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/tpu_compile.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v0 {

using ::tensorflow::tpu::FunctionToHloArgs;
using ::tensorflow::tpu::GuaranteedConsts;
using ::tensorflow::tpu::MlirToHloArgs;
using ::tensorflow::tpu::ShardingAndIndex;

auto* mlir_second_phase_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/v0/mlir_second_phase_count" /* metric name */,
    "Counts the number of graphs that were analyzed prior deciding whether "
    "the MLIR or the old bridge will be used" /* metric description */,
    "status" /* metric label */);

auto* phase2_bridge_compilation_time = tsl::monitoring::Sampler<1>::New(
    {"/tensorflow/core/tf2xla/v0/bridge_phase_2_compilation_time",
     "The wall-clock time spent on executing graphs in milliseconds.",
     "configuration"},
    // Power of 1.5 with bucket count 45 (> 23 hours)
    {tsl::monitoring::Buckets::Exponential(1, 1.5, 45)});

// There were no MLIR ops so the old bridge was called successfully.
constexpr char kOldBridgeNoMlirSuccess[] = "kOldBridgeNoMlirSuccess";
// There were no MLIR ops so the old bridge was called but it failed.
constexpr char kOldBridgeNoMlirFailure[] = "kOldBridgeNoMlirFailure";

namespace {

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

// Populates input_output_alias field in the HLO Module proto.
Status PopulateInputOutputAliasing(
    mlir::func::FuncOp main_fn,
    XlaCompiler::CompilationResult* compilation_result, bool use_tuple_args) {
  constexpr char kAliasingAttr[] = "tf.aliasing_output";
  llvm::SmallDenseMap<unsigned, unsigned> output_to_input_alias;
  unsigned num_arguments = main_fn.getNumArguments();
  for (unsigned arg_index = 0; arg_index < num_arguments; ++arg_index) {
    if (auto aliasing_output = main_fn.getArgAttrOfType<mlir::IntegerAttr>(
            arg_index, kAliasingAttr))
      output_to_input_alias[aliasing_output.getInt()] = arg_index;
  }

  if (output_to_input_alias.empty()) return OkStatus();

  xla::HloModuleProto* module_proto =
      compilation_result->computation->mutable_proto();
  StatusOr<xla::ProgramShape> program_shape_or_status =
      compilation_result->computation->GetProgramShape();
  TF_RET_CHECK(program_shape_or_status.ok());

  xla::ProgramShape& program_shape = program_shape_or_status.value();
  if (!program_shape.result().IsTuple())
    return errors::Internal("Expect result to have tuple shape");

  xla::HloInputOutputAliasConfig config(program_shape.result());
  for (auto alias : output_to_input_alias) {
    if (use_tuple_args) {
      TF_RETURN_IF_ERROR(config.SetUpAlias(
          xla::ShapeIndex({alias.first}), 0, xla::ShapeIndex({alias.second}),
          xla::HloInputOutputAliasConfig::AliasKind::kMayAlias));
    } else {
      TF_RETURN_IF_ERROR(config.SetUpAlias(
          xla::ShapeIndex({alias.first}), alias.second, xla::ShapeIndex({}),
          xla::HloInputOutputAliasConfig::AliasKind::kMayAlias));
    }
  }
  *module_proto->mutable_input_output_alias() = config.ToProto();
  return OkStatus();
}

// Transforms the given module to be suitable for export to TensorFlow GraphDef
// and then exports all functions to the given library.
Status PrepareAndExportToLibrary(mlir::ModuleOp module,
                                 FunctionLibraryDefinition* flib_def) {
  // Pass pipeline is defined here instead of leveraging the phase one export
  // pipeline because only the functional to executor dialect conversion and
  // breakup islands passes are common between the export pipeline and here.
  // Reconsider this if there is more commonality in the future with more
  // passes.
  mlir::PassManager manager(module.getContext());
  applyTensorflowAndCLOptions(manager);
  manager.addPass(mlir::TF::CreatePrepareTpuComputationForTfExportPass());
  manager.addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());
  manager.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  manager.addPass(mlir::CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  if (failed(manager.run(module))) return diag_handler.ConsumeStatus();

  GraphExportConfig config;
  config.export_entry_func_to_flib = true;
  return tensorflow::ConvertMlirToGraph(module, config, /*graph=*/nullptr,
                                        flib_def);
}

}  // namespace

tsl::Status CompileTensorflowGraphToHlo(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns
        shape_determination_funcs,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client,
    XlaCompiler::CompilationResult* compilation_result) {
  LOG_FIRST_N(INFO, 1) << "Compiling MLIR computation to XLA HLO using the "
                          "old (non-MLIR) tf2xla bridge";

  *compilation_result = {};
  bool has_mlir = computation.index() == 0;

  std::string mlir_string = has_mlir ? "has_mlir" : "has_function_to_hlo";
  const std::string kMlirBridgeFallback =
      absl::StrCat("graph_old_bridge_", mlir_string);
  CompilationTimer timer;

  if (!has_mlir) {
    FunctionToHloArgs function_computation = std::get<1>(computation);
    Status comp_status = CompileTFFunctionToHlo(
        *function_computation.flib_def, function_computation.graph_def_version,
        shape_determination_funcs, arg_shapes,
        function_computation.guaranteed_constants,
        *function_computation.function, metadata, client, arg_core_mapping,
        per_core_arg_shapes, use_tuple_args, compilation_result);
    if (comp_status.ok()) {
      mlir_second_phase_count->GetCell(kOldBridgeNoMlirSuccess)->IncrementBy(1);
    } else {
      mlir_second_phase_count->GetCell(kOldBridgeNoMlirFailure)->IncrementBy(1);
    }

    phase2_bridge_compilation_time->GetCell(kMlirBridgeFallback)
        ->Add(timer.ElapsedCyclesInMilliseconds());
    return comp_status;
  }

  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  TF_RETURN_IF_ERROR(DeserializeMlirModule(std::get<0>(computation).mlir_module,
                                           &context, &mlir_module));
  if (!mlir::SetTPUInfeedLayout(mlir_module))
    return errors::Internal("Failed to set layouts attribute");

  if (VLOG_IS_ON(2)) {
    tensorflow::DumpMlirOpToFile("legalize_with_old_bridge", mlir_module.get());
  }
  constexpr char kEntryFuncName[] = "main";
  auto main_fn = mlir_module->lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!main_fn) {
    return errors::Internal(
        "TPU compile op requires module with a entry function main");
  }

  // Export functions to the library.
  auto flib_def = std::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), FunctionDefLibrary());
  TF_RETURN_IF_ERROR(PrepareAndExportToLibrary(*mlir_module, flib_def.get()));

  if (VLOG_IS_ON(2)) {
    tensorflow::DumpMlirOpToFile("legalize_with_old_bridge_post_transform",
                                 mlir_module.get());
  }
  VersionDef versions;
  if (mlir::failed(ExtractTfVersions(*mlir_module, &versions))) {
    return errors::Internal(
        "module attribute in _TPUCompileMlir op is missing tf versions.");
  }

  NameAttrList func;
  func.set_name(kEntryFuncName);
  GuaranteedConsts consts;

  *compilation_result = {};

  TF_RETURN_IF_ERROR(CompileTFFunctionToHlo(
      *flib_def, versions.producer(), shape_determination_funcs, arg_shapes,
      consts, func, metadata, client, arg_core_mapping, per_core_arg_shapes,
      use_tuple_args, compilation_result));

  phase2_bridge_compilation_time->GetCell(kMlirBridgeFallback)
      ->Add(timer.ElapsedCyclesInMilliseconds());

  return PopulateInputOutputAliasing(main_fn, compilation_result,
                                     use_tuple_args);
}

};  // namespace v0
};  // namespace tf2xla
};  // namespace tensorflow
