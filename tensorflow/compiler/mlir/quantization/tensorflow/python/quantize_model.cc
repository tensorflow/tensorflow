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
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/save_variables.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace quantization {
namespace internal {
namespace {

// Options when running passes for exporting an MLIR ModuleOp.
struct ExportOptions {
  // If set to `true`, it runs `DuplicateShapeDeterminingConstantsPass` before
  // lowering to tf_executor dialect.
  bool duplicate_shape_determining_constants = true;

  // If set to `true`, unfreezes constants into variables and saves them to a
  // checkpoint file. Setting this to `true` is an experimental feature that has
  // no stability guarantees.
  bool unfreeze_constants = false;

  // Path to the directory where checkpoint files are saved.
  absl::string_view checkpoint_dir = "";

  // Name used to identify the ModuleOp this is exporting. Only used for
  // debugging and does not modify the behavior of the export.
  absl::string_view debug_name = "tf_quant";
};

// Add passes for transforming the MLIR module op so that it can be exported
// back to GraphDef. Roughly, this consists of:
//   1) Inserting the @main function, which will become the main Graph.
//   2) Duplicates shape-determining constants.
//   3) Converting TF dialect -> tf_executor dialect.
//   4) Adding initializer function's ops into @main function for correct
//      resource initialization when loading the exported model.
//
// Duplicating shape-determining constants is required to place constants that
// affect the shape of a tensor to be placed in the TPU graph instead of in the
// CPU graph, when the graph gets converted for TPU inference. This allows these
// constants to be known at XLA compilation time.
void AddExportPasses(const bool duplicate_shape_determining_constants,
                     mlir::PassManager &pm) {
  if (duplicate_shape_determining_constants) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::quant::CreateDuplicateShapeDeterminingConstantsPass());
  }

  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());
  pm.addPass(mlir::quant::CreateMergeInitializerFunctionOpsToMainPass());
}

// Returns the name of the initializer node from a set of control return nodes.
// Returns an empty string if no initializer node exists. This assumes that
// there is only one node for initialization.
std::string GetInitNodeName(
    const absl::flat_hash_set<Node *> &control_ret_nodes) {
  for (Node *control_ret_node : control_ret_nodes) {
    if (absl::StrContains(control_ret_node->name(), kInitOpNamePrefix)) {
      VLOG(1) << "Init node found: " << control_ret_node->name();
      return control_ret_node->name();
    }
  }
  return "";
}

[[nodiscard]] ExportedModel CreateExportedModel(
    GraphDef &&graph_def, const absl::string_view init_node_name,
    const absl::string_view checkpoint_dir) {
  ExportedModel exported_model{};
  *exported_model.mutable_graph_def() = graph_def;
  exported_model.set_init_node_name(std::string(init_node_name));
  exported_model.set_checkpoint_dir(std::string(checkpoint_dir));

  return exported_model;
}

// Converts MLIR ModuleOp to ExportedModel. Returns InternalError status
// when the conversion fails.
absl::StatusOr<ExportedModel> ConvertMlirModuleToExportedModel(
    const mlir::ModuleOp module_op, const absl::string_view checkpoint_dir) {
  const GraphExportConfig config{};
  FunctionLibraryDefinition flib_def{OpRegistry::Global(),
                                     FunctionDefLibrary()};
  std::unique_ptr<Graph> graph;
  absl::flat_hash_set<Node *> control_ret_nodes{};
  if (const auto status = ConvertMlirToGraph(module_op, config, &graph,
                                             &flib_def, &control_ret_nodes);
      !status.ok()) {
    return absl::InternalError("Failed to convert MLIR to GraphDef. " +
                               status.error_message());
  }

  GraphDef graph_def{};
  graph->ToGraphDef(&graph_def);

  return CreateExportedModel(
      std::move(graph_def), GetInitNodeName(control_ret_nodes), checkpoint_dir);
}

// Runs MLIR passes with `module_op`. The passes are added by calling
// `add_passes_func`, which is a callable receiving mlir::PassManager& as its
// only argument. `name` identifies the set of passes added by `add_passes_func`
// and is used for debugging. Changing the `name` does not modify the behavior
// of the passes.
//
// It will try to dump intermediate MLIRs if certain conditions are met. See the
// description from `MaybeEnableIrPrinting` for the details about the
// conditions.
//
// Returns a non-OK status when the pass run fails or it fails to create an MLIR
// dump file.
template <typename FuncT>
absl::Status RunPasses(const absl::string_view name, FuncT add_passes_func,
                       mlir::MLIRContext &ctx, mlir::ModuleOp module_op) {
  mlir::PassManager pm{&ctx};
  add_passes_func(pm);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler{&ctx};
  const absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> out_dump_file =
      MaybeEnableIrPrinting(pm, name);
  if (!out_dump_file.ok()) {
    return absl::InternalError(out_dump_file.status().message());
  }

  if (failed(pm.run(module_op))) {
    return absl::InternalError(
        absl::StrFormat("Failed to run pass: %s. %s", name,
                        diagnostic_handler.ConsumeStatus().error_message()));
  }

  return absl::OkStatus();
}

// Create a unique local temporary filename. It only creates the name, not the
// actual file.
absl::StatusOr<std::string> GetLocalTempFilename() {
  auto *env = Env::Default();
  std::string tmp_fname{};
  if (!env->LocalTempFilename(&tmp_fname)) {
    return absl::InternalError("Failed to create a local temp file name.");
  }

  return tmp_fname;
}

// Unfreezes constants into variables and saves them to a checkpoint files under
// `checkpoint_dir`. `checkpoint_dir` will be created within this function. It
// will return a non-OK status if it already exists or permission is denied.
// TODO(b/261652258): Make sure this works for when there are non-frozen
// variables in the model.
absl::Status UnfreezeConstantsAndSaveVariables(
    const absl::string_view checkpoint_dir, mlir::MLIRContext &ctx,
    mlir::ModuleOp module_op) {
  if (const absl::Status pass_run_status =
          RunPasses(/*name=*/kTfQuantConstantUnfreezingStepName,
                    /*add_passes_func=*/
                    [](mlir::PassManager &pm) {
                      pm.addPass(mlir::quant::CreateUnfreezeConstantsPass());
                    },
                    ctx, module_op);
      !pass_run_status.ok()) {
    return pass_run_status;
  }

  if (const tsl::Status create_dir_status =
          Env::Default()->CreateDir(std::string(checkpoint_dir));
      !create_dir_status.ok()) {
    LOG(ERROR) << "Failed to create checkpoint directory at: "
               << checkpoint_dir;
    return tsl::ToAbslStatus(create_dir_status);
  }

  return SaveVariablesToCheckpoint(checkpoint_dir, module_op);
}

// Sets up and runs the passes for exporting `module_op`. The behavior of the
// exporting passes is controlled by `export_opts`.
absl::Status RunExportPasses(const ExportOptions &export_opts,
                             mlir::MLIRContext &ctx, mlir::ModuleOp module_op) {
  if (export_opts.unfreeze_constants) {
    if (const absl::Status unfreeze_constant_status =
            UnfreezeConstantsAndSaveVariables(export_opts.checkpoint_dir, ctx,
                                              module_op);
        !unfreeze_constant_status.ok()) {
      return unfreeze_constant_status;
    }
    LOG(INFO) << "Unfrozen constants and saved variables to checkpoint file: "
              << export_opts.checkpoint_dir;
  }

  return RunPasses(
      /*name=*/export_opts.debug_name,
      /*add_passes_func=*/
      [dup_constants = export_opts.duplicate_shape_determining_constants](
          mlir::PassManager &pm) { AddExportPasses(dup_constants, pm); },
      ctx, module_op);
}

// Creates MLIRContext where the dialects required for quantization are
// registered.
mlir::MLIRContext CreateMlirContextForTfQuantization() {
  mlir::DialectRegistry registry{};
  registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect,
                  mlir::TF::TensorFlowDialect, mlir::shape::ShapeDialect,
                  mlir::quant::QuantizationDialect>();
  return mlir::MLIRContext{registry};
}

}  // namespace

absl::StatusOr<ExportedModel> QuantizeQatModel(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const Status status = PreprocessAndFreezeGraph(
      module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
  if (!status.ok()) {
    return absl::InternalError("Failed to preprocess graph: " +
                               status.error_message());
  }

  if (const absl::Status qat_status =
          RunPasses(/*name=*/kTfQuantQatStepName,
                    /*add_passes_func=*/
                    [&quantization_options](mlir::PassManager &pm) {
                      AddQuantizeQatPasses(pm, quantization_options);
                    },
                    context, *module_ref);
      !qat_status.ok()) {
    return qat_status;
  }

  const bool unfreeze_constants =
      !quantization_options.freeze_all_variables().enabled();
  const absl::StatusOr<std::string> checkpoint_dir = GetLocalTempFilename();
  if (!checkpoint_dir.ok()) {
    LOG(ERROR) << "Failed to get checkpoint directory name.";
    return checkpoint_dir.status();
  }
  const auto export_opts =
      ExportOptions{/*duplicate_shape_determining_constants=*/true,
                    unfreeze_constants, *checkpoint_dir,
                    /*debug_name=*/kTfQuantQatStepName};
  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir);
}

absl::StatusOr<ExportedModel> QuantizePtqModelPreCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const Status status = PreprocessAndFreezeGraph(
      module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
  if (!status.ok()) {
    return absl::InternalError("Failed to preprocess graph: " +
                               status.error_message());
  }

  if (const absl::Status pre_calib_pass_status = RunPasses(
          /*name=*/kTfQuantPtqPreCalibrationStepName,
          /*add_passes_func=*/
          [&quantization_options](mlir::PassManager &pm) {
            AddQuantizePtqPreCalibrationPasses(pm, quantization_options);
          },
          context, *module_ref);
      !pre_calib_pass_status.ok()) {
    return pre_calib_pass_status;
  }

  const bool unfreeze_constants =
      !quantization_options.freeze_all_variables().enabled();
  const absl::StatusOr<std::string> checkpoint_dir = GetLocalTempFilename();
  if (!checkpoint_dir.ok()) {
    return checkpoint_dir.status();
  }
  // `duplicate_shape_determining_constants = false` because the
  // resulting graph of this step is not expected to be loaded on TPU.
  const auto export_opts =
      ExportOptions{/*duplicate_shape_determining_constants=*/false,
                    unfreeze_constants, *checkpoint_dir,
                    /*debug_name=*/kTfQuantPtqPreCalibrationStepName};
  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir);
}

absl::StatusOr<ExportedModel> QuantizePtqModelPostCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/true, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  if (const absl::Status pre_calib_pass_status = RunPasses(
          /*name=*/kTfQuantPtqPostCalibrationStepName,
          /*add_passes_func=*/
          [&quantization_options](mlir::PassManager &pm) {
            AddQuantizePtqPostCalibrationPasses(pm, quantization_options);
          },
          context, *module_ref);
      !pre_calib_pass_status.ok()) {
    return pre_calib_pass_status;
  }

  const bool unfreeze_constants =
      !quantization_options.freeze_all_variables().enabled();
  const absl::StatusOr<std::string> checkpoint_dir = GetLocalTempFilename();
  if (!checkpoint_dir.ok()) {
    return checkpoint_dir.status();
  }
  const auto export_opts =
      ExportOptions{/*duplicate_shape_determining_constants=*/true,
                    unfreeze_constants, *checkpoint_dir,
                    /*debug_name=*/kTfQuantPtqPostCalibrationStepName};
  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir);
}

absl::StatusOr<ExportedModel> QuantizePtqDynamicRange(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const Status status = PreprocessAndFreezeGraph(
      module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
  if (!status.ok()) {
    return absl::InternalError("Failed to preprocess graph: " +
                               status.error_message());
  }

  if (const absl::Status ptq_dynamic_range_status = RunPasses(
          /*name=*/kTfQuantPtqDynamicRangeStepName,
          /*add_passes_func=*/
          [&quantization_options](mlir::PassManager &pm) {
            AddQuantizePtqDynamicRangePasses(pm, quantization_options);
          },
          context, *module_ref);
      !ptq_dynamic_range_status.ok()) {
    return ptq_dynamic_range_status;
  }

  const bool unfreeze_constants =
      !quantization_options.freeze_all_variables().enabled();
  const absl::StatusOr<std::string> checkpoint_dir = GetLocalTempFilename();
  if (!checkpoint_dir.ok()) {
    return checkpoint_dir.status();
  }
  const auto export_opts =
      ExportOptions{/*duplicate_shape_determining_constants=*/true,
                    unfreeze_constants, *checkpoint_dir,
                    /*debug_name=*/kTfQuantPtqDynamicRangeStepName};
  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir);
}

}  // namespace internal
}  // namespace quantization
}  // namespace tensorflow
