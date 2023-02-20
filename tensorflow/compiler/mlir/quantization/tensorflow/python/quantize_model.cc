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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/save_variables.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace quantization {
namespace {

using ::mlir::quant::kTfQuantSaveOpName;
using ::mlir::tf_saved_model::kTfSavedModelInitializerInitType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;

// Suffix string for the module export step. Used for debugging.
constexpr absl::string_view kExportStepSuffix = "_export";

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
  std::string checkpoint_dir = "";

  // Name used to identify the ModuleOp this is exporting. Only used for
  // debugging and does not modify the behavior of the export.
  std::string debug_name = "tf_quant";
};

// Add passes for transforming the MLIR module op so that it can be exported
// back to GraphDef. Roughly, this consists of:
//   1) Inserting the @main function, which will become the main Graph.
//   2) Duplicating shape-determining constants.
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
  pm.addPass(mlir::quant::CreateMergeSaveFunctionOpsToMainPass());

  // Used to clean up the "tf._noinliner" attribute that is previously used to
  // prevent certain functions from being inlined (see
  // `MarkFunctionsNoinlinePass`). InlinerPass must not come after this pass.
  pm.addPass(mlir::TF::CreateStripNoinlineAttributePass());
}

// Finds and returns the name of the node from a set of control output nodes.
// The name should contain the string `contains`. Returns an empty string if no
// node whose name contains `contains` is found. Assumes there is at most one
// such a node.
std::string GetNodeName(const absl::flat_hash_set<Node *> &control_ret_nodes,
                        const absl::string_view contains) {
  for (Node *control_ret_node : control_ret_nodes) {
    if (absl::StrContains(control_ret_node->name(), contains)) {
      VLOG(1) << "Node found: " << control_ret_node->name()
              << ", contains: " << contains;
      return control_ret_node->name();
    }
  }
  VLOG(1) << "Could not find node whose name conatins: " << contains;
  return "";
}

[[nodiscard]] ExportedModel CreateExportedModel(
    GraphDef &&graph_def, const absl::string_view init_node_name,
    const absl::string_view restore_node_name,
    const absl::string_view save_node_name,
    const absl::string_view checkpoint_dir,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  ExportedModel exported_model{};
  *exported_model.mutable_graph_def() = graph_def;
  exported_model.set_init_node_name(std::string(init_node_name));
  exported_model.set_restore_node_name(std::string(restore_node_name));
  exported_model.set_save_node_name(std::string(save_node_name));
  exported_model.set_checkpoint_dir(std::string(checkpoint_dir));
  exported_model.mutable_function_aliases()->insert(function_aliases.begin(),
                                                    function_aliases.end());

  return exported_model;
}

// Converts MLIR ModuleOp to ExportedModel. Returns InternalError status
// when the conversion fails.
absl::StatusOr<ExportedModel> ConvertMlirModuleToExportedModel(
    const mlir::ModuleOp module_op, const absl::string_view checkpoint_dir,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
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

  const std::string init_node_name =
      GetNodeName(control_ret_nodes, kTfSavedModelInitializerInitType);
  const std::string restore_node_name =
      GetNodeName(control_ret_nodes, kTfSavedModelInitializerRestoreType);
  const std::string save_node_name =
      GetNodeName(control_ret_nodes, kTfQuantSaveOpName);

  return CreateExportedModel(std::move(graph_def), init_node_name,
                             restore_node_name, save_node_name, checkpoint_dir,
                             function_aliases);
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
// TODO(b/262189534): Move this to a separate file for better testing.
absl::Status UnfreezeConstantsAndSaveVariables(
    const absl::string_view checkpoint_dir, mlir::MLIRContext &ctx,
    mlir::ModuleOp module_op) {
  if (const absl::Status pass_run_status = RunPasses(
          /*name=*/kTfQuantConstantUnfreezingStepName,
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

  const auto variable_save_status =
      SaveVariablesToCheckpoint(checkpoint_dir, module_op);
  if (!variable_save_status.ok()) {
    return variable_save_status.status();
  }

  return RunPasses(
      /*name=*/kTfQuantInsertRestoreOpStepName,
      /*add_passes_func=*/
      [](mlir::PassManager &pm) {
        pm.addPass(mlir::quant::CreateInsertRestoreOpPass());
        pm.addPass(mlir::quant::CreateInsertSaveOpPass());
        // Initialization by `tf.ConstOp` is no longer required as there is
        // a `tf.RestoreV2Op` now.
        pm.addPass(
            mlir::quant::CreateRemoveVariableInitializationByConstPass());
      },
      ctx, module_op);
}

// Sets up and runs the passes for exporting `module_op`. The behavior of the
// exporting passes is controlled by `export_opts`.
absl::Status RunExportPasses(const ExportOptions &export_opts,
                             mlir::MLIRContext &ctx, mlir::ModuleOp module_op) {
  if (export_opts.unfreeze_constants) {
    if (const absl::Status unfreeze_constants_status =
            UnfreezeConstantsAndSaveVariables(export_opts.checkpoint_dir, ctx,
                                              module_op);
        !unfreeze_constants_status.ok()) {
      return unfreeze_constants_status;
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
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options) {
  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tags,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  if (const absl::Status preprocess_status = PreprocessAndFreezeGraph(
          module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
      !preprocess_status.ok()) {
    return preprocess_status;
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

  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/true, unfreeze_constants,
      *checkpoint_dir,
      /*debug_name=*/absl::StrCat(kTfQuantQatStepName, kExportStepSuffix)};

  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir,
                                          /*function_aliases=*/{});
}

// Returns the updated function aliases. `module_op` may have different function
// names from the original model, so it re-associates the aliases with the new
// function names. Both the input `function_aliases` and the returned value
// are function name -> alias mappings. `function_aliases` is the function alias
// mapping of the original function.
absl::flat_hash_map<std::string, std::string> UpdateFunctionAliases(
    const absl::flat_hash_map<std::string, std::string> function_aliases,
    mlir::ModuleOp module_op) {
  absl::flat_hash_map<std::string, std::string> updated_function_aliases;

  module_op->walk([&](mlir::func::FuncOp func_op) {
    // We may retrieve the original function's name from the attribute.
    // Functions without this attribute are ignored.
    auto original_func_name =
        func_op->getAttrOfType<mlir::StringAttr>("tf._original_func_name");
    if (original_func_name) {
      if (auto alias_itr = function_aliases.find(original_func_name.str());
          alias_itr != function_aliases.end()) {
        const std::string alias = alias_itr->second;
        const std::string new_func_name = func_op.getSymName().str();

        updated_function_aliases[new_func_name] = alias;

        VLOG(1) << "Updated function alias. Alias: " << alias
                << ", New function name: " << new_func_name
                << ", Old function name: " << original_func_name.str();
      }
    }
  });

  return updated_function_aliases;
}

absl::StatusOr<ExportedModel> QuantizePtqModelPreCalibration(
    const absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tags,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<std::string, std::string> updated_function_aliases =
      UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  if (const absl::Status preprocess_status = PreprocessAndFreezeGraph(
          /*mlir_dump_file_prefix=*/kTfQuantPtqPreCalibrationStepName,
          /*is_inliner_run=*/true,
          /*noinline_functions=*/aliased_function_names, module_ref.get(),
          &context, bundle ? bundle->GetSession() : nullptr);
      !preprocess_status.ok()) {
    return preprocess_status;
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
  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/false, unfreeze_constants,
      *checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(kTfQuantPtqPreCalibrationStepName, kExportStepSuffix)};

  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir,
                                          updated_function_aliases);
}

absl::StatusOr<ExportedModel> QuantizePtqModelPostCalibration(
    const absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tags,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<std::string, std::string> updated_function_aliases =
      UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  // Freezing is required again since variables might have been produced during
  // the pre-calibration step. `is_inliner_run = false` to prevent the functions
  // lifted for quantization from being inlined.
  if (const absl::Status preprocess_status = PreprocessAndFreezeGraph(
          /*mlir_dump_file_prefix=*/kTfQuantPtqPostCalibrationStepName,
          /*is_inliner_run=*/false,
          /*noinline_functions=*/aliased_function_names, module_ref.get(),
          &context, bundle ? bundle->GetSession() : nullptr);
      !preprocess_status.ok()) {
    return preprocess_status;
  }

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
  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/true, unfreeze_constants,
      *checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(kTfQuantPtqPostCalibrationStepName, kExportStepSuffix)};

  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir,
                                          updated_function_aliases);
}

absl::StatusOr<ExportedModel> QuantizePtqDynamicRange(
    const absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options) {
  // Convert the SavedModelBundle to an MLIR module.
  mlir::MLIRContext context = CreateMlirContextForTfQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tags,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  if (const absl::Status preprocess_status = PreprocessAndFreezeGraph(
          module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
      !preprocess_status.ok()) {
    return preprocess_status;
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
  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/true, unfreeze_constants,
      *checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(kTfQuantPtqDynamicRangeStepName, kExportStepSuffix)};
  if (const absl::Status export_status =
          RunExportPasses(export_opts, context, *module_ref);
      !export_status.ok()) {
    return export_status;
  }

  return ConvertMlirModuleToExportedModel(*module_ref, *checkpoint_dir,
                                          /*function_aliases=*/{});
}

}  // namespace quantization
}  // namespace tensorflow
