/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/python/mlir.h"

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/saved_model_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mlprogram_util.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tosa/tf_passes.h"
#include "tensorflow/compiler/mlir/tosa/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"
#include "xla/mlir/framework/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {
// All the passes we will make available to Python by default.
// TODO(tf): this should be sharded instead of being monolithic like that.
static void RegisterPasses() {
  static bool unique_registration = [] {
    mlir::registerAllPasses();
    mlir::registerTensorFlowPasses();
    mlir::TFDevice::registerTensorFlowDevicePasses();
    mlir::mhlo::registerAllMhloPasses();
    // These are in compiler/mlir/xla and not part of the above MHLO
    // passes.
    mlir::mhlo::registerTfXlaPasses();
    mlir::mhlo::registerLegalizeTFPass();
    mlir::quant::stablehlo::registerBridgePasses();
    mlir::tosa::registerLegalizeTosaPasses();
    mlir::tosa::registerTFtoTOSALegalizationPipeline();
    mlir::tosa::registerTFLtoTOSALegalizationPipeline();
    mlir::tosa::registerTFTFLtoTOSALegalizationPipeline();
    mlir::tf_saved_model::registerTensorFlowSavedModelPasses();
    mlir::xla_framework::registerXlaFrameworkPasses();
    tensorflow::RegisterMlProgramPasses();
    return true;
  }();
  (void)unique_registration;
}

// Runs pass pipeline `pass_pipeline` on `module` if `pass_pipeline` is not
// empty.
std::string RunPassPipelineOnModule(mlir::ModuleOp module,
                                    const std::string& pass_pipeline,
                                    bool show_debug_info, TF_Status* status) {
  RegisterPasses();
  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(module.getContext());
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   ("Invalid pass_pipeline: " + error_stream.str()).c_str());
      return "// error";
    }

    mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext());
    if (failed(pm.run(module))) {
      tsl::Set_TF_Status_from_Status(status, statusHandler.ConsumeStatus());
      return "// error";
    }
  }
  return MlirModuleToString(module, show_debug_info);
}

}  // anonymous namespace

static std::string ImportGraphDefImpl(const std::string& proto,
                                      const std::string& pass_pipeline,
                                      bool show_debug_info,
                                      GraphDebugInfo& debug_info,
                                      GraphImportConfig& specs,
                                      TF_Status* status) {
  GraphDef graphdef;
  auto s = tensorflow::LoadProtoFromBuffer(proto, &graphdef);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  auto module = ConvertGraphdefToMlir(graphdef, debug_info, specs, &context);
  if (!module.ok()) {
    tsl::Set_TF_Status_from_Status(status, module.status());
    return "// error";
  }

  return RunPassPipelineOnModule(module->get(), pass_pipeline, show_debug_info,
                                 status);
}

std::string ImportFunction(const std::string& functiondef_proto,
                           const std::string& pass_pipeline,
                           bool show_debug_info, TFE_Context* tfe_context,
                           TF_Status* status) {
  FunctionDef functiondef;
  auto s = tensorflow::LoadProtoFromBuffer(functiondef_proto, &functiondef);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  const std::string& function_name = functiondef.signature().name();
  EagerContext* cpp_context = ContextFromInterface(unwrap(tfe_context));
  FunctionLibraryDefinition& flib_def = *cpp_context->FuncLibDef();
  const tensorflow::FunctionDef* fdef = flib_def.Find(function_name);
  if (fdef == nullptr) {
    s = tensorflow::errors::NotFound("Cannot find function ", function_name);
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  std::unique_ptr<tensorflow::FunctionBody> fbody;
  s = FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice(), &flib_def,
                              &fbody);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  auto module = ConvertFunctionToMlir(fbody.get(), flib_def, &context);
  if (!module.ok()) {
    tsl::Set_TF_Status_from_Status(status, module.status());
    return "// error";
  }

  return RunPassPipelineOnModule(module->get(), pass_pipeline, show_debug_info,
                                 status);
}

std::string ImportGraphDef(const std::string& proto,
                           const std::string& pass_pipeline,
                           bool show_debug_info, TF_Status* status) {
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  return ImportGraphDefImpl(proto, pass_pipeline, show_debug_info, debug_info,
                            specs, status);
}

std::string ImportGraphDef(const std::string& proto,
                           const std::string& pass_pipeline,
                           bool show_debug_info, absl::string_view input_names,
                           absl::string_view input_data_types,
                           absl::string_view input_data_shapes,
                           absl::string_view output_names, TF_Status* status) {
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  auto s = ParseInputArrayInfo(input_names, input_data_types, input_data_shapes,
                               &specs.inputs);
  if (!s.ok()) {
    tsl::Set_TF_Status_from_Status(status, s);
    return "// error";
  }
  if (!output_names.empty()) {
    specs.outputs = absl::StrSplit(output_names, ',');
  }
  return ImportGraphDefImpl(proto, pass_pipeline, show_debug_info, debug_info,
                            specs, status);
}

std::string ExperimentalConvertSavedModelToMlir(
    const std::string& saved_model_path, const std::string& exported_names_str,
    bool show_debug_info, TF_Status* status) {
  // Load the saved model into a SavedModelV2Bundle.

  tensorflow::SavedModelV2Bundle bundle;
  auto load_status =
      tensorflow::SavedModelV2Bundle::Load(saved_model_path, &bundle);
  if (!load_status.ok()) {
    tsl::Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelV2Bundle to an MLIR module.

  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  auto module_or = ConvertSavedModelToMlir(
      &bundle, &context, absl::Span<std::string>(exported_names));
  if (!module_or.status().ok()) {
    tsl::Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*std::move(module_or).value(), show_debug_info);
}

std::string ExperimentalConvertSavedModelV1ToMlirLite(
    const std::string& saved_model_path, const std::string& exported_names_str,
    const std::string& tags, bool upgrade_legacy, bool show_debug_info,
    TF_Status* status) {
  std::unordered_set<string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());

  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);

  tensorflow::SavedModelImportOptions import_options;
  import_options.upgrade_legacy = upgrade_legacy;
  auto module_or = SavedModelSignatureDefsToMlirImportLite(
      saved_model_path, tag_set, absl::Span<std::string>(exported_names),
      &context, import_options);
  if (!module_or.status().ok()) {
    tsl::Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*module_or.value(), show_debug_info);
}

std::string ExperimentalConvertSavedModelV1ToMlir(
    const std::string& saved_model_path, const std::string& exported_names_str,
    const std::string& tags, bool lift_variables,
    bool include_variables_in_initializers, bool upgrade_legacy,
    bool show_debug_info, TF_Status* status) {
  // Load the saved model into a SavedModelBundle.

  std::unordered_set<string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());

  tensorflow::SavedModelBundle bundle;
  auto load_status =
      tensorflow::LoadSavedModel({}, {}, saved_model_path, tag_set, &bundle);
  if (!load_status.ok()) {
    tsl::Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelBundle to an MLIR module.
  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  tensorflow::SavedModelImportOptions import_options;
  import_options.upgrade_legacy = upgrade_legacy;
  import_options.lift_variables = lift_variables;
  import_options.include_variables_in_initializers =
      include_variables_in_initializers;
  auto module_or =
      ConvertSavedModelV1ToMlir(bundle, absl::Span<std::string>(exported_names),
                                &context, import_options);
  if (!module_or.status().ok()) {
    tsl::Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  // Run the tf standard pipeline by default and then, run passes that lift
  // variables if the flag is set on the module.
  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(module_or).value();
  mlir::PassManager pm(&context);
  std::string error;
  llvm::raw_string_ostream error_stream(error);

  mlir::TF::StandardPipelineOptions tf_options;
  mlir::TF::CreateTFStandardPipeline(pm, tf_options);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module))) {
    tsl::Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
    return "// error";
  }
  return MlirModuleToString(*module, show_debug_info);
}

std::string ExperimentalRunPassPipeline(const std::string& mlir_txt,
                                        const std::string& pass_pipeline,
                                        bool show_debug_info,
                                        TF_Status* status) {
  RegisterPasses();
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_txt, &context);
    if (!module) {
      tsl::Set_TF_Status_from_Status(status,
                                     diagnostic_handler.ConsumeStatus());
      return "// error";
    }
  }

  // Run the pass_pipeline on the module.
  mlir::PassManager pm(&context);
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Invalid pass_pipeline: " + error_stream.str()).c_str());
    return "// error";
  }

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module))) {
    tsl::Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
    return "// error";
  }
  return MlirModuleToString(*module, show_debug_info);
}

void ExperimentalWriteBytecode(const std::string& filename,
                               const std::string& mlir_txt, TF_Status* status) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  {
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_txt, &context);
    if (!module) {
      tsl::Set_TF_Status_from_Status(status,
                                     diagnostic_handler.ConsumeStatus());
      return;
    }
  }
  mlir::FallbackAsmResourceMap fallback_resource_map;
  mlir::BytecodeWriterConfig writer_config(fallback_resource_map);
  // TODO(jpienaar): Make this an option to the call.
  writer_config.setDesiredBytecodeVersion(1);
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(filename, &error);
  if (!error.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Unable to create output file " + error).c_str());
    return;
  }
  outputFile->keep();
  if (failed(mlir::writeBytecodeToFile(*module, outputFile->os(),
                                       writer_config))) {
    tsl::Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
  }
}

void ExperimentalTFLiteToTosaBytecode(
    const std::string& flatbuffer_file, const std::string& tosa_bytecode_file,
    bool use_external_constant,
    const std::vector<std::string>& ordered_input_arrays,
    const std::vector<std::string>& ordered_output_arrays, TF_Status* status) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  registry.insert<mlir::tosa::TosaDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  {
    mlir::Location loc = mlir::UnknownLoc::get(&context);
    std::string error;
    std::unique_ptr<llvm::MemoryBuffer> buffer =
        mlir::openInputFile(flatbuffer_file, &error);
    if (buffer == nullptr) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   ("Unable to load input file " + error).c_str());
      return;
    }

    auto buffer_view =
        std::string_view(buffer->getBufferStart(), buffer->getBufferSize());
    module = tflite::FlatBufferToMlir(
        buffer_view, &context, loc, use_external_constant, ordered_input_arrays,
        ordered_output_arrays);
    mlir::PassManager pm(&context, module.get()->getName().getStringRef(),
                         mlir::PassManager::Nesting::Implicit);
    mlir::tosa::TOSATFLLegalizationPipelineOptions opts;
    // This flow is specific to compilation backend, so set to true.
    opts.target_compilation_backend = true;
    // Temporary work-around for https://github.com/openxla/iree/issues/8974
    opts.dequantize_tfl_softmax = true;
    createTFLtoTOSALegalizationPipeline(pm, opts);
    if (failed(pm.run(*module))) {
      tsl::Set_TF_Status_from_Status(status,
                                     diagnostic_handler.ConsumeStatus());
      return;
    }
  }
  mlir::FallbackAsmResourceMap fallback_resource_map;
  mlir::BytecodeWriterConfig writer_config(fallback_resource_map);
  // TODO(jpienaar): Make this an option to the call.
  writer_config.setDesiredBytecodeVersion(1);
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(tosa_bytecode_file, &error);
  if (!error.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Unable to create output file" + error).c_str());
    return;
  }
  outputFile->keep();
  if (failed(mlir::writeBytecodeToFile(*module, outputFile->os(),
                                       writer_config))) {
    tsl::Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
  }
}

}  // namespace tensorflow
