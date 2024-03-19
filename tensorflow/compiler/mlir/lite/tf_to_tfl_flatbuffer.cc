/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/debug/debug.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "tensorflow/compiler/mlir/lite/quantization/stablehlo/quantization.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/op_stat_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_util.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/transforms.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/remat/metadata_util.h"
#include "tensorflow/lite/python/metrics/converter_error_data.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/quantize_weights.h"
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace tensorflow {
namespace {

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningOpRef;
using ::stablehlo::quantization::QuantizationConfig;
using ::tensorflow::quantization::PyFunctionLibrary;

bool IsControlFlowV1Op(Operation* op) {
  return mlir::isa<mlir::tf_executor::SwitchOp, mlir::tf_executor::MergeOp,
                   mlir::tf_executor::EnterOp, mlir::tf_executor::ExitOp,
                   mlir::tf_executor::NextIterationSinkOp,
                   mlir::tf_executor::NextIterationSourceOp>(op);
}

mlir::LogicalResult IsValidGraph(mlir::ModuleOp module) {
  auto result = module.walk([&](Operation* op) {
    return IsControlFlowV1Op(op) ? mlir::WalkResult::interrupt()
                                 : mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    mlir::TFL::AttachErrorCode(
        module.emitError(
            "The graph has Control Flow V1 ops. TFLite converter doesn't "
            "support Control Flow V1 ops. Consider using Control Flow V2 ops "
            "instead. See https://www.tensorflow.org/api_docs/python/tf/compat/"
            "v1/enable_control_flow_v2."),
        tflite::metrics::ConverterErrorData::ERROR_UNSUPPORTED_CONTROL_FLOW_V1);
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult GraphContainsStatefulPartitionedOp(mlir::ModuleOp module) {
  auto result = module.walk([&](Operation* op) {
    return llvm::isa_and_nonnull<mlir::TF::StatefulPartitionedCallOp>(op)
               ? mlir::WalkResult::interrupt()
               : mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    // StatefulPartitionedCall ops are not supported by the tflite runtime.
    mlir::TFL::AttachErrorCode(
        module.emitError(
            "The Graph contains unsupported `StatefulPartionedCallOp`(s), will "
            "retry with `guarantee_all_funcs_used_once`"),
        tflite::metrics::ConverterErrorData::
            ERROR_STATEFUL_PARTITIONED_CALL_IN_FINAL_IR);
    return mlir::failure();
  }
  return mlir::success();
}

// Util that registers 'extra_tf_opdefs' to the TF global registry.
// Return OK on success, failure if registering failed.
absl::Status RegisterExtraTfOpDefs(
    absl::Span<const std::string> extra_tf_opdefs) {
  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    OpDef opdef;
    // NOLINTNEXTLINE: Use tsl::protobuf to be compatible with OSS.
    if (!tsl::protobuf::TextFormat::ParseFromString(tf_opdefs_string, &opdef)) {
      LOG(ERROR) << "OpDef parsing failed for: " << tf_opdefs_string;
      return absl::InvalidArgumentError("fail to parse extra OpDef");
    }
    // Register extra opdefs.
    // TODO: b/133770952 - Support shape functions.
    OpRegistry::Global()->Register(
        [opdef](OpRegistrationData* op_reg_data) -> absl::Status {
          *op_reg_data = OpRegistrationData(opdef);
          return absl::OkStatus();
        });
  }
  return absl::OkStatus();
}

// The hlo->tf conversion is done in three steps; pre-quantization,
// quantization, and post-quantization. Quantization is optional, enabled only
// when `pass_config.enable_stablehlo_quantizer` is `true`. If quantization is
// not run, it only performs the hlo->tf conversion.
//
// All parameters except for `pass_config`, `pass_manager`, `status_handler`,
// and `module` are only required for quantization. See the comments of
// `RunQuantization` for details. If quantization is not performed, they will be
// ignored.
//
// Returns a failure status when any of the three steps fail. `pass_manager`
// will be cleared before returning.
mlir::LogicalResult RunHloToTfConversion(
    const mlir::TFL::PassConfig& pass_config,
    const absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& saved_model_tags,
    QuantizationConfig* quantization_config,
    const PyFunctionLibrary* quantization_py_function_lib,
    const SavedModelBundle* saved_model_bundle, mlir::PassManager& pass_manager,
    mlir::StatusScopedDiagnosticHandler& status_handler, ModuleOp& module) {
  // TODO: b/194747383 - We need to valid that indeed the "main" func is
  // presented.
  AddPreQuantizationStableHloToTfPasses(/*entry_function_name=*/"main",
                                        pass_config, pass_manager);
  if (failed(pass_manager.run(module))) {
    return mlir::failure();
  }
  pass_manager.clear();

  if (pass_config.enable_stablehlo_quantizer) {
    const absl::StatusOr<mlir::ModuleOp> quantized_module_op = RunQuantization(
        saved_model_bundle, saved_model_dir, saved_model_tags,
        *quantization_config, quantization_py_function_lib, module);
    if (!quantized_module_op.ok()) {
      LOG(ERROR) << "Failed to run quantization: "
                 << quantized_module_op.status();
      return mlir::failure();
    }
    module = *quantized_module_op;
  }

  AddPostQuantizationStableHloToTfPasses(pass_config, pass_manager);
  if (failed(pass_manager.run(module))) {
    return mlir::failure();
  }
  pass_manager.clear();

  return mlir::success();
}

}  // namespace

absl::StatusOr<OwningOpRef<ModuleOp>> LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string>& extra_tf_opdefs,
    const GraphImportConfig& specs, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view control_output_arrays, llvm::SourceMgr* source_mgr,
    MLIRContext* context) {
  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!file) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open input file: ", error_message));
  }

  if (input_mlir) {
    source_mgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningOpRef<ModuleOp>(
        mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, context));
  }

  // Register extra TF ops passed as OpDef.
  auto extra_opdefs_status = RegisterExtraTfOpDefs(extra_tf_opdefs);
  if (!extra_opdefs_status.ok()) return extra_opdefs_status;

  GraphdefToMlirOptions graphdef_conversion_options{
      std::string(debug_info_file),
      /*xla_compile_device_type=*/"",
      /*prune_unused_nodes=*/specs.prune_unused_nodes,
      /*convert_legacy_fed_inputs=*/true,
      /*graph_as_function=*/false,
      specs.upgrade_legacy,
      /*enable_shape_inference=*/false,
      /*unconditionally_use_set_output_shapes=*/true,
      /*enable_soft_placement=*/false};

  if (use_splatted_constant) {
    return GraphdefToSplattedMlirTranslateFunction(
        file->getBuffer(), input_arrays, input_dtypes, input_shapes,
        output_arrays, control_output_arrays, graphdef_conversion_options,
        context);
  }
  return GraphdefToMlirTranslateFunction(file->getBuffer(), input_arrays,
                                         input_dtypes, input_shapes,
                                         output_arrays, control_output_arrays,
                                         graphdef_conversion_options, context);
}

// Applying post-training dynamic range quantization from the old TOCO quantizer
// on the translated_result using quant_specs and saving the final output in
// result.
absl::Status ApplyDynamicRangeQuantizationFromOldQuantizer(
    const mlir::quant::QuantizationSpecs& quant_specs,
    std::string translated_result, std::string* result) {
  flatbuffers::FlatBufferBuilder q_builder(/*initial_size=*/10240);
  const uint8_t* buffer =
      reinterpret_cast<const uint8_t*>(translated_result.c_str());
  const ::tflite::Model* input_model = ::tflite::GetModel(buffer);

  ::tflite::optimize::BufferType quantized_type;
  switch (quant_specs.inference_type) {
    case DT_QINT8:
      quantized_type = ::tflite::optimize::BufferType::QUANTIZED_INT8;
      break;
    case DT_HALF:
      quantized_type = ::tflite::optimize::BufferType::QUANTIZED_FLOAT16;
      break;
    default:
      return absl::InvalidArgumentError("Quantized type not supported");
      break;
  }

  bool use_updated_hybrid_scheme = !quant_specs.disable_per_channel;
  if (::tflite::optimize::QuantizeWeights(
          &q_builder, input_model, quantized_type, use_updated_hybrid_scheme,
          ::tflite::optimize::QuantizerType::OLD_QUANTIZER) != kTfLiteOk) {
    return absl::InvalidArgumentError(
        "Quantize weights transformation failed.");
  }
  const uint8_t* q_buffer = q_builder.GetBufferPointer();
  *result =
      std::string(reinterpret_cast<const char*>(q_buffer), q_builder.GetSize());

  return absl::OkStatus();
}

absl::Status ConvertTFExecutorToStablehloFlatbuffer(
    mlir::PassManager& pass_manager, mlir::ModuleOp module, bool export_to_mlir,
    mlir::StatusScopedDiagnosticHandler& status_handler,
    const toco::TocoFlags& toco_flags, const mlir::TFL::PassConfig& pass_config,
    std::optional<Session*> session, std::string* result,
    const std::unordered_set<std::string>& saved_model_tags) {
  // Currently, TF quantization only support dynamic range quant, as such
  // when toco flag post training quantization is specified with converting to
  // stablehlo, we automatically enable dynamic range quantization

  if (toco_flags.post_training_quantize()) {
    const auto status = quantization::PreprocessAndFreezeGraph(
        module, module.getContext(), session);
    if (!status.ok()) {
      return status_handler.Combine(
          absl::InternalError("Failed to preprocess & freeze TF graph."));
    }

    // TODO: b/264218457 - Refactor the component below once StableHLO Quantizer
    // can run DRQ. Temporarily using TF Quantization for StableHLO DRQ.
    if (!toco_flags.has_quantization_options()) {
      // The default minimum number of elements a weights array must have to be
      // quantized by this transformation.
      const int kWeightsMinNumElementsDefault = 1024;

      quantization::QuantizationOptions quantization_options;

      quantization_options.mutable_quantization_method()->set_preset_method(
          quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8);
      quantization_options.set_op_set(quantization::UNIFORM_QUANTIZED);
      quantization_options.set_min_num_elements_for_weights(
          kWeightsMinNumElementsDefault);
      quantization::AddQuantizePtqDynamicRangePasses(pass_manager,
                                                     quantization_options);
    }
    if (failed(pass_manager.run(module))) {
      return status_handler.ConsumeStatus();
    }
  }

  pass_manager.clear();
  mlir::odml::AddTFToStablehloPasses(pass_manager, /*skip_resize=*/true,
                                     /*smuggle_disallowed_ops=*/true);
  // Print out a detailed report of non-converted stats.
  pass_manager.addPass(mlir::odml::createPrintOpStatsPass(
      mlir::odml::GetAcceptedStableHLODialects()));
  mlir::odml::AddStablehloOptimizationPasses(pass_manager);
  if (toco_flags.has_quantization_options()) {
    stablehlo::quantization::AddQuantizationPasses(
        pass_manager, toco_flags.quantization_options());
  }
  if (failed(pass_manager.run(module))) {
    return status_handler.ConsumeStatus();
  }
  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return status_handler.ConsumeStatus();
  }
  pass_manager.clear();
  pass_manager.addPass(mlir::odml::createLegalizeStablehloToVhloPass());
  if (failed(pass_manager.run(module))) {
    return status_handler.Combine(
        absl::InvalidArgumentError("VHLO lowering failed"));
  }

  // Write MLIR Stablehlo dialect into FlatBuffer
  OpOrArgLocNameMapper op_or_arg_name_mapper;
  tflite::FlatbufferExportOptions options;
  options.toco_flags = toco_flags;
  options.saved_model_tags = saved_model_tags;
  options.op_or_arg_name_mapper = &op_or_arg_name_mapper;
  options.metadata[tflite::kModelUseStablehloTensorKey] = "true";
  if (!tflite::MlirToFlatBufferTranslateFunction(module, options, result,
                                                 true)) {
    return status_handler.Combine(
        absl::InternalError("Could not translate MLIR to FlatBuffer."));
  }

  return absl::OkStatus();
}

absl::Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, toco::TocoFlags& toco_flags,
    const mlir::TFL::PassConfig& pass_config,
    const std::unordered_set<std::string>& saved_model_tags,
    llvm::StringRef saved_model_dir, SavedModelBundle* saved_model_bundle,
    std::string* result, bool serialize_stablehlo_ops,
    const PyFunctionLibrary* quantization_py_function_lib) {
  // Explicitly disable dumping Op details on failures.
  module.getContext()->printOpOnDiagnostic(false);

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  module.getContext()->appendDialectRegistry(registry);

  mlir::StatusScopedDiagnosticHandler status_handler(module.getContext(),
                                                     /*propagate=*/true);

  mlir::PassManager pass_manager(module.getContext());
  mlir::registerPassManagerCLOptions();
  if (mlir::failed(mlir::applyPassManagerCLOptions(pass_manager))) {
    return absl::InternalError("Failed to apply MLIR pass manager CL options.");
  }
  InitPassManager(pass_manager, toco_flags.debug_options());

  pass_manager.addInstrumentation(
      std::make_unique<mlir::TFL::ErrorCollectorInstrumentation>(
          pass_manager.getContext()));

  if (failed(IsValidGraph(module))) {
    return status_handler.ConsumeStatus();
  }

  Session* session = saved_model_bundle == nullptr
                         ? nullptr
                         : saved_model_bundle->GetSession();
  if (pass_config.enable_stablehlo_conversion) {
    // `ConvertTFExecutorToStablehloFlatbuffer` expects a `std::nullopt` if the
    // `Session*` is a nullptr.
    std::optional<Session*> session_opt =
        session == nullptr ? std::nullopt : std::make_optional(session);

    // return to avoid adding TFL converter path
    return ConvertTFExecutorToStablehloFlatbuffer(
        pass_manager, module, export_to_mlir, status_handler, toco_flags,
        pass_config, std::move(session_opt), result, saved_model_tags);
  }

  if (pass_config.enable_hlo_to_tf_conversion) {
    if (failed(RunHloToTfConversion(
            pass_config, saved_model_dir, saved_model_tags,
            toco_flags.mutable_quantization_config(),
            quantization_py_function_lib, saved_model_bundle, pass_manager,
            status_handler, module))) {
      return status_handler.ConsumeStatus();
    }
  }

  AddPreVariableFreezingTFToTFLConversionPasses(pass_config, &pass_manager);
  if (failed(pass_manager.run(module))) {
    return status_handler.ConsumeStatus();
  }

  // Freeze variables if a session is provided.
  if (session != nullptr &&
      failed(mlir::tf_saved_model::FreezeVariables(module, session))) {
    return status_handler.Combine(absl::InvalidArgumentError(
        "Variable constant folding is failed. Please consider using "
        "enabling `experimental_enable_resource_variables` flag in the "
        "TFLite converter object. For example, "
        "converter.experimental_enable_resource_variables = True"));
  }

  pass_manager.clear();

  AddPostVariableFreezingTFToTFLConversionPasses(saved_model_dir, toco_flags,
                                                 pass_config, &pass_manager);
  if (failed(pass_manager.run(module))) {
    return status_handler.Combine(absl::InvalidArgumentError(
        "Variable constant folding is failed. Please consider using "
        "enabling `experimental_enable_resource_variables` flag in the "
        "TFLite converter object. For example, "
        "converter.experimental_enable_resource_variables = True"));
  }

  if (failed(GraphContainsStatefulPartitionedOp(module))) {
    return status_handler.ConsumeStatus();
  }

  if (export_to_mlir) {
    pass_manager.clear();
    // Print out a detailed report of ops that are not converted to TFL ops.
    pass_manager.addPass(mlir::odml::createPrintOpStatsPass(
        mlir::odml::GetAcceptedTFLiteDialects()));
    if (failed(pass_manager.run(module))) {
      return status_handler.ConsumeStatus();
    }

    llvm::raw_string_ostream os(*result);
    module.print(os);
    return status_handler.ConsumeStatus();
  }

  // Write MLIR TFLite dialect into FlatBuffer
  const mlir::quant::QuantizationSpecs& quant_specs = pass_config.quant_specs;
  OpOrArgLocNameMapper op_or_arg_name_mapper;
  tflite::FlatbufferExportOptions options;
  std::string translated_result;
  options.toco_flags = toco_flags;
  options.saved_model_tags = saved_model_tags;
  options.op_or_arg_name_mapper = &op_or_arg_name_mapper;
  if (quant_specs.support_mask !=
      tflite::optimize::ReducedPrecisionSupport::None) {
    options.metadata.insert(
        MetadataForReducedPrecisionSupport(quant_specs.support_mask));
  }
  pass_manager.clear();
  pass_manager.addPass(mlir::odml::createLegalizeStablehloToVhloPass());
  if (failed(pass_manager.run(module))) {
    return status_handler.Combine(
        absl::InvalidArgumentError("VHLO lowering failed"));
  }

  if (!tflite::MlirToFlatBufferTranslateFunction(
          module, options, &translated_result, serialize_stablehlo_ops)) {
    return status_handler.Combine(
        absl::InternalError("Could not translate MLIR to FlatBuffer."));
  }

  // TODO: b/176267167 - Quantize flex fallback in the MLIR pipeline
  if (quant_specs.weight_quantization &&
      (!quant_specs.RunAndRewriteDynamicRangeQuantizationPasses() ||
       !pass_config.emit_builtin_tflite_ops)) {
    // Apply post-training dynamic range quantization from the old TOCO
    // quantizer.Once MLIR has support for this, we can remove this if
    // statement.
    auto status = ApplyDynamicRangeQuantizationFromOldQuantizer(
        quant_specs, translated_result, result);
    if (!status.ok()) {
      return status_handler.Combine(status);
    }
  } else {
    *result = translated_result;
  }

  if (mlir::failed(module.verifyInvariants())) {
    return status_handler.Combine(
        absl::InternalError("Final module is invalid."));
  }
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    const std::string& input_filename, const int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    bool enable_variable_lifting, mlir::MLIRContext* context,
    std::unique_ptr<SavedModelBundle>* saved_model_bundle) {
  // Register extra TF ops passed as OpDef.
  auto extra_opdefs_status = RegisterExtraTfOpDefs(extra_tf_opdefs);
  if (!extra_opdefs_status.ok()) return extra_opdefs_status;

  if (saved_model_version == 2) {
    auto module_or = SavedModelObjectGraphToMlirImport(
        input_filename, tags, exported_names, context,
        /*unconditionally_use_set_output_shapes=*/true);
    if (!module_or.status().ok()) return module_or.status();
    return std::move(module_or).value();
  } else if (saved_model_version == 1) {
    MLIRImportOptions options;
    options.upgrade_legacy = specs.upgrade_legacy;
    options.unconditionally_use_set_output_shapes = true;
    options.lift_variables = enable_variable_lifting;
    auto module_or = SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, context, options,
        saved_model_bundle);

    if (!module_or.status().ok()) return module_or.status();
    return std::move(module_or).value();
  } else {
    return absl::InvalidArgumentError("Should be either saved model v1 or v2.");
  }
}

}  // namespace tensorflow
