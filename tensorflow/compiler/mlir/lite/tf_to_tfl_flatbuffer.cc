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

#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/serializer/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/op_stat_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_tfl_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/transforms.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/tools/optimize/quantize_weights.h"
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace {
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningOpRef;
using tsl::StatusOr;

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

// Util that registers 'extra_tf_opdefs' to the TF global registry.
// Return OK on success, failure if registering failed.
Status RegisterExtraTfOpDefs(absl::Span<const std::string> extra_tf_opdefs) {
  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs_string,
                                                           &opdef)) {
      LOG(ERROR) << "OpDef parsing failed for: " << tf_opdefs_string;
      return errors::InvalidArgument("fail to parse extra OpDef");
    }
    // Register extra opdefs.
    // TODO(b/133770952): Support shape functions.
    tensorflow::OpRegistry::Global()->Register(
        [opdef](tensorflow::OpRegistrationData* op_reg_data) -> Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return OkStatus();
        });
  }
  return OkStatus();
}
}  // namespace

StatusOr<OwningOpRef<ModuleOp>> LoadFromGraphdefOrMlirSource(
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
    llvm::errs() << error_message << "\n";
    return errors::InvalidArgument("fail to open input file");
  }

  if (input_mlir) {
    source_mgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningOpRef<ModuleOp>(
        mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, context));
  }

  // Register extra TF ops passed as OpDef.
  auto extra_opdefs_status = RegisterExtraTfOpDefs(extra_tf_opdefs);
  if (!extra_opdefs_status.ok()) return extra_opdefs_status;

  if (use_splatted_constant) {
    return tensorflow::GraphdefToSplattedMlirTranslateFunction(
        file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
        input_shapes, output_arrays, control_output_arrays,
        specs.prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
        /*graph_as_function=*/false, specs.upgrade_legacy,
        /*enable_shape_inference=*/false,
        /*unconditionally_use_set_output_shapes=*/true, context);
  }
  return tensorflow::GraphdefToMlirTranslateFunction(
      file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
      input_shapes, output_arrays, control_output_arrays,
      specs.prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
      /*graph_as_function=*/false, specs.upgrade_legacy,
      /*enable_shape_inference=*/false,
      /*unconditionally_use_set_output_shapes=*/true, context);
}

// Applying post-training dynamic range quantization from the old TOCO quantizer
// on the translated_result using quant_specs and saving the final output in
// result.
Status ApplyDynamicRangeQuantizationFromOldQuantizer(
    const mlir::quant::QuantizationSpecs& quant_specs,
    std::string translated_result, std::string* result) {
  flatbuffers::FlatBufferBuilder q_builder(/*initial_size=*/10240);
  const uint8_t* buffer =
      reinterpret_cast<const uint8_t*>(translated_result.c_str());
  const ::tflite::Model* input_model = ::tflite::GetModel(buffer);

  ::tflite::optimize::BufferType quantized_type;
  switch (quant_specs.inference_type) {
    case tensorflow::DT_QINT8:
      quantized_type = ::tflite::optimize::BufferType::QUANTIZED_INT8;
      break;
    case tensorflow::DT_HALF:
      quantized_type = ::tflite::optimize::BufferType::QUANTIZED_FLOAT16;
      break;
    default:
      return errors::InvalidArgument("Quantized type not supported");
      break;
  }

  bool use_updated_hybrid_scheme = !quant_specs.disable_per_channel;
  if (::tflite::optimize::QuantizeWeights(
          &q_builder, input_model, quantized_type, use_updated_hybrid_scheme,
          ::tflite::optimize::QuantizerType::OLD_QUANTIZER) != kTfLiteOk) {
    return errors::InvalidArgument("Quantize weights transformation failed.");
  }
  const uint8_t* q_buffer = q_builder.GetBufferPointer();
  *result =
      string(reinterpret_cast<const char*>(q_buffer), q_builder.GetSize());

  return OkStatus();
}

Status ConvertTFExecutorToStablehloFlatbuffer(
    mlir::PassManager& pass_manager, mlir::ModuleOp module, bool export_to_mlir,
    mlir::StatusScopedDiagnosticHandler& statusHandler,
    const toco::TocoFlags& toco_flags, const mlir::TFL::PassConfig& pass_config,
    llvm::Optional<tensorflow::Session*> session, std::string* result) {
  // Currently, TF quantization only support dynamic range quant, as such
  // when toco flag post training quantization is specified with converting to
  // stablehlo, we automatically enable dynamic range quantization

  if (toco_flags.post_training_quantize()) {
    const auto status = tensorflow::quantization::PreprocessAndFreezeGraph(
        module, module.getContext(), session);
    if (!status.ok()) {
      return errors::Aborted("Failed to preprocess & freeze TF graph");
    }

    // The default minimum number of elements a weights array must have to be
    // quantized by this transformation.
    const int kWeightsMinNumElementsDefault = 1024;

    tensorflow::quantization::QuantizationOptions quantization_options;

    quantization_options.mutable_quantization_method()->set_experimental_method(
        tensorflow::quantization::QuantizationMethod::DYNAMIC_RANGE);
    quantization_options.set_op_set(
        tensorflow::quantization::UNIFORM_QUANTIZED);
    quantization_options.set_min_num_elements_for_weights(
        kWeightsMinNumElementsDefault);
    tensorflow::quantization::AddQuantizePtqDynamicRangePasses(
        pass_manager, quantization_options);
    if (failed(pass_manager.run(module))) {
      return statusHandler.ConsumeStatus();
    }
  }

  pass_manager.clear();
  mlir::odml::AddTFToStablehloPasses(pass_manager, /*skip_resize*/ true,
                                     /*smuggle_disallowed_ops*/ true);
  // Print out a detailed report of non-converted stats.
  pass_manager.addPass(mlir::odml::createPrintOpStatsPass());
  mlir::odml::AddStablehloOptimizationPasses(pass_manager);
  if (failed(pass_manager.run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return statusHandler.ConsumeStatus();
  }

  mlir::odml::FlatbufferExportOptions options;
  if (!mlir::odml::MlirToFlatBufferTranslateFunction(module, options, result)) {
    return statusHandler.ConsumeStatus();
  }

  return OkStatus();
}

Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir,
    const toco::TocoFlags& toco_flags, const mlir::TFL::PassConfig& pass_config,
    const std::unordered_set<std::string>& saved_model_tags,
    llvm::StringRef saved_model_dir,
    llvm::Optional<tensorflow::Session*> session, std::string* result) {
  // Explicitly disable dumping Op details on failures.
  module.getContext()->printOpOnDiagnostic(false);

  // Register a warning handler only log to std out.
  mlir::ScopedDiagnosticHandler s(
      module.getContext(), [](mlir::Diagnostic& diag) {
        if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
          for (auto& note : diag.getNotes()) {
            std::cout << note.str() << "\n";
            LOG(WARNING) << note.str() << "\n";
          }
        }
        return mlir::failure();
      });

  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);

  if (failed(IsValidGraph(module))) {
    return statusHandler.ConsumeStatus();
  }

  mlir::PassManager pass_manager(module.getContext());
  mlir::registerPassManagerCLOptions();
  mlir::applyPassManagerCLOptions(pass_manager);
  pass_manager.addInstrumentation(
      std::make_unique<mlir::TFL::ErrorCollectorInstrumentation>(
          pass_manager.getContext()));

  if (pass_config.enable_stablehlo_conversion) {
    // return to avoid adding TFL converter path
    return ConvertTFExecutorToStablehloFlatbuffer(
        pass_manager, module, export_to_mlir, statusHandler, toco_flags,
        pass_config, session, result);
  }

  tensorflow::AddPreVariableFreezingTFToTFLConversionPasses(pass_config,
                                                            &pass_manager);
  if (failed(pass_manager.run(module))) {
    return statusHandler.ConsumeStatus();
  }
  // Freeze variables if a session is provided.
  if (session.has_value()) {
    mlir::TFL::ErrorCollectorInstrumentation collector(module.getContext());
    if (failed(
            mlir::tf_saved_model::FreezeVariables(module, session.value()))) {
      auto status = statusHandler.ConsumeStatus();
      mlir::TFL::ErrorCollector* collector =
          mlir::TFL::ErrorCollector::GetErrorCollector();
      if (!collector->CollectedErrors().empty()) {
        // LINT.IfChange
        return errors::InvalidArgument(
            "Variable constant folding is failed. Please consider using "
            "enabling `experimental_enable_resource_variables` flag in the "
            "TFLite converter object. For example, "
            "converter.experimental_enable_resource_variables = True");
        // LINT.ThenChange(//tensorflow/lite/python/lite_v2_test.py)
      }
      return status;
    }
  }

  pass_manager.clear();

  tensorflow::AddPostVariableFreezingTFToTFLConversionPasses(
      saved_model_dir, toco_flags, pass_config, &pass_manager);
  if (failed(pass_manager.run(module))) {
    auto status = statusHandler.ConsumeStatus();
    mlir::TFL::ErrorCollector* collector =
        mlir::TFL::ErrorCollector::GetErrorCollector();
    for (const auto& error_data : collector->CollectedErrors()) {
      if (error_data.subcomponent() == "FreezeGlobalTensorsPass") {
        // LINT.IfChange
        return errors::InvalidArgument(
            "Variable constant folding is failed. Please consider using "
            "enabling `experimental_enable_resource_variables` flag in the "
            "TFLite converter object. For example, "
            "converter.experimental_enable_resource_variables = True");
        // LINT.ThenChange(//tensorflow/lite/python/lite_v2_test.py)
      }
    }
    return status;
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return statusHandler.ConsumeStatus();
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
  if (!tflite::MlirToFlatBufferTranslateFunction(module, options,
                                                 &translated_result)) {
    return statusHandler.ConsumeStatus();
  }

  // TODO(b/176267167): Quantize flex fallback in the MLIR pipeline
  if (quant_specs.weight_quantization &&
      (!quant_specs.RunAndRewriteDynamicRangeQuantizationPasses() ||
       !pass_config.emit_builtin_tflite_ops)) {
    // Apply post-training dynamic range quantization from the old TOCO
    // quantizer.Once MLIR has support for this, we can remove this if
    // statement.
    auto status = ApplyDynamicRangeQuantizationFromOldQuantizer(
        quant_specs, translated_result, result);
    if (!status.ok()) return status;
  } else {
    *result = translated_result;
  }

  if (mlir::failed(module.verifyInvariants())) {
    return tensorflow::errors::Unknown("Final module is invalid");
  }
  return OkStatus();
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    const std::string& input_filename, const int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    bool enable_variable_lifting, mlir::MLIRContext* context,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle) {
  // Register extra TF ops passed as OpDef.
  auto extra_opdefs_status = RegisterExtraTfOpDefs(extra_tf_opdefs);
  if (!extra_opdefs_status.ok()) return extra_opdefs_status;

  if (saved_model_version == 2) {
    auto module_or = tensorflow::SavedModelObjectGraphToMlirImport(
        input_filename, tags, exported_names, context,
        /*unconditionally_use_set_output_shapes=*/true);
    if (!module_or.status().ok()) return module_or.status();
    return std::move(module_or).value();
  } else if (saved_model_version == 1) {
    MLIRImportOptions options;
    options.upgrade_legacy = specs.upgrade_legacy;
    options.unconditionally_use_set_output_shapes = true;
    auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, context, options,
        enable_variable_lifting, saved_model_bundle);

    if (!module_or.status().ok()) return module_or.status();
    return std::move(module_or).value();
  } else {
    return tensorflow::errors::InvalidArgument(
        "Should be either saved model v1 or v2");
  }
}

}  // namespace tensorflow
