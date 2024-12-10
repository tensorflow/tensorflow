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

#include <stdlib.h>

#include <cstddef>
#include <cstdint>
#include <memory>
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
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/debug/debug.h"
#include "tensorflow/compiler/mlir/lite/experimental/remat/metadata_util.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/metrics/converter_error_data.pb.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/quantize_weights.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/op_stat_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_util.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/transforms.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tools/optimize/reduced_precision_metadata.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/const_tensor_utils.h"
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
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningOpRef;
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

// This function estimates the size of the module in bytes. It does so by
// iterating through all the constant-like attributes and tensors in the module
// and summing up their sizes.
//
// This function is used to reserve space in the buffer before serializing the
// module to avoid reallocating the buffer during serialization.
//
// This function may need to be improved to give more accurate size of the
// module if the current estimate is not good enough and causes huge
// reallocations during serialization.
size_t GetApproximateModuleSize(mlir::ModuleOp module) {
  size_t module_size_estimate = 0;
  mlir::DenseMap<mlir::Attribute, size_t> unique_tensors;

  module.walk([&](Operation* op) {
    mlir::DenseElementsAttr attr;
    if (mlir::detail::constant_op_binder<mlir::DenseElementsAttr>(&attr).match(
            op)) {
      auto it = unique_tensors.find(attr);

      // If the tensor hasn't been seen before
      if (it == unique_tensors.end()) {
        size_t tensor_size =
            mlir::TFL::GetSizeInBytes(op->getResult(0).getType());
        unique_tensors[attr] = tensor_size;  // Store the size in the map
        module_size_estimate += tensor_size;
      }
    }
  });
  return module_size_estimate;
}

// Cloning MLIR modules requires serializing the source and deserializing
// into the target. We do this when we need Garbage collection of
// types/attributes after running the pass pipeline.
// This function-
// 1. Get a rough estimate of the size of the source_module, in bytes.
// 2. Serialize the source module into a buffer with size reserved.
// 3. Deletes the existing source module and context.
// 4. Parses the serialized buffer into the new module to create it in the
// destination context
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CloneModuleInto(
    std::unique_ptr<mlir::MLIRContext> source_context,
    mlir::OwningOpRef<mlir::ModuleOp> source_module,
    mlir::MLIRContext& destination_context) {
  // 1. Get the module size. Module size is a rough estimate of all the
  // constant-like attributes and tensors in the module, plus the size of the
  // module itself without the attributes and constants.
  size_t module_size_estimate = GetApproximateModuleSize(source_module.get());

  // 2. Serialize the module into a buffer with size reserved.
  llvm::SmallString<1024> buffer;
  buffer.reserve(module_size_estimate);

  llvm::raw_svector_ostream out(buffer);
  if (failed(mlir::writeBytecodeToFile(source_module.get(), out))) {
    return absl::InternalError("Failed to serialize module");
  }

  // 3. Delete the existing source module and context.
  source_module = nullptr;
  source_context.reset();

  // 4. Parse the serialized buffer into the new module to create it in the
  // destination context.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      buffer.str(), mlir::ParserConfig(&destination_context));
  buffer.clear();

  return module;
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

  mlir::lite::toco_legacy::BufferType quantized_type;
  switch (quant_specs.inference_type) {
    case DT_QINT8:
      quantized_type = mlir::lite::toco_legacy::BufferType::QUANTIZED_INT8;
      break;
    case DT_HALF:
      quantized_type = mlir::lite::toco_legacy::BufferType::QUANTIZED_FLOAT16;
      break;
    default:
      return absl::InvalidArgumentError("Quantized type not supported");
      break;
  }

  bool use_updated_hybrid_scheme = !quant_specs.disable_per_channel;
  absl::Status quantize_weights_status =
      mlir::lite::toco_legacy::QuantizeWeights(
          &q_builder, input_model, quantized_type, use_updated_hybrid_scheme,
          mlir::lite::toco_legacy::QuantizerType::OLD_QUANTIZER);
  if (!quantize_weights_status.ok()) return quantize_weights_status;
  const uint8_t* q_buffer = q_builder.GetBufferPointer();
  *result =
      std::string(reinterpret_cast<const char*>(q_buffer), q_builder.GetSize());

  return absl::OkStatus();
}

absl::Status ConvertTFExecutorToStablehloFlatbuffer(
    mlir::PassManager& pass_manager, mlir::ModuleOp module, bool export_to_mlir,
    mlir::StatusScopedDiagnosticHandler& status_handler,
    const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config, std::string* result,
    const std::unordered_set<std::string>& saved_model_tags) {
  // Currently, TF quantization only support dynamic range quant, as such
  // when converter flag post training quantization is specified with converting
  // to stablehlo, we automatically enable dynamic range quantization

  if (converter_flags.post_training_quantize()) {
    const absl::Status status =
        quantization::PreprocessAndFreezeGraph(module, module.getContext());
    if (!status.ok()) {
      return status_handler.Combine(
          absl::InternalError("Failed to preprocess & freeze TF graph."));
    }

    // TODO: b/264218457 - Refactor the component below once StableHLO Quantizer
    // can run DRQ. Temporarily using TF Quantization for StableHLO DRQ.
    if (!converter_flags.has_quantization_options()) {
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
  if (converter_flags.has_quantization_options()) {
    stablehlo::quantization::AddQuantizationPasses(
        pass_manager, converter_flags.quantization_options());
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
  pass_manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (failed(pass_manager.run(module))) {
    return status_handler.Combine(
        absl::InvalidArgumentError("VHLO lowering failed"));
  }

  // Write MLIR Stablehlo dialect into FlatBuffer
  OpOrArgLocNameMapper op_or_arg_name_mapper;
  tflite::FlatbufferExportOptions options;
  options.converter_flags = converter_flags;
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
    std::unique_ptr<mlir::MLIRContext>&& context,
    mlir::OwningOpRef<mlir::ModuleOp> module,
    tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    const std::unordered_set<std::string>& saved_model_tags,
    llvm::StringRef saved_model_dir, std::string* result,
    bool serialize_stablehlo_ops, bool export_to_mlir,
    const PyFunctionLibrary* quantization_py_function_lib) {
  // TODO: b/353597396 - Remove this once the StableHLO Quantizer is fully
  // eliminated from the TFLite Converter.
  (void)quantization_py_function_lib;

  // Explicitly disable dumping Op details on failures.
  context->printOpOnDiagnostic(false);

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  context->appendDialectRegistry(registry);

  auto status_handler =
      std::make_unique<mlir::StatusScopedDiagnosticHandler>(context.get(),
                                                            /*propagate=*/true);

  auto pass_manager = std::make_unique<mlir::PassManager>(context.get());

  mlir::registerPassManagerCLOptions();
  if (mlir::failed(mlir::applyPassManagerCLOptions(*pass_manager))) {
    return absl::InternalError("Failed to apply MLIR pass manager CL options.");
  }

  InitPassManager(*pass_manager, converter_flags.debug_options());

  if (failed(IsValidGraph(module.get()))) {
    return status_handler->ConsumeStatus();
  }

  if (pass_config.enable_stablehlo_conversion) {
    // return to avoid adding TFL converter path
    return ConvertTFExecutorToStablehloFlatbuffer(
        *pass_manager, module.get(), export_to_mlir, *status_handler,
        converter_flags, pass_config, result, saved_model_tags);
  }
  if (pass_config.enable_hlo_to_tf_conversion) {
    // TODO: b/194747383 - We need to valid that indeed the "main" func is
    // presented.
    AddPreQuantizationStableHloToTfPasses(/*entry_function_name=*/"main",
                                          pass_config, *pass_manager);
    if (failed(pass_manager->run(module.get()))) {
      return status_handler->ConsumeStatus();
    }
    pass_manager->clear();

    AddPostQuantizationStableHloToTfPasses(pass_config, *pass_manager);
    if (failed(pass_manager->run(module.get()))) {
      return status_handler->ConsumeStatus();
    }
    pass_manager->clear();
  }

  AddPreVariableFreezingTFToTFLConversionPasses(pass_config,
                                                pass_manager.get());
  if (failed(pass_manager->run(module.get()))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager->clear();

  AddVariableFreezingFromGlobalTensorsPasses(converter_flags, pass_config,
                                             pass_manager.get());
  if (failed(pass_manager->run(module.get()))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager->clear();

  AddPostVariableFreezingTFToTFLConversionPasses(
      saved_model_dir, converter_flags, pass_config, pass_manager.get());
  if (failed(pass_manager->run(module.get()))) {
    return status_handler->Combine(absl::InvalidArgumentError(
        "Variable constant folding is failed. Please consider using "
        "enabling `experimental_enable_resource_variables` flag in the "
        "TFLite converter object. For example, "
        "converter.experimental_enable_resource_variables = True"));
  }

  if (failed(GraphContainsStatefulPartitionedOp(module.get()))) {
    return status_handler->ConsumeStatus();
  }

  pass_manager->clear();
  pass_manager->addPass(mlir::odml::createLegalizeStablehloToVhloPass());
  pass_manager->addPass(mlir::createReconcileUnrealizedCastsPass());
  if (failed(pass_manager->run(module.get()))) {
    return status_handler->Combine(
        absl::InvalidArgumentError("VHLO lowering failed"));
  }

  // Clear the pass manager and status handler to avoid any traces of the
  // previous MLIRContext.
  pass_manager.reset();
  status_handler.reset();

  // We can clone the module into a new context to avoid any issues with
  // resource variables.
  // TODO(b/349914241): Remove this once the resource variable are read as
  // DenseResourceElementAttr.
  mlir::DialectRegistry new_registry;
  mlir::func::registerAllExtensions(new_registry);
  new_registry
      .insert<mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
              mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect,
              mlir::arith::ArithDialect, mlir::func::FuncDialect,
              mlir::quant::QuantDialect,
              mlir::quantfork::QuantizationForkDialect,
              mlir::tf_saved_model::TensorFlowSavedModelDialect,
              mlir::tf_type::TFTypeDialect,
              mlir::tf_executor::TensorFlowExecutorDialect>();

  auto new_context = std::make_unique<mlir::MLIRContext>(
      new_registry, mlir::MLIRContext::Threading::DISABLED);

  TF_ASSIGN_OR_RETURN(
      auto new_module,
      CloneModuleInto(std::move(context), std::move(module), *new_context));

  module = std::move(new_module);
  context = std::move(new_context);
  new_module = nullptr;
  new_context = nullptr;

  pass_manager = std::make_unique<mlir::PassManager>(context.get());
  mlir::registerPassManagerCLOptions();
  if (mlir::failed(mlir::applyPassManagerCLOptions(*pass_manager))) {
    return absl::InternalError("Failed to apply MLIR pass manager CL options.");
  }
  InitPassManager(*pass_manager, converter_flags.debug_options());

  status_handler =
      std::make_unique<mlir::StatusScopedDiagnosticHandler>(context.get(),
                                                            /*propagate=*/true);

  if (export_to_mlir) {
    pass_manager->clear();
    // Print out a detailed report of ops that are not converted to TFL ops.
    pass_manager->addPass(mlir::odml::createPrintOpStatsPass(
        mlir::odml::GetAcceptedTFLiteDialects()));
    if (failed(pass_manager->run(module.get()))) {
      return status_handler->ConsumeStatus();
    }

    llvm::raw_string_ostream os(*result);
    module->print(os);
    return status_handler->ConsumeStatus();
  }

  // Write MLIR TFLite dialect into FlatBuffer
  const mlir::quant::QuantizationSpecs& quant_specs = pass_config.quant_specs;
  OpOrArgLocNameMapper op_or_arg_name_mapper;
  tflite::FlatbufferExportOptions options;
  std::string translated_result;
  options.converter_flags = converter_flags;
  options.saved_model_tags = saved_model_tags;
  options.op_or_arg_name_mapper = &op_or_arg_name_mapper;
  if (quant_specs.support_mask !=
      tflite::optimize::ReducedPrecisionSupport::None) {
    options.metadata.insert(
        MetadataForReducedPrecisionSupport(quant_specs.support_mask));
  }

  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &translated_result, false)) {
    return status_handler->Combine(
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
      return status_handler->Combine(status);
    }
  } else {
    *result = translated_result;
  }

  if (mlir::failed(module->verifyInvariants())) {
    return status_handler->Combine(
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
