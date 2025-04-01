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

#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace lite {

namespace {

using llvm::StringRef;

// Convert op represented in TFLite builtin_code to its corresponding MLIR
// OperationName.
void TfLiteBuiltinOpToMlir(const BuiltinOperatorSet& tflite_builtin_codes,
                           absl::flat_hash_set<std::string>& mlir_op_names) {
  for (const auto& entry : tflite_builtin_codes) {
    StringRef tflite_op_name = EnumNameBuiltinOperator(entry);
    std::string mlir_name = llvm::Twine("tfl.", tflite_op_name.lower()).str();
    mlir_op_names.insert(std::move(mlir_name));
  }
}

std::string TfLiteToMlir(absl::string_view tflite_op_name) {
  StringRef op_name(tflite_op_name.data(), tflite_op_name.size());
  return op_name.lower();
}

std::unique_ptr<tflite::ModelT> CreateMutableModelFromFile(
    const tflite::Model* input_model) {
  auto copied_model = std::make_unique<tflite::ModelT>();
  input_model->UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}
}  // namespace

// TODO(b/214314076): Support MLIR model as an input for the C++ dynamic range
// quantization API
absl::Status QuantizeWeights(
    flatbuffers::FlatBufferBuilder* builder, const tflite::Model* input_model,
    const tflite::TensorType& inference_type,
    const absl::flat_hash_set<std::string>& denylisted_ops,
    const CustomOpMap& custom_op_map, int64_t minimum_elements_for_weights,
    bool disable_per_channel, bool weight_only_quantization,
    bool legacy_float_scale) {
  // Translate TFLite names to mlir op names.
  absl::flat_hash_set<std::string> denylisted_mlir_op_names;
  for (auto& entry : denylisted_ops) {
    denylisted_mlir_op_names.insert(TfLiteToMlir(entry));
  }

  DialectRegistry registry;
  MLIRContext context(registry);
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              /*propagate=*/true);

  // Import input_model to a MLIR module
  flatbuffers::FlatBufferBuilder input_builder;
  flatbuffers::Offset<tflite::Model> input_model_location = tflite::Model::Pack(
      input_builder, CreateMutableModelFromFile(input_model).get());
  tflite::FinishModelBuffer(input_builder, input_model_location);

  std::string serialized_model(
      reinterpret_cast<const char*>(input_builder.GetBufferPointer()),
      input_builder.GetSize());

  OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      serialized_model, &context, UnknownLoc::get(&context));

  // Apply quantization passes.
  PassManager pm((*module)->getName(), OpPassManager::Nesting::Implicit);
  quant::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tflite::TflTypeToTfType(inference_type);
  quant_specs.weight_quantization = true;
  quant_specs.weight_only_quantization = weight_only_quantization;
  quant_specs.minimum_elements_for_weights = minimum_elements_for_weights;
  quant_specs.disable_per_channel = disable_per_channel;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = denylisted_mlir_op_names;
  for (const auto& entry : custom_op_map) {
    quant_specs.custom_map[entry.first].quantizable_input_indices =
        entry.second.quantizable_input_indices;
    quant_specs.custom_map[entry.first].is_weight_only =
        entry.second.is_weight_only;
    quant_specs.custom_map[entry.first].no_side_effect =
        entry.second.no_side_effect;
  }

  if (quant_specs.inference_type == tensorflow::DT_INT8)
    quant_specs.inference_type = tensorflow::DT_QINT8;

  if (!(quant_specs.inference_type == tensorflow::DT_HALF ||
        quant_specs.inference_type == tensorflow::DT_QINT8)) {
    LOG(ERROR) << "Couldn't apply dynamic range quantization since unsupported "
                  "inference_type is passed.";
    return absl::InvalidArgumentError(
        "Quantize weights transformation failed.");
  }

  llvm::dbgs() << "weight_quantization: " << true
               << ", weight_only_quantization: "
               << quant_specs.weight_only_quantization << ", mlir_quantizer: "
               << quant_specs.enable_mlir_dynamic_range_quantizer
               << ", inference_type: " << quant_specs.inference_type << "\n";
  Builder mlir_builder(&context);

  tensorflow::AddDynamicRangeQuantizationPasses(
      mlir::TFL::PassConfig(quant_specs), pm);

  if (failed(pm.run(module.get()))) {
    absl::string_view err = statusHandler.ConsumeStatus().message();
    LOG(ERROR) << "Failed to quantize: " << err;
    return absl::InvalidArgumentError(
        "Quantize weights transformation failed.");
  }

  // Export the results to the builder
  std::string result;
  tflite::FlatbufferExportOptions options;
  options.converter_flags.set_force_select_tf_ops(false);
  options.converter_flags.set_enable_select_tf_ops(true);
  options.converter_flags.set_allow_custom_ops(true);
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &result)) {
    LOG(ERROR) << "Failed to export MLIR to flatbuffer.";
    return absl::InvalidArgumentError(
        "Quantize weights transformation failed.");
  }
  builder->PushFlatBuffer(reinterpret_cast<const uint8_t*>(result.data()),
                          result.size());

  return absl::OkStatus();
}

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             int64_t weights_min_num_elements,
                             bool use_hybrid_evaluation) {
  return QuantizeWeights(
      builder, input_model,
      /*inference_type=*/tflite::TensorType_INT8,
      /*denylisted_ops=*/{},
      /*custom_op_map=*/{},
      /*minimum_elements_for_weights=*/weights_min_num_elements,
      /*disable_per_channel=*/false,
      /*weight_only_quantization=*/!use_hybrid_evaluation,
      /*legacy_float_scale=*/true);
}

// In MLIR use_updated_hybrid_scheme = true means per-channel operation.
absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             BufferType quant_type,
                             bool use_updated_hybrid_scheme) {
  tflite::TensorType inference_type;
  switch (quant_type) {
    case BufferType::QUANTIZED_FLOAT16:
      inference_type = tflite::TensorType_FLOAT16;
      break;
    default:
      inference_type = tflite::TensorType_INT8;
  }
  return QuantizeWeights(builder, input_model, inference_type,
                         /*denylisted_ops=*/{},
                         /*custom_op_map=*/{},
                         /*minimum_elements_for_weights=*/1024,
                         /*disable_per_channel=*/!use_updated_hybrid_scheme,
                         /*weight_only_quantization=*/false,
                         /*legacy_float_scale=*/true);
}

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const tflite::Model* input_model,
                             int64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             bool use_updated_hybrid_scheme,
                             const BuiltinOperatorSet& op_denylist) {
  const tflite::TensorType inference_type = tflite::TensorType_INT8;

  absl::flat_hash_set<std::string> mlir_op_denylist;
  TfLiteBuiltinOpToMlir(op_denylist, mlir_op_denylist);

  return QuantizeWeights(
      builder, input_model, inference_type,
      /*denylisted_ops=*/mlir_op_denylist,
      /*custom_op_map=*/custom_op_map,
      /*minimum_elements_for_weights=*/weights_min_num_elements,
      /*disable_per_channel=*/!use_updated_hybrid_scheme,
      /*weight_only_quantization=*/false,
      /*legacy_float_scale=*/true);
}

}  // namespace lite
}  // namespace mlir
