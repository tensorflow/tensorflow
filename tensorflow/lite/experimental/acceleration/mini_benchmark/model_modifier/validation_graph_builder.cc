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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/validation_graph_builder.h"

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/grafter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::GetRoot;

absl::Status ValidationGraphBuilder::BuildIntermediateModel(
    FlatBufferBuilder* fbb) {
  fbb_.Reset();
  auto model = MakeModel(/* intermediate_only */ true,
                         /* subgraph_with_golden_outputs */ nullptr);
  if (!model.ok()) {
    return model.status();
  }
  fbb_.Finish(*model, "TFL3");
  std::vector<const Model*> models{
      main_model_, flatbuffers::GetRoot<Model>(fbb_.GetBufferPointer())};
  std::vector<std::string> subgraph_names_not_important(2);
  return CombineModels(fbb, models, subgraph_names_not_important, schema_);
}

absl::Status ValidationGraphBuilder::BuildFinalModel(
    FlatBufferBuilder* fbb, Subgraph* subgraph_with_golden_outputs) {
  fbb_.Reset();
  auto model =
      MakeModel(/* intermediate_only */ false, subgraph_with_golden_outputs);
  if (!model.ok()) {
    return model.status();
  }
  fbb_.Finish(*model, "TFL3");
  std::vector<const Model*> models{
      main_model_, GetRoot<Model>(fbb_.GetBufferPointer()), validation_model_};
  std::vector<std::string> subgraph_names;
  auto main_subgraph_name = main_model_->subgraphs()->Get(0)->name();
  subgraph_names.push_back(main_subgraph_name ? main_subgraph_name->str() : "");
  subgraph_names.push_back("VALIDATION:main");
  subgraph_names.push_back("VALIDATION:metrics");
  return CombineModels(fbb, models, subgraph_names, schema_);
}

absl::StatusOr<flatbuffers::Offset<Model>> ValidationGraphBuilder::MakeModel(
    bool intermediate_only, Subgraph* subgraph_with_golden_outputs) {
  TensorInfo tensor_info;
  auto operator_codes = OperatorCodes();
  if (!operator_codes.ok()) {
    return operator_codes.status();
  }
  auto subgraphs = SubGraphs(intermediate_only, &tensor_info);
  if (!subgraphs.ok()) {
    return subgraphs.status();
  }
  auto buffers =
      Buffers(intermediate_only, tensor_info, subgraph_with_golden_outputs);
  if (!buffers.ok()) {
    return buffers.status();
  }
  return CreateModel(fbb_, kModelVersion, *operator_codes, *subgraphs,
                     fbb_.CreateString("validation"), *buffers,
                     /* metadata_buffer */ 0, /* metadata */ 0,
                     /* signature_defs */ 0);
}

absl::StatusOr<
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>>
ValidationGraphBuilder::OperatorCodes() {
#define RET_CHECK_INDEX(constant, code_index)                              \
  do {                                                                     \
    if ((constant) != (code_index)) {                                      \
      return absl::InternalError(absl::StrFormat(                          \
          "Operator code indexing mismatch %s (%d) != %s (%d)", #constant, \
          (constant), #code_index, (code_index)));                         \
    }                                                                      \
  } while (0)
  std::vector<flatbuffers::Offset<OperatorCode>> codes;
  RET_CHECK_INDEX(kCallOperatorCode, codes.size());
  codes.push_back(CreateOperatorCode(fbb_, BuiltinOperator_CUSTOM,
                                     fbb_.CreateString("validation/call")));
  RET_CHECK_INDEX(kDequantizeOperatorCode, codes.size());
  codes.push_back(CreateOperatorCode(fbb_, BuiltinOperator_DEQUANTIZE));
  RET_CHECK_INDEX(kDecodeJpegOperatorCode, codes.size());
  codes.push_back(
      CreateOperatorCode(fbb_, BuiltinOperator_CUSTOM,
                         fbb_.CreateString("validation/decode_jpeg")));
  return fbb_.CreateVector(codes);
#undef RET_CHECK_INDEX
}

absl::StatusOr<
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Tensor>>>>
ValidationGraphBuilder::Tensors(bool intermediate_only,
                                TensorInfo* tensor_info) {
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  int buffer_count = 0;

  // Copy tensors from a source subgraph, overriding the batch_size where
  // necessary (the called subgraph always uses batch size 1, the calling
  // subgraph always uses batch size equal jpeg_data_.size()).
  auto copy =
      [&tensors, this, &buffer_count](
          const SubGraph* from_subgraph,
          const flatbuffers::Vector<int32_t>* indices,
          std::vector<int32_t>* store_indices_into, int batch_size,
          const std::string prefix = "",
          std::function<absl::StatusOr<bool>(const Tensor*, int)> filter =
              nullptr) -> absl::Status {
    int counter = 0;
    for (auto index = indices->cbegin(); index != indices->cend();
         index++, counter++) {
      const Tensor* tensor = from_subgraph->tensors()->Get(*index);
      if (filter) {
        auto statusor = filter(tensor, counter);
        if (!statusor.ok()) {
          return statusor.status();
        } else if (!statusor.value()) {
          store_indices_into->push_back(kSkippedIndex);
          continue;
        }
      }
      std::vector<int32_t> shape{tensor->shape()->cbegin(),
                                 tensor->shape()->cend()};
      if (shape.size() >= 2 && shape[0] == 1 && batch_size > 0) {
        shape[0] = batch_size;
      }
      std::vector<int32_t> shape_signature;
      if (tensor->shape_signature()) {
        shape_signature.assign(tensor->shape_signature()->cbegin(),
                               tensor->shape_signature()->cend());
        if (shape_signature.size() >= 2 && shape_signature[0] == 1 &&
            batch_size > 0) {
          shape_signature[0] = batch_size;
        }
      }
      auto quantization_parameters = helper_.CopyTable(
          "tflite.QuantizationParameters", tensor->quantization());
      if (!quantization_parameters.ok()) {
        return quantization_parameters.status();
      }
      auto sparsity_parameters =
          helper_.CopyTable("tflite.SparsityParameters", tensor->sparsity());
      if (!sparsity_parameters.ok()) {
        return sparsity_parameters.status();
      }
      store_indices_into->push_back(tensors.size());
      std::string name = tensor->name()->str();
      if (!prefix.empty() && name.find(prefix) != 0) {  // NOLINT
        name = prefix + name;
      }
      tensors.push_back(CreateTensor(
          fbb_, fbb_.CreateVector(shape), tensor->type(), buffer_count,
          fbb_.CreateString(name), *quantization_parameters,
          tensor->is_variable(), *sparsity_parameters,
          shape_signature.empty() ? 0 : fbb_.CreateVector(shape_signature)));
      buffer_count++;
    }
    return absl::OkStatus();
  };
  // Input image, jpeg data.
  tensor_info->jpeg_images.push_back(tensors.size());
  DynamicBuffer jpeg_buffer;
  for (int i = 0; i < jpeg_data_.size(); i++) {
    jpeg_buffer.AddString(jpeg_data_[i].data(), jpeg_data_[i].size());
  }
  tensor_info->jpeg_buffer_length =
      jpeg_buffer.WriteToBuffer(&(tensor_info->jpeg_buffer_contents));
  tensors.push_back(CreateTensor(fbb_,
                                 fbb_.CreateVector(std::vector<int32_t>{
                                     static_cast<int32_t>(jpeg_data_.size())}),
                                 TensorType::TensorType_STRING, buffer_count,
                                 fbb_.CreateString("call/jpeg_images")));
  buffer_count++;

  // Input image.
  const SubGraph* main_subgraph = main_model_->subgraphs()->Get(0);
  const Tensor* input_tensor =
      main_subgraph->tensors()->Get(main_subgraph->inputs()->Get(0));
  tensor_info->jpeg_height = input_tensor->shape()->Get(1);
  tensor_info->jpeg_width = input_tensor->shape()->Get(2);
  if (input_tensor->type() == TensorType_FLOAT32) {
    // Quantized.
    std::vector<int32_t> input_shape{input_tensor->shape()->cbegin(),
                                     input_tensor->shape()->cend()};
    input_shape[0] = static_cast<int32_t>(jpeg_data_.size());
    tensor_info->quantized_images.push_back(tensors.size());
    tensors.push_back(CreateTensor(
        fbb_, fbb_.CreateVector(input_shape), TensorType::TensorType_UINT8,
        buffer_count, fbb_.CreateString("call/quant_image"),
        CreateQuantizationParameters(
            fbb_, 0, 0, fbb_.CreateVector(std::vector<float>{scale_}),
            fbb_.CreateVector(std::vector<int64_t>{zero_point_}))));
    buffer_count++;
    // Float.
    tensor_info->float_images.push_back(tensors.size());
    tensors.push_back(CreateTensor(fbb_, fbb_.CreateVector(input_shape),
                                   TensorType::TensorType_FLOAT32, buffer_count,
                                   fbb_.CreateString("call/float_image")));
    buffer_count++;
  } else {
    // Quantized only.
    auto status = copy(main_model_->subgraphs()->Get(0),
                       main_model_->subgraphs()->Get(0)->inputs(),
                       &tensor_info->quantized_images, jpeg_data_.size());
    if (!status.ok()) {
      return status;
    }
  }

  // Validation inputs, actual.
  auto status = copy(main_model_->subgraphs()->Get(0),
                     main_model_->subgraphs()->Get(0)->outputs(),
                     &tensor_info->main_outputs, jpeg_data_.size());
  if (!status.ok()) {
    return status;
  }
  if (intermediate_only) {
    return fbb_.CreateVector(tensors);
  }
  // Validation inputs, golden.
  tensor_info->validation_inputs = tensor_info->main_outputs;
  status = copy(main_model_->subgraphs()->Get(0),
                main_model_->subgraphs()->Get(0)->outputs(),
                &tensor_info->validation_inputs, jpeg_data_.size());
  if (!status.ok()) {
    return status;
  }
  // Entrypoint inputs. Golden first (validator relies on this).
  for (int i = tensor_info->validation_inputs.size() / 2;
       i < tensor_info->validation_inputs.size(); i++) {
    tensor_info->entrypoint_inputs.push_back(tensor_info->validation_inputs[i]);
  }
  tensor_info->entrypoint_inputs.push_back(tensor_info->jpeg_images[0]);
  // Validation inputs, dequantized.
  status = copy(
      validation_model_->subgraphs()->Get(0),
      validation_model_->subgraphs()->Get(0)->inputs(),
      &tensor_info->dequantized_validation_inputs, jpeg_data_.size(), "",
      [&tensors, &tensor_info, this](const Tensor* validation_model_input,
                                     int i) -> absl::StatusOr<bool> {
        // validation_model_input is the tensor for metrics calculation.
        // validation_graph_input is the under-construction graph will be
        // given to the metrics calculation but need to be dequantized
        // first.
        const Tensor* validation_graph_input = flatbuffers::GetTemporaryPointer(
            fbb_, tensors[tensor_info->validation_inputs[i]]);
        if (validation_model_input->type() == TensorType_FLOAT32 &&
            (validation_graph_input->type() == TensorType_UINT8 ||
             validation_graph_input->type() == TensorType_INT8)) {
          return true;
        } else if (validation_model_input->type() !=
                   validation_graph_input->type()) {
          const char* name = "(null)";
          if (validation_model_input->name()) {
            name = validation_model_input->name()->c_str();
          }
          return absl::InvalidArgumentError(
              absl::StrFormat("Validation model input %s with type %d is "
                              "incompatible with main model output type %d",
                              name, validation_model_input->type(),
                              validation_graph_input->type()));
        } else {
          return false;
        }
      });
  if (!status.ok()) {
    return status;
  }
  // Validation outputs.
  status =
      copy(validation_model_->subgraphs()->Get(0),
           validation_model_->subgraphs()->Get(0)->outputs(),
           &tensor_info->validation_outputs, jpeg_data_.size(), metric_prefix_);
  if (!status.ok()) {
    return status;
  }
  // Outputs from entrypoint graph
  // Actuals first (validator relies on this);
  for (int i = 0; i < tensor_info->validation_inputs.size() / 2; i++) {
    tensor_info->entrypoint_outputs.push_back(
        tensor_info->validation_inputs[i]);
  }
  // Metrics.
  for (int i = 0; i < tensor_info->validation_outputs.size(); i++) {
    tensor_info->entrypoint_outputs.push_back(
        tensor_info->validation_outputs[i]);
  }
  return fbb_.CreateVector(tensors);
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
ValidationGraphBuilder::CallOpCustomOptions(int subgraph) {
  flexbuffers::Builder fbb;
  fbb.Map([&] {
    fbb.Int("subgraph_index", subgraph);
    fbb.Int("loop_count", static_cast<int32_t>(jpeg_data_.size()));
  });
  fbb.Finish();
  return fbb_.CreateVector(fbb.GetBuffer());
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
ValidationGraphBuilder::JpegOpCustomOptions(int height, int width,
                                            int channels) {
  flexbuffers::Builder fbb;
  fbb.Map([&] {
    fbb.Int("height", height);
    fbb.Int("width", width);
    fbb.Int("channels", channels);
    fbb.Int("num_images", jpeg_data_.size());
  });
  fbb.Finish();
  return fbb_.CreateVector(fbb.GetBuffer());
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Operator>>>
ValidationGraphBuilder::Operators(bool intermediate_only,
                                  const TensorInfo& tensor_info) {
  std::vector<flatbuffers::Offset<Operator>> ops;
  // Jpeg decode.
  ops.push_back(CreateOperator(
      fbb_, kDecodeJpegOperatorCode, fbb_.CreateVector(tensor_info.jpeg_images),
      fbb_.CreateVector(tensor_info.quantized_images),
      tflite::BuiltinOptions_NONE, 0,
      JpegOpCustomOptions(tensor_info.jpeg_height, tensor_info.jpeg_width,
                          jpeg_output_channels_)));
  if (!tensor_info.float_images.empty()) {
    // Dequantize.
    ops.push_back(
        CreateOperator(fbb_, kDequantizeOperatorCode,
                       fbb_.CreateVector(tensor_info.quantized_images),
                       fbb_.CreateVector(tensor_info.float_images),
                       BuiltinOptions_DequantizeOptions, 0));
    // Call main model.
    ops.push_back(CreateOperator(
        fbb_, kCallOperatorCode, fbb_.CreateVector(tensor_info.float_images),
        fbb_.CreateVector(tensor_info.main_outputs),
        tflite::BuiltinOptions_NONE, 0, CallOpCustomOptions(kMainSubgraphIndex),
        tflite::CustomOptionsFormat_FLEXBUFFERS));
  } else {
    // Call main model.
    ops.push_back(CreateOperator(
        fbb_, kCallOperatorCode,
        fbb_.CreateVector(tensor_info.quantized_images),
        fbb_.CreateVector(tensor_info.main_outputs),
        tflite::BuiltinOptions_NONE, 0, CallOpCustomOptions(kMainSubgraphIndex),
        tflite::CustomOptionsFormat_FLEXBUFFERS));
  }
  if (intermediate_only) {
    return fbb_.CreateVector(ops);
  }
  // Call validation model.
  std::vector<int32_t> validation_input_indices;
  for (int i = 0; i < tensor_info.dequantized_validation_inputs.size(); i++) {
    int32_t validation_input_index;
    if (tensor_info.dequantized_validation_inputs[i] == kSkippedIndex) {
      validation_input_index = tensor_info.validation_inputs[i];
    } else {
      validation_input_index = tensor_info.dequantized_validation_inputs[i];
      std::vector<int32_t> dequantize_inputs{tensor_info.validation_inputs[i]};
      std::vector<int32_t> dequantize_outputs{
          tensor_info.dequantized_validation_inputs[i]};
      ops.push_back(CreateOperator(fbb_, kDequantizeOperatorCode,
                                   fbb_.CreateVector(dequantize_inputs),
                                   fbb_.CreateVector(dequantize_outputs),
                                   BuiltinOptions_DequantizeOptions, 0));
    }
    validation_input_indices.push_back(validation_input_index);
  }
  ops.push_back(CreateOperator(
      fbb_, kCallOperatorCode, fbb_.CreateVector(validation_input_indices),
      fbb_.CreateVector(tensor_info.validation_outputs),
      tflite::BuiltinOptions_NONE, 0,
      CallOpCustomOptions(kValidationSubgraphIndex),
      tflite::CustomOptionsFormat_FLEXBUFFERS));
  return fbb_.CreateVector(ops);
}

absl::StatusOr<
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<SubGraph>>>>
ValidationGraphBuilder::SubGraphs(bool intermediate_only,
                                  TensorInfo* tensor_info) {
  auto tensors = Tensors(intermediate_only, tensor_info);
  if (!tensors.ok()) {
    return tensors.status();
  }
  std::vector<flatbuffers::Offset<SubGraph>> graphs;
  if (intermediate_only) {
    graphs.push_back(CreateSubGraph(
        fbb_, *tensors, fbb_.CreateVector(tensor_info->jpeg_images),
        fbb_.CreateVector(tensor_info->main_outputs),
        Operators(intermediate_only, *tensor_info), fbb_.CreateString("call")));
  } else {
    graphs.push_back(CreateSubGraph(
        fbb_, *tensors, fbb_.CreateVector(tensor_info->entrypoint_inputs),
        fbb_.CreateVector(tensor_info->entrypoint_outputs),
        Operators(intermediate_only, *tensor_info), fbb_.CreateString("call")));
  }
  return fbb_.CreateVector(graphs);
}

absl::StatusOr<
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>>
ValidationGraphBuilder::Buffers(bool intermediate_only,
                                const TensorInfo& tensor_info,
                                Subgraph* subgraph_with_golden_outputs) {
  std::vector<flatbuffers::Offset<Buffer>> buffers;

  // The buffers created in this method map 1:1 to the
  // tensors created in Tensors() - a tensor at index X
  // uses buffer at index X. The numbering is checked along
  // the way using the RET_CHECK_INDEX macro below.
#define RET_CHECK_INDEX(tensor_index, buffer_index)                         \
  do {                                                                      \
    if ((tensor_index) != (buffer_index)) {                                 \
      return absl::InternalError(absl::StrFormat(                           \
          "%s:%d, Tensor/buffer indexing mismatch %s (%d) != %s (%d)",      \
          __FILE__, __LINE__, #tensor_index, (tensor_index), #buffer_index, \
          (buffer_index)));                                                 \
    }                                                                       \
  } while (0)

  // Jpeg input.
  RET_CHECK_INDEX(tensor_info.jpeg_images.size(), 1);
  RET_CHECK_INDEX(tensor_info.jpeg_images[0], buffers.size());
  std::vector<uint8_t> jpeg_buffer_vec{
      reinterpret_cast<const uint8_t*>(tensor_info.jpeg_buffer_contents),
      reinterpret_cast<const uint8_t*>(tensor_info.jpeg_buffer_contents) +
          tensor_info.jpeg_buffer_length};
  buffers.push_back(CreateBuffer(fbb_, fbb_.CreateVector(jpeg_buffer_vec)));

  // Decoded and dequantized image.
  RET_CHECK_INDEX(tensor_info.quantized_images.size(), 1);
  RET_CHECK_INDEX(tensor_info.quantized_images[0], buffers.size());
  buffers.push_back(CreateBuffer(fbb_));
  if (!tensor_info.float_images.empty()) {
    RET_CHECK_INDEX(tensor_info.float_images.size(), 1);
    RET_CHECK_INDEX(tensor_info.float_images[0], buffers.size());
    buffers.push_back(CreateBuffer(fbb_));
  }

  // Main graph outputs / first half of validation inputs.
  auto main_subgraph = main_model_->subgraphs()->Get(0);
  RET_CHECK_INDEX(main_subgraph->outputs()->size(),
                  tensor_info.main_outputs.size());
  int main_output_index = 0;
  int validation_graph_input_index = 0;
  for (auto i = main_subgraph->outputs()->cbegin();
       i != main_subgraph->outputs()->cend(); i++) {
    RET_CHECK_INDEX(tensor_info.main_outputs[main_output_index],
                    buffers.size());
    main_output_index++;
    if (!intermediate_only) {
      RET_CHECK_INDEX(
          tensor_info.validation_inputs[validation_graph_input_index],
          buffers.size());
      validation_graph_input_index++;
    }
    auto t = main_subgraph->tensors()->Get(*i);
    auto status = helper_.CopyTableToVector(
        "tflite.Buffer", main_model_->buffers()->Get(t->buffer()), &buffers);
    if (!status.ok()) {
      return status;
    }
  }
  if (intermediate_only) {
    return fbb_.CreateVector(buffers);
  }

  // Golden outputs / second half of validation inputs.
  RET_CHECK_INDEX(tensor_info.validation_inputs.size(),
                  validation_graph_input_index +
                      subgraph_with_golden_outputs->outputs().size());
  for (auto i : subgraph_with_golden_outputs->outputs()) {
    RET_CHECK_INDEX(tensor_info.validation_inputs[validation_graph_input_index],
                    buffers.size());
    validation_graph_input_index++;
    auto t = subgraph_with_golden_outputs->tensor(i);
    if (!use_ondevice_cpu_for_golden_) {
      std::vector<uint8_t> output_data{
          reinterpret_cast<const uint8_t*>(t->data.raw),
          reinterpret_cast<const uint8_t*>(t->data.raw + t->bytes)};
      buffers.push_back(CreateBuffer(fbb_, fbb_.CreateVector(output_data)));
    } else {
      buffers.push_back(CreateBuffer(fbb_));
    }
  }

  auto validation_model_subgraph = validation_model_->subgraphs()->Get(0);
  // Dequantized validation inputs.
  RET_CHECK_INDEX(tensor_info.dequantized_validation_inputs.size(),
                  validation_model_subgraph->inputs()->size());
  int validation_graph_dequantized_input_index = 0;
  for (auto i = validation_model_subgraph->inputs()->cbegin();
       i != validation_model_subgraph->inputs()->cend(); i++) {
    if (tensor_info.dequantized_validation_inputs
            [validation_graph_dequantized_input_index] == kSkippedIndex) {
      validation_graph_dequantized_input_index++;
      continue;
    }
    RET_CHECK_INDEX(tensor_info.dequantized_validation_inputs
                        [validation_graph_dequantized_input_index],
                    buffers.size());
    validation_graph_dequantized_input_index++;
    auto t = validation_model_subgraph->tensors()->Get(*i);
    auto status = helper_.CopyTableToVector(
        "tflite.Buffer", validation_model_->buffers()->Get(t->buffer()),
        &buffers);
    if (!status.ok()) {
      return status;
    }
  }

  // Validation outputs.
  RET_CHECK_INDEX(tensor_info.validation_outputs.size(),
                  validation_model_subgraph->outputs()->size());
  int validation_graph_output_index = 0;
  for (auto i = validation_model_subgraph->outputs()->cbegin();
       i != validation_model_subgraph->outputs()->cend(); i++) {
    RET_CHECK_INDEX(
        tensor_info.validation_outputs[validation_graph_output_index],
        buffers.size());
    validation_graph_output_index++;
    auto t = validation_model_subgraph->tensors()->Get(*i);
    auto status = helper_.CopyTableToVector(
        "tflite.Buffer", validation_model_->buffers()->Get(t->buffer()),
        &buffers);
    if (!status.ok()) {
      return status;
    }
  }
  return fbb_.CreateVector(buffers);
#undef RET_CHECK_INDEX
}

}  // namespace acceleration

}  // namespace tflite
