/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedder.h"

#include <cstdint>
#include <functional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_common.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"

namespace fb = flatbuffers;

namespace tflite {
namespace acceleration {

namespace {
// Class for building the validation entry-point graph that calls into the main
// graph and a metrics graph. Like this (boxes are tensors with plural names
// meaning possibly multiple tensors, arrows are ops and numbers in parentheses
// are subgraph indices):
// +--------------------------------------+
// | Graph created by this class (1)      |
// |                                      |
// | +-----------input-+                  |
// | |jpeg input       |                  |
// | +-----+-----------+                  |
// |       |                              |
// |       | decode                       |
// |       v                              |
// | +-----+-----------+                  |
// | |quantized image  |                  |
// | +-----+-----------+                  |  +-----------------------+
// |       |                              |  |'main_model' (0)       |
// |       | dequantize (optional)        |  | +---------------+     |
// |       v                              |  | |input          +---+ |
// | +-----+-----------+                  |  | +---------------+   | |
// | |float image      |                  |  |                     ~ |
// | +-----+-----------+                  |  | +---------------+   | |
// |       |  call                        |  | |outputs        +<--+ |
// |       +<------------------------------->+ +---------------+     |
// |       v                              |  |                       |
// | +-----+-----output+ +---------input+ |  +-----------------------+
// | |actual outputs   | |golden outputs| |
// | +-----+-----------+ +-----------+--+ |
// |       |                         |    |
// |       | dequantize (optional)   |    |
// |       |                         |    |
// | +-----+-------------------------+-+  |
// | | dequantized actual and golden   |  |
// | | outputs (validation inputs)     |  |
// | +-----+---------------------------+  |  +-----------------------+
// |       |  call                        |  |'validation model' (2) |
// |       +<------------------------------->+                       |
// |       v                              |  | +---------------+     |
// | +-----+-----output+                  |  | |inputs         +---+ |
// | |results          |                  |  | +---------------+   | |
// | +-----------------+                  |  |                     ~ |
// |                                      |  | +---------------+   | |
// |                                      |  | |outputs        +<--+ |
// |                                      |  | +---------------+     |
// |                                      |  |                       |
// +--------------------------------------+  +-----------------------+
//
// It's important the 'main_model' has subgraph index 0 so that it is used as
// the primary subgraph by the TFLite interpreter. The other indices are
// arbitrary.
// TODO(b/172541832): Handle a main model with more than one subgraph.
//
// Note that the jpeg input is marked as an input in this graph, as TFLite
// graphs must have inputs. However, it will be pre-filled from the jpeg_data
// and doesn't need to be filled by the user of the model.
//

constexpr char kMetricPrefix[] = "metrics/";

class ValidationGraphBuilder {
 public:
  ValidationGraphBuilder(const Model* main_model,
                         std::vector<std::string> jpeg_data,
                         int32_t jpeg_output_channels, float scale,
                         int64_t zero_point, const Model* validation_model,
                         const reflection::Schema* schema,
                         bool use_ondevice_cpu_for_golden);

  // Builds the part of the model drawn above until the call to the validation
  // graph. The model is used to generate golden outputs. Calls Finish on the
  // FlatbufferBuilder.
  absl::Status BuildIntermediateModel(fb::FlatBufferBuilder* fbb);
  // Builds the whole model as drawn above. The subgraph_with_golden_outputs
  // should be the result of invoking subgraph 1 on the output of
  // BuildIntermediateModel(). Calls Finish on the FlatbufferBuilder.
  absl::Status BuildFinalModel(fb::FlatBufferBuilder* fbb,
                               Subgraph* subgraph_with_golden_outputs);

  ValidationGraphBuilder(const ValidationGraphBuilder&) = delete;
  ValidationGraphBuilder& operator=(const ValidationGraphBuilder&) = delete;

 private:
  static const int32_t kModelVersion = 3;
  static const int32_t kSkippedIndex = -1;
  // Operator code numbering.
  static const int32_t kCallOperatorCode = 0;
  static const int32_t kDequantizeOperatorCode = 1;
  static const int32_t kDecodeJpegOperatorCode = 2;
  // Subgraph numbering.
  static const int32_t kMainSubgraphIndex = 0;
  static const int32_t kValidationSubgraphIndex = 2;

  // Allocation of tensors, for communication between methods that create the
  // tensors, the operations and the buffers.
  // (Some of these vectors will always contain only one element, but using the
  // same type for them simplifies the code a lot).
  struct TensorInfo {
    std::vector<int32_t> entrypoint_inputs;
    std::vector<int32_t> entrypoint_outputs;
    std::vector<int32_t> jpeg_images;

    // With float main model, both quantized_images and float_images are set,
    // and float_images is the same as main input. With a quantized model
    // only quantized_images is set and it's the same as main input.
    std::vector<int32_t> quantized_images;
    std::vector<int32_t> float_images;

    std::vector<int32_t> main_outputs;  // First half of validation_inputs.
    std::vector<int32_t> validation_inputs;
    // With a float model, validation_inputs is used directly. With a quantized
    // model, the inputs are first dequantized.
    // Some models have a mixture of quantized outputs that need to be
    // dequantized to floats; and integer outputs. For integer outputs
    // kSkippedIndex is used.
    std::vector<int32_t> dequantized_validation_inputs;
    std::vector<int32_t> validation_outputs;

    char* jpeg_buffer_contents = nullptr;
    int32_t jpeg_buffer_length = -1;
    int32_t jpeg_height = -1;
    int32_t jpeg_width = -1;

    ~TensorInfo() { free(jpeg_buffer_contents); }
  };

  absl::StatusOr<fb::Offset<Model>> MakeModel(
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

  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<OperatorCode>>>>
  OperatorCodes() {
#define RET_CHECK_INDEX(constant, code_index)                              \
  do {                                                                     \
    if ((constant) != (code_index)) {                                      \
      return absl::InternalError(absl::StrFormat(                          \
          "Operator code indexing mismatch %s (%d) != %s (%d)", #constant, \
          (constant), #code_index, (code_index)));                         \
    }                                                                      \
  } while (0)
    std::vector<fb::Offset<OperatorCode>> codes;
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

  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<Tensor>>>> Tensors(
      bool intermediate_only, TensorInfo* tensor_info) {
    std::vector<fb::Offset<Tensor>> tensors;
    int buffer_count = 0;

    // Copy tensors from a source subgraph, overriding the batch_size where
    // necessary (the called subgraph always uses batch size 1, the calling
    // subgraph always uses batch size equal jpeg_data_.size()).
    auto copy =
        [&tensors, this, &buffer_count](
            const SubGraph* from_subgraph, const fb::Vector<int32_t>* indices,
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
    tensors.push_back(CreateTensor(
        fbb_,
        fbb_.CreateVector(
            std::vector<int32_t>{static_cast<int32_t>(jpeg_data_.size())}),
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
      tensors.push_back(CreateTensor(
          fbb_, fbb_.CreateVector(input_shape), TensorType::TensorType_FLOAT32,
          buffer_count, fbb_.CreateString("call/float_image")));
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
      tensor_info->entrypoint_inputs.push_back(
          tensor_info->validation_inputs[i]);
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
          // given to the metrics calculation but need to be dequantized first.
          const Tensor* validation_graph_input = fb::GetTemporaryPointer(
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
    status = copy(validation_model_->subgraphs()->Get(0),
                  validation_model_->subgraphs()->Get(0)->outputs(),
                  &tensor_info->validation_outputs, jpeg_data_.size(),
                  kMetricPrefix);
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

  // Create the options for the custom call op (see call.cc for the options
  // format).
  fb::Offset<fb::Vector<uint8_t>> CallOpCustomOptions(int subgraph) {
    flexbuffers::Builder fbb;
    fbb.Map([&] {
      fbb.Int("subgraph_index", subgraph);
      fbb.Int("loop_count", static_cast<int32_t>(jpeg_data_.size()));
    });
    fbb.Finish();
    return fbb_.CreateVector(fbb.GetBuffer());
  }

  // Create the options for the custom jpeg op (see decode_jpeg.cc for the
  // options format).
  fb::Offset<fb::Vector<uint8_t>> JpegOpCustomOptions(int height, int width,
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

  fb::Offset<fb::Vector<fb::Offset<Operator>>> Operators(
      bool intermediate_only, const TensorInfo& tensor_info) {
    std::vector<fb::Offset<Operator>> ops;
    // Jpeg decode.
    ops.push_back(CreateOperator(
        fbb_, kDecodeJpegOperatorCode,
        fbb_.CreateVector(tensor_info.jpeg_images),
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
      ops.push_back(CreateOperator(fbb_, kCallOperatorCode,
                                   fbb_.CreateVector(tensor_info.float_images),
                                   fbb_.CreateVector(tensor_info.main_outputs),
                                   tflite::BuiltinOptions_NONE, 0,
                                   CallOpCustomOptions(kMainSubgraphIndex),
                                   tflite::CustomOptionsFormat_FLEXBUFFERS));
    } else {
      // Call main model.
      ops.push_back(
          CreateOperator(fbb_, kCallOperatorCode,
                         fbb_.CreateVector(tensor_info.quantized_images),
                         fbb_.CreateVector(tensor_info.main_outputs),
                         tflite::BuiltinOptions_NONE, 0,
                         CallOpCustomOptions(kMainSubgraphIndex),
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
        std::vector<int32_t> dequantize_inputs{
            tensor_info.validation_inputs[i]};
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

  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<SubGraph>>>> SubGraphs(
      bool intermediate_only, TensorInfo* tensor_info) {
    auto tensors = Tensors(intermediate_only, tensor_info);
    if (!tensors.ok()) {
      return tensors.status();
    }
    std::vector<fb::Offset<SubGraph>> graphs;
    if (intermediate_only) {
      graphs.push_back(CreateSubGraph(
          fbb_, *tensors, fbb_.CreateVector(tensor_info->jpeg_images),
          fbb_.CreateVector(tensor_info->main_outputs),
          Operators(intermediate_only, *tensor_info),
          fbb_.CreateString("call")));
    } else {
      graphs.push_back(CreateSubGraph(
          fbb_, *tensors, fbb_.CreateVector(tensor_info->entrypoint_inputs),
          fbb_.CreateVector(tensor_info->entrypoint_outputs),
          Operators(intermediate_only, *tensor_info),
          fbb_.CreateString("call")));
    }
    return fbb_.CreateVector(graphs);
  }

  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<Buffer>>>> Buffers(
      bool intermediate_only, const TensorInfo& tensor_info,
      Subgraph* subgraph_with_golden_outputs) {
    std::vector<fb::Offset<Buffer>> buffers;

    // The buffers created in this method map 1:1 to the tensors created in
    // Tensors() - a tensor at index X uses buffer at index X. The numbering
    // is checked along the way using the RET_CHECK_INDEX macro below.
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
      RET_CHECK_INDEX(
          tensor_info.validation_inputs[validation_graph_input_index],
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

  const Model* main_model_;
  std::vector<std::string> jpeg_data_;
  int32_t jpeg_output_channels_;
  float scale_;
  int64_t zero_point_;
  const Model* validation_model_;
  const reflection::Schema* schema_;
  fb::FlatBufferBuilder fbb_;
  FlatbufferHelper helper_;
  bool use_ondevice_cpu_for_golden_;
};

// Define the static constant members. Without definition they can be used as
// compile-time constants but can not be passed by reference (e.g., used with
// absl::StrFormat).
const int32_t ValidationGraphBuilder::kModelVersion;
const int32_t ValidationGraphBuilder::kSkippedIndex;
const int32_t ValidationGraphBuilder::kCallOperatorCode;
const int32_t ValidationGraphBuilder::kDequantizeOperatorCode;
const int32_t ValidationGraphBuilder::kDecodeJpegOperatorCode;
const int32_t ValidationGraphBuilder::kMainSubgraphIndex;
const int32_t ValidationGraphBuilder::kValidationSubgraphIndex;

ValidationGraphBuilder::ValidationGraphBuilder(
    const Model* main_model, std::vector<std::string> jpeg_data,
    int32_t jpeg_output_channels, float scale, int64_t zero_point,
    const Model* validation_model, const reflection::Schema* schema,
    bool use_ondevice_cpu_for_golden)
    : main_model_(main_model),
      jpeg_data_(jpeg_data),
      jpeg_output_channels_(jpeg_output_channels),
      scale_(scale),
      zero_point_(zero_point),
      validation_model_(validation_model),
      schema_(schema),
      helper_(&fbb_, schema_),
      use_ondevice_cpu_for_golden_(use_ondevice_cpu_for_golden) {}

absl::Status ValidationGraphBuilder::BuildIntermediateModel(
    fb::FlatBufferBuilder* fbb) {
  fbb_.Reset();
  auto model = MakeModel(/* intermediate_only */ true,
                         /* subgraph_with_golden_outputs */ nullptr);
  if (!model.ok()) {
    return model.status();
  }
  fbb_.Finish(*model, "TFL3");
  std::vector<const Model*> models{main_model_,
                                   fb::GetRoot<Model>(fbb_.GetBufferPointer())};
  std::vector<std::string> subgraph_names_not_important(2);
  return CombineModels(fbb, models, subgraph_names_not_important, schema_);
}

absl::Status ValidationGraphBuilder::BuildFinalModel(
    fb::FlatBufferBuilder* fbb, Subgraph* subgraph_with_golden_outputs) {
  fbb_.Reset();
  auto model =
      MakeModel(/* intermediate_only */ false, subgraph_with_golden_outputs);
  if (!model.ok()) {
    return model.status();
  }
  fbb_.Finish(*model, "TFL3");
  std::vector<const Model*> models{main_model_,
                                   fb::GetRoot<Model>(fbb_.GetBufferPointer()),
                                   validation_model_};
  std::vector<std::string> subgraph_names;
  auto main_subgraph_name = main_model_->subgraphs()->Get(0)->name();
  subgraph_names.push_back(main_subgraph_name ? main_subgraph_name->str() : "");
  subgraph_names.push_back("VALIDATION:main");
  subgraph_names.push_back("VALIDATION:metrics");
  return CombineModels(fbb, models, subgraph_names, schema_);
}

std::string DescribeShape(const fb::Vector<int32_t>* shape) {
  std::string desc = "[";
  for (int i = 0; i < shape->size(); i++) {
    if (i != 0) {
      desc += ", ";
    }
    desc += absl::StrFormat("%d", shape->Get(i));
  }
  desc += "]";
  return desc;
}

}  // namespace

Embedder::Embedder(const Model* main_model,
                   const std::vector<std::string>& jpeg_data, float scale,
                   int64_t zero_point, const Model* validation_model,
                   const reflection::Schema* schema,
                   bool use_ondevice_cpu_for_golden)
    : main_model_(main_model),
      jpeg_data_(jpeg_data),
      scale_(scale),
      zero_point_(zero_point),
      validation_model_(validation_model),
      schema_(schema),
      use_ondevice_cpu_for_golden_(use_ondevice_cpu_for_golden) {}

absl::Status Embedder::ValidateInputs() {
#define VALIDATE(condition, ...)                                     \
  if (!(condition)) {                                                \
    return absl::InvalidArgumentError(absl::StrFormat(__VA_ARGS__)); \
  }
  VALIDATE(main_model_, "main_model may not be null");
  VALIDATE(main_model_->subgraphs()->size(), "main model must have subgraphs");
  const SubGraph* main_subgraph = main_model_->subgraphs()->Get(0);
  VALIDATE(main_subgraph->inputs()->size() == 1,
           "main subgraph must have 1 input (got %d)",
           main_subgraph->inputs()->size());
  const auto* shape =
      main_subgraph->tensors()->Get(main_subgraph->inputs()->Get(0))->shape();
  VALIDATE(shape->size() == 4,
           "main subgraph input must have 4 dimensions (got %d)",
           shape->size());
  jpeg_output_channels_ = shape->Get(3);
  VALIDATE(shape->Get(0) == 1,
           "main subgraph input must have batch size 1 (got %d)",
           shape->Get(0));
  VALIDATE(shape->Get(3) == 1 || shape->Get(3) == 3 || shape->Get(3) == 4,
           "main subgraph input must have 1 or 3 or 4 channels (got %d)",
           shape->Get(3));

  VALIDATE(!jpeg_data_.empty(), "must have at least 1 jpeg input");
  int jpeg_number = 0;
  for (const std::string& jpeg_image_data : jpeg_data_) {
    int width, height, components;
    decode_jpeg_kernel::JpegHeader header{0};
    auto status = decode_jpeg_kernel::ReadJpegHeader(
        {jpeg_image_data.data(), static_cast<int>(jpeg_image_data.size())},
        &header);
    VALIDATE(status.code == kTfLiteOk,
             "Failed to decompress jpeg data at index %d: %s", jpeg_number,
             status.error_message.c_str());
    width = header.width;
    height = header.height;
    components = header.channels;
    VALIDATE(height == shape->Get(1) && width == shape->Get(2) &&
                 (components == shape->Get(3) ||
                  // A workaround to allow RGBA channels extracted from RGB
                  // images with alpha channel as 255 (fully opaque) by default.
                  components == 3 && shape->Get(3) == 4),
             "Jpeg input at index %d has different size from input tensor "
             "(jpeg h: %d, w: %d, c: %d; tensor h: %d, w: %d, c: %d)",
             jpeg_number, height, width, components, shape->Get(1),
             shape->Get(2), shape->Get(3));
    if (components < shape->Get(3)) {
      TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                      "Jpeg input at index %d has %d channels. Lower than the "
                      "expected %d channels.",
                      jpeg_number, components, shape->Get(3));
    }
    jpeg_number++;
  }

  int main_output_count = main_subgraph->outputs()->size();
  VALIDATE(main_output_count > 0,
           "main subgraph must have at least 1 output (got %d)",
           main_output_count);
  VALIDATE(validation_model_->subgraphs()->size(),
           "validation model must have subgraphs");
  const SubGraph* validation_subgraph = validation_model_->subgraphs()->Get(0);
  int validation_input_count = validation_subgraph->inputs()->size();
  VALIDATE(
      validation_input_count == main_output_count * 2,
      "validation subgraph input count must be 2 times main subgraph output "
      "count (validation output count: %d, main subgraph output count: %d)",
      validation_input_count, main_output_count);
  for (int i = 0; i < main_output_count; i++) {
    auto main_output_tensor =
        main_subgraph->tensors()->Get(main_subgraph->outputs()->Get(i));
    auto main_output_shape = DescribeShape(main_output_tensor->shape());
    VALIDATE(main_output_shape != "[]",
             "All main outputs must be tensors, %d is a scalar", i);
    VALIDATE(main_output_tensor->name()->str().find(kMetricPrefix) != 0,
             "Main output %d name %s clashes with metrics/ prefix", i,
             main_output_tensor->name()->c_str());
    auto validation_input_shape_1 =
        DescribeShape(validation_subgraph->tensors()
                          ->Get(validation_subgraph->inputs()->Get(i))
                          ->shape());
    auto validation_input_shape_2 = DescribeShape(
        validation_subgraph->tensors()
            ->Get(validation_subgraph->inputs()->Get(main_output_count + i))
            ->shape());
    VALIDATE(main_output_shape == validation_input_shape_1,
             "Main output %d dimensions %s do not match validation input %d "
             "dimensions %s",
             i, main_output_shape, i, validation_input_shape_1);
    VALIDATE(main_output_shape == validation_input_shape_2,
             "Main output %d dimensions %s do not match validation input %d "
             "dimensions %s",
             i, main_output_shape, main_output_count + i,
             validation_input_shape_2);
  }
  int validation_output_count = validation_subgraph->outputs()->size();
  VALIDATE(validation_output_count >= 2,
           "validation output count must be at least 2 (got "
           "%d)",
           validation_output_count);
  bool seen_ok = false;
  const std::string kOk = "ok";
  const std::string kPrefixedOk = kMetricPrefix + kOk;
  std::string names = "";
  for (int i = 0; i < validation_output_count; i++) {
    const Tensor* t = validation_subgraph->tensors()->Get(
        validation_subgraph->outputs()->Get(i));
    VALIDATE(t->shape()->size(),
             "validation outputs must be tensors, %d is a scalar", i);
    seen_ok = (seen_ok || (kOk == t->name()->str()) ||
               (kPrefixedOk == t->name()->str()));
    if (i != 0) {
      names += ", ";
    }
    names += t->name()->str();
  }
  VALIDATE(seen_ok, "validation must have an output named 'ok' (saw %s)",
           names);
#undef VALIDATE
  return absl::OkStatus();
}

absl::Status Embedder::CreateModelWithEmbeddedValidation(
    fb::FlatBufferBuilder* fbb, ops::builtin::BuiltinOpResolver* resolver) {
  auto status = ValidateInputs();
  if (!status.ok()) {
    return status;
  }
  fb::FlatBufferBuilder intermediate_fbb;
  ValidationGraphBuilder builder(main_model_, jpeg_data_, jpeg_output_channels_,
                                 scale_, zero_point_, validation_model_,
                                 schema_, use_ondevice_cpu_for_golden_);
  status = builder.BuildIntermediateModel(&intermediate_fbb);
  if (!status.ok()) {
    return status;
  }
  auto intermediate_model = FlatBufferModel::VerifyAndBuildFromBuffer(
      reinterpret_cast<const char*>(intermediate_fbb.GetBufferPointer()),
      intermediate_fbb.GetSize());
  if (!intermediate_model) {
    return absl::InternalError("Failed to load intermediate model");
  }
  std::unique_ptr<Interpreter> interpreter;
  InterpreterBuilder(*intermediate_model, *resolver)(&interpreter);
  if (!interpreter) {
    return absl::InternalError(
        "Failed to build interpreter from intermediate model");
  }
  Subgraph* subgraph = interpreter->subgraph(1);
  if (subgraph->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError(
        "Failed to AllocateTensors() on validation subgraph of intermediate "
        "model");
  }
  if (subgraph->Invoke() != kTfLiteOk) {
    return absl::InternalError(
        "Failed to Invoke() on validation subgraph of intermediate model");
  }
  return builder.BuildFinalModel(fbb, subgraph);
}

}  // namespace acceleration
}  // namespace tflite
