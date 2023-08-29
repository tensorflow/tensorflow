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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/embedder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_common.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/validation_graph_builder.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace fb = flatbuffers;

namespace tflite {
namespace acceleration {
namespace {

constexpr char kMetricPrefix[] = "metrics/";

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
        {jpeg_image_data.data(), jpeg_image_data.size()}, &header);
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
      "count (validation input count: %d, main subgraph output count: %d)",
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
  ValidationGraphBuilder builder(
      kMetricPrefix, main_model_, jpeg_data_, jpeg_output_channels_, scale_,
      zero_point_, validation_model_, schema_, use_ondevice_cpu_for_golden_);
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
