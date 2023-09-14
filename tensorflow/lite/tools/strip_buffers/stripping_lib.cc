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

#include "tensorflow/lite/tools/strip_buffers/stripping_lib.h"

#include <stdint.h>

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/model.h"

#define TFLITE_SCHEMA_VERSION 3

namespace tflite {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::Offset;

// Parameters for a simple Gaussian distribution to generate values roughly in
// [0, 1).
constexpr float kGaussianFloatMean = 0.5;
constexpr float kGaussianStdDev = 1.0 / 3;

template <typename Type, typename TypeT>
void CopyToOffsetVector(FlatBufferBuilder* builder, const Type* data,
                        std::vector<Offset<Type>>& vec) {
  std::unique_ptr<TypeT> unpacked(data->UnPack());
  flatbuffers::Offset<Type> offset = Type::Pack(*builder, unpacked.get());
  vec.push_back(offset);
}

int GetNumElements(const std::vector<int>& dims) {
  int num_elements = 1;
  for (int i = 0; i < dims.size(); i++) {
    num_elements *= dims[i];
  }
  return num_elements;
}

// TODO(b/141023954): Reconcile this with the function in
// inference_profiler_stage.
template <typename T>
void GenerateRandomGaussianData(int64_t num_elements, float min, float max,
                                std::vector<T>* data) {
  data->clear();
  data->reserve(num_elements);

  static std::normal_distribution<double> distribution(kGaussianFloatMean,
                                                       kGaussianStdDev);
  static std::default_random_engine generator;
  for (int i = 0; i < num_elements; ++i) {
    auto rand_n = distribution(generator);
    while (rand_n < 0 || rand_n >= 1) {
      rand_n = distribution(generator);
    }
    auto rand_float = min + (max - min) * static_cast<float>(rand_n);
    data->push_back(static_cast<T>(rand_float));
  }
}

TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type) {
  *type = kTfLiteNoType;
  switch (tensor_type) {
    case TensorType_FLOAT32:
      *type = kTfLiteFloat32;
      break;
    case TensorType_INT32:
      *type = kTfLiteInt32;
      break;
    case TensorType_UINT32:
      *type = kTfLiteUInt32;
      break;
    case TensorType_UINT8:
      *type = kTfLiteUInt8;
      break;
    case TensorType_INT8:
      *type = kTfLiteInt8;
      break;
    default:
      break;
  }
  if (*type == kTfLiteNoType) {
    VLOG(0) << "Unsupported data type %d in tensor: " << tensor_type;
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus StripWeightsFromFlatbuffer(
    const Model* input_model,
    flatbuffers::FlatBufferBuilder* new_model_builder) {
  // TODO(b/141023954): Generalize to N subgraphs.
  if (input_model->subgraphs()->size() != 1) {
    VLOG(0) << "Only 1 subgraph supported for now: "
            << input_model->subgraphs()->size();
    return kTfLiteError;
  }

  // Data structures for output flatbuffer.
  std::vector<Offset<SubGraph>> output_subgraphs;
  std::vector<Offset<OperatorCode>> output_opcodes;
  std::vector<Offset<Buffer>> output_buffers;

  const SubGraph* input_subgraph = (*input_model->subgraphs())[0];
  std::unique_ptr<SubGraphT> mutable_subgraph(input_subgraph->UnPack());

  // For constant tensors that meet requirements:
  // 1. Set the buffer-id to something larger than total number of buffers.
  // This indicates to reconstitute_weights_into_fb that random data should be
  // generated for them.
  // 2. Mark that buffer for not getting carried into the output flatbuffer.
  absl::flat_hash_set<uint32_t> erased_buffers;
  const int num_buffers = input_model->buffers()->size();
  int i = 0;
  for (auto& tensor : mutable_subgraph->tensors) {
    // We don't support Int32 tensors because they could contain
    // non-randomisable information like Reshape dims.
    if (tensor->type == TensorType_INT32 &&
        GetNumElements(tensor->shape) < 10) {
      // Int32 tensors of elements < 10 could be non-randomisable: for example,
      // 'shape' input to a Reshape op.
      continue;
    } else if (tensor->type != TensorType_INT32 &&
               tensor->type != TensorType_FLOAT32 &&
               tensor->type != TensorType_UINT8 &&
               tensor->type != TensorType_INT8) {
      continue;
    }

    if (auto* buffer = (*input_model->buffers())[tensor->buffer]) {
      if (auto* array = buffer->data()) {
        VLOG(1) << "Tensor " << i
                << " is constant, with buffer = " << tensor->buffer;
        // Set tensor buffer to a high value (num_buffers * 2) & put an empty
        // buffer in place of the original one.
        erased_buffers.insert(tensor->buffer);
        tensor->buffer = num_buffers * 2;
      }
    }
    ++i;
  }

  i = 0;
  for (const Buffer* buffer : *(input_model->buffers())) {
    if (erased_buffers.find(i) == erased_buffers.end()) {
      // If buffer is not to be erased, insert it into the output flatbuffer to
      // retain data.
      CopyToOffsetVector<Buffer, BufferT>(new_model_builder, buffer,
                                          output_buffers);
    } else {
      output_buffers.push_back(CreateBuffer(*new_model_builder));
    }
    ++i;
  }

  flatbuffers::Offset<SubGraph> output_subgraph =
      SubGraph::Pack(*new_model_builder, mutable_subgraph.get());
  output_subgraphs.push_back(output_subgraph);

  // Write all ops as they are.
  for (const OperatorCode* opcode : *(input_model->operator_codes())) {
    CopyToOffsetVector<OperatorCode, OperatorCodeT>(new_model_builder, opcode,
                                                    output_opcodes);
  }

  // Generate output model.
  auto description =
      new_model_builder->CreateString("Generated by strip_buffers_from_fb");
  auto new_model_offset =
      CreateModel(*new_model_builder, TFLITE_SCHEMA_VERSION,
                  new_model_builder->CreateVector(output_opcodes),
                  new_model_builder->CreateVector(output_subgraphs),
                  description, new_model_builder->CreateVector(output_buffers),
                  /* metadata_buffer */ 0, /* metadatas */ 0);
  FinishModelBuffer(*new_model_builder, new_model_offset);

  return kTfLiteOk;
}

string StripWeightsFromFlatbuffer(const absl::string_view input_flatbuffer) {
  auto input_model = FlatBufferModel::BuildFromBuffer(input_flatbuffer.data(),
                                                      input_flatbuffer.size());

  FlatBufferBuilder builder(/*initial_size=*/10240);
  if (StripWeightsFromFlatbuffer(input_model->GetModel(), &builder) !=
      kTfLiteOk) {
    return string();
  }

  return string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                builder.GetSize());
}

bool FlatbufferHasStrippedWeights(const Model* input_model) {
  if (input_model->subgraphs()->size() != 1) {
    VLOG(0) << "Only 1 subgraph supported for now";
    return false;
  }
  const SubGraph* input_subgraph = (*input_model->subgraphs())[0];
  std::unique_ptr<SubGraphT> mutable_subgraph(input_subgraph->UnPack());

  // For all tensors that have buffer > num_buffers + 1 (set to be so in
  // strip_buffers_from_fb), create a buffer with random data & assign to them.
  // For others, just copy over the original buffer from source model.
  const int num_buffers = input_model->buffers()->size();
  for (auto& tensor : mutable_subgraph->tensors) {
    if (tensor->buffer > num_buffers + 1) {
      return true;
    }
  }
  return false;
}

TfLiteStatus ReconstituteConstantTensorsIntoFlatbuffer(
    const Model* input_model,
    flatbuffers::FlatBufferBuilder* new_model_builder) {
  // TODO(b/141023954): Generalize to N subgraphs.
  if (input_model->subgraphs()->size() != 1) {
    VLOG(0) << "Only 1 subgraph supported for now";
    return kTfLiteError;
  }
  const SubGraph* input_subgraph = (*input_model->subgraphs())[0];
  std::unique_ptr<SubGraphT> mutable_subgraph(input_subgraph->UnPack());

  // Containers for output flatbuffer.
  std::vector<Offset<::tflite::SubGraph>> output_subgraphs;
  std::vector<Offset<::tflite::OperatorCode>> output_opcodes;
  std::vector<Offset<::tflite::Buffer>> output_buffers;

  // An empty buffer, needed as a TFLite convention.
  output_buffers.push_back(CreateBuffer(*new_model_builder));

  // For all tensors that have buffer > num_buffers + 1 (set to be so in
  // strip_buffers_from_fb), create a buffer with random data & assign to them.
  // For others, just copy over the original buffer from source model.
  const int num_buffers = input_model->buffers()->size();
  for (auto& tensor : mutable_subgraph->tensors) {
    int buffer_id = output_buffers.size();
    if (tensor->buffer > num_buffers + 1) {
      // Num tensor elements.
      int num_elements = GetNumElements(tensor->shape);
      // Tensor type.
      TfLiteType type;
      if (ConvertTensorType(tensor->type, &type) != kTfLiteOk) {
        return kTfLiteError;
      }
      // Generate buffer random data.
      // Use different min/max bounds per tensor-type to ensure that random data
      // 'appears' similar to typical values.
      if (type == kTfLiteUInt8) {
        std::vector<uint8_t> data;
        GenerateRandomGaussianData(num_elements,
                                   std::numeric_limits<uint8_t>::min(),
                                   std::numeric_limits<uint8_t>::max(), &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(uint8_t) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      } else if (type == kTfLiteInt8) {
        std::vector<int8_t> data;
        GenerateRandomGaussianData(num_elements,
                                   std::numeric_limits<int8_t>::min(),
                                   std::numeric_limits<int8_t>::max(), &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(int8_t) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      } else if (type == kTfLiteFloat32) {
        std::vector<float> data;
        GenerateRandomGaussianData(num_elements, -1, 1, &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(float) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      } else if (type == kTfLiteInt32) {
        std::vector<int32_t> data;
        GenerateRandomGaussianData(num_elements, 10, 10, &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(int32_t) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      }
    } else {
      // For intermediate tensors, create a new buffer & assign the id to them.
      // output_buffers.push_back(CreateBuffer(*new_model_builder));
      CopyToOffsetVector<Buffer, BufferT>(
          new_model_builder, input_model->buffers()->Get(tensor->buffer),
          output_buffers);
    }
    tensor->buffer = buffer_id;
  }

  for (const ::tflite::OperatorCode* opcode :
       *(input_model->operator_codes())) {
    CopyToOffsetVector<::tflite::OperatorCode, ::tflite::OperatorCodeT>(
        new_model_builder, opcode, output_opcodes);
  }

  flatbuffers::Offset<::tflite::SubGraph> output_subgraph =
      ::tflite::SubGraph::Pack(*new_model_builder, mutable_subgraph.get());
  output_subgraphs.push_back(output_subgraph);

  auto description = new_model_builder->CreateString(
      "Generated by TFLite strip_buffers/reconstitution");
  auto new_model_offset =
      CreateModel(*new_model_builder, TFLITE_SCHEMA_VERSION,
                  new_model_builder->CreateVector(output_opcodes),
                  new_model_builder->CreateVector(output_subgraphs),
                  description, new_model_builder->CreateVector(output_buffers),
                  /* metadata_buffer */ 0, /* metadatas */ 0);
  FinishModelBuffer(*new_model_builder, new_model_offset);

  return kTfLiteOk;
}

string ReconstituteConstantTensorsIntoFlatbuffer(
    const absl::string_view input_flatbuffer) {
  auto input_model = FlatBufferModel::BuildFromBuffer(input_flatbuffer.data(),
                                                      input_flatbuffer.size());

  FlatBufferBuilder builder(/*initial_size=*/10240);
  if (ReconstituteConstantTensorsIntoFlatbuffer(input_model->GetModel(),
                                                &builder) != kTfLiteOk) {
    return string();
  }

  return string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                builder.GetSize());
}

}  // namespace tflite
