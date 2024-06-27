/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/core/light_model_builder.h"

#include <stddef.h>

#include <cstring>
#include <memory>
#include <utility>

#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/lite/allocation.h"

namespace mlir {

#ifndef TFLITE_MCU
// Loads a model from `filename`. If `mmap_file` is true then use mmap,
// otherwise make a copy of the model in a buffer.
std::unique_ptr<tflite::Allocation> GetAllocationFromFile(
    const char* filename) {
  std::unique_ptr<tflite::Allocation> allocation;
  if (tflite::MMAPAllocation::IsSupported()) {
    allocation = std::make_unique<tflite::MMAPAllocation>(filename, nullptr);
  } else {
    allocation =
        std::make_unique<tflite::FileCopyAllocation>(filename, nullptr);
  }
  return allocation;
}

std::unique_ptr<LightFlatBufferModel> LightFlatBufferModel::BuildFromFile(
    const char* filename) {
  std::unique_ptr<LightFlatBufferModel> model =
      BuildFromAllocation(GetAllocationFromFile(filename));
#if FLATBUFFERS_LITTLEENDIAN == 1
  return model;
#else
  return ByteConvertModel(std::move(model));
#endif
}

#endif

std::unique_ptr<LightFlatBufferModel> LightFlatBufferModel::BuildFromBuffer(
    const char* caller_owned_buffer, size_t buffer_size) {
  std::unique_ptr<tflite::Allocation> allocation(
      new tflite::MemoryAllocation(caller_owned_buffer, buffer_size, nullptr));
  return BuildFromAllocation(std::move(allocation));
}

std::unique_ptr<LightFlatBufferModel>
LightFlatBufferModel::VerifyAndBuildFromBuffer(const char* caller_owned_buffer,
                                               size_t buffer_size) {
  std::unique_ptr<tflite::Allocation> allocation(
      new tflite::MemoryAllocation(caller_owned_buffer, buffer_size, nullptr));
  return VerifyAndBuildFromAllocation(std::move(allocation));
}

#if FLATBUFFERS_LITTLEENDIAN == 0

void LightFlatBufferModel::ByteSwapSerializedModel(
    std::string* serialized_model, bool from_big_endian) {
  const uint8_t* buffer =
      reinterpret_cast<const uint8_t*>(serialized_model->c_str());
  const tflite::Model* input_model = tflite::GetModel(buffer);
  ByteSwapTFLiteModel(input_model, from_big_endian);
}

void LightFlatBufferModel::ByteSwapBuffer(int8_t tensor_type,
                                          size_t buffer_size, uint8_t* buffer,
                                          bool from_big_endian) {
  switch (tensor_type) {
    case tflite::TensorType_STRING: {
      auto bp = reinterpret_cast<int32_t*>(buffer);
      int num_of_strings =
          from_big_endian ? bp[0] : flatbuffers::EndianSwap(bp[0]);
      for (int i = 0; i < num_of_strings + 2; i++)
        bp[i] = flatbuffers::EndianSwap(bp[i]);
      break;
    }
    // 16-bit types
    case tflite::TensorType_FLOAT16:
    case tflite::TensorType_INT16:
    case tflite::TensorType_UINT16: {
      auto bp = reinterpret_cast<uint16_t*>(buffer);
      for (int i = 0; i < buffer_size / 2; i++)
        bp[i] = flatbuffers::EndianSwap(bp[i]);
      break;
    }
    // 32-bit types
    case tflite::TensorType_FLOAT32:
    case tflite::TensorType_INT32:
    case tflite::TensorType_UINT32:
    case tflite::TensorType_COMPLEX64: {
      auto bp = reinterpret_cast<uint32_t*>(buffer);
      for (int i = 0; i < buffer_size / 4; i++)
        bp[i] = flatbuffers::EndianSwap(bp[i]);
      break;
    }
    // 64-bit types
    case tflite::TensorType_INT64:
    case tflite::TensorType_FLOAT64:
    case tflite::TensorType_UINT64:
    case tflite::TensorType_COMPLEX128: {
      auto bp = reinterpret_cast<uint64_t*>(buffer);
      for (int i = 0; i < buffer_size / 8; i++)
        bp[i] = flatbuffers::EndianSwap(bp[i]);
      break;
    }
    default:
      break;
  }
}

void LightFlatBufferModel::ByteSwapTFLiteModel(const tflite::Model* tfl_model,
                                               bool from_big_endian) {
  bool buffer_swapped[tfl_model->buffers()->size()] = {};
  for (size_t subgraph_idx = 0; subgraph_idx < tfl_model->subgraphs()->size();
       subgraph_idx++) {
    const tflite::SubGraph* subgraph =
        tfl_model->subgraphs()->Get(subgraph_idx);
    for (size_t ts_idx = 0; ts_idx < subgraph->tensors()->size(); ts_idx++) {
      const tflite::Tensor* tensor = subgraph->tensors()->Get(ts_idx);
      if (tensor->buffer() > 0 &&
          tensor->buffer() < tfl_model->buffers()->size() &&
          !buffer_swapped[tensor->buffer()]) {
        const tflite::Buffer* buffer_ =
            (*tfl_model->buffers())[tensor->buffer()];
        if (!buffer_ || !buffer_->data()) continue;
        auto* buffer = buffer_->data();
        uint8_t* buff_ = const_cast<uint8_t*>(buffer->data());
        ByteSwapBuffer(tensor->type(), buffer->size(), buff_, from_big_endian);
        buffer_swapped[tensor->buffer()] = true;
      }
    }
  }
}

std::unique_ptr<LightFlatBufferModel> LightFlatBufferModel::ByteConvertModel(
    std::unique_ptr<LightFlatBufferModel> model, bool from_big_endian) {
  if (model == nullptr) return model;
  auto tfl_model = model->GetModel();
  if (tfl_model->subgraphs()->size() == 0) return model;
  if (tfl_model->subgraphs()->Get(0)->tensors()->size() == 0) return model;
  if (tfl_model->buffers()->size() < 2) return model;
  return ByteSwapFlatBufferModel(std::move(model), from_big_endian);
}

std::unique_ptr<LightFlatBufferModel>
LightFlatBufferModel::ByteSwapFlatBufferModel(
    std::unique_ptr<LightFlatBufferModel> model, bool from_big_endian) {
  FlatBufferModel* modelp = model.release();
  auto tflite_model = modelp->GetModel();
  auto copied_model = std::make_unique<tflite::ModelT>();
  tflite_model->UnPackTo(copied_model.get(), nullptr);
  ByteSwapTFLiteModelT(copied_model.get(), from_big_endian);
  std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(
      new flatbuffers::FlatBufferBuilder());
  auto packed_model = tflite::Model::Pack(*builder, copied_model.get());
  tflite::FinishModelBuffer(*builder, packed_model);
  flatbuffers::FlatBufferBuilder* builder_ = builder.release();
  return BuildFromBuffer(
      reinterpret_cast<const char*>(builder_->GetBufferPointer()),
      builder_->GetSize());
}

void LightFlatBufferModel::ByteSwapTFLiteModelT(tflite::ModelT* tfl_modelt,
                                                bool from_big_endian) {
  size_t bytes_per_elem = 0;
  bool buffer_swapped[tfl_modelt->buffers.size()] = {};
  for (size_t subgraph_idx = 0; subgraph_idx < tfl_modelt->subgraphs.size();
       subgraph_idx++) {
    tflite::SubGraphT* subgraph = tfl_modelt->subgraphs.at(subgraph_idx).get();
    for (size_t ts_idx = 0; ts_idx < subgraph->tensors.size(); ts_idx++) {
      tflite::TensorT* tensor = subgraph->tensors[ts_idx].get();
      if (tensor->buffer > 0 && tensor->buffer < tfl_modelt->buffers.size() &&
          !buffer_swapped[tensor->buffer]) {
        const auto* buffer = &(tfl_modelt->buffers[tensor->buffer].get()->data);
        if (buffer && buffer->data()) {
          uint8_t* buff_ = const_cast<uint8_t*>(buffer->data());
          ByteSwapBuffer(tensor->type, buffer->size(), buff_, from_big_endian);
          buffer_swapped[tensor->buffer] = true;
        }
      }
    }
  }
}

#endif

std::unique_ptr<LightFlatBufferModel> LightFlatBufferModel::BuildFromAllocation(
    std::unique_ptr<tflite::Allocation> allocation) {
  std::unique_ptr<LightFlatBufferModel> model(
      new LightFlatBufferModel(std::move(allocation)));
  if (!model->initialized()) {
    model.reset();
  }
  return model;
}

std::unique_ptr<LightFlatBufferModel>
LightFlatBufferModel::VerifyAndBuildFromAllocation(
    std::unique_ptr<tflite::Allocation> allocation) {
  if (!allocation || !allocation->valid()) {
    return nullptr;
  }

  {
    // Flatbuffers can only be smaller than 2GB. The file format appends some
    // data after the actual flabuffer. We truncate the allocation size to 2GB
    // so that the verifier doesn't early exit on us.
    size_t allocation_size =
        std::min(allocation->bytes(),
                 static_cast<size_t>(FLATBUFFERS_MAX_BUFFER_SIZE - 1));
    flatbuffers::Verifier base_verifier(
        reinterpret_cast<const uint8_t*>(allocation->base()), allocation_size);
    if (!tflite::VerifyModelBuffer(base_verifier)) {
      return nullptr;
    }
  }

  return BuildFromAllocation(std::move(allocation));
}

LightFlatBufferModel::LightFlatBufferModel(
    std::unique_ptr<tflite::Allocation> allocation)
    : allocation_(std::move(allocation)) {
  if (!allocation_ || !allocation_->valid()) {
    return;
  }

  model_ = ::tflite::GetModel(allocation_->base());
}

LightFlatBufferModel::~LightFlatBufferModel() = default;

}  // namespace mlir
