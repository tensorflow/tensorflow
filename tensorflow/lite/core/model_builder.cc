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
#include "tensorflow/lite/core/model_builder.h"

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/verifier.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace {

// Ensure that ErrorReporter is non-null.
ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
  return e ? e : DefaultErrorReporter();
}

}  // namespace

#ifndef TFLITE_MCU
// Loads a model from `filename`. If `mmap_file` is true then use mmap,
// otherwise make a copy of the model in a buffer.
std::unique_ptr<Allocation> GetAllocationFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  std::unique_ptr<Allocation> allocation;
  if (MMAPAllocation::IsSupported()) {
    allocation = std::make_unique<MMAPAllocation>(filename, error_reporter);
  } else {
    allocation = std::make_unique<FileCopyAllocation>(filename, error_reporter);
  }
  return allocation;
}

namespace impl {

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<FlatBufferModel> model = BuildFromAllocation(
      GetAllocationFromFile(filename, error_reporter), error_reporter);
#if FLATBUFFERS_LITTLEENDIAN == 1
  return model;
#else
  return ByteConvertModel(std::move(model), error_reporter);
#endif
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromFile(
    const char* filename, TfLiteVerifier* extra_verifier,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<FlatBufferModel> model = VerifyAndBuildFromAllocation(
      GetAllocationFromFile(filename, error_reporter), extra_verifier,
      error_reporter);
#if FLATBUFFERS_LITTLEENDIAN == 1
  return model;
#else
  return ByteConvertModel(std::move(model), error_reporter);
#endif
}

}  // namespace impl

#endif

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromBuffer(
    const char* caller_owned_buffer, size_t buffer_size,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<Allocation> allocation(
      new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
  return BuildFromAllocation(std::move(allocation), error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromBuffer(
    const char* caller_owned_buffer, size_t buffer_size,
    TfLiteVerifier* extra_verifier, ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<Allocation> allocation(
      new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
  return VerifyAndBuildFromAllocation(std::move(allocation), extra_verifier,
                                      error_reporter);
}

#if FLATBUFFERS_LITTLEENDIAN == 0

void FlatBufferModel::ByteSwapSerializedModel(std::string* serialized_model,
                                              bool from_big_endian) {
  const uint8_t* buffer =
      reinterpret_cast<const uint8_t*>(serialized_model->c_str());
  const tflite::Model* input_model = tflite::GetModel(buffer);
  ByteSwapTFLiteModel(input_model, from_big_endian);
}

void FlatBufferModel::ByteSwapBuffer(int8_t tensor_type, size_t buffer_size,
                                     uint8_t* buffer, bool from_big_endian) {
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

void FlatBufferModel::ByteSwapTFLiteModel(const tflite::Model* tfl_model,
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

std::unique_ptr<FlatBufferModel> FlatBufferModel::ByteConvertModel(
    std::unique_ptr<FlatBufferModel> model, ErrorReporter* error_reporter,
    bool from_big_endian) {
  if (model == nullptr) return model;
  auto tfl_model = model->GetModel();
  if (tfl_model->subgraphs()->size() == 0) return model;
  if (tfl_model->subgraphs()->Get(0)->tensors()->size() == 0) return model;
  if (tfl_model->buffers()->size() < 2) return model;
  return ByteSwapFlatBufferModel(std::move(model), error_reporter,
                                 from_big_endian);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::ByteSwapFlatBufferModel(
    std::unique_ptr<FlatBufferModel> model, ErrorReporter* error_reporter,
    bool from_big_endian) {
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
      builder_->GetSize(), error_reporter);
}

void FlatBufferModel::ByteSwapTFLiteModelT(tflite::ModelT* tfl_modelt,
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

void FlatBufferModel::ValidateModelBuffers(ErrorReporter* error_reporter) {
  auto buffers = model_->buffers();
  if (buffers && buffers->size() > 0) {
    auto first_buffer = buffers->Get(0);
    if (first_buffer && first_buffer->size() != 0) {
      // Note the 0th entry of this array must be an empty buffer (sentinel).
      // This is a convention so that tensors without a buffer can provide 0
      // as their buffer.
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "The 0th entry of the model buffer must be an empty buffer.");
    }
  }
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromAllocation(
    std::unique_ptr<Allocation> allocation, ErrorReporter* error_reporter) {
  std::unique_ptr<FlatBufferModel> model(new FlatBufferModel(
      std::move(allocation), ValidateErrorReporter(error_reporter)));
  if (!model->initialized()) {
    model.reset();
  } else {
    model->ValidateModelBuffers(error_reporter);
  }
  return model;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromAllocation(
    std::unique_ptr<Allocation> allocation, TfLiteVerifier* extra_verifier,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  if (!allocation || !allocation->valid()) {
    TF_LITE_REPORT_ERROR(error_reporter, "The model allocation is null/empty");
    return nullptr;
  }

  // Only run validator on models less than 2GB
  if (allocation->bytes() < flatbuffer_size_max) {
    flatbuffers::Verifier base_verifier(
        reinterpret_cast<const uint8_t*>(allocation->base()),
        allocation->bytes());
    if (!VerifyModelBuffer(base_verifier)) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "The model is not a valid Flatbuffer buffer");
      return nullptr;
    }
  }

  if (extra_verifier &&
      !extra_verifier->Verify(static_cast<const char*>(allocation->base()),
                              allocation->bytes(), error_reporter)) {
    // The verifier will have already logged an appropriate error message.
    return nullptr;
  }

  return BuildFromAllocation(std::move(allocation), error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromModel(
    const tflite::Model* caller_owned_model_spec,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);

  if (CheckBufferOutsideModel(caller_owned_model_spec)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "The model contains weights not accessible from "
                         "tflite::Model *, please use other api");
    return nullptr;
  }

  std::unique_ptr<FlatBufferModel> model(
      new FlatBufferModel(caller_owned_model_spec, error_reporter));
  if (!model->initialized()) {
    model.reset();
  } else {
    model->ValidateModelBuffers(error_reporter);
  }
  return model;
}

bool FlatBufferModel::CheckBufferOutsideModel(const tflite::Model* model) {
  if (!model || !model->metadata()) return false;

  for (int i = 0; i < model->metadata()->size(); ++i) {
    auto metadata = model->metadata()->Get(i);
    if (metadata->name()->str() == tflite_metadata_buffer_location) {
      return true;
    }
  }
  return false;
}

string FlatBufferModel::GetMinimumRuntime() const {
  if (!model_ || !model_->metadata()) return "";

  for (int i = 0; i < model_->metadata()->size(); ++i) {
    auto metadata = model_->metadata()->Get(i);
    if (metadata->name()->str() == tflite_metadata_min_runtime_version) {
      auto buf = metadata->buffer();
      auto* buffer = (*model_->buffers())[buf];
      auto* array = buffer->data();
      // Get the real length of the runtime string, since there might be
      // trailing
      // '\0's in the buffer.
      for (int len = 0; len < array->size(); ++len) {
        if (array->data()[len] == '\0') {
          return string(reinterpret_cast<const char*>(array->data()), len);
        }
      }
      // If there is no '\0' in the buffer, this indicates that the flatbuffer
      // is malformed.
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Min_runtime_version in model metadata is malformed");
      break;
    }
  }
  return "";
}

std::map<std::string, std::string> FlatBufferModel::ReadAllMetadata() const {
  return ReadAllMetadata(model_);
}

std::map<std::string, std::string> FlatBufferModel::ReadAllMetadata(
    const tflite::Model* model) {
  std::map<std::string, std::string> keys_values;
  if (!model || !model->metadata() || !model->buffers()) return keys_values;

  for (int i = 0; i < model->metadata()->size(); ++i) {
    auto metadata = model->metadata()->Get(i);
    auto buf = metadata->buffer();
    if (buf >= model->buffers()->size()) continue;
    const tflite::Buffer* buffer = (*model->buffers())[buf];
    if (!buffer || !buffer->data()) continue;
    const flatbuffers::Vector<uint8_t>* array = buffer->data();
    if (!array) continue;
    std::string val =
        string(reinterpret_cast<const char*>(array->data()), array->size());
    // Skip if key or value of metadata is empty.
    if (!metadata->name() || val.empty()) continue;
    keys_values[metadata->name()->str()] = val;
  }
  return keys_values;
}

bool FlatBufferModel::CheckModelIdentifier() const {
  if (allocation_->bytes() < 7) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Model provided must have at least 7 bytes to hold identifier.\n");
    return false;
  }
  if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
    const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
        ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
    return false;
  }
  return true;
}

FlatBufferModel::FlatBufferModel(const Model* model,
                                 ErrorReporter* error_reporter)
    : model_(model), error_reporter_(ValidateErrorReporter(error_reporter)) {}

FlatBufferModel::FlatBufferModel(std::unique_ptr<Allocation> allocation,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)),
      allocation_(std::move(allocation)) {
  if (!allocation_ || !allocation_->valid() || !CheckModelIdentifier()) {
    return;
  }

  model_ = ::tflite::GetModel(allocation_->base());
}

FlatBufferModel::~FlatBufferModel() {}

}  // namespace tflite
