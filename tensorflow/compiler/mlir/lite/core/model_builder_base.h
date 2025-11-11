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
/// \file
///
/// Deserialization infrastructure for tflite. Provides functionality
/// to go from a serialized tflite model in flatbuffer format to an
/// in-memory representation of the model.
///
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include "third_party/tensorflow/lite/model_builder.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
// IWYU pragma: private, include "third_party/tensorflow/lite/model_builder.h"

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_MODEL_BUILDER_BASE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_MODEL_BUILDER_BASE_H_

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "tensorflow/compiler/mlir/lite/core/api/verifier.h"
#include "tensorflow/compiler/mlir/lite/core/macros.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace tflite {

std::unique_ptr<Allocation> GetAllocationFromFile(
    const char* filename, ErrorReporter* error_reporter,
    bool allow_modifications = false);

std::unique_ptr<Allocation> GetAllocationFromFile(
    int fd, ErrorReporter* error_reporter, bool allow_modifications = false);

namespace impl {

/// An RAII object that represents a read-only tflite model, copied from disk,
/// or mmapped. This uses flatbuffers as the serialization format.
///
/// NOTE: The current API requires that a FlatBufferModelBase instance be kept
/// alive by the client as long as it is in use by any dependent Interpreter
/// instances. As the FlatBufferModelBase instance is effectively immutable
/// after creation, the client may safely use a single model with multiple
/// dependent Interpreter instances, even across multiple threads (though note
/// that each Interpreter instance is *not* thread-safe).
///
/// <pre><code>
/// using namespace tflite;
/// StderrReporter error_reporter;
/// auto model = FlatBufferModelBase::BuildFromFile("interesting_model.tflite",
///                                             &error_reporter);
/// MyOpResolver resolver;  // You need to subclass OpResolver to provide
///                         // implementations.
/// InterpreterBuilder builder(*model, resolver);
/// std::unique_ptr<Interpreter> interpreter;
/// if(builder(&interpreter) == kTfLiteOk) {
///   .. run model inference with interpreter
/// }
/// </code></pre>
///
/// OpResolver must be defined to provide your kernel implementations to the
/// interpreter. This is environment specific and may consist of just the
/// builtin ops, or some custom operators you defined to extend tflite.
template <typename T>
class FlatBufferModelBase {
 public:
  /// Builds a model based on a file.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModelBase instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<T> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<T> model = BuildFromAllocation(
        GetAllocationFromFile(filename, error_reporter), error_reporter);
#if FLATBUFFERS_LITTLEENDIAN == 1
    return model;
#else
    return ByteConvertModel(std::move(model), error_reporter);
#endif
  }

  /// Verifies whether the content of the file is legit, then builds a model
  /// based on the file.
  /// The extra_verifier argument is an additional optional verifier for the
  /// file contents. By default, we always check with tflite::VerifyModelBuffer.
  /// If extra_verifier is supplied, the file contents is also checked against
  /// the extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModelBase instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<T> VerifyAndBuildFromFile(
      const char* filename, TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<T> model = VerifyAndBuildFromAllocation(
        GetAllocationFromFile(filename, error_reporter), extra_verifier,
        error_reporter);
#if FLATBUFFERS_LITTLEENDIAN == 1
    return model;
#else
    return ByteConvertModel(std::move(model), error_reporter);
#endif
  }

  /// Builds a model based on a file descriptor.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModelBase instance. Caller retains ownership
  /// of `fd` and must ensure it is closed after BuildFromFile returns. Returns
  /// a nullptr in case of failure.
  static std::unique_ptr<T> BuildFromFileDescriptor(
      int fd, ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<T> model = BuildFromAllocation(
        GetAllocationFromFile(fd, error_reporter), error_reporter);
#if FLATBUFFERS_LITTLEENDIAN == 1
    return model;
#else
    return ByteConvertModel(std::move(model), error_reporter);
#endif
  }

  /// Verifies whether the content of the file descriptor is legit, then builds
  /// a model based on the file.
  /// The extra_verifier argument is an additional optional verifier for the
  /// file contents. By default, we always check with tflite::VerifyModelBuffer.
  /// If extra_verifier is supplied, the file contents is also checked against
  /// the extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModelBase instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<T> VerifyAndBuildFromFileDescriptor(
      int fd, TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<FlatBufferModelBase<T>> model =
        VerifyAndBuildFromAllocation(GetAllocationFromFile(fd, error_reporter),
                                     extra_verifier, error_reporter);
#if FLATBUFFERS_LITTLEENDIAN == 1
    return model;
#else
    return ByteConvertModel(std::move(model), error_reporter);
#endif
  }

  /// Builds a model based on a pre-loaded flatbuffer.
  /// Caller retains ownership of the buffer and should keep it alive until
  /// the returned object is destroyed. Caller also retains ownership of
  /// `error_reporter` and must ensure its lifetime is longer than the
  /// FlatBufferModelBase instance.
  /// Returns a nullptr in case of failure.
  /// NOTE: this does NOT validate the buffer so it should NOT be called on
  /// invalid/untrusted input. Use VerifyAndBuildFromBuffer in that case
  static std::unique_ptr<T> BuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<Allocation> allocation(
        new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
    return BuildFromAllocation(std::move(allocation), error_reporter);
  }

  /// Verifies whether the content of the buffer is legit, then builds a model
  /// based on the pre-loaded flatbuffer.
  /// The extra_verifier argument is an additional optional verifier for the
  /// buffer. By default, we always check with tflite::VerifyModelBuffer. If
  /// extra_verifier is supplied, the buffer is checked against the
  /// extra_verifier after the check against tflite::VerifyModelBuilder. The
  /// caller retains ownership of the buffer and should keep it alive until the
  /// returned object is destroyed. Caller retains ownership of `error_reporter`
  /// and must ensure its lifetime is longer than the FlatBufferModelBase
  /// instance. Returns a nullptr in case of failure.
  static std::unique_ptr<T> VerifyAndBuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size,
      TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);
    std::unique_ptr<Allocation> allocation(
        new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
    return VerifyAndBuildFromAllocation(std::move(allocation), extra_verifier,
                                        error_reporter);
  }

#if FLATBUFFERS_LITTLEENDIAN == 0

  void ByteSwapSerializedModel(std::string* serialized_model,
                               bool from_big_endian) {
    const uint8_t* buffer =
        reinterpret_cast<const uint8_t*>(serialized_model->c_str());
    const tflite::Model* input_model = tflite::GetModel(buffer);
    ByteSwapTFLiteModel(input_model, from_big_endian);
  }

  void ByteSwapBuffer(int8_t tensor_type, size_t buffer_size, uint8_t* buffer,
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

  void ByteSwapTFLiteModel(const tflite::Model* tfl_model,
                           bool from_big_endian) {
    std::vector<bool> buffer_swapped(tfl_model->buffers()->size(), false);
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
          ByteSwapBuffer(tensor->type(), buffer->size(), buff_,
                         from_big_endian);
          buffer_swapped[tensor->buffer()] = true;
        }
      }
    }
  }

  std::unique_ptr<T> ByteConvertModel(std::unique_ptr<T> model,
                                      ErrorReporter* error_reporter,
                                      bool from_big_endian) {
    if (model == nullptr) return model;
    auto tfl_model = model->GetModel();
    if (tfl_model->subgraphs()->size() == 0) return model;
    if (tfl_model->subgraphs()->Get(0)->tensors()->size() == 0) return model;
    if (tfl_model->buffers()->size() < 2) return model;
    return ByteSwapFlatBufferModelBase<T>(std::move(model), error_reporter,
                                          from_big_endian);
  }

  std::unique_ptr<T> ByteSwapFlatBufferModelBase(std::unique_ptr<T> model,
                                                 ErrorReporter* error_reporter,
                                                 bool from_big_endian) {
    FlatBufferModelBase<T>* modelp = model.release();
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

  void ByteSwapTFLiteModelT(tflite::ModelT* tfl_modelt, bool from_big_endian) {
    size_t bytes_per_elem = 0;
    std::vector<bool> buffer_swapped(tfl_modelt->buffers.size(), false);
    for (size_t subgraph_idx = 0; subgraph_idx < tfl_modelt->subgraphs.size();
         subgraph_idx++) {
      tflite::SubGraphT* subgraph =
          tfl_modelt->subgraphs.at(subgraph_idx).get();
      for (size_t ts_idx = 0; ts_idx < subgraph->tensors.size(); ts_idx++) {
        tflite::TensorT* tensor = subgraph->tensors[ts_idx].get();
        if (tensor->buffer > 0 && tensor->buffer < tfl_modelt->buffers.size() &&
            !buffer_swapped[tensor->buffer]) {
          const auto* buffer =
              &(tfl_modelt->buffers[tensor->buffer].get()->data);
          if (buffer && buffer->data()) {
            uint8_t* buff_ = const_cast<uint8_t*>(buffer->data());
            ByteSwapBuffer(tensor->type, buffer->size(), buff_,
                           from_big_endian);
            buffer_swapped[tensor->buffer] = true;
          }
        }
      }
    }
  }

#endif

  /// Builds a model directly from an allocation.
  /// Ownership of the allocation is passed to the model, but the caller
  /// retains ownership of `error_reporter` and must ensure its lifetime is
  /// longer than the FlatBufferModelBase instance.
  /// Returns a nullptr in case of failure (e.g., the allocation is invalid).
  static std::unique_ptr<T> BuildFromAllocation(
      std::unique_ptr<Allocation> allocation,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    std::unique_ptr<T> model(
        new T(std::move(allocation), ValidateErrorReporter(error_reporter)));
    if (!model->initialized()) {
      model.reset();
    } else {
      model->ValidateModelBuffers(error_reporter);
    }
    return model;
  }

  /// Verifies whether the content of the allocation is legit, then builds a
  /// model based on the provided allocation.
  /// The extra_verifier argument is an additional optional verifier for the
  /// buffer. By default, we always check with tflite::VerifyModelBuffer. If
  /// extra_verifier is supplied, the buffer is checked against the
  /// extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Ownership of the allocation is passed to the model, but the caller
  /// retains ownership of `error_reporter` and must ensure its lifetime is
  /// longer than the FlatBufferModelBase instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<T> VerifyAndBuildFromAllocation(
      std::unique_ptr<Allocation> allocation,
      TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);
    if (!allocation || !allocation->valid()) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "The model allocation is null/empty");
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
          reinterpret_cast<const uint8_t*>(allocation->base()), allocation_size,
          flatbuffers::Verifier::Options());
      if (!VerifyModelBuffer(base_verifier)) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "The model is not a valid Flatbuffer buffer");
        return nullptr;
      }

      if (extra_verifier &&
          !extra_verifier->Verify(static_cast<const char*>(allocation->base()),
                                  allocation_size, error_reporter)) {
        // The verifier will have already logged an appropriate error message.
        return nullptr;
      }
    }

    return BuildFromAllocation(std::move(allocation), error_reporter);
  }

  /// Builds a model directly from a flatbuffer pointer
  /// Caller retains ownership of the buffer and should keep it alive until the
  /// returned object is destroyed. Caller retains ownership of `error_reporter`
  /// and must ensure its lifetime is longer than the FlatBufferModelBase
  /// instance. Returns a nullptr in case of failure.
  static std::unique_ptr<T> BuildFromModel(
      const tflite::Model* caller_owned_model_spec,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter()) {
    error_reporter = ValidateErrorReporter(error_reporter);

    if (CheckBufferOutsideModel(caller_owned_model_spec)) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "The model contains weights not accessible from "
                           "tflite::Model *, please use other api");
      return nullptr;
    }

    std::unique_ptr<T> model(new T(caller_owned_model_spec, error_reporter));
    if (!model->initialized()) {
      model.reset();
    } else {
      model->ValidateModelBuffers(error_reporter);
    }
    return model;
  }

  // Releases memory or unmaps mmaped memory.
  ~FlatBufferModelBase() = default;

  // Copying or assignment is disallowed to simplify ownership semantics.
  FlatBufferModelBase(const FlatBufferModelBase&) = delete;
  FlatBufferModelBase& operator=(const FlatBufferModelBase&) = delete;

  bool initialized() const { return model_ != nullptr; }
  const tflite::Model* operator->() const { return model_; }
  const tflite::Model* GetModel() const { return model_; }
  ErrorReporter* error_reporter() const { return error_reporter_; }
  const Allocation* allocation() const { return allocation_.get(); }

  // Returns the minimum runtime version from the flatbuffer. This runtime
  // version encodes the minimum required interpreter version to run the
  // flatbuffer model. If the minimum version can't be determined, an empty
  // string will be returned.
  // Note that the returned minimum version is a lower-bound but not a strict
  // lower-bound; ops in the graph may not have an associated runtime version,
  // in which case the actual required runtime might be greater than the
  // reported minimum.
  std::string GetMinimumRuntime() const {
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
            return std::string(reinterpret_cast<const char*>(array->data()),
                               len);
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

  // Return model metadata as a mapping of name & buffer strings.
  // See Metadata table in TFLite schema.
  std::map<std::string, std::string> ReadAllMetadata() const {
    return ReadAllMetadata(model_);
  }

  // // Return model metadata as a mapping of name & buffer strings.
  // // See Metadata table in TFLite schema.
  static std::map<std::string, std::string> ReadAllMetadata(
      const ::tflite::Model* model) {
    std::map<std::string, std::string> keys_values;
    if (!model || !model->metadata() || !model->buffers()) return keys_values;

    for (size_t i = 0; i < model->metadata()->size(); ++i) {
      auto metadata = model->metadata()->Get(i);
      auto buf = metadata->buffer();
      if (buf >= model->buffers()->size()) continue;
      const tflite::Buffer* buffer = (*model->buffers())[buf];
      if (!buffer || !buffer->data()) continue;
      const flatbuffers::Vector<uint8_t>* array = buffer->data();
      if (!array) continue;
      std::string val = std::string(
          reinterpret_cast<const char*>(array->data()), array->size());
      // Skip if key or value of metadata is empty.
      if (!metadata->name() || val.empty()) continue;
      keys_values[metadata->name()->str()] = val;
    }
    return keys_values;
  }

  // Validates if the FlatBufferModelBase's buffer is well-formed. Specifically,
  // it checks if the 0th entry of the model buffers is an empty buffer
  // (sentinel). This is a convention so that tensors without a buffer can
  // provide 0 as their buffer. NOTE: The function doesn't explicitly fail for
  // backward compatibility reasons; it just provides a warning in case of
  // failures.
  void ValidateModelBuffers(ErrorReporter* error_reporter) {
    auto buffers = model_->buffers();
    if (buffers && !buffers->empty()) {
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

  /// Returns true if the model identifier is correct (otherwise false and
  /// reports an error).
  bool CheckModelIdentifier() const {
    if (allocation_->bytes() < 7) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Model provided must have at least 7 bytes to hold identifier.\n");
      return false;
    }
    if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
      const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
      // Suppress unused variable warning.
      (void)ident;
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
          ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
      return false;
    }
    return true;
  }

  /// Check If the buffer is stored as part of the Flatbuffer or outside
  /// Return false if the buffers are part of the Flatbuffer
  static bool CheckBufferOutsideModel(const tflite::Model* model) {
    if (!model || !model->metadata()) return false;

    for (int i = 0; i < model->metadata()->size(); ++i) {
      auto metadata = model->metadata()->Get(i);
      if (metadata->name()->str() == tflite_metadata_buffer_location) {
        return true;
      }
    }
    return false;
  }

 protected:
  /// Loads a model from a given allocation. FlatBufferModelBase will take over
  /// the ownership of `allocation`, and delete it in destructor. The ownership
  /// of `error_reporter`remains with the caller and must have lifetime at least
  /// as much as FlatBufferModelBase. This is to allow multiple models to use
  /// the same ErrorReporter instance.
  explicit FlatBufferModelBase(
      std::unique_ptr<Allocation> allocation,
      ErrorReporter* error_reporter = T::GetDefaultErrorReporter())
      : error_reporter_(ValidateErrorReporter(error_reporter)),
        allocation_(std::move(allocation)) {
    if (!allocation_ || !allocation_->valid() || !CheckModelIdentifier()) {
      return;
    }

    model_ = ::tflite::GetModel(allocation_->base());
  }

  /// Loads a model from Model flatbuffer. The `model` has to remain alive and
  /// unchanged until the end of this flatbuffer model's lifetime.
  FlatBufferModelBase(const Model* model, ErrorReporter* error_reporter)
      : model_(model), error_reporter_(ValidateErrorReporter(error_reporter)) {}

  static ErrorReporter* ValidateErrorReporter(ErrorReporter* error_reporter) {
    return error_reporter ? error_reporter : T::GetDefaultErrorReporter();
  }

  /// Flatbuffer traverser pointer. (Model* is a pointer that is within the
  /// allocated memory of the data allocated by allocation's internals.
  const tflite::Model* model_ = nullptr;
  /// The error reporter to use for model errors and subsequent errors when
  /// the interpreter is created
  ErrorReporter* error_reporter_;
  /// The allocator used for holding memory of the model. Note that this will
  /// be null if the client provides a tflite::Model directly.
  std::unique_ptr<Allocation> allocation_;
};

}  // namespace impl

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_MODEL_BUILDER_BASE_H_
