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
#ifndef TENSORFLOW_LITE_CORE_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_CORE_MODEL_BUILDER_H_

#include <stddef.h>

#include <map>
#include <memory>
#include <string>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/verifier.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

/// An RAII object that represents a read-only tflite model, copied from disk,
/// or mmapped. This uses flatbuffers as the serialization format.
///
/// NOTE: The current API requires that a FlatBufferModel instance be kept alive
/// by the client as long as it is in use by any dependent Interpreter
/// instances. As the FlatBufferModel instance is effectively immutable after
/// creation, the client may safely use a single model with multiple dependent
/// Interpreter instances, even across multiple threads (though note that each
/// Interpreter instance is *not* thread-safe).
///
/// <pre><code>
/// using namespace tflite;
/// StderrReporter error_reporter;
/// auto model = FlatBufferModel::BuildFromFile("interesting_model.tflite",
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
namespace impl {

class FlatBufferModel {
 public:
  /// Builds a model based on a file.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Verifies whether the content of the file is legit, then builds a model
  /// based on the file.
  /// The extra_verifier argument is an additional optional verifier for the
  /// file contents. By default, we always check with tflite::VerifyModelBuffer.
  /// If extra_verifier is supplied, the file contents is also checked against
  /// the extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(
      const char* filename, TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Builds a model based on a file descriptor.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModel instance. Caller retains ownership of
  /// `fd` and must ensure it is closed after BuildFromFile returns.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFileDescriptor(
      int fd,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Verifies whether the content of the file descriptor is legit, then builds
  /// a model based on the file.
  /// The extra_verifier argument is an additional optional verifier for the
  /// file contents. By default, we always check with tflite::VerifyModelBuffer.
  /// If extra_verifier is supplied, the file contents is also checked against
  /// the extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Caller retains ownership of `error_reporter` and must ensure its lifetime
  /// is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFileDescriptor(
      int fd, TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Builds a model based on a pre-loaded flatbuffer.
  /// Caller retains ownership of the buffer and should keep it alive until
  /// the returned object is destroyed. Caller also retains ownership of
  /// `error_reporter` and must ensure its lifetime is longer than the
  /// FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  /// NOTE: this does NOT validate the buffer so it should NOT be called on
  /// invalid/untrusted input. Use VerifyAndBuildFromBuffer in that case
  static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Verifies whether the content of the buffer is legit, then builds a model
  /// based on the pre-loaded flatbuffer.
  /// The extra_verifier argument is an additional optional verifier for the
  /// buffer. By default, we always check with tflite::VerifyModelBuffer. If
  /// extra_verifier is supplied, the buffer is checked against the
  /// extra_verifier after the check against tflite::VerifyModelBuilder. The
  /// caller retains ownership of the buffer and should keep it alive until the
  /// returned object is destroyed. Caller retains ownership of `error_reporter`
  /// and must ensure its lifetime is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size,
      TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Builds a model directly from an allocation.
  /// Ownership of the allocation is passed to the model, but the caller
  /// retains ownership of `error_reporter` and must ensure its lifetime is
  /// longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure (e.g., the allocation is invalid).
  static std::unique_ptr<FlatBufferModel> BuildFromAllocation(
      std::unique_ptr<Allocation> allocation,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Verifies whether the content of the allocation is legit, then builds a
  /// model based on the provided allocation.
  /// The extra_verifier argument is an additional optional verifier for the
  /// buffer. By default, we always check with tflite::VerifyModelBuffer. If
  /// extra_verifier is supplied, the buffer is checked against the
  /// extra_verifier after the check against tflite::VerifyModelBuilder.
  /// Ownership of the allocation is passed to the model, but the caller
  /// retains ownership of `error_reporter` and must ensure its lifetime is
  /// longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromAllocation(
      std::unique_ptr<Allocation> allocation,
      TfLiteVerifier* extra_verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Builds a model directly from a flatbuffer pointer
  /// Caller retains ownership of the buffer and should keep it alive until the
  /// returned object is destroyed. Caller retains ownership of `error_reporter`
  /// and must ensure its lifetime is longer than the FlatBufferModel instance.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromModel(
      const tflite::Model* caller_owned_model_spec,
      ErrorReporter* error_reporter = DefaultErrorReporter());

#if FLATBUFFERS_LITTLEENDIAN == 0
  /// Byte swap a constant buffer in place.
  static void ByteSwapBuffer(int8_t tensor_type, size_t buffer_size,
                             uint8_t* buffer, bool from_big_endian = true);

  /// Byte swap the buffers field of a TFLite Model instance in place.
  static void ByteSwapTFLiteModel(const tflite::Model* tfl_model,
                                  bool from_big_endian = true);

  /// Byte swap the buffers field of a TFLite ModelT instance in place.
  static void ByteSwapTFLiteModelT(tflite::ModelT* tfl_modelt,
                                   bool from_big_endian = true);

  /// Convert the TFLite buffers field between LE and BE format in a
  /// FlatBufferModel which is not empty and return the converted instance.
  static std::unique_ptr<FlatBufferModel> ByteConvertModel(
      std::unique_ptr<FlatBufferModel> model,
      ErrorReporter* error_reporter = DefaultErrorReporter(),
      bool from_big_endian = false);

  /// Byte Swap the TFLite buffers field in a FlatBufferModel and return the
  /// swapped instance.
  static std::unique_ptr<FlatBufferModel> ByteSwapFlatBufferModel(
      std::unique_ptr<FlatBufferModel> model,
      ErrorReporter* error_reporter = DefaultErrorReporter(),
      bool from_big_endian = false);

  /// Byte Swap the serialized String of a TFLite model in place.
  static void ByteSwapSerializedModel(std::string* serialized_model,
                                      bool from_big_endian = true);
#endif

  // Releases memory or unmaps mmaped memory.
  ~FlatBufferModel();

  // Copying or assignment is disallowed to simplify ownership semantics.
  FlatBufferModel(const FlatBufferModel&) = delete;
  FlatBufferModel& operator=(const FlatBufferModel&) = delete;

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
  std::string GetMinimumRuntime() const;

  // Return model metadata as a mapping of name & buffer strings.
  // See Metadata table in TFLite schema.
  std::map<std::string, std::string> ReadAllMetadata() const;

  // Return model metadata as a mapping of name & buffer strings.
  // See Metadata table in TFLite schema.
  static std::map<std::string, std::string> ReadAllMetadata(
      const ::tflite::Model* model);

  // If the buffer is stored as part of the Flatbuffer or outside
  // return false if the buffers are part of the Flatbuffer
  static bool CheckBufferOutsideModel(const tflite::Model* model);

  // Validates if the FlatBufferModel's buffer is well-formed. Specifically, it
  // checks if the 0th entry of the model buffers is an empty buffer (sentinel).
  // This is a convention so that tensors without a buffer can provide 0
  // as their buffer.
  // NOTE: The function doesn't explicitly fail for backward compatibility
  // reasons; it just provides a warning in case of failures.
  void ValidateModelBuffers(ErrorReporter* error_reporter);

  /// Returns true if the model identifier is correct (otherwise false and
  /// reports an error).
  bool CheckModelIdentifier() const;

 private:
  /// Loads a model from a given allocation. FlatBufferModel will take over the
  /// ownership of `allocation`, and delete it in destructor. The ownership of
  /// `error_reporter`remains with the caller and must have lifetime at least
  /// as much as FlatBufferModel. This is to allow multiple models to use the
  /// same ErrorReporter instance.
  explicit FlatBufferModel(
      std::unique_ptr<Allocation> allocation,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  /// Loads a model from Model flatbuffer. The `model` has to remain alive and
  /// unchanged until the end of this flatbuffermodel's lifetime.
  FlatBufferModel(const Model* model, ErrorReporter* error_reporter);

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

using FlatBufferModel = impl::FlatBufferModel;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_MODEL_BUILDER_H_
