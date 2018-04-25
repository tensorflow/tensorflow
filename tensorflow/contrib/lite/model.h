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
// Deserialization infrastructure for tflite. Provides functionality
// to go from a serialized tflite model in flatbuffer format to an
// interpreter.
//
// using namespace tflite;
// StderrReporter error_reporter;
// auto model = FlatBufferModel::BuildFromFile("interesting_model.tflite",
//                                             &error_reporter);
// MyOpResolver resolver;  // You need to subclass OpResolver to provide
//                         // implementations.
// InterpreterBuilder builder(*model, resolver);
// std::unique_ptr<Interpreter> interpreter;
// if(builder(&interpreter) == kTfLiteOk) {
//   .. run model inference with interpreter
// }
//
// OpResolver must be defined to provide your kernel implementations to the
// interpreter. This is environment specific and may consist of just the builtin
// ops, or some custom operators you defined to extend tflite.
#ifndef TENSORFLOW_CONTRIB_LITE_MODEL_H_
#define TENSORFLOW_CONTRIB_LITE_MODEL_H_

#include <memory>
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"

namespace tflite {

// Abstract interface that verifies whether a given model is legit.
// It facilitates the use-case to verify and build a model without loading it
// twice.
class TfLiteVerifier {
 public:
  // Returns true if the model is legit.
  virtual bool Verify(const char* data, int length,
                      ErrorReporter* reporter) = 0;
  virtual ~TfLiteVerifier() {}
};

// An RAII object that represents a read-only tflite model, copied from disk,
// or mmapped. This uses flatbuffers as the serialization format.
class FlatBufferModel {
 public:
  // Builds a model based on a file.
  // Caller retains ownership of `error_reporter` and must ensure its lifetime
  // is longer than the FlatBufferModel instance.
  // Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  // Verifies whether the content of the file is legit, then builds a model
  // based on the file.
  // Caller retains ownership of `error_reporter` and must ensure its lifetime
  // is longer than the FlatBufferModel instance.
  // Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> VerifyAndBuildFromFile(
      const char* filename, TfLiteVerifier* verifier = nullptr,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  // Builds a model based on a pre-loaded flatbuffer. The caller retains
  // ownership of the buffer and should keep it alive until the returned object
  // is destroyed. Caller retains ownership of `error_reporter` and must ensure
  // its lifetime is longer than the FlatBufferModel instance.
  // Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
      const char* buffer, size_t buffer_size,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  // Builds a model directly from a flatbuffer pointer. The caller retains
  // ownership of the buffer and should keep it alive until the returned object
  // is destroyed. Caller retains ownership of `error_reporter` and must ensure
  // its lifetime is longer than the FlatBufferModel instance.
  // Returns a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromModel(
      const tflite::Model* model_spec,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  // Releases memory or unmaps mmaped memory.
  ~FlatBufferModel();

  // Copying or assignment is disallowed to simplify ownership semantics.
  FlatBufferModel(const FlatBufferModel&) = delete;
  FlatBufferModel& operator=(const FlatBufferModel&) = delete;

  bool initialized() const { return model_ != nullptr; }
  const tflite::Model* operator->() const { return model_; }
  const tflite::Model* GetModel() const { return model_; }
  ErrorReporter* error_reporter() const { return error_reporter_; }
  const Allocation* allocation() const { return allocation_; }

  // Returns true if the model identifier is correct (otherwise false and
  // reports an error).
  bool CheckModelIdentifier() const;

 private:
  // Loads a model from a given allocation. FlatBufferModel will take over the
  // ownership of `allocation`, and delete it in destructor. The ownership of
  // `error_reporter`remains with the caller and must have lifetime at least
  // as much as FlatBufferModel. This is to allow multiple models to use the
  // same ErrorReporter instance.
  FlatBufferModel(Allocation* allocation,
                  ErrorReporter* error_reporter = DefaultErrorReporter());

  // Loads a model from Model flatbuffer. The `model` has to remain alive and
  // unchanged until the end of this flatbuffermodel's lifetime.
  FlatBufferModel(const Model* model, ErrorReporter* error_reporter);

  // Flatbuffer traverser pointer. (Model* is a pointer that is within the
  // allocated memory of the data allocated by allocation's internals.
  const tflite::Model* model_ = nullptr;
  // The error reporter to use for model errors and subsequent errors when
  // the interpreter is created
  ErrorReporter* error_reporter_;
  // The allocator used for holding memory of the model.
  Allocation* allocation_ = nullptr;
};

// Abstract interface that returns TfLiteRegistrations given op codes or custom
// op names. This is the mechanism that ops being referenced in the flatbuffer
// model are mapped to executable function pointers (TfLiteRegistrations).
class OpResolver {
 public:
  // Finds the op registration for a builtin operator by enum code.
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  // Finds the op registration of a custom operator by op name.
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual ~OpResolver() {}
};

// Build an interpreter capable of interpreting `model`.
//
// model: a scoped model whose lifetime must be at least as long as
//   the interpreter. In principle multiple interpreters can be made from
//   a single model.
// op_resolver: An instance that implements the Resolver interface which maps
//   custom op names and builtin op codes to op registrations.
// reportError: a functor that is called to report errors that handles
//   printf var arg semantics. The lifetime of the reportError object must
//   be greater than or equal to the Interpreter created by operator().
//
// Returns a kTfLiteOk when successful and sets interpreter to a valid
// Interpreter. Note: the user must ensure the model lifetime is at least as
// long as interpreter's lifetime.
class InterpreterBuilder {
 public:
  InterpreterBuilder(const FlatBufferModel& model,
                     const OpResolver& op_resolver);
  // Builds an interpreter given only the raw flatbuffer Model object (instead
  // of a FlatBufferModel). Mostly used for testing.
  // If `error_reporter` is null, then DefaultErrorReporter() is used.
  InterpreterBuilder(const ::tflite::Model* model,
                     const OpResolver& op_resolver,
                     ErrorReporter* error_reporter = DefaultErrorReporter());
  InterpreterBuilder(const InterpreterBuilder&) = delete;
  InterpreterBuilder& operator=(const InterpreterBuilder&) = delete;
  TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter);
  TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter,
                          int num_threads);

 private:
  TfLiteStatus BuildLocalIndexToRegistrationMapping();
  TfLiteStatus ParseNodes(
      const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
      Interpreter* interpreter);
  TfLiteStatus ParseTensors(
      const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
      const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
      Interpreter* interpreter);

  const ::tflite::Model* model_;
  const OpResolver& op_resolver_;
  ErrorReporter* error_reporter_;

  std::vector<TfLiteRegistration*> flatbuffer_op_index_to_registration_;
  const Allocation* allocation_ = nullptr;
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_MODEL_H_
