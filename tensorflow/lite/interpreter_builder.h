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
/// Provides functionality to construct an interpreter for a model.
///
#ifndef TENSORFLOW_LITE_INTERPRETER_BUILDER_H_
#define TENSORFLOW_LITE_INTERPRETER_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {

/// Build an interpreter capable of interpreting `model`.
///
/// `model`: A model whose lifetime must be at least as long as any
///   interpreter(s) created by the builder. In principle multiple interpreters
///   can be made from a single model.
/// `op_resolver`: An instance that implements the `OpResolver` interface, which
///   maps custom op names and builtin op codes to op registrations. The
///   lifetime of the provided `op_resolver` object must be at least as long as
///   the `InterpreterBuilder`; unlike `model` and `error_reporter`, the
///   `op_resolver` does not need to exist for the duration of any created
///   `Interpreter` objects.
/// `error_reporter`: a functor that is called to report errors that handles
///   printf var arg semantics. The lifetime of the `error_reporter` object must
///   be greater than or equal to the `Interpreter` created by `operator()`.
/// `options_experimental`: Options that can change behavior of interpreter.
///   WARNING: this parameter is an experimental API and is subject to change.
///
/// Returns a kTfLiteOk when successful and sets interpreter to a valid
/// Interpreter. Note: The user must ensure the lifetime of the model (and error
/// reporter, if provided) is at least as long as interpreter's lifetime, and
/// a single model instance may safely be used with multiple interpreters.
class InterpreterBuilder {
 public:
  /// For this constructor, the ErrorReporter will be extracted from the
  /// FlatBufferModel.
  /// `options` object is copied during construction. So caller can release it
  // after calling the constructor.
  InterpreterBuilder(const FlatBufferModel& model,
                     const OpResolver& op_resolver,
                     const InterpreterOptions* options_experimental = nullptr);
  /// Builds an interpreter given only the raw flatbuffer Model object (instead
  /// of a FlatBufferModel). Mostly used for testing.
  /// If `error_reporter` is null, then DefaultErrorReporter() is used.
  /// `options` object is copied during construction. So caller can release it
  // after calling the constructor.
  InterpreterBuilder(const ::tflite::Model* model,
                     const OpResolver& op_resolver,
                     ErrorReporter* error_reporter = DefaultErrorReporter(),
                     const InterpreterOptions* options_experimental = nullptr);
  ~InterpreterBuilder();
  InterpreterBuilder(const InterpreterBuilder&) = delete;
  InterpreterBuilder& operator=(const InterpreterBuilder&) = delete;

  /// Builds an interpreter and stores it in `*interpreter`.
  /// On success, returns kTfLiteOk and sets `*interpreter` to a valid
  /// Interpreter.
  /// On failure, returns an error status and sets `*interpreter` to nullptr.
  TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter);

  /// Same as above, but also sets the number of CPU threads to use
  /// (overriding any previous call to SetNumThreads).
  /// Deprecated: use the SetNumThreads method instead.
  TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter,
                          int num_threads);

  /// Sets the number of CPU threads to use for the interpreter.
  /// Returns kTfLiteOk on success, kTfLiteError on error.
  TfLiteStatus SetNumThreads(int num_threads);

  /// Enables preserving intermediates for debugging. Otherwise, by default
  /// intermediates are undefined due to memory planning and reuse.
  InterpreterBuilder& PreserveAllTensorsExperimental();

  /// Any delegates added with AddDelegate will be applied to the Interpreter
  /// generated by operator(), in the order that they were added.  (The delegate
  /// parameter passed to AddDelegate should be non-null, otherwise an error
  /// will be reported, and the call to AddDelegate will have no other effect.)
  /// The lifetime of the delegate must be at least as long as the lifetime of
  /// any Interpreter generated by this InterpreterBuilder.
  void AddDelegate(TfLiteDelegate* delegate);

 private:
  TfLiteStatus BuildLocalIndexToRegistrationMapping();
  TfLiteStatus ParseNodes(
      const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
      Subgraph* subgraph);
  TfLiteStatus ParseTensors(
      const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
      const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
      Subgraph* subgraph);
  TfLiteStatus ApplyDelegates(Interpreter* interpreter);
  TfLiteStatus ParseQuantization(const QuantizationParameters* src_quantization,
                                 TfLiteQuantization* quantization,
                                 const std::vector<int>& dims);
  TfLiteStatus ParseSparsity(const SparsityParameters* src_sparsity,
                             TfLiteSparsity** sparsity);
  TfLiteStatus ParseSignatureDefs(
      const flatbuffers::Vector<flatbuffers::Offset<SignatureDef>>*
          signature_def_list,
      Interpreter* interpreter);

  const ::tflite::Model* model_;
  const OpResolver& op_resolver_;
  ErrorReporter* error_reporter_;
  std::vector<TfLiteDelegate*> delegates_;
  // Model metadata stored as mapping of name (key) to buffer (value).
  // Data is mapped from the Metadata in TFLite flatbuffer model.
  // TODO(b/188185962): Consider mapping to std::pair<const char*, size_t> if
  // this increases runtime memory usage for large metadata.
  std::map<std::string, std::string> metadata_;

  std::vector<const TfLiteRegistration*> flatbuffer_op_index_to_registration_;
  std::vector<TfLiteRegistration> unresolved_custom_ops_;
  std::vector<BuiltinOperator> flatbuffer_op_index_to_registration_types_;
  const Allocation* allocation_ = nullptr;

  bool has_flex_op_ = false;
  int num_fp32_tensors_ = 0;
  int num_threads_ = -1;
  InterpreterOptions options_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_INTERPRETER_BUILDER_H_
