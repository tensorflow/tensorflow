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
#ifndef TENSORFLOW_LITE_CORE_SIGNATURE_RUNNER_H_
#define TENSORFLOW_LITE_CORE_SIGNATURE_RUNNER_H_
/// \file
///
/// An abstraction for invoking the TF Lite interpreter.
/// Provides support for named parameters, and for including multiple
/// named computations in a single model, each with its own inputs/outputs.
///
/// Do NOT include this file directly,
/// instead include third_party/tensorflow/lite/signature_riunner.h
/// See third_party/tensorflow/lite/c/common.h for the API for defining
/// operations (TfLiteRegistration).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/internal/signature_def.h"

namespace tflite {
namespace impl {
class Interpreter;  // Class for friend declarations.
}
class SignatureRunnerHelper;     // Class for friend declarations.
class SignatureRunnerJNIHelper;  // Class for friend declarations.
class TensorHandle;              // Class for friend declarations.

namespace impl {
/// SignatureRunner class for running TFLite models using SignatureDef.
///
/// Usage:
///
/// <pre><code>
/// // Create model from file. Note that the model instance must outlive the
/// // interpreter instance.
/// auto model = tflite::FlatBufferModel::BuildFromFile(...);
/// if (model == nullptr) {
///   // Return error.
/// }
///
/// // Create an Interpreter with an InterpreterBuilder.
/// std::unique_ptr<tflite::Interpreter> interpreter;
/// tflite::ops::builtin::BuiltinOpResolver resolver;
/// if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
///   // Return failure.
/// }
///
/// // Get the list of signatures and check it.
/// auto signature_defs = interpreter->signature_keys();
/// if (signature_defs.empty()) {
///   // Return error.
/// }
///
/// // Get pointer to the SignatureRunner instance corresponding to a signature.
/// // Note that the pointed SignatureRunner instance has lifetime same as the
/// // Interpreter instance.
/// tflite::SignatureRunner* runner =
///                interpreter->GetSignatureRunner(signature_defs[0]->c_str());
/// if (runner == nullptr) {
///   // Return error.
/// }
/// if (runner->AllocateTensors() != kTfLiteOk) {
///   // Return failure.
/// }
///
/// // Set input data. In this example, the input tensor has float type.
/// float* input = runner->input_tensor(0)->data.f;
/// for (int i = 0; i < input_size; i++) {
///   input[i] = ...;
//  }
/// runner->Invoke();
/// </code></pre>
///
/// WARNING: This class is *not* thread-safe. The client is responsible for
/// ensuring serialized interaction to avoid data races and undefined behavior.
///
/// SignatureRunner and Interpreter share the same underlying data. Calling
/// methods on an Interpreter object will affect the state in corresponding
/// SignatureRunner objects. Therefore, it is recommended not to call other
/// Interpreter methods after calling GetSignatureRunner to create
/// SignatureRunner instances.
class SignatureRunner {
 public:
  /// Returns the key for the corresponding signature.
  const std::string& signature_key() { return signature_def_->signature_key; }

  /// Returns the number of inputs.
  size_t input_size() const { return subgraph_->inputs().size(); }

  /// Returns the number of outputs.
  size_t output_size() const { return subgraph_->outputs().size(); }

  /// Read-only access to list of signature input names.
  const std::vector<const char*>& input_names() { return input_names_; }

  /// Read-only access to list of signature output names.
  const std::vector<const char*>& output_names() { return output_names_; }

  /// Returns the input tensor identified by 'input_name' in the
  /// given signature. Returns nullptr if the given name is not valid.
  TfLiteTensor* input_tensor(const char* input_name);

  /// Returns the output tensor identified by 'output_name' in the
  /// given signature. Returns nullptr if the given name is not valid.
  const TfLiteTensor* output_tensor(const char* output_name) const;

  /// Change a dimensionality of a given tensor. Note, this is only acceptable
  /// for tensors that are inputs.
  /// Returns status of failure or success. Note that this doesn't actually
  /// resize any existing buffers. A call to AllocateTensors() is required to
  /// change the tensor input buffer.
  TfLiteStatus ResizeInputTensor(const char* input_name,
                                 const std::vector<int>& new_size);

  /// Change the dimensionality of a given tensor. This is only acceptable for
  /// tensor indices that are inputs or variables.
  ///
  /// Difference from ResizeInputTensor: Only unknown dimensions can be resized
  /// with this function. Unknown dimensions are indicated as `-1` in the
  /// `dims_signature` attribute of a TfLiteTensor.
  ///
  /// Returns status of failure or success. Note that this doesn't actually
  /// resize any existing buffers. A call to AllocateTensors() is required to
  /// change the tensor input buffer.
  TfLiteStatus ResizeInputTensorStrict(const char* input_name,
                                       const std::vector<int>& new_size);

  /// Updates allocations for all tensors, related to the given signature.
  TfLiteStatus AllocateTensors() { return subgraph_->AllocateTensors(); }

  /// Invokes the signature runner (run the graph identified by the given
  /// signature in dependency order).
  TfLiteStatus Invoke();

  /// Attempts to cancel in flight invocation if any.
  /// This will not affect calls to `Invoke` that happened after this.
  /// Non blocking and thread safe.
  /// Returns kTfLiteError if cancellation is not enabled, otherwise returns
  /// kTfLiteOk.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus Cancel() { return subgraph_->Cancel(); }

  /// \brief Assigns (or reassigns) a custom memory allocation for the given
  /// tensor name. `flags` is a bitmask, see TfLiteCustomAllocationFlags.
  /// The runtime does NOT take ownership of the underlying memory.
  ///
  /// NOTE: User needs to call AllocateTensors() after this.
  /// Invalid/insufficient buffers will cause an error during AllocateTensors or
  /// Invoke (in case of dynamic shapes in the graph).
  ///
  /// Parameters should satisfy the following conditions:
  /// 1. tensor->allocation_type == kTfLiteArenaRw or kTfLiteArenaRwPersistent
  ///    In general, this is true for I/O tensors & variable tensors.
  /// 2. allocation->data has the appropriate permissions for runtime access
  ///    (Read-only for inputs, Read-Write for others), and outlives
  ///    Interpreter.
  /// 3. allocation->bytes >= tensor->bytes.
  ///    This condition is checked again if any tensors are resized.
  /// 4. allocation->data should be aligned to kDefaultTensorAlignment
  ///    defined in lite/util.h. (Currently 64 bytes)
  ///    This check is skipped if kTfLiteCustomAllocationFlagsSkipAlignCheck is
  ///    set through `flags`.
  /// \warning This is an experimental API and subject to change. \n
  TfLiteStatus SetCustomAllocationForInputTensor(
      const char* input_name, const TfLiteCustomAllocation& allocation,
      int64_t flags = kTfLiteCustomAllocationFlagsNone);

  /// \brief Assigns (or reassigns) a custom memory allocation for the given
  /// tensor name. `flags` is a bitmask, see TfLiteCustomAllocationFlags.
  /// The runtime does NOT take ownership of the underlying memory.
  ///
  /// NOTE: User needs to call AllocateTensors() after this.
  /// Invalid/insufficient buffers will cause an error during AllocateTensors or
  /// Invoke (in case of dynamic shapes in the graph).
  ///
  /// Parameters should satisfy the following conditions:
  /// 1. tensor->allocation_type == kTfLiteArenaRw or kTfLiteArenaRwPersistent
  ///    In general, this is true for I/O tensors & variable tensors.
  /// 2. allocation->data has the appropriate permissions for runtime access
  ///    (Read-only for inputs, Read-Write for others), and outlives
  ///    Interpreter.
  /// 3. allocation->bytes >= tensor->bytes.
  ///    This condition is checked again if any tensors are resized.
  /// 4. allocation->data should be aligned to kDefaultTensorAlignment
  ///    defined in lite/util.h. (Currently 64 bytes)
  ///    This check is skipped if kTfLiteCustomAllocationFlagsSkipAlignCheck is
  ///    set through `flags`.
  /// \warning This is an experimental API and subject to change. \n
  TfLiteStatus SetCustomAllocationForOutputTensor(
      const char* output_name, const TfLiteCustomAllocation& allocation,
      int64_t flags = kTfLiteCustomAllocationFlagsNone);

  /// \brief Set if buffer handle output is allowed.
  ///
  /// When using hardware delegation, Interpreter will make the data of output
  /// tensors available in `tensor->data` by default. If the application can
  /// consume the buffer handle directly (e.g. reading output from OpenGL
  /// texture), it can set this flag to true, so Interpreter won't copy the
  /// data from buffer handle to CPU memory.
  /// \warning This is an experimental API and subject to change. \n
  void SetAllowBufferHandleOutput(bool allow_buffer_handle_output) {
    allow_buffer_handle_output_ = allow_buffer_handle_output;
  }

  /// \warning This is an experimental API and subject to change. \n
  /// \brief Set the delegate buffer handle to a input tensor.
  /// TfLiteDelegate should be aware of how to handle the buffer handle.
  /// `release_existing_buffer_handle`: If true, the existing buffer handle
  // will be released by TfLiteDelegate::FreeBufferHandle.
  TfLiteStatus SetInputBufferHandle(const char* input_name,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteDelegate* delegate,
                                    bool release_existing_buffer_handle = true);

  /// \warning This is an experimental API and subject to change. \n
  /// \brief Set the delegate buffer handle to a output tensor.
  /// TfLiteDelegate should be aware of how to handle the buffer handle.
  /// `release_existing_buffer_handle`: If true, the existing buffer handle
  /// will be released by TfLiteDelegate::FreeBufferHandle.
  TfLiteStatus SetOutputBufferHandle(
      const char* output_name, TfLiteBufferHandle buffer_handle,
      TfLiteDelegate* delegate, bool release_existing_buffer_handle = true);

 private:
  // The life cycle of SignatureRunner depends on the life cycle of Subgraph,
  // which is owned by an Interpreter. Therefore, the Interpreter will takes the
  // responsibility to create and manage SignatureRunner objects to make sure
  // SignatureRunner objects don't outlive their corresponding Subgraph objects.
  SignatureRunner(const internal::SignatureDef* signature_def,
                  Subgraph* subgraph);
  friend class ::tflite::impl::Interpreter;
  friend class ::tflite::SignatureRunnerHelper;
  friend class ::tflite::SignatureRunnerJNIHelper;
  friend class ::tflite::TensorHandle;

  // The SignatureDef object is owned by the interpreter.
  const internal::SignatureDef* signature_def_;
  // The Subgraph object is owned by the interpreter.
  Subgraph* subgraph_;
  // The list of input tensor names.
  std::vector<const char*> input_names_;
  // The list of output tensor names.
  std::vector<const char*> output_names_;

  bool allow_buffer_handle_output_ = false;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_SIGNATURE_RUNNER_H_
