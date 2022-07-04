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
#ifndef TENSORFLOW_LITE_SIGNATURE_RUNNER_H_
#define TENSORFLOW_LITE_SIGNATURE_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/internal/signature_def.h"

namespace tflite {
class Interpreter;  // Class for friend declarations.
class SignatureRunnerJNIHelper;  // Class for friend declarations.
class TensorHandle;              // Class for friend declarations.

/// WARNING: Experimental interface, subject to change
///
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
/// auto signature_defs = interpreter->signature_def_names();
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
  /// tensor indices that are inputs or variables. Only unknown dimensions can
  /// be resized with this function. Unknown dimensions are indicated as `-1` in
  /// the `dims_signature` attribute of a TfLiteTensor.
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

 private:
  // The life cycle of SignatureRunner depends on the life cycle of Subgraph,
  // which is owned by an Interpreter. Therefore, the Interpreter will takes the
  // responsibility to create and manage SignatureRunner objects to make sure
  // SignatureRunner objects don't outlive their corresponding Subgraph objects.
  SignatureRunner(const internal::SignatureDef* signature_def,
                  Subgraph* subgraph);
  friend class Interpreter;
  friend class SignatureRunnerJNIHelper;
  friend class TensorHandle;

  // The SignatureDef object is owned by the interpreter.
  const internal::SignatureDef* signature_def_;
  // The Subgraph object is owned by the interpreter.
  Subgraph* subgraph_;
  // The list of input tensor names.
  std::vector<const char*> input_names_;
  // The list of output tensor names.
  std::vector<const char*> output_names_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_SIGNATURE_RUNNER_H_
