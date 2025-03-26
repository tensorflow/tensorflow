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

#ifndef TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_COMMON_DELEGATE_COMPATIBILITY_CHECKER_BASE_H_
#define TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_COMMON_DELEGATE_COMPATIBILITY_CHECKER_BASE_H_

#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {

// TODO(b/243489631): Remove online mode support.

// Base class for the delegate compatibility checker (DCC). Extracts the logic
// of iterating through the model, and lets each specific DCC to check if the
// operation of a node is compatible with that delegate. TfLiteNode and
// operator (in schema.fbs) are equivalent.
// There are two modes supported: online and offline. Online mode needs
// TfLiteContext while offline mode doesn't. Online mode is supported as an
// intermediate stage and it is recommended that delegates support offline
// mode in validation logic, if possible.
class DelegateCompatibilityCheckerBase {
 public:
  virtual ~DelegateCompatibilityCheckerBase() = default;

  // Iterates over the subgraphs in the model, and for each operator, checks if
  // it is compatible with the delegate.
  // Stores the compatibility for each operator in the result structure.
  absl::Status checkModelCompatibilityOffline(
      tflite::FlatBufferModel* model_buffer,
      tflite::proto::CompatibilityResult* result);

  // This function is implemented differently by each specific DCC.
  // Stores the compatibility for each operator in the result structure.
  virtual absl::Status checkModelCompatibilityOnline(
      tflite::FlatBufferModel* model_buffer,
      tflite::proto::CompatibilityResult* result) = 0;

  // This function gets the operation signature (OpSignature) from the
  // op_code, op, subgraph and model, and then call to checkOpSigCompatibility()
  // with the operation signature.
  // Params:
  //   op_code: Used to get the built in code of the operator, in
  //               order to know which operator is being used (e.g. MUL,
  //               FULLY_CONNECTED…).
  //   op: Used to get the input and output tensors indexes, so that obtains the
  //               tensors with the subgraph.
  //   subgraph: Used to get the tensors to finally get the OpSignature.
  //   model: Used to get the buffer in order to check if the tensor is
  //               a constant tensor.
  //   op_result: Stores whether the operation is compatible or not and why.
  // Returns: absl::OkStatus() if the function is completed without exceptions.
  absl::Status checkOpCompatibilityOffline(
      const tflite::OperatorCode* op_code, const tflite::Operator* op,
      const tflite::SubGraph* subgraph, const tflite::Model* model,
      tflite::proto::OpCompatibilityResult* op_result);

  // Returns the dictionary with the initialized keys. This is an abstract
  // function because each specific DCC needs different parameters, which will
  // be the keys in the returned dictionary.
  virtual std::unordered_map<std::string, std::string>
  getDccConfigurations() = 0;

  // Sets the parameters needed in the specific DCC. Also checks if the
  // value types are correct.
  virtual absl::Status setDccConfigurations(
      const std::unordered_map<std::string, std::string>& dcc_configs) = 0;

 private:
  // This function is implemented differently by each specific DCC because
  // they contain the logic for checking if the operation in op_sig is
  // compatible for that specific DCC. op_result stores whether the operation
  // is supported or not, and why. By using offline mode, only op_signature is
  // used to perform the checks.
  virtual absl::Status checkOpSigCompatibility(
      const tflite::OpSignature& op_sig,
      tflite::proto::OpCompatibilityResult* op_result) = 0;
};
}  // namespace tools
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_DELEGATES_COMPATIBILITY_COMMON_DELEGATE_COMPATIBILITY_CHECKER_BASE_H_
