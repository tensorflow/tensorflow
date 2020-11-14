/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_COMMON_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_COMMON_H_

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace optimize {
namespace calibration {
using BuiltinOperatorKey = std::pair<BuiltinOperator, int>;

using CustomOperatorKey = std::pair<std::string, int>;

using BuiltinOpsSet = std::unordered_set<
    BuiltinOperatorKey,
    op_resolver_hasher::OperatorKeyHasher<BuiltinOperatorKey>>;

using CustomOpsSet = std::unordered_set<
    CustomOperatorKey,
    op_resolver_hasher::OperatorKeyHasher<CustomOperatorKey>>;

template <typename T>
class BuiltinOpsMap
    : public std::unordered_map<
          BuiltinOperatorKey, T,
          op_resolver_hasher::OperatorKeyHasher<BuiltinOperatorKey>> {};

template <typename T>
class CustomOpsMap
    : public std::unordered_map<
          CustomOperatorKey, T,
          op_resolver_hasher::OperatorKeyHasher<CustomOperatorKey>> {};

// An alias for |TfLiteRegistration.invoke|.
using KernelEvalFuncPtr = TfLiteStatus (*)(TfLiteContext*, TfLiteNode*);

enum class OperatorTensorType { kNone, kInput, kOutput, kIntermediate };

// Information about an operator in the TfLite graph.
struct OperatorInfo {
  int node_index;
  std::string name;
  BuiltinOperator builtin_op_code;
  bool is_custom_op;
  std::vector<int> inputs;
  std::vector<int> outputs;
  // Inputs that need to be logged.
  std::vector<int> loggable_inputs;
  // Outputs that need to be logged.
  std::vector<int> loggable_outputs;
  const TfLiteRegistration* registration;
  int version;
};

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_COMMON_H_
