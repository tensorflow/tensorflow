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
#include <string>
#include <vector>

#include "re2/re2.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

namespace tflite {

string NormalizeCustomOpName(const string& op) {
  string method(op);
  RE2::GlobalReplace(&method, "([a-z])([A-Z])", "\\1_\\2");
  std::transform(method.begin(), method.end(), method.begin(), ::toupper);
  return method;
}

void ReadOpsFromModel(const ::tflite::Model* model,
                      tflite::RegisteredOpMap* builtin_ops,
                      tflite::RegisteredOpMap* custom_ops) {
  if (!model) return;
  auto opcodes = model->operator_codes();
  if (!opcodes) return;
  for (const auto* opcode : *opcodes) {
    const int version = opcode->version();
    if (opcode->builtin_code() != ::tflite::BuiltinOperator_CUSTOM) {
      auto iter_and_bool = builtin_ops->insert(std::make_pair(
          tflite::EnumNameBuiltinOperator(opcode->builtin_code()),
          std::make_pair(version, version)));
      auto& versions = iter_and_bool.first->second;
      versions.first = std::min(versions.first, version);
      versions.second = std::max(versions.second, version);
    } else {
      auto iter_and_bool = custom_ops->insert(std::make_pair(
          opcode->custom_code()->c_str(), std::make_pair(version, version)));
      auto& versions = iter_and_bool.first->second;
      versions.first = std::min(versions.first, version);
      versions.second = std::max(versions.second, version);
    }
  }
}

}  // namespace tflite
