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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_EXPORT_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_EXPORT_H_

#include "tensorflow/contrib/lite/toco/model.h"

namespace toco {

namespace tflite {

// Transform the given tf.mini model into a TF Lite flatbuffer and deposit the
// result in the given string.
void Export(const Model& model, bool allow_custom_ops,
            string* output_file_contents);
// This if backward-compatibility.
inline void Export(const Model& model, string* output_file_contents) {
  Export(model, true, output_file_contents);
}

namespace details {

// A maps from tensor name to its final position in the TF Lite buffer.
using TensorsMap = std::unordered_map<string, int>;

// A key to identify an operator.
// Only when `type` is `kTensorFlowUnsupported`, `custom_code` is filled to
// identify which operation is used.
struct OperatorKey {
  OperatorKey(OperatorType type, const std::string& custom_code)
      : type(type), custom_code(custom_code) {}
  const OperatorType type;
  const std::string custom_code;

  bool operator<(const OperatorKey& other) const {
    if (type < other.type) return true;
    if (type > other.type) return false;
    return custom_code < other.custom_code;
  }

  bool operator==(const OperatorKey& other) const {
    return type == other.type && custom_code == other.custom_code;
  }

  struct Hash {
    std::size_t operator()(const OperatorKey& key) const {
      return std::hash<size_t>()(static_cast<size_t>(key.type)) ^
             std::hash<std::string>()(key.custom_code);
    }
  };
};

// A maps from operator type to its final position in the TF Lite buffer.
using OperatorsMap = std::unordered_map<OperatorKey, int, OperatorKey::Hash>;

void LoadTensorsMap(const Model& model, TensorsMap* tensors_map);
void LoadOperatorsMap(const Model& model, OperatorsMap* operators_map);

}  // namespace details
}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_EXPORT_H_
