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
#include "tensorflow/contrib/lite/toco/tflite/operator.h"

namespace toco {

namespace tflite {

// Transform the given tf.mini model into a TF Lite flatbuffer and deposit the
// result in the given string.
void Export(const Model& model, bool allow_custom_ops,
            string* output_file_contents);

// This if backward-compatibility.
// TODO(ycling): Remove the deprecated entry functions.
inline void Export(const Model& model, string* output_file_contents) {
  Export(model, true, output_file_contents);
}

// Export API with custom TFLite operator mapping.
void Export(
    const Model& model, bool allow_custom_ops, string* output_file_contents,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type);

namespace details {

// A maps from tensor name to its final position in the TF Lite buffer.
using TensorsMap = std::unordered_map<string, int>;

// A key to identify an operator.
// Only when `type` is `kTensorFlowUnsupported`, `custom_code` is filled to
// identify which operation is used.
struct OperatorKey {
  OperatorKey(OperatorType type, const std::string& custom_code, int version)
      : type(type), custom_code(custom_code), version(version) {}
  const OperatorType type;
  const std::string custom_code;
  const int version;

  bool operator<(const OperatorKey& other) const {
    if (type < other.type) return true;
    else if (type > other.type)
      return false;
    else if (custom_code < other.custom_code)
      return true;
    else if (custom_code > other.custom_code)
      return false;
    else
      return version < other.version;
  }

  bool operator==(const OperatorKey& other) const {
    return type == other.type && custom_code == other.custom_code &&
           version == other.version;
  }

  struct Hash {
    size_t operator()(const OperatorKey& key) const {
      return CombineHashes({std::hash<size_t>()(static_cast<size_t>(key.type)),
                            std::hash<std::string>()(key.custom_code),
                            std::hash<int>()(key.version)});
    }

   private:
    // TODO(ycling): Refactoring and extract this function into a common
    // utility module.
    static size_t CombineHashes(std::initializer_list<size_t> hashes) {
      size_t result = 0;
      // Hash combiner used by TensorFlow core.
      for (size_t hash : hashes) {
        result = result ^ (hash + 0x9e3779b97f4a7800ULL + (result << 10) +
                           (result >> 4));
      }
      return result;
    }
  };
};

// A maps from operator type to its final position in the TF Lite buffer.
using OperatorsMap = std::unordered_map<OperatorKey, int, OperatorKey::Hash>;

void LoadTensorsMap(const Model& model, TensorsMap* tensors_map);
void LoadOperatorsMap(
    const Model& model, OperatorsMap* operators_map,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type);

}  // namespace details
}  // namespace tflite
}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_EXPORT_H_
