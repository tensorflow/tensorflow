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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_

#include <string>

#include "absl/log/log.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/util.h"

namespace toco {

namespace tflite {

enum class QuantizedBufferType { NONE, INT8, FLOAT16 };

// The parameters for exporting a TFLite model.
struct ExportParams {
  bool allow_custom_ops = false;
  bool allow_dynamic_tensors = true;
  bool enable_select_tf_ops = false;
  QuantizedBufferType quantize_weights = QuantizedBufferType::NONE;
  // Whether to use per-tensor (false) or per-channel (true) for hybrid quant.
  bool disable_per_channel = false;
};

// Transform the given tf.mini model into a TF Lite flatbuffer and deposit the
// result in the given string.
tensorflow::Status Export(const Model& model, std::string* output_file_contents,
                          const ExportParams& params);

// Export API with custom TFLite operator mapping.
tensorflow::Status Export(
    const Model& model, std::string* output_file_contents,
    const ExportParams& params,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type);

// This is for backward-compatibility.
inline void Export(const Model& model, bool allow_custom_ops,
                   bool quantize_weights, std::string* output_file_contents) {
  ExportParams params;
  params.allow_custom_ops = allow_custom_ops;
  params.quantize_weights =
      quantize_weights ? QuantizedBufferType::INT8 : QuantizedBufferType::NONE;
  auto status = Export(model, output_file_contents, params);
  if (!status.ok()) LOG(QFATAL) << status.message();
}

// This is for backward-compatibility.
inline void Export(
    const Model& model, bool allow_custom_ops, bool quantize_weights,
    std::string* output_file_contents,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type) {
  ExportParams params;
  params.allow_custom_ops = allow_custom_ops;
  params.quantize_weights =
      quantize_weights ? QuantizedBufferType::INT8 : QuantizedBufferType::NONE;
  auto status = Export(model, output_file_contents, params, ops_by_type);
  if (!status.ok()) LOG(QFATAL) << status.message();
}

// This is for backward-compatibility.
inline void Export(const Model& model, std::string* output_file_contents) {
  ExportParams params;
  params.allow_custom_ops = true;
  auto status = Export(model, output_file_contents, params);
  if (!status.ok()) LOG(QFATAL) << status.message();
}

namespace details {

// A map from tensor name to its final position in the TF Lite buffer.
using TensorsMap = std::unordered_map<std::string, int>;

// A key to identify an operator.
// Only when `type` is `kUnsupported`, `custom_code` is filled to
// identify which operation is used.
class OperatorKey {
 public:
  OperatorKey() {}

  // Construct OperatorKey by Toco op.
  OperatorKey(
      const ::toco::OperatorSignature& op_signature,
      const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
      bool enable_select_tf_ops);

  // Construct OperatorKey by type, custom code and version.
  // Note that this construct doesn't set the additional information including
  // `is_custom_op`, `is_flex_op`, `is_unsupported_flex_op`.
  OperatorKey(::tflite::BuiltinOperator type, const std::string& custom_code,
              int version)
      : type_(type), custom_code_(custom_code), version_(version) {}

  // Only `type`, `custom_code` and `version` is used to compute hash and
  // identity.
  ::tflite::BuiltinOperator type() const { return type_; }
  const std::string& custom_code() const { return custom_code_; }
  int version() const { return version_; }

  // The attributes below are not used to compute hash and identity.
  //
  // Return true if the op is a custom op. Note it will return false for Flex
  // ops.
  bool is_custom_op() const { return is_custom_op_; }
  // Return true if the op is a Flex op.
  bool is_flex_op() const { return is_flex_op_; }
  // Return true if the op is a Flex op but it's knwon that the op is not
  // supported by Flex runtime.
  bool is_unsupported_flex_op() const { return is_unsupported_flex_op_; }
  // Return the original TensorFlow op name for a Flex op.
  const std::string& flex_tensorflow_op() const { return flex_tensorflow_op_; }

  bool operator<(const OperatorKey& other) const {
    if (type_ < other.type_)
      return true;
    else if (type_ > other.type_)
      return false;
    else if (custom_code_ < other.custom_code_)
      return true;
    else if (custom_code_ > other.custom_code_)
      return false;
    else
      return version_ < other.version_;
  }

  bool operator==(const OperatorKey& other) const {
    return type_ == other.type_ && custom_code_ == other.custom_code_ &&
           version_ == other.version_;
  }

  struct Hash {
    size_t operator()(const OperatorKey& key) const {
      return ::tflite::CombineHashes(
          {std::hash<size_t>()(static_cast<size_t>(key.type())),
           std::hash<std::string>()(key.custom_code()),
           std::hash<int>()(key.version())});
    }
  };

 private:
  ::tflite::BuiltinOperator type_ = ::tflite::BuiltinOperator_CUSTOM;
  std::string custom_code_;
  int version_ = 1;

  bool is_custom_op_ = false;
  bool is_flex_op_ = false;
  bool is_unsupported_flex_op_ = false;
  // The original TensorFlow op name for the flex op. Filled only when
  // `is_flex_op` is true.
  std::string flex_tensorflow_op_;
};

// A map from OperatorKey to its final position in the TF Lite buffer.
using OperatorsMap = std::unordered_map<OperatorKey, int, OperatorKey::Hash>;

void LoadTensorsMap(const Model& model, TensorsMap* tensors_map);
void LoadOperatorsMap(
    const Model& model, OperatorsMap* operators_map,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    bool enable_select_tf_ops);

}  // namespace details
}  // namespace tflite
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_
