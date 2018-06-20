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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_BUILTIN_OPERATOR_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_BUILTIN_OPERATOR_H_

#include "absl/memory/memory.h"
#include "tensorflow/contrib/lite/toco/tflite/operator.h"

namespace toco {

namespace tflite {

// Builtin operators have special TF Lite objects describing their options.
// This class has the boilerplate code for creating those.
//
// Template arguments:
//   - T1 must derive from ::toco::Operator.
//   - T2 must be one of TF Lite's objects defining Builtin Options, such as
//     ::tflite::Conv2DOptions.
template <typename T1, typename T2, ::tflite::BuiltinOptions TfLiteEnum>
class BuiltinOperator : public BaseOperator {
 public:
  using TocoOperator = T1;
  using TfLiteOptions = T2;

  BuiltinOperator(::tflite::BuiltinOperator op, OperatorType type)
      : BaseOperator(::tflite::EnumNameBuiltinOperator(op), type) {}

  // Build the configuration object in the given flatbuffer builder. Return
  // its offset.
  virtual flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const = 0;

  // Read options from the TF Lite object and set the corresponding values in
  // the tf.mini operator.
  virtual void ReadOptions(const TfLiteOptions& opt,
                           TocoOperator* op) const = 0;

  Options Serialize(const Operator& op,
                    flatbuffers::FlatBufferBuilder* builder) const override {
    auto options = WriteOptions(static_cast<const TocoOperator&>(op), builder);
    return Options::Builtin(TfLiteEnum, options.Union());
  }

  std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const override {
    auto op = absl::make_unique<TocoOperator>();
    auto* options = static_cast<const TfLiteOptions*>(builtin_options);
    if (options) {
      ReadOptions(*options, op.get());
    }
    return std::unique_ptr<Operator>(op.release());
  }
};

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_BUILTIN_OPERATOR_H_
