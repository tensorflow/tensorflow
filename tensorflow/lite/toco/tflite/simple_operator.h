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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_SIMPLE_OPERATOR_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_SIMPLE_OPERATOR_H_

#include "tensorflow/lite/toco/tflite/operator.h"

namespace toco {

namespace tflite {

// Simple operators don't have any configuration options and can be trivially
// serialized and deserialized. Note that most of toco's operators will
// likely be supported as builtin operators in TF Lite.  Simple (and custom)
// operators are mostly a convenience for the times when tf.mini supports more
// operators than TF Lite.
//
// Template argument T must derive from ::toco::Operator.
template <typename T>
class SimpleOperator : public BaseOperator {
 public:
  using BaseOperator::BaseOperator;

  SimpleOperator(::tflite::BuiltinOperator op, OperatorType type)
      : BaseOperator(::tflite::EnumNameBuiltinOperator(op), type),
        builtin_op_(op) {}

  Options Serialize(const Operator& op,
                    flatbuffers::FlatBufferBuilder* builder) const override {
    return Options();
  }
  std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const override {
    return std::unique_ptr<Operator>(new T);
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    return ::tflite::GetBuiltinOperatorVersion(
        GetVersioningOpSig(builtin_op_, op_signature));
  }

  ::tflite::BuiltinOperator builtin_op() const { return builtin_op_; }

 private:
  const ::tflite::BuiltinOperator builtin_op_;
};

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_SIMPLE_OPERATOR_H_
