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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_CUSTOM_OPERATOR_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_CUSTOM_OPERATOR_H_

#include "flatbuffers/flexbuffers.h"
#include "absl/memory/memory.h"
#include "tensorflow/contrib/lite/toco/tflite/operator.h"

namespace toco {

namespace tflite {

// Custom operators have a generic byte buffer describing their options. This
// class provides the boilerplate code for populating those options using
// flexbuffers. Note that most of toco's operators will likely be supported
// as builtin operators in TF Lite.
//
// Template argument T must derive from ::toco::Operator.
template <typename T>
class CustomOperator : public BaseOperator {
 public:
  using TocoOperator = T;
  using BaseOperator::BaseOperator;

  // Populate the given flexbuffer with options obtained from the tf.mini
  // operator.
  virtual void WriteOptions(const TocoOperator& op,
                            flexbuffers::Builder* fbb) const {}

  // Set options in the given tf.mini operator using values from the flexbuffer
  // map.
  virtual void ReadOptions(const flexbuffers::Map& m, TocoOperator* op) const {}

  Options Serialize(const Operator& op,
                    flatbuffers::FlatBufferBuilder* builder) const override {
    flexbuffers::Builder fbb;
    fbb.Map(
        [&]() { WriteOptions(static_cast<const TocoOperator&>(op), &fbb); });
    fbb.Finish();
    return Options::Custom(builder->CreateVector(fbb.GetBuffer()));
  }

  std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const override {
    auto op = absl::make_unique<TocoOperator>();
    if (custom_options) {
      auto flexbuffer_map =
          flexbuffers::GetRoot(custom_options->data(), custom_options->size())
              .AsMap();
      ReadOptions(flexbuffer_map, op.get());
    }
    return std::unique_ptr<Operator>(op.release());
  }
};

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_CUSTOM_OPERATOR_H_
