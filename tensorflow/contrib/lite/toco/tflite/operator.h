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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_OPERATOR_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_OPERATOR_H_

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/toco/model.h"

namespace toco {

namespace tflite {

class BaseOperator;

// Return a map contained all know TF Lite Operators, keyed by their names.
std::map<string, std::unique_ptr<BaseOperator>> BuildOperatorByNameMap();

// Return a map contained all know TF Lite Operators, keyed by the type of
// their tf.mini counterparts.
std::map<OperatorType, std::unique_ptr<BaseOperator>> BuildOperatorByTypeMap();

// These are the flatbuffer types for custom and builtin options.
using CustomOptions = flatbuffers::Vector<uint8_t>;
using BuiltinOptions = void;

// A simple wrapper around the flatbuffer objects used to describe options that
// configure operators.
struct Options {
  // Build custom options.
  static Options Custom(flatbuffers::Offset<CustomOptions> offset) {
    return {::tflite::BuiltinOptions_NONE, 0, offset};
  }

  // Build builtin options of the given type.
  static Options Builtin(::tflite::BuiltinOptions type,
                         flatbuffers::Offset<BuiltinOptions> offset) {
    return {type, offset, 0};
  }

  ::tflite::BuiltinOptions type;
  flatbuffers::Offset<BuiltinOptions> builtin;
  flatbuffers::Offset<CustomOptions> custom;
};

// A BaseOperator encapsulates the relationship between operators in tf.mini
// and TF lite, and provides methods for converting between those two formats.
class BaseOperator {
 public:
  // Build an operator with the given TF Lite name and tf.mini type.
  BaseOperator(const string& name, OperatorType type)
      : name_(name), type_(type) {}
  virtual ~BaseOperator() = default;

  string name() const { return name_; }
  OperatorType type() const { return type_; }

  // Given a tf.mini operator, create the corresponding flatbuffer options and
  // return their offsets.
  virtual Options Serialize(const Operator& op,
                            flatbuffers::FlatBufferBuilder* builder) const = 0;

  // Read TF Lite options and create the appropriate tf.mini operator.
  virtual std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const = 0;

 private:
  string name_;
  OperatorType type_;
};

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TFLITE_OPERATOR_H_
