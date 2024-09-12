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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_OPERATOR_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_OPERATOR_H_

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"
#include "tensorflow/lite/tools/versioning/op_version.h"

namespace toco {

namespace tflite {

class BaseOperator;

// Return a map contained all know TF Lite Operators, keyed by their names.
std::map<std::string, std::unique_ptr<BaseOperator>> BuildOperatorByNameMap(
    bool enable_select_tf_ops = false);

// Return a map contained all know TF Lite Operators, keyed by the type of
// their tf.mini counterparts.
std::map<OperatorType, std::unique_ptr<BaseOperator>> BuildOperatorByTypeMap(
    bool enable_select_tf_ops = false);

// Write the custom option FlexBuffer with a serialized TensorFlow NodeDef
// for a Flex op.
std::unique_ptr<flexbuffers::Builder> WriteFlexOpOptions(
    const std::string& tensorflow_node_def);

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
  BaseOperator(const std::string& name, OperatorType type)
      : name_(name), type_(type) {}
  virtual ~BaseOperator() = default;

  std::string name() const { return name_; }
  OperatorType type() const { return type_; }

  // Given a tf.mini operator, create the corresponding flatbuffer options and
  // return their offsets.
  virtual Options Serialize(const Operator& op,
                            flatbuffers::FlatBufferBuilder* builder) const = 0;

  // Read TF Lite options and create the appropriate tf.mini operator.
  virtual std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const = 0;

  // Get the op version using the OperatorSignature.
  // The function needs to be overridden to return the op version based on the
  // parameters. Note:
  // * The first version for each op should be 1 (to be consistent with the
  //   default value in Flatbuffer. `return 1;` is okay for newly implemented
  //   ops.
  // * When multiple versions are defined for an op, this function could be
  //   overridden. (See example in `operator_test.cc` and
  //   'tools/versioning/op_version.cc`)
  virtual int GetVersion(const OperatorSignature& op_signature) const = 0;

  // Given a Toco `Operator`, return a list of booleans indicating the op
  // mutates which input variables.
  // * If the op mutates any input variables, it should return a list of bool
  //   with the same length as inputs.
  // * Otherwise, it will return an empty list.
  virtual std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const {
    // Most ops don't have variable tensors. This function can be overridden.
    return std::vector<bool>();
  }

 private:
  std::string name_;
  OperatorType type_;
};

// Helper function to create ::tflite::OpSignature from the given
// ::tflite::BuiltinOperator and OperatorSignature.
::tflite::OpSignature GetVersioningOpSig(const ::tflite::BuiltinOperator op,
                                         const OperatorSignature& op_signature);

// Helper function to determine if a unsupported TensorFlow op should be
// exported as an Flex op or a regular custom op.
bool ShouldExportAsFlexOp(bool enable_select_tf_ops,
                          const std::string& tensorflow_op_name);

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_OPERATOR_H_
