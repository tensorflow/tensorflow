/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/logging/conversion_log_util.h"

#ifdef __linux__
#include <sys/utsname.h>
#endif

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tflite/export.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/lite/version.h"

namespace toco {

namespace {

string TryGetOperatorName(const Operator& op) {
  string op_name;
  if (!op.tensorflow_node_def.empty()) {
    // Parse op name from serialized NodeDef.
    tensorflow::NodeDef node_def;
    if (!node_def.ParseFromString(op.tensorflow_node_def)) {
      LOG(ERROR) << "Failed to parse Tensorflow NodeDef";
    } else {
      op_name = node_def.op();
      if (!op_name.empty()) return op_name;
    }
  }
  if (op.type == OperatorType::kUnsupported) {
    // If we failed to get op name from serialized NodeDef (either because
    // the tensorflow_node_def is an empty string, or we failed to parse
    // from it), fall back to use 'tensorflow_op' field if this op is a
    // TensorflowUnsupportedOperator.
    const TensorFlowUnsupportedOperator& unsupported_op =
        static_cast<const TensorFlowUnsupportedOperator&>(op);
    if (!unsupported_op.tensorflow_op.empty()) {
      op_name = unsupported_op.tensorflow_op;
      return op_name;
    }
  }
  // If this is a built-in op.
  op_name = OperatorTypeName(op.type);
  return op_name;
}

string GetOSVersion() {
  string os_info;
#ifdef __linux__
  utsname info;
  if (uname(&info)) {
    // Failed
    LOG(ERROR) << "Cannot get OS info.";
    return "";
  }
  os_info = string(info.sysname) + ";OSVer=" + string(info.release) + ";";
#endif
  return os_info;
}

string ShapeToStringNoSpace(const Shape& shape) {
  if (shape.dimensions_count() == 0) {
    return "[]";
  }

  return absl::StrCat("[", absl::StrJoin(shape.dims(), ","), "]");
}

string GetOperatorSignature(
    const Model& model, const Operator& op,
    const std::map<OperatorType, std::unique_ptr<tflite::BaseOperator>>&
        op_types_map) {
  // The signature of an op has the following schema:
  // INPUT:SHAPE::TYPE::OUTPUT:SHAPE::TYPE::NAME:VERSION:
  string op_signature;
  constexpr char delimiter[] = "::";

  // Get input shapes and types.
  op_signature.append("INPUT:");
  for (const auto& input : op.inputs) {
    const auto& array = model.GetArray(input);
    if (array.has_shape()) {
      op_signature.append(ShapeToStringNoSpace(array.shape()));
    } else {
      op_signature.append("None");
    }
    op_signature.append(delimiter);
    op_signature.append(ArrayDataTypeName(array.data_type) + delimiter);
  }
  // Get output shapes and types.
  op_signature.append("OUTPUT:");
  for (const auto& output : op.outputs) {
    const auto& array = model.GetArray(output);
    if (array.has_shape()) {
      op_signature.append(ShapeToStringNoSpace(array.shape()));
    } else {
      op_signature.append("None");
    }
    op_signature.append(delimiter);
    op_signature.append(ArrayDataTypeName(array.data_type) + delimiter);
  }
  // Append Op name.
  op_signature.append("NAME:");
  op_signature.append(TryGetOperatorName(op) + delimiter);
  // Append Op version.
  op_signature.append("VERSION:");
  OperatorSignature toco_op_signature;
  toco_op_signature.op = &op;
  toco_op_signature.model = &model;
  if (op_types_map.find(op.type) != op_types_map.end()) {
    const int version = op_types_map.at(op.type)->GetVersion(toco_op_signature);
    op_signature.append(std::to_string(version));
  } else {
    op_signature.append("None");
  }
  return op_signature;
}

}  // namespace

std::vector<string> GetOperatorNames(const Model& model) {
  std::vector<string> op_names;
  for (const auto& op : model.operators) {
    op_names.push_back(TryGetOperatorName(*op));
  }
  return op_names;
}

void CountOperatorsByType(const Model& model,
                          std::map<string, int>* built_in_ops,
                          std::map<string, int>* custom_ops,
                          std::map<string, int>* select_ops) {
  for (const auto& op : model.operators) {
    OperatorSignature op_signature = {op.get(), &model};
    const auto ops_by_type =
        tflite::BuildOperatorByTypeMap(true /*enable_select_tf_ops*/);
    tflite::details::OperatorKey op_key(op_signature, ops_by_type,
                                        true /*enable_select_tf_ops*/);

    const string op_name = TryGetOperatorName(*op);
    if (op_key.is_custom_op()) {
      (*custom_ops)[op_name]++;
    } else if (op_key.is_flex_op()) {
      (*select_ops)[op_name]++;
    } else {
      (*built_in_ops)[op_name]++;
    }
  }
}

void GetInputAndOutputTypes(
    const Model& model, TFLITE_PROTO_NS::RepeatedPtrField<string>* input_types,
    TFLITE_PROTO_NS::RepeatedPtrField<string>* output_types) {
  for (const auto& input_array : model.flags.input_arrays()) {
    const Array& array = model.GetArray(input_array.name());
    input_types->Add(ArrayDataTypeName(array.data_type));
  }
  for (const auto& output_array : model.flags.output_arrays()) {
    const Array& array = model.GetArray(output_array);
    output_types->Add(ArrayDataTypeName(array.data_type));
  }
}

string GetTfLiteVersion() { return TFLITE_VERSION_STRING; }

string GetCachedOSVersion() {
  static string* version = new string(GetOSVersion());
  return *version;
}

void GetOpSignatures(const Model& model,
                     TFLITE_PROTO_NS::RepeatedPtrField<string>* op_signatures) {
  const auto& op_types_map =
      tflite::BuildOperatorByTypeMap(true /*enable_select_tf_ops*/);
  for (const auto& op : model.operators) {
    op_signatures->Add(GetOperatorSignature(model, *op, op_types_map));
  }
}

string GetModelHash(const Model& model) {
  // TODO(b/123519920): Implement the hash function for Model.
  // Need to consider different implementations for public/private models.
  return "";
}

void PopulateConversionLog(const Model& model, TocoConversionLog* log) {
  // Get the list of ops after conversion.
  const std::vector<string> op_names = GetOperatorNames(model);
  for (const auto& op_name : op_names) {
    log->add_op_list(op_name);
  }

  // Get op signatures.
  TFLITE_PROTO_NS::RepeatedPtrField<string> op_signatures;
  GetOpSignatures(model, &op_signatures);
  log->mutable_op_signatures()->CopyFrom(op_signatures);

  // Get op counts by category: custom, built-in or select.
  std::map<string, int> custom_ops, select_ops, built_in_ops;
  CountOperatorsByType(model, &built_in_ops, &custom_ops, &select_ops);
  log->mutable_custom_ops()->insert(custom_ops.cbegin(), custom_ops.cend());
  log->mutable_built_in_ops()->insert(built_in_ops.cbegin(),
                                      built_in_ops.cend());
  log->mutable_select_ops()->insert(select_ops.cbegin(), select_ops.cend());

  // Get the model's input and output types.
  TFLITE_PROTO_NS::RepeatedPtrField<string> input_types, output_types;
  GetInputAndOutputTypes(model, &input_types, &output_types);
  log->mutable_input_tensor_types()->CopyFrom(input_types);
  log->mutable_output_tensor_types()->CopyFrom(output_types);

  log->set_log_generation_ts(absl::ToUnixMicros(absl::Now()));

  log->set_model_size(model.operators.size());
  log->set_tf_lite_version(GetTfLiteVersion());
  log->set_os_version(GetCachedOSVersion());
  log->set_model_hash(GetModelHash(model));
  // TODO(b/123519920): Populate TOCO error logs.
  // Currently we will focus on external installation of TOCO via pip, where
  // the C++ TOCO binary is invoked via subprocess command, this will make our
  // life easier collecting the error logs emitted by TOCO. However, note that
  // if a user directly invokes the C++ TOCO binary, this log might not be
  // available.
}

}  // namespace toco
