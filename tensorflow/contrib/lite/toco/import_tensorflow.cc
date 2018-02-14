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
#include "tensorflow/contrib/lite/toco/import_tensorflow.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/map.h"
#include "google/protobuf/text_format.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/tensorflow_graph_matching/resolve_cluster.h"
#include "tensorflow/contrib/lite/toco/tensorflow_util.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

using tensorflow::AttrValue;
using tensorflow::DT_BOOL;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_QUINT8;
using tensorflow::DT_STRING;
using tensorflow::DT_UINT8;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::TensorProto;
using tensorflow::TensorShapeProto;

namespace toco {
namespace {
bool HasAttr(const NodeDef& node, const string& attr_name) {
  return node.attr().count(attr_name) > 0;
}

const string& GetStringAttr(const NodeDef& node, const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kS);
  return attr.s();
}

int GetIntAttr(const NodeDef& node, const string& attr_name) {
  CHECK(HasAttr(node, attr_name)) << attr_name << " not found in:\n"
                                  << node.DebugString();
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kI);
  return attr.i();
}

float GetFloatAttr(const NodeDef& node, const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kF);
  return attr.f();
}

bool GetBoolAttr(const NodeDef& node, const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kB);
  return attr.b();
}

tensorflow::DataType GetDataTypeAttr(const NodeDef& node,
                                     const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kType);
  return attr.type();
}

const TensorShapeProto& GetShapeAttr(const NodeDef& node,
                                     const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kShape);
  return attr.shape();
}

const TensorProto& GetTensorAttr(const NodeDef& node, const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kTensor);
  return attr.tensor();
}

const AttrValue::ListValue& GetListAttr(const NodeDef& node,
                                        const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kList);
  return attr.list();
}

ArrayDataType ConvertDataType(tensorflow::DataType dtype) {
  if (dtype == DT_UINT8)
    return ArrayDataType::kUint8;
  else if (dtype == DT_FLOAT)
    return ArrayDataType::kFloat;
  else if (dtype == DT_BOOL)
    return ArrayDataType::kBool;
  else if (dtype == DT_INT32)
    return ArrayDataType::kInt32;
  else if (dtype == DT_INT64)
    return ArrayDataType::kInt64;
  else if (dtype == DT_STRING)
    return ArrayDataType::kString;
  else
    LOG(INFO) << "Unsupported data type in placehoder op: " << dtype;
  return ArrayDataType::kNone;
}

void ImportShape(const TFLITE_PROTO_NS::RepeatedPtrField<
                     tensorflow::TensorShapeProto_Dim>& input_dims,
                 Shape* shape) {
  std::vector<int> input_dims_only_sizes;
  for (auto& d : input_dims) {
    if (d.size() == 0) {
      // Some TensorFlow shapes contain a 0 dim, effectively making
      // them of flat size 0 even though they have other nonzero dims.
      // This breaks our invariant, that array dims can't be 0.
      // For now, tweaking this to record a 0-D shape instead.
      input_dims_only_sizes.clear();
      break;
    }
    input_dims_only_sizes.push_back(d.size());
  }
  *shape->mutable_dims() = input_dims_only_sizes;
}

void ImportFloatArray(const TensorProto& input_tensor, Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_FLOAT);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  ImportShape(input_shape.dim(), output_array->mutable_shape());
  int input_flat_size = 1;
  for (int k = 0; k < input_shape.dim_size(); k++) {
    input_flat_size *= input_shape.dim(k).size();
  }
  auto& output_float_data =
      output_array->GetMutableBuffer<ArrayDataType::kFloat>().data;
  output_float_data.resize(RequiredBufferSizeForShape(output_array->shape()),
                           0.f);
  if (input_tensor.float_val_size() == 1) {
    for (int i = 0; i < input_flat_size; i++) {
      output_float_data[i] = input_tensor.float_val(0);
    }
  } else if (input_tensor.float_val_size() == input_flat_size) {
    for (int i = 0; i < input_tensor.float_val_size(); i++) {
      output_float_data[i] = input_tensor.float_val(i);
    }
  } else if (input_tensor.tensor_content().size() ==
             input_flat_size * sizeof(float)) {
    toco::port::CopyToBuffer(input_tensor.tensor_content(),
                             reinterpret_cast<char*>(output_float_data.data()));
  } else {
    LOG(FATAL) << "Neither input_content nor float_val have the right "
                  "dimensions for this float tensor.";
  }
}

void ImportQuint8Array(const TensorProto& input_tensor, Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_QUINT8);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  ImportShape(input_shape.dim(), output_array->mutable_shape());
  int input_flat_size = 1;
  for (int k = 0; k < input_shape.dim_size(); k++) {
    input_flat_size *= input_shape.dim(k).size();
  }
  auto& output_int_data =
      output_array->GetMutableBuffer<ArrayDataType::kUint8>().data;
  output_int_data.resize(RequiredBufferSizeForShape(output_array->shape()), 0);
  if (input_tensor.int_val_size()) {
    for (int i = 0; i < input_tensor.int_val_size(); i++) {
      output_int_data[i] = input_tensor.int_val(i);
    }
  } else if (input_tensor.tensor_content().size() ==
             input_flat_size * sizeof(uint8_t)) {
    toco::port::CopyToBuffer(input_tensor.tensor_content(),
                             reinterpret_cast<char*>(output_int_data.data()));
  } else {
    LOG(FATAL) << "Neither input_content nor int_val have the right "
                  "dimensions for this uint8 tensor.";
  }
}

void ImportInt32Array(const TensorProto& input_tensor, Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_INT32);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  ImportShape(input_shape.dim(), output_array->mutable_shape());
  int input_flat_size = 1;
  for (int k = 0; k < input_shape.dim_size(); k++) {
    input_flat_size *= input_shape.dim(k).size();
  }
  auto& output_int_data =
      output_array->GetMutableBuffer<ArrayDataType::kInt32>().data;
  output_int_data.resize(RequiredBufferSizeForShape(output_array->shape()), 0);
  if (input_tensor.int_val_size()) {
    for (int i = 0; i < input_tensor.int_val_size(); i++) {
      output_int_data[i] = input_tensor.int_val(i);
    }
  } else if (input_tensor.tensor_content().size() ==
             input_flat_size * sizeof(int32)) {
    toco::port::CopyToBuffer(input_tensor.tensor_content(),
                             reinterpret_cast<char*>(output_int_data.data()));
  } else {
    LOG(FATAL) << "Neither input_content nor int_val have the right "
                  "dimensions for this int32 tensor.";
  }
}

void ImportInt64Array(const TensorProto& input_tensor, Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_INT64);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  ImportShape(input_shape.dim(), output_array->mutable_shape());
  int input_flat_size = 1;
  for (int k = 0; k < input_shape.dim_size(); k++) {
    input_flat_size *= input_shape.dim(k).size();
  }
  auto& output_int_data =
      output_array->GetMutableBuffer<ArrayDataType::kInt64>().data;
  output_int_data.resize(RequiredBufferSizeForShape(output_array->shape()), 0);
  if (input_tensor.int64_val_size()) {
    for (int i = 0; i < input_tensor.int64_val_size(); i++) {
      output_int_data[i] = input_tensor.int64_val(i);
    }
  } else if (input_tensor.tensor_content().size() ==
             input_flat_size * sizeof(int64)) {
    toco::port::CopyToBuffer(input_tensor.tensor_content(),
                             reinterpret_cast<char*>(output_int_data.data()));
  } else {
    LOG(FATAL) << "Neither input_content nor int64_val have the right "
                  "dimensions for this int64 tensor.";
  }
}

void ImportStringArray(const TensorProto& input_tensor, Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_STRING);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  ImportShape(input_shape.dim(), output_array->mutable_shape());
  int input_flat_size = 1;
  for (int k = 0; k < input_shape.dim_size(); k++) {
    input_flat_size *= input_shape.dim(k).size();
  }
  auto& output_string_data =
      output_array->GetMutableBuffer<ArrayDataType::kString>().data;
  output_string_data.resize(RequiredBufferSizeForShape(output_array->shape()));
  if (input_flat_size != input_tensor.string_val_size()) {
    LOG(FATAL) << "Input_content string_val doesn't have the right "
                  "dimensions for this string tensor.";
  }
  for (int i = 0; i < input_flat_size; ++i) {
    output_string_data[i] = input_tensor.string_val(i);
  }
}

// Count the number of inputs of a given node. If
// `tf_import_flags.drop_control_dependency` is true, count the number of
// non-control-dependency inputs.
int GetInputsCount(const NodeDef& node,
                   const TensorFlowImportFlags& tf_import_flags) {
  if (tf_import_flags.drop_control_dependency) {
    for (size_t i = 0; i < node.input_size(); ++i) {
      if (node.input(i)[0] == '^') {
        return i;
      }
    }
    return node.input_size();
  } else {
    return node.input_size();
  }
}

void CheckInputsCount(const NodeDef& node,
                      const TensorFlowImportFlags& tf_import_flags,
                      int expected_input_count) {
  QCHECK_EQ(GetInputsCount(node, tf_import_flags), expected_input_count)
      << node.op() << " node expects " << expected_input_count
      << " input(s) other than control dependencies: " << node.DebugString();
}

void ConvertConstOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Const");
  const auto& tensor = GetTensorAttr(node, "value");
  const auto dtype = GetDataTypeAttr(node, "dtype");

  auto& array = model->GetOrCreateArray(node.name());
  switch (dtype) {
    case DT_FLOAT:
      array.data_type = ArrayDataType::kFloat;
      ImportFloatArray(tensor, &array);
      break;
    case DT_INT32:
      array.data_type = ArrayDataType::kInt32;
      ImportInt32Array(tensor, &array);
      break;
    case DT_QUINT8:
      array.data_type = ArrayDataType::kUint8;
      ImportQuint8Array(tensor, &array);
      break;
    case DT_INT64:
      array.data_type = ArrayDataType::kInt64;
      ImportInt64Array(tensor, &array);
      break;
    case DT_STRING:
      array.data_type = ArrayDataType::kString;
      ImportStringArray(tensor, &array);
      break;
    default:
      array.data_type = ArrayDataType::kNone;
      // do nothing, silently ignore the Const data.
      // We just make a dummy buffer to indicate that
      // this array does not rely on external input.
      array.GetMutableBuffer<ArrayDataType::kNone>();
      break;
  }
}

void ConvertConvOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Conv2D");
  CheckInputsCount(node, tf_import_flags, 2);

  // We only support NHWC, which is the default data_format.
  // So if data_format is not defined, we're all good.
  if (node.attr().count("data_format")) {
    CHECK_EQ(GetStringAttr(node, "data_format"), "NHWC");
  }
  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);

  const auto& input_name = node.input(0);
  const auto& weights_name = node.input(1);
  const auto& reordered_weights_name = weights_name + "_reordered";
  // Check if a ReorderAxesOperator was already created for these weights
  // (that happens when multiple layers share the same weights).
  const Operator* existing_reorder =
      GetOpWithOutput(*model, reordered_weights_name);
  if (existing_reorder) {
    // Check that it is safe to rely on the _reordered naming of the output
    // array!
    CHECK(existing_reorder->type == OperatorType::kReorderAxes);
  } else {
    // Create a new ReorderAxesOperator
    auto* reorder = new ReorderAxesOperator;
    reorder->inputs = {weights_name};
    reorder->outputs = {reordered_weights_name};
    reorder->input_axes_order = AxesOrder::kHWIO;
    reorder->output_axes_order = AxesOrder::kOHWI;
    model->operators.emplace_back(reorder);
  }
  auto* conv = new ConvOperator;
  conv->inputs = {input_name, reordered_weights_name};
  conv->outputs = {node.name()};
  const auto& strides = GetListAttr(node, "strides");
  CHECK_EQ(strides.i_size(), 4);
  CHECK_EQ(strides.i(0), 1);
  CHECK_EQ(strides.i(3), 1);
  conv->stride_height = strides.i(1);
  conv->stride_width = strides.i(2);
  const auto& padding = GetStringAttr(node, "padding");
  if (padding == "SAME") {
    conv->padding.type = PaddingType::kSame;
  } else if (padding == "VALID") {
    conv->padding.type = PaddingType::kValid;
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  model->operators.emplace_back(conv);
}

void ConvertDepthwiseConvOperator(const NodeDef& node,
                                  const TensorFlowImportFlags& tf_import_flags,
                                  Model* model) {
  CHECK_EQ(node.op(), "DepthwiseConv2dNative");
  CheckInputsCount(node, tf_import_flags, 2);

  // We only support NHWC, which is the default data_format.
  // So if data_format is not defined, we're all good.
  if (node.attr().count("data_format")) {
    CHECK_EQ(GetStringAttr(node, "data_format"), "NHWC");
  }
  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);

  const auto& input_name = node.input(0);
  const auto& weights_name = node.input(1);
  const auto& reordered_weights_name = weights_name + "_reordered";
  // Check if a ReorderAxesOperator was already created for these weights
  // (that happens when multiple layers share the same weights).
  const Operator* existing_reorder =
      GetOpWithOutput(*model, reordered_weights_name);
  if (existing_reorder) {
    // Check that it is safe to rely on the _reordered naming of the output
    // array!
    CHECK(existing_reorder->type == OperatorType::kReorderAxes);
  } else {
    // Create a new ReorderAxesOperator
    auto* reorder = new ReorderAxesOperator;
    reorder->inputs = {weights_name};
    reorder->outputs = {reordered_weights_name};
    reorder->input_axes_order = AxesOrder::kHWIM;
    reorder->output_axes_order = AxesOrder::k1HWO;
    model->operators.emplace_back(reorder);
  }
  auto* conv = new DepthwiseConvOperator;
  conv->inputs = {input_name, reordered_weights_name};
  conv->outputs = {node.name()};
  const auto& strides = GetListAttr(node, "strides");
  CHECK_EQ(strides.i_size(), 4);
  CHECK_EQ(strides.i(0), 1);
  CHECK_EQ(strides.i(3), 1);
  conv->stride_height = strides.i(1);
  conv->stride_width = strides.i(2);
  const auto& padding = GetStringAttr(node, "padding");
  if (padding == "SAME") {
    conv->padding.type = PaddingType::kSame;
  } else if (padding == "VALID") {
    conv->padding.type = PaddingType::kValid;
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  model->operators.emplace_back(conv);
}

void ConvertDepthToSpaceOperator(const NodeDef& node,
                                 const TensorFlowImportFlags& tf_import_flags,
                                 Model* model) {
  CHECK_EQ(node.op(), "DepthToSpace");
  CheckInputsCount(node, tf_import_flags, 1);

  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);
  auto* op = new DepthToSpaceOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  op->block_size = GetIntAttr(node, "block_size");
  QCHECK_GE(op->block_size, 2);
  model->operators.emplace_back(op);
}

void ConvertSpaceToDepthOperator(const NodeDef& node,
                                 const TensorFlowImportFlags& tf_import_flags,
                                 Model* model) {
  CHECK_EQ(node.op(), "SpaceToDepth");
  CheckInputsCount(node, tf_import_flags, 1);

  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);
  auto* op = new SpaceToDepthOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  op->block_size = GetIntAttr(node, "block_size");
  QCHECK_GE(op->block_size, 2);
  model->operators.emplace_back(op);
}

void ConvertBiasAddOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "BiasAdd");
  CheckInputsCount(node, tf_import_flags, 2);

  const auto& input_name = node.input(0);
  const auto& bias_name = node.input(1);
  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);
  auto* biasadd = new AddOperator;
  biasadd->inputs.push_back(input_name);
  biasadd->inputs.push_back(bias_name);
  biasadd->outputs.push_back(node.name());
  model->operators.emplace_back(biasadd);
}

void ConvertReluOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Relu");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto& input_name = node.input(0);
  auto* relu = new ReluOperator;
  relu->inputs.push_back(input_name);
  relu->outputs.push_back(node.name());
  model->operators.emplace_back(relu);
}

void ConvertRelu6Operator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Relu6");
  CheckInputsCount(node, tf_import_flags, 1);

  const auto& input_name = node.input(0);
  auto* op = new Relu6Operator;
  op->inputs.push_back(input_name);
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertLogisticOperator(const NodeDef& node,
                             const TensorFlowImportFlags& tf_import_flags,
                             Model* model) {
  CHECK_EQ(node.op(), "Sigmoid");
  CheckInputsCount(node, tf_import_flags, 1);

  const auto& input_name = node.input(0);
  auto* op = new LogisticOperator;
  op->inputs.push_back(input_name);
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertTanhOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Tanh");
  CheckInputsCount(node, tf_import_flags, 1);

  const auto& input_name = node.input(0);
  auto* op = new TanhOperator;
  op->inputs.push_back(input_name);
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertDivOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK(node.op() == "Div" || node.op() == "RealDiv");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new DivOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertIdentityOperator(const NodeDef& node,
                             const TensorFlowImportFlags& tf_import_flags,
                             Model* model) {
  CHECK(node.op() == "Identity" || node.op() == "CheckNumerics" ||
        node.op() == "PlaceholderWithDefault" || node.op() == "StopGradient");
  auto* op = new TensorFlowIdentityOperator;
  // Amazingly, some TensorFlow graphs (at least rajeev_lstm.pb) have
  // identity nodes with multiple inputs, but the other inputs seem
  // to be gratuitous (in the case of rajeev_lstm.pb, these are
  // enumerating the LSTM state arrays). We will just ignore extra
  // inputs beyond the first input.
  QCHECK_GE(node.input_size(), 1)
      << node.op()
      << " node expects at least 1 input other than control dependencies: "
      << node.DebugString();
  const auto& input_name = node.input(0);
  op->inputs.push_back(input_name);
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertFakeQuantWithMinMaxArgs(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "FakeQuantWithMinMaxArgs");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new FakeQuantOperator;
  op->inputs.push_back(node.input(0));
  op->minmax.reset(new MinMax);
  auto& minmax = *op->minmax;
  minmax.min = GetFloatAttr(node, "min");
  minmax.max = GetFloatAttr(node, "max");
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertFakeQuantWithMinMaxVars(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "FakeQuantWithMinMaxVars");
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  QCHECK(num_inputs == 3 || num_inputs == 4)
      << "FakeQuantWithMinMaxVars node expects 3 or 4 inputs other than "
         "control dependencies: "
      << node.DebugString();
  auto* op = new FakeQuantOperator;
  for (int i = 0; i < 3; i++) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertNegOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Neg");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new NegOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertRsqrtOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Rsqrt");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new TensorFlowRsqrtOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertSqrtOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Sqrt");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new TensorFlowSqrtOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertSqueezeOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "Squeeze");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new SqueezeOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());

  const auto& squeeze_dims = GetListAttr(node, "squeeze_dims");
  for (int i = 0; i < squeeze_dims.i_size(); ++i) {
    op->squeeze_dims.push_back(squeeze_dims.i(i));
  }

  model->operators.emplace_back(op);
}

void ConvertSquareOperator(const NodeDef& node,
                           const TensorFlowImportFlags& tf_import_flags,
                           Model* model) {
  CHECK_EQ(node.op(), "Square");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new TensorFlowSquareOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertAddOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Add");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new AddOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertAddNOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "AddN");
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  auto* op = new AddNOperator;
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertMulOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Mul");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new MulOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertSubOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Sub");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new SubOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertSumOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Sum");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowSumOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  if (HasAttr(node, "keep_dims")) {
    op->keep_dims = GetBoolAttr(node, "keep_dims");
  }
}

void ConvertTileOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Tile");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowTileOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertSliceOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Slice");
  CheckInputsCount(node, tf_import_flags, 3);
  auto* op = new SliceOperator;
  for (int i = 0; i < 3; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertPadOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Pad");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new PadOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertShapeOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Shape");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new TensorFlowShapeOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertSplitOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Split");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowSplitOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  const int num_split = GetIntAttr(node, "num_split");
  op->outputs.push_back(node.name());
  for (int i = 1; i < num_split; i++) {
    op->outputs.push_back(absl::StrCat(node.name(), ":", i));
  }
  op->num_split = num_split;
  model->operators.emplace_back(op);
}

void ConvertMergeOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Merge");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowMergeOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertSwitchOperator(const NodeDef& node,
                           const TensorFlowImportFlags& tf_import_flags,
                           Model* model) {
  CHECK_EQ(node.op(), "Switch");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowSwitchOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  // Switch operators have two outputs: "name" and "name:1".
  op->outputs.push_back(node.name() + ":1");
  model->operators.emplace_back(op);
}

void ConvertSoftmaxOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "Softmax");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto& input_name = node.input(0);
  auto* softmax = new SoftmaxOperator;
  softmax->inputs.push_back(input_name);
  softmax->outputs.push_back(node.name());
  // TensorFlow's Softmax doesn't seem to admit a 'beta' parameter.
  CHECK(!node.attr().count("beta"));  // Stab in the dark, just in case.
  softmax->beta = 1.f;
  model->operators.emplace_back(softmax);
}

void ConvertLogSoftmaxOperator(const NodeDef& node,
                               const TensorFlowImportFlags& tf_import_flags,
                               Model* model) {
  CHECK_EQ(node.op(), "LogSoftmax");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto& input_name = node.input(0);
  auto* log_softmax = new LogSoftmaxOperator;
  log_softmax->inputs.push_back(input_name);
  log_softmax->outputs.push_back(node.name());
  model->operators.emplace_back(log_softmax);
}

void ConvertLRNOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "LRN");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto& input_name = node.input(0);
  auto* lrn = new LocalResponseNormalizationOperator;
  lrn->inputs.push_back(input_name);
  lrn->outputs.push_back(node.name());
  lrn->range = GetIntAttr(node, "depth_radius");
  lrn->bias = GetFloatAttr(node, "bias");
  lrn->alpha = GetFloatAttr(node, "alpha");
  lrn->beta = GetFloatAttr(node, "beta");
  model->operators.emplace_back(lrn);
}

void ConvertMaxPoolOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "MaxPool");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto& input_name = node.input(0);
  // We only support NHWC, which is the default data_format.
  // So if data_format is not defined, we're all good.
  if (node.attr().count("data_format")) {
    CHECK_EQ(GetStringAttr(node, "data_format"), "NHWC");
  }
  if (HasAttr(node, "T")) {
    CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);
  } else {
    LOG(WARNING) << "Found MaxPool operator missing 'T' attribute";
  }
  auto* maxpool = new MaxPoolOperator;
  maxpool->inputs.push_back(input_name);
  maxpool->outputs.push_back(node.name());
  const auto& strides = GetListAttr(node, "strides");
  CHECK_EQ(strides.i_size(), 4);
  CHECK_EQ(strides.i(0), 1);
  CHECK_EQ(strides.i(3), 1);
  maxpool->stride_height = strides.i(1);
  maxpool->stride_width = strides.i(2);
  const auto& ksize = GetListAttr(node, "ksize");
  CHECK_EQ(ksize.i_size(), 4);
  CHECK_EQ(ksize.i(0), 1);
  CHECK_EQ(ksize.i(3), 1);
  maxpool->kheight = ksize.i(1);
  maxpool->kwidth = ksize.i(2);
  const auto& padding = GetStringAttr(node, "padding");
  if (padding == "SAME") {
    maxpool->padding.type = PaddingType::kSame;
  } else if (padding == "VALID") {
    maxpool->padding.type = PaddingType::kValid;
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  model->operators.emplace_back(maxpool);
}

void ConvertAvgPoolOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "AvgPool");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto& input_name = node.input(0);
  // We only support NHWC, which is the default data_format.
  // So if data_format is not defined, we're all good.
  if (node.attr().count("data_format")) {
    CHECK_EQ(GetStringAttr(node, "data_format"), "NHWC");
  }
  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);
  auto* avgpool = new AveragePoolOperator;
  avgpool->inputs.push_back(input_name);
  avgpool->outputs.push_back(node.name());
  const auto& strides = GetListAttr(node, "strides");
  CHECK_EQ(strides.i_size(), 4);
  CHECK_EQ(strides.i(0), 1);
  CHECK_EQ(strides.i(3), 1);
  avgpool->stride_height = strides.i(1);
  avgpool->stride_width = strides.i(2);
  const auto& ksize = GetListAttr(node, "ksize");
  CHECK_EQ(ksize.i_size(), 4);
  CHECK_EQ(ksize.i(0), 1);
  CHECK_EQ(ksize.i(3), 1);
  avgpool->kheight = ksize.i(1);
  avgpool->kwidth = ksize.i(2);
  const auto& padding = GetStringAttr(node, "padding");
  if (padding == "SAME") {
    avgpool->padding.type = PaddingType::kSame;
  } else if (padding == "VALID") {
    avgpool->padding.type = PaddingType::kValid;
  } else {
    LOG(FATAL) << "Bad padding (only SAME and VALID are supported)";
  }
  model->operators.emplace_back(avgpool);
}

void ConvertReshapeOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "Reshape");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowReshapeOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertBatchMatMulOperator(const NodeDef& node,
                                const TensorFlowImportFlags& tf_import_flags,
                                Model* model) {
  CheckInputsCount(node, tf_import_flags, 2);

  // https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions
  CHECK(!HasAttr(node, "adj_a") || (GetBoolAttr(node, "adj_a") == false));
  CHECK(!HasAttr(node, "adj_b") || (GetBoolAttr(node, "adj_b") == false));

  auto* batch_matmul = new BatchMatMulOperator;
  batch_matmul->inputs = {node.input(0), node.input(1)};
  batch_matmul->outputs = {node.name()};
  model->operators.emplace_back(batch_matmul);
}

void ConvertMatMulOperator(const NodeDef& node,
                           const TensorFlowImportFlags& tf_import_flags,
                           Model* model) {
  CheckInputsCount(node, tf_import_flags, 2);

  // Transpose flags should be easy to support, but we don't have a
  // GraphDef with them to test on at the moment.
  CHECK_EQ(GetBoolAttr(node, "transpose_a"), false);
  CHECK_EQ(GetBoolAttr(node, "transpose_b"), false);
  CHECK(!HasAttr(node, "adjoint_a") ||
        (GetBoolAttr(node, "adjoint_a") == false));
  CHECK(!HasAttr(node, "adjoint_b") ||
        (GetBoolAttr(node, "adjoint_b") == false));

  auto* matmul = new TensorFlowMatMulOperator;
  matmul->inputs = {node.input(0), node.input(1)};
  matmul->outputs = {node.name()};
  model->operators.emplace_back(matmul);
}

void ConvertConcatOperator(const NodeDef& node,
                           const TensorFlowImportFlags& tf_import_flags,
                           Model* model) {
  Operator* op = nullptr;
  if (node.op() == "Concat") {
    op = new TensorFlowConcatOperator;
  } else if (node.op() == "ConcatV2") {
    op = new TensorFlowConcatV2Operator;
  } else {
    LOG(FATAL) << "Expected Concat or ConcatV2";
  }
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  QCHECK_GE(num_inputs, 2)
      << node.op()
      << " node expects at least 2 inputs other than control dependencies: "
      << node.DebugString();
  CHECK_EQ(num_inputs, 1 + GetIntAttr(node, "N"));
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertAllOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "All");
  auto* op = new TensorFlowAllOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertAssertOperator(const NodeDef& node,
                           const TensorFlowImportFlags& tf_import_flags,
                           Model* model) {
  CHECK_EQ(node.op(), "Assert");
  auto* op = new TensorFlowAssertOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertLessOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Less");
  auto* op = new TensorFlowLessOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertLessEqualOperator(const NodeDef& node,
                              const TensorFlowImportFlags& tf_import_flags,
                              Model* model) {
  CHECK_EQ(node.op(), "LessEqual");
  auto* op = new TensorFlowLessEqualOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertGreaterOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "Greater");
  auto* op = new TensorFlowGreaterOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertGreaterEqualOperator(const NodeDef& node,
                                 const TensorFlowImportFlags& tf_import_flags,
                                 Model* model) {
  CHECK_EQ(node.op(), "GreaterEqual");
  auto* op = new TensorFlowGreaterEqualOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertMaxOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Max");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowMaxOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  if (HasAttr(node, "keep_dims")) {
    op->keep_dims = GetBoolAttr(node, "keep_dims");
  }
}

void ConvertMinOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Min");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowMinOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  if (HasAttr(node, "keep_dims")) {
    op->keep_dims = GetBoolAttr(node, "keep_dims");
  }
}

void ConvertMaximumOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "Maximum");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowMaximumOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertMinimumOperator(const NodeDef& node,
                            const TensorFlowImportFlags& tf_import_flags,
                            Model* model) {
  CHECK_EQ(node.op(), "Minimum");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TensorFlowMinimumOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertUnsupportedOperator(const NodeDef& node,
                                const TensorFlowImportFlags& tf_import_flags,
                                Model* model) {
  LOG(INFO) << "Converting unsupported operation: " << node.op();
  auto* op = new TensorFlowUnsupportedOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  op->tensorflow_op = node.op();
  node.SerializeToString(&op->tensorflow_node_def);
  model->operators.emplace_back(op);
  if (HasAttr(node, "_output_quantized")) {
    op->quantized = GetBoolAttr(node, "_output_quantized");
  }
  if (HasAttr(node, "_output_types")) {
    const auto& output_types = GetListAttr(node, "_output_types");
    for (int i = 0; i < output_types.type_size(); ++i) {
      op->output_data_types.push_back(ConvertDataType(output_types.type(i)));
    }
  }
}

void ConvertStridedSliceOperator(const NodeDef& node,
                                 const TensorFlowImportFlags& tf_import_flags,
                                 Model* model) {
  CHECK_EQ(node.op(), "StridedSlice");
  // TODO(soroosh): The 4th input (strides) should be e optional, to be
  // consistent with TF.
  CheckInputsCount(node, tf_import_flags, 4);

  auto* op = new StridedSliceOperator;
  for (const auto& input : node.input()) {
    op->inputs.push_back(input);
  }
  op->outputs.push_back(node.name());

  op->begin_mask = GetIntAttr(node, "begin_mask");
  op->ellipsis_mask = GetIntAttr(node, "ellipsis_mask");
  op->end_mask = GetIntAttr(node, "end_mask");
  op->new_axis_mask = GetIntAttr(node, "new_axis_mask");
  op->shrink_axis_mask = GetIntAttr(node, "shrink_axis_mask");
  model->operators.emplace_back(op);
}

void ConvertPlaceholderOperator(const NodeDef& node,
                                const TensorFlowImportFlags& tf_import_flags,
                                Model* model) {
  CHECK(node.op() == "Placeholder" || node.op() == "LegacyFedInput");
  if (node.op() == "Placeholder") {
    CheckInputsCount(node, tf_import_flags, 0);
  }
  auto& array = model->GetOrCreateArray(node.name());
  if (node.attr().count("dtype")) {
    array.data_type = ConvertDataType(GetDataTypeAttr(node, "dtype"));
  }
  if (node.attr().count("shape")) {
    const auto& shape = GetShapeAttr(node, "shape");
    auto num_dims = shape.dim_size();
    bool has_wildcard = false;
    for (std::size_t i = 0; i < num_dims; i++) {
      if (shape.dim(i).size() == -1) {
        has_wildcard = true;
      }
    }
    // TODO(b/62716978): This logic needs to be revisted.  During dims
    // refactoring it is an interim fix.
    if (num_dims > 0 && !has_wildcard) {
      auto& dst_array_dims = *array.mutable_shape()->mutable_dims();
      dst_array_dims.resize(num_dims);
      for (std::size_t i = 0; i < num_dims; i++) {
        dst_array_dims[i] = shape.dim(i).size();
      }
    }
  }
}

void ConvertNoOpOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {}

void ConvertCastOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Cast");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto tf_src_dtype = GetDataTypeAttr(node, "SrcT");
  const auto tf_dst_dtype = GetDataTypeAttr(node, "DstT");
  auto* op = new CastOperator;
  op->src_data_type = ConvertDataType(tf_src_dtype);
  op->dst_data_type = ConvertDataType(tf_dst_dtype);
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertFloorOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Floor");
  CheckInputsCount(node, tf_import_flags, 1);
  const auto data_type = GetDataTypeAttr(node, "T");
  CHECK(data_type == DT_FLOAT);
  auto* op = new FloorOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertGatherOperator(const NodeDef& node,
                           const TensorFlowImportFlags& tf_import_flags,
                           Model* model) {
  CHECK_EQ(node.op(), "Gather");
  CheckInputsCount(node, tf_import_flags, 2);
  const auto indices_data_type = GetDataTypeAttr(node, "Tindices");
  CHECK(indices_data_type == DT_INT32 || indices_data_type == DT_INT64);
  auto* op = new GatherOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertArgMaxOperator(const NodeDef& node,
                           const TensorFlowImportFlags& tf_import_flags,
                           Model* model) {
  CHECK_EQ(node.op(), "ArgMax");
  CheckInputsCount(node, tf_import_flags, 2);
  const auto axis_data_type = GetDataTypeAttr(node, "Tidx");
  const auto output_type = GetDataTypeAttr(node, "output_type");
  CHECK(axis_data_type == DT_INT64 || axis_data_type == DT_INT32);
  CHECK(output_type == DT_INT64 || output_type == DT_INT32);
  auto* op = new ArgMaxOperator;
  op->output_data_type = ConvertDataType(output_type);
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertResizeBilinearOperator(const NodeDef& node,
                                   const TensorFlowImportFlags& tf_import_flags,
                                   Model* model) {
  CHECK_EQ(node.op(), "ResizeBilinear");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new ResizeBilinearOperator;

  op->align_corners = false;
  if (HasAttr(node, "align_corners")) {
    op->align_corners = GetBoolAttr(node, "align_corners");
  }

  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertBatchNormWithGlobalNormalizationOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "BatchNormWithGlobalNormalization");
  CheckInputsCount(node, tf_import_flags, 5);

  // TODO(ahentz): to really match tensorflow we need to add variance_epsilon
  // to the input, before feeding it into TensorFlowRsqrtOperator.
  // CHECK_EQ(GetFloatAttr(node, "variance_epsilon"), 0.001f);

  string multiplier = node.name() + "_mul";
  if (GetBoolAttr(node, "scale_after_normalization")) {
    // Create graph:
    //   v -> RSQRT ->
    //                 MUL  -> multiplier
    //   gamma  ----->
    string rsqrt = node.name() + "_rsqrt";

    auto* rsqrt_op = new TensorFlowRsqrtOperator;
    rsqrt_op->inputs.push_back(node.input(2));
    rsqrt_op->outputs.push_back(rsqrt);
    model->operators.emplace_back(rsqrt_op);

    auto* mul_op = new MulOperator;
    mul_op->inputs.push_back(rsqrt);
    mul_op->inputs.push_back(node.input(4));
    mul_op->outputs.push_back(multiplier);
    model->operators.emplace_back(mul_op);
  } else {
    // Create graph:
    //   v -> RSQRT -> multiplier
    auto* rsqrt_op = new TensorFlowRsqrtOperator;
    rsqrt_op->inputs.push_back(node.input(2));
    rsqrt_op->outputs.push_back(multiplier);
    model->operators.emplace_back(rsqrt_op);
  }

  auto* op = new BatchNormalizationOperator;
  op->global_normalization = true;

  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(multiplier);
  op->inputs.push_back(node.input(3));
  op->outputs.push_back(node.name());

  model->operators.emplace_back(op);
}

void ConvertFusedBatchNormOperator(const NodeDef& node,
                                   const TensorFlowImportFlags& tf_import_flags,
                                   Model* model) {
  CHECK_EQ(node.op(), "FusedBatchNorm");
  CheckInputsCount(node, tf_import_flags, 5);

  // Declare shortcuts for the inputs.
  const string& gamma_input = node.input(1);
  const string& beta_input = node.input(2);
  const string& moving_mean_input = node.input(3);
  const string& moving_variance_input = node.input(4);

  // Create an array holding the epsilon value (typically, 0.001).
  const string epsilon_array_name = node.name() + "_epsilon_array";
  auto& epsilon_array = model->GetOrCreateArray(epsilon_array_name);
  epsilon_array.data_type = ArrayDataType::kFloat;
  *epsilon_array.mutable_shape()->mutable_dims() = {1};
  epsilon_array.GetMutableBuffer<ArrayDataType::kFloat>().data.push_back(
      GetFloatAttr(node, "epsilon"));

  // Add epsilon to the moving variance.
  const string epsilon_add_op_name = node.name() + "_epsilon";
  auto* epsilon_add_op = new AddOperator;
  epsilon_add_op->inputs.push_back(moving_variance_input);
  epsilon_add_op->inputs.push_back(epsilon_array_name);
  epsilon_add_op->outputs.push_back(epsilon_add_op_name);
  model->operators.emplace_back(epsilon_add_op);

  // Take the inverse square root of the (variance + epsilon).
  const string rsqrt_op_name = node.name() + "_rsqrt";
  auto* rsqrt_op = new TensorFlowRsqrtOperator;
  rsqrt_op->inputs.push_back(epsilon_add_op_name);
  rsqrt_op->outputs.push_back(rsqrt_op_name);
  model->operators.emplace_back(rsqrt_op);

  // Multiply the result by gamma.
  const string multiplier = node.name() + "_mul";
  auto* mul_op = new MulOperator;
  mul_op->inputs.push_back(rsqrt_op_name);
  mul_op->inputs.push_back(gamma_input);
  mul_op->outputs.push_back(multiplier);
  model->operators.emplace_back(mul_op);

  // Now we have all required inputs for the BatchNormalizationOperator.
  auto* op = new BatchNormalizationOperator;
  op->global_normalization = true;

  op->inputs.push_back(node.input(0));
  op->inputs.push_back(moving_mean_input);
  op->inputs.push_back(multiplier);
  op->inputs.push_back(beta_input);
  op->outputs.push_back(node.name());

  model->operators.emplace_back(op);
}

void ConvertSpaceToBatchNDOperator(const NodeDef& node,
                                   const TensorFlowImportFlags& tf_import_flags,
                                   Model* model) {
  CHECK_EQ(node.op(), "SpaceToBatchND");
  CheckInputsCount(node, tf_import_flags, 3);
  CHECK_EQ(GetDataTypeAttr(node, "Tblock_shape"), DT_INT32);
  CHECK_EQ(GetDataTypeAttr(node, "Tpaddings"), DT_INT32);
  auto* op = new SpaceToBatchNDOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(2));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertBatchToSpaceNDOperator(const NodeDef& node,
                                   const TensorFlowImportFlags& tf_import_flags,
                                   Model* model) {
  CHECK_EQ(node.op(), "BatchToSpaceND");
  CheckInputsCount(node, tf_import_flags, 3);
  CHECK_EQ(GetDataTypeAttr(node, "Tblock_shape"), DT_INT32);
  CHECK_EQ(GetDataTypeAttr(node, "Tcrops"), DT_INT32);
  auto* op = new BatchToSpaceNDOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(2));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertExpOperator(const NodeDef& node,
                        const TensorFlowImportFlags& tf_import_flags,
                        Model* model) {
  CHECK_EQ(node.op(), "Exp");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new ExpOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertMeanOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Mean");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new MeanOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  if (HasAttr(node, "keep_dims")) {
    op->keep_dims = GetBoolAttr(node, "keep_dims");
  }
}

void ConvertSvdfOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Svdf");
  const int input_size = GetInputsCount(node, tf_import_flags);
  QCHECK(input_size == 3 || input_size == 4)
      << "Svdf node expects 3 or 4 inputs other than control dependencies: "
      << node.DebugString();
  bool has_bias = (input_size == 4);
  auto* op = new SvdfOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(2));
  if (has_bias) {
    op->inputs.push_back(node.input(3));
  }
  op->outputs.push_back(node.name() + "_state");
  op->outputs.push_back(node.name());
  if (node.attr().at("ActivationFunction").s() == "Relu") {
    op->fused_activation_function = FusedActivationFunctionType::kRelu;
  } else {
    op->fused_activation_function = FusedActivationFunctionType::kNone;
  }
  op->rank = node.attr().at("Rank").i();
  model->operators.emplace_back(op);
}

// This is just bare bones support to get the shapes to propagate.
void ConvertTransposeConvOperator(const NodeDef& node,
                                  const TensorFlowImportFlags& tf_import_flags,
                                  Model* model) {
  CHECK_EQ(node.op(), "Conv2DBackpropInput");
  CheckInputsCount(node, tf_import_flags, 3);
  auto* op = new TransposeConvOperator;
  op->inputs.push_back(node.input(2));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  const auto& strides = GetListAttr(node, "strides");
  CHECK_EQ(strides.i_size(), 4);
  CHECK_EQ(strides.i(0), 1);
  op->stride_height = strides.i(1);
  op->stride_width = strides.i(2);
  CHECK_EQ(strides.i(3), 1);
  auto const& padding = GetStringAttr(node, "padding");
  if (padding == "SAME") {
    op->padding.type = PaddingType::kSame;
  } else if (padding == "VALID") {
    op->padding.type = PaddingType::kValid;
  } else {
    LOG(FATAL) << "Only SAME and VALID padding supported on "
                  "Conv2DBackpropInput nodes.";
  }
  model->operators.emplace_back(op);
}

void ConvertExpandDimsOperator(const NodeDef& node,
                               const TensorFlowImportFlags& tf_import_flags,
                               Model* model) {
  CHECK_EQ(node.op(), "ExpandDims");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new ExpandDimsOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertFillOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Fill");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new FillOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertFloorDivOperator(const NodeDef& node,
                             const TensorFlowImportFlags& tf_import_flags,
                             Model* model) {
  CHECK_EQ(node.op(), "FloorDiv");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new FloorDivOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertFloorModOperator(const NodeDef& node,
                             const TensorFlowImportFlags& tf_import_flags,
                             Model* model) {
  CHECK(node.op() == "FloorMod");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new FloorModOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertRangeOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK_EQ(node.op(), "Range");
  CheckInputsCount(node, tf_import_flags, 3);
  auto* op = new RangeOperator;
  if (HasAttr(node, "Tidx")) {
    const auto dtype = toco::GetDataTypeAttr(node, "Tidx");
    CHECK(dtype == DT_UINT8 || dtype == DT_INT32 || dtype == DT_INT64 ||
          dtype == DT_FLOAT);
    op->dtype = ConvertDataType(dtype);
  }
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(2));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertRankOperator(const NodeDef& node,
                         const TensorFlowImportFlags& tf_import_flags,
                         Model* model) {
  CHECK_EQ(node.op(), "Rank");
  CheckInputsCount(node, tf_import_flags, 1);
  auto* op = new RankOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertStackOperator(const NodeDef& node,
                          const TensorFlowImportFlags& tf_import_flags,
                          Model* model) {
  CHECK((node.op() == "Stack") || (node.op() == "Pack"));
  auto* op = new StackOperator;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  QCHECK_GE(num_inputs, 1)
      << node.op()
      << " node expects at least 1 input other than control dependencies: "
      << node.DebugString();
  CHECK_EQ(num_inputs, GetIntAttr(node, "N"));
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  // Both "Stack" and "Pack" have the "axis" attribute.
  op->axis = GetIntAttr(node, "axis");
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

void ConvertTransposeOperator(const NodeDef& node,
                              const TensorFlowImportFlags& tf_import_flags,
                              Model* model) {
  CHECK_EQ(node.op(), "Transpose");
  CheckInputsCount(node, tf_import_flags, 2);
  auto* op = new TransposeOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
}

// Some TensorFlow ops only occur in graph cycles, representing
// control flow. We do not currently support control flow, so we wouldn't
// be able to fully support such graphs, including performing inference,
// anyway. However, rather than erroring out early on graphs being cyclic,
// it helps to at least support these just enough to allow getting a
// graph visualization. This is not trivial, as we require graphs to be
// acyclic aside from RNN back-edges. The solution is to special-case
// such ops as RNN back-edges, which is technically incorrect (does not
// allow representing the op's semantics) but good enough to get a
// graph visualization.
void ConvertOperatorSpecialCasedAsRNNBackEdge(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  // At the moment, the only type of operator special-cased in this way is
  // NextIteration, occurring only in control-flow cycles.
  CHECK_EQ(node.op(), "NextIteration");
  CHECK_EQ(node.input_size(), 1);
  auto* rnn_state = model->flags.add_rnn_states();
  // This RNN state is not explicitly created by the user, so it's
  // OK for some later graph transformation to discard it.
  rnn_state->set_discardable(true);
  rnn_state->set_state_array(node.name());
  rnn_state->set_back_edge_source_array(node.input(0));
}

void StripCaretFromArrayNames(Model* model) {
  for (auto& op : model->operators) {
    for (auto& input : op->inputs) {
      input = string(absl::StripPrefix(input, "^"));
    }
    for (auto& output : op->outputs) {
      output = string(absl::StripPrefix(output, "^"));
    }
  }
  for (auto& array : model->GetArrayMap()) {
    if (absl::StartsWith(array.first, "^")) {
      LOG(FATAL) << "What?";
    }
  }
}

void StripZeroOutputIndexFromInputs(NodeDef* node) {
  for (auto& input : *node->mutable_input()) {
    input = string(absl::StripSuffix(input, ":0"));
  }
}

// In TensorFlow GraphDef, when a node has multiple outputs, they are named
// name:0, name:1, ...
// where 'name' is the node's name(). Just 'name' is an equivalent shorthand
// form for name:0.
// A TensorFlow GraphDef does not explicitly list all the outputs of each node
// (unlike inputs), it being implied by the node's name and operator type
// (the latter implies the number of outputs).
// This makes it non-trivial for us to reconstruct the list of all arrays
// present in the graph and, for each operator, the list of its outputs.
// We do that by taking advantage of the fact that
// at least each node lists explicitly its inputs, so after we've loaded
// all nodes, we can use that information.
void AddExtraOutputs(Model* model) {
  // Construct the list of all arrays consumed by anything in the graph.
  std::vector<string> consumed_arrays;
  // Add arrays consumed by an op.
  for (const auto& consumer_op : model->operators) {
    for (const string& input : consumer_op->inputs) {
      consumed_arrays.push_back(input);
    }
  }
  // Add global outputs of the model.
  for (const string& output_array : model->flags.output_arrays()) {
    consumed_arrays.push_back(output_array);
  }
  // Add arrays consumed by a RNN back-edge.
  for (const auto& rnn_state : model->flags.rnn_states()) {
    consumed_arrays.push_back(rnn_state.back_edge_source_array());
  }
  // Now add operator outputs so that all arrays that are consumed,
  // are produced.
  for (const string& consumed_array : consumed_arrays) {
    // Split the consumed array name into the form name:output_index.
    const std::vector<string>& split = absl::StrSplit(consumed_array, ':');
    // If not of the form name:output_index, then this is not an additional
    // output of a node with multiple outputs, so nothing to do here.
    if (split.size() != 2) {
      continue;
    }
    int output_index = 0;
    if (!absl::SimpleAtoi(split[1], &output_index)) {
      continue;
    }
    // Each op is initially recorded as producing at least the array that
    // has its name. We use that to identify the producer node.
    auto* producer_op = GetOpWithOutput(*model, split[0]);
    if (!producer_op) {
      continue;
    }
    // Add extra outputs to that producer node, all the way to the
    // output_index.
    while (producer_op->outputs.size() <= output_index) {
      using toco::port::StringF;
      producer_op->outputs.push_back(
          StringF("%s:%d", split[0], producer_op->outputs.size()));
    }
  }
}

bool InlineAllFunctions(GraphDef* graphdef) {
  if (graphdef->library().function().empty()) {
    VLOG(kLogLevelModelUnchanged) << "No functions to inline.";
    return false;
  }

  // Override "_noinline" attribute on all functions
  GraphDef graphdef_copy(*graphdef);
  for (auto& function :
       (*graphdef_copy.mutable_library()->mutable_function())) {
    auto* attributes = function.mutable_attr();
    if (attributes->count(tensorflow::kNoInlineAttr) != 0) {
      (*attributes)[tensorflow::kNoInlineAttr].set_b(false);
    }
  }

  // Construct minimum resources needed to use ExpandInlineFunctions().
  tensorflow::SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", 1});
  std::vector<tensorflow::Device*> devices;
  TF_CHECK_OK(tensorflow::DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));

  tensorflow::FunctionLibraryDefinition fld(tensorflow::OpRegistry::Global(),
                                            graphdef_copy.library());
  tensorflow::DeviceMgr device_mgr(devices);
  tensorflow::OptimizerOptions o_opts;
  tensorflow::ProcessFunctionLibraryRuntime pflr(
      &device_mgr, tensorflow::Env::Default(), TF_GRAPH_DEF_VERSION, &fld,
      o_opts, nullptr);
  tensorflow::FunctionLibraryRuntime* flr;
  flr = pflr.GetFLR("/job:localhost/replica:0/task:0/cpu:0");

  tensorflow::Graph graph(fld);
  tensorflow::ImportGraphDefOptions gc_opts;
  gc_opts.validate_shape = false;
  const auto& tf_convert_status = tensorflow::ImportGraphDef(
      gc_opts, graphdef_copy, &graph, nullptr, nullptr);
  if (!tf_convert_status.ok()) {
    LOG(ERROR) << "tensorflow::ImportGraphDef failed with status: "
               << tf_convert_status.ToString();
    return false;
  }

  // Iterate over the graph until there are no more nodes to be inlined.
  bool graph_modified = false;
  while (tensorflow::ExpandInlineFunctions(flr, &graph)) {
    graph_modified = true;
  }

  // Output inlined graph
  if (graph_modified) {
    LOG(INFO) << "Found and inlined TensorFlow functions.";
    graph.ToGraphDef(graphdef);
  }
  return graph_modified;
}
}  // namespace

std::unique_ptr<Model> ImportTensorFlowGraphDef(
    const ModelFlags& model_flags, const TensorFlowImportFlags& tf_import_flags,
    const GraphDef& tf_graph) {
  LogDumpGraphDef(kLogLevelModelChanged, "AT IMPORT", tf_graph);

  GraphDef inlined_graph(tf_graph);
  if (InlineAllFunctions(&inlined_graph)) {
    LogDumpGraphDef(kLogLevelModelChanged, "AFTER INLINING", inlined_graph);
  }

  // Check input and output specification.
  for (const auto& specified_input_array : model_flags.input_arrays()) {
    CHECK(!absl::EndsWith(specified_input_array.name(), ":0"))
        << "Unsupported explicit zero output index: "
        << specified_input_array.name();
  }
  for (const string& specified_output_array : model_flags.output_arrays()) {
    CHECK(!absl::EndsWith(specified_output_array, ":0"))
        << "Unsupported explicit zero output index: " << specified_output_array;
  }

  Model* model = new Model;

  for (auto node : inlined_graph.node()) {
    StripZeroOutputIndexFromInputs(&node);
    if (node.op() == "Const") {
      ConvertConstOperator(node, tf_import_flags, model);
    } else if (node.op() == "Conv2D") {
      ConvertConvOperator(node, tf_import_flags, model);
    } else if (node.op() == "Conv2DBackpropInput") {
      ConvertTransposeConvOperator(node, tf_import_flags, model);
    } else if (node.op() == "DepthwiseConv2dNative") {
      ConvertDepthwiseConvOperator(node, tf_import_flags, model);
    } else if (node.op() == "DepthToSpace") {
      ConvertDepthToSpaceOperator(node, tf_import_flags, model);
    } else if (node.op() == "SpaceToDepth") {
      ConvertSpaceToDepthOperator(node, tf_import_flags, model);
    } else if (node.op() == "BiasAdd") {
      ConvertBiasAddOperator(node, tf_import_flags, model);
    } else if (node.op() == "Relu") {
      ConvertReluOperator(node, tf_import_flags, model);
    } else if (node.op() == "Relu6") {
      ConvertRelu6Operator(node, tf_import_flags, model);
    } else if (node.op() == "Sigmoid") {
      ConvertLogisticOperator(node, tf_import_flags, model);
    } else if (node.op() == "Tanh") {
      ConvertTanhOperator(node, tf_import_flags, model);
    } else if (node.op() == "MaxPool") {
      ConvertMaxPoolOperator(node, tf_import_flags, model);
    } else if (node.op() == "AvgPool") {
      ConvertAvgPoolOperator(node, tf_import_flags, model);
    } else if (node.op() == "Reshape") {
      ConvertReshapeOperator(node, tf_import_flags, model);
    } else if (node.op() == "BatchMatMul") {
      ConvertBatchMatMulOperator(node, tf_import_flags, model);
    } else if (node.op() == "MatMul") {
      ConvertMatMulOperator(node, tf_import_flags, model);
    } else if (node.op() == "Div" || node.op() == "RealDiv") {
      ConvertDivOperator(node, tf_import_flags, model);
    } else if (node.op() == "Identity" || node.op() == "CheckNumerics" ||
               node.op() == "StopGradient") {
      ConvertIdentityOperator(node, tf_import_flags, model);
    } else if (node.op() == "FakeQuantWithMinMaxVars") {
      ConvertFakeQuantWithMinMaxVars(node, tf_import_flags, model);
    } else if (node.op() == "FakeQuantWithMinMaxArgs") {
      ConvertFakeQuantWithMinMaxArgs(node, tf_import_flags, model);
    } else if (node.op() == "Neg") {
      ConvertNegOperator(node, tf_import_flags, model);
    } else if (node.op() == "Rsqrt") {
      ConvertRsqrtOperator(node, tf_import_flags, model);
    } else if (node.op() == "Squeeze") {
      ConvertSqueezeOperator(node, tf_import_flags, model);
    } else if (node.op() == "Sqrt") {
      ConvertSqrtOperator(node, tf_import_flags, model);
    } else if (node.op() == "Square") {
      ConvertSquareOperator(node, tf_import_flags, model);
    } else if (node.op() == "Add") {
      ConvertAddOperator(node, tf_import_flags, model);
    } else if (node.op() == "AddN") {
      ConvertAddNOperator(node, tf_import_flags, model);
    } else if (node.op() == "Mul") {
      ConvertMulOperator(node, tf_import_flags, model);
    } else if (node.op() == "Sub") {
      ConvertSubOperator(node, tf_import_flags, model);
    } else if (node.op() == "Sum") {
      ConvertSumOperator(node, tf_import_flags, model);
    } else if (node.op() == "Tile") {
      ConvertTileOperator(node, tf_import_flags, model);
    } else if (node.op() == "Concat" || node.op() == "ConcatV2") {
      ConvertConcatOperator(node, tf_import_flags, model);
    } else if (node.op() == "LRN") {
      ConvertLRNOperator(node, tf_import_flags, model);
    } else if (node.op() == "Softmax") {
      ConvertSoftmaxOperator(node, tf_import_flags, model);
    } else if (node.op() == "LogSoftmax") {
      ConvertLogSoftmaxOperator(node, tf_import_flags, model);
    } else if (node.op() == "All") {
      ConvertAllOperator(node, tf_import_flags, model);
    } else if (node.op() == "Assert") {
      ConvertAssertOperator(node, tf_import_flags, model);
    } else if (node.op() == "Less") {
      ConvertLessOperator(node, tf_import_flags, model);
    } else if (node.op() == "LessEqual") {
      ConvertLessEqualOperator(node, tf_import_flags, model);
    } else if (node.op() == "Greater") {
      ConvertGreaterOperator(node, tf_import_flags, model);
    } else if (node.op() == "GreaterEqual") {
      ConvertGreaterEqualOperator(node, tf_import_flags, model);
    } else if (node.op() == "Max") {
      ConvertMaxOperator(node, tf_import_flags, model);
    } else if (node.op() == "Min") {
      ConvertMinOperator(node, tf_import_flags, model);
    } else if (node.op() == "Maximum") {
      ConvertMaximumOperator(node, tf_import_flags, model);
    } else if (node.op() == "Minimum") {
      ConvertMinimumOperator(node, tf_import_flags, model);
    } else if (node.op() == "Merge") {
      ConvertMergeOperator(node, tf_import_flags, model);
    } else if (node.op() == "Pad") {
      ConvertPadOperator(node, tf_import_flags, model);
    } else if (node.op() == "StridedSlice") {
      ConvertStridedSliceOperator(node, tf_import_flags, model);
    } else if (node.op() == "Shape") {
      ConvertShapeOperator(node, tf_import_flags, model);
    } else if (node.op() == "Slice") {
      ConvertSliceOperator(node, tf_import_flags, model);
    } else if (node.op() == "Split") {
      ConvertSplitOperator(node, tf_import_flags, model);
    } else if (node.op() == "Switch") {
      ConvertSwitchOperator(node, tf_import_flags, model);
    } else if (node.op() == "Placeholder") {
      ConvertPlaceholderOperator(node, tf_import_flags, model);
    } else if (node.op() == "PlaceholderWithDefault") {
      ConvertIdentityOperator(node, tf_import_flags, model);
    } else if (node.op() == "LegacyFedInput") {
      ConvertPlaceholderOperator(node, tf_import_flags, model);
    } else if (node.op() == "NoOp") {
      ConvertNoOpOperator(node, tf_import_flags, model);
    } else if (node.op() == "Cast") {
      ConvertCastOperator(node, tf_import_flags, model);
    } else if (node.op() == "Floor") {
      ConvertFloorOperator(node, tf_import_flags, model);
    } else if (node.op() == "Gather") {
      ConvertGatherOperator(node, tf_import_flags, model);
    } else if (node.op() == "ResizeBilinear") {
      ConvertResizeBilinearOperator(node, tf_import_flags, model);
    } else if (node.op() == "BatchNormWithGlobalNormalization") {
      ConvertBatchNormWithGlobalNormalizationOperator(node, tf_import_flags,
                                                      model);
    } else if (node.op() == "FusedBatchNorm") {
      ConvertFusedBatchNormOperator(node, tf_import_flags, model);
    } else if (node.op() == "SpaceToBatchND") {
      ConvertSpaceToBatchNDOperator(node, tf_import_flags, model);
    } else if (node.op() == "BatchToSpaceND") {
      ConvertBatchToSpaceNDOperator(node, tf_import_flags, model);
    } else if (node.op() == "Mean") {
      ConvertMeanOperator(node, tf_import_flags, model);
    } else if (node.op() == "Svdf") {
      ConvertSvdfOperator(node, tf_import_flags, model);
    } else if (node.op() == "NextIteration") {
      ConvertOperatorSpecialCasedAsRNNBackEdge(node, tf_import_flags, model);
    } else if (node.op() == "ExpandDims") {
      ConvertExpandDimsOperator(node, tf_import_flags, model);
    } else if (node.op() == "Fill") {
      ConvertFillOperator(node, tf_import_flags, model);
    } else if (node.op() == "FloorDiv") {
      ConvertFloorDivOperator(node, tf_import_flags, model);
    } else if (node.op() == "FloorMod") {
      ConvertFloorModOperator(node, tf_import_flags, model);
    } else if (node.op() == "Range") {
      ConvertRangeOperator(node, tf_import_flags, model);
    } else if (node.op() == "Rank") {
      ConvertRankOperator(node, tf_import_flags, model);
    } else if (node.op() == "Stack" || node.op() == "Pack") {
      ConvertStackOperator(node, tf_import_flags, model);
    } else if (node.op() == "Transpose") {
      ConvertTransposeOperator(node, tf_import_flags, model);
    } else if (node.op() == "ArgMax") {
      ConvertArgMaxOperator(node, tf_import_flags, model);
    } else if (node.op() == "Exp") {
      ConvertExpOperator(node, tf_import_flags, model);
    } else {
      ConvertUnsupportedOperator(node, tf_import_flags, model);
    }
  }

  ResolveModelFlags(model_flags, model);

  StripCaretFromArrayNames(model);
  AddExtraOutputs(model);
  FixNoMissingArray(model);
  FixNoOrphanedArray(model);
  FixOperatorOrdering(model);
  CheckInvariants(*model);

  // if rnn state arrays are constant, make them transient
  for (const auto& rnn_state : model->flags.rnn_states()) {
    model->GetArray(rnn_state.state_array()).buffer = nullptr;
  }

  return std::unique_ptr<Model>(model);
}

std::unique_ptr<Model> ImportTensorFlowGraphDef(
    const ModelFlags& model_flags, const TensorFlowImportFlags& tf_import_flags,
    const string& input_file_contents) {
  std::unique_ptr<GraphDef> tf_graph(new GraphDef);
  CHECK(ParseFromStringEitherTextOrBinary(input_file_contents, tf_graph.get()));

  std::unique_ptr<GraphDef> pruned_graph =
      MaybeReplaceCompositeSubgraph(*tf_graph);
  if (pruned_graph) {
    tf_graph = std::move(pruned_graph);
  }
  return ImportTensorFlowGraphDef(model_flags, tf_import_flags, *tf_graph);
}
}  // namespace toco
