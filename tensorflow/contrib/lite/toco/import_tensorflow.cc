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
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/tensorflow_graph_matching/resolve_cluster.h"
#include "tensorflow/contrib/lite/toco/tensorflow_util.h"
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
#include "tensorflow/core/lib/core/status.h"
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

int64 GetIntAttr(const NodeDef& node, const string& attr_name) {
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
  CHECK(HasAttr(node, attr_name)) << "No attr named '" << attr_name << "'";
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

tensorflow::Status CheckOptionalAttr(const NodeDef& node,
                                     const string& attr_name,
                                     const string& expected_value) {
  if (HasAttr(node, attr_name)) {
    const string& value = GetStringAttr(node, attr_name);
    if (value != expected_value) {
      return tensorflow::errors::InvalidArgument(
          "Unexpected value for attribute '" + attr_name + "'. Expected '" +
          expected_value + "'");
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status CheckOptionalAttr(
    const NodeDef& node, const string& attr_name,
    const tensorflow::DataType& expected_value) {
  if (HasAttr(node, attr_name)) {
    const tensorflow::DataType& value = GetDataTypeAttr(node, attr_name);
    if (value != expected_value) {
      return tensorflow::errors::InvalidArgument(
          "Unexpected value for attribute '" + attr_name + "'. Expected '" +
          tensorflow::DataType_Name(expected_value) + "'");
    }
  }
  return tensorflow::Status::OK();
}

template <typename T1, typename T2>
tensorflow::Status ExpectValue(const T1& v1, const T2& v2,
                               const string& description) {
  if (v1 == v2) return tensorflow::Status::OK();
  return tensorflow::errors::InvalidArgument(absl::StrCat(
      "Unexpected ", description, ": got ", v1, ", expected ", v2));
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
    LOG(INFO) << "Unsupported data type in placeholder op: " << dtype;
  return ArrayDataType::kNone;
}

tensorflow::Status ImportShape(
    const TFLITE_PROTO_NS::RepeatedPtrField<tensorflow::TensorShapeProto_Dim>&
        input_dims,
    int* input_flat_size, Shape* shape) {
  std::vector<int> input_dims_only_sizes;
  for (auto& d : input_dims) {
    if (d.size() == 0) {
      // Some TensorFlow shapes contain a 0 dim, effectively making
      // them of flat size 0 even though they have other nonzero dims.
      // This breaks our invariant, that array dims can't be 0.
      // For now, tweaking this to record a 0-D shape instead.
      shape->mutable_dims()->clear();
      if (input_flat_size != nullptr) *input_flat_size = 0;
      return tensorflow::Status::OK();
    }
    // TensorFlow's shapes use int64s, while TOCO uses ints.
    if (d.size() > std::numeric_limits<int>::max()) {
      return tensorflow::errors::InvalidArgument("Shape element overflows");
    }

    input_dims_only_sizes.push_back(d.size());
  }
  *shape->mutable_dims() = input_dims_only_sizes;

  if (input_flat_size == nullptr) return tensorflow::Status::OK();

  return NumElements(input_dims_only_sizes, input_flat_size);
}

tensorflow::Status ImportFloatArray(const TensorProto& input_tensor,
                                    Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_FLOAT);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  int input_flat_size;
  auto status = ImportShape(input_shape.dim(), &input_flat_size,
                            output_array->mutable_shape());
  if (!status.ok()) return status;

  auto& output_float_data =
      output_array->GetMutableBuffer<ArrayDataType::kFloat>().data;
  output_float_data.resize(RequiredBufferSizeForShape(output_array->shape()),
                           0.f);
  CHECK_GE(output_float_data.size(), input_flat_size);
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
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Neither input_content (",
                     input_tensor.tensor_content().size() / sizeof(float),
                     ") nor float_val (", input_tensor.float_val_size(),
                     ") have the right dimensions (", input_flat_size,
                     ") for this float tensor"));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ImportQuint8Array(const TensorProto& input_tensor,
                                     Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_QUINT8);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  int input_flat_size;
  auto status = ImportShape(input_shape.dim(), &input_flat_size,
                            output_array->mutable_shape());
  if (!status.ok()) return status;

  auto& output_int_data =
      output_array->GetMutableBuffer<ArrayDataType::kUint8>().data;
  output_int_data.resize(RequiredBufferSizeForShape(output_array->shape()), 0);
  CHECK_GE(output_int_data.size(), input_flat_size);
  if (input_tensor.int_val_size() == 1) {
    for (int i = 0; i < input_flat_size; i++) {
      output_int_data[i] = input_tensor.int_val(0);
    }
  } else if (input_tensor.int_val_size() == input_flat_size) {
    for (int i = 0; i < input_tensor.int_val_size(); i++) {
      output_int_data[i] = input_tensor.int_val(i);
    }
  } else if (input_tensor.tensor_content().size() ==
             input_flat_size * sizeof(uint8_t)) {
    toco::port::CopyToBuffer(input_tensor.tensor_content(),
                             reinterpret_cast<char*>(output_int_data.data()));
  } else {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Neither input_content (",
                     input_tensor.tensor_content().size() / sizeof(uint8_t),
                     ") nor int_val (", input_tensor.int_val_size(),
                     ") have the right dimensions (", input_flat_size,
                     ") for this uint8 tensor"));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ImportInt32Array(const TensorProto& input_tensor,
                                    Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_INT32);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  int input_flat_size;
  auto status = ImportShape(input_shape.dim(), &input_flat_size,
                            output_array->mutable_shape());
  if (!status.ok()) return status;

  auto& output_int_data =
      output_array->GetMutableBuffer<ArrayDataType::kInt32>().data;
  output_int_data.resize(RequiredBufferSizeForShape(output_array->shape()), 0);
  CHECK_GE(output_int_data.size(), input_flat_size);
  if (input_tensor.int_val_size() == 1) {
    for (int i = 0; i < input_flat_size; i++) {
      output_int_data[i] = input_tensor.int_val(0);
    }
  } else if (input_tensor.int_val_size() == input_flat_size) {
    for (int i = 0; i < input_tensor.int_val_size(); i++) {
      output_int_data[i] = input_tensor.int_val(i);
    }
  } else if (input_tensor.tensor_content().size() ==
             input_flat_size * sizeof(int32)) {
    toco::port::CopyToBuffer(input_tensor.tensor_content(),
                             reinterpret_cast<char*>(output_int_data.data()));
  } else {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Neither input_content (",
        input_tensor.tensor_content().size() / sizeof(int32), ") nor int_val (",
        input_tensor.int_val_size(), ") have the right dimensions (",
        input_flat_size, ") for this int32 tensor"));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ImportInt64Array(const TensorProto& input_tensor,
                                    Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_INT64);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  int input_flat_size;
  auto status = ImportShape(input_shape.dim(), &input_flat_size,
                            output_array->mutable_shape());
  if (!status.ok()) return status;

  auto& output_int_data =
      output_array->GetMutableBuffer<ArrayDataType::kInt64>().data;
  output_int_data.resize(RequiredBufferSizeForShape(output_array->shape()), 0);
  CHECK_GE(output_int_data.size(), input_flat_size);
  if (input_tensor.int64_val_size() == 1) {
    for (int i = 0; i < input_flat_size; i++) {
      output_int_data[i] = input_tensor.int64_val(0);
    }
  } else if (input_tensor.int64_val_size() == input_flat_size) {
    for (int i = 0; i < input_tensor.float_val_size(); i++) {
      output_int_data[i] = input_tensor.int64_val(i);
    }
  } else if (input_tensor.tensor_content().size() ==
             input_flat_size * sizeof(int64)) {
    toco::port::CopyToBuffer(input_tensor.tensor_content(),
                             reinterpret_cast<char*>(output_int_data.data()));
  } else {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Neither input_content (",
                     input_tensor.tensor_content().size() / sizeof(int64),
                     ") nor int64_val (", input_tensor.int64_val_size(),
                     ") have the right dimensions (", input_flat_size,
                     ") for this int64 tensor"));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ImportBoolArray(const TensorProto& input_tensor,
                                   Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_BOOL);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  int input_flat_size;
  auto status = ImportShape(input_shape.dim(), &input_flat_size,
                            output_array->mutable_shape());
  if (!status.ok()) return status;

  auto& output_bool_data =
      output_array->GetMutableBuffer<ArrayDataType::kBool>().data;
  output_bool_data.resize(RequiredBufferSizeForShape(output_array->shape()),
                          false);
  CHECK_GE(output_bool_data.size(), input_flat_size);
  if (input_tensor.bool_val_size() == 1) {
    for (int i = 0; i < input_flat_size; i++) {
      output_bool_data[i] = input_tensor.bool_val(0);
    }
  } else if (input_tensor.bool_val_size() == input_flat_size) {
    for (int i = 0; i < input_tensor.bool_val_size(); i++) {
      output_bool_data[i] = input_tensor.bool_val(i);
    }
  } else if (input_tensor.tensor_content().size() == input_flat_size) {
    std::vector<char> buf(input_tensor.tensor_content().size());
    toco::port::CopyToBuffer(input_tensor.tensor_content(), buf.data());
    for (int i = 0; i < input_tensor.tensor_content().size(); i++) {
      output_bool_data[i] = static_cast<bool>(buf[i]);
    }
  } else {
    // Some graphs have bool const nodes without actual value...
    // assuming that 'false' is implied.
    // So far only encountered that in an array with 1 entry, let's
    // require that until we encounter a graph where that's not the case.
    if (output_bool_data.size() != 1) {
      return tensorflow::errors::InvalidArgument(absl::StrCat(
          "Neither input_content (", input_tensor.tensor_content().size(),
          ") nor bool_val (", input_tensor.bool_val_size(),
          ") have the right dimensions (", input_flat_size,
          ") for this bool tensor"));
    }
    output_bool_data[0] = false;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ImportStringArray(const TensorProto& input_tensor,
                                     Array* output_array) {
  CHECK_EQ(input_tensor.dtype(), DT_STRING);
  const auto& input_shape = input_tensor.tensor_shape();
  CHECK_LE(input_shape.dim_size(), 4);
  int input_flat_size;
  auto status = ImportShape(input_shape.dim(), &input_flat_size,
                            output_array->mutable_shape());
  if (!status.ok()) return status;

  if (input_flat_size != input_tensor.string_val_size()) {
    return tensorflow::errors::InvalidArgument(
        "Input_content string_val doesn't have the right dimensions "
        "for this string tensor");
  }

  auto& output_string_data =
      output_array->GetMutableBuffer<ArrayDataType::kString>().data;
  output_string_data.resize(RequiredBufferSizeForShape(output_array->shape()));
  CHECK_GE(output_string_data.size(), input_flat_size);
  for (int i = 0; i < input_flat_size; ++i) {
    output_string_data[i] = input_tensor.string_val(i);
  }
  return tensorflow::Status::OK();
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
  }
  return node.input_size();
}

tensorflow::Status CheckInputsCount(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    int expected_input_count) {
  if (GetInputsCount(node, tf_import_flags) != expected_input_count) {
    return tensorflow::errors::FailedPrecondition(
        node.op(), " node expects ", expected_input_count,
        " input(s) other than control dependencies: ", node.DebugString());
  }
  return tensorflow::Status::OK();
}

template <ArrayDataType T>
string CreateConstArray(Model* model, string const& name,
                        std::vector<typename toco::DataType<T> > const& data) {
  // Utility function to create a const 1D array, useful for input parameters.
  string array_name = toco::AvailableArrayName(*model, name);
  auto& array = model->GetOrCreateArray(array_name);
  array.data_type = T;
  array.mutable_shape()->mutable_dims()->emplace_back(data.size());
  array.GetMutableBuffer<T>().data = data;
  return array_name;
}

tensorflow::Status ConvertConstOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Const");
  const auto& tensor = GetTensorAttr(node, "value");
  const auto dtype = GetDataTypeAttr(node, "dtype");

  tensorflow::Status status = tensorflow::Status::OK();

  auto& array = model->GetOrCreateArray(node.name());
  switch (dtype) {
    case DT_FLOAT:
      array.data_type = ArrayDataType::kFloat;
      status = ImportFloatArray(tensor, &array);
      break;
    case DT_INT32:
      array.data_type = ArrayDataType::kInt32;
      status = ImportInt32Array(tensor, &array);
      break;
    case DT_QUINT8:
      array.data_type = ArrayDataType::kUint8;
      status = ImportQuint8Array(tensor, &array);
      break;
    case DT_INT64:
      array.data_type = ArrayDataType::kInt64;
      status = ImportInt64Array(tensor, &array);
      break;
    case DT_STRING:
      array.data_type = ArrayDataType::kString;
      status = ImportStringArray(tensor, &array);
      break;
    case DT_BOOL:
      array.data_type = ArrayDataType::kBool;
      status = ImportBoolArray(tensor, &array);
      break;
    default:
      array.data_type = ArrayDataType::kNone;
      // do nothing, silently ignore the Const data.
      // We just make a dummy buffer to indicate that
      // this array does not rely on external input.
      array.GetMutableBuffer<ArrayDataType::kNone>();
      break;
  }
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      status, " (while processing node '" + node.name() + "')");
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConvOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Conv2D");
  TF_RETURN_IF_ERROR(CheckInputsCount(node, tf_import_flags, 2));

  // We only support NHWC, which is the default data_format.
  // So if data_format is not defined, we're all good.
  TF_RETURN_IF_ERROR(CheckOptionalAttr(node, "data_format", "NHWC"));
  TF_RETURN_IF_ERROR(CheckOptionalAttr(node, "T", DT_FLOAT));

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
  if (!HasAttr(node, "strides")) {
    return tensorflow::errors::InvalidArgument("Missing attribute 'strides'");
  }
  const auto& strides = GetListAttr(node, "strides");
  TF_RETURN_IF_ERROR(ExpectValue(strides.i_size(), 4, "number of strides"));
  TF_RETURN_IF_ERROR(ExpectValue(strides.i(0), 1, "strides(0)"));
  TF_RETURN_IF_ERROR(ExpectValue(strides.i(3), 1, "strides(3)"));
  conv->stride_height = strides.i(1);
  conv->stride_width = strides.i(2);
  if (HasAttr(node, "dilations")) {
    const auto& dilations = GetListAttr(node, "dilations");
    TF_RETURN_IF_ERROR(
        ExpectValue(dilations.i_size(), 4, "number of dilations"));
    if (dilations.i(0) != 1 || dilations.i(3) != 1) {
      return tensorflow::errors::InvalidArgument(absl::StrCat(
          "Can only import Conv ops with dilation along the height "
          "(1st) or width (2nd) axis. TensorFlow op \"",
          node.name(), "\" had dilations:[ ", dilations.i(0), ", ",
          dilations.i(1), ", ", dilations.i(2), ", ", dilations.i(3), "]."));
    }
    conv->dilation_height_factor = dilations.i(1);
    conv->dilation_width_factor = dilations.i(2);
  } else {
    conv->dilation_height_factor = 1;
    conv->dilation_width_factor = 1;
  }
  const auto& padding = GetStringAttr(node, "padding");
  if (padding == "SAME") {
    conv->padding.type = PaddingType::kSame;
  } else if (padding == "VALID") {
    conv->padding.type = PaddingType::kValid;
  } else {
    return tensorflow::errors::InvalidArgument(
        "Bad padding (only SAME and VALID are supported)");
  }
  model->operators.emplace_back(conv);

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertDepthwiseConvOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "DepthwiseConv2dNative");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));

  // We only support NHWC, which is the default data_format.
  // So if data_format is not defined, we're all good.
  if (HasAttr(node, "data_format")) {
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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertDepthToSpaceOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "DepthToSpace");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));

  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);
  auto* op = new DepthToSpaceOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  op->block_size = GetIntAttr(node, "block_size");
  QCHECK_GE(op->block_size, 2);
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSpaceToDepthOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "SpaceToDepth");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));

  tensorflow::DataType dtype = GetDataTypeAttr(node, "T");
  if (dtype != DT_FLOAT && dtype != DT_UINT8 && dtype != DT_INT32 &&
      dtype != DT_INT64) {
    const auto* enum_descriptor = tensorflow::DataType_descriptor();
    LOG(FATAL) << "TFLite does not support SpaceToDepth with type T:"
               << enum_descriptor->FindValueByNumber(dtype)->name() << ". "
               << "T must be one of {DT_FLOAT, DT_INT8, DT_INT32, DT_INT64}.";
  }
  auto* op = new SpaceToDepthOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  op->block_size = GetIntAttr(node, "block_size");
  QCHECK_GE(op->block_size, 2);
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertBiasAddOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "BiasAdd");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));

  const auto& input_name = node.input(0);
  const auto& bias_name = node.input(1);
  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_FLOAT);
  auto* biasadd = new AddOperator;
  biasadd->inputs.push_back(input_name);
  biasadd->inputs.push_back(bias_name);
  biasadd->outputs.push_back(node.name());
  model->operators.emplace_back(biasadd);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertRandomUniform(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "RandomUniform");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));

  CHECK_EQ(GetDataTypeAttr(node, "T"), DT_INT32);
  auto op = absl::make_unique<RandomUniformOperator>();
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  op->dtype = ConvertDataType(GetDataTypeAttr(node, "dtype"));
  op->seed = GetIntAttr(node, "seed");
  op->seed2 = GetIntAttr(node, "seed2");
  CHECK(model != nullptr);
  model->operators.emplace_back(std::move(op));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertIdentityOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertFakeQuantWithMinMaxArgs(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "FakeQuantWithMinMaxArgs");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
  auto* op = new FakeQuantOperator;
  op->inputs.push_back(node.input(0));
  op->minmax.reset(new MinMax);
  auto& minmax = *op->minmax;
  minmax.min = GetFloatAttr(node, "min");
  minmax.max = GetFloatAttr(node, "max");
  op->outputs.push_back(node.name());
  // tf.fake_quant_with_min_max_args num_bits defaults to 8.
  op->num_bits = HasAttr(node, "num_bits") ? GetIntAttr(node, "num_bits") : 8;
  if (HasAttr(node, "narrow_range")) {
    op->narrow_range = GetBoolAttr(node, "narrow_range");
  }
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertFakeQuantWithMinMaxVars(
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
  op->num_bits = HasAttr(node, "num_bits") ? GetIntAttr(node, "num_bits") : 8;
  if (HasAttr(node, "narrow_range")) {
    op->narrow_range = GetBoolAttr(node, "narrow_range");
  }
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSqueezeOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Squeeze");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
  auto* op = new SqueezeOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());

  // When omitted we are to squeeze all dimensions == 1.
  if (HasAttr(node, "squeeze_dims")) {
    const auto& squeeze_dims = GetListAttr(node, "squeeze_dims");
    for (int i = 0; i < squeeze_dims.i_size(); ++i) {
      op->squeeze_dims.push_back(squeeze_dims.i(i));
    }
  }

  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSplitOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Split");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSwitchOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Switch");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
  auto* op = new TensorFlowSwitchOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  // Switch operators have two outputs: "name" and "name:1".
  op->outputs.push_back(node.name() + ":1");
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSoftmaxOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Softmax");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
  const auto& input_name = node.input(0);
  auto* softmax = new SoftmaxOperator;
  softmax->inputs.push_back(input_name);
  softmax->outputs.push_back(node.name());
  // TensorFlow's Softmax doesn't seem to admit a 'beta' parameter.
  CHECK(!node.attr().count("beta"));  // Stab in the dark, just in case.
  softmax->beta = 1.f;
  model->operators.emplace_back(softmax);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertLRNOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "LRN");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
  const auto& input_name = node.input(0);
  auto* lrn = new LocalResponseNormalizationOperator;
  lrn->inputs.push_back(input_name);
  lrn->outputs.push_back(node.name());
  lrn->range = GetIntAttr(node, "depth_radius");
  lrn->bias = GetFloatAttr(node, "bias");
  lrn->alpha = GetFloatAttr(node, "alpha");
  lrn->beta = GetFloatAttr(node, "beta");
  model->operators.emplace_back(lrn);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertMaxPoolOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "MaxPool");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertAvgPoolOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "AvgPool");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertBatchMatMulOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));

  // https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions
  CHECK(!HasAttr(node, "adj_a") || (GetBoolAttr(node, "adj_a") == false));
  CHECK(!HasAttr(node, "adj_b") || (GetBoolAttr(node, "adj_b") == false));

  auto* batch_matmul = new BatchMatMulOperator;
  batch_matmul->inputs = {node.input(0), node.input(1)};
  batch_matmul->outputs = {node.name()};
  model->operators.emplace_back(batch_matmul);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertMatMulOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));

  CHECK(!HasAttr(node, "adjoint_a") ||
        (GetBoolAttr(node, "adjoint_a") == false));
  CHECK(!HasAttr(node, "adjoint_b") ||
        (GetBoolAttr(node, "adjoint_b") == false));

  auto* matmul = new TensorFlowMatMulOperator;
  if (HasAttr(node, "transpose_a")) {
    matmul->transpose_a = GetBoolAttr(node, "transpose_a");
  }
  if (HasAttr(node, "transpose_b")) {
    matmul->transpose_b = GetBoolAttr(node, "transpose_b");
  }

  matmul->inputs = {node.input(0), node.input(1)};
  matmul->outputs = {node.name()};
  model->operators.emplace_back(matmul);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConcatOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
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
  return tensorflow::Status::OK();
}

// This method supports simple operators without additional attributes.
template <typename Op>
tensorflow::Status ConvertSimpleOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  auto* op = new Op;
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

// This method supports simple operators without additional attributes.
template <typename Op, unsigned int NumInputs>
tensorflow::Status ConvertSimpleOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, NumInputs));
  return ConvertSimpleOperator<Op>(node, tf_import_flags, model);
}

tensorflow::Status ConvertUnsupportedOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
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
  } else if (HasAttr(node, "Tout")) {
    const auto& output_type = GetDataTypeAttr(node, "Tout");
    op->output_data_types.push_back(ConvertDataType(output_type));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertStridedSliceOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "StridedSlice");
  // TODO(soroosh): The 4th input (strides) should be e optional, to be
  // consistent with TF.
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 4));

  auto* op = new StridedSliceOperator;
  for (const auto& input : node.input()) {
    op->inputs.push_back(input);
  }
  op->outputs.push_back(node.name());

  op->begin_mask =
      HasAttr(node, "begin_mask") ? GetIntAttr(node, "begin_mask") : 0;
  op->ellipsis_mask =
      HasAttr(node, "ellipsis_mask") ? GetIntAttr(node, "ellipsis_mask") : 0;
  op->end_mask = HasAttr(node, "end_mask") ? GetIntAttr(node, "end_mask") : 0;
  op->new_axis_mask =
      HasAttr(node, "new_axis_mask") ? GetIntAttr(node, "new_axis_mask") : 0;
  op->shrink_axis_mask = HasAttr(node, "shrink_axis_mask")
                             ? GetIntAttr(node, "shrink_axis_mask")
                             : 0;

  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPlaceholderOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK(node.op() == "Placeholder" || node.op() == "LegacyFedInput");
  if (node.op() == "Placeholder") {
    TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 0));
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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertNoOpOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertCastOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Cast");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
  const auto tf_src_dtype = GetDataTypeAttr(node, "SrcT");
  const auto tf_dst_dtype = GetDataTypeAttr(node, "DstT");
  auto* op = new CastOperator;
  op->src_data_type = ConvertDataType(tf_src_dtype);
  op->dst_data_type = ConvertDataType(tf_dst_dtype);
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertFloorOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Floor");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
  const auto data_type = GetDataTypeAttr(node, "T");
  CHECK(data_type == DT_FLOAT);
  auto* op = new FloorOperator;
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertGatherOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK(node.op() == "Gather" || node.op() == "GatherV2");
  if (node.op() == "Gather")
    TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
  if (node.op() == "GatherV2")
    TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 3));
  const auto indices_data_type = GetDataTypeAttr(node, "Tindices");
  CHECK(indices_data_type == DT_INT32 || indices_data_type == DT_INT64);
  auto* op = new GatherOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  if (node.input_size() >= 3) {
    // GatherV2 form where we are provided an axis. It may be either a constant
    // or runtime defined value, so we just wire up the array and let
    // ResolveGatherAttributes take care of it later on.
    const auto axis_data_type = GetDataTypeAttr(node, "Taxis");
    CHECK(axis_data_type == DT_INT32 || axis_data_type == DT_INT64);
    op->inputs.push_back(node.input(2));
  } else {
    // Gather form that assumes axis=0.
    op->axis = {0};
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

template <typename Op, const char* op_name>
tensorflow::Status ConvertArgMinMaxOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), op_name);
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
  const auto axis_data_type =
      HasAttr(node, "Tidx") ? GetDataTypeAttr(node, "Tidx") : DT_INT32;
  const auto output_type = HasAttr(node, "output_type")
                               ? GetDataTypeAttr(node, "output_type")
                               : DT_INT64;
  CHECK(axis_data_type == DT_INT64 || axis_data_type == DT_INT32);
  CHECK(output_type == DT_INT64 || output_type == DT_INT32);
  auto* op = new Op;
  op->output_data_type = ConvertDataType(output_type);
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertResizeBilinearOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "ResizeBilinear");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
  auto* op = new ResizeBilinearOperator;

  op->align_corners = false;
  if (HasAttr(node, "align_corners")) {
    op->align_corners = GetBoolAttr(node, "align_corners");
  }

  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertBatchNormWithGlobalNormalizationOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "BatchNormWithGlobalNormalization");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 5));

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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertFusedBatchNormOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "FusedBatchNorm");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 5));

  // Declare shortcuts for the inputs.
  const string& gamma_input = node.input(1);
  const string& beta_input = node.input(2);
  const string& moving_mean_input = node.input(3);
  const string& moving_variance_input = node.input(4);

  // Create an array holding the epsilon value (typically, 0.001).
  const string epsilon_array_name = CreateConstArray<ArrayDataType::kFloat>(
      model, node.name() + "_epsilon_array", {GetFloatAttr(node, "epsilon")});

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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSpaceToBatchNDOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "SpaceToBatchND");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 3));
  CHECK_EQ(GetDataTypeAttr(node, "Tblock_shape"), DT_INT32);
  CHECK_EQ(GetDataTypeAttr(node, "Tpaddings"), DT_INT32);
  auto* op = new SpaceToBatchNDOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(2));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertBatchToSpaceNDOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "BatchToSpaceND");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 3));
  CHECK_EQ(GetDataTypeAttr(node, "Tblock_shape"), DT_INT32);
  CHECK_EQ(GetDataTypeAttr(node, "Tcrops"), DT_INT32);
  auto* op = new BatchToSpaceNDOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(2));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status ConvertReduceOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
  auto* op = new T;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op);
  if (HasAttr(node, "keepdims")) {
    op->keep_dims = GetBoolAttr(node, "keepdims");
  } else if (HasAttr(node, "keep_dims")) {
    op->keep_dims = GetBoolAttr(node, "keep_dims");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSvdfOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
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
  return tensorflow::Status::OK();
}

// This is just bare bones support to get the shapes to propagate.
tensorflow::Status ConvertTransposeConvOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Conv2DBackpropInput");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 3));
  auto* op = new TransposeConvOperator;
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->inputs.push_back(node.input(2));
  op->outputs.push_back(node.name());
  const auto& strides = GetListAttr(node, "strides");
  op->stride_height = strides.i(1);
  op->stride_width = strides.i(2);
  CHECK_EQ(strides.i_size(), 4)
      << "Can only import TransposeConv ops with 4D strides. TensorFlow op \""
      << node.name() << "\" has " << strides.i_size() << "D strides.";
  CHECK((strides.i(0) == 1) && (strides.i(3) == 1))
      << "Can only import TransposeConv ops with striding along the height "
         "(1st) or width (2nd) axis. TensorFlow op \""
      << node.name() << "\" had strides:[ " << strides.i(0) << ", "
      << strides.i(1) << ", " << strides.i(2) << ", " << strides.i(3) << "].";
  op->stride_height = strides.i(1);
  op->stride_width = strides.i(2);
  if (HasAttr(node, "dilations")) {
    const auto& dilations = GetListAttr(node, "dilations");
    CHECK_EQ(dilations.i_size(), 4)
        << "Dilation unsupported in TransposeConv. TensorFlow op \""
        << node.name() << "\" had dilations";
    CHECK((dilations.i(0) == 1) && (dilations.i(1) == 1) &&
          (dilations.i(1) == 1) && (dilations.i(3) == 1))
        << "Dilation unsupported in TransposeConv. TensorFlow op \""
        << node.name() << "\" had dilations:[ " << dilations.i(0) << ", "
        << dilations.i(1) << ", " << dilations.i(2) << ", " << dilations.i(3)
        << "].";
  }

  const string& weights_name = node.input(TransposeConvOperator::WEIGHTS);
  const string& transposed_weights_name = weights_name + "_transposed";
  // Check if a TransposeOperator was already created for these weights
  // (can happen when multiple layers share the same weights).
  const Operator* existing_transpose =
      GetOpWithOutput(*model, transposed_weights_name);
  if (existing_transpose) {
    CHECK(existing_transpose->type == OperatorType::kTranspose);
  } else {
    // Transpose weights from HWOI order to OHWI order, which is more efficient
    // for computation. (Note that TensorFlow considers the order as HWIO
    // because they consider this a backward conv, inverting the sense of
    // input/output.)
    TransposeOperator* transpose = new TransposeOperator;
    string perm_array = CreateConstArray<ArrayDataType::kInt32>(
        model, node.name() + "_transpose_perm", {2, 0, 1, 3});
    transpose->inputs = {weights_name, perm_array};
    transpose->outputs = {transposed_weights_name};
    model->operators.emplace_back(transpose);
  }
  op->inputs[1] = transposed_weights_name;

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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertRangeOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Range");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 3));
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
  return tensorflow::Status::OK();
}

// Note that it's easy to confuse/conflate "Stack" and "Pack" operators, but
// they aren't the same thing.  tf.stack results in a "Pack" operator.  "Stack"
// operators also exist, but involve manipulating the TF runtime stack, and are
// not directly related to tf.stack() usage.
tensorflow::Status ConvertPackOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Pack");
  auto op = absl::make_unique<PackOperator>();
  const int num_inputs = GetInputsCount(node, tf_import_flags);
  QCHECK_GE(num_inputs, 1)
      << node.op()
      << " node expects at least 1 input other than control dependencies: "
      << node.DebugString();
  CHECK_EQ(num_inputs, GetIntAttr(node, "N"));
  for (int i = 0; i < num_inputs; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->axis = HasAttr(node, "axis") ? GetIntAttr(node, "axis") : 0;
  op->dtype = ConvertDataType(toco::GetDataTypeAttr(node, "T"));
  op->outputs.push_back(node.name());
  model->operators.emplace_back(std::move(op));
  return tensorflow::Status::OK();
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
tensorflow::Status ConvertOperatorSpecialCasedAsRNNBackEdge(
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
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertShapeOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Shape");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 1));
  const auto out_type =
      HasAttr(node, "out_type") ? GetDataTypeAttr(node, "out_type") : DT_INT32;
  CHECK(out_type == DT_INT64 || out_type == DT_INT32);
  auto op = absl::make_unique<TensorFlowShapeOperator>();
  op->output_data_type = ConvertDataType(out_type);
  op->inputs.push_back(node.input(0));
  op->outputs.push_back(node.name());
  model->operators.push_back(std::move(op));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertAnyOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "Any");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
  const auto idx_type =
      HasAttr(node, "Tidx") ? GetDataTypeAttr(node, "Tidx") : DT_INT32;
  CHECK(idx_type == DT_INT32);
  auto op = absl::make_unique<AnyOperator>();
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  op->outputs.push_back(node.name());
  op->keep_dims =
      HasAttr(node, "keep_dims") ? GetBoolAttr(node, "keep_dims") : false;
  model->operators.push_back(std::move(op));
  return tensorflow::Status::OK();
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

tensorflow::Status ConvertTopKV2Operator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK((node.op() == "TopK") || (node.op() == "TopKV2"));
  auto op = absl::make_unique<TopKV2Operator>();
  op->inputs.push_back(node.input(0));
  // K can be encoded as attr (TopK) convert it to a const.
  if (HasAttr(node, "k")) {
    string k_array = CreateConstArray<ArrayDataType::kInt32>(
        model, node.name() + "k", {static_cast<int32>(GetIntAttr(node, "k"))});
    op->inputs.push_back(k_array);
  } else {
    TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
    op->inputs.push_back(node.input(1));
  }
  // The op has two outputs.
  op->outputs.push_back(node.name());
  op->outputs.push_back(node.name() + ":1");
  model->operators.emplace_back(op.release());
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertDynamicPartitionOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  auto op = absl::make_unique<DynamicPartitionOperator>();
  CHECK(HasAttr(node, "num_partitions"));
  op->num_partitions = GetIntAttr(node, "num_partitions");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 2));
  op->inputs.push_back(node.input(0));
  op->inputs.push_back(node.input(1));
  CHECK_GT(op->num_partitions, 1);
  op->outputs.push_back(node.name());  // Implicit :0.
  for (int i = 1; i < op->num_partitions; ++i) {
    op->outputs.push_back(node.name() + ":" + std::to_string(i));
  }
  model->operators.emplace_back(op.release());
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertDynamicStitchOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  // The parallel and non-parallel variants are the same besides whether they
  // have a parallel loop; there are no behavioral differences.
  CHECK(node.op() == "DynamicStitch" || node.op() == "ParallelDynamicStitch");
  auto op = absl::make_unique<DynamicStitchOperator>();
  CHECK(HasAttr(node, "N"));
  op->num_partitions = GetIntAttr(node, "N");
  // Expect all ID partitions + all value partitions.
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, op->num_partitions * 2));
  for (int i = 0; i < op->num_partitions * 2; ++i) {
    op->inputs.push_back(node.input(i));
  }
  op->outputs.push_back(node.name());
  model->operators.emplace_back(op.release());
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSparseToDenseOperator(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model) {
  CHECK_EQ(node.op(), "SparseToDense");
  TF_QCHECK_OK(CheckInputsCount(node, tf_import_flags, 4));

  auto* op = new SparseToDenseOperator;
  for (const string& input : node.input()) {
    op->inputs.push_back(input);
  }
  op->outputs.push_back(node.name());

  op->validate_indices = HasAttr(node, "validate_indices")
                             ? GetBoolAttr(node, "validate_indices")
                             : true;
  model->operators.emplace_back(op);
  return tensorflow::Status::OK();
}

}  // namespace

namespace internal {

using ConverterType = tensorflow::Status (*)(
    const NodeDef& node, const TensorFlowImportFlags& tf_import_flags,
    Model* model);
using ConverterMapType = std::unordered_map<std::string, ConverterType>;

constexpr char kArgMax[] = "ArgMax";
constexpr char kArgMin[] = "ArgMin";

ConverterMapType GetTensorFlowNodeConverterMap() {
  return std::unordered_map<std::string, ConverterType>({
      {"Add", ConvertSimpleOperator<AddOperator, 2>},
      {"AddN", ConvertSimpleOperator<AddNOperator>},
      {"All", ConvertSimpleOperator<TensorFlowAllOperator>},
      {"Any", ConvertAnyOperator},
      {"ArgMax", ConvertArgMinMaxOperator<ArgMaxOperator, kArgMax>},
      {"ArgMin", ConvertArgMinMaxOperator<ArgMinOperator, kArgMin>},
      {"Assert", ConvertSimpleOperator<TensorFlowAssertOperator>},
      {"AvgPool", ConvertAvgPoolOperator},
      {"BatchMatMul", ConvertBatchMatMulOperator},
      {"BatchNormWithGlobalNormalization",
       ConvertBatchNormWithGlobalNormalizationOperator},
      {"BatchToSpaceND", ConvertBatchToSpaceNDOperator},
      {"BiasAdd", ConvertBiasAddOperator},
      {"Cast", ConvertCastOperator},
      {"CheckNumerics", ConvertIdentityOperator},
      {"Concat", ConvertConcatOperator},
      {"ConcatV2", ConvertConcatOperator},
      {"Const", ConvertConstOperator},
      {"Conv2D", ConvertConvOperator},
      {"Conv2DBackpropInput", ConvertTransposeConvOperator},
      {"DepthToSpace", ConvertDepthToSpaceOperator},
      {"DepthwiseConv2dNative", ConvertDepthwiseConvOperator},
      {"Div", ConvertSimpleOperator<DivOperator, 2>},
      {"DynamicPartition", ConvertDynamicPartitionOperator},
      {"DynamicStitch", ConvertDynamicStitchOperator},
      {"Equal", ConvertSimpleOperator<TensorFlowEqualOperator, 2>},
      {"Exp", ConvertSimpleOperator<ExpOperator, 1>},
      {"ExpandDims", ConvertSimpleOperator<ExpandDimsOperator, 2>},
      {"FakeQuantWithMinMaxArgs", ConvertFakeQuantWithMinMaxArgs},
      {"FakeQuantWithMinMaxVars", ConvertFakeQuantWithMinMaxVars},
      {"Fill", ConvertSimpleOperator<FillOperator, 2>},
      {"Floor", ConvertFloorOperator},
      {"FloorDiv", ConvertSimpleOperator<FloorDivOperator, 2>},
      {"FloorMod", ConvertSimpleOperator<FloorModOperator, 2>},
      {"FusedBatchNorm", ConvertFusedBatchNormOperator},
      {"Gather", ConvertGatherOperator},
      {"GatherV2", ConvertGatherOperator},
      {"Greater", ConvertSimpleOperator<TensorFlowGreaterOperator, 2>},
      {"GreaterEqual",
       ConvertSimpleOperator<TensorFlowGreaterEqualOperator, 2>},
      {"Identity", ConvertIdentityOperator},
      {"LRN", ConvertLRNOperator},
      {"LegacyFedInput", ConvertPlaceholderOperator},
      {"Less", ConvertSimpleOperator<TensorFlowLessOperator, 2>},
      {"LessEqual", ConvertSimpleOperator<TensorFlowLessEqualOperator, 2>},
      {"Log", ConvertSimpleOperator<LogOperator, 1>},
      {"LogSoftmax", ConvertSimpleOperator<LogSoftmaxOperator, 1>},
      {"LogicalAnd", ConvertSimpleOperator<LogicalAndOperator, 2>},
      {"LogicalNot", ConvertSimpleOperator<LogicalNotOperator, 1>},
      {"MatMul", ConvertMatMulOperator},
      {"Max", ConvertReduceOperator<TensorFlowMaxOperator>},
      {"MaxPool", ConvertMaxPoolOperator},
      {"Maximum", ConvertSimpleOperator<TensorFlowMaximumOperator, 2>},
      {"Mean", ConvertReduceOperator<MeanOperator>},
      {"Merge", ConvertSimpleOperator<TensorFlowMergeOperator, 2>},
      {"Min", ConvertReduceOperator<TensorFlowMinOperator>},
      {"Minimum", ConvertSimpleOperator<TensorFlowMinimumOperator, 2>},
      {"Mul", ConvertSimpleOperator<MulOperator, 2>},
      {"Neg", ConvertSimpleOperator<NegOperator, 1>},
      {"NextIteration", ConvertOperatorSpecialCasedAsRNNBackEdge},
      {"NoOp", ConvertNoOpOperator},
      {"NotEqual", ConvertSimpleOperator<TensorFlowNotEqualOperator, 2>},
      {"Pack", ConvertPackOperator},
      {"Pad", ConvertSimpleOperator<PadOperator, 2>},
      {"PadV2", ConvertSimpleOperator<PadV2Operator, 3>},
      {"ParallelDynamicStitch", ConvertDynamicStitchOperator},
      {"Placeholder", ConvertPlaceholderOperator},
      {"PlaceholderWithDefault", ConvertIdentityOperator},
      {"Pow", ConvertSimpleOperator<PowOperator, 2>},
      {"Prod", ConvertReduceOperator<TensorFlowProdOperator>},
      {"RandomUniform", ConvertRandomUniform},
      {"Range", ConvertRangeOperator},
      {"Rank", ConvertSimpleOperator<RankOperator, 1>},
      {"RealDiv", ConvertSimpleOperator<DivOperator, 2>},
      {"Relu", ConvertSimpleOperator<ReluOperator, 1>},
      {"Relu6", ConvertSimpleOperator<Relu6Operator, 1>},
      {"Reshape", ConvertSimpleOperator<TensorFlowReshapeOperator, 2>},
      {"ResizeBilinear", ConvertResizeBilinearOperator},
      {"Rsqrt", ConvertSimpleOperator<TensorFlowRsqrtOperator, 1>},
      {"Select", ConvertSimpleOperator<SelectOperator, 3>},
      {"Shape", ConvertShapeOperator},
      {"Sigmoid", ConvertSimpleOperator<LogisticOperator, 1>},
      {"Sin", ConvertSimpleOperator<SinOperator, 1>},
      {"Slice", ConvertSimpleOperator<SliceOperator, 3>},
      {"Softmax", ConvertSoftmaxOperator},
      {"SpaceToBatchND", ConvertSpaceToBatchNDOperator},
      {"SpaceToDepth", ConvertSpaceToDepthOperator},
      {"SparseToDense", ConvertSparseToDenseOperator},
      {"Split", ConvertSplitOperator},
      {"Sqrt", ConvertSimpleOperator<TensorFlowSqrtOperator, 1>},
      {"Square", ConvertSimpleOperator<TensorFlowSquareOperator, 1>},
      {"Squeeze", ConvertSqueezeOperator},
      {"StopGradient", ConvertIdentityOperator},
      {"StridedSlice", ConvertStridedSliceOperator},
      {"Sub", ConvertSimpleOperator<SubOperator, 2>},
      {"Sum", ConvertReduceOperator<TensorFlowSumOperator>},
      {"Svdf", ConvertSvdfOperator},
      {"Switch", ConvertSwitchOperator},
      {"Tanh", ConvertSimpleOperator<TanhOperator, 1>},
      {"Tile", ConvertSimpleOperator<TensorFlowTileOperator, 2>},
      {"TopK", ConvertTopKV2Operator},
      {"TopKV2", ConvertTopKV2Operator},
      {"Transpose", ConvertSimpleOperator<TransposeOperator, 2>},
  });
}

tensorflow::Status ImportTensorFlowNode(
    const tensorflow::NodeDef& node,
    const TensorFlowImportFlags& tf_import_flags, Model* model,
    const ConverterMapType& converter_map) {
  auto converter = converter_map.find(node.op());
  if (converter == converter_map.end()) {
    return ConvertUnsupportedOperator(node, tf_import_flags, model);
  } else {
    return converter->second(node, tf_import_flags, model);
  }
}
}  // namespace internal

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
  const internal::ConverterMapType& converter_map =
      internal::GetTensorFlowNodeConverterMap();

  for (auto node : inlined_graph.node()) {
    StripZeroOutputIndexFromInputs(&node);
    auto status = internal::ImportTensorFlowNode(node, tf_import_flags, model,
                                                 converter_map);
    CHECK(status.ok()) << status.error_message();
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
