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
#include "tensorflow/lite/testing/tf_driver.h"

#include <fstream>
#include <iostream>

#include "absl/strings/escaping.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {

namespace {

tensorflow::Tensor CreateTensor(const tensorflow::DataType type,
                                const std::vector<int64_t>& dim) {
  tensorflow::TensorShape shape{tensorflow::gtl::ArraySlice<int64_t>{
      reinterpret_cast<const int64_t*>(dim.data()), dim.size()}};
  return {type, shape};
}

template <typename T>
int FillTensorWithData(tensorflow::Tensor* tensor,
                       const string& values_as_string) {
  const auto& values = testing::Split<T>(values_as_string, ",");

  if (values.size() == tensor->NumElements()) {
    auto data = tensor->flat<T>();
    for (int i = 0; i < values.size(); i++) {
      data(i) = values[i];
    }
  }

  return values.size();
}

// Assumes 'values_as_string' is a hex string that gets converted into a
// TF Lite DynamicBuffer. Strings are then extracted and copied into the
// TensorFlow tensor.
int FillTensorWithTfLiteHexString(tensorflow::Tensor* tensor,
                                  const string& values_as_string) {
  string s = absl::HexStringToBytes(values_as_string);

  int num_strings = values_as_string.empty() ? 0 : GetStringCount(s.data());

  if (num_strings == tensor->NumElements()) {
    auto data = tensor->flat<tensorflow::tstring>();
    for (size_t i = 0; i < num_strings; ++i) {
      auto ref = GetString(s.data(), i);
      data(i).assign(ref.str, ref.len);
    }
  }

  return num_strings;
}

template <typename T>
void FillTensorWithZeros(tensorflow::Tensor* tensor) {
  auto data = tensor->flat<T>();
  for (int i = 0; i < tensor->NumElements(); i++) {
    data(i) = 0;
  }
}

template <typename T>
string TensorDataToCsvString(const tensorflow::Tensor& tensor) {
  const auto& data = tensor.flat<T>();
  return Join(data.data(), data.size(), ",");
}

string TensorDataToTfLiteHexString(const tensorflow::Tensor& tensor) {
  DynamicBuffer dynamic_buffer;

  auto data = tensor.flat<tensorflow::tstring>();
  for (int i = 0; i < tensor.NumElements(); ++i) {
    dynamic_buffer.AddString(data(i).data(), data(i).size());
  }

  char* char_buffer = nullptr;
  size_t size = dynamic_buffer.WriteToBuffer(&char_buffer);
  string s = absl::BytesToHexString({char_buffer, size});
  free(char_buffer);

  return s;
}

}  // namespace

TfDriver::TfDriver(const std::vector<string>& input_layer,
                   const std::vector<string>& input_layer_type,
                   const std::vector<string>& input_layer_shape,
                   const std::vector<string>& output_layer)
    : input_names_(input_layer), output_names_(output_layer) {
  CHECK_EQ(input_layer.size(), input_layer_type.size());
  CHECK_EQ(input_layer.size(), input_layer_shape.size());

  input_ids_.resize(input_layer.size());
  input_tensors_.reserve(input_layer.size());
  input_types_.resize(input_layer.size());
  input_shapes_.resize(input_layer.size());
  for (int i = 0; i < input_layer.size(); i++) {
    input_ids_[i] = i;
    input_tensors_[input_layer[i]] = {};
    CHECK(DataTypeFromString(input_layer_type[i], &input_types_[i]));
    input_shapes_[i] = Split<int64_t>(input_layer_shape[i], ",");
    input_name_to_id_[input_layer[i]] = i;
  }

  output_ids_.resize(output_layer.size());
  output_tensors_.reserve(output_layer.size());
  for (int i = 0; i < output_layer.size(); i++) {
    output_ids_[i] = i;
    output_name_to_id_[output_layer[i]] = i;
  }
}

void TfDriver::LoadModel(const string& bin_file_path) {
  if (!IsValid()) return;
  std::ifstream model(bin_file_path);
  if (model.fail()) {
    Invalidate("Failed to find the model " + bin_file_path);
    return;
  }

  tensorflow::GraphDef graphdef;
  if (!graphdef.ParseFromIstream(&model)) {
    Invalidate("Failed to parse tensorflow graphdef");
    return;
  }

  tensorflow::SessionOptions options;
  session_.reset(tensorflow::NewSession(options));
  auto status = session_->Create(graphdef);
  if (!status.ok()) {
    Invalidate("Failed to create session. " + status.error_message());
  }
}

void TfDriver::ReshapeTensor(const string& name, const string& csv_values) {
  if (!IsValid()) return;
  int id = input_name_to_id_[name];
  input_shapes_[id] = Split<int64_t>(csv_values, ",");
  input_tensors_[input_names_[id]] =
      CreateTensor(input_types_[id], input_shapes_[id]);
  ResetTensor(name);
}

void TfDriver::ResetTensor(const std::string& name) {
  if (!IsValid()) return;
  int id = input_name_to_id_[name];
  auto tensor = input_tensors_[input_names_[id]];
  switch (input_types_[id]) {
    case tensorflow::DT_FLOAT: {
      FillTensorWithZeros<float>(&tensor);
      break;
    }
    case tensorflow::DT_INT32: {
      FillTensorWithZeros<int32_t>(&tensor);
      break;
    }
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ", input_types_[id],
                              tensorflow::DataType_Name(input_types_[id]),
                              " in ResetInput"));
      return;
  }
}
string TfDriver::ReadOutput(const string& name) {
  if (!IsValid()) return "";
  return ReadOutput(output_tensors_[output_name_to_id_[name]]);
}
void TfDriver::Invoke(const std::vector<std::pair<string, string>>& inputs) {
  if (!IsValid()) return;
  for (const auto& input : inputs) {
    auto id = input_name_to_id_[input.first];
    auto tensor = CreateTensor(input_types_[id], input_shapes_[id]);
    SetInput(input.second, &tensor);
    input_tensors_[input_names_[id]] = tensor;
  }
  auto status = session_->Run({input_tensors_.begin(), input_tensors_.end()},
                              output_names_, {}, &output_tensors_);
  if (!status.ok()) {
    Invalidate(absl::StrCat("TensorFlow failed to run graph:",
                            status.error_message()));
  }
}

void TfDriver::SetInput(const string& values_as_string,
                        tensorflow::Tensor* tensor) {
  int num_values_available = 0;
  switch (tensor->dtype()) {
    case tensorflow::DT_FLOAT:
      num_values_available =
          FillTensorWithData<float>(tensor, values_as_string);
      break;
    case tensorflow::DT_INT32:
      num_values_available =
          FillTensorWithData<int32_t>(tensor, values_as_string);
      break;
    case tensorflow::DT_UINT32:
      num_values_available =
          FillTensorWithData<uint32_t>(tensor, values_as_string);
      break;
    case tensorflow::DT_UINT8:
      num_values_available =
          FillTensorWithData<uint8_t>(tensor, values_as_string);
      break;
    case tensorflow::DT_STRING:
      num_values_available =
          FillTensorWithTfLiteHexString(tensor, values_as_string);
      break;
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              tensorflow::DataType_Name(tensor->dtype()),
                              " in SetInput"));
      return;
  }

  if (tensor->NumElements() != num_values_available) {
    Invalidate(absl::StrCat("Needed ", tensor->NumElements(),
                            " values for input tensor, but was given ",
                            num_values_available, " instead."));
  }
}

string TfDriver::ReadOutput(const tensorflow::Tensor& tensor) {
  switch (tensor.dtype()) {
    case tensorflow::DT_FLOAT:
      return TensorDataToCsvString<float>(tensor);
    case tensorflow::DT_INT32:
      return TensorDataToCsvString<int32_t>(tensor);
    case tensorflow::DT_UINT32:
      return TensorDataToCsvString<uint32_t>(tensor);
    case tensorflow::DT_INT64:
      return TensorDataToCsvString<int64_t>(tensor);
    case tensorflow::DT_UINT8:
      return TensorDataToCsvString<uint8_t>(tensor);
    case tensorflow::DT_STRING:
      return TensorDataToTfLiteHexString(tensor);
    case tensorflow::DT_BOOL:
      return TensorDataToCsvString<bool>(tensor);
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              tensorflow::DataType_Name(tensor.dtype()),
                              " in ReadOutput"));
      return "";
  }
}

}  // namespace testing
}  // namespace tflite
