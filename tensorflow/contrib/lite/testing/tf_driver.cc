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
#include "tensorflow/contrib/lite/testing/tf_driver.h"

#include <fstream>
#include <iostream>

#include "tensorflow/contrib/lite/testing/join.h"
#include "tensorflow/contrib/lite/testing/split.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tflite {
namespace testing {

namespace {

tensorflow::Tensor CreateTensor(const tensorflow::DataType type,
                                const std::vector<int64_t>& dim) {
  tensorflow::TensorShape shape{gtl::ArraySlice<int64>{
      reinterpret_cast<const int64*>(dim.data()), dim.size()}};
  return {type, shape};
}

template <typename T>
void FillTensorWithData(tensorflow::Tensor* tensor, const string& csv_values) {
  auto data = tensor->flat<T>();

  const auto& values = testing::Split<T>(csv_values, ",");
  for (int i = 0; i < values.size(); i++) {
    data(i) = values[i];
  }
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
  }

  output_ids_.resize(output_layer.size());
  output_tensors_.reserve(output_layer.size());
  for (int i = 0; i < output_layer.size(); i++) {
    output_ids_[i] = i;
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

void TfDriver::SetInput(int id, const string& csv_values) {
  if (!IsValid()) return;

  auto tensor = CreateTensor(input_types_[id], input_shapes_[id]);
  switch (input_types_[id]) {
    case tensorflow::DT_FLOAT: {
      FillTensorWithData<float>(&tensor, csv_values);
      break;
    }
    case tensorflow::DT_INT32: {
      FillTensorWithData<int32_t>(&tensor, csv_values);
      break;
    }
    case tensorflow::DT_UINT8: {
      FillTensorWithData<uint8_t>(&tensor, csv_values);
      break;
    }
    default:
      fprintf(stderr, "Unsupported type %d in SetInput\n", input_types_[id]);
      Invalidate("Unsupported tensor data type");
      return;
  }
  input_tensors_[input_names_[id]] = tensor;
}

void TfDriver::ResetTensor(int id) {
  if (!IsValid()) return;
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
      fprintf(stderr, "Unsupported type %d in ResetTensor\n", input_types_[id]);
      Invalidate("Unsupported tensor data type");
      return;
  }
}

void TfDriver::ReshapeTensor(int id, const string& csv_values) {
  input_shapes_[id] = Split<int64_t>(csv_values, ",");
  input_tensors_[input_names_[id]] =
      CreateTensor(input_types_[id], input_shapes_[id]);
  ResetTensor(id);
}

string TfDriver::ReadOutput(int id) {
  if (!IsValid()) return "";
  switch (output_tensors_[id].dtype()) {
    case tensorflow::DT_FLOAT:
      return TensorDataToCsvString<float>(output_tensors_[id]);
    case tensorflow::DT_INT32:
      return TensorDataToCsvString<int32_t>(output_tensors_[id]);
    case tensorflow::DT_UINT8:
      return TensorDataToCsvString<uint8_t>(output_tensors_[id]);
    default:
      fprintf(stderr, "Unsupported type %d in ResetTensor\n", input_types_[id]);
      Invalidate("Unsupported tensor data type");
      return "";
  }
}

void TfDriver::Invoke() {
  if (!IsValid()) return;
  auto status = session_->Run({input_tensors_.begin(), input_tensors_.end()},
                              output_names_, {}, &output_tensors_);
  if (!status.ok()) {
    Invalidate("Failed to invoke interpreter");
  }
}

}  // namespace testing
}  // namespace tflite
