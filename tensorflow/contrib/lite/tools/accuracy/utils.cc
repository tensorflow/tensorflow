/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/tools/accuracy/utils.h"

#include <sys/stat.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/op_resolver.h"

namespace tensorflow {
namespace metrics {

namespace utils {

DataType GetTFDataType(TfLiteType tflite_type) {
  switch (tflite_type) {
    case kTfLiteFloat32:
      return DT_FLOAT;
    case kTfLiteUInt8:
      return DT_UINT8;
    default:
      return DT_INVALID;
  }
}

TensorShape GetTFLiteTensorShape(const TfLiteTensor& tflite_tensor) {
  TensorShape shape;
  for (int i = 0; i < tflite_tensor.dims->size; i++) {
    shape.AddDim(tflite_tensor.dims->data[i]);
  }
  return shape;
}

Status ReadFileLines(const string& file_path,
                     std::vector<string>* lines_output) {
  if (!lines_output) {
    return errors::InvalidArgument("Invalid output");
  }
  std::vector<string> lines;
  std::ifstream stream(file_path, std::ios_base::in);
  if (!stream) {
    return errors::InvalidArgument("Unable to open file: ", file_path);
  }
  std::string line;
  while (std::getline(stream, line)) {
    lines_output->push_back(line);
  }
  return Status::OK();
}

Status GetTFliteModelInfo(const string& model_file_path,
                          ModelInfo* model_info) {
  if (model_file_path.empty()) {
    return errors::InvalidArgument("Invalid model file.");
  }
  struct stat stat_buf;
  if (stat(model_file_path.c_str(), &stat_buf) != 0) {
    int error_num = errno;
    return errors::InvalidArgument("Invalid model file: ", model_file_path,
                                   std::strerror(error_num));
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(model_file_path.data());
  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    return errors::InvalidArgument("Invalid model", model_file_path);
  }
  for (int i : interpreter->inputs()) {
    TfLiteTensor* tensor = interpreter->tensor(i);
    model_info->input_shapes.push_back(utils::GetTFLiteTensorShape(*tensor));
    model_info->input_types.push_back(utils::GetTFDataType(tensor->type));
  }
  return Status::OK();
}

}  // namespace utils
}  // namespace metrics
}  // namespace tensorflow
