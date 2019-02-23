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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/lite/context.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace metrics {

namespace utils {

struct ModelInfo {
  std::vector<TensorShape> input_shapes;
  std::vector<DataType> input_types;
};

Status GetTFliteModelInfo(const string& model_file_path, ModelInfo* model_info);

DataType GetTFDataType(TfLiteType tflite_type);

TensorShape GetTFLiteTensorShape(const TfLiteTensor& tflite_tensor);

Status ReadFileLines(const string& file_path,
                     std::vector<string>* lines_output);
}  // namespace utils
}  // namespace metrics
}  // namespace tensorflow
#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_UTILS_H_
