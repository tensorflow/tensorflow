/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h"

#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

std::ostream& operator<<(std::ostream& os, const TfLiteTensor& tensor) {
  std::string shape;
  absl::optional<std::string> result = tflite::ShapeToString(tensor.dims);
  if (result.has_value()) {
    shape = std::move(result.value());
  } else {
    shape = "[error: unsupported number of dimensions]";
  }
  return os << "tensor of shape " << shape;
}

namespace tflite {

absl::optional<std::string> ShapeToString(TfLiteIntArray* shape) {
  std::string result;
  int* data = shape->data;
  switch (shape->size) {
    case 1:
      result = absl::Substitute("Linear=[$0]", data[0]);
      break;
    case 2:
      result = absl::Substitute("HW=[$0, $1]", data[0], data[1]);
      break;
    case 3:
      result = absl::Substitute("HWC=[$0, $1, $2]", data[0], data[1], data[2]);
      break;
    case 4:
      result = absl::Substitute("BHWC=[$0, $1, $2, $3]", data[0], data[1],
                                data[2], data[3]);
      break;
    default:
      // This printer doesn't expect shapes of more than 4 dimensions.
      return absl::nullopt;
  }
  return result;
}

absl::optional<std::string> CoordinateToString(TfLiteIntArray* shape,
                                               int linear) {
  std::string result;
  switch (shape->size) {
    case 1: {
      result = absl::Substitute("[$0]", linear);
      break;
    } break;
    case 2: {
      const int tensor_width = shape->data[1];
      const int h_coord = linear / tensor_width;
      const int w_coord = linear % tensor_width;
      result = absl::Substitute("[$0, $1]", h_coord, w_coord);
      break;
    } break;
    case 3: {
      const int tensor_width = shape->data[1];
      const int tensor_channels = shape->data[2];
      const int h_coord = linear / (tensor_width * tensor_channels);
      const int w_coord =
          (linear % (tensor_width * tensor_channels)) / tensor_channels;
      const int c_coord =
          (linear % (tensor_width * tensor_channels)) % tensor_channels;
      result = absl::Substitute("[$0, $1, $2]", h_coord, w_coord, c_coord);
      break;
    } break;
    case 4: {
      const int tensor_height = shape->data[1];
      const int tensor_width = shape->data[2];
      const int tensor_channels = shape->data[3];
      const int b_coord =
          linear / (tensor_height * tensor_width * tensor_channels);
      const int h_coord =
          (linear % (tensor_height * tensor_width * tensor_channels)) /
          (tensor_width * tensor_channels);
      const int w_coord =
          ((linear % (tensor_height * tensor_width * tensor_channels)) %
           (tensor_width * tensor_channels)) /
          tensor_channels;
      const int c_coord =
          ((linear % (tensor_height * tensor_width * tensor_channels)) %
           (tensor_width * tensor_channels)) %
          tensor_channels;
      result = absl::Substitute("[$0, $1, $2, $3]", b_coord, h_coord, w_coord,
                                c_coord);
      break;
    }
    default:
      // This printer doesn't expect shapes of more than 4 dimensions.
      return absl::nullopt;
  }
  return result;
}

// Builds intepreter for a model, allocates tensors.
absl::Status BuildInterpreter(const Model* model,
                              std::unique_ptr<Interpreter>* interpreter) {
  TfLiteStatus status =
      InterpreterBuilder(model, ops::builtin::BuiltinOpResolver())(interpreter);
  if (status != kTfLiteOk || !*interpreter) {
    return absl::InternalError(
        "Failed to initialize interpreter with model binary.");
  }
  return absl::OkStatus();
}

absl::Status AllocateTensors(std::unique_ptr<Interpreter>* interpreter) {
  if ((*interpreter)->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("Failed to allocate tensors.");
  }
  return absl::OkStatus();
}

absl::Status ModifyGraphWithDelegate(std::unique_ptr<Interpreter>* interpreter,
                                     TfLiteDelegate* delegate) {
  if ((*interpreter)->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    return absl::InternalError("Failed modify graph with delegate.");
  }
  return absl::OkStatus();
}

void InitializeInputs(int left, int right,
                      std::unique_ptr<Interpreter>* interpreter) {
  for (int id : (*interpreter)->inputs()) {
    float* input_data = (*interpreter)->typed_tensor<float>(id);
    int input_size = (*interpreter)->input_tensor(id)->bytes;
    for (int i = 0; i < input_size; i++) {
      input_data[i] = left + i % right;
    }
  }
}

absl::Status Invoke(std::unique_ptr<Interpreter>* interpreter) {
  if ((*interpreter)->Invoke() != kTfLiteOk) {
    return absl::InternalError("Failed during inference.");
  }
  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os, const TestParams& param) {
  return os << param.name;
}

}  // namespace tflite
