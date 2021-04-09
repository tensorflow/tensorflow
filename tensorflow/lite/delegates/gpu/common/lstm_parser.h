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
#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_LSTM_PARSER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_LSTM_PARSER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/object_reader.h"

namespace tflite {
namespace gpu {

absl::Status ParseLSTMAttributes(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    GraphFloat32* graph, ObjectReader* reader, const TfLiteLSTMParams* params,
    absl::flat_hash_map<int, ValueId>* new_variable_input_values);
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_LSTM_PARSER_H_
