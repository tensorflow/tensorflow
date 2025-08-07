/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace aeq_hadamard_rotation {

static const int kInputTensor = 0;
static const int kOutputTensor = 0;

struct OpData {
  bool is_initialized = false;
  int hadamard_size = 0;
  std::vector<int> random_binary_vector;
};

// Fast Walsh Hadamard Transform. Updates `data` in place.
void FWHTGeneral(float* data, int n, bool normalize) {
  if ((n & (n - 1)) != 0) {
    std::cerr << "Error: Input size must be a power of 2." << std::endl;
    return;
  }

  int h = 1;
  while (h < n) {
    for (int i = 0; i < n; i += h * 2) {
      for (int j = i; j < i + h; ++j) {
        float x = data[j];
        float y = data[j + h];
        data[j] = x + y;
        data[j + h] = x - y;
      }
    }
    h *= 2;
  }
  if (normalize) {
    // Calculate the inverse square root once.
    const float norm_factor = 1.0f / std::sqrt(static_cast<float>(n));
    for (int k = 0; k < n; ++k) {
      data[k] *= norm_factor;
    }
  }
}

// Same FWHT algorithm, with loops explicitly unrolled for sizes >= 16.
void FWHTFast(float* data, int hadamard_size) {
  std::vector<float> output(hadamard_size);
  int num_chunks = hadamard_size / 16;

  float* in = data;
  // Use general, iterative loops algorithm for sizes up to 16.
  for (int chunk = 0; chunk < num_chunks; ++chunk, in += 16) {
    FWHTGeneral(in, 16, false);
  }
  // Finish the bigger butterflies with explicit unrolling.
  for (int chunk_size = 16; chunk_size < hadamard_size; chunk_size *= 2) {
    float* in1 = &data[0];
    float* in2 = &data[chunk_size];
    for (int i = 0; i < hadamard_size;
         i += chunk_size * 2, in1 += chunk_size, in2 += chunk_size) {
      for (int j = i; j < i + chunk_size; j += 16) {
        // Compiler will unroll this fixed size loop easily.
        for (int k = 0; k < 16; k++) {
          float x = *in1;
          float y = *in2;
          *in1++ = x + y;
          *in2++ = x - y;
        }
      }
    }
  }

  // Calculate the inverse square root once.
  const float norm_factor = 1.0f / std::sqrt(hadamard_size);
  for (int i = 0; i < hadamard_size; ++i) {
    data[i] *= norm_factor;
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* op_data = new OpData();
  op_data->is_initialized = false;
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  if (!op_data->is_initialized) {
    const uint8_t* buffer =
        reinterpret_cast<const uint8_t*>(node->custom_initial_data);
    const size_t length = node->custom_initial_data_size;
    auto flexbuffer_map = flexbuffers::GetRoot(buffer, length).AsMap();
    int32_t hadamard_size = flexbuffer_map["hadamard_size"].AsInt32();
    std::vector<int> vec;
    const auto& vector = flexbuffer_map["random_binary_vector"].AsVector();
    vec.reserve(vector.size());
    for (size_t i = 0; i < vector.size(); i++) {
      vec.push_back(vector[i].AsInt8());
    }
    op_data->hadamard_size = hadamard_size;
    op_data->random_binary_vector = vec;
    op_data->is_initialized = true;
  }

  // Prepare the inputs.
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &input_tensor));

  TF_LITE_ENSURE(context, input_tensor->type == kTfLiteFloat32 ||
                              input_tensor->type == kTfLiteInt32);

  return kTfLiteOk;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &input_tensor));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int hadamard_size = op_data->hadamard_size;
  int input_batch = 1;
  int input_features = input_tensor->dims->data[0];
  int input_feature_size = input_tensor->dims->data[1];
  if (input_tensor->dims->size == 3) {
    input_batch = input_tensor->dims->data[0];
    input_features = input_tensor->dims->data[1];
    input_feature_size = input_tensor->dims->data[2];
  }

  memcpy(output->data.f, input_tensor->data.f, input_tensor->bytes);

  int num_hadamards_per_feature = input_feature_size / hadamard_size;
  const int total_transforms =
      input_batch * input_features * num_hadamards_per_feature;
  for (int i = 0; i < total_transforms; ++i) {
    int chunk_start = i * hadamard_size;
    // Update output->data.f in place.
    if (hadamard_size < 16) {
      FWHTGeneral(&output->data.f[chunk_start], hadamard_size, true);
    } else {
      FWHTFast(&output->data.f[chunk_start], hadamard_size);
    }
  }

  return kTfLiteOk;
}

}  // namespace aeq_hadamard_rotation

TfLiteRegistration* Register_HADAMARD_ROTATION() {
  static TfLiteRegistration r = {
      aeq_hadamard_rotation::Init, aeq_hadamard_rotation::Free,
      aeq_hadamard_rotation::Prepare, aeq_hadamard_rotation::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
