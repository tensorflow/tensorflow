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

template <size_t N, size_t H>
void FwhtStaticSize(float* data) {
  for (size_t h = H; h < N; h *= 2) {
    for (size_t i = 0; i < N; i += 2 * h) {
      for (size_t j = i; j < i + h; ++j) {
        const float x = data[j];
        const float y = data[j + h];
        data[j] = x + y;
        data[j + h] = x - y;
      }
    }
  }
}

// Fast Walsh Hadamard Transform. Updates `data` in place.
template <size_t FixedSize = 128>
void FwhtFast(float* data, int hadamard_size) {
  if ((hadamard_size & (hadamard_size - 1)) != 0) {
    std::cerr << "hadamard_size needs to be a power of 2\n";
    return;
  }

  if (hadamard_size < FixedSize) {
    // Note: Manually unrolling like this may cause binary size inflation. This
    // can be traded against performance by implementing the algorithm for a
    // variable size.
    switch (hadamard_size) {
      case 2:
        FwhtStaticSize<2, 1>(data);
        break;
      case 4:
        FwhtStaticSize<4, 1>(data);
        break;
      case 8:
        FwhtStaticSize<8, 1>(data);
        break;
      case 16:
        FwhtStaticSize<16, 1>(data);
        break;
      case 32:
        FwhtStaticSize<32, 1>(data);
        break;
      case 64:
        FwhtStaticSize<64, 1>(data);
        break;
      case 128:
        FwhtStaticSize<128, 1>(data);
        break;
    }
    static_assert(FixedSize <= 128,
                  "Missing cases for unrolling array sizes bigger than 128.");
  } else {
    const int num_chunks = hadamard_size / FixedSize;
    float* in = data;
    // Use general, iterative loops algorithm for sizes up to FixedSize.
    for (int chunk = 0; chunk < num_chunks; ++chunk, in += FixedSize) {
      FwhtStaticSize<FixedSize, 1>(in);
    }
    // Finish the bigger butterflies with explicit unrolling.
    for (int chunk_size = FixedSize; chunk_size < hadamard_size;
         chunk_size *= 2) {
      float* in1 = &data[0];
      float* in2 = &data[chunk_size];
      for (int i = 0; i < hadamard_size;
           i += chunk_size * 2, in1 += chunk_size, in2 += chunk_size) {
        for (int j = i; j < i + chunk_size; j += FixedSize) {
          // Compiler will unroll this fixed size loop easily.
          for (int k = 0; k < FixedSize; k++) {
            float x = *in1;
            float y = *in2;
            *in1++ = x + y;
            *in2++ = x - y;
          }
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
    FwhtFast(&output->data.f[chunk_start], hadamard_size);
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
