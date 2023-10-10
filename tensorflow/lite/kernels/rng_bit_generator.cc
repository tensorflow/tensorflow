/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/rng_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_rng_bit_generator {

namespace {

// Inputs and outputs tensor index.
constexpr int kInitialState = 0;
constexpr int kOutputKey = 0;
constexpr int kOutput = 1;

template <typename T, size_t K>
void FillOutputBuffer(uint32_t* output_buffer, uint32_t* output_state_buffer,
                      int64_t output_num_elements, T fn,
                      std::array<uint32_t, K>& ctr, uint32_t key_0,
                      uint32_t key_1) {
  int64_t i = 0;
  while (i < output_num_elements) {
    auto val = fn(key_0, key_1, ctr);
    int64_t copy_size = (output_num_elements - i >= val.size())
                            ? val.size()
                            : output_num_elements - i;
    memcpy(output_buffer + i, &val, copy_size * sizeof(uint32_t));
    i += copy_size;
    if (!++ctr[0]) {
      ++ctr[1];
    }
  }

  output_state_buffer[0] = key_0;
  output_state_buffer[1] = key_1;
  output_state_buffer[2] = ctr[0];
  output_state_buffer[3] = ctr[1];
}

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  // Initial state is 1D vector.
  const TfLiteTensor* initial_state;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInitialState, &initial_state));
  TF_LITE_ENSURE_EQ(context, initial_state->type, kTfLiteUInt64);
  TF_LITE_ENSURE_EQ(context, NumDimensions(initial_state), 1);

  TfLiteTensor* output_key;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputKey, &output_key));
  TF_LITE_ENSURE_EQ(context, output_key->type, kTfLiteUInt64);
  TF_LITE_ENSURE(context, HaveSameShapes(output_key, initial_state));
  TfLiteIntArray* output_key_size_array = TfLiteIntArrayCopy(output_key->dims);
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_key,
                                                   output_key_size_array));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutput, &output));
  TF_LITE_ENSURE(context, output->type == kTfLiteInt32 ||
                              output->type == kTfLiteInt64 ||
                              output->type == kTfLiteUInt32 ||
                              output->type == kTfLiteUInt64);
  // Output tensor has a static shape.
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCopy(output->dims);
  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteStablehloRngBitGeneratorParams*>(
      node->builtin_data);
  TfLiteRngAlgorithm algorithm = params->algorithm;
  const TfLiteTensor* initial_state = GetInput(context, node, 0);
  TfLiteTensor* output_key = GetOutput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 1);
  TF_LITE_ENSURE(context, !IsDynamicTensor(output));

  int64_t output_num_elements = NumElements(output);
  switch (output->type) {
    case kTfLiteUInt64:
    case kTfLiteInt64:
      output_num_elements *= sizeof(uint64_t) / sizeof(uint32_t);
      break;
    case kTfLiteUInt32:
    case kTfLiteInt32:
      // no-op here as the byte size is equal to sizeof(uint32_t).
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Unsupported output data type: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  switch (algorithm) {
    case TfLiteRngAlgorithm::kTfLiteRngAlgorithmThreefry: {
      // Initial state for the THREEFRY algorithm should be a u64[2].
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(initial_state, 0), 2);
      // Deliberately cast uint64_t* to uint32_t* here.
      const uint32_t* state_vals = GetTensorData<uint32_t>(initial_state);
      std::array<uint32_t, 2> ctr{state_vals[2], state_vals[3]};
      FillOutputBuffer<decltype(tflite::rng::Threefry2x32), 2>(
          static_cast<uint32_t*>(output->data.data),
          static_cast<uint32_t*>(output_key->data.data), output_num_elements,
          tflite::rng::Threefry2x32, ctr,
          /*key_0=*/state_vals[0], /*key_1=*/state_vals[1]);
      break;
    }
    case TfLiteRngAlgorithm::kTfLiteRngAlgorithmPhilox:
    case TfLiteRngAlgorithm::kTfLiteRngAlgorithmDefault: {
      // Initial state for the PHILOX algorithm should be a u64[2] or u64[3].
      int state_dim_0_size = SizeOfDimension(initial_state, 0);
      TF_LITE_ENSURE(context, state_dim_0_size == 2 || state_dim_0_size == 3);
      // Deliberately cast uint64_t* to uint32_t* here.
      const uint32_t* state_vals = GetTensorData<uint32_t>(initial_state);
      std::array<uint32_t, 4> ctr{state_vals[2], state_vals[3],
                                  state_vals[state_dim_0_size == 3 ? 4 : 0],
                                  state_vals[state_dim_0_size == 3 ? 5 : 1]};
      // First copy over the initial state.
      memcpy(output_key->data.data, state_vals,
             state_dim_0_size * sizeof(uint64_t));
      FillOutputBuffer<decltype(tflite::rng::Philox4x32), 4>(
          static_cast<uint32_t*>(output->data.data),
          static_cast<uint32_t*>(output_key->data.data), output_num_elements,
          tflite::rng::Philox4x32, ctr, /*key_0=*/state_vals[0],
          /*key_1=*/state_vals[1]);
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Unknown RNG algorithm: %d", algorithm);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace stablehlo_rng_bit_generator

TfLiteRegistration* Register_STABLEHLO_RNG_BIT_GENERATOR() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 stablehlo_rng_bit_generator::Prepare,
                                 stablehlo_rng_bit_generator::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
