/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>

#include <algorithm>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace bucketize {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  // boundaries array is owned by the buffer housing TfLiteBucketizeParams.
  const float* boundaries;
  int num_boundaries;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  const auto* params = reinterpret_cast<const TfLiteBucketizeParams*>(buffer);

  if (!FLATBUFFERS_LITTLEENDIAN) {
    int32_t* p =
        reinterpret_cast<int32_t*>(const_cast<float*>(params->boundaries));
    for (size_t i = 0; i < params->num_boundaries; i++, p++)
      *p = flatbuffers::EndianSwap(*p);
  }

  op_data->boundaries = params->boundaries;
  op_data->num_boundaries = params->num_boundaries;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  if (!std::is_sorted(opdata->boundaries,
                      opdata->boundaries + opdata->num_boundaries)) {
    TF_LITE_KERNEL_LOG(context, "Expected sorted boundaries");
    return kTfLiteError;
  }

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteInt64 && input->type != kTfLiteFloat64) {
    TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by bucketize.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = kTfLiteInt32;

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

template <typename T>
inline void Bucketize(const RuntimeShape& input_shape, const T* input_data,
                      const float* boundaries, int num_boundaries,
                      const RuntimeShape& output_shape, int32_t* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    auto first_bigger_it = std::upper_bound(
        boundaries, boundaries + num_boundaries, input_data[i]);
    output_data[i] = first_bigger_it - boundaries;
  }
}

template <typename T>
TfLiteStatus BucketizeImpl(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt32);

  Bucketize<T>(GetTensorShape(input), GetTensorData<T>(input),
               opdata->boundaries, opdata->num_boundaries,
               GetTensorShape(output), GetTensorData<int32_t>(output));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  switch (input->type) {
    case kTfLiteFloat32: {
      return BucketizeImpl<float>(context, node);
    }
    case kTfLiteFloat64: {
      return BucketizeImpl<double>(context, node);
    }
    case kTfLiteInt32: {
      return BucketizeImpl<int32_t>(context, node);
    }
    case kTfLiteInt64: {
      return BucketizeImpl<int64_t>(context, node);
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by bucketize.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace
}  // namespace bucketize

TfLiteRegistration* Register_BUCKETIZE() {
  static TfLiteRegistration r = {bucketize::Init, bucketize::Free,
                                 bucketize::Prepare, bucketize::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
