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

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include "arm_nnfunctions.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
};

TfLiteStatus CalculateSoftmaxOpData(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    OpData* data) {
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
    }

    TF_LITE_ENSURE(context, (output->params.scale == 1.f / 256) ||
                                (output->params.scale == 1.f / 255));

    static const int kScaledDiffIntegerBits = 5;

    tflite::PreprocessSoftmaxScaling(
        params->beta, input->params.scale, kScaledDiffIntegerBits,
        &data->input_multiplier, &data->input_left_shift);
    data->diff_min = -1.0 * tflite::CalculateInputRadius(
                                kScaledDiffIntegerBits, data->input_left_shift);
  }
  return kTfLiteOk;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

// Takes a 1D tensor and performs softmax along it.
void Softmax1DFloat(const TfLiteTensor* input, TfLiteTensor* output,
                    TfLiteSoftmaxParams* params) {
  const int input_size = input->dims->data[0];
  tflite::reference_ops::Softmax(input->data.f, input_size, 1, params->beta,
                                 output->data.f);
}

// Takes a 2D tensor and perform softmax along the last dimension.
void Softmax2DFloat(const TfLiteTensor* input, TfLiteTensor* output,
                    TfLiteSoftmaxParams* params) {
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  tflite::reference_ops::Softmax(input->data.f, input_size, batch_size,
                                 params->beta, output->data.f);
}

void Softmax1DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params, OpData* data) {
  const int input_size = input->dims->data[0];
  const int32_t shape_data[4] = {1, 1, 1, input_size};
  RuntimeShape shape(4, shape_data);
  SoftmaxParams op_params;
  op_params.input_multiplier = data->input_multiplier;
  op_params.input_left_shift = data->input_left_shift;
  op_params.diff_min = data->diff_min;
  if (input->type == kTfLiteUInt8) {
    tflite::reference_ops::Softmax(op_params, shape,
                                   GetTensorData<uint8_t>(input), shape,
                                   GetTensorData<uint8_t>(output));
  } else {
    arm_softmax_s8(GetTensorData<int8_t>(input), shape_data[0], shape_data[3],
                   op_params.input_multiplier, op_params.input_left_shift,
                   op_params.diff_min, GetTensorData<int8_t>(output));
  }
}

void Softmax2DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params, OpData* data) {
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int32_t shape_data[4] = {batch_size, 1, 1, input_size};
  RuntimeShape shape(4, shape_data);
  SoftmaxParams op_params;
  op_params.input_multiplier = data->input_multiplier;
  op_params.input_left_shift = data->input_left_shift;
  op_params.diff_min = data->diff_min;
  if (input->type == kTfLiteUInt8) {
    tflite::reference_ops::Softmax(op_params, shape,
                                   GetTensorData<uint8_t>(input), shape,
                                   GetTensorData<uint8_t>(output));
  } else {
    arm_softmax_s8(GetTensorData<int8_t>(input), shape_data[0], shape_data[3],
                   op_params.input_multiplier, op_params.input_left_shift,
                   op_params.diff_min, GetTensorData<int8_t>(output));
  }
}

// Takes a 4D tensor and perform softmax along the forth dimension.
void Softmax4DFloat(const TfLiteTensor* input, TfLiteTensor* output,
                    TfLiteSoftmaxParams* params) {
  SoftmaxParams op_params;
  op_params.beta = params->beta;
  tflite::reference_ops::Softmax(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(output), GetTensorData<float>(output));
}

void Softmax4DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params, OpData* data) {
  SoftmaxParams op_params;
  op_params.input_multiplier = data->input_multiplier;
  op_params.input_left_shift = data->input_left_shift;
  op_params.diff_min = data->diff_min;
  if (input->type == kTfLiteUInt8) {
    tflite::reference_ops::Softmax(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  } else {
    arm_softmax_s8(
        GetTensorData<int8_t>(input),
        input->dims->data[0] * input->dims->data[1] * input->dims->data[2],
        input->dims->data[3], op_params.input_multiplier,
        op_params.input_left_shift, op_params.diff_min,
        GetTensorData<int8_t>(output));
  }
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  OpData local_data_object;
  OpData* data = &local_data_object;
  TF_LITE_ENSURE_STATUS(
      CalculateSoftmaxOpData(context, input, output, params, data));

  switch (input->type) {
    case kTfLiteFloat32: {
      if (NumDimensions(input) == 1) {
        Softmax1DFloat(input, output, params);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 2) {
        Softmax2DFloat(input, output, params);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 4) {
        Softmax4DFloat(input, output, params);
        return kTfLiteOk;
      }
      TF_LITE_KERNEL_LOG(
          context, "Only 1D, 2D and 4D tensors supported currently, got %dD.",
          NumDimensions(input));
      return kTfLiteError;
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      if (NumDimensions(input) == 1) {
        Softmax1DQuantized(input, output, params, data);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 2) {
        Softmax2DQuantized(input, output, params, data);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 4) {
        Softmax4DQuantized(input, output, params, data);
        return kTfLiteOk;
      }
      TF_LITE_KERNEL_LOG(context,
                         "Only 2D and 4D tensors supported currently, got %dD.",
                         NumDimensions(input));
      return kTfLiteError;
    }
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Only float32, uint8_t and int8_t supported currently, got %d.",
          input->type);
      return kTfLiteError;
  }
}
}  // namespace activations

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::SoftmaxPrepare,
                                 activations::SoftmaxEval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
