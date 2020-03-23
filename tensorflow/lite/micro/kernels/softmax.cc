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

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {
namespace {

TfLiteStatus CalculateQuantizedSoftmaxParams(TfLiteContext* context,
                                             const TfLiteTensor* input,
                                             TfLiteTensor* output,
                                             const TfLiteSoftmaxParams* params,
                                             SoftmaxParams* data) {
  if (input->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt8);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);
    if (output->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, -32768);
      // NOTE: Current int16 softmax output does not require symmetric scaling
      // - so no need to verify scale here.
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
      TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
    }
  }

  static const int kScaledDiffIntegerBits = 5;

  int input_left_shift;
  tflite::PreprocessSoftmaxScaling(static_cast<double>(params->beta),
                                   static_cast<double>(input->params.scale),
                                   kScaledDiffIntegerBits,
                                   &data->input_multiplier, &input_left_shift);
  data->input_left_shift = input_left_shift;
  data->diff_min = -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                                       data->input_left_shift);
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
                        TfLiteSoftmaxParams* params,
                        const SoftmaxParams& op_params) {
  // TODO(ahentz): this is arguably a dirty trick. Since the implementation
  // always traverses the last dimension of a 4D tensor, we will pretend our 1D
  // tensor is 4D in a special way. We will convert a (Y) shape into a (1,
  // 1, 1, Y) shape.
  const int input_size = input->dims->data[0];
  const int32_t shape_data[4] = {1, 1, 1, input_size};
  RuntimeShape shape(4, shape_data);
  if (input->type == kTfLiteUInt8) {
    tflite::reference_ops::Softmax(op_params, shape,
                                   GetTensorData<uint8_t>(input), shape,
                                   GetTensorData<uint8_t>(output));
  } else {
    if (output->type == kTfLiteInt16) {
      tflite::reference_ops::Softmax(op_params, shape,
                                     GetTensorData<int8_t>(input), shape,
                                     GetTensorData<int16_t>(output));
    } else {
      tflite::reference_ops::Softmax(op_params, shape,
                                     GetTensorData<int8_t>(input), shape,
                                     GetTensorData<int8_t>(output));
    }
  }
}

void Softmax2DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params,
                        const SoftmaxParams& op_params) {
  // TODO(ahentz): this is arguably a dirty trick. Since the implementation
  // always traverses the last dimension of a 4D tensor, we will pretend our 2D
  // tensor is 4D in a special way. We will convert a (X, Y) shape into a (X,
  // 1, 1, Y) shape.
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int32_t shape_data[4] = {batch_size, 1, 1, input_size};
  RuntimeShape shape(4, shape_data);
  if (input->type == kTfLiteUInt8) {
    tflite::reference_ops::Softmax(op_params, shape,
                                   GetTensorData<uint8_t>(input), shape,
                                   GetTensorData<uint8_t>(output));
  } else {
    if (output->type == kTfLiteInt16) {
      tflite::reference_ops::Softmax(op_params, shape,
                                     GetTensorData<int8_t>(input), shape,
                                     GetTensorData<int16_t>(output));
    } else {
      tflite::reference_ops::Softmax(op_params, shape,
                                     GetTensorData<int8_t>(input), shape,
                                     GetTensorData<int8_t>(output));
    }
  }
}

// Takes a 4D tensor and perform softmax along the forth dimension.
void Softmax4DFloat(const TfLiteTensor* input, TfLiteTensor* output,
                    TfLiteSoftmaxParams* params) {
  SoftmaxParams op_params;
  op_params.beta = static_cast<double>(params->beta);
  tflite::reference_ops::Softmax(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(output), GetTensorData<float>(output));
}

void Softmax4DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params,
                        const SoftmaxParams& op_params) {
  if (input->type == kTfLiteUInt8) {
    tflite::reference_ops::Softmax(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  } else {
    if (output->type == kTfLiteInt16) {
      tflite::reference_ops::Softmax(
          op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output));
    } else {
      tflite::reference_ops::Softmax(
          op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
    }
  }
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  // TODO(ahentz): consider an implementation that works for many (all?)
  // dimensions.
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
    case kTfLiteInt8:
    case kTfLiteUInt8: {
      SoftmaxParams op_params;
      TF_LITE_ENSURE_STATUS(CalculateQuantizedSoftmaxParams(
          context, input, output, params, &op_params));
      if (NumDimensions(input) == 1) {
        Softmax1DQuantized(input, output, params, op_params);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 2) {
        Softmax2DQuantized(input, output, params, op_params);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 4) {
        Softmax4DQuantized(input, output, params, op_params);
        return kTfLiteOk;
      }
      TF_LITE_KERNEL_LOG(
          context, "Only 1D, 2D and 4D tensors supported currently, got %dD.",
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
  static TfLiteRegistration r = {};
  r.init = activations::Init;
  r.free = activations::Free;
  r.prepare = activations::SoftmaxPrepare;
  r.invoke = activations::SoftmaxEval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
