/******************************************************************************
 * Copyright (C) 2019 Cadence Design Systems, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to use this Software with Cadence processor cores only and
 * not with any other processors and platforms, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

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
#include "tensorflow/lite/kernels/internal/reference/integer_ops/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "xtensa_tf_micro_common.h"

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
      if (output->type == kTfLiteInt16) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -32768);
        // NOTE: Current int16 softmax output does not require symmetric scaling
        // - so no need to verify scale here.
      } else {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
        TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
      }
    }

    static const int kScaledDiffIntegerBits = 5;

    tflite::PreprocessSoftmaxScaling(
        static_cast<double>(params->beta),
        static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
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
TfLiteStatus Softmax2DFloat(TfLiteContext* context, const TfLiteTensor* input,
                            TfLiteTensor* output, TfLiteSoftmaxParams* params) {
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];

  ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
  float* p_scratch = (float*)xtensa_nnlib_scratch_buf;

  if (input->dims->data[1] * sizeof(float) > XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
    TF_LITE_KERNEL_LOG(context, "Softmax: insufficient scratch memory");
    return kTfLiteError;
  }

  for (int i = 0; i < batch_size * input_size; ++i) {
    p_scratch[i] = input->data.f[i] * params->beta;
  }

  for (int i = 0; i < batch_size; ++i) {
    int err = xa_nn_vec_softmax_f32_f32(&output->data.f[i * input_size],
                                        &p_scratch[i * input_size], input_size);
    CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_softmax_f32_f32 failed");
  }
  return kTfLiteOk;
}

void Softmax1DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params, OpData* data) {
  // (ahentz): this is arguably a dirty trick. Since the implementation
  // always traverses the last dimension of a 4D tensor, we will pretend our 1D
  // tensor is 4D in a special way. We will convert a (Y) shape into a (1,
  // 1, 1, Y) shape.
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
    if (output->type == kTfLiteInt16) {
      tflite::reference_integer_ops::Softmax(
          op_params, shape, GetTensorData<int8_t>(input), shape,
          GetTensorData<int16_t>(output));
    } else {
      tflite::reference_integer_ops::Softmax(
          op_params, shape, GetTensorData<int8_t>(input), shape,
          GetTensorData<int8_t>(output));
    }
  }
}

TfLiteStatus Softmax2DQuantized(TfLiteContext* context,
                                const TfLiteTensor* input, TfLiteTensor* output,
                                TfLiteSoftmaxParams* params, OpData* data) {
  // (ahentz): this is arguably a dirty trick. Since the implementation
  // always traverses the last dimension of a 4D tensor, we will pretend our 2D
  // tensor is 4D in a special way. We will convert a (X, Y) shape into a (X,
  // 1, 1, Y) shape.
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int32_t shape_data[4] = {batch_size, 1, 1, input_size};
  RuntimeShape shape(4, shape_data);
  SoftmaxParams op_params;
  op_params.input_multiplier = data->input_multiplier;
  op_params.input_left_shift = data->input_left_shift;
  op_params.diff_min = data->diff_min;

  if (input->type == kTfLiteUInt8) {
    ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM;
    void* p_scratch = (void*)xtensa_nnlib_scratch_buf;

    if (get_softmax_scratch_size(PREC_ASYM8, PREC_ASYM8, input_size) >
        XTENSA_NNLIB_MAX_SCRATCH_SIZE) {
      TF_LITE_KERNEL_LOG(context, "Softmax: insufficient scratch memory");
      return kTfLiteError;
    }

    for (int i = 0; i < batch_size; ++i) {
      int err = xa_nn_vec_softmax_asym8_asym8(
          &output->data.uint8[i * input_size],
          &input->data.uint8[i * input_size], op_params.diff_min,
          op_params.input_left_shift, op_params.input_multiplier, input_size,
          p_scratch);
      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_softmax_asym8_asym8 failed");
    }
  } else {
    if (output->type == kTfLiteInt16) {
      tflite::reference_integer_ops::Softmax(
          op_params, shape, GetTensorData<int8_t>(input), shape,
          GetTensorData<int16_t>(output));
    } else {
      tflite::reference_integer_ops::Softmax(
          op_params, shape, GetTensorData<int8_t>(input), shape,
          GetTensorData<int8_t>(output));
    }
  }
  return kTfLiteOk;
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
    if (output->type == kTfLiteInt16) {
      tflite::reference_integer_ops::Softmax(
          op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int16_t>(output));
    } else {
      tflite::reference_integer_ops::Softmax(
          op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(output), GetTensorData<int8_t>(output));
    }
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

  // (ahentz): consider an implementation that works for many (all?)
  // dimensions.
  switch (input->type) {
    case kTfLiteFloat32: {
      if (NumDimensions(input) == 1) {
        Softmax1DFloat(input, output, params);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 2) {
        return Softmax2DFloat(context, input, output, params);
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
      if (NumDimensions(input) == 1) {
        Softmax1DQuantized(input, output, params, data);
        return kTfLiteOk;
      }
      if (NumDimensions(input) == 2) {
        return Softmax2DQuantized(context, input, output, params, data);
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
