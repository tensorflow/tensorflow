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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_SVDF_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_SVDF_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

struct OpData {
  int32_t effective_scale_1_a;
  int32_t effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
  int scratch_tensor_index;
  int scratch_output_tensor_index;

  // Cached tensor zero point values for quantized operations.
  int input_zero_point;
  int output_zero_point;
};

// Input tensors.
extern const int kSvdfInputTensor;
extern const int kSvdfWeightsFeatureTensor;
extern const int kSvdfWeightsTimeTensor;
extern const int kSvdfBiasTensor;
// This is a variable tensor, and will be modified by this op.
extern const int kSvdfInputActivationStateTensor;

// Output tensor.
extern const int kSvdfOutputTensor;

// TensorflowLite Micro-specific reference implementation for Integer SVDF.
void EvalIntegerSvdfReference(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteEvalTensor* input_tensor,
                              const TfLiteEvalTensor* weights_feature_tensor,
                              const TfLiteEvalTensor* weights_time_tensor,
                              const TfLiteEvalTensor* bias_tensor,
                              const TfLiteSVDFParams* params,
                              TfLiteEvalTensor* activation_state_tensor,
                              TfLiteEvalTensor* output_tensor,
                              const OpData& data);

void EvalFloatSvdfReference(
    TfLiteContext* context, TfLiteNode* node, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* weights_feature,
    const TfLiteEvalTensor* weights_time, const TfLiteEvalTensor* bias,
    const TfLiteSVDFParams* params, int scratch_tensor_index,
    TfLiteEvalTensor* activation_state, TfLiteEvalTensor* output);

TfLiteStatus PrepareSvdf(TfLiteContext* context, TfLiteNode* node);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_SVDF_H_
