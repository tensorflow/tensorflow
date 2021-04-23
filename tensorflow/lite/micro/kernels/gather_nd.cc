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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kParams = 0;
constexpr int kIndices = 1;
constexpr int kOutputTensor = 0;
constexpr int MAX_INDICES_ND = 5;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* params;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kParams, &params));
  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kIndices, &indices));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (params->type) {
    case kTfLiteFloat32:
    case kTfLiteInt8:
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Params of type '%s' are not supported by gather_nd.",
                         TfLiteTypeGetName(params->type));
      return kTfLiteError;
      break;
  }
  switch (indices->type) {
    case kTfLiteInt32:
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Indices of type '%s' are not supported by gather_nd.",
                         TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }

  const int params_rank = NumDimensions(params);
  const int indices_rank = NumDimensions(indices);
  const int indices_nd = SizeOfDimension(indices, indices_rank - 1);
  if (params_rank < 1) {
    TF_LITE_KERNEL_LOG(context, "Params must be at least a vector.");
    return kTfLiteError;
  }
  if (indices_rank < 1) {
    TF_LITE_KERNEL_LOG(context, "Indices must be at least a vector.");
    return kTfLiteError;
  }
  if (indices_nd > params_rank) {
    TF_LITE_KERNEL_LOG(
        context, "Index innermost dimension length must be <= params rank.");
    return kTfLiteError;
  }
  if (indices_nd > MAX_INDICES_ND) {
    TF_LITE_KERNEL_LOG(context,
                       "Index innermost dimension length must not exceed %d.",
                       MAX_INDICES_ND);
    return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = params->type;

  // TFLM gather_nd does not create the output tensor, but it needs to ensure
  // that the output shape is correct. The result shape is
  // indices.shape[:-1] + params.shape[indices.shape[-1]:]
  TfLiteIntArray* output_shape = output->dims;
  int output_index = 0;
  for (int i = 0; i < indices_rank - 1; ++i) {
    output_shape->data[output_index++] = indices->dims->data[i];
  }
  for (int i = indices_nd; i < params_rank; ++i) {
    output_shape->data[output_index++] = params->dims->data[i];
  }
  output_shape->size = output_index;
  return kTfLiteOk;
}

template <typename ParamsT, typename IndicesT>
TfLiteStatus GatherNd(const TfLiteEvalTensor* params,
                      const TfLiteEvalTensor* indices,
                      TfLiteEvalTensor* output) {
  const int indices_dims = indices->dims->size;
  const int indices_nd = indices->dims->data[indices_dims - 1];
  const int params_dims = params->dims->size;
  const IndicesT* index_data = tflite::micro::GetTensorData<IndicesT>(indices);
  const ParamsT* param_data = tflite::micro::GetTensorData<ParamsT>(params);
  ParamsT* output_data = tflite::micro::GetTensorData<ParamsT>(output);

  int n_slices = 1;
  for (int i = 0; i < indices_dims - 1; ++i) {
    n_slices *= indices->dims->data[i];
  }

  // If indices[-1] == params.rank, fetch single elements.
  // If indices[-1] < params.rank, fetch slices.
  int slice_size = 1;
  for (int i = indices_nd; i < params_dims; ++i) {
    slice_size *= params->dims->data[i];
  }

  int remain_flat_size = ElementCount(*params->dims);

  // Number of elements per dimension
  int dims_to_count[MAX_INDICES_ND];
  for (int i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / params->dims->data[i];
    remain_flat_size = dims_to_count[i];
  }

  for (int i = 0; i < n_slices; ++i) {
    int from_pos = 0;
    for (int j = 0; j < indices_nd; ++j) {
      int offset = i * indices_nd + j;
      IndicesT index = index_data[offset];
      from_pos += index * dims_to_count[j];
    }
    std::memcpy(output_data + i * slice_size, param_data + from_pos,
                sizeof(ParamsT) * slice_size);
  }
  return kTfLiteOk;
}

template <typename IndicesT>
TfLiteStatus EvalGatherNd(TfLiteContext* context,
                          const TfLiteEvalTensor* params,
                          const TfLiteEvalTensor* indices,
                          TfLiteEvalTensor* output) {
  switch (params->type) {
    case kTfLiteFloat32:
      return GatherNd<float, IndicesT>(params, indices, output);
      break;
    case kTfLiteInt8:
      return GatherNd<int8_t, IndicesT>(params, indices, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Params type '%s' are not supported by gather_nd.",
                         TfLiteTypeGetName(params->type));
      return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* params =
      tflite::micro::GetEvalInput(context, node, kParams);
  const TfLiteEvalTensor* indices =
      tflite::micro::GetEvalInput(context, node, kIndices);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (indices->type) {
    case kTfLiteInt32:
      return EvalGatherNd<int32_t>(context, params, indices, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Indices of type '%s' are not supported by gather_nd.",
                         TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }
}
}  // namespace

TfLiteRegistration Register_GATHER_ND() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
