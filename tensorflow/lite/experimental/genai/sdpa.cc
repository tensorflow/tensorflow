/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace llm {

static const int kQueryTensor = 0;
static const int kKeyTensor = 1;
static const int kValueTensor = 2;
static const int kAttentionMaskTensor = 3;
static const int kOutputTensor = 0;

static const int kNumTempTensors = 10;
static const int kTransposeQueryTempTensorIndex = 0;
static const int kTransposeKeyTempTensorIndex = 1;
static const int kMatMul1TempTensorIndex = 2;
static const int kAddTempTensorIndex = 3;
static const int kTransposeValueTempTensorIndex = 4;
static const int kMatMul2TempTensorIndex = 5;
static const int kReshape1TempTensorIndex = 6;
static const int kReshape2TempTensorIndex = 7;
static const int kBroadcastKTempTensorIndex = 8;
static const int kBroadcastVTempTensorIndex = 9;

struct OpData {
  float scale;
  int scratch_tensor_index;
};

void* SDPAInit(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* op_data = new OpData();
  op_data->scale = 0.0f;
  context->AddTensors(context, kNumTempTensors, &op_data->scratch_tensor_index);
  return op_data;
}

TfLiteStatus SDPAPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* q_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kQueryTensor, &q_tensor));
  const TfLiteTensor* k_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeyTensor, &k_tensor));
  const TfLiteTensor* v_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueTensor, &v_tensor));
  const TfLiteTensor* mask_tensor;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kAttentionMaskTensor, &mask_tensor));
  TF_LITE_ENSURE_EQ(context, NumDimensions(q_tensor), NumDimensions(k_tensor));
  TF_LITE_ENSURE_EQ(context, NumDimensions(k_tensor), NumDimensions(v_tensor));
  TF_LITE_ENSURE_EQ(context, NumDimensions(v_tensor),
                    NumDimensions(mask_tensor));
  TF_LITE_ENSURE_EQ(context, NumDimensions(mask_tensor), 4);

  // Get custom op params
  const uint8_t* buffer =
      reinterpret_cast<const uint8_t*>(node->custom_initial_data);
  const size_t length = node->custom_initial_data_size;
  auto flexbuffer_map = flexbuffers::GetRoot(buffer, length).AsMap();
  float scale = flexbuffer_map["scale"].AsFloat();
  op_data->scale = scale > 0.0f ? scale : 0.0f;

  // If scale is not set, use sqrt(q_tensor->dims->data[3])
  if (op_data->scale == 0.0f)
    op_data->scale = 1 / sqrt(q_tensor->dims->data[3]);

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kNumTempTensors);
  bool mqa = k_tensor->dims->data[2] == 1;

  // Temp tensor for Transposed Q;
  {
    node->temporaries->data[kTransposeQueryTempTensorIndex] =
        op_data->scratch_tensor_index + kTransposeQueryTempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node,
                                       /*index=*/kTransposeQueryTempTensorIndex,
                                       &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    for (int i = 0; i < 4; ++i) {
      scratch_buffer_size->data[i] = q_tensor->dims->data[i];
    }
    // Swap middle two dimensions.
    scratch_buffer_size->data[1] = q_tensor->dims->data[2];
    scratch_buffer_size->data[2] = q_tensor->dims->data[1];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Transposed K;
  {
    node->temporaries->data[kTransposeKeyTempTensorIndex] =
        op_data->scratch_tensor_index + kTransposeKeyTempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node,
                                       /*index=*/kTransposeKeyTempTensorIndex,
                                       &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    for (int i = 0; i < 4; ++i) {
      scratch_buffer_size->data[i] = k_tensor->dims->data[i];
    }
    // Swap to middle two dimensions.
    scratch_buffer_size->data[1] = k_tensor->dims->data[2];
    scratch_buffer_size->data[2] = k_tensor->dims->data[1];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  TfLiteIntArray* add_broadcast_shape = nullptr;
  // Temp tensor for Matmul1 output;
  {
    node->temporaries->data[kMatMul1TempTensorIndex] =
        op_data->scratch_tensor_index + kMatMul1TempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node,
                         /*index=*/kMatMul1TempTensorIndex, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    // mha/gqa: [permute_q[0], permute_q[1], permute_q[2], permute_k[2]]
    int matmul_out_shape[4] = {q_tensor->dims->data[0], q_tensor->dims->data[2],
                               q_tensor->dims->data[1],
                               k_tensor->dims->data[1]};
    for (int i = 0; i < 4; ++i) {
      scratch_buffer_size->data[i] = matmul_out_shape[i];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
    // get dims from attention_mask, matmul1_out for add broadcast
    CalculateShapeForBroadcast(context, mask_tensor, scratch_buffer,
                               &add_broadcast_shape);
  }

  // Temp tensor for add output;
  int add_out_shape[4];
  {
    node->temporaries->data[kAddTempTensorIndex] =
        op_data->scratch_tensor_index + kAddTempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                                /*index=*/kAddTempTensorIndex,
                                                &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = add_broadcast_shape;
    for (int i = 0; i < 4; ++i) {
      add_out_shape[i] = scratch_buffer_size->data[i];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Transposed V;
  {
    node->temporaries->data[kTransposeValueTempTensorIndex] =
        op_data->scratch_tensor_index + kTransposeValueTempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node,
                                       /*index=*/kTransposeValueTempTensorIndex,
                                       &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    // Swap to {0, 2, 3, 1} dimensions.
    scratch_buffer_size->data[0] = v_tensor->dims->data[0];
    scratch_buffer_size->data[1] = v_tensor->dims->data[2];
    scratch_buffer_size->data[2] = v_tensor->dims->data[3];
    scratch_buffer_size->data[3] = v_tensor->dims->data[1];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Matmul2 output;
  {
    node->temporaries->data[kMatMul2TempTensorIndex] =
        op_data->scratch_tensor_index + kMatMul2TempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node,
                         /*index=*/kMatMul2TempTensorIndex, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);
    // logits_out_shape = add_out_shape
    // mha/gqa: [logits_out[0], logits_out[1], logits_out[2], permute_v[2]]
    scratch_buffer_size->data[0] = add_out_shape[0];
    scratch_buffer_size->data[1] = add_out_shape[1];
    scratch_buffer_size->data[2] = add_out_shape[2];
    scratch_buffer_size->data[3] = v_tensor->dims->data[3];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Reshape K / Transpose Q;
  {
    node->temporaries->data[kReshape1TempTensorIndex] =
        op_data->scratch_tensor_index + kReshape1TempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node,
                         /*index=*/kReshape1TempTensorIndex, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size;
    if (mqa)
      scratch_buffer_size = TfLiteIntArrayCreate(2);
    else
      scratch_buffer_size = TfLiteIntArrayCreate(4);
    if (mqa) {
      scratch_buffer_size->data[0] = k_tensor->dims->data[1];
      scratch_buffer_size->data[1] = k_tensor->dims->data[3];
    } else {
      scratch_buffer_size->data[0] = q_tensor->dims->data[0];
      scratch_buffer_size->data[1] = q_tensor->dims->data[2];
      scratch_buffer_size->data[2] = q_tensor->dims->data[3];
      scratch_buffer_size->data[3] = q_tensor->dims->data[1];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Reshape V / Add_out (softmax_out);
  {
    node->temporaries->data[kReshape2TempTensorIndex] =
        op_data->scratch_tensor_index + kReshape2TempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node,
                         /*index=*/kReshape2TempTensorIndex, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size;
    if (mqa)
      scratch_buffer_size = TfLiteIntArrayCreate(2);
    else
      scratch_buffer_size = TfLiteIntArrayCreate(4);
    if (mqa) {
      scratch_buffer_size->data[0] = v_tensor->dims->data[3];
      scratch_buffer_size->data[1] = v_tensor->dims->data[1];
    } else {
      scratch_buffer_size->data[0] = add_out_shape[0];
      scratch_buffer_size->data[1] = add_out_shape[1];
      scratch_buffer_size->data[2] = add_out_shape[3];
      scratch_buffer_size->data[3] = add_out_shape[2];
    }

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Broadcast K
  {
    node->temporaries->data[kBroadcastKTempTensorIndex] =
        op_data->scratch_tensor_index + kBroadcastKTempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node,
                                       /*index=*/kBroadcastKTempTensorIndex,
                                       &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);

    scratch_buffer_size->data[0] = k_tensor->dims->data[0];
    scratch_buffer_size->data[1] = q_tensor->dims->data[2];  // num_heads
    scratch_buffer_size->data[2] = k_tensor->dims->data[1];
    scratch_buffer_size->data[3] = k_tensor->dims->data[3];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // Temp tensor for Broadcast V
  {
    node->temporaries->data[kBroadcastVTempTensorIndex] =
        op_data->scratch_tensor_index + kBroadcastVTempTensorIndex;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node,
                                       /*index=*/kBroadcastVTempTensorIndex,
                                       &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(4);

    scratch_buffer_size->data[0] = v_tensor->dims->data[0];
    scratch_buffer_size->data[1] = q_tensor->dims->data[2];  // num_heads
    scratch_buffer_size->data[2] = v_tensor->dims->data[3];
    scratch_buffer_size->data[3] = v_tensor->dims->data[1];

    scratch_buffer->type = kTfLiteFloat32;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  return kTfLiteOk;
}

void SDPAFree(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus SDPAEval(TfLiteContext* context, TfLiteNode* node) {
  /*
  Simple implementation of Scaled Dot Product Attention.
  Takes query_proj, key_proj, value_proj, mask tensors as inputs, and
  outputs the attention result.

  Notes:
  Scale is computed using 1/sqrt(head_dim),
  head_dim = q[-1] = embedding_dim // num_q_heads
  Only support for FLOAT32 inputs for now.
  Only support static tensors for now (k/v[1] = max sequence length)
  */

  const TfLiteTensor* query_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kQueryTensor, &query_tensor));
  auto query_shape = GetTensorShape(query_tensor);
  auto query_data = GetTensorData<float>(query_tensor);
  const TfLiteTensor* key_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kKeyTensor, &key_tensor));
  auto key_shape = GetTensorShape(key_tensor);
  auto key_data = GetTensorData<float>(key_tensor);
  const TfLiteTensor* value_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueTensor, &value_tensor));
  auto value_shape = GetTensorShape(value_tensor);
  auto value_data = GetTensorData<float>(value_tensor);
  const TfLiteTensor* attention_mask_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAttentionMaskTensor,
                                          &attention_mask_tensor));
  auto attention_mask_shape = GetTensorShape(attention_mask_tensor);
  auto attention_mask_data = GetTensorData<float>(attention_mask_tensor);
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputTensor, &output_tensor));
  auto output_shape = GetTensorShape(output_tensor);
  auto output_data = GetTensorData<float>(output_tensor);

  // temporaries
  TfLiteTensor* transpose_q_out_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, /*index=*/kTransposeQueryTempTensorIndex,
                       &transpose_q_out_tensor));
  auto transpose_q_out_shape = GetTensorShape(transpose_q_out_tensor);
  auto transpose_q_out_data = GetTensorData<float>(transpose_q_out_tensor);
  TfLiteTensor* transpose_k_out_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, /*index=*/kTransposeKeyTempTensorIndex,
                       &transpose_k_out_tensor));
  auto transpose_k_out_shape = GetTensorShape(transpose_k_out_tensor);
  auto transpose_k_out_data = GetTensorData<float>(transpose_k_out_tensor);
  TfLiteTensor* matmul1_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                              /*index=*/kMatMul1TempTensorIndex,
                                              &matmul1_out_tensor));
  auto matmul1_out_shape = GetTensorShape(matmul1_out_tensor);
  auto matmul1_out_data = GetTensorData<float>(matmul1_out_tensor);
  TfLiteTensor* add_out_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/kAddTempTensorIndex,
                                &add_out_tensor));
  auto add_out_shape = GetTensorShape(add_out_tensor);
  auto add_out_data = GetTensorData<float>(add_out_tensor);
  TfLiteTensor* transpose_v_out_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, /*index=*/kTransposeValueTempTensorIndex,
                       &transpose_v_out_tensor));
  auto transpose_v_out_shape = GetTensorShape(transpose_v_out_tensor);
  auto transpose_v_out_data = GetTensorData<float>(transpose_v_out_tensor);
  TfLiteTensor* matmul2_out_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                              /*index=*/kMatMul2TempTensorIndex,
                                              &matmul2_out_tensor));
  auto matmul2_out_shape = GetTensorShape(matmul2_out_tensor);
  auto matmul2_out_data = GetTensorData<float>(matmul2_out_tensor);
  TfLiteTensor* reshape_k_or_q_out_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, /*index=*/kReshape1TempTensorIndex,
                       &reshape_k_or_q_out_tensor));
  auto reshape_k_or_q_out_shape = GetTensorShape(reshape_k_or_q_out_tensor);
  auto reshape_k_or_q_out_data =
      GetTensorData<float>(reshape_k_or_q_out_tensor);
  TfLiteTensor* reshape_v_or_add_out_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, /*index=*/kReshape2TempTensorIndex,
                       &reshape_v_or_add_out_tensor));
  auto reshape_v_or_add_out_shape = GetTensorShape(reshape_v_or_add_out_tensor);
  auto reshape_v_or_add_out_data =
      GetTensorData<float>(reshape_v_or_add_out_tensor);
  TfLiteTensor* broadcast_k_out_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, /*index=*/kBroadcastKTempTensorIndex,
                       &broadcast_k_out_tensor));
  auto broadcast_k_out_shape = GetTensorShape(broadcast_k_out_tensor);
  auto broadcast_k_out_data = GetTensorData<float>(broadcast_k_out_tensor);
  TfLiteTensor* broadcast_v_out_tensor;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, /*index=*/kBroadcastVTempTensorIndex,
                       &broadcast_v_out_tensor));
  auto broadcast_v_out_shape = GetTensorShape(broadcast_v_out_tensor);
  auto broadcast_v_out_data = GetTensorData<float>(broadcast_v_out_tensor);

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  bool mqa = key_tensor->dims->data[2] == 1;
  bool gqa = !mqa && (key_tensor->dims->data[2] != query_tensor->dims->data[2]);

  // scale * q
  float scale = op_data->scale;
  int flat_size = query_shape.FlatSize();
  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();
  for (int i = 0; i < flat_size; ++i) {
    query_tensor->data.f[i] = ActivationFunctionWithMinMax(
        query_tensor->data.f[i] * scale, output_min, output_max);
  }

  // permute q {0, 2, 1, 3}
  tflite::TransposeParams transpose_q_params;
  transpose_q_params.perm_count = 4;
  transpose_q_params.perm[0] = 0;
  transpose_q_params.perm[1] = 2;
  transpose_q_params.perm[2] = 1;
  transpose_q_params.perm[3] = 3;
  reference_ops::Transpose(transpose_q_params, query_shape, query_data,
                           transpose_q_out_shape, transpose_q_out_data);

  // permute k {0, 2, 1, 3}
  tflite::TransposeParams transpose_k_params;
  transpose_k_params.perm_count = 4;
  transpose_k_params.perm[0] = 0;
  transpose_k_params.perm[1] = 2;
  transpose_k_params.perm[2] = 1;
  transpose_k_params.perm[3] = 3;
  reference_ops::Transpose(transpose_k_params, key_shape, key_data,
                           transpose_k_out_shape, transpose_k_out_data);

  // broadcast k to match num_heads
  // broadcasting similar to torch.repeat_interleave
  if (gqa) {
    float* transpose_k_ptr = transpose_k_out_data;
    float* broadcast_k_ptr = broadcast_k_out_data;
    int num_elements =
        transpose_k_out_shape.Dims(2) * transpose_k_out_shape.Dims(3);
    int num_repeat =
        broadcast_k_out_shape.Dims(1) / transpose_k_out_shape.Dims(1);
    for (int i = 0; i < transpose_k_out_shape.Dims(0); ++i) {
      for (int j = 0; j < transpose_k_out_shape.Dims(1); ++j) {
        for (int k = 0; k < num_repeat; ++k) {
          memcpy(broadcast_k_ptr, transpose_k_ptr,
                 num_elements * sizeof(float));
          broadcast_k_ptr += num_elements;
        }
        transpose_k_ptr += num_elements;
      }
    }
  }

  // reshape k for MQA, or transpose q for MHA
  if (mqa) {
    TF_LITE_ENSURE_EQ(context, transpose_k_out_tensor->bytes,
                      reshape_k_or_q_out_tensor->bytes);
    memcpy(reshape_k_or_q_out_tensor->data.data,
           transpose_k_out_tensor->data.data, transpose_k_out_tensor->bytes);
  } else {
    // permute q2 {0, 1, 3, 2}
    tflite::TransposeParams transpose_q2_params;
    transpose_q2_params.perm_count = 4;
    transpose_q2_params.perm[0] = 0;
    transpose_q2_params.perm[1] = 1;
    transpose_q2_params.perm[2] = 3;
    transpose_q2_params.perm[3] = 2;
    reference_ops::Transpose(transpose_q2_params, transpose_q_out_shape,
                             transpose_q_out_data, reshape_k_or_q_out_shape,
                             reshape_k_or_q_out_data);
  }

  // mqa FC (q, squeezed_k)
  // mha BMM(q, k) transpose_b = true
  if (mqa) {
    tflite::FullyConnectedParams fc_params;
    fc_params.float_activation_min = output_min;
    fc_params.float_activation_max = output_max;
    reference_ops::FullyConnected(
        fc_params, transpose_q_out_shape, transpose_q_out_data,
        reshape_k_or_q_out_shape, reshape_k_or_q_out_data, RuntimeShape(),
        nullptr, matmul1_out_shape, matmul1_out_data);
  } else if (gqa) {
    // pass rhs first (this is why we transpose q above)
    reference_ops::BatchMatMul(
        broadcast_k_out_shape, broadcast_k_out_data, reshape_k_or_q_out_shape,
        reshape_k_or_q_out_data, matmul1_out_shape, matmul1_out_data);
  } else {
    reference_ops::BatchMatMul(
        transpose_k_out_shape, transpose_k_out_data, reshape_k_or_q_out_shape,
        reshape_k_or_q_out_data, matmul1_out_shape, matmul1_out_data);
  }

  // add matmul_out + mask
  tflite::ArithmeticParams add_params;
  SetActivationParams(output_min, output_max, &add_params);
  reference_ops::BroadcastAdd6DSlow(
      add_params, attention_mask_shape, attention_mask_data, matmul1_out_shape,
      matmul1_out_data, add_out_shape, add_out_data);

  // softmax, can do in-place
  tflite::SoftmaxParams softmax_params;
  softmax_params.beta = 1.0f;
  reference_ops::Softmax(softmax_params, add_out_shape, add_out_data,
                         add_out_shape, add_out_data);

  // permute v {0, 2, 3, 1}
  tflite::TransposeParams transpose_v_params;
  transpose_v_params.perm_count = 4;
  transpose_v_params.perm[0] = 0;
  transpose_v_params.perm[1] = 2;
  transpose_v_params.perm[2] = 3;
  transpose_v_params.perm[3] = 1;
  reference_ops::Transpose(transpose_v_params, value_shape, value_data,
                           transpose_v_out_shape, transpose_v_out_data);

  // broadcast v to match num_heads
  // broadcasting similar to torch.repeat_interleave
  if (gqa) {
    float* transpose_v_ptr = transpose_v_out_data;
    float* broadcast_v_ptr = broadcast_v_out_data;
    int num_elements =
        transpose_v_out_shape.Dims(2) * transpose_v_out_shape.Dims(3);
    int num_repeat =
        broadcast_v_out_shape.Dims(1) / transpose_v_out_shape.Dims(1);
    for (int i = 0; i < transpose_v_out_shape.Dims(0); ++i) {
      for (int j = 0; j < transpose_v_out_shape.Dims(1); ++j) {
        for (int k = 0; k < num_repeat; ++k) {
          memcpy(broadcast_v_ptr, transpose_v_ptr,
                 num_elements * sizeof(float));
          broadcast_v_ptr += num_elements;
        }
        transpose_v_ptr += num_elements;
      }
    }
  }

  // reshape v for MQA, or add_out (softmax_out)
  if (mqa) {
    TF_LITE_ENSURE_EQ(context, transpose_v_out_tensor->bytes,
                      reshape_v_or_add_out_tensor->bytes);
    memcpy(reshape_v_or_add_out_tensor->data.data,
           transpose_v_out_tensor->data.data, transpose_v_out_tensor->bytes);
  } else {
    // permute softmax_out {0, 1, 3, 2}
    tflite::TransposeParams transpose_softmax_out_params;
    transpose_softmax_out_params.perm_count = 4;
    transpose_softmax_out_params.perm[0] = 0;
    transpose_softmax_out_params.perm[1] = 1;
    transpose_softmax_out_params.perm[2] = 3;
    transpose_softmax_out_params.perm[3] = 2;
    reference_ops::Transpose(transpose_softmax_out_params, add_out_shape,
                             add_out_data, reshape_v_or_add_out_shape,
                             reshape_v_or_add_out_data);
  }

  // mqa FC (softmax_out, squeezed_v)
  // mha BMM(softmax_out, v) transpose_b = true
  if (mqa) {
    tflite::FullyConnectedParams fc_params;
    fc_params.float_activation_min = output_min;
    fc_params.float_activation_max = output_max;
    reference_ops::FullyConnected(fc_params, add_out_shape, add_out_data,
                                  reshape_v_or_add_out_shape,
                                  reshape_v_or_add_out_data, RuntimeShape(),
                                  nullptr, matmul2_out_shape, matmul2_out_data);
  } else if (gqa) {
    // pass rhs first (this is why we transpose add_out above)
    reference_ops::BatchMatMul(
        broadcast_v_out_shape, broadcast_v_out_data, reshape_v_or_add_out_shape,
        reshape_v_or_add_out_data, matmul2_out_shape, matmul2_out_data);
  } else {
    reference_ops::BatchMatMul(
        transpose_v_out_shape, transpose_v_out_data, reshape_v_or_add_out_shape,
        reshape_v_or_add_out_data, matmul2_out_shape, matmul2_out_data);
  }

  // permute out {0, 2, 1, 3}
  tflite::TransposeParams transpose_out_params;
  transpose_out_params.perm_count = 4;
  transpose_out_params.perm[0] = 0;
  transpose_out_params.perm[1] = 2;
  transpose_out_params.perm[2] = 1;
  transpose_out_params.perm[3] = 3;
  reference_ops::Transpose(transpose_out_params, matmul2_out_shape,
                           matmul2_out_data, output_shape, output_data);

  return kTfLiteOk;
}

}  // namespace llm

TfLiteRegistration* Register_SDPA() {
  static TfLiteRegistration r = {llm::SDPAInit, llm::SDPAFree, llm::SDPAPrepare,
                                 llm::SDPAEval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
