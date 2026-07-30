/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/ynnpack/attention_model.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ynnpack {

TfLiteStatus RuntimeBmmPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size >= 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor* input_a = &context->tensors[node->inputs->data[0]];
  const TfLiteTensor* input_b = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  int rank_a = input_a->dims->size;
  int rank_b = input_b->dims->size;
  TF_LITE_ENSURE(context, rank_a >= 2);
  TF_LITE_ENSURE(context, rank_b >= 2);

  bool is_src = false;
  if (node->custom_initial_data != nullptr) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(node->custom_initial_data),
            node->custom_initial_data_size)
            .AsMap();
    if (!flexbuffer_map["is_src"].IsNull()) {
      is_src = flexbuffer_map["is_src"].AsBool();
    }
  }

  if (!is_src) {
    if (input_a->dims->data[rank_a - 1] == input_b->dims->data[rank_b - 2] &&
        input_a->dims->data[rank_a - 1] != input_b->dims->data[rank_b - 1]) {
      is_src = true;
    }
  }

  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(rank_a);
  for (int i = 0; i < rank_a - 2; ++i) {
    output_dims->data[i] = input_a->dims->data[i];
  }

  if (!is_src) {
    output_dims->data[rank_a - 2] = input_a->dims->data[rank_a - 2];
    output_dims->data[rank_a - 1] = input_b->dims->data[rank_b - 2];
  } else {
    output_dims->data[rank_a - 2] = input_a->dims->data[rank_a - 2];
    output_dims->data[rank_a - 1] = input_b->dims->data[rank_b - 2];
  }

  return context->ResizeTensor(context, output, output_dims);
}

TfLiteStatus RuntimeBmmEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_a = &context->tensors[node->inputs->data[0]];
  const TfLiteTensor* input_b = &context->tensors[node->inputs->data[1]];
  const TfLiteTensor* s_active_tensor =
      &context->tensors[node->inputs->data[2]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  int s_active = s_active_tensor->data.i32[0];

  bool is_src = false;
  if (node->custom_initial_data != nullptr) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(node->custom_initial_data),
            node->custom_initial_data_size)
            .AsMap();
    if (!flexbuffer_map["is_src"].IsNull()) {
      is_src = flexbuffer_map["is_src"].AsBool();
    }
  }

  int rank_a = input_a->dims->size;
  int rank_b = input_b->dims->size;

  TF_LITE_ENSURE_EQ(context, rank_a, 4);
  TF_LITE_ENSURE_EQ(context, rank_b, 4);

  int B = input_a->dims->data[0];
  int N = input_a->dims->data[1];

  const float* a_data = input_a->data.f;
  const float* b_data = input_b->data.f;
  float* o_data = output->data.f;

  memset(o_data, 0, output->bytes);

  if (!is_src) {
    int T = input_a->dims->data[2];
    int H = input_a->dims->data[3];
    int S = input_b->dims->data[2];

    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int t = 0; t < T; ++t) {
          for (int s = 0; s < s_active; ++s) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
              int a_idx = ((b * N + n) * T + t) * H + h;
              int b_idx = ((b * N + n) * S + s) * H + h;
              sum += a_data[a_idx] * b_data[b_idx];
            }
            int o_idx = ((b * N + n) * T + t) * S + s;
            o_data[o_idx] = sum;
          }
        }
      }
    }
  } else {
    int T = input_a->dims->data[2];
    int S = input_a->dims->data[3];
    int H = input_b->dims->data[2];

    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int t = 0; t < T; ++t) {
          for (int h = 0; h < H; ++h) {
            float sum = 0.0f;
            for (int s = 0; s < s_active; ++s) {
              int a_idx = ((b * N + n) * T + t) * S + s;
              int b_idx = ((b * N + n) * H + h) * S + s;
              sum += a_data[a_idx] * b_data[b_idx];
            }
            int o_idx = ((b * N + n) * T + t) * H + h;
            o_data[o_idx] = sum;
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_RuntimeBmm() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/RuntimeBmmPrepare,
      /*.invoke=*/RuntimeBmmEval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
      /*.custom_name=*/"odml.runtime_bmm",
      /*.version=*/1,
  };
  return &reg;
}

AttentionModel::AttentionModel(
    int b, int t, int s, int h, int n, float scale, bool transpose_io,
    bool use_delegate, const TfLiteYNNPackDelegateOptions& delegate_options) {
  std::vector<int> query_shape = transpose_io ? std::vector<int>{b, t, n, h}
                                              : std::vector<int>{b, n, t, h};
  std::vector<int> key_shape = transpose_io ? std::vector<int>{b, s, n, h}
                                            : std::vector<int>{b, n, s, h};
  // V shape has H and S swapped for runtime_bmm compatibility.
  std::vector<int> value_shape = transpose_io ? std::vector<int>{b, h, n, s}
                                              : std::vector<int>{b, n, h, s};
  // Mask is broadcasted across heads (n), but not batch (b).
  std::vector<int> mask_shape = transpose_io ? std::vector<int>{b, t, 1, s}
                                             : std::vector<int>{b, 1, t, s};

  query_id_ = AddInput({TensorType_FLOAT32, query_shape});
  key_id_ = AddInput({TensorType_FLOAT32, key_shape});
  value_id_ = AddInput({TensorType_FLOAT32, value_shape});
  runtime_bmm_params_id_ = AddInput({TensorType_INT32, {1}});
  mask_id_ = AddInput({TensorType_FLOAT32, mask_shape});

  int current_query = query_id_;
  int current_key = key_id_;
  int current_value = value_id_;
  int current_mask = mask_id_;

  if (transpose_io) {
    int perm_id = AddConstInput<int32_t>(TensorType_INT32, {0, 2, 1, 3}, {4});

    int transposed_query = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union(),
                 {current_query, perm_id}, {transposed_query});
    current_query = transposed_query;

    int transposed_key = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union(),
                 {current_key, perm_id}, {transposed_key});
    current_key = transposed_key;

    int transposed_value = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union(),
                 {current_value, perm_id}, {transposed_value});
    current_value = transposed_value;

    int transposed_mask = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union(),
                 {current_mask, perm_id}, {transposed_mask});
    current_mask = transposed_mask;
  }

  // BMM1: Q @ K^T (is_src = false)
  int scores_id = AddIntermediate(TensorType_FLOAT32, {}, {});
  {
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Bool("is_src", false); });
    fbb.Finish();
    AddCustomOp("odml.runtime_bmm", fbb.GetBuffer(), Register_RuntimeBmm,
                {current_query, current_key, runtime_bmm_params_id_},
                {scores_id});
  }

  // Add mask
  int masked_scores_id = AddIntermediate(TensorType_FLOAT32, {}, {});
  AddBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
               CreateAddOptions(builder_).Union(), {scores_id, current_mask},
               {masked_scores_id});

  int probs_id = AddIntermediate(TensorType_FLOAT32, {}, {});
  AddBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
               CreateSoftmaxOptions(builder_, scale).Union(),
               {masked_scores_id}, {probs_id});

  // BMM2: Probs @ V (is_src = true)
  int output_id;
  if (transpose_io) {
    output_id = AddIntermediate(TensorType_FLOAT32, {}, {});
  } else {
    output_id = AddOutput({TensorType_FLOAT32, {}});
  }
  {
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Bool("is_src", true); });
    fbb.Finish();
    AddCustomOp("odml.runtime_bmm", fbb.GetBuffer(), Register_RuntimeBmm,
                {probs_id, current_value, runtime_bmm_params_id_}, {output_id});
  }

  int current_output = output_id;

  if (transpose_io) {
    int perm_id = AddConstInput<int32_t>(TensorType_INT32, {0, 2, 1, 3}, {4});
    int transposed_output = AddOutput({TensorType_FLOAT32, {}});
    AddBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union(),
                 {current_output, perm_id}, {transposed_output});
    current_output = transposed_output;
  }

  output_id_ = current_output;

  BuildInterpreter({query_shape, key_shape, value_shape, {1}, mask_shape}, -1,
                   false,
                   /*apply_delegate=*/false, /*allocate_and_delegate=*/false);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "Failed to allocate tensors\n");
  }

  if (use_delegate) {
    auto delegate = TfLiteYNNPackDelegateCreateUnique(&delegate_options);
    SetDelegate(std::move(delegate));
    if (ApplyDelegate() != kTfLiteOk) {
      fprintf(stderr, "Failed to apply delegate\n");
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      fprintf(stderr, "Failed to allocate tensors after delegation\n");
    }
  }
}

}  // namespace ynnpack
}  // namespace tflite
