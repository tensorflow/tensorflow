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

// This file implements the TensorFlow Lite's broadcast gradient argument
// operator.

#include <algorithm>
#include <array>
#include <cmath>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace {

static const int kInputOneTensor = 0;
static const int kInputTwoTensor = 1;
static const int kOutputOneTensor = 0;
static const int kOutputTwoTensor = 1;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check inputs and output.
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  const TfLiteTensor* input1 = GetInput(context, node, kInputOneTensor);
  TF_LITE_ENSURE(context, input1 != nullptr);
  const RuntimeShape input1_shape = GetTensorShape(input1);
  TF_LITE_ENSURE(context,
                 input1->type == kTfLiteInt32 || input1->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, input1_shape.DimensionsCount(), 1);

  const TfLiteTensor* input2 = GetInput(context, node, kInputTwoTensor);
  TF_LITE_ENSURE(context, input2 != nullptr);
  const RuntimeShape input2_shape = GetTensorShape(input2);
  TF_LITE_ENSURE_TYPES_EQ(context, input2->type, input1->type);
  TF_LITE_ENSURE_EQ(context, input2_shape.DimensionsCount(), 1);

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
  TfLiteTensor* output1 = GetOutput(context, node, kOutputOneTensor);
  TF_LITE_ENSURE(context, output1 != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, output1->type, input1->type);
  TfLiteTensor* output2 = GetOutput(context, node, kOutputTwoTensor);
  TF_LITE_ENSURE(context, output2 != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, output2->type, input1->type);
  SetTensorToDynamic(output1);
  SetTensorToDynamic(output2);
  return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1 = GetInput(context, node, kInputOneTensor);
  TF_LITE_ENSURE(context, input1 != nullptr);
  const RuntimeShape input1_shape = GetTensorShape(input1);

  const TfLiteTensor* input2 = GetInput(context, node, kInputTwoTensor);
  TF_LITE_ENSURE(context, input2 != nullptr);
  const RuntimeShape input2_shape = GetTensorShape(input2);

  TfLiteTensor* output1 = GetOutput(context, node, kOutputOneTensor);
  TF_LITE_ENSURE(context, output1 != nullptr);
  TfLiteTensor* output2 = GetOutput(context, node, kOutputTwoTensor);
  TF_LITE_ENSURE(context, output2 != nullptr);

  std::vector<int64_t> input1_vec;
  std::vector<int64_t> input2_vec;
  if (input1->type == kTfLiteInt32) {
    input1_vec = std::vector<int64_t>(input1->data.i32,
                                      input1->data.i32 + input1_shape.Dims(0));
  } else {
    input1_vec = std::vector<int64_t>(input1->data.i64,
                                      input1->data.i64 + input1_shape.Dims(0));
  }
  if (input2->type == kTfLiteInt32) {
    input2_vec = std::vector<int64_t>(input2->data.i32,
                                      input2->data.i32 + input2_shape.Dims(0));
  } else {
    input2_vec = std::vector<int64_t>(input2->data.i64,
                                      input2->data.i64 + input2_shape.Dims(0));
  }

  if (input1_vec == input2_vec) {
    // All equals.
    TfLiteIntArray* output1_shape = TfLiteIntArrayCreate(1);
    output1_shape->data[0] = 0;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output1, output1_shape));

    TfLiteIntArray* output2_shape = TfLiteIntArrayCreate(1);
    output2_shape->data[0] = 0;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output2, output2_shape));
    return kTfLiteOk;
  }

  size_t largest_rank = std::max(input1_vec.size(), input2_vec.size());

  // Reverse all the shapes for convenience
  // After the reverse, 0-th is the inner-most dimension.
  std::vector<int64_t> copy[2];
  copy[0] = std::vector<int64_t>(input1_vec.rbegin(), input1_vec.rend());
  copy[1] = std::vector<int64_t>(input2_vec.rbegin(), input2_vec.rend());

  // 1-extend and align all vectors.
  for (int i = 0; i < 2; ++i) {
    if (copy[i].size() < largest_rank) {
      copy[i].resize(largest_rank, 1);
    }
  }
  // Going through each dimension starting from the inner-most
  // dimension, compares dimension of x and y. They are compatible if
  // they are equal or either is 1.

  // indices of j-th component of each input.
  std::array<bool, 2> prev_is_one = {false, false};
  std::array<bool, 2> current_is_one = {false, false};
  bool set_one = false;
  // indices of gradient reduction of each input.
  std::vector<int64_t> grad_reduce_idx[2];

  for (int j = 0; j < largest_rank; ++j) {
    int output_dim = -1;
    int output_dim_set = false;
    bool none_is_one = true;
    // Find which indices are 1.
    for (int i = 0; i < 2; ++i) {
      // Keep track of which indices are 1.
      if (copy[i][j] == 1) {
        current_is_one[i] = true;
        none_is_one = false;
      } else {
        current_is_one[i] = false;
        if (!output_dim_set || copy[i][j] == output_dim) {
          output_dim = copy[i][j];
          output_dim_set = true;
        } else {
          // Not broadcastable shapes.
          return kTfLiteError;
        }
      }
    }
    // All dimensions are 1.
    if (!output_dim_set) {
      for (int i = 0; i < 2; ++i) {
        grad_reduce_idx[i].push_back(largest_rank - 1 - j);
      }
      continue;
    } else if (current_is_one == prev_is_one && set_one) {
      // It is a run of the same broadcasting case as last time.
      // We can reshape the input so that fewer dimensions
      // are involved in the intermediate computation.
      for (int i = 0; i < 2; ++i) {
        if (current_is_one[i] && !none_is_one) {
          grad_reduce_idx[i].push_back(largest_rank - 1 - j);
        }
      }
    } else {
      for (int i = 0; i < 2; ++i) {
        if (current_is_one[i] && !none_is_one) {
          grad_reduce_idx[i].push_back(largest_rank - 1 - j);
        }
      }
    }
    set_one = true;
    for (int i = 0; i < 2; ++i) {
      prev_is_one[i] = current_is_one[i];
    }
  }
  for (int i = 0; i < 2; ++i) {
    std::reverse(grad_reduce_idx[i].begin(), grad_reduce_idx[i].end());
  }
  TfLiteIntArray* output1_shape = TfLiteIntArrayCreate(1);
  output1_shape->data[0] = grad_reduce_idx[0].size();
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output1, output1_shape));
  if (output1->type == kTfLiteInt32) {
    for (int i = 0; i < grad_reduce_idx[0].size(); ++i) {
      output1->data.i32[i] = grad_reduce_idx[0][i];
    }
  } else if (output1->type == kTfLiteInt64) {
    for (int i = 0; i < grad_reduce_idx[0].size(); ++i) {
      output1->data.i64[i] = grad_reduce_idx[0][i];
    }
  }

  TfLiteIntArray* output2_shape = TfLiteIntArrayCreate(1);
  output2_shape->data[0] = grad_reduce_idx[1].size();
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output2, output2_shape));
  if (output2->type == kTfLiteInt32) {
    for (int i = 0; i < grad_reduce_idx[1].size(); ++i) {
      output2->data.i32[i] = grad_reduce_idx[1][i];
    }
  } else if (output2->type == kTfLiteInt64) {
    for (int i = 0; i < grad_reduce_idx[1].size(); ++i) {
      output2->data.i64[i] = grad_reduce_idx[1][i];
    }
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration* Register_BROADCAST_GRADIENT_ARGS() {
  static TfLiteRegistration reg = {/*init=*/nullptr,
                                   /*free=*/nullptr,
                                   /*prepare=*/Prepare,
                                   /*invoke=*/Invoke};
  return &reg;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
