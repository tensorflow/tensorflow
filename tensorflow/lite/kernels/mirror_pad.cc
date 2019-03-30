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

#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace mirror_pad {
namespace {

// Nil value for paddingMode/offset.
const int kUnsetOffset = -1;

// Wrapper for data used by the op.
struct OpData {
  // Holds computed value (memoized value) of an internal fill state of a
  // subarray.
  // State is (Dimension to fill, index in tensor as flattened array)
  // The value is start and end in the output array which has the padded result.
  std::vector<std::pair<int, int>> cache;
};

// Wrapper for params passed to the Eval<T> function.
template <typename T>
struct EvalData {
  OpData* op_data = nullptr;
  const TfLiteTensor* padding_matrix = nullptr;
  const TfLiteIntArray* input_dims = nullptr;
  // Holds number of elements at the nth dimension.
  // value at last dimension = 1, at second to last = sizeof last dimension.
  const std::vector<int>* dimension_num_elements = nullptr;
  const T* input_data = nullptr;

  int offset = kUnsetOffset;
  T* output_data = nullptr;
  int input_size = 0;
  int output_size = 0;
  int num_dims = 0;
};

// Helper method that fills the left and right pads.
template <typename T>
inline void GetPadding(const T* data, int offset, int64_t* left_pad,
                       int64_t* right_pad) {
  *left_pad = static_cast<int64_t>(*(data + offset * 2));
  *right_pad = static_cast<int64_t>(*(data + offset * 2 + 1));
}

inline void GetPadding(const TfLiteTensor* padding_matrix, int dimension,
                       int64_t* left_pad, int64_t* right_pad) {
  switch (padding_matrix->type) {
    case kTfLiteInt32:
      GetPadding(padding_matrix->data.i32, dimension, left_pad, right_pad);
      break;
    case kTfLiteInt64:
      GetPadding(padding_matrix->data.i64, dimension, left_pad, right_pad);
      break;
    default:
      return;
  }
}

template <typename T>
int Eval(EvalData<T>* eval_data, int current_dim, int flat_index,
         int output_index) {
  if (current_dim == eval_data->num_dims) {
    // Base case if we finished evaluating.
    if (output_index >= eval_data->output_size) {
      return output_index;
    }
    eval_data->output_data[output_index] = eval_data->input_data[flat_index];
    return output_index + 1;
  }
  // Check if the value is computed already.
  const int cache_index = current_dim * eval_data->input_size + flat_index;
  auto& cache_entry = eval_data->op_data->cache[cache_index];
  if (cache_entry.first != -1) {
    // Cache value is (start, end) interval. We can just copy the interval
    // directly.
    const int count = cache_entry.second - cache_entry.first;
    memcpy(eval_data->output_data + output_index,
           eval_data->output_data + cache_entry.first, count * sizeof(T));
    return output_index + count;
  }
  cache_entry.first = output_index;
  int64_t left_pad = 0, right_pad = 0;
  const int multiplier = (*eval_data->dimension_num_elements)[current_dim];
  const TfLiteTensor* padding_matrix = eval_data->padding_matrix;
  const auto offset = eval_data->offset;
  auto* dims = eval_data->input_dims;

  GetPadding(padding_matrix, current_dim, &left_pad, &right_pad);
  // Left padding
  for (int i = left_pad + offset - 1; i >= offset && left_pad > 0;
       --i, --left_pad) {
    output_index = Eval(eval_data, current_dim + 1, flat_index + i * multiplier,
                        output_index);
  }
  // Original values.
  for (int i = 0; i < dims->data[current_dim]; ++i) {
    output_index = Eval(eval_data, current_dim + 1, flat_index + i * multiplier,
                        output_index);
  }
  // Right padding.
  for (int i = dims->data[current_dim] - (1 + offset); i >= 0 && right_pad > 0;
       --i, --right_pad) {
    output_index = Eval(eval_data, current_dim + 1, flat_index + i * multiplier,
                        output_index);
  }
  cache_entry.second = output_index;
  return output_index;
}

// Returns the shape of the final output after padding.
std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> GetPaddedOutputShape(
    const TfLiteTensor* input, const TfLiteTensor* padding_matrix) {
  const int input_dims = NumDimensions(input);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(input_dims), TfLiteIntArrayFree);

  int64_t left_pad = 0, right_pad = 0;
  for (int i = 0; i < input_dims; ++i) {
    GetPadding(padding_matrix, i, &left_pad, &right_pad);
    shape->data[i] = SizeOfDimension(input, i) + left_pad + right_pad;
  }
  return shape;
}

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, 0);
  const TfLiteTensor* padding_matrix = GetInput(context, node, 1);
  auto* params =
      reinterpret_cast<TfLiteMirrorPaddingParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  if (params == nullptr) {
    return kTfLiteError;
  }
  const int input_dims = NumDimensions(input_tensor);

  TfLiteTensor* output_tensor = GetOutput(context, node, 0);
  if (IsDynamicTensor(output_tensor)) {
    auto output_size = GetPaddedOutputShape(input_tensor, padding_matrix);
    if (output_size == nullptr) {
      return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(
        context->ResizeTensor(context, output_tensor, output_size.release()));
  }

  std::vector<int> dimension_num_elements(input_dims, 1);
  for (int i = input_dims - 2; i >= 0; i--) {
    dimension_num_elements[i] =
        dimension_num_elements[i + 1] * input_tensor->dims->data[i + 1];
  }
  const int input_size = NumElements(input_tensor);

  const int offset =
      params->mode != TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect ? 0
                                                                           : 1;
  TfLiteStatus status = kTfLiteOk;
  int output_index = 0;
  // Reset cache array.
  std::fill(op_data->cache.begin(), op_data->cache.end(),
            std::make_pair(-1, -1));
#define TF_LITE_MIRROR_PAD(type)                              \
  EvalData<type> eval_data;                                   \
  eval_data.input_data = GetTensorData<type>(input_tensor);   \
  eval_data.input_dims = input_tensor->dims;                  \
  eval_data.input_size = input_size;                          \
  eval_data.dimension_num_elements = &dimension_num_elements; \
  eval_data.num_dims = input_dims;                            \
  eval_data.offset = offset;                                  \
  eval_data.op_data = op_data;                                \
  eval_data.output_data = GetTensorData<type>(output_tensor); \
  eval_data.output_size = NumElements(output_tensor);         \
  eval_data.padding_matrix = padding_matrix;                  \
  Eval<type>(&eval_data, 0, 0, output_index);

  switch (output_tensor->type) {
    case kTfLiteFloat32: {
      TF_LITE_MIRROR_PAD(float);
      break;
    }
    case kTfLiteInt32: {
      TF_LITE_MIRROR_PAD(int32_t);
      break;
    }
    case kTfLiteUInt8: {
      TF_LITE_MIRROR_PAD(uint8_t);
      break;
    }
    case kTfLiteInt64: {
      TF_LITE_MIRROR_PAD(int64_t);
      break;
    }
    default:
      status = kTfLiteError;
      break;
  }
#undef TF_LITE_MIRROR_PAD
  return status;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, 0);
  const TfLiteTensor* padding_matrix = GetInput(context, node, 1);
  TfLiteTensor* output_tensor = GetOutput(context, node, 0);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumDimensions(padding_matrix), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(padding_matrix, 0),
                    NumDimensions(input_tensor));

  int num_elements = NumElements(input_tensor) * NumDimensions(input_tensor);
  op_data->cache.resize(num_elements + 1);

  if (!IsConstantTensor(padding_matrix)) {
    SetTensorToDynamic(output_tensor);
    return kTfLiteOk;
  }
  // We have constant padding, so we can infer output size.

  auto output_size = GetPaddedOutputShape(input_tensor, padding_matrix);
  if (output_size == nullptr) {
    return kTfLiteError;
  }
  return context->ResizeTensor(context, output_tensor, output_size.release());
}

}  // namespace mirror_pad
TfLiteRegistration* Register_MIRROR_PAD() {
  static TfLiteRegistration r = {mirror_pad::Init, mirror_pad::Free,
                                 mirror_pad::Prepare, mirror_pad::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
