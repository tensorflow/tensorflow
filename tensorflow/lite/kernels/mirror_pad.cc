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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace mirror_pad {
namespace {

// Nil value for paddingMode/offset.
const int kUnsetOffset = -1;

struct PaddingMatrixData {
  TfLiteType type = kTfLiteNoType;
  absl::Span<const int32_t> i32;
  absl::Span<const int64_t> i64;
  std::vector<int> left_padding;
  std::vector<int> right_padding;
  size_t num_dims = 0;
};

// Wrapper for params passed to the Eval<T> function.
template <typename T>
struct EvalData {
  const TfLiteIntArray* input_dims = nullptr;
  // Holds number of elements at the nth dimension.
  // value at last dimension = 1, at second to last = sizeof last dimension.
  const std::vector<int>* output_dims_num_elements = nullptr;
  const std::vector<int>* input_dims_num_elements = nullptr;
  const T* input_data = nullptr;
  const int* left_padding = nullptr;
  const int* right_padding = nullptr;

  int offset = kUnsetOffset;
  T* output_data = nullptr;
  size_t num_dims = 0;
};

/// Reads one `[left, right]` padding pair from a flattened padding matrix.
template <typename T>
inline void GetPadding(absl::Span<const T> data, size_t dimension,
                       int64_t* left_pad, int64_t* right_pad) {
  const size_t offset = dimension * 2;
  *left_pad = static_cast<int64_t>(data[offset]);
  *right_pad = static_cast<int64_t>(data[offset + 1]);
}

/// Reads one `[left, right]` padding pair from typed padding matrix data.
inline void GetPadding(const PaddingMatrixData& padding_matrix,
                       size_t dimension, int64_t* left_pad,
                       int64_t* right_pad) {
  switch (padding_matrix.type) {
    case kTfLiteInt32:
      GetPadding(padding_matrix.i32, dimension, left_pad, right_pad);
      break;
    case kTfLiteInt64:
      GetPadding(padding_matrix.i64, dimension, left_pad, right_pad);
      break;
    default:
      return;
  }
}

/// Builds a typed span view over the padding matrix tensor data.
TfLiteStatus GetPaddingMatrixData(TfLiteContext* context,
                                  const TfLiteTensor* padding_matrix,
                                  PaddingMatrixData* padding_data) {
  size_t padding_count = 0;
  TF_LITE_ENSURE_MSG(
      context, CheckedNumElements(padding_matrix, padding_count) == kTfLiteOk,
      "MirrorPad paddings size overflowed.");
  TF_LITE_ENSURE(context, padding_count % 2 == 0);
  padding_data->type = padding_matrix->type;
  padding_data->num_dims = padding_count / 2;
  switch (padding_matrix->type) {
    case kTfLiteInt32: {
      const int32_t* data = GetTensorData<int32_t>(padding_matrix);
      TF_LITE_ENSURE(context, data != nullptr || padding_count == 0);
      padding_data->i32 = absl::MakeConstSpan(data, padding_count);
      return kTfLiteOk;
    }
    case kTfLiteInt64: {
      const int64_t* data = GetTensorData<int64_t>(padding_matrix);
      TF_LITE_ENSURE(context, data != nullptr || padding_count == 0);
      padding_data->i64 = absl::MakeConstSpan(data, padding_count);
      return kTfLiteOk;
    }
    default:
      return kTfLiteError;
  }
}

/// Computes the MirrorPad mode offset used by symmetric and reflect padding.
TfLiteStatus GetOffset(TfLiteContext* context, TfLiteNode* node, int* offset) {
  auto* params =
      reinterpret_cast<TfLiteMirrorPaddingParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  TF_LITE_ENSURE(
      context,
      params->mode == TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect ||
          params->mode ==
              TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingSymmetric);
  *offset = params->mode != TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect
                ? 0
                : 1;
  return kTfLiteOk;
}

/// Validates the padding matrix type and shape against the input tensor rank.
TfLiteStatus ValidatePaddingMatrix(TfLiteContext* context,
                                   const TfLiteTensor* input,
                                   const TfLiteTensor* padding_matrix) {
  TF_LITE_ENSURE(context, padding_matrix->type == kTfLiteInt32 ||
                              padding_matrix->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, NumDimensions(padding_matrix), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(padding_matrix, 0),
                    NumDimensions(input));
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(padding_matrix, 1), 2);
  return kTfLiteOk;
}

/// Computes and validates the output shape after applying mirror padding.
TfLiteStatus GetPaddedOutputShape(
    TfLiteContext* context, const TfLiteTensor* input, int offset,
    PaddingMatrixData* padding_matrix,
    std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)>* shape) {
  const int input_dims = NumDimensions(input);
  TF_LITE_ENSURE(context,
                 padding_matrix->num_dims == static_cast<size_t>(input_dims));
  shape->reset(TfLiteIntArrayCreate(input_dims));
  TF_LITE_ENSURE(context, *shape != nullptr);
  padding_matrix->left_padding.resize(input_dims);
  padding_matrix->right_padding.resize(input_dims);

  int64_t left_pad = 0, right_pad = 0;
  for (size_t i = 0; i < padding_matrix->num_dims; ++i) {
    GetPadding(*padding_matrix, i, &left_pad, &right_pad);
    TF_LITE_ENSURE_MSG(context, left_pad >= 0 && right_pad >= 0,
                       "MirrorPad paddings must be non-negative.");
    const int input_dim_size = input->dims->data[i];
    TF_LITE_ENSURE_MSG(context, input_dim_size >= 0,
                       "MirrorPad input dimensions must be non-negative.");
    const int64_t max_padding = static_cast<int64_t>(input_dim_size) - offset;
    TF_LITE_ENSURE_MSG(
        context, left_pad <= max_padding && right_pad <= max_padding,
        "MirrorPad paddings must be no greater than the input dimension size "
        "minus the mode offset. Dimension %zu has input size %d, mode offset "
        "%d, max padding %lld, left padding %lld, and right padding %lld.",
        i, input_dim_size, offset, static_cast<long long>(max_padding),
        static_cast<long long>(left_pad), static_cast<long long>(right_pad));
    CheckedInt<int> left_pad_int(left_pad);
    CheckedInt<int> right_pad_int(right_pad);
    CheckedInt<int> output_dim =
        CheckedInt<int>(input_dim_size) + left_pad_int + right_pad_int;
    TF_LITE_ENSURE_MSG(context, output_dim.Status() == kTfLiteOk,
                       "MirrorPad output dimension overflowed.");
    padding_matrix->left_padding[i] = left_pad_int.Value();
    padding_matrix->right_padding[i] = right_pad_int.Value();
    (*shape)->data[i] = output_dim.Value();
  }
  size_t output_num_elements = 0;
  TF_LITE_ENSURE_MSG(
      context,
      CheckedNumElements(shape->get(), output_num_elements) == kTfLiteOk,
      "MirrorPad output size overflowed.");
  return kTfLiteOk;
}

// Given dimension index and the left/right padding.
// Returns the corresponding dimension in the input array.
inline int GetInputDimension(int padded_dimension, int left_pad, int right_pad,
                             int input_dim_size, int offset) {
  if (padded_dimension < left_pad) {
    const int original_ind = left_pad + offset - 1;
    return original_ind - (std::min(padded_dimension, original_ind - offset));
  }
  padded_dimension -= left_pad;
  if (padded_dimension >= input_dim_size) {
    padded_dimension -= input_dim_size;
    const int original_ind = input_dim_size - (1 + offset);
    return original_ind - std::min(padded_dimension, original_ind);
  }
  return padded_dimension;
}

// Given and index in output array, returns the index of the value
// in input array.
template <typename T>
int GetFlatIndex(int index, EvalData<T>* eval_data) {
  int flat_index = 0;
  int dimension_index, index_in_input;
  for (size_t i = 0; i < eval_data->num_dims; ++i) {
    const int left_pad = eval_data->left_padding[i];
    const int right_pad = eval_data->right_padding[i];
    dimension_index = index / (*eval_data->output_dims_num_elements)[i];
    index_in_input =
        GetInputDimension(dimension_index, left_pad, right_pad,
                          eval_data->input_dims->data[i], eval_data->offset);
    flat_index += index_in_input * (*eval_data->input_dims_num_elements)[i];
    index %= (*eval_data->output_dims_num_elements)[i];
  }
  return flat_index;
}

template <typename T>
struct MirrorPadWorkerTask : cpu_backend_threadpool::Task {
  MirrorPadWorkerTask(EvalData<T>* eval_data, int start, int end)
      : eval_data(eval_data), start(start), end(end) {}
  void Run() override {
    auto* input_data = eval_data->input_data;
    auto* output_data = eval_data->output_data;
    for (int i = start; i < end; ++i) {
      output_data[i] = input_data[GetFlatIndex(i, eval_data)];
    }
  }

 private:
  EvalData<T>* eval_data;
  int start;
  int end;
};

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  ruy::profiler::ScopeLabel label("MirrorPad");
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input_tensor));
  const TfLiteTensor* padding_matrix;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &padding_matrix));
  int offset = kUnsetOffset;
  TF_LITE_ENSURE_OK(context, GetOffset(context, node, &offset));
  TF_LITE_ENSURE_OK(
      context, ValidatePaddingMatrix(context, input_tensor, padding_matrix));
  PaddingMatrixData padding_data;
  TF_LITE_ENSURE_OK(
      context, GetPaddingMatrixData(context, padding_matrix, &padding_data));
  const int input_dims = NumDimensions(input_tensor);

  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, input_tensor->type, output_tensor->type);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> expected_shape(
      nullptr, TfLiteIntArrayFree);
  TF_LITE_ENSURE_OK(context,
                    GetPaddedOutputShape(context, input_tensor, offset,
                                         &padding_data, &expected_shape));
  if (IsDynamicTensor(output_tensor)) {
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output_tensor,
                                                expected_shape.release()));
  } else {
    TF_LITE_ENSURE_EQ(context, NumDimensions(output_tensor),
                      expected_shape->size);
    for (int i = 0; i < expected_shape->size; ++i) {
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(output_tensor, i),
                        expected_shape->data[i]);
    }
  }

  std::vector<int> output_dims_num_elements(input_dims, 1);
  std::vector<int> input_dims_num_elements(input_dims, 1);
  const RuntimeShape output_shape = GetTensorShape(output_tensor);
  const RuntimeShape input_shape = GetTensorShape(input_tensor);
  for (int i = input_dims - 2; i >= 0; i--) {
    TF_LITE_ENSURE_MSG(context,
                       output_shape.CheckedSizeFromDimension(
                           /*start=*/i + 1, output_dims_num_elements[i]),
                       "MirrorPad output stride overflowed.");
    TF_LITE_ENSURE_MSG(context,
                       input_shape.CheckedSizeFromDimension(
                           /*start=*/i + 1, input_dims_num_elements[i]),
                       "MirrorPad input stride overflowed.");
  }

  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  const int thread_count = std::max(1, cpu_backend_context->max_num_threads());
  TfLiteStatus status = kTfLiteOk;
  int output_size = 0;
  TF_LITE_ENSURE_MSG(
      context, CheckedNumElements(output_tensor, output_size) == kTfLiteOk,
      "MirrorPad output size overflowed.");
#define TF_LITE_MIRROR_PAD(type)                                           \
  EvalData<type> eval_data;                                                \
  eval_data.input_data = GetTensorData<type>(input_tensor);                \
  eval_data.input_dims = input_tensor->dims;                               \
  eval_data.output_dims_num_elements = &output_dims_num_elements;          \
  eval_data.input_dims_num_elements = &input_dims_num_elements;            \
  eval_data.num_dims = padding_data.num_dims;                              \
  eval_data.offset = offset;                                               \
  eval_data.output_data = GetTensorData<type>(output_tensor);              \
  eval_data.left_padding = padding_data.left_padding.data();               \
  eval_data.right_padding = padding_data.right_padding.data();             \
  std::vector<MirrorPadWorkerTask<type>> tasks;                            \
  tasks.reserve(thread_count);                                             \
  int start = 0;                                                           \
  for (int i = 0; i < thread_count; ++i) {                                 \
    int end = start + (output_size - start) / (thread_count - i);          \
    tasks.emplace_back(MirrorPadWorkerTask<type>(&eval_data, start, end)); \
    start = end;                                                           \
  }                                                                        \
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),              \
                                  cpu_backend_context);

  switch (TfLiteTypeGetSizeBits(output_tensor->type)) {
    case 8: {
      TF_LITE_MIRROR_PAD(uint8_t);
      break;
    }
    case 16: {
      TF_LITE_MIRROR_PAD(uint16_t);
      break;
    }
    case 32: {
      TF_LITE_MIRROR_PAD(uint32_t);
      break;
    }
    case 64: {
      TF_LITE_MIRROR_PAD(uint64_t);
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
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input_tensor));
  const TfLiteTensor* padding_matrix;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &padding_matrix));
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, input_tensor->type, output_tensor->type);
  int offset = kUnsetOffset;
  TF_LITE_ENSURE_OK(context, GetOffset(context, node, &offset));

  TF_LITE_ENSURE_OK(
      context, ValidatePaddingMatrix(context, input_tensor, padding_matrix));

  if (input_tensor->type == kTfLiteUInt8 || input_tensor->type == kTfLiteInt8 ||
      input_tensor->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input_tensor->params.scale,
                      output_tensor->params.scale);
    TF_LITE_ENSURE_EQ(context, input_tensor->params.zero_point,
                      output_tensor->params.zero_point);
  }

  if (input_tensor->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input_tensor->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output_tensor->params.zero_point, 0);
  }

  if (!IsConstantOrPersistentTensor(padding_matrix)) {
    SetTensorToDynamic(output_tensor);
    return kTfLiteOk;
  }
  // We have constant padding, so we can infer output size.
  PaddingMatrixData padding_data;
  TF_LITE_ENSURE_OK(
      context, GetPaddingMatrixData(context, padding_matrix, &padding_data));
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> output_size(
      nullptr, TfLiteIntArrayFree);
  TF_LITE_ENSURE_OK(context, GetPaddedOutputShape(context, input_tensor, offset,
                                                  &padding_data, &output_size));
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
