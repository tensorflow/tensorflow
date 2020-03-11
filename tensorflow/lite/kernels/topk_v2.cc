/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace topk_v2 {
constexpr int kInputTensor = 0;
constexpr int kInputTopK = 1;
constexpr int kOutputValues = 0;
constexpr int kOutputIndexes = 1;

namespace {
TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* top_k = GetInput(context, node, kInputTopK);
  // INT32 number of top results is supported.
  TF_LITE_ENSURE_EQ(context, top_k->type, kTfLiteInt32);
  // Check that the tensor contains only one value.
  TF_LITE_ENSURE_EQ(context, NumElements(top_k), 1);
  const int32 k = *GetTensorData<int32_t>(top_k);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const int num_dimensions = NumDimensions(input);
  // Check that input has one or more dimensions.
  TF_LITE_ENSURE_MSG(context, input->dims->size >= 1,
                     "TopK k input must have 1 or more dimensions.");
  // Check that k is less or equal the internal dimension.
  TF_LITE_ENSURE_MSG(context, k <= input->dims->data[num_dimensions - 1],
                     "TopK k is higher than the internal dimension.");

  TfLiteIntArray* output_indexes_shape = TfLiteIntArrayCreate(num_dimensions);
  TfLiteIntArray* output_values_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions - 1; ++i) {
    output_indexes_shape->data[i] = input->dims->data[i];
    output_values_shape->data[i] = input->dims->data[i];
  }
  output_indexes_shape->data[num_dimensions - 1] = k;
  output_values_shape->data[num_dimensions - 1] = k;
  TfLiteTensor* output_indexes = GetOutput(context, node, kOutputIndexes);
  TfLiteTensor* output_values = GetOutput(context, node, kOutputValues);
  // Force output types.
  output_indexes->type = kTfLiteInt32;
  output_values->type = input->type;
  auto resize_tensor = [context](TfLiteTensor* tensor, TfLiteIntArray* new_size,
                                 TfLiteIntArray* delete_on_error) {
    TfLiteStatus status = context->ResizeTensor(context, tensor, new_size);
    if (status != kTfLiteOk) {
      if (delete_on_error != nullptr) {
        TfLiteIntArrayFree(delete_on_error);
      }
    }
    return status;
  };
  TF_LITE_ENSURE_OK(context, resize_tensor(output_indexes, output_indexes_shape,
                                           output_values_shape));
  TF_LITE_ENSURE_OK(context,
                    resize_tensor(output_values, output_values_shape, nullptr));
  return kTfLiteOk;
}

// Class that collects indices of top k values.  Based on template
// tensorflow::gtl::TopN<> but, for optimization, it re-uses the same container.
template <typename T>
class TopContainer {
 public:
  TopContainer() = delete;
  TopContainer(int32 k, int32 row_size) : k_(k) {
    container_.reserve(std::min(k, row_size) + 1);
  }

  void start_collecting(const T* values) {
    values_ = values;
    container_.clear();
  }
  void push(int32 a) {
    auto comparator = [this](int32 a, int32 b) { return compare_fun(a, b); };
    if (container_.size() <= k_) {
      container_.push_back(a);
      if (container_.size() == k_ + 1) {
        std::make_heap(container_.begin(), container_.end(), comparator);
        std::pop_heap(container_.begin(), container_.end(), comparator);
      }
    } else if (comparator(a, container_.front())) {
      // Due to how we defined comparator / compare_fun, container_.front()
      // contains the index of the smallest of the top-k elements seen so far.
      //
      // If control reaches this point, we know that the current index a
      // corresponds to an element which is bigger than the smallest of the
      // top-k elements seen so far.  Hence, we have to update the indices of
      // the top-k elements, by removing the index of the smallest top-k
      // element, adding a, and making sure container_[0:k] is still a heap.

      // Store index a into container_[k].
      container_.back() = a;

      // Swap container_[0] and container_[k], and rearrange elements from
      // container_[0,k) such that they are a heap according to comparator.  For
      // more info, see https://en.cppreference.com/w/cpp/algorithm/pop_heap.
      std::pop_heap(container_.begin(), container_.end(), comparator);
    }
  }

  const std::vector<int32>& sorted_result() {
    auto comparator = [this](int32 a, int32 b) { return compare_fun(a, b); };
    if (container_.size() <= k_) {
      // Note: due to the way we defined compare_fun (see comments for that
      // function) std::sort puts the indices from container_ in decreasing
      // order of the corresponding elements.
      std::sort(container_.begin(), container_.end(), comparator);
    } else {
      std::sort_heap(container_.begin(), container_.end() - 1, comparator);
      container_.resize(k_);
    }
    return container_;
  }

 private:
  const int32 k_;

  // container_[0,k) holds the indices of the largest k elements from values_
  // seen so far and are maintained in a min-heap order: container_.front() is
  // the index of the smallest of the top-k elements see so far.
  //
  // container_[k] is used as temporary space (not part of the min-heap).
  std::vector<int32> container_;

  const T* values_ = nullptr;

  // Compares indices a and b based on the corresponding elements from values_.
  //
  // Intuitively, compare_fun(a, b) returns true iff values_[b] < values_[a]
  // (notice the inversion of direction, not a typo); ties (==) are broken in
  // favor of earlier elements (i.e., a < b).
  bool compare_fun(int32 a, int32 b) const {
    if (values_[b] < values_[a]) {
      return true;
    } else if (values_[b] > values_[a]) {
      return false;
    } else {
      return a < b;
    }
  }
};

// Mostly modeled on tensorflow/core/kernels/topk_op.cc for CPU.
template <typename T>
void TopK(int32 row_size, int32 num_rows, const T* data, int32 k,
          int32* output_indexes, T* output_values) {
  TopContainer<T> topc(k, row_size);
  for (int row = 0; row < num_rows; ++row) {
    const T* values_row = data + row * row_size;
    topc.start_collecting(values_row);
    for (int32 c = 0; c < row_size; ++c) {
      topc.push(c);
    }

    // Prepare output buffers.
    int32* indexes_row = output_indexes + row * k;
    T* output_row = output_values + row * k;
    // We always assume that the output is sorted.
    const auto& top_k = topc.sorted_result();
    std::copy(top_k.begin(), top_k.end(), indexes_row);
    std::transform(top_k.begin(), top_k.end(), output_row,
                   [values_row](const int32 loc) { return values_row[loc]; });
  }
}

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check that the inputs and outputs have the right sizes and types.
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output_values = GetOutput(context, node, kOutputValues);
  TF_LITE_ENSURE_EQ(context, input->type, output_values->type);

  const TfLiteTensor* top_k = GetInput(context, node, kInputTopK);
  TF_LITE_ENSURE_EQ(context, top_k->type, kTfLiteInt32);

  // Set output dynamic if the input is not const.
  if (IsConstantTensor(top_k)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  } else {
    TfLiteTensor* output_indexes = GetOutput(context, node, kOutputIndexes);
    TfLiteTensor* output_values = GetOutput(context, node, kOutputValues);
    SetTensorToDynamic(output_indexes);
    SetTensorToDynamic(output_values);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output_values = GetOutput(context, node, kOutputValues);
  TfLiteTensor* output_indexes = GetOutput(context, node, kOutputIndexes);
  if (IsDynamicTensor(output_values)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }
  const TfLiteTensor* top_k = GetInput(context, node, kInputTopK);
  const int32 k = top_k->data.i32[0];
  // The tensor can have more than 2 dimensions or even be a vector, the code
  // anyway calls the internal dimension as row;
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const int32 row_size = input->dims->data[input->dims->size - 1];
  int32 num_rows = 1;
  for (int i = 0; i < input->dims->size - 1; ++i) {
    num_rows *= input->dims->data[i];
  }
  switch (output_values->type) {
    case kTfLiteFloat32:
      TopK(row_size, num_rows, GetTensorData<float>(input), k,
           output_indexes->data.i32, GetTensorData<float>(output_values));
      break;
    case kTfLiteUInt8:
      TopK(row_size, num_rows, input->data.uint8, k, output_indexes->data.i32,
           output_values->data.uint8);
      break;
    case kTfLiteInt8:
      TopK(row_size, num_rows, input->data.int8, k, output_indexes->data.i32,
           output_values->data.int8);
      break;
    case kTfLiteInt32:
      TopK(row_size, num_rows, input->data.i32, k, output_indexes->data.i32,
           output_values->data.i32);
      break;
    case kTfLiteInt64:
      TopK(row_size, num_rows, input->data.i64, k, output_indexes->data.i32,
           output_values->data.i64);
      break;
    default:
      context->ReportError(context,
                           "Type %d is currently not supported by TopK.",
                           output_values->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace topk_v2
TfLiteRegistration* Register_TOPK_V2() {
  static TfLiteRegistration r = {nullptr, nullptr, topk_v2::Prepare,
                                 topk_v2::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
