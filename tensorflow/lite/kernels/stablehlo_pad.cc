/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
         //
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_pad {
namespace {

constexpr int kMaxDims = TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT;

// Fills a buffer with the given data.
//
// WARNING: This expects buffer_bytes to be a multiple of data_bytes.
void FillBuffer(char* buffer, int64_t buffer_bytes, const char* data,
                int64_t data_bytes) {
  if (buffer_bytes == 0) {
    return;
  }
  TFLITE_DCHECK(buffer_bytes % data_bytes == 0);
  std::memcpy(buffer, data, data_bytes);
  buffer_bytes -= data_bytes;
  while (buffer_bytes) {
    const int64_t bytes = std::min(buffer_bytes, data_bytes);
    std::memcpy(buffer + data_bytes, buffer, bytes);
    buffer_bytes -= bytes;
    data_bytes += bytes;
  }
}

// Recursive implementation of a strided copy of a tensor.
void StridedCopy(const int rank, const char* input, const int64_t* input_shape,
                 const int64_t* input_strides, char* output,
                 const int64_t* output_strides, const int64_t element_size,
                 const int depth) {
  if (input_shape[depth] <= 0) {
    return;
  }
  if (depth + 1 == rank) {
    if (output_strides[depth] == element_size &&
        input_strides[depth] == element_size) {
      std::memcpy(output, input, element_size * input_shape[depth]);
    } else {
      for (int64_t i = 0; i < input_shape[depth]; ++i) {
        std::memcpy(output, input, element_size);
        input += input_strides[depth];
        output += output_strides[depth];
      }
    }
  } else {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      StridedCopy(rank, input, input_shape, input_strides, output,
                  output_strides, element_size, depth + 1);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  }
}

// Holds the main implementation of the Pad operation.
//
// The StableHLO pad operation can add interior padding and edge padding to a
// tensor. The edge padding may be negative in which case it is considered as a
// cropping specification.
//
// This is implemented as a strided copy where:
//
// - interior padding affects the output strides.
// - positive edge padding affects the output shape, strides and initial offset.
// - negative edge padding affects the input shape and initial offset as well as
// the output initial offset.
//
// See https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad for more
// information.
class PadData {
 public:
  static constexpr int kInput = 0;
  static constexpr int kPaddingValue = 1;
  static constexpr int kOutput = 0;

  explicit PadData(const TfLiteStablehloPadParams& params) {
    std::memcpy(edge_pad_low_, params.edge_padding_low, sizeof(edge_pad_low_));
    std::memcpy(edge_pad_high_, params.edge_padding_high,
                sizeof(edge_pad_high_));
    std::memcpy(interior_pad_, params.interior_padding, sizeof(interior_pad_));
  }

  // Computes the shapes and strides that are needed for the final strided copy.
  TfLiteStatus Setup(TfLiteContext* context, const int* dims, const int rank,
                     const int64_t element_size) {
    TF_LITE_ENSURE(context, rank > 0 && rank <= kMaxDims);
    rank_ = rank;
    element_size_ = element_size;
    input_offset_ = 0;
    output_offset_ = 0;
    output_size_ = 0;

    // Compute the output shape.
    for (int i = 0; i < rank; ++i) {
      TF_LITE_ENSURE(context,
                     interior_pad_[i] >= 0 && interior_pad_[i] <= INT_MAX);
      TF_LITE_ENSURE(
          context, edge_pad_low_[i] >= -INT_MAX && edge_pad_low_[i] <= INT_MAX);
      TF_LITE_ENSURE(context, edge_pad_high_[i] >= -INT_MAX &&
                                  edge_pad_high_[i] <= INT_MAX);
      int64_t interior_gaps = std::max<int64_t>(0, dims[i] - 1);
      int64_t out_dim = dims[i] + interior_gaps * interior_pad_[i] +
                        edge_pad_low_[i] + edge_pad_high_[i];
      TF_LITE_ENSURE(context, out_dim <= INT_MAX);
      output_shape_[i] = std::max<int64_t>(0, out_dim);
    }
    if (std::any_of(output_shape_, output_shape_ + rank,
                    [](auto s) { return s <= 0; })) {
      std::memset(input_shape_, 0, sizeof(input_shape_));
      std::memset(output_shape_, 0, sizeof(output_shape_));
      output_size_ = 0;
      return kTfLiteOk;
    }
    // Compute the output size for each dimension.
    //
    // This is different from the output strides because of the interior
    // padding: the output strides take it into account to "jump" over the
    // interior padding elements.
    output_dimension_sizes_[rank - 1] = element_size;
    for (int i = rank - 2; i >= 0; --i) {
      if (__builtin_mul_overflow(output_shape_[i + 1],
                                 output_dimension_sizes_[i + 1],
                                 &output_dimension_sizes_[i])) {
        return kTfLiteError;
      }
    }
    // Compute the output stride for each dimension.
    //
    // This is the stride between two elements that are copied from the input
    // tensor (i.e. not generated by interior padding).
    output_strides_[rank - 1] = element_size * (interior_pad_[rank - 1] + 1);
    for (int i = rank - 2; i >= 0; --i) {
      if (__builtin_mul_overflow(output_dimension_sizes_[i],
                                 interior_pad_[i] + 1, &output_strides_[i])) {
        return kTfLiteError;
      }
    }
    // Compute the output offset from the eventual pads.
    for (int i = 0; i < rank; ++i) {
      int64_t pad_offset;
      if (__builtin_mul_overflow(std::max<int64_t>(edge_pad_low_[i], 0),
                                 output_dimension_sizes_[i], &pad_offset)) {
        return kTfLiteError;
      }
      if (__builtin_add_overflow(output_offset_, pad_offset, &output_offset_)) {
        return kTfLiteError;
      }
    }
    // Compute the final output size.
    output_size_ = element_size;
    for (int i = 0; i < rank; ++i) {
      if (__builtin_mul_overflow(output_size_, output_shape_[i],
                                 &output_size_)) {
        return kTfLiteError;
      }
    }
    // Compute input strides.
    input_strides_[rank - 1] = element_size;
    for (int i = rank - 1; i >= 1; --i) {
      if (__builtin_mul_overflow(dims[i], input_strides_[i],
                                 &input_strides_[i - 1])) {
        return kTfLiteError;
      }
    }
    // Helper that computes the division between a negative num and a positive
    // denum, rounding away from 0, or returns 0 if num is positive.
    auto DivNegRoundAwayOrZero = [](int64_t num, int64_t denum) -> int64_t {
      TFLITE_DCHECK(denum > 0);
      return num < 0 ? (num - denum + 1) / denum : 0;
    };
    // Compute the input bounds from the eventual crops.
    //
    // If negative padding is applied, we can treat this as copying a subtensor
    // of the input. We modify the input shape in place as we don't use it for
    // anything else.
    for (int i = 0; i < rank; ++i) {
      input_shape_[i] =
          dims[i] +
          DivNegRoundAwayOrZero(edge_pad_low_[i], interior_pad_[i] + 1) +
          DivNegRoundAwayOrZero(edge_pad_high_[i], interior_pad_[i] + 1);
    }
    // Compute the input offset from the eventual crops.
    //
    // When computing the subtensor from the negative padding, we need to find
    // out the offset to its first element in addition to its shape (see
    // previous comment).
    //
    // Cropping also means that the interior padding can become edge padding so
    // we also need to update the output offset:
    //
    // > `1 0 0 0 2 0 0 0 3` cropped by 1 low element becomes `0 0 0 2 0 0 0 3`
    // > which effectlvely means pad `2 3` with an interior padding of 3 and a
    // > low edge padding of 3.
    for (int i = 0; i < rank; ++i) {
      input_offset_ -=
          DivNegRoundAwayOrZero(edge_pad_low_[i], interior_pad_[i] + 1) *
          input_strides_[i];
      if (edge_pad_low_[i] < 0) {
        int64_t tmp_offset = ((interior_pad_[i] + 1 + edge_pad_low_[i]) %
                              (interior_pad_[i] + 1));
        if (tmp_offset < 0) {
          tmp_offset += interior_pad_[i] + 1;
        }
        output_offset_ += tmp_offset * output_dimension_sizes_[i];
      }
    }
    return kTfLiteOk;
  }

  void Apply(const char* input, const char* padding_value, char* output) const {
    // Fill the output tensor with the padding value.
    FillBuffer(output, output_size_, padding_value, element_size_);
    StridedCopy(rank_, input + input_offset_, input_shape_, input_strides_,
                output + output_offset_, output_strides_, element_size_,
                /*depth=*/0);
  }

  TfLiteIntArray* BuildOutputTensorDims() const {
    TfLiteIntArray* dims = TfLiteIntArrayCreate(rank_);
    for (int64_t i = 0; i < rank_; ++i) {
      dims->data[i] = static_cast<int>(output_shape_[i]);
    }
    return dims;
  }

 private:
  int64_t edge_pad_low_[kMaxDims];
  int64_t edge_pad_high_[kMaxDims];
  int64_t interior_pad_[kMaxDims];
  int64_t rank_ = 0;
  int64_t element_size_ = 0;
  int64_t input_shape_[kMaxDims];
  int64_t output_shape_[kMaxDims];
  int64_t input_strides_[kMaxDims];
  int64_t output_strides_[kMaxDims];
  int64_t output_dimension_sizes_[kMaxDims];
  int64_t input_offset_ = 0;
  int64_t output_offset_ = 0;
  int64_t output_size_ = 0;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  return new PadData(
      *reinterpret_cast<const TfLiteStablehloPadParams*>(options));
}

void Free(TfLiteContext* context, void* node_data) {
  delete reinterpret_cast<PadData*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Input checks.
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, PadData::kInput, &input_tensor));
  const TfLiteTensor* padding_value_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, PadData::kPaddingValue,
                                          &padding_value_tensor));
  TF_LITE_ENSURE(context, input_tensor->type == padding_value_tensor->type);
  // PadData computations.
  size_t element_size;
  TF_LITE_ENSURE(context, GetSizeOfType(context, input_tensor->type,
                                        &element_size) == kTfLiteOk);
  PadData& pad_data = *reinterpret_cast<PadData*>(node->user_data);
  TF_LITE_ENSURE_STATUS(pad_data.Setup(context, input_tensor->dims->data,
                                       input_tensor->dims->size, element_size));
  // Output tensor setup.
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, PadData::kOutput, &output_tensor));
  TF_LITE_ENSURE(context, input_tensor->type == output_tensor->type);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(
      context, output_tensor, pad_data.BuildOutputTensorDims()));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, PadData::kInput, &input_tensor));
  const TfLiteTensor* padding_value_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, PadData::kPaddingValue,
                                          &padding_value_tensor));
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, PadData::kOutput, &output_tensor));
  // Pad using PadData
  PadData& pad_data = *reinterpret_cast<PadData*>(node->user_data);
  pad_data.Apply(input_tensor->data.raw_const,
                 padding_value_tensor->data.raw_const, output_tensor->data.raw);
  return kTfLiteOk;
}

}  // namespace
}  // namespace stablehlo_pad

TfLiteRegistration* Register_STABLEHLO_PAD() {
  static TfLiteRegistration r = {/*.init=*/stablehlo_pad::Init,
                                 /*.free=*/stablehlo_pad::Free,
                                 /*.prepare=*/stablehlo_pad::Prepare,
                                 /*.invoke=*/stablehlo_pad::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
