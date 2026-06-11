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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_pad {
namespace {

static constexpr int kMaxDims = TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT;

/// Fills a buffer by repeatedly copying the given pattern.
///
/// WARNING: This expects `buffer.size()` to be a multiple of `data.size()`.
void FillBuffer(absl::Span<char> buffer, absl::Span<const char> data) {
  if (buffer.empty()) {
    return;
  }
  assert(!data.empty());
  assert(buffer.size() >= data.size());
  if (data.empty() || buffer.size() < data.size()) {
    return;
  }
  assert(buffer.size() % data.size() == 0);
  std::memcpy(buffer.data(), data.data(), data.size());
  size_t bytes_filled = data.size();
  while (bytes_filled < buffer.size()) {
    const size_t bytes = std::min(buffer.size() - bytes_filled, bytes_filled);
    std::memcpy(buffer.data() + bytes_filled, buffer.data(), bytes);
    bytes_filled += bytes;
  }
}

/// Recursively copies a tensor region with independent input and output
/// strides.
void StridedCopy(const char* input, absl::Span<const int64_t> input_shape,
                 absl::Span<const int64_t> input_strides, char* output,
                 absl::Span<const int64_t> output_strides, int64_t element_size,
                 int depth) {
  const int rank = static_cast<int>(input_shape.size());
  assert(input_strides.size() == input_shape.size());
  assert(output_strides.size() == input_shape.size());
  if (depth + 1 == rank) {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      std::memcpy(output, input, element_size);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  } else {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      StridedCopy(input, input_shape, input_strides, output, output_strides,
                  element_size, depth + 1);
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
  enum { kInput, kPaddingValue, kInputTensorCount };
  enum { kOutput, kOutputTensorCount };

  explicit PadData(const TfLiteStablehloPadParams& params) {
    std::memcpy(
        edge_pad_low_, params.edge_padding_low,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
    std::memcpy(
        edge_pad_high_, params.edge_padding_high,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
    std::memcpy(
        interior_pad_, params.interior_padding,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
  }

  /// Computes the shapes and strides that are needed for the final strided
  /// copy.
  TfLiteStatus Setup(TfLiteContext* context, absl::Span<const int> dims,
                     int64_t element_size) {
    TF_LITE_ENSURE_MSG(
        context, !dims.empty() && dims.size() <= kMaxDims,
        "StableHLO Pad input rank must be between 1 and the maximum "
        "supported rank.");
    TF_LITE_ENSURE(context, dims.data() != nullptr);
    TF_LITE_ENSURE_MSG(context, element_size > 0,
                       "StableHLO Pad element size must be positive.");
    const int rank = static_cast<int>(dims.size());
    rank_ = rank;
    element_size_ = element_size;
    input_offset_ = 0;
    output_offset_ = 0;
    output_size_ = 0;
    has_input_copy_ = false;

    // Compute the output shape.
    int input_dims[kMaxDims];
    int output_dims[kMaxDims];
    for (int i = 0; i < rank; ++i) {
      TF_LITE_ENSURE_MSG(context, dims[i] >= 0,
                         "StableHLO Pad input dimensions must be "
                         "non-negative.");
      TF_LITE_ENSURE_MSG(
          context, interior_pad_[i] >= 0,
          "StableHLO Pad interior padding must be non-negative.");
      const CheckedInt<int64_t> interior_step =
          CheckedInt<int64_t>(interior_pad_[i]) + 1;
      TF_LITE_ENSURE_MSG(context, interior_step.Status() == kTfLiteOk,
                         "StableHLO Pad interior padding overflowed.");
      interior_step_[i] = interior_step.Value();
      const CheckedInt<int64_t> output_dim =
          (CheckedInt<int64_t>(dims[i]) - 1) * interior_step + 1 +
          edge_pad_low_[i] + edge_pad_high_[i];
      TF_LITE_ENSURE_MSG(context, output_dim.Status() == kTfLiteOk,
                         "StableHLO Pad output dimension overflowed.");
      output_shape_[i] = output_dim.Value();
      input_dims[i] = dims[i];
    }
    if (absl::c_any_of(absl::MakeConstSpan(output_shape_, rank),
                       [](auto s) { return s <= 0; })) {
      std::memset(input_shape_, 0, sizeof(input_shape_));
      std::memset(output_shape_, 0, sizeof(output_shape_));
      output_size_ = 0;
      return kTfLiteOk;
    }
    for (int i = 0; i < rank; ++i) {
      TF_LITE_ENSURE_MSG(
          context, output_shape_[i] <= std::numeric_limits<int>::max(),
          "StableHLO Pad output dimension does not fit in TfLiteIntArray.");
      output_dims[i] = static_cast<int>(output_shape_[i]);
    }
    size_t input_elements = 0;
    TF_LITE_ENSURE_OK(
        context, CheckedShapeProduct(
                     context, absl::MakeConstSpan(input_dims, rank),
                     "StableHLO Pad input size overflowed.", input_elements));
    size_t input_bytes = 0;
    TF_LITE_ENSURE_MSG(context,
                       MultiplyAndCheckOverflow(
                           input_elements, static_cast<size_t>(element_size),
                           &input_bytes) == kTfLiteOk,
                       "StableHLO Pad input byte size overflowed.");
    size_t output_elements = 0;
    TF_LITE_ENSURE_OK(
        context, CheckedShapeProduct(
                     context, absl::MakeConstSpan(output_dims, rank),
                     "StableHLO Pad output size overflowed.", output_elements));
    size_t output_bytes = 0;
    TF_LITE_ENSURE_MSG(context,
                       MultiplyAndCheckOverflow(
                           output_elements, static_cast<size_t>(element_size),
                           &output_bytes) == kTfLiteOk,
                       "StableHLO Pad output byte size overflowed.");
    TF_LITE_ENSURE_MSG(context,
                       output_bytes <= static_cast<size_t>(
                                           std::numeric_limits<int64_t>::max()),
                       "StableHLO Pad output byte size overflowed.");
    output_size_ = output_bytes;

    // Compute the output size for each dimension.
    //
    // This is different from the output strides because of the interior
    // padding: the output strides take it into account to "jump" over the
    // interior padding elements.
    output_dimension_sizes_[rank - 1] = element_size;
    for (int i = rank - 2; i >= 0; --i) {
      const CheckedInt<int64_t> dimension_size =
          CheckedInt<int64_t>(output_shape_[i + 1]) *
          output_dimension_sizes_[i + 1];
      TF_LITE_ENSURE_MSG(context, dimension_size.Status() == kTfLiteOk,
                         "StableHLO Pad output stride overflowed.");
      output_dimension_sizes_[i] = dimension_size.Value();
    }
    // Compute the output stride for each dimension.
    //
    // This is the stride between two elements that are copied from the input
    // tensor (i.e. not generated by interior padding).
    CheckedInt<int64_t> output_stride =
        CheckedInt<int64_t>(element_size) * interior_step_[rank - 1];
    TF_LITE_ENSURE_MSG(context, output_stride.Status() == kTfLiteOk,
                       "StableHLO Pad output stride overflowed.");
    output_strides_[rank - 1] = output_stride.Value();
    for (int i = rank - 2; i >= 0; --i) {
      output_stride =
          CheckedInt<int64_t>(output_dimension_sizes_[i]) * interior_step_[i];
      TF_LITE_ENSURE_MSG(context, output_stride.Status() == kTfLiteOk,
                         "StableHLO Pad output stride overflowed.");
      output_strides_[i] = output_stride.Value();
    }
    // Compute the output offset from the eventual pads.
    CheckedInt<int64_t> output_offset = 0;
    for (int i = 0; i < rank; ++i) {
      output_offset +=
          CheckedInt<int64_t>(std::max<int64_t>(edge_pad_low_[i], 0)) *
          output_dimension_sizes_[i];
      TF_LITE_ENSURE_MSG(context, output_offset.Status() == kTfLiteOk,
                         "StableHLO Pad output offset overflowed.");
    }
    output_offset_ = output_offset.Value();
    // Compute input strides.
    input_strides_[rank - 1] = element_size;
    for (int i = rank - 1; i >= 1; --i) {
      const CheckedInt<int64_t> input_stride =
          CheckedInt<int64_t>(dims[i]) * input_strides_[i];
      TF_LITE_ENSURE_MSG(context, input_stride.Status() == kTfLiteOk,
                         "StableHLO Pad input stride overflowed.");
      input_strides_[i - 1] = input_stride.Value();
    }
    // Helper that computes the division between a negative num and a positive
    // denum, rounding away from 0, or returns 0 if num is positive.
    auto DivNegRoundAwayOrZero = [](int64_t num, int64_t denum,
                                    int64_t* result) -> TfLiteStatus {
      if (denum <= 0) {
        return kTfLiteError;
      }
      if (num >= 0) {
        *result = 0;
        return kTfLiteOk;
      }
      const CheckedInt<int64_t> numerator =
          CheckedInt<int64_t>(num) - denum + 1;
      if (numerator.Status() != kTfLiteOk) {
        return kTfLiteError;
      }
      *result = numerator.Value() / denum;
      return kTfLiteOk;
    };
    // Compute the input bounds from the eventual crops.
    //
    // If negative padding is applied, we can treat this as copying a subtensor
    // of the input. We modify the input shape in place as we don't use it for
    // anything else.
    for (int i = 0; i < rank; ++i) {
      int64_t low_crop = 0;
      int64_t high_crop = 0;
      TF_LITE_ENSURE_MSG(
          context,
          DivNegRoundAwayOrZero(edge_pad_low_[i], interior_step_[i],
                                &low_crop) == kTfLiteOk &&
              DivNegRoundAwayOrZero(edge_pad_high_[i], interior_step_[i],
                                    &high_crop) == kTfLiteOk,
          "StableHLO Pad crop computation overflowed.");
      const CheckedInt<int64_t> input_shape =
          CheckedInt<int64_t>(dims[i]) + low_crop + high_crop;
      TF_LITE_ENSURE_MSG(context, input_shape.Status() == kTfLiteOk,
                         "StableHLO Pad input shape overflowed.");
      input_shape_[i] = input_shape.Value();
    }
    has_input_copy_ = absl::c_all_of(absl::MakeConstSpan(input_shape_, rank),
                                     [](auto s) { return s > 0; });
    if (!has_input_copy_) {
      return kTfLiteOk;
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
    CheckedInt<int64_t> input_offset = 0;
    for (int i = 0; i < rank; ++i) {
      int64_t low_crop = 0;
      TF_LITE_ENSURE_MSG(
          context,
          DivNegRoundAwayOrZero(edge_pad_low_[i], interior_step_[i],
                                &low_crop) == kTfLiteOk,
          "StableHLO Pad crop computation overflowed.");
      input_offset -= CheckedInt<int64_t>(low_crop) * input_strides_[i];
      TF_LITE_ENSURE_MSG(context, input_offset.Status() == kTfLiteOk,
                         "StableHLO Pad input offset overflowed.");
      if (edge_pad_low_[i] < 0) {
        int64_t tmp_offset =
            ((interior_step_[i] + edge_pad_low_[i]) % interior_step_[i]);
        if (tmp_offset < 0) {
          tmp_offset += interior_step_[i];
        }
        output_offset +=
            CheckedInt<int64_t>(tmp_offset) * output_dimension_sizes_[i];
        TF_LITE_ENSURE_MSG(context, output_offset.Status() == kTfLiteOk,
                           "StableHLO Pad output offset overflowed.");
      }
    }
    input_offset_ = input_offset.Value();
    output_offset_ = output_offset.Value();
    TF_LITE_ENSURE_MSG(
        context,
        input_offset_ >= 0 && static_cast<size_t>(input_offset_) < input_bytes,
        "StableHLO Pad input offset is out of bounds.");
    TF_LITE_ENSURE_MSG(context,
                       output_offset_ >= 0 &&
                           static_cast<size_t>(output_offset_) < output_size_,
                       "StableHLO Pad output offset is out of bounds.");
    return kTfLiteOk;
  }

  void Apply(const char* input, const char* padding_value, char* output) const {
    // Fill the output tensor with the padding value.
    FillBuffer(
        absl::MakeSpan(output, output_size_),
        absl::MakeConstSpan(padding_value, static_cast<size_t>(element_size_)));
    if (output_size_ == 0 || !has_input_copy_) {
      return;
    }
    const int rank = static_cast<int>(rank_);
    StridedCopy(input + input_offset_, absl::MakeConstSpan(input_shape_, rank),
                absl::MakeConstSpan(input_strides_, rank),
                output + output_offset_,
                absl::MakeConstSpan(output_strides_, rank), element_size_,
                /*depth=*/0);
  }

  TfLiteIntArray* BuildOuputTensorDims() const {
    const int rank = static_cast<int>(rank_);
    TfLiteIntArray* dims = TfLiteIntArrayCreate(rank);
    if (dims == nullptr) {
      return nullptr;
    }
    for (int i = 0; i < rank; ++i) {
      dims->data[i] = static_cast<int>(output_shape_[i]);
    }
    return dims;
  }

 private:
  int64_t edge_pad_low_[kMaxDims];
  int64_t edge_pad_high_[kMaxDims];
  int64_t interior_pad_[kMaxDims];
  int64_t interior_step_[kMaxDims];
  int64_t rank_ = 0;
  int64_t element_size_ = 0;
  int64_t input_shape_[kMaxDims];
  int64_t output_shape_[kMaxDims];
  int64_t input_strides_[kMaxDims];
  int64_t output_strides_[kMaxDims];
  int64_t output_dimension_sizes_[kMaxDims];
  int64_t input_offset_ = 0;
  int64_t output_offset_ = 0;
  size_t output_size_ = 0;
  bool has_input_copy_ = false;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  if (options == nullptr) {
    return nullptr;
  }
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
  int padding_value_count = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(padding_value_tensor,
                                        padding_value_count) == kTfLiteOk,
                     "StableHLO Pad padding value size overflowed.");
  TF_LITE_ENSURE_EQ(context, padding_value_count, 1);
  // PadData computations.
  size_t element_size;
  TF_LITE_ENSURE(context, GetSizeOfType(context, input_tensor->type,
                                        &element_size) == kTfLiteOk);
  TF_LITE_ENSURE_MSG(
      context,
      element_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "StableHLO Pad element size overflowed.");
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  PadData& pad_data = *reinterpret_cast<PadData*>(node->user_data);
  TF_LITE_ENSURE_OK(
      context, pad_data.Setup(context,
                              absl::MakeConstSpan(input_tensor->dims->data,
                                                  input_tensor->dims->size),
                              static_cast<int64_t>(element_size)));
  // Output tensor setup.
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, PadData::kOutput, &output_tensor));
  TF_LITE_ENSURE(context, input_tensor->type == output_tensor->type);
  TfLiteIntArray* output_dims = pad_data.BuildOuputTensorDims();
  TF_LITE_ENSURE(context, output_dims != nullptr);
  return context->ResizeTensor(context, output_tensor, output_dims);
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
  TF_LITE_ENSURE(context, input_tensor->type == output_tensor->type);
  TF_LITE_ENSURE(context, node->user_data != nullptr);
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
