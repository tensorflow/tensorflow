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
#ifndef TENSORFLOW_LITE_KERNELS_PADDING_H_
#define TENSORFLOW_LITE_KERNELS_PADDING_H_

#include <cstdint>
#include <limits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

inline TfLiteStatus CheckedNarrowPaddingValue(int64_t value, int* result) {
  if (result == nullptr || value > std::numeric_limits<int>::max() ||
      value < std::numeric_limits<int>::min()) {
    return kTfLiteError;
  }
  *result = static_cast<int>(value);
  return kTfLiteOk;
}

inline int64_t ComputeEffectiveFilterSize(int filter_size, int dilation_rate) {
  return (static_cast<int64_t>(filter_size) - 1) * dilation_rate + 1;
}

inline TfLiteStatus ValidatePaddingArguments(TfLitePadding padding,
                                             int image_size, int filter_size,
                                             int stride, int dilation_rate) {
  if ((padding != kTfLitePaddingSame && padding != kTfLitePaddingValid) ||
      image_size < 0 || filter_size <= 0 || stride <= 0 || dilation_rate <= 0) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

inline TfLiteStatus ComputePaddingWithOffsetChecked(
    int stride, int dilation_rate, int in_size, int filter_size, int out_size,
    int* offset, int* padding) {
  if (offset == nullptr || padding == nullptr || in_size < 0 ||
      filter_size <= 0 || out_size < 0 || stride <= 0 || dilation_rate <= 0) {
    return kTfLiteError;
  }
  const int64_t effective_filter_size =
      ComputeEffectiveFilterSize(filter_size, dilation_rate);
  int64_t total_padding = ((static_cast<int64_t>(out_size) - 1) * stride +
                           effective_filter_size - in_size);
  total_padding = total_padding > 0 ? total_padding : 0;
  *offset = static_cast<int>(total_padding % 2);
  return CheckedNarrowPaddingValue(total_padding / 2, padding);
}

inline int ComputePadding(int stride, int dilation_rate, int in_size,
                          int filter_size, int out_size) {
  int offset = 0;
  int padding = 0;
  return ComputePaddingWithOffsetChecked(stride, dilation_rate, in_size,
                                         filter_size, out_size, &offset,
                                         &padding) == kTfLiteOk
             ? padding
             : 0;
}

// It's not guaranteed that padding is symmetric. It's important to keep
// offset for algorithms need all paddings.
inline int ComputePaddingWithOffset(int stride, int dilation_rate, int in_size,
                                    int filter_size, int out_size,
                                    int* offset) {
  int padding = 0;
  if (ComputePaddingWithOffsetChecked(stride, dilation_rate, in_size,
                                      filter_size, out_size, offset,
                                      &padding) != kTfLiteOk) {
    if (offset != nullptr) *offset = 0;
    return 0;
  }
  return padding;
}

// Matching GetWindowedOutputSize in TensorFlow.
inline TfLiteStatus ComputeOutSizeChecked(TfLitePadding padding, int image_size,
                                          int filter_size, int stride,
                                          int dilation_rate, int* out_size) {
  if (out_size == nullptr ||
      ValidatePaddingArguments(padding, image_size, filter_size, stride,
                               dilation_rate) != kTfLiteOk) {
    return kTfLiteError;
  }
  const int64_t effective_filter_size =
      ComputeEffectiveFilterSize(filter_size, dilation_rate);

  int64_t value = 0;
  switch (padding) {
    case kTfLitePaddingSame:
      value = (static_cast<int64_t>(image_size) + stride - 1) / stride;
      break;
    case kTfLitePaddingValid:
      value =
          (static_cast<int64_t>(image_size) + stride - effective_filter_size) /
          stride;
      break;
    default:
      return kTfLiteError;
  }
  if (value < 0) return kTfLiteError;
  return CheckedNarrowPaddingValue(value, out_size);
}

inline int ComputeOutSize(TfLitePadding padding, int image_size,
                          int filter_size, int stride, int dilation_rate = 1) {
  int out_size = 0;
  return ComputeOutSizeChecked(padding, image_size, filter_size, stride,
                               dilation_rate, &out_size) == kTfLiteOk
             ? out_size
             : 0;
}

inline TfLiteStatus ComputePaddingHeightWidthChecked(
    int stride_height, int stride_width, int dilation_rate_height,
    int dilation_rate_width, int in_height, int in_width, int filter_height,
    int filter_width, TfLitePadding padding, int* out_height, int* out_width,
    TfLitePaddingValues* padding_values) {
  if (out_height == nullptr || out_width == nullptr ||
      padding_values == nullptr) {
    return kTfLiteError;
  }
  TF_LITE_ENSURE_STATUS(ComputeOutSizeChecked(padding, in_width, filter_width,
                                              stride_width, dilation_rate_width,
                                              out_width));
  TF_LITE_ENSURE_STATUS(
      ComputeOutSizeChecked(padding, in_height, filter_height, stride_height,
                            dilation_rate_height, out_height));

  int offset = 0;
  TF_LITE_ENSURE_STATUS(ComputePaddingWithOffsetChecked(
      stride_height, dilation_rate_height, in_height, filter_height,
      *out_height, &offset, &padding_values->height));
  padding_values->height_offset = offset;
  TF_LITE_ENSURE_STATUS(ComputePaddingWithOffsetChecked(
      stride_width, dilation_rate_width, in_width, filter_width, *out_width,
      &offset, &padding_values->width));
  padding_values->width_offset = offset;
  return kTfLiteOk;
}

inline TfLitePaddingValues ComputePaddingHeightWidth(
    int stride_height, int stride_width, int dilation_rate_height,
    int dilation_rate_width, int in_height, int in_width, int filter_height,
    int filter_width, TfLitePadding padding, int* out_height, int* out_width) {
  TfLitePaddingValues padding_values;
  if (out_height != nullptr) *out_height = 0;
  if (out_width != nullptr) *out_width = 0;
  padding_values.height = 0;
  padding_values.height_offset = 0;
  padding_values.width = 0;
  padding_values.width_offset = 0;
  ComputePaddingHeightWidthChecked(
      stride_height, stride_width, dilation_rate_height, dilation_rate_width,
      in_height, in_width, filter_height, filter_width, padding, out_height,
      out_width, &padding_values);
  return padding_values;
}

inline Padding3DValues ComputePadding3DValues(
    int stride_height, int stride_width, int stride_depth,
    int dilation_rate_height, int dilation_rate_width, int dilation_rate_depth,
    int in_height, int in_width, int in_depth, int filter_height,
    int filter_width, int filter_depth, TfLitePadding padding, int* out_height,
    int* out_width, int* out_depth) {
  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);
  *out_depth = ComputeOutSize(padding, in_depth, filter_depth, stride_depth,
                              dilation_rate_depth);

  Padding3DValues padding_values;
  int offset = 0;
  padding_values.depth =
      ComputePaddingWithOffset(stride_depth, dilation_rate_depth, in_depth,
                               filter_depth, *out_depth, &offset);
  padding_values.depth_offset = offset;
  padding_values.height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  padding_values.height_offset = offset;
  padding_values.width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  padding_values.width_offset = offset;
  return padding_values;
}
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_PADDING_H_
