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

#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

absl::Status CheckKernels(int kernel_h, int kernel_w) {
  if (kernel_h <= 0 || kernel_w <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect kernel values: kernel_height = ", kernel_h,
                     ", kernel_width = ", kernel_w));
  }
  return absl::OkStatus();
}

absl::Status CheckKernelsAndStrides(int kernel_h, int kernel_w, int strides_h,
                                    int strides_w) {
  RETURN_IF_ERROR(CheckKernels(kernel_h, kernel_w));
  RETURN_IF_ERROR(CheckStrides(strides_h, strides_w));
  return absl::OkStatus();
}

absl::Status CheckMaxSupportedOpVersion(const TfLiteRegistration* registration,
                                        int max_version) {
  const int op_version = registration->version;
  if (op_version > max_version) {
    return absl::UnimplementedError(
        absl::StrCat("Max version supported: ", max_version,
                     ". Requested version ", op_version, "."));
  }
  return absl::OkStatus();
}

absl::Status CheckStrides(int strides_h, int strides_w) {
  if (strides_h <= 0 || strides_w <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect stride values: stride_height = ", strides_h,
                     ", stride_width = ", strides_w));
  }
  return absl::OkStatus();
}

absl::Status CheckTensorIsAvailable(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node, int idx) {
  // If tensor id is in range, it's guaranteed that it'll be available.
  if (idx >= tflite_node->inputs->size) {
    return absl::OutOfRangeError(
        absl::StrCat("Requested index goes beyond array size: ", idx, " vs ",
                     idx, tflite_node->inputs->size));
  }
  return absl::OkStatus();
}

HW ToHW(int32_t h, int32_t w) { return HW(h > 0 ? h : 1, w > 0 ? w : 1); }

absl::Status ParsePoolingAttributes(const TfLitePoolParams* tf_options,
                                    const BHWC& input_shape,
                                    Pooling2DAttributes* attr) {
  attr->kernel = ToHW(tf_options->filter_height, tf_options->filter_width);
  attr->strides = ToHW(tf_options->stride_height, tf_options->stride_width);
  UpdatePadding(tf_options->padding, input_shape, attr);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
