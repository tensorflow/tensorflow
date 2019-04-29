/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// Utility functions used for interpolarity between TFLite and ACL
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_UTILS_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"

#include "arm_compute/core/Types.h"

#define ACL_RETURN_ERROR_ON(cond) \
  do {                            \
    if ((cond)) {                 \
      return kTfLiteError;        \
    }                             \
  } while (0)

namespace tflite {
namespace internal {
namespace backends {
namespace acl {

// Maps tflite data type to ACL
arm_compute::DataType map_datatype_to_acl(TfLiteType type);

// Maps tflite tensor dimensions to ACL
arm_compute::TensorShape map_shape_to_acl(const TfLiteIntArray &dims);

// Maps tflite activation type to ACL
arm_compute::ActivationLayerInfo map_activation_info_to_acl(
    TfLiteFusedActivation activation);

// Maps tflite quantization information to ACL
arm_compute::QuantizationInfo map_quantization_info_to_acl(
    const TfLiteQuantization &quantization);

// Maps tflite pad stride info to ACL
arm_compute::PadStrideInfo map_pad_stride_info_to_acl(
    const arm_compute::Size2D &strides, TfLitePadding padding,
    const arm_compute::TensorShape &input_shape,
    const arm_compute::TensorShape &weights_shape);
}  // namespace acl
}  // namespace backends
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_UTILS_H_
