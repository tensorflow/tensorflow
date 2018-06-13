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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/legacy_reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

inline RuntimeShape DimsToShape(const tflite::Dims<4>& dims) {
  return RuntimeShape(
      {dims.sizes[3], dims.sizes[2], dims.sizes[1], dims.sizes[0]});
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  return L2Normalization<Ac>(input_data, DimsToShape(input_dims), output_data,
                             DimsToShape(output_dims));
}

inline void L2Normalization(const uint8* input_data, const Dims<4>& input_dims,
                            int32 input_zero_point, uint8* output_data,
                            const Dims<4>& output_dims) {
  return L2Normalization(input_data, DimsToShape(input_dims), input_zero_point,
                         output_data, DimsToShape(output_dims));
}

}  // namespace optimized_ops
}  // namespace tflite
#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_
