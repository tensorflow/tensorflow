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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {
template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  // Note that the output_shape is not used herein.
  tflite::StridedSliceParams params_copy = op_params;

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  // Reverse and pad to 4 dimensions because that is what the runtime code
  // requires (ie. all shapes must be 4D and are given backwards).
  strided_slice::StridedSlicePadIndices(&params_copy, 4);

  const int start_b = strided_slice::StartForAxis(params_copy, input_shape, 0);
  const int stop_b =
      strided_slice::StopForAxis(params_copy, input_shape, 0, start_b);
  const int start_h = strided_slice::StartForAxis(params_copy, input_shape, 1);
  const int stop_h =
      strided_slice::StopForAxis(params_copy, input_shape, 1, start_h);
  const int start_w = strided_slice::StartForAxis(params_copy, input_shape, 2);
  const int stop_w =
      strided_slice::StopForAxis(params_copy, input_shape, 2, start_w);
  const int start_d = strided_slice::StartForAxis(params_copy, input_shape, 3);
  const int stop_d =
      strided_slice::StopForAxis(params_copy, input_shape, 3, start_d);

  T* out_ptr = output_data;
  for (int in_b = start_b;
       !strided_slice::LoopCondition(in_b, stop_b, params_copy.strides[0]);
       in_b += params_copy.strides[0]) {
    for (int in_h = start_h;
         !strided_slice::LoopCondition(in_h, stop_h, params_copy.strides[1]);
         in_h += params_copy.strides[1]) {
      for (int in_w = start_w;
           !strided_slice::LoopCondition(in_w, stop_w, params_copy.strides[2]);
           in_w += params_copy.strides[2]) {
        for (int in_d = start_d; !strided_slice::LoopCondition(
                 in_d, stop_d, params_copy.strides[3]);
             in_d += params_copy.strides[3]) {
          *out_ptr++ = input_data[Offset(input_shape, in_b, in_h, in_w, in_d)];
        }
      }
    }
  }
}
}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_
