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

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const RuntimeShape& unextended_output_shape,
                         SequentialTensorWriter<T>* writer) {
  ruy::profiler::ScopeLabel label("StridedSlice");

  // Note that the output_shape is not used herein.
  tflite::StridedSliceParams params_copy = op_params;

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 5);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(5, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(5, unextended_output_shape);

  // Reverse and pad to 5 dimensions because that is what the runtime code
  // requires (ie. all shapes must be 5D and are given backwards).
  strided_slice::StridedSlicePadIndices(&params_copy, 5);

  const int start_0 =
      strided_slice::StridedSliceStartForAxis(params_copy, input_shape, 0);
  const int stop_0 = strided_slice::StridedSliceEndForAxis(
      params_copy, input_shape, 0, start_0);
  const int start_1 =
      strided_slice::StridedSliceStartForAxis(params_copy, input_shape, 1);
  const int stop_1 = strided_slice::StridedSliceEndForAxis(
      params_copy, input_shape, 1, start_1);
  const int start_2 =
      strided_slice::StridedSliceStartForAxis(params_copy, input_shape, 2);
  const int stop_2 = strided_slice::StridedSliceEndForAxis(
      params_copy, input_shape, 2, start_2);
  const int start_3 =
      strided_slice::StridedSliceStartForAxis(params_copy, input_shape, 3);
  const int stop_3 = strided_slice::StridedSliceEndForAxis(
      params_copy, input_shape, 3, start_3);
  const int start_4 =
      strided_slice::StridedSliceStartForAxis(params_copy, input_shape, 4);
  const int stop_4 = strided_slice::StridedSliceEndForAxis(
      params_copy, input_shape, 4, start_4);

  auto lc = [&](int end, int stride, int index) {
    if (stride < 0) {
      return index > end;
    } else {
      return index < end;
    }
  };
  // With a static_cast it is not possible to initialize
  // a variable of type 'const int *'
  // with an rvalue of type 'const int32_t *' (aka 'const long *').
  // reinterpret_cast is required to handle this casting.
  const int* shape = reinterpret_cast<const int*>(input_shape.DimsData());
  const int* stride = reinterpret_cast<const int*>(params_copy.strides);
  const bool inner_stride_is_1 = params_copy.strides[4] == 1;

  for (int offset_0 = start_0; lc(stop_0, stride[0], offset_0);
       offset_0 += stride[0]) {
    for (int offset_1 = start_1; lc(stop_1, stride[1], offset_1);
         offset_1 += stride[1]) {
      for (int offset_2 = start_2; lc(stop_2, stride[2], offset_2);
           offset_2 += stride[2]) {
        for (int offset_3 = start_3; lc(stop_3, stride[3], offset_3);
             offset_3 += stride[3]) {
          // When the stride is 1, the inner loop is equivalent to the
          // optimized slice inner loop. Otherwise, it is identical to the
          // strided_slice reference implementation inner loop.
          if (inner_stride_is_1) {
            const int len = stop_4 - start_4;
            int index = start_4 + offset_3 * shape[4] +
                        offset_2 * shape[3] * shape[4] +
                        offset_1 * shape[2] * shape[3] * shape[4] +
                        offset_0 * shape[1] * shape[2] * shape[3] * shape[4];
            if (len > 0) {
              writer->WriteN(index, len);
            }
          } else {
            for (int offset_4 = start_4; lc(stop_4, stride[4], offset_4);
                 offset_4 += stride[4]) {
              int index = offset_4 + offset_3 * shape[4] +
                          offset_2 * shape[3] * shape[4] +
                          offset_1 * shape[2] * shape[3] * shape[4] +
                          offset_0 * shape[1] * shape[2] * shape[3] * shape[4];
              writer->Write(index);
            }
          }
        }
      }
    }
  }
}

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  SequentialTensorWriter<T> writer(input_data, output_data);
  StridedSlice<T>(op_params, unextended_input_shape, unextended_output_shape,
                  &writer);
}

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const TfLiteTensor* input,
                         const RuntimeShape& unextended_output_shape,
                         TfLiteTensor* output) {
  SequentialTensorWriter<T> writer(input, output);
  StridedSlice<T>(op_params, unextended_input_shape, unextended_output_shape,
                  &writer);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_
