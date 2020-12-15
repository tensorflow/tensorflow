/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TRANSPOSE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TRANSPOSE_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T, int N>
void TransposeImpl(const TransposeParams& params,
                   const RuntimeShape& unextended_input_shape,
                   const T* input_data,
                   const RuntimeShape& unextended_output_shape,
                   T* output_data) {
  const int unextended_input_size = unextended_input_shape.DimensionsCount();
  const int unextended_output_size = unextended_output_shape.DimensionsCount();
  TFLITE_DCHECK_LE(unextended_input_size, N);
  TFLITE_DCHECK_LE(unextended_output_size, N);
  TFLITE_DCHECK_EQ(unextended_output_size, params.perm_count);
  const int input_ext_size = N - unextended_input_size;
  const int output_ext_size = N - unextended_output_size;
  NdArrayDesc<N> input_desc;
  NdArrayDesc<N> output_desc;
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_input_shape),
                 &input_desc);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                 &output_desc);

  // The perm data is extended to match the output, each index incremented by
  // the amount of front padding of the input shape.
  int extended_perm[N];
  for (int i = 0; i < N; ++i) {
    extended_perm[i] = i < output_ext_size
                           ? i
                           : params.perm[i - output_ext_size] + input_ext_size;
  }

  // Permutes the input shape so we don't need to permute the indexes inside
  // the loop. Check to make sure output_dims is matching input_dims.
  NdArrayDesc<N> perm_input_desc;
  for (int k = 0; k < N; ++k) {
    TFLITE_DCHECK_EQ(input_desc.extents[extended_perm[k]],
                     output_desc.extents[k]);
    perm_input_desc.extents[k] = input_desc.extents[extended_perm[k]];
    perm_input_desc.strides[k] = input_desc.strides[extended_perm[k]];
  }

  // Naive transpose loop (iterate on output index and compute input index).
  auto tranpose_func = [&](int indexes[N]) {
    output_data[SubscriptToIndex(output_desc, indexes)] =
        input_data[SubscriptToIndex(perm_input_desc, indexes)];
  };
  NDOpsHelper<N>(output_desc, tranpose_func);
}

template <typename T, int N = 5>
void Transpose(const TransposeParams& params,
               const RuntimeShape& unextended_input_shape, const T* input_data,
               const RuntimeShape& unextended_output_shape, T* output_data) {
  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (sizeof(T)) {
    case 1:
      TransposeImpl<int8_t, N>(params, unextended_input_shape,
                               reinterpret_cast<const int8_t*>(input_data),
                               unextended_output_shape,
                               reinterpret_cast<int8_t*>(output_data));
      break;
    case 2:
      TransposeImpl<int16_t, N>(params, unextended_input_shape,
                                reinterpret_cast<const int16_t*>(input_data),
                                unextended_output_shape,
                                reinterpret_cast<int16_t*>(output_data));
      break;

    case 4:
      TransposeImpl<int32_t, N>(params, unextended_input_shape,
                                reinterpret_cast<const int32_t*>(input_data),
                                unextended_output_shape,
                                reinterpret_cast<int32_t*>(output_data));
      break;
    case 8:
      TransposeImpl<int64_t, N>(params, unextended_input_shape,
                                reinterpret_cast<const int64_t*>(input_data),
                                unextended_output_shape,
                                reinterpret_cast<int64_t*>(output_data));
      break;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TRANSPOSE_H_
