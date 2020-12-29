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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_MAXIMUM_MINIMUM_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_MAXIMUM_MINIMUM_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T, typename Op, int N = 5>
void MaximumMinimumBroadcastSlow(const RuntimeShape& unextended_input1_shape,
                                 const T* input1_data,
                                 const RuntimeShape& unextended_input2_shape,
                                 const T* input2_data,
                                 const RuntimeShape& unextended_output_shape,
                                 T* output_data, Op op) {
  // Uses element-wise calculation if broadcast is not required.
  if (unextended_input1_shape == unextended_input2_shape) {
    const int flat_size =
        MatchingElementsSize(unextended_input1_shape, unextended_input2_shape,
                             unextended_output_shape);
    for (int i = 0; i < flat_size; ++i) {
      output_data[i] = op(input1_data[i], input2_data[i]);
    }
  } else {
    TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), N);
    TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), N);
    TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), N);

    NdArrayDesc<N> desc1;
    NdArrayDesc<N> desc2;
    NdArrayDesc<N> output_desc;
    NdArrayDescsForElementwiseBroadcast(
        unextended_input1_shape, unextended_input2_shape, &desc1, &desc2);
    CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                   &output_desc);

    auto maxmin_func = [&](int indexes[N]) {
      output_data[SubscriptToIndex(output_desc, indexes)] =
          op(input1_data[SubscriptToIndex(desc1, indexes)],
             input2_data[SubscriptToIndex(desc2, indexes)]);
    };
    NDOpsHelper<N>(output_desc, maxmin_func);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_MAXIMUM_MINIMUM_H_
