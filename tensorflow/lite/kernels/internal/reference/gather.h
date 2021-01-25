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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_GATHER_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_GATHER_H_

#include <cstring>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

template <typename T, typename CoordsT = int32>
inline void Gather(const tflite::GatherParams& op_params,
                   const RuntimeShape& input_shape, const T* input_data,
                   const RuntimeShape& coords_shape, const CoordsT* coords_data,
                   const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Gather");
  int axis = op_params.axis;
  if (axis < 0) {
    axis += input_shape.DimensionsCount();
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, input_shape.DimensionsCount());
  const int axis_size = input_shape.Dims(axis);
  const int coords_count = coords_shape.FlatSize();

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }

  int inner_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i) {
    inner_size *= input_shape.Dims(i);
  }

  for (int outer = 0; outer < outer_size; ++outer) {
    for (int i = 0; i < coords_count; ++i) {
      TFLITE_DCHECK_GE(coords_data[i], 0);
      TFLITE_DCHECK_LT(coords_data[i], axis_size);
      // TODO(rsun): replace memcpy with a for loop
      std::memcpy(
          output_data + (outer * coords_count + i) * inner_size,
          input_data + (outer * axis_size + coords_data[i]) * inner_size,
          sizeof(T) * inner_size);
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_GATHER_H_
