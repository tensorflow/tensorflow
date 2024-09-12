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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GATHER_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GATHER_H_

#include <cstdint>
#include <cstring>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T, typename CoordsT = int32_t>
inline TfLiteStatus Gather(const tflite::GatherParams& op_params,
                           const RuntimeShape& input_shape, const T* input_data,
                           const RuntimeShape& coords_shape,
                           const CoordsT* coords_data,
                           const RuntimeShape& output_shape, T* output_data,
                           bool int4_input = false) {
  ruy::profiler::ScopeLabel label("Gather");
  int axis = op_params.axis;
  if (axis < 0) {
    axis += input_shape.DimensionsCount();
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, input_shape.DimensionsCount());

  int batch_dims = op_params.batch_dims;
  if (batch_dims < 0) {
    batch_dims += coords_shape.DimensionsCount();
  }
  TFLITE_DCHECK_GE(batch_dims, 0);
  TFLITE_DCHECK_LT(batch_dims, input_shape.DimensionsCount());
  TFLITE_DCHECK_LE(batch_dims, coords_shape.DimensionsCount());
  TFLITE_DCHECK_GE(axis, batch_dims);
  for (int i = 0; i < batch_dims; ++i) {
    TFLITE_DCHECK_EQ(input_shape.Dims(i), coords_shape.Dims(i));
  }

  const int axis_size = input_shape.Dims(axis);

  int batch_size = 1;
  for (int i = 0; i < batch_dims; ++i) {
    batch_size *= input_shape.Dims(i);
  }

  int outer_size = 1;
  for (int i = batch_dims; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }

  int inner_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i) {
    inner_size *= input_shape.Dims(i);
  }

  int input_flat_size = input_shape.FlatSize();
  int output_flat_size = output_shape.FlatSize();

  if (int4_input) {
    // TODO(b/298210669) It doesn't handle the case when sizes are not
    // divisible by 2.
    TFLITE_DCHECK_EQ(inner_size % 2, 0);
    inner_size /= 2;
    TFLITE_DCHECK_EQ(input_flat_size % 2, 0);
    input_flat_size /= 2;
    TFLITE_DCHECK_EQ(output_flat_size % 2, 0);
    output_flat_size /= 2;
  }

  int coord_size = 1;
  for (int i = batch_dims; i < coords_shape.DimensionsCount(); ++i) {
    coord_size *= coords_shape.Dims(i);
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int outer = 0; outer < outer_size; ++outer) {
      for (int i = 0; i < coord_size; ++i) {
        // TODO(rsun): replace memcpy with a for loop
        const int64_t coord = coords_data[batch * coord_size + i];
        if (coord < 0 || coord >= axis_size) {
          return kTfLiteError;
        }
        const int64_t from_pos =
            (((batch * outer_size) + outer) * axis_size + coord) * inner_size;
        TFLITE_DCHECK(from_pos >= 0);
        TFLITE_DCHECK(from_pos + inner_size <= input_flat_size);
        const int64_t to_pos =
            (((batch * outer_size) + outer) * coord_size + i) * inner_size;
        TFLITE_DCHECK(to_pos >= 0);
        TFLITE_DCHECK(to_pos + inner_size <= output_flat_size);
        std::memcpy(&output_data[to_pos], &input_data[from_pos],
                    sizeof(T) * inner_size);
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GATHER_H_
