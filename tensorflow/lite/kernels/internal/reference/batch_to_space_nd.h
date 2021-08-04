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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_TO_SPACE_ND_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_TO_SPACE_ND_H_

#include <cmath>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

// TODO(b/135760455): Move this method anonymous namespace in a cc file.
inline RuntimeShape ExtendShapeBatchToSpace(const RuntimeShape& shape) {
  if (shape.DimensionsCount() == 4) {
    return shape;
  }
  RuntimeShape new_shape(4, 1);
  new_shape.SetDim(0, shape.Dims(0));
  new_shape.SetDim(1, shape.Dims(1));
  new_shape.SetDim(3, shape.Dims(2));
  return new_shape;
}

template <typename T>
inline void BatchToSpaceND(const RuntimeShape& unextended_input1_shape,
                           const T* input1_data,
                           const RuntimeShape& unextended_input2_shape,
                           const int32_t* block_shape_data,
                           const RuntimeShape& unextended_input3_shape,
                           const int32_t* crops_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  ruy::profiler::ScopeLabel label("BatchToSpaceND");
  TFLITE_DCHECK_GE(unextended_input1_shape.DimensionsCount(), 3);
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(unextended_input1_shape.DimensionsCount(),
                   unextended_output_shape.DimensionsCount());

  const RuntimeShape input1_shape =
      ExtendShapeBatchToSpace(unextended_input1_shape);
  const RuntimeShape output_shape =
      ExtendShapeBatchToSpace(unextended_output_shape);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int block_shape_height = block_shape_data[0];
  const int block_shape_width =
      unextended_input1_shape.DimensionsCount() == 4 ? block_shape_data[1] : 1;
  const int crops_top = crops_data[0];
  const int crops_left =
      unextended_input1_shape.DimensionsCount() == 4 ? crops_data[2] : 0;
  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch) {
    const int out_batch = in_batch % output_batch_size;
    const int spatial_offset = in_batch / output_batch_size;
    for (int in_h = 0; in_h < input_height; ++in_h) {
      const int out_h = in_h * block_shape_height +
                        spatial_offset / block_shape_width - crops_top;
      if (out_h < 0 || out_h >= output_height) {
        continue;
      }
      for (int in_w = 0; in_w < input_width; ++in_w) {
        const int out_w = in_w * block_shape_width +
                          spatial_offset % block_shape_width - crops_left;

        if (out_w < 0 || out_w >= output_width) {
          continue;
        }
        T* out = output_data + Offset(output_shape, out_batch, out_h, out_w, 0);
        const T* in =
            input1_data + Offset(input1_shape, in_batch, in_h, in_w, 0);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_TO_SPACE_ND_H_
