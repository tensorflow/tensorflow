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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SLICE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SLICE_H_

#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape,
                  const RuntimeShape& output_shape,
                  SequentialTensorWriter<T>* writer) {
  const RuntimeShape ext_shape = RuntimeShape::ExtendedShape(5, input_shape);
  TFLITE_DCHECK_LE(op_params.begin_count, 5);
  TFLITE_DCHECK_LE(op_params.size_count, 5);
  const int begin_count = op_params.begin_count;
  const int size_count = op_params.size_count;
  // We front-pad the begin and size vectors.
  int start[5];
  int stop[5];
  for (int i = 0; i < 5; ++i) {
    int padded_i = 5 - i;
    start[i] =
        begin_count < padded_i ? 0 : op_params.begin[begin_count - padded_i];
    stop[i] =
        (size_count < padded_i || op_params.size[size_count - padded_i] == -1)
            ? ext_shape.Dims(i)
            : start[i] + op_params.size[size_count - padded_i];
  }

  for (int i0 = start[0]; i0 < stop[0]; ++i0) {
    for (int i1 = start[1]; i1 < stop[1]; ++i1) {
      for (int i2 = start[2]; i2 < stop[2]; ++i2) {
        for (int i3 = start[3]; i3 < stop[3]; ++i3) {
          for (int i4 = start[4]; i4 < stop[4]; ++i4) {
            writer->Write(Offset(ext_shape, i0, i1, i2, i3, i4));
          }
        }
      }
    }
  }
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  SequentialTensorWriter<T> writer(input_data, output_data);
  return Slice(op_params, input_shape, output_shape, &writer);
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape, const TfLiteTensor* input,
                  const RuntimeShape& output_shape, TfLiteTensor* output) {
  SequentialTensorWriter<T> writer(input, output);
  return Slice(op_params, input_shape, output_shape, &writer);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SLICE_H_
