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
						 const RuntimeShape& input_shape,
						 const T* input_data,
						 const RuntimeShape& output_shape,
						 T* output_data) {
  const int slice_dimensions = input_shape.DimensionsCount();
  T* out_ptr = output_data;

  std::vector<int> start;
  std::vector<int> stop;
  int start_i;

  // Compute Start & Stop for all axes
  for (int i = 0; i < op_params.start_indices_count; i++) {
	start_i = strided_slice::StartForAxis(op_params, input_shape, i);
	start.emplace_back(start_i);
	stop.emplace_back(strided_slice::StopForAxis(op_params, input_shape, i, start_i));
  }

  std::function<void(int, int)> compute_strided_slice = [&compute_strided_slice, slice_dimensions, input_data, input_shape, start, stop, op_params, &out_ptr](int axis, int curr_indices) {
	if (axis == (slice_dimensions - 1)) {
	  for (int in_x = start[axis]; !strided_slice::LoopCondition(in_x, stop[axis], op_params.strides[axis]); in_x += op_params.strides[axis]) {
		int index = in_x + curr_indices;
		*out_ptr++ = input_data[index];
	  }
	  return;
	}
	else {
	  for (int in_x = start[axis]; !strided_slice::LoopCondition(in_x, stop[axis], op_params.strides[axis]); in_x += op_params.strides[axis]) {
		int indices = (in_x + curr_indices) * input_shape.Dims(axis + 1);
		compute_strided_slice(axis + 1, indices);
	  }
	  return;
	}
  };

  compute_strided_slice(0, 0);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_
