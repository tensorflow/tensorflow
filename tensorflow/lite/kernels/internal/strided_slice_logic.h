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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_

#include <limits>
#include <vector>
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace strided_slice {

// Use until std::clamp() is available from C++17.
inline int Clamp(const int v, const int lo, const int hi) {
  TFLITE_DCHECK(!(hi < lo));
  if (hi < v) return hi;
  if (v < lo) return lo;
  return v;
}

inline void StridedSlicePadIndices(tflite::StridedSliceParams* p,
                                   int dim_count) {
  // Add indices and mask bits to fully include extra dimensions
  TFLITE_CHECK_LE(dim_count, 5);
  TFLITE_CHECK_GE(dim_count, p->start_indices_count);
  TFLITE_CHECK_EQ(p->start_indices_count, p->stop_indices_count);
  TFLITE_CHECK_EQ(p->stop_indices_count, p->strides_count);

  const int pad_count = dim_count - p->start_indices_count;

  // Pad indices at start, so move arrays by pad_count.
  for (int i = p->start_indices_count - 1; i >= 0; --i) {
    p->strides[i + pad_count] = p->strides[i];
    p->start_indices[i + pad_count] = p->start_indices[i];
    p->stop_indices[i + pad_count] = p->stop_indices[i];
  }
  for (int i = 0; i < pad_count; ++i) {
    p->start_indices[i] = 0;
    p->stop_indices[i] = 1;
    p->strides[i] = 1;
  }

  // Pad masks with 0s or 1s as required.
  p->shrink_axis_mask <<= pad_count;
  p->ellipsis_mask <<= pad_count;
  p->new_axis_mask <<= pad_count;
  p->begin_mask <<= pad_count;
  p->end_mask <<= pad_count;
  p->begin_mask |= (1 << pad_count) - 1;
  p->end_mask |= (1 << pad_count) - 1;

  p->start_indices_count = dim_count;
  p->stop_indices_count = dim_count;
  p->strides_count = dim_count;
}

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axis_size - 1] that can be used to index
// directly into the data.
inline int StartForAxis(const tflite::StridedSliceParams& params,
                        const RuntimeShape& input_shape, int axis) {
  const auto begin_mask = params.begin_mask;
  const auto* start_indices = params.start_indices;
  const auto* strides = params.strides;
  const int axis_size = input_shape.Dims(axis);
  if (axis_size == 0) {
    return 0;
  }
  // Begin with the specified index.
  int start = start_indices[axis];

  // begin_mask override
  if (begin_mask & 1 << axis) {
    if (strides[axis] > 0) {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = std::numeric_limits<int>::lowest();
    } else {
      // Backward iteration - use the last element.
      start = std::numeric_limits<int>::max();
    }
  }

  // Handle negative indices
  if (start < 0) {
    start += axis_size;
  }

  // Clamping
  start = Clamp(start, 0, axis_size - 1);

  return start;
}

// Return the "real" index for the end of iteration along that axis. This is an
// "end" in the traditional C sense, in that it points to one past the last
// element. ie. So if you were iterating through all elements of a 1D array of
// size 4, this function would return 4 as the stop, because it is one past the
// "real" indices of 0, 1, 2 & 3.
inline int StopForAxis(const tflite::StridedSliceParams& params,
                       const RuntimeShape& input_shape, int axis,
                       int start_for_axis) {
  const auto end_mask = params.end_mask;
  const auto shrink_axis_mask = params.shrink_axis_mask;
  const auto* stop_indices = params.stop_indices;
  const auto* strides = params.strides;
  const int axis_size = input_shape.Dims(axis);
  if (axis_size == 0) {
    return 0;
  }

  // Begin with the specified index
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  int stop = stop_indices[axis];

  // When shrinking an axis, the end position does not matter (and can be
  // incorrect when negative indexing is used, see Issue #19260). Always use
  // start_for_axis + 1 to generate a length 1 slice, since start_for_axis has
  // already been adjusted for negative indices.
  if (shrink_axis) {
    stop = start_for_axis + 1;
  }

  // end_mask override
  if (end_mask & (1 << axis)) {
    if (strides[axis] > 0) {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = std::numeric_limits<int>::max();
    } else {
      // Backward iteration - use the first element.
      stop = std::numeric_limits<int>::lowest();
    }
  }

  // Handle negative indices
  if (stop < 0) {
    stop += axis_size;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (strides[axis] > 0) {
    // Forward iteration
    stop = Clamp(stop, 0, axis_size);
  } else {
    // Backward iteration
    stop = Clamp(stop, -1, axis_size - 1);
  }

  return stop;
}

inline bool LoopCondition(int index, int stop, int stride) {
  // True when we have reached the end of an axis and should loop.
  return stride > 0 ? index >= stop : index <= stop;
}

inline tflite::StridedSliceParams BuildStridedSliceParams(
    int begin_mask, int end_mask, int shrink_axis_mask,
    const std::vector<int>& start_indices, const std::vector<int>& stop_indices,
    const std::vector<int>& strides) {
  tflite::StridedSliceParams op_params;
  const int dims_count = start_indices.size();

  op_params.start_indices_count = dims_count;
  op_params.stop_indices_count = dims_count;
  op_params.strides_count = dims_count;
  for (int i = 0; i < dims_count; ++i) {
    op_params.start_indices[i] = start_indices[i];
    op_params.stop_indices[i] = stop_indices[i];
    op_params.strides[i] = strides[i];
  }

  op_params.begin_mask = begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = shrink_axis_mask;

  return op_params;
}

}  // namespace strided_slice

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
