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

#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_

#include <limits>
#include <vector>
#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"

namespace tflite {

namespace strided_slice {

// Use until std::clamp() is available from C++17.
inline int Clamp(const int v, const int lo, const int hi) {
  TFLITE_DCHECK(!(hi < lo));
  if (hi < v) return hi;
  if (v < lo) return lo;
  return v;
}

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axis_size - 1] that can be used to index
// directly into the data.
template <typename IntType>
inline int StartForAxis(int begin_mask,
                        std::vector<IntType> const& start_indices,
                        std::vector<IntType> const& strides,
                        int const* input_shape, int axis) {
  // Begin with the specified index
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
  int axis_size = input_shape[axis];
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
template <typename IntType>
inline int StopForAxis(int end_mask, int shrink_axis_mask,
                       std::vector<IntType> const& stop_indices,
                       std::vector<IntType> const& strides,
                       int const* input_shape, int axis, int start_for_axis) {
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
  const int axis_size = input_shape[axis];
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

}  // namespace strided_slice

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
