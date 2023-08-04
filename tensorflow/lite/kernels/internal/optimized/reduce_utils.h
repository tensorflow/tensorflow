/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_UTILS_H_

#include <stdint.h>

#include <algorithm>
#include <cstring>

namespace tflite {
namespace reduce_utils {

inline void RemoveSize1Dims(int* shape_out, int& out_num_dims, int* axis_out,
                            int& out_num_axis) {
  for (int64_t i = 0; i < out_num_dims;) {
    if (shape_out[i] == 1) {
      for (int64_t j = i + 1; j < out_num_dims; ++j) {
        shape_out[j - 1] = shape_out[j];
      }
      for (int64_t j = 0; j < out_num_axis; ++j) {
        if (axis_out[j] == i) {
          for (int64_t k = j + 1; k < out_num_axis; ++k) {
            axis_out[k - 1] = axis_out[k];
          }
          out_num_axis -= 1;
          break;
        }
      }
      for (int64_t j = 0; j < out_num_axis; ++j) {
        if (axis_out[j] > i) {
          axis_out[j] -= 1;
        }
      }
      --out_num_dims;
    } else {
      ++i;
    }
  }
}

// This method parses the input 'axis' to remove duplicates, handle negative
// values and remove redundant dimensions. It returns a valid 'axis_out' and
// 'shape_out' contains the flattened input shape. 'out_num_dims' contains the
// reduced number of dimensions.
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* axis_out,
                        int& out_num_axis, const int* shape_in, int* shape_out,
                        int& out_num_dims) {
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    out_num_axis = 0;
    out_num_dims = 0;
    return true;
  }
  out_num_axis = 0;
  out_num_dims = num_dims;
  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    // Handle negative index. A positive index 'p_idx' can be represented as a
    // negative index 'n_idx' as: n_idx = p_idx-num_dims
    // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]  */
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    if (current < 0 || current >= num_dims) {
      return false;
    }
    bool is_dup = false;
    for (int j = 0; j < out_num_axis; ++j) {
      if (axis_out[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      axis_out[out_num_axis] = current;
      out_num_axis += 1;
    }
  }
  // If two or more adjacent dimensions are either reduced
  // over or not, then the second and subsequent dimensions may be flattened.
  memcpy(shape_out, shape_in, num_dims * sizeof(int));
  std::sort(&axis_out[0], &axis_out[out_num_axis]);

  RemoveSize1Dims(shape_out, out_num_dims, axis_out, out_num_axis);
  if (out_num_axis > 0) {
    int64_t j = out_num_axis - 1;
    // true if the previous index is present in axis_out.
    bool previous_here = (axis_out[j] == out_num_dims - 1);
    if (previous_here) {
      j -= 1;
    }

    for (int64_t i = out_num_dims - 2; i >= 0; --i) {
      // true if the current index is present in axis_out.
      bool current_here = j >= 0 ? (axis_out[j] == i) : false;
      if (current_here == previous_here) {
        shape_out[i] *= shape_out[i + 1];
        for (int64_t k = i + 1; k + 1 < out_num_dims; ++k) {
          shape_out[k] = shape_out[k + 1];
        }
        // All axis bigger than this need to be reduced by 1.
        for (int64_t k = 0; k < out_num_axis; ++k) {
          if (axis_out[k] > i) {
            axis_out[k] -= 1;
          }
        }
        if (current_here) {
          for (int64_t k = j + 1; k + 1 < out_num_axis; ++k) {
            axis_out[k] = axis_out[k + 1];
          }
          out_num_axis -= 1;
        }
        out_num_dims -= 1;
      }
      if (current_here) {
        j -= 1;
      }
      previous_here = current_here;
    }
  }
  return true;
}
}  // namespace reduce_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_UTILS_H_
