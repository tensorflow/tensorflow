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

namespace tflite {
namespace reduce_utils {

// This method parses the input 'axis' to remove duplicates, handle negative
// values and remove redundant dimensions. It returns a valid 'out_axis' and
// 'shape_out' contains the flattened input shape. 'out_num_dims' contains the
// reduced number of dimensions.
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis, const int* shape_in, int* shape_out,
                        int* out_num_dims) {
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    *out_num_axis = 0;
    *out_num_dims = 0;
    return true;
  }
  int num_out_axis = 0;
  int dims_out = num_dims;
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
    for (int j = 0; j < num_out_axis; ++j) {
      if (out_axis[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      out_axis[num_out_axis] = current;
      num_out_axis += 1;
    }
  }
  // If two or more adjacent dimensions are either reduced
  // over or not, then the second and subsequent dimensions may be flattened.
  memcpy(shape_out, shape_in, num_dims * sizeof(int));
  if (num_out_axis > 0) {
    std::sort(&out_axis[0], &out_axis[num_out_axis]);

    int64_t j = num_out_axis - 1;
    // true if the previous index is present in out_axis.
    bool previous_here = (out_axis[j] == num_dims - 1);
    if (previous_here) {
      j -= 1;
    }

    for (int64_t i = num_dims - 2; i >= 0; --i) {
      // true if the current index is present in out_axis.
      bool current_here = j >= 0 ? (out_axis[j] == i) : false;
      if (current_here == previous_here) {
        shape_out[i] *= shape_out[i + 1];
        for (int64_t k = i + 1; k + 1 < num_dims; ++k) {
          shape_out[k] = shape_out[k + 1];
        }
        // All axis bigger than this need to be reduced by 1.
        for (int64_t k = 0; k < num_out_axis; ++k) {
          if (out_axis[k] > i) {
            out_axis[k] -= 1;
          }
        }
        if (current_here) {
          for (int64_t k = j + 1; k + 1 < num_out_axis; ++k) {
            out_axis[k] = out_axis[k + 1];
          }
          num_out_axis -= 1;
        }
        dims_out -= 1;
      }
      if (current_here) {
        j -= 1;
      }
      previous_here = current_here;
    }
  }
  *out_num_axis = num_out_axis;
  *out_num_dims = dims_out;
  return true;
}
}  // namespace reduce_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_UTILS_H_
