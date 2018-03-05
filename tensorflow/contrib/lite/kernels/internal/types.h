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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TYPES_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TYPES_H_

#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"

namespace tflite {

enum class FusedActivationFunctionType : uint8 { kNone, kRelu6, kRelu1, kRelu };

template <int N>
struct Dims {
  int sizes[N];
  int strides[N];
};

// Gets next index to iterate through a multidimensional array.
inline bool NextIndex(const int num_dims, const int* dims, int* current) {
  TFLITE_DCHECK_GT(num_dims, 0);
  TFLITE_DCHECK(dims != nullptr);
  TFLITE_DCHECK(current != nullptr);
  int carry = 1;
  for (int idx = num_dims - 1; idx >= 0; --idx) {
    int current_val = current[idx] + carry;
    TFLITE_DCHECK_GE(dims[idx], current_val);
    if (dims[idx] == current_val) {
      current[idx] = 0;
    } else {
      current[idx] = current_val;
      carry = 0;
      break;
    }
  }
  return (carry == 0);
}

// Gets offset of index if reducing on axis. When reducing, the flattened offset
// will not change, if the input index changes on the given axis. For example,
// if you have a 3D tensor and you are reducing to 2D by eliminating axis 0,
// then index (0, 1, 2) and index (1, 1, 2) will map to the same flattened
// offset.
// TODO(kanlig): uses Dims to represent dimensions.
inline size_t ReducedOutputOffset(const int num_dims, const int* dims,
                                  const int* index, const int num_axis,
                                  const int* axis) {
  TFLITE_DCHECK_GT(num_dims, 0);
  TFLITE_DCHECK(dims != nullptr);
  TFLITE_DCHECK(index != nullptr);
  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx) {
    // if we need to skip this axis
    bool is_axis = false;
    if (axis != nullptr) {
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (idx == axis[axis_idx]) {
          is_axis = true;
          break;
        }
      }
    }
    if (!is_axis) {
      offset = offset * static_cast<size_t>(dims[idx]) +
               static_cast<size_t>(index[idx]);
    }
  }
  return offset;
}

inline int Offset(const Dims<4>& dims, int i0, int i1, int i2, int i3) {
  TFLITE_DCHECK(i0 >= 0 && i0 < dims.sizes[0]);
  TFLITE_DCHECK(i1 >= 0 && i1 < dims.sizes[1]);
  TFLITE_DCHECK(i2 >= 0 && i2 < dims.sizes[2]);
  TFLITE_DCHECK(i3 >= 0 && i3 < dims.sizes[3]);
  return i0 * dims.strides[0] + i1 * dims.strides[1] + i2 * dims.strides[2] +
         i3 * dims.strides[3];
}

inline int Offset(const Dims<4>& dims, int* index) {
  return Offset(dims, index[0], index[1], index[2], index[3]);
}

// Get array size, DCHECKing that the dim index is in range.
template <int N>
int ArraySize(const Dims<N>& array, int index) {
  TFLITE_DCHECK(index >= 0 && index < N);
  return array.sizes[index];
}

// Get common array size, DCHECKing that they all agree.
template <typename ArrayType1, typename ArrayType2>
int MatchingArraySize(const ArrayType1& array1, int index1,
                      const ArrayType2& array2, int index2) {
  TFLITE_DCHECK_EQ(ArraySize(array1, index1), ArraySize(array2, index2));
  return ArraySize(array1, index1);
}

template <typename ArrayType1, typename ArrayType2, typename... Args>
int MatchingArraySize(const ArrayType1& array1, int index1,
                      const ArrayType2& array2, int index2, Args... args) {
  TFLITE_DCHECK_EQ(ArraySize(array1, index1), ArraySize(array2, index2));
  return MatchingArraySize(array1, index1, args...);
}

inline int RequiredBufferSizeForDims(const Dims<4>& dims) {
  int max_offset = 0;
  for (int i = 0; i < 4; i++) {
    max_offset += (dims.sizes[i] - 1) * dims.strides[i];
  }
  return max_offset + 1;
}

template <int N>
bool IsPackedWithoutStrides(const Dims<N>& dims) {
  int expected_stride = 1;
  for (int d = 0; d < N; d++) {
    if (dims.strides[d] != expected_stride) return false;
    expected_stride *= dims.sizes[d];
  }
  return true;
}

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TYPES_H_
