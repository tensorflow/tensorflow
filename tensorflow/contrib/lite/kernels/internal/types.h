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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TYPES_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TYPES_H_

#include <cstring>
#include <iterator>

#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"

namespace tflite {

enum class FusedActivationFunctionType : uint8 { kNone, kRelu6, kRelu1, kRelu };
enum class PaddingType { kNone, kSame, kValid };

// Quantization parameters, determining the mapping of quantized values
// to real values (i.e. determining how quantized values are mathematically
// interpreted).
//
// The correspondence is as follows:
//
//   real_value = scale * (quantized_value - zero_point);
//
// In other words, zero_point designates which quantized value corresponds to
// the real 0 value, and scale designates the difference between the real values
// corresponding to consecutive quantized values differing by 1.
struct QuantizationParams {
  int32 zero_point = 0;
  double scale = 0.0;
};

template <int N>
struct Dims {
  int sizes[N];
  int strides[N];
};

class RuntimeShape {
 public:
  // Shapes with dimensions up to 4 are stored directly in the structure, while
  // larger shapes are separately allocated.
  static constexpr int kMaxSmallSize = 4;

  RuntimeShape() : size_(0) {}

  explicit RuntimeShape(int dimensions_count) : size_(dimensions_count) {
    if (dimensions_count > kMaxSmallSize) {
      dims_pointer_ = new int32[dimensions_count];
    }
  }

  RuntimeShape(int dimensions_count, const int32* dims_data) : size_(0) {
    ReplaceWith(dimensions_count, dims_data);
  }

  RuntimeShape(const std::initializer_list<int> init_list) : size_(0) {
    BuildFrom(init_list);
  }

  ~RuntimeShape() {
    if (size_ > kMaxSmallSize) {
      delete[] dims_pointer_;
    }
  }

  inline int32 DimensionsCount() const { return size_; }
  inline int32 Dims(int i) const {
    TFLITE_DCHECK_GE(i, 0);
    TFLITE_DCHECK_LT(i, size_);
    return size_ > kMaxSmallSize ? dims_pointer_[i] : dims_[i];
  }
  inline void SetDim(int i, int32 val) {
    TFLITE_DCHECK_GE(i, 0);
    TFLITE_DCHECK_LT(i, size_);
    if (size_ > kMaxSmallSize) {
      dims_pointer_[i] = val;
    } else {
      dims_[i] = val;
    }
  }
  inline int32* DimsData() {
    return size_ > kMaxSmallSize ? dims_pointer_ : dims_;
  }
  inline const int32* DimsData() const {
    return size_ > kMaxSmallSize ? dims_pointer_ : dims_;
  }

  inline void Resize(int dimensions_count) {
    if (size_ > kMaxSmallSize) {
      delete[] dims_pointer_;
    }
    size_ = dimensions_count;
    if (dimensions_count > kMaxSmallSize) {
      dims_pointer_ = new int32[dimensions_count];
    }
  }

  inline void ReplaceWith(int dimensions_count, const int32* dims_data) {
    Resize(dimensions_count);
    int32* dst_dims = DimsData();
    std::memcpy(dst_dims, dims_data, dimensions_count * sizeof(int32));
  }

  template <typename T>
  inline void BuildFrom(const T& src_iterable) {
    const int dimensions_count =
        std::distance(src_iterable.begin(), src_iterable.end());
    Resize(dimensions_count);
    int32* data = DimsData();
    for (auto it : src_iterable) {
      *data = it;
      ++data;
    }
  }

  inline void BuildFrom(const std::initializer_list<int> init_list) {
    BuildFrom<const std::initializer_list<int>>(init_list);
  }

  // Returns the total count of elements, that is the size when flattened into a
  // vector.
  inline int FlatSize() const {
    int buffer_size = 1;
    const int* dims_data = DimsData();
    for (int i = 0; i < size_; i++) {
      const int dim = dims_data[i];
      TFLITE_DCHECK_GE(dim, 1);
      buffer_size *= dim;
    }
    return buffer_size;
  }

 private:
  int32 size_;
  union {
    int32 dims_[kMaxSmallSize];
    int32* dims_pointer_;
  };
};

// Converts inference-style shape to legacy tflite::Dims<4>.
inline tflite::Dims<4> ToRuntimeDims(const tflite::RuntimeShape& array_shape) {
  tflite::Dims<4> result;
  const int dimensions_count = array_shape.DimensionsCount();
  TFLITE_CHECK_LE(dimensions_count, 4);
  int cum_prod = 1;
  for (int i = 0; i < 4; i++) {
    const int new_dim =
        (i < dimensions_count) ? array_shape.Dims(dimensions_count - 1 - i) : 1;
    result.sizes[i] = new_dim;
    result.strides[i] = cum_prod;
    cum_prod *= new_dim;
  }
  return result;
}

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

inline int Offset(const RuntimeShape& shape, int i0, int i1, int i2, int i3) {
  TFLITE_DCHECK(i0 >= 0 && i0 < shape.Dims(0));
  TFLITE_DCHECK(i1 >= 0 && i1 < shape.Dims(1));
  TFLITE_DCHECK(i2 >= 0 && i2 < shape.Dims(2));
  TFLITE_DCHECK(i3 >= 0 && i3 < shape.Dims(3));
  const int* dims_data = shape.DimsData();
  return ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3;
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
//
// Note that this will be phased out with Dims<4>, since RuntimeShape::Dims()
// already performs this check.
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

// Get common shape dim, DCHECKing that they all agree.
inline int MatchingDim(const RuntimeShape& shape1, int index1,
                       const RuntimeShape& shape2, int index2) {
  TFLITE_DCHECK_EQ(shape1.Dims(index1), shape2.Dims(index2));
  return shape1.Dims(index1);
}

template <typename... Args>
int MatchingDim(const RuntimeShape& shape1, int index1,
                const RuntimeShape& shape2, int index2, Args... args) {
  TFLITE_DCHECK_EQ(shape1.Dims(index1), shape2.Dims(index2));
  return MatchingDim(shape1, index1, args...);
}

// Will be phased out with Dims<4>, replaced by RuntimeShape::FlatSize().
template <int N>
inline int FlatSize(const Dims<N>& dims) {
  int flat_size = 1;
  for (int i = 0; i < N; ++i) {
    flat_size *= dims.sizes[i];
  }
  return flat_size;
}

// Deprecated. Prefer FlatSize.
inline int RequiredBufferSizeForDims(const Dims<4>& dims) {
  return FlatSize(dims);
}

// Flat size calculation, checking that dimensions match with one or more other
// arrays.
inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return shape.FlatSize();
}

inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0,
                            const RuntimeShape& check_shape_1) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return MatchingFlatSize(shape, check_shape_1);
}

inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0,
                            const RuntimeShape& check_shape_1,
                            const RuntimeShape& check_shape_2) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return MatchingFlatSize(shape, check_shape_1, check_shape_2);
}

inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0,
                            const RuntimeShape& check_shape_1,
                            const RuntimeShape& check_shape_2,
                            const RuntimeShape& check_shape_3) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return MatchingFlatSize(shape, check_shape_1, check_shape_2, check_shape_3);
}

// Flat size calculation, checking that dimensions match with one or more other
// arrays.
template <int N>
inline int MatchingFlatSize(const Dims<N>& dims, const Dims<N>& check_dims_0) {
  for (int i = 0; i < N; ++i) {
    TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return FlatSize(dims);
}

template <int N>
inline int MatchingFlatSize(const Dims<N>& dims, const Dims<N>& check_dims_0,
                            const Dims<N>& check_dims_1) {
  for (int i = 0; i < N; ++i) {
    TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return MatchingFlatSize(dims, check_dims_1);
}

template <int N>
inline int MatchingFlatSize(const Dims<N>& dims, const Dims<N>& check_dims_0,
                            const Dims<N>& check_dims_1,
                            const Dims<N>& check_dims_2) {
  for (int i = 0; i < N; ++i) {
    TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return MatchingFlatSize(dims, check_dims_1, check_dims_2);
}

template <int N>
inline int MatchingFlatSize(const Dims<N>& dims, const Dims<N>& check_dims_0,
                            const Dims<N>& check_dims_1,
                            const Dims<N>& check_dims_2,
                            const Dims<N>& check_dims_3) {
  for (int i = 0; i < N; ++i) {
    TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
  }
  return MatchingFlatSize(dims, check_dims_1, check_dims_2, check_dims_3);
}

// Data is required to be contiguous, and so many operators can use either the
// full array flat size or the flat size with one dimension skipped (commonly
// the depth).
template <int N>
inline int FlatSizeSkipDim(const Dims<N>& dims, int skip_dim) {
  TFLITE_DCHECK(skip_dim >= 0 && skip_dim < N);
  int flat_size = 1;
  for (int i = 0; i < N; ++i) {
    flat_size *= (i == skip_dim) ? 1 : dims.sizes[i];
  }
  return flat_size;
}

// A combination of MatchingFlatSize() and FlatSizeSkipDim().
template <int N>
inline int MatchingFlatSizeSkipDim(const Dims<N>& dims, int skip_dim,
                                   const Dims<N>& check_dims_0) {
  for (int i = 0; i < N; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
    }
  }
  return FlatSizeSkipDim(dims, skip_dim);
}

template <int N>
inline int MatchingFlatSizeSkipDim(const Dims<N>& dims, int skip_dim,
                                   const Dims<N>& check_dims_0,
                                   const Dims<N>& check_dims_1) {
  for (int i = 0; i < N; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
    }
  }
  return MatchingFlatSizeSkipDim(dims, skip_dim, check_dims_1);
}

template <int N>
inline int MatchingFlatSizeSkipDim(const Dims<N>& dims, int skip_dim,
                                   const Dims<N>& check_dims_0,
                                   const Dims<N>& check_dims_1,
                                   const Dims<N>& check_dims_2) {
  for (int i = 0; i < N; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
    }
  }
  return MatchingFlatSizeSkipDim(dims, skip_dim, check_dims_1, check_dims_2);
}

template <int N>
inline int MatchingFlatSizeSkipDim(const Dims<N>& dims, int skip_dim,
                                   const Dims<N>& check_dims_0,
                                   const Dims<N>& check_dims_1,
                                   const Dims<N>& check_dims_2,
                                   const Dims<N>& check_dims_3) {
  for (int i = 0; i < N; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
    }
  }
  return MatchingFlatSizeSkipDim(dims, skip_dim, check_dims_1, check_dims_2,
                                 check_dims_3);
}

// Data is required to be contiguous, and so many operators can use either the
// full array flat size or the flat size with one dimension skipped (commonly
// the depth).
inline int FlatSizeSkipDim(const RuntimeShape& shape, int skip_dim) {
  const int dims_count = shape.DimensionsCount();
  TFLITE_DCHECK(skip_dim >= 0 && skip_dim < dims_count);
  const auto* dims_data = shape.DimsData();
  int flat_size = 1;
  for (int i = 0; i < dims_count; ++i) {
    flat_size *= (i == skip_dim) ? 1 : dims_data[i];
  }
  return flat_size;
}

// A combination of MatchingFlatSize() and FlatSizeSkipDim().
inline int MatchingFlatSizeSkipDim(const RuntimeShape& shape, int skip_dim,
                                   const RuntimeShape& check_shape_0) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
    }
  }
  return FlatSizeSkipDim(shape, skip_dim);
}

inline int MatchingFlatSizeSkipDim(const RuntimeShape& shape, int skip_dim,
                                   const RuntimeShape& check_shape_0,
                                   const RuntimeShape& check_shape_1) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
    }
  }
  return MatchingFlatSizeSkipDim(shape, skip_dim, check_shape_1);
}

inline int MatchingFlatSizeSkipDim(const RuntimeShape& shape, int skip_dim,
                                   const RuntimeShape& check_shape_0,
                                   const RuntimeShape& check_shape_1,
                                   const RuntimeShape& check_shape_2) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
    }
  }
  return MatchingFlatSizeSkipDim(shape, skip_dim, check_shape_1, check_shape_2);
}

inline int MatchingFlatSizeSkipDim(const RuntimeShape& shape, int skip_dim,
                                   const RuntimeShape& check_shape_0,
                                   const RuntimeShape& check_shape_1,
                                   const RuntimeShape& check_shape_2,
                                   const RuntimeShape& check_shape_3) {
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
    }
  }
  return MatchingFlatSizeSkipDim(shape, skip_dim, check_shape_1, check_shape_2,
                                 check_shape_3);
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

template <int N>
void ComputeStrides(Dims<N>* dims) {
  dims->strides[0] = 1;
  for (int d = 1; d < N; d++) {
    dims->strides[d] = dims->strides[d - 1] * dims->sizes[d - 1];
  }
}

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TYPES_H_
