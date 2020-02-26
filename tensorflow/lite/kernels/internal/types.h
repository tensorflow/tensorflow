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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TYPES_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TYPES_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {

enum class FusedActivationFunctionType : uint8 { kNone, kRelu6, kRelu1, kRelu };
enum class PaddingType : uint8 { kNone, kSame, kValid };

struct PaddingValues {
  int16 width;
  int16 height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16 width_offset;
  // Same as width_offset except it's over the height dimension.
  int16 height_offset;
};

// This enumeration allows for non-default formats for the weights array
// of a fully-connected operator, allowing the use of special optimized
// runtime paths.
enum class FullyConnectedWeightsFormat : uint8 {
  // Default format (flat 2D layout, the inner contiguous dimension
  // is input_depth, the outer non-contiguous dimension is output_depth)
  kDefault,
  // Summary: optimized layout for fast CPU runtime implementation,
  // aimed specifically at ARM CPUs at the moment, and specialized for
  // 8-bit quantized layers.
  //
  // The use case we're concerned with here is: 8-bit quantization,
  // large weights matrix that doesn't fit in cache (e.g. 4096x2048 in
  // a key application that drove this), very small batch size (e.g. 1 -- 4).
  //
  // Even with 8-bit quantization of weights, the performance of memory
  // accesses to the weights can become the dominant issue when
  // the batch size is small, so each weight value is used in only a few
  // arithmetic ops, i.e. the fully-connected node has a low arithmetic
  // intensity. The specific issues that arise are of three kinds:
  // (1) One may, ideally, max out DRAM bandwidth, i.e. be truly memory
  //     bound. That's the "good" issue to run into.
  // (2) One may run into sub-optimal pre-fetching: the data hasn't been
  //     prefetched into the cache by the time we need it.
  // (3) One may run into cache aliasing: multiple values that are
  //     pre-fetched, alias each other in the L1 cache (which typically
  //     has only 4-way set associativity in ARM CPUs) and thus evict
  //     each other before we get to using them.
  //
  // The point of this shuffling is to avoid issues (2) and (3) so that
  // we get as fast as possible given only the hard constraint (1).
  // This is achieved by turning the difficulty into a solution: the
  // difficulty, that each value loaded from memory is used only in
  // one kernel iteration, making this operation memory-intensive, hints at
  // the solution, of shuffling the weights so that they are stored in the
  // exact order as the kernel needs to load them, so that the memory
  // accesses made by the kernel are trivial. This solves (2) because the
  // trivial memory access pattern allows the CPU's automatic prefetching
  // to perform very well (no need even for preload instructions), and this
  // solves (3) because the values being loaded concurrently are now
  // contiguous in the address space, thus don't alias each other in the cache.
  //
  // On ARM, we typically want our kernel to process a 4x16 block of weights
  // at a time, because:
  //   - 16 is the number of bytes in a NEON register.
  //   - 4 is how many rows we need to handle concurrently in the kernel in
  //     order to have sufficient mutual independence of instructions to
  //     maximize arithmetic throughput.
  //
  // Finally, the 'Int8' part in the name refers to the fact that this
  // weights format has each weights value encoded as a signed int8 value,
  // even if the data type of the weights buffer is uint8.  This is intended
  // to save runtime kernels the effort to have to XOR the top bit of these
  // bytes before using them in signed arithmetic, see this file for more
  // explanations on the 'signed int8 trick' in matrix multiplication kernels:
  //
  //   tensorflow/lite/toco/graph_transformations/ensure_uint8_weights_safe_for_fast_int8_kernels.cc
  //
  kShuffled4x16Int8,
};

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

inline bool operator==(const QuantizationParams& qp1,
                       const QuantizationParams& qp2) {
  return qp1.zero_point == qp2.zero_point && qp1.scale == qp2.scale;
}

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

  RuntimeShape& operator=(RuntimeShape const&) = delete;

  RuntimeShape() : size_(0) {}

  explicit RuntimeShape(int dimensions_count) : size_(dimensions_count) {
    if (dimensions_count > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      dims_pointer_ = new int32[dimensions_count];
#endif  // TF_LITE_STATIC_MEMORY
    }
  }

  RuntimeShape(int shape_size, int32 value) : size_(0) {
    Resize(shape_size);
    for (int i = 0; i < shape_size; ++i) {
      SetDim(i, value);
    }
  }

  RuntimeShape(int dimensions_count, const int32* dims_data) : size_(0) {
    ReplaceWith(dimensions_count, dims_data);
  }

  RuntimeShape(const std::initializer_list<int> init_list) : size_(0) {
    BuildFrom(init_list);
  }

  // Avoid using this constructor.  We should be able to delete it when C++17
  // rolls out.
  RuntimeShape(RuntimeShape const& other) : size_(other.DimensionsCount()) {
    if (size_ > kMaxSmallSize) {
      dims_pointer_ = new int32[size_];
    }
    std::memcpy(DimsData(), other.DimsData(), sizeof(int32) * size_);
  }

  bool operator==(const RuntimeShape& comp) const {
    return this->size_ == comp.size_ &&
           std::memcmp(DimsData(), comp.DimsData(), size_ * sizeof(int32)) == 0;
  }

  ~RuntimeShape() {
    if (size_ > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      delete[] dims_pointer_;
#endif  // TF_LITE_STATIC_MEMORY
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
  // The caller must ensure that the shape is no bigger than 4-D.
  inline const int32* DimsDataUpTo4D() const { return dims_; }

  inline void Resize(int dimensions_count) {
    if (size_ > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      delete[] dims_pointer_;
#endif  // TF_LITE_STATIC_MEMORY
    }
    size_ = dimensions_count;
    if (dimensions_count > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      dims_pointer_ = new int32[dimensions_count];
#endif  // TF_LITE_STATIC_MEMORY
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

  // This will probably be factored out. Old code made substantial use of 4-D
  // shapes, and so this function is used to extend smaller shapes. Note that
  // (a) as Dims<4>-dependent code is eliminated, the reliance on this should be
  // reduced, and (b) some kernels are stricly 4-D, but then the shapes of their
  // inputs should already be 4-D, so this function should not be needed.
  inline static RuntimeShape ExtendedShape(int new_shape_size,
                                           const RuntimeShape& shape) {
    return RuntimeShape(new_shape_size, shape, 1);
  }

  inline void BuildFrom(const std::initializer_list<int> init_list) {
    BuildFrom<const std::initializer_list<int>>(init_list);
  }

  // Returns the total count of elements, that is the size when flattened into a
  // vector.
  inline int FlatSize() const {
    int buffer_size = 1;
    const int* dims_data = reinterpret_cast<const int*>(DimsData());
    for (int i = 0; i < size_; i++) {
      buffer_size *= dims_data[i];
    }
    return buffer_size;
  }

  bool operator!=(const RuntimeShape& comp) const { return !((*this) == comp); }

 private:
  // For use only by ExtendedShape(), written to guarantee (return-value) copy
  // elision in C++17.
  // This creates a shape padded to the desired size with the specified value.
  RuntimeShape(int new_shape_size, const RuntimeShape& shape, int pad_value)
      : size_(0) {
    // If the following check fails, it is likely because a 4D-only kernel is
    // being used with an array of larger dimension count.
    TFLITE_CHECK_GE(new_shape_size, shape.DimensionsCount());
    Resize(new_shape_size);
    const int size_increase = new_shape_size - shape.DimensionsCount();
    for (int i = 0; i < size_increase; ++i) {
      SetDim(i, pad_value);
    }
    std::memcpy(DimsData() + size_increase, shape.DimsData(),
                sizeof(int32) * shape.DimensionsCount());
  }

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

// TODO(b/80418076): Move to legacy ops file, update invocations.
inline RuntimeShape DimsToShape(const tflite::Dims<4>& dims) {
  return RuntimeShape(
      {dims.sizes[3], dims.sizes[2], dims.sizes[1], dims.sizes[0]});
}

// Gets next index to iterate through a multidimensional array.
inline bool NextIndex(const int num_dims, const int* dims, int* current) {
  if (num_dims == 0) {
    return false;
  }
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
  if (num_dims == 0) {
    return 0;
  }
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
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), 4);
  const int* dims_data = reinterpret_cast<const int*>(shape.DimsDataUpTo4D());
  TFLITE_DCHECK(i0 >= 0 && i0 < dims_data[0]);
  TFLITE_DCHECK(i1 >= 0 && i1 < dims_data[1]);
  TFLITE_DCHECK(i2 >= 0 && i2 < dims_data[2]);
  TFLITE_DCHECK(i3 >= 0 && i3 < dims_data[3]);
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

inline int Offset(const RuntimeShape& shape, int* index) {
  return Offset(shape, index[0], index[1], index[2], index[3]);
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

TFLITE_DEPRECATED("Prefer FlatSize.")
inline int RequiredBufferSizeForDims(const Dims<4>& dims) {
  return FlatSize(dims);
}

inline int MatchingElementsSize(const RuntimeShape& shape,
                                const RuntimeShape& check_shape_0) {
  const int size_1 = shape.FlatSize();
  const int size_2 = check_shape_0.FlatSize();
  TFLITE_CHECK_EQ(size_1, size_2);
  return size_1;
}

inline int MatchingElementsSize(const RuntimeShape& shape,
                                const RuntimeShape& check_shape_0,
                                const RuntimeShape& check_shape_1) {
  const int size_1 = shape.FlatSize();
  const int size_2 = check_shape_0.FlatSize();
  const int size_3 = check_shape_1.FlatSize();
  TFLITE_CHECK_EQ(size_1, size_2);
  TFLITE_CHECK_EQ(size_2, size_3);
  return size_1;
}

// Flat size calculation, checking that dimensions match with one or more other
// arrays.
inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0) {
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), check_shape_0.DimensionsCount());
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(i), check_shape_0.Dims(i));
  }
  return shape.FlatSize();
}

inline int MatchingFlatSize(const RuntimeShape& shape,
                            const RuntimeShape& check_shape_0,
                            const RuntimeShape& check_shape_1) {
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), check_shape_0.DimensionsCount());
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
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), check_shape_0.DimensionsCount());
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
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), check_shape_0.DimensionsCount());
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

enum class BroadcastableOpCategory : uint8 {
  kNone,
  kNonBroadcast,               // Matching input shapes.
  kFirstInputBroadcastsFast,   // Fivefold nested loops.
  kSecondInputBroadcastsFast,  // Fivefold nested loops.
  kGenericBroadcast,           // Fall-back.
};

struct MinMax {
  float min;
  float max;
};
static_assert(sizeof(MinMax) == 8, "");

struct ActivationParams {
  FusedActivationFunctionType activation_type;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
};

struct ReluParams : public ActivationParams {
  int32 input_offset;
  int32 output_offset;
  int32 output_multiplier;
  int32 output_shift;
};

// Styles of resizing op usages. For example, kImageStyle can be used with a Pad
// op for pattern-specific optimization.
enum class ResizingCategory : uint8 {
  kNone,
  kImageStyle,  // 4D, operating on inner dimensions, say {0, a, b, 0}.
  kGenericResize,
};

// For Add, Sub, Mul ops.
struct ArithmeticParams {
  // Shape dependent / common to data / op types.
  BroadcastableOpCategory broadcast_category;
  // uint8 inference params.
  int32 input1_offset;
  int32 input2_offset;
  int32 output_offset;
  int32 output_multiplier;
  int output_shift;
  // Add / Sub, not Mul, uint8 inference params.
  int left_shift;
  int32 input1_multiplier;
  int input1_shift;
  int32 input2_multiplier;
  int input2_shift;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;

  // Processed output dimensions.
  // Let input "a" be the one that broadcasts in the faster-changing dimension.
  // Then, after coalescing, for shapes {a0, a1, a2, a3, a4} and
  // {b0, b1, b2, b3, b4},
  // broadcast_shape[4] = b0 = a0.
  // broadcast_shape[3] = b1; a1 = 1.
  // broadcast_shape[2] = b2 = a2.
  // broadcast_shape[1] = a3; b3 = 1.
  // broadcast_shape[0] = b4 = a4.
  int broadcast_shape[5];
};

struct ConcatenationParams {
  int8 axis;
  const int32* input_zeropoint;
  const float* input_scale;
  uint16 inputs_count;
  int32 output_zeropoint;
  float output_scale;
};

struct ComparisonParams {
  // uint8 inference params.
  int left_shift;
  int32 input1_offset;
  int32 input1_multiplier;
  int input1_shift;
  int32 input2_offset;
  int32 input2_multiplier;
  int input2_shift;
  // Shape dependent / common to inference types.
  bool is_broadcast;
};

struct ConvParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16 stride_width;
  int16 stride_height;
  int16 dilation_width_factor;
  int16 dilation_height_factor;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32 input_offset;
  int32 weights_offset;
  int32 output_offset;
  int32 output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct DepthToSpaceParams {
  int32 block_size;
};

struct DepthwiseParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  int16 stride_width;
  int16 stride_height;
  int16 dilation_width_factor;
  int16 dilation_height_factor;
  int16 depth_multiplier;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32 input_offset;
  int32 weights_offset;
  int32 output_offset;
  int32 output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  const int32* output_multiplier_per_channel;
  const int32* output_shift_per_channel;
};

struct DequantizationParams {
  double scale;
  int32 zero_point;
};

struct FakeQuantParams {
  MinMax minmax;
  int32 num_bits;
};

struct FullyConnectedParams {
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32 input_offset;
  int32 weights_offset;
  int32 output_offset;
  int32 output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  // Mark the operands as cacheable if they are unchanging, e.g. weights.
  bool lhs_cacheable;
  bool rhs_cacheable;
  FullyConnectedWeightsFormat weights_format;
};

struct GatherParams {
  int16 axis;
};

struct L2NormalizationParams {
  // uint8 inference params.
  int32 input_zero_point;
};

struct LocalResponseNormalizationParams {
  int32 range;
  double bias;
  double alpha;
  double beta;
};

struct HardSwishParams {
  // zero_point of the input activations.
  int16_t input_zero_point;
  // zero_point of the output activations.
  int16_t output_zero_point;
  // 16bit fixed-point component of the multiplier to apply to go from the
  // "high-res input scale", which is the input scale multiplied by 2^7, to the
  // "relu-ish scale", which 3.0/32768.
  // See the implementation of HardSwishPrepare.
  int16_t reluish_multiplier_fixedpoint_int16;
  // exponent/bit-shift component of the aforementioned multiplier.
  int reluish_multiplier_exponent;
  // 16bit fixed-point component of the multiplier to apply to go from the
  // "high-res input scale", which is the input scale multiplied by 2^7, to the
  // output scale.
  // See the implementation of HardSwishPrepare.
  int16_t output_multiplier_fixedpoint_int16;
  // exponent/bit-shift component of the aforementioned multiplier.
  int output_multiplier_exponent;
};

struct LogisticParams {
  // uint8 inference params.
  int32 input_zero_point;
  int32 input_range_radius;
  int32 input_multiplier;
  int input_left_shift;
};

struct LstmCellParams {
  int32 weights_zero_point;
  int32 accum_multiplier;
  int accum_shift;
  int state_integer_bits;
};

struct MeanParams {
  int8 axis_count;
  int16 axis[4];
};

struct PackParams {
  int8 axis;
  const int32* input_zeropoint;
  const float* input_scale;
  uint16 inputs_count;
  int32 output_zeropoint;
  float output_scale;
};

struct PadParams {
  int8 left_padding_count;
  int32 left_padding[4];
  int8 right_padding_count;
  int32 right_padding[4];
  ResizingCategory resizing_category;
};

struct PreluParams {
  int32 input_offset;
  int32 alpha_offset;
  int32 output_offset;
  int32 output_multiplier;
  int output_shift;
};

struct PoolParams {
  FusedActivationFunctionType activation;
  PaddingType padding_type;
  PaddingValues padding_values;
  int stride_height;
  int stride_width;
  int filter_height;
  int filter_width;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct ReshapeParams {
  int8 shape_count;
  int32 shape[4];
};

struct ResizeBilinearParams {
  bool align_corners;
  // half_pixel_centers assumes pixels are of half the actual dimensions, and
  // yields more accurate resizes. Corresponds to the same argument for the
  // original TensorFlow op in TF2.0.
  bool half_pixel_centers;
};

struct ResizeNearestNeighborParams {
  bool align_corners;
};

struct SliceParams {
  int8 begin_count;
  int32 begin[4];
  int8 size_count;
  int32 size[4];
};

struct SoftmaxParams {
  // beta is not really used (not a Tensorflow parameter) and not implemented
  // for LogSoftmax.
  double beta;
  // uint8 inference params.  Used even when beta defaults to 1.0.
  int32 input_multiplier;
  int32 input_left_shift;
  // Reverse scaling is only used by LogSoftmax.
  int32 reverse_scaling_divisor;
  int32 reverse_scaling_right_shift;
  int diff_min;
  int32_t zero_point;
  float scale;
  float* table;
};

struct SpaceToBatchParams {
  // "Zero" padding for uint8 means padding with the output offset.
  int32 output_offset;
};

struct SpaceToDepthParams {
  int32 block_size;
};

struct SplitParams {
  // Graphs that split into, say, 2000 nodes are encountered.  The indices in
  // OperatorEdges are of type uint16.
  uint16 num_split;
  int16 axis;
};

struct SqueezeParams {
  int8 squeeze_dims_count;
  int32 squeeze_dims[4];
};

struct StridedSliceParams {
  int8 start_indices_count;
  int32 start_indices[4];
  int8 stop_indices_count;
  int32 stop_indices[4];
  int8 strides_count;
  int32 strides[4];

  int16 begin_mask;
  int16 ellipsis_mask;
  int16 end_mask;
  int16 new_axis_mask;
  int16 shrink_axis_mask;
};

struct TanhParams {
  int32 input_zero_point;
  int32 input_range_radius;
  int32 input_multiplier;
  int input_left_shift;
};

struct TransposeParams {
  int8 perm_count;
  int32 perm[4];
};

struct UnpackParams {
  uint16 num_split;
  int16 axis;
};

struct LeakyReluParams {
  float alpha;
  int32 input_offset;
  int32 alpha_offset;
  int32 output_offset;
  int32 output_multiplier;
  int output_shift;
};

template <typename P>
inline void SetActivationParams(float min, float max, P* params) {
  params->float_activation_min = min;
  params->float_activation_max = max;
}

template <typename P>
inline void SetActivationParams(int32 min, int32 max, P* params) {
  params->quantized_activation_min = min;
  params->quantized_activation_max = max;
}

template <typename P>
inline void GetActivationParams(const P& params, int32* min, int32* max) {
  *min = params.quantized_activation_min;
  *max = params.quantized_activation_max;
}

template <typename P>
inline void GetActivationParams(const P& params, float* min, float* max) {
  *min = params.float_activation_min;
  *max = params.float_activation_max;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TYPES_H_
