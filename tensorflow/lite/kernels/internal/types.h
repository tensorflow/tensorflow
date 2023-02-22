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
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace tflite {

enum class FusedActivationFunctionType : uint8_t {
  kNone,
  kRelu6,
  kRelu1,
  kRelu
};
enum class PaddingType : uint8_t { kNone, kSame, kValid };

struct PaddingValues {
  int16_t width;
  int16_t height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16_t width_offset;
  // Same as width_offset except it's over the height dimension.
  int16_t height_offset;
};

struct Padding3DValues {
  int16_t width;
  int16_t height;
  int16_t depth;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16_t width_offset;
  // Same as width_offset except it's over the height dimension.
  int16_t height_offset;
  // Same as width_offset except it's over the depth dimension.
  int16_t depth_offset;
};

// This enumeration allows for non-default formats for the weights array
// of a fully-connected operator, allowing the use of special optimized
// runtime paths.
enum class FullyConnectedWeightsFormat : uint8_t {
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
  // weights format has each weights value encoded as a signed int8_t value,
  // even if the data type of the weights buffer is uint8_t.  This is intended
  // to save runtime kernels the effort to have to XOR the top bit of these
  // bytes before using them in signed arithmetic, see this file for more
  // explanations on the 'signed int8_t trick' in matrix multiplication kernels:
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
  int32_t zero_point = 0;
  double scale = 0.0;
};

inline bool operator==(const QuantizationParams& qp1,
                       const QuantizationParams& qp2) {
  return qp1.zero_point == qp2.zero_point && qp1.scale == qp2.scale;
}

// Quantization parameters for each channel, determining the mapping of
// quantized values to real values. See QuantizationParams for a single set of
// parameters per tensor. This has one parameters set per each channel.
//
// The correspondence is as follows:
//
//   real_value = scale[channel] * (quantized_value - zero_point[channel]);
//
struct PerChannelQuantizationParams {
  // The following members typically point to the corresponding members of a
  // TfLiteAffineQuantization struct.
  const float* scale;
  const int32_t* zero_point;
  int32_t quantized_dimension;
};

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

// Since tensors with '0' in their shape are valid in TF, these offset functions
// allow that as long as the corresponding index is also 0. It is upto the
// calling ops to ensure that they perform verification checks on tensor shapes
// if they don't support a particular behavior.

inline int Offset(const Dims<4>& dims, int i0, int i1, int i2, int i3) {
  TFLITE_DCHECK((i0 == 0 && dims.sizes[0] == 0) ||
                (i0 >= 0 && i0 < dims.sizes[0]));
  TFLITE_DCHECK((i1 == 0 && dims.sizes[1] == 0) ||
                (i1 >= 0 && i1 < dims.sizes[1]));
  TFLITE_DCHECK((i2 == 0 && dims.sizes[2] == 0) ||
                (i2 >= 0 && i2 < dims.sizes[2]));
  TFLITE_DCHECK((i3 == 0 && dims.sizes[3] == 0) ||
                (i3 >= 0 && i3 < dims.sizes[3]));
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
  return std::min(shape1.Dims(index1), shape2.Dims(index2));
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

// Flat size calculation, checking if their extended shapes match.
inline int MatchingExtendedShapeFlatSize(const RuntimeShape& shape,
                                         const RuntimeShape& check_shape_0) {
  const int shape_dims = shape.DimensionsCount();
  const int check_shape_0_dims = check_shape_0.DimensionsCount();
  const int min_dims = std::min(shape_dims, check_shape_0_dims);

  for (int i = 0; i < min_dims; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(shape_dims - 1 - i),
                     check_shape_0.Dims(check_shape_0_dims - 1 - i));
  }
  for (int i = min_dims; i < shape_dims; ++i) {
    TFLITE_DCHECK_EQ(shape.Dims(shape_dims - 1 - i), 1);
  }
  for (int i = min_dims; i < check_shape_0_dims; ++i) {
    TFLITE_DCHECK_EQ(check_shape_0.Dims(check_shape_0_dims - 1 - i), 1);
  }
  return shape.FlatSize();
}

inline int MatchingExtendedShapeFlatSize(const RuntimeShape& shape,
                                         const RuntimeShape& check_shape_0,
                                         const RuntimeShape& check_shape_1) {
  const int flat_size = MatchingExtendedShapeFlatSize(shape, check_shape_0);
  TFLITE_DCHECK_EQ(MatchingExtendedShapeFlatSize(shape, check_shape_1),
                   flat_size);
  return flat_size;
}

inline int MatchingExtendedShapeFlatSize(const RuntimeShape& shape,
                                         const RuntimeShape& check_shape_0,
                                         const RuntimeShape& check_shape_1,
                                         const RuntimeShape& check_shape_2) {
  const int flat_size = MatchingExtendedShapeFlatSize(shape, check_shape_0);
  TFLITE_DCHECK_EQ(
      MatchingExtendedShapeFlatSize(shape, check_shape_1, check_shape_2),
      flat_size);
  return flat_size;
}

inline int MatchingExtendedShapeFlatSize(const RuntimeShape& shape,
                                         const RuntimeShape& check_shape_0,
                                         const RuntimeShape& check_shape_1,
                                         const RuntimeShape& check_shape_2,
                                         const RuntimeShape& check_shape_3) {
  const int flat_size = MatchingExtendedShapeFlatSize(shape, check_shape_0);
  TFLITE_DCHECK_EQ(MatchingExtendedShapeFlatSize(shape, check_shape_1,
                                                 check_shape_2, check_shape_3),
                   flat_size);
  return flat_size;
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

enum class BroadcastableOpCategory : uint8_t {
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
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
};

struct ReluParams : public ActivationParams {
  int32_t input_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
};

// Styles of resizing op usages. For example, kImageStyle can be used with a Pad
// op for pattern-specific optimization.
enum class ResizingCategory : uint8_t {
  kNone,
  kImageStyle,  // 4D, operating on inner dimensions, say {0, a, b, 0}.
  kGenericResize,
};

// For Add, Sub, Mul ops.
struct ArithmeticParams {
  // Shape dependent / common to data / op types.
  BroadcastableOpCategory broadcast_category;
  // uint8_t inference params.
  int32_t input1_offset;
  int32_t input2_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // Add / Sub, not Mul, uint8_t inference params.
  int left_shift;
  int32_t input1_multiplier;
  int input1_shift;
  int32_t input2_multiplier;
  int input2_shift;

  // TODO(b/158622529): Union the following activation params.
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  // int64_t activation params.
  int64_t int64_activation_min;
  int64_t int64_activation_max;

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
  int8_t axis;
  const int32_t* input_zeropoint;
  const float* input_scale;
  uint16_t inputs_count;
  int32_t output_zeropoint;
  float output_scale;
};

struct ComparisonParams {
  // uint8_t inference params.
  int left_shift;
  int32_t input1_offset;
  int32_t input1_multiplier;
  int input1_shift;
  int32_t input2_offset;
  int32_t input2_multiplier;
  int input2_shift;
  // Shape dependent / common to inference types.
  bool is_broadcast;
};

struct ConvParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct Conv3DParams {
  Padding3DValues padding_values;
  int stride_width;
  int stride_height;
  int stride_depth;
  int dilation_width;
  int dilation_height;
  int dilation_depth;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

typedef Conv3DParams Conv3DTransposeParams;

struct DepthToSpaceParams {
  int32_t block_size;
};

struct DepthwiseParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  int16_t depth_multiplier;
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  const int32_t* output_multiplier_per_channel;
  const int32_t* output_shift_per_channel;
};

struct DequantizationParams {
  double scale;
  int32_t zero_point;
};

struct PerChannelDequantizationParams {
  const float* scale;
  const int32_t* zero_point;
  int32_t quantized_dimension;
};

struct FakeQuantParams {
  MinMax minmax;
  int32_t num_bits;
};

struct FullyConnectedParams {
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  // Mark the operands as cacheable if they are unchanging, e.g. weights.
  bool lhs_cacheable;
  bool rhs_cacheable;
  FullyConnectedWeightsFormat weights_format;
};

struct GatherParams {
  int16_t axis;
  int16_t batch_dims;
};

struct L2NormalizationParams {
  // uint8_t inference params.
  int32_t input_zero_point;
};

struct LocalResponseNormalizationParams {
  int32_t range;
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
  // uint8_t inference params.
  int32_t input_zero_point;
  int32_t input_range_radius;
  int32_t input_multiplier;
  int input_left_shift;
};

struct LstmCellParams {
  int32_t weights_zero_point;
  int32_t accum_multiplier;
  int accum_shift;
  int state_integer_bits;
};

struct MeanParams {
  int8_t axis_count;
  int16_t axis[4];
};

struct PackParams {
  int8_t axis;
  const int32_t* input_zeropoint;
  const float* input_scale;
  uint16_t inputs_count;
  int32_t output_zeropoint;
  float output_scale;
};

struct PadParams {
  int8_t left_padding_count;
  int32_t left_padding[5];
  int8_t right_padding_count;
  int32_t right_padding[5];
  ResizingCategory resizing_category;
};

struct PreluParams {
  int32_t input_offset;
  int32_t alpha_offset;
  int32_t output_offset;
  int32_t output_multiplier_1;
  int output_shift_1;
  int32_t output_multiplier_2;
  int output_shift_2;
};

struct PoolParams {
  FusedActivationFunctionType activation;
  PaddingType padding_type;
  PaddingValues padding_values;
  int stride_height;
  int stride_width;
  int filter_height;
  int filter_width;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct ReshapeParams {
  int8_t shape_count;
  int32_t shape[4];
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
  bool half_pixel_centers;
};

struct SliceParams {
  int8_t begin_count;
  int32_t begin[5];
  int8_t size_count;
  int32_t size[5];
};

struct SoftmaxParams {
  // beta is not really used (not a Tensorflow parameter) and not implemented
  // for LogSoftmax.
  double beta;
  // uint8_t inference params.  Used even when beta defaults to 1.0.
  int32_t input_multiplier;
  int32_t input_left_shift;
  // Reverse scaling is only used by LogSoftmax.
  int32_t reverse_scaling_divisor;
  int32_t reverse_scaling_right_shift;
  int diff_min;
  int32_t zero_point;
  float scale;
  float* table;
  // int16 LUT for exp(x), where x uniform distributed between [-10.0 , 0.0]
  int16_t* exp_lut;
  // int16 LUT for 1 / (1 + x), where x uniform distributed between [0.0 , 1.0]
  int16_t* one_over_one_plus_x_lut;
  uint8_t* uint8_table1;
  uint8_t* uint8_table2;
};

struct SpaceToBatchParams {
  // "Zero" padding for uint8_t means padding with the output offset.
  int32_t output_offset;
};

struct SpaceToDepthParams {
  int32_t block_size;
};

struct SplitParams {
  // Graphs that split into, say, 2000 nodes are encountered.  The indices in
  // OperatorEdges are of type uint16_t.
  uint16_t num_split;
  int16_t axis;
};

struct SqueezeParams {
  int8_t squeeze_dims_count;
  int32_t squeeze_dims[4];
};

struct StridedSliceParams {
  int8_t start_indices_count;
  int32_t start_indices[5];
  int8_t stop_indices_count;
  int32_t stop_indices[5];
  int8_t strides_count;
  int32_t strides[5];

  uint16_t begin_mask;
  uint16_t ellipsis_mask;
  uint16_t end_mask;
  uint16_t new_axis_mask;
  uint16_t shrink_axis_mask;
};

struct TanhParams {
  int32_t input_zero_point;
  int32_t input_range_radius;
  int32_t input_multiplier;
  int input_left_shift;
};

constexpr int kTransposeMaxDimensions = 6;

struct TransposeParams {
  int8_t perm_count;
  int32_t perm[kTransposeMaxDimensions];
};

struct UnpackParams {
  uint16_t num_split;
  int16_t axis;
};

struct LeakyReluParams {
  float alpha;
  int32_t input_offset;
  int32_t output_offset;
  int32_t output_multiplier_alpha;
  int32_t output_shift_alpha;
  int32_t output_multiplier_identity;
  int32_t output_shift_identity;
};

template <typename P>
inline void SetActivationParams(float min, float max, P* params) {
  params->float_activation_min = min;
  params->float_activation_max = max;
}

template <typename P>
inline void SetActivationParams(int32_t min, int32_t max, P* params) {
  params->quantized_activation_min = min;
  params->quantized_activation_max = max;
}

template <typename P>
inline void SetActivationParams(int64_t min, int64_t max, P* params) {
  params->int64_activation_min = min;
  params->int64_activation_max = max;
}

template <typename P>
inline void GetActivationParams(const P& params, int32_t* min, int32_t* max) {
  *min = params.quantized_activation_min;
  *max = params.quantized_activation_max;
}

template <typename P>
inline void GetActivationParams(const P& params, float* min, float* max) {
  *min = params.float_activation_min;
  *max = params.float_activation_max;
}

template <typename P>
inline void GetActivationParams(const P& params, int64_t* min, int64_t* max) {
  *min = params.int64_activation_min;
  *max = params.int64_activation_max;
}

// Type trait to check of given type has size smaller than 4 bytes.
template <typename T>
struct is_small_integer
    : public std::integral_constant<bool,
                                    std::is_same<T, int8_t>::value ||
                                        std::is_same<T, uint8_t>::value ||
                                        std::is_same<T, int16_t>::value ||
                                        std::is_same<T, uint16_t>::value> {};

// Type trait to check of given type is int32 or int64.
template <typename T>
struct is_int32_or_int64
    : public std::integral_constant<bool, std::is_same<T, int32_t>::value ||
                                              std::is_same<T, int64_t>::value> {
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TYPES_H_
