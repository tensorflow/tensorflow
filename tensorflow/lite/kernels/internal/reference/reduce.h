/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REDUCE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REDUCE_H_

#include <algorithm>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/max.h"
#include "tensorflow/lite/kernels/internal/min.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

// Check if the reduction at index is the first one along the dimensions given
// in axis.
inline bool IsFirstReduction(const int* index, const int num_axis,
                             const int* axis) {
  if (num_axis == 0) {
    return true;
  }

  TFLITE_DCHECK(index != nullptr);
  TFLITE_DCHECK(axis != nullptr);
  for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
    if (index[axis[axis_idx]] != 0) {
      return false;
    }
  }

  return true;
}

namespace tflite {

enum ReduceType {
  kSum,
  kProd,
  kMax,
  kMin,
  kAny,
  kAll,
};

template <typename T>
struct SumOp {
  inline T operator()(const T& a, const T& b) { return a + b; }
  static constexpr T kNeutralElement = T(0);
};

template <typename T, typename U>
struct CastSumOp {
  inline U operator()(const U& a, const T& b) { return a + static_cast<U>(b); }
  static constexpr U kNeutralElement = U(0);
};

template <typename T>
struct ProdOp {
  inline T operator()(const T& a, const T& b) { return a * b; }
  static constexpr T kNeutralElement = T(1);
};

template <typename T>
struct MaxOp {
  inline T operator()(const T& a, const T& b) { return (a > b) ? a : b; }
  static constexpr T kNeutralElement = std::numeric_limits<T>::lowest();
};

template <typename T>
struct MinOp {
  inline T operator()(const T& a, const T& b) { return (a < b) ? a : b; }
  static constexpr T kNeutralElement = std::numeric_limits<T>::max();
};

struct AndOp {
  inline bool operator()(bool a, bool b) { return a && b; }
  static constexpr bool kNeutralElement = true;
};

struct OrOp {
  inline bool operator()(bool a, bool b) { return a || b; }
  static constexpr bool kNeutralElement = false;
};

namespace reference_ops {

// When the number of axis is zero, the reduction is simply a copy.
template <typename T>
void ReduceIsCopy(const T* input_data, const int* input_dims,
                  const int input_num_dims, T* output_data) {
  int num_elems = 1;
  for (int i = 0; i < input_num_dims; ++i) {
    num_elems *= input_dims[i];
  }
  memcpy(output_data, input_data, num_elems * sizeof(T));
}

// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis.
template <typename In, typename Out, typename Op>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int* output_dims, const int input_num_dims,
                   const int output_num_dims, const int* axis,
                   const int num_axis, int* input_iter, Out* output_data) {
  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx) {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do {
    size_t input_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset = ReducedOutputOffset(input_num_dims, input_dims,
                                               input_iter, num_axis, axis);
    output_data[output_offset] =
        Op()(output_data[output_offset], input_data[input_offset]);
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

// Similar to above Reduce function but takes two reducer functions.
// The 'reducer_first' is called with the first value of the reduction,
// 'reducer_next' is then called for all the others.
template <typename In, typename Out>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int* output_dims, const int input_num_dims,
                   const int output_num_dims, const int* axis,
                   const int num_axis, int* input_iter,
                   const std::function<Out(In in)>& reducer_first,
                   const std::function<Out(Out current, In in)>& reducer_next,
                   Out* output_data) {
  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx) {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do {
    size_t input_offset =
        ReducedOutputOffset(input_num_dims, input_dims, input_iter, 0, nullptr);
    size_t output_offset = ReducedOutputOffset(input_num_dims, input_dims,
                                               input_iter, num_axis, axis);
    if (IsFirstReduction(input_iter, num_axis, axis)) {
      output_data[output_offset] = reducer_first(input_data[input_offset]);
    } else {
      output_data[output_offset] =
          reducer_next(output_data[output_offset], input_data[input_offset]);
    }
  } while (NextIndex(input_num_dims, input_dims, input_iter));
  return true;
}

// Bubble sort for sorting small inputs. std::sort may dynamically allocate
// memory so is not suitable for use in TFLM.
static void sort(int* input, int size) {
  for (int i = 0; i < size - 1; ++i) {
    for (int j = 0; j < size - i - 1; ++j) {
      if (input[j] > input[j + 1]) {
        std::swap(input[j], input[j + 1]);
      }
    }
  }
}

// This method parses the input 'axis' to remove duplicates, handle negative
// values and remove redundant dimensions. It returns a valid 'out_axis' and
// 'shape_out' contains the flattened input shape. 'out_num_dims' contains the
// reduced number of dimensions.
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis, const int* shape_in, int* shape_out,
                        int* out_num_dims) {
  int num_out_axis = 0;
  int dims_out = num_dims;
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    return true;
  }
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
    sort(out_axis, num_out_axis);

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

// This method expects that output_data has been initialized.
template <typename In, typename Out>
inline bool ReduceSumImpl(const In* input_data, const int* input_dims,
                          const int* output_dims, const int input_num_dims,
                          const int output_num_dims, const int* axis,
                          const int num_axis, int* input_iter,
                          Out* output_data) {
  return Reduce<In, Out, CastSumOp<In, Out>>(
      input_data, input_dims, output_dims, input_num_dims, output_num_dims,
      axis, num_axis, input_iter, output_data);
}

template <typename T>
inline bool InitTensorDataForReduce(const int* dims, const int num_dims,
                                    const T init_value, T* data) {
  size_t num_elements = 1;
  for (int idx = 0; idx < num_dims; ++idx) {
    size_t current = static_cast<size_t>(dims[idx]);
    // Overflow prevention.
    if (current > 0 &&
        num_elements > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_elements *= current;
  }
  for (size_t idx = 0; idx < num_elements; ++idx) {
    data[idx] = init_value;
  }
  return true;
}

// Computes the generic value (i.e., sum/max/min/prod) of elements across
// dimensions given in axis. It needs to pass in reducer.
template <typename T>
inline bool ReduceGeneric(const T* input_data, const int* input_dims,
                          const int input_num_dims, T* output_data,
                          const int* output_dims, const int output_num_dims,
                          const int* axis, const int64_t num_axis_dimensions,
                          bool keep_dims, int* temp_index, int* resolved_axis,
                          int* normalized_dims, ReduceType reduce_type) {
  T init_value;
  switch (reduce_type) {
    case kProd:
      init_value = ProdOp<T>::kNeutralElement;
      break;
    case kSum:
      init_value = SumOp<T>::kNeutralElement;
      break;
    case kMin:
      init_value = MinOp<T>::kNeutralElement;
      break;
    case kMax:
      init_value = MaxOp<T>::kNeutralElement;
      break;
    case kAny:
      init_value = OrOp::kNeutralElement;
      break;
    case kAll:
      init_value = AndOp::kNeutralElement;
      break;
    default:
      return false;
  }
  // Reset output data.
  if (!InitTensorDataForReduce(output_dims, output_num_dims, init_value,
                               output_data)) {
    return false;
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  int normalized_num_dims = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis, input_dims, normalized_dims,
                   &normalized_num_dims)) {
    return false;
  }

  // Return early when input shape has zero dim. This is done after initializing
  // data for output tensor because there are cases that the input tensor is
  // empty but output tensor is not. In that case, output tensor should be
  // filled with Op::kNeutralElement.
  for (int i = 0; i < input_num_dims; ++i) {
    if (input_dims[i] == 0) return true;
  }

  switch (reduce_type) {
    case kProd:
      return Reduce<T, T, ProdOp<T>>(input_data, normalized_dims, output_dims,
                                     normalized_num_dims, output_num_dims,
                                     resolved_axis, num_resolved_axis,
                                     temp_index, output_data);
    case kSum:
      return Reduce<T, T, SumOp<T>>(input_data, normalized_dims, output_dims,
                                    normalized_num_dims, output_num_dims,
                                    resolved_axis, num_resolved_axis,
                                    temp_index, output_data);
    case kMin:
      return Reduce<T, T, MinOp<T>>(input_data, normalized_dims, output_dims,
                                    normalized_num_dims, output_num_dims,
                                    resolved_axis, num_resolved_axis,
                                    temp_index, output_data);
    case kMax:
      return Reduce<T, T, MaxOp<T>>(input_data, normalized_dims, output_dims,
                                    normalized_num_dims, output_num_dims,
                                    resolved_axis, num_resolved_axis,
                                    temp_index, output_data);
    case kAll:
      return Reduce<T, T, AndOp>(input_data, normalized_dims, output_dims,
                                 normalized_num_dims, output_num_dims,
                                 resolved_axis, num_resolved_axis, temp_index,
                                 output_data);
    case kAny:
      return Reduce<T, T, OrOp>(input_data, normalized_dims, output_dims,
                                normalized_num_dims, output_num_dims,
                                resolved_axis, num_resolved_axis, temp_index,
                                output_data);
    default:
      return false;
  }
}

// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis.
template <typename T, typename U>
inline bool Mean(const T* input_data, const int* input_dims,
                 const int input_num_dims, T* output_data,
                 const int* output_dims, const int output_num_dims,
                 const int* axis, const int num_axis_dimensions, bool keep_dims,
                 int* temp_index, int* resolved_axis, int* normalized_dims,
                 U* temp_sum) {
  if (num_axis_dimensions == 0) {
    ReduceIsCopy(input_data, input_dims, input_num_dims, output_data);
    return true;
  }
  ruy::profiler::ScopeLabel label("Mean");
  // Reset output data.
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx) {
    size_t current = static_cast<size_t>(output_dims[idx]);
    // Overflow prevention.
    if (num_outputs > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_outputs *= current;
  }
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    output_data[idx] = T();
    temp_sum[idx] = U();
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  int normalized_num_dims = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis, input_dims, normalized_dims,
                   &normalized_num_dims)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, normalized_dims, output_dims,
                           normalized_num_dims, output_num_dims, resolved_axis,
                           num_resolved_axis, temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    size_t current = static_cast<size_t>(normalized_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (std::numeric_limits<size_t>::max() / num_elements_in_axis)) {
      return false;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis > 0) {
    for (size_t idx = 0; idx < num_outputs; ++idx) {
      output_data[idx] =
          static_cast<T>(temp_sum[idx] / static_cast<U>(num_elements_in_axis));
    }
  }
  return true;
}

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const float* input_data,
                 const RuntimeShape& unextended_output_shape,
                 float* output_data) {
  ruy::profiler::ScopeLabel label("Mean4D");

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          value += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          value / (input_width * input_height);
    }
  }
}

// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis for quantized values.
template <typename T, typename U>
inline bool QuantizedMeanOrSum(
    const T* input_data, int32_t input_zero_point, const int* input_dims,
    const int input_num_dims, T* output_data, int32_t output_multiplier,
    int output_shift, int32_t output_zero_point, const int* output_dims,
    const int output_num_dims, const int* axis, const int num_axis_dimensions,
    bool keep_dims, int* temp_index, int* resolved_axis, int* normalized_dims,
    U* temp_sum, bool compute_sum) {
  if (num_axis_dimensions == 0) {
    ReduceIsCopy(input_data, input_dims, input_num_dims, output_data);
    return true;
  }
  const int32_t kMinValue = std::numeric_limits<T>::min();
  const int32_t kMaxValue = std::numeric_limits<T>::max();
  const bool uint8_case = std::is_same<T, uint8_t>::value;
  const bool int16_case = std::is_same<T, int16_t>::value;
  if (uint8_case) {
    ruy::profiler::ScopeLabel label(compute_sum ? "Sum/Uint8" : "Mean/Uint8");
  } else if (int16_case) {
    ruy::profiler::ScopeLabel label(compute_sum ? "Sum/Int16" : "Mean/Int16");
  } else {
    ruy::profiler::ScopeLabel label(compute_sum ? "Sum/Int8" : "Mean/Int8");
  }
  // Reset output data.
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx) {
    size_t current = static_cast<size_t>(output_dims[idx]);
    // Overflow prevention.
    if (num_outputs > std::numeric_limits<size_t>::max() / current) {
      return false;
    }
    num_outputs *= current;
  }
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    output_data[idx] = T();
    temp_sum[idx] = U();
  }

  // Return early when input shape has zero dim. This is done after initializing
  // data for output tensor because there are cases that the input tensor is
  // empty but output tensor is not. In that case, output tensor should be
  // filled with init_value.
  for (int i = 0; i < input_num_dims; ++i) {
    if (input_dims[i] == 0) return true;
  }

  // Resolve axis.
  int num_resolved_axis = 0;
  int normalized_num_dims = 0;
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis, input_dims, normalized_dims,
                   &normalized_num_dims)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, normalized_dims, output_dims,
                           normalized_num_dims, output_num_dims, resolved_axis,
                           num_resolved_axis, temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  int64_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    int64_t current = static_cast<int64_t>(normalized_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current >
        (std::numeric_limits<int64_t>::max() / num_elements_in_axis)) {
      return false;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis == 0) {
    return true;
  }

  // Readapt output rescaling when calculating the mean to integrate a
  // 1/num_elements_in_axis multiplier.
  if (!compute_sum) {
    TFLITE_DCHECK_GE(num_elements_in_axis, 0);
    int shift =
        63 - CountLeadingZeros(static_cast<uint64_t>(num_elements_in_axis));
    // To avoid any overflow risk 'shift' should be <= 32 and to satisfy
    // 'MultiplyByQuantizedMultiplier' pre-conditions 'output_shift - shift'
    // should be >= -31. Clamp the value at the price of some precision loss.
    shift = std::min(shift, 32);
    shift = std::min(shift, 31 + output_shift);
    output_multiplier = static_cast<int32_t>(
        (static_cast<int64_t>(output_multiplier) << shift) /
        num_elements_in_axis);
    output_shift = output_shift - shift;
  }

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    const U shifted_sum =
        static_cast<U>(temp_sum[idx] - input_zero_point * num_elements_in_axis);
    int32_t output = MultiplyByQuantizedMultiplier(
                         shifted_sum, output_multiplier, output_shift) +
                     output_zero_point;
    output = std::min(std::max(output, kMinValue), kMaxValue);
    output_data[idx] = static_cast<T>(output);
  }
  return true;
}

template <typename T>
inline bool QuantizedReduceProd(
    const T* input_data, int32_t input_zero_point,
    const RuntimeShape& input_shape, T* output_data, int32_t output_zero_point,
    const RuntimeShape& output_shape, const int* axis,
    const int64_t num_axis_dimensions, bool keep_dims, int* temp_index,
    int* resolved_axis, int* normalized_dims, int32_t* temp_prod,
    int32_t scaling_multiplier, int scaling_shift) {
  const int32_t kMinValue = std::numeric_limits<T>::min();
  const int32_t kMaxValue = std::numeric_limits<T>::max();

  // Resolve axis.
  int num_resolved_axis = 0;
  int normalized_num_dims = 0;
  // reinterpret_cast(input_shape.DimsData) is needed to build with the Hexagon
  // toolchain. See http://b/234097792 for more context.
  if (!ResolveAxis(input_shape.DimensionsCount(), axis, num_axis_dimensions,
                   resolved_axis, &num_resolved_axis,
                   reinterpret_cast<const int*>(input_shape.DimsData()),
                   normalized_dims, &normalized_num_dims)) {
    return false;
  }

  // Calculate the reduced product by rescaling each multiplication step to
  // avoid an overflow.
  auto reducer_first = [&](T in) -> int32_t { return in - input_zero_point; };

  auto reducer_next = [&](int32_t current, T in) -> int32_t {
    const int64_t result =
        static_cast<int64_t>(current) * (in - input_zero_point);
    return MultiplyByQuantizedMultiplier(result, scaling_multiplier,
                                         scaling_shift);
  };

  if (!Reduce<T, int32_t>(input_data, normalized_dims, output_shape.DimsData(),
                          normalized_num_dims, output_shape.DimensionsCount(),
                          resolved_axis, num_resolved_axis, temp_index,
                          reducer_first, reducer_next, temp_prod)) {
    return false;
  }

  for (int i = 0; i < output_shape.FlatSize(); i++) {
    int32_t result =
        MultiplyByQuantizedMultiplier(static_cast<int64_t>(temp_prod[i]),
                                      scaling_multiplier, scaling_shift) +
        output_zero_point;
    result = std::min(std::max(result, kMinValue), kMaxValue);
    output_data[i] = static_cast<T>(result);
  }

  return true;
}

}  // namespace reference_ops

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REDUCE_H_
