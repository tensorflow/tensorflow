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

namespace reference_ops {

// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis.
template <typename In, typename Out>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int* output_dims, const int input_num_dims,
                   const int output_num_dims, const int* axis,
                   const int num_axis, int* input_iter,
                   Out reducer(Out current, const In in), Out* output_data) {
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
        reducer(output_data[output_offset], input_data[input_offset]);
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

// This method parses the input 'axis' to remove duplicates and handle negative
// values, and returns a valid 'out_axis'
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis) {
  *out_num_axis = 0;  // Just in case.
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
    TFLITE_DCHECK(current >= 0 && current < num_dims);
    if (current < 0 || current >= num_dims) {
      return false;
    }
    bool is_dup = false;
    for (int j = 0; j < *out_num_axis; ++j) {
      if (out_axis[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

// This method expects that output_data has been initialized.
template <typename In, typename Out>
inline bool ReduceSumImpl(const In* input_data, const int* input_dims,
                          const int* output_dims, const int input_num_dims,
                          const int output_num_dims, const int* axis,
                          const int num_axis, int* input_iter,
                          Out* output_data) {
  auto reducer = [](const Out current, const In in) -> Out {
    const Out actual_in = static_cast<Out>(in);
    return current + actual_in;
  };
  return Reduce<In, Out>(input_data, input_dims, output_dims, input_num_dims,
                         output_num_dims, axis, num_axis, input_iter, reducer,
                         output_data);
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
// dimensions given in axis. It needs to pass in init_value and reducer.
template <typename T>
inline bool ReduceGeneric(const T* input_data, const int* input_dims,
                          const int input_num_dims, T* output_data,
                          const int* output_dims, const int output_num_dims,
                          const int* axis, const int64_t num_axis_dimensions,
                          bool keep_dims, int* temp_index, int* resolved_axis,
                          T init_value,
                          T reducer(const T current, const T in)) {
  // Reset output data.
  if (!InitTensorDataForReduce(output_dims, output_num_dims, init_value,
                               output_data)) {
    return false;
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
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  return Reduce<T, T>(input_data, input_dims, output_dims, input_num_dims,
                      output_num_dims, resolved_axis, num_resolved_axis,
                      temp_index, reducer, output_data);
}

// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis.
template <typename T, typename U>
inline bool Mean(const T* input_data, const int* input_dims,
                 const int input_num_dims, T* output_data,
                 const int* output_dims, const int output_num_dims,
                 const int* axis, const int num_axis_dimensions, bool keep_dims,
                 int* temp_index, int* resolved_axis, U* temp_sum) {
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
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, input_dims, output_dims, input_num_dims,
                           output_num_dims, resolved_axis, num_resolved_axis,
                           temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
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

template <typename T>
inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const T* input_data,
                 const RuntimeShape& unextended_output_shape, T* output_data) {
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

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const uint8_t* input_data, int32_t input_zero_point,
                 float input_scale, const RuntimeShape& unextended_output_shape,
                 uint8_t* output_data, int32_t output_zero_point,
                 float output_scale) {
  ruy::profiler::ScopeLabel label("Mean4D/Uint8");

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
  const float num_elements_in_axis = input_width * input_height;

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  constexpr int32_t kMinValue = std::numeric_limits<uint8_t>::min();
  constexpr int32_t kMaxValue = std::numeric_limits<uint8_t>::max();

  float temp = input_zero_point * input_scale / output_scale;
  temp = temp > 0 ? temp + 0.5f : temp - 0.5f;
  int32_t bias = output_zero_point - static_cast<int32_t>(temp);
  double real_scale =
      static_cast<double>(input_scale / (num_elements_in_axis * output_scale));

  int32_t multiplier;
  int shift;
  QuantizeMultiplier(real_scale, &multiplier, &shift);
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      int32_t acc = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          acc += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }
      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += bias;
      acc = std::min(std::max(acc, kMinValue), kMaxValue);
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          static_cast<uint8_t>(acc);
    }
  }
}

// Computes the mean of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis for quantized values.
template <typename T, typename U>
inline bool QuantizedMeanOrSum(const T* input_data, int32_t input_zero_point,
                               float input_scale, const int* input_dims,
                               const int input_num_dims, T* output_data,
                               int32_t output_zero_point, float output_scale,
                               const int* output_dims,
                               const int output_num_dims, const int* axis,
                               const int num_axis_dimensions, bool keep_dims,
                               int* temp_index, int* resolved_axis, U* temp_sum,
                               bool compute_sum) {
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
  if (!ResolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis,
                   &num_resolved_axis)) {
    return false;
  }

  if (!ReduceSumImpl<T, U>(input_data, input_dims, output_dims, input_num_dims,
                           output_num_dims, resolved_axis, num_resolved_axis,
                           temp_index, temp_sum)) {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx) {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (std::numeric_limits<size_t>::max() / num_elements_in_axis)) {
      return false;
    }
    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis > 0) {
    const float scale = input_scale / output_scale;
    if (compute_sum) {
      // TODO(b/116341117): Eliminate float and do this completely in 8bit.
      const float bias = -input_zero_point * scale * num_elements_in_axis;
      for (size_t idx = 0; idx < num_outputs; ++idx) {
        const U value =
            static_cast<U>(TfLiteRound(temp_sum[idx] * scale + bias)) +
            output_zero_point;
        output_data[idx] = static_cast<T>(value);
      }
    } else {
      const float bias = -input_zero_point * scale;
      for (size_t idx = 0; idx < num_outputs; ++idx) {
        float float_mean = static_cast<float>(temp_sum[idx]) /
                           static_cast<float>(num_elements_in_axis);
        float result = TfLiteMin(
            TfLiteRound(float_mean * scale + bias) + output_zero_point,
            static_cast<float>(std::numeric_limits<T>::max()));
        result = TfLiteMax(result,
                           static_cast<float>(std::numeric_limits<T>::min()));
        output_data[idx] = static_cast<T>(result);
      }
    }
  }
  return true;
}

template <typename T, typename U>
inline bool QuantizedMeanOrSumExtraArgs(
    const T* input_data, int32_t input_zero_point, float input_scale,
    const int* input_dims, const int input_num_dims, T* output_data,
    float output_scale, int32_t output_multiplier, int output_shift,
    int32_t output_zero_point, const int* output_dims,
    const int output_num_dims, const int* axis, const int num_axis_dimensions,
    bool keep_dims, int* temp_index, int* resolved_axis, U* temp_sum,
    bool compute_sum) {
  return QuantizedMeanOrSum<T, U>(
      input_data, input_zero_point, input_scale, input_dims, input_num_dims,
      output_data, output_zero_point, output_scale, output_dims,
      output_num_dims, axis, num_axis_dimensions, keep_dims, temp_index,
      resolved_axis, temp_sum, compute_sum);
}

template <typename T>
inline bool QuantizedReduceProd(const T* input_data, int32_t input_zero_point,
                                const RuntimeShape& input_shape, T* output_data,
                                int32_t output_zero_point,
                                const RuntimeShape& output_shape,
                                const int* axis,
                                const int64_t num_axis_dimensions,
                                bool keep_dims, int* temp_index,
                                int* resolved_axis, int32_t* temp_prod,
                                int32_t scaling_multiplier, int scaling_shift) {
  const int32_t kMinValue = std::numeric_limits<T>::min();
  const int32_t kMaxValue = std::numeric_limits<T>::max();

  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input_shape.DimensionsCount(), axis, num_axis_dimensions,
                   resolved_axis, &num_resolved_axis)) {
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

  if (!Reduce<T, int32_t>(
          input_data, input_shape.DimsData(), output_shape.DimsData(),
          input_shape.DimensionsCount(), output_shape.DimensionsCount(),
          resolved_axis, num_resolved_axis, temp_index, reducer_first,
          reducer_next, temp_prod)) {
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
