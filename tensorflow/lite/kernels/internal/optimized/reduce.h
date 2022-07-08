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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_REDUCE_OPS_H_

#include <stdint.h>

#include <algorithm>
#include <limits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops_utils.h"
#include "tensorflow/lite/kernels/internal/optimized/reduce_utils.h"
#include "tensorflow/lite/kernels/internal/reduce_common.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

inline void MeanImpl(const tflite::MeanParams& op_params,
                     const RuntimeShape& input_shape, const uint8_t* input_data,
                     int32 multiplier, int32 shift, int32 bias,
                     const RuntimeShape& output_shape, uint8_t* output_data,
                     int start_depth, int end_depth) {
  ruy::profiler::ScopeLabel label("Mean4D/Uint8/MeanImpl");

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  constexpr int32_t kMinValue = std::numeric_limits<uint8_t>::min();
  constexpr int32_t kMaxValue = std::numeric_limits<uint8_t>::max();

#ifdef USE_NEON
  const int32x4_t bias_dup = vdupq_n_s32(bias);
  const int32x4_t min_dup = vdupq_n_s32(kMinValue);
  const int32x4_t max_dup = vdupq_n_s32(kMaxValue);
#endif  // USE_NEON

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    int out_d = start_depth;
#ifdef USE_NEON

    for (; out_d <= end_depth - 16; out_d += 16) {
      int32x4x4_t temp_sum;
      temp_sum.val[0] = vdupq_n_s32(0);
      temp_sum.val[1] = vdupq_n_s32(0);
      temp_sum.val[2] = vdupq_n_s32(0);
      temp_sum.val[3] = vdupq_n_s32(0);
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const uint8_t* input_data_ptr =
              input_data + Offset(input_shape, out_b, in_h, in_w, out_d);
          uint8x16_t input_data_val = vld1q_u8(input_data_ptr);

          int16x8_t input_data_low_shift =
              vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_data_val)));
          int16x8_t input_data_high_shift =
              vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_data_val)));

          int32x4_t input_low_low =
              vmovl_s16(vget_low_s16(input_data_low_shift));
          int32x4_t input_high_low =
              vmovl_s16(vget_high_s16(input_data_low_shift));
          int32x4_t input_low_high =
              vmovl_s16(vget_low_s16(input_data_high_shift));
          int32x4_t input_high_high =
              vmovl_s16(vget_high_s16(input_data_high_shift));

          temp_sum.val[0] = vaddq_s32(temp_sum.val[0], input_low_low);
          temp_sum.val[1] = vaddq_s32(temp_sum.val[1], input_high_low);
          temp_sum.val[2] = vaddq_s32(temp_sum.val[2], input_low_high);
          temp_sum.val[3] = vaddq_s32(temp_sum.val[3], input_high_high);
        }
      }

      temp_sum =
          MultiplyByQuantizedMultiplier4Rows(temp_sum, multiplier, shift);

      temp_sum.val[0] = vaddq_s32(temp_sum.val[0], bias_dup);
      temp_sum.val[1] = vaddq_s32(temp_sum.val[1], bias_dup);
      temp_sum.val[2] = vaddq_s32(temp_sum.val[2], bias_dup);
      temp_sum.val[3] = vaddq_s32(temp_sum.val[3], bias_dup);

      temp_sum.val[0] = vminq_s32(vmaxq_s32(temp_sum.val[0], min_dup), max_dup);
      temp_sum.val[1] = vminq_s32(vmaxq_s32(temp_sum.val[1], min_dup), max_dup);
      temp_sum.val[2] = vminq_s32(vmaxq_s32(temp_sum.val[2], min_dup), max_dup);
      temp_sum.val[3] = vminq_s32(vmaxq_s32(temp_sum.val[3], min_dup), max_dup);

      uint16x4_t narrowed_low_low =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[0]));
      uint16x4_t narrowed_high_low =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[1]));
      uint16x4_t narrowed_low_high =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[2]));
      uint16x4_t narrowed_high_high =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[3]));

      uint16x8_t combined_low =
          vcombine_u16(narrowed_low_low, narrowed_high_low);
      uint16x8_t combined_high =
          vcombine_u16(narrowed_low_high, narrowed_high_high);

      uint8x8_t narrowed_low = vmovn_u16(combined_low);
      uint8x8_t narrowed_high = vmovn_u16(combined_high);

      uint8x16_t combined_output = vcombine_u8(narrowed_low, narrowed_high);

      uint8_t* output_data_ptr =
          output_data + Offset(output_shape, out_b, 0, 0, out_d);
      vst1q_u8(output_data_ptr, combined_output);
    }
#endif  // USE_NEON

    for (; out_d < end_depth; ++out_d) {
      int acc = 0;
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

struct MeanWorkerTask : cpu_backend_threadpool::Task {
  MeanWorkerTask(const tflite::MeanParams& op_params,
                 const RuntimeShape& input_shape, const uint8_t* input_data,
                 int32 multiplier, int32 shift, int32 bias,
                 const RuntimeShape& output_shape, uint8_t* output_data,
                 int start_height, int end_height)
      : op_params(op_params),
        input_shape(input_shape),
        input_data(input_data),
        multiplier(multiplier),
        shift(shift),
        bias(bias),
        output_shape(output_shape),
        output_data(output_data),
        start_height(start_height),
        end_height(end_height) {}

  void Run() override {
    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, start_height, end_height);
  }

 private:
  const tflite::MeanParams& op_params;
  const RuntimeShape& input_shape;
  const uint8_t* input_data;
  int32 multiplier;
  int32 shift;
  int32 bias;
  const RuntimeShape& output_shape;
  uint8_t* output_data;
  int start_height;
  int end_height;
};

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const uint8_t* input_data, int32 input_zero_point,
                 float input_scale, const RuntimeShape& unextended_output_shape,
                 uint8_t* output_data, int32 output_zero_point,
                 float output_scale, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("Mean4D/Uint8");
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const float num_elements_in_axis = input_width * input_height;

  float temp = input_zero_point * input_scale / output_scale;
  temp = temp > 0 ? temp + 0.5f : temp - 0.5f;
  int32_t bias = output_zero_point - static_cast<int32_t>(temp);
  float real_scale = input_scale / (num_elements_in_axis * output_scale);

  int32 multiplier, shift;
  QuantizeMultiplier(real_scale, &multiplier, &shift);

  constexpr int kMinDepthPerThread = 8;
  int thread_count = output_depth / kMinDepthPerThread;
  thread_count = thread_count > 0 ? thread_count : 1;
  const int capped_thread_count =
      std::min(thread_count, cpu_backend_context->max_num_threads());

  if (capped_thread_count == 1) {
    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, 0, output_depth);
  } else {
    // Instead parallel for batch, we loop for the output_depth since batch
    // is typical 1.
    std::vector<MeanWorkerTask> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(capped_thread_count);
    int depth_start = 0;
    for (int i = 0; i < capped_thread_count; ++i) {
      // Try to distribute the tasks as even as possible.
      int depth_end = depth_start +
                      (output_depth - depth_start) / (capped_thread_count - i);
      tasks.emplace_back(op_params, input_shape, input_data, multiplier, shift,
                         bias, output_shape, output_data, depth_start,
                         depth_end);
      depth_start = depth_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

template <typename T, typename U>
inline bool MeanGeneral(const T* input_data, const int* input_dims,
                        const int input_num_dims, T* output_data,
                        const int* output_dims, const int output_num_dims,
                        const int* axis, const int num_axis_dimensions,
                        bool keep_dims, int* temp_index, int* resolved_axis,
                        U* temp_sum) {
  return reference_ops::Mean(input_data, input_dims, input_num_dims,
                             output_data, output_dims, output_num_dims, axis,
                             num_axis_dimensions, keep_dims, temp_index,
                             resolved_axis, temp_sum);
}

template <>
inline bool MeanGeneral<float, float>(
    const float* input_data, const int* input_dims, const int input_num_dims,
    float* output_data, const int* output_dims, const int output_num_dims,
    const int* axis, const int num_axis_dimensions, bool keep_dims,
    int* temp_index, int* resolved_axis, float* temp_sum) {
  // Handle reduce_mean for the last dimensions.
  if (num_axis_dimensions == 1 && axis[0] == (input_num_dims - 1)) {
    ruy::profiler::ScopeLabel label("MeanLastDim/Float");
    int output_size = 1;
    for (int i = 0; i < input_num_dims - 1; ++i) {
      output_size *= input_dims[i];
    }
    const int last_input_dim = input_dims[axis[0]];

    // TODO(b/152563685): Consider use eigen to cover more general cases.
    const MatrixMap<const float> in_mat(input_data, last_input_dim,
                                        output_size);
    VectorMap<float> out(output_data, output_size, 1);
    out = (in_mat.array().colwise().sum()) / static_cast<float>(last_input_dim);
    return true;
  }

  return reference_ops::Mean(input_data, input_dims, input_num_dims,
                             output_data, output_dims, output_num_dims, axis,
                             num_axis_dimensions, keep_dims, temp_index,
                             resolved_axis, temp_sum);
}

template <typename T>
struct SumOp {
  inline T operator()(const T& a) const { return a; }
  inline T operator()(const T& a, const T& b) const { return a + b; }
  static constexpr T kNeutralElement = T(0);
};

template <typename T, typename U>
struct CastSumOp {
  inline U operator()(const T& a) const { return static_cast<U>(a); }
  inline U operator()(const U& a, const T& b) const {
    return a + static_cast<U>(b);
  }
  static constexpr U kNeutralElement = U(0);
};

template <typename T>
struct ProdOp {
  inline T operator()(const T& a) const { return a; }
  inline T operator()(const T& a, const T& b) const { return a * b; }
  static constexpr T kNeutralElement = T(1);
};

template <typename T>
struct MaxOp {
  inline T operator()(const T& a) const { return a; }
  inline T operator()(const T& a, const T& b) const { return (a > b) ? a : b; }
  static constexpr T kNeutralElement = std::numeric_limits<T>::lowest();
};

template <typename T>
struct MinOp {
  inline T operator()(const T& a) const { return a; }
  inline T operator()(const T& a, const T& b) const { return (a < b) ? a : b; }
  static constexpr T kNeutralElement = std::numeric_limits<T>::max();
};

struct AndOp {
  inline bool operator()(bool a) const { return a; }
  inline bool operator()(bool a, bool b) const { return a && b; }
  static constexpr bool kNeutralElement = true;
};

struct OrOp {
  inline bool operator()(bool a) const { return a; }
  inline bool operator()(bool a, bool b) const { return a || b; }
  static constexpr bool kNeutralElement = false;
};

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

// Reduces the input over either odd or even dimensions using Op.
// One recursive call for each dimension is made.
// 'depth' is the depth of recursion.
// 'parity' indicates whether odd or even dimensions are being reduced.
// ReducerFirst is applied to the first element to be written to each output
// position.
// ReducerNext is applied to each subsequent element to be written to each
// output position.
template <typename T, typename U, typename ReducerFirst, typename ReducerNext>
inline std::pair<const T*, U*> ReduceImpl(const T* input_data,
                                          const int* input_dims, U* output_data,
                                          int depth, int parity, bool next,
                                          const ReducerFirst& reducer_first,
                                          const ReducerNext& reducer_next) {
  // The output pointer is incremented conditionally depending on whether the
  // odd or even dimension is being reduced.
  // The input pointer is always incremented as each input is read once.
  if (depth > 0) {
    U* future_output = output_data;
    bool update_output = (depth % 2) == parity;
    for (int i = 0; i < input_dims[0]; ++i) {
      if (i > 0 && !update_output) {
        next = true;
      }
      std::tie(input_data, future_output) =
          ReduceImpl(input_data, &input_dims[1], output_data, depth - 1, parity,
                     next, reducer_first, reducer_next);
      if (update_output) {
        output_data = future_output;
      }
    }
    output_data = future_output;
  } else {
    // Reduce the final dimension.
    if (parity) {
      // Reduce the even dimension. The entire dimension is reduced into one
      // value.
      U res = next ? reducer_next(*output_data, *input_data++)
                   : reducer_first(*input_data++);
      for (int i = 1; i < input_dims[0]; ++i) {
        res = reducer_next(res, *input_data++);
      }
      *output_data++ = res;
    } else {
      // Reduce the odd dimension. Each input is accumulated into a separate
      // output.
      if (!next) {
        for (int i = 0; i < input_dims[0]; ++i) {
          U res = reducer_first(*input_data++);
          *output_data++ = res;
        }
      } else {
        for (int i = 0; i < input_dims[0]; ++i) {
          U res = *output_data;
          res = reducer_next(res, *input_data++);
          *output_data++ = res;
        }
      }
    }
  }
  return {input_data, output_data};
}

// A generic reduce method that can be used for reduce_sum, reduce_mean, etc.
// This method iterates through input data and reduce elements along the
// dimensions given in axis. ReducerFirst is used the first time each output
// element is written and ReducerNext is used for all subsequent writes.
template <typename In, typename Out, typename ReducerFirst,
          typename ReducerNext>
inline bool Reduce(const In* input_data, const int* input_dims,
                   const int input_num_dims, const int* axis,
                   const int num_axis, Out* output_data,
                   const ReducerFirst& reducer_first,
                   const ReducerNext& reducer_next) {
  const int parity = (axis[num_axis - 1] == input_num_dims - 1) ? 1 : 0;
  ReduceImpl(input_data, input_dims, output_data, input_num_dims - 1, parity,
             /*next=*/false, reducer_first, reducer_next);
  return true;
}

// Computes the mean or sum of elements across dimensions given in axis.
// It does so in two stages, first calculates the sum of elements along the axis
// then divides it by the number of element in axis for quantized values.
template <typename T, typename U>
bool QuantizedMeanOrSum(const T* input_data, int32_t input_zero_point,
                        float input_scale, const int* input_dims,
                        const int input_num_dims, T* output_data,
                        int32_t output_zero_point, float output_scale,
                        const int* output_dims, const int output_num_dims,
                        const int* axis, const int num_axis_dimensions,
                        bool keep_dims, int* normalized_dims,
                        int* resolved_axis, U* temp_sum, bool compute_sum) {
  ruy::profiler::ScopeLabel label(compute_sum ? "QuantizedSum"
                                              : "QuantizedMean");
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
  if (!reduce_utils::ResolveAxis(input_num_dims, axis, num_axis_dimensions,
                                 resolved_axis, &num_resolved_axis, input_dims,
                                 normalized_dims, &normalized_num_dims)) {
    return false;
  }

  if (!Reduce<T, U, CastSumOp<T, U>, CastSumOp<T, U>>(
          input_data, normalized_dims, normalized_num_dims, resolved_axis,
          num_resolved_axis, temp_sum, CastSumOp<T, U>(), CastSumOp<T, U>())) {
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
    const float scale = input_scale / output_scale;
    if (compute_sum) {
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

using ops::builtin::reduce::ReduceType;

template <typename T>
inline bool ReduceDispatcher(const T* input_data, const int* input_dims,
                             const int input_num_dims, const int* output_dims,
                             int output_num_dims, T* output_data,
                             const int* axis, const int64_t num_axis_dimensions,
                             ReduceType reduce_type) {
  T init_value;
  switch (reduce_type) {
    case ReduceType::kProd:
      init_value = ProdOp<T>::kNeutralElement;
      break;
    case ReduceType::kSum:
      init_value = SumOp<T>::kNeutralElement;
      break;
    case ReduceType::kMin:
      init_value = MinOp<T>::kNeutralElement;
      break;
    case ReduceType::kMax:
      init_value = MaxOp<T>::kNeutralElement;
      break;
    default:
      return false;
  }
  // Return early when input shape has zero dim. This is done after initializing
  // data for output tensor because there are cases that the input tensor is
  // empty but output tensor is not. In that case, output tensor should be
  // filled with Op::kNeutralElement.
  for (int i = 0; i < input_num_dims; ++i) {
    if (input_dims[i] == 0) {
      return reference_ops::InitTensorDataForReduce(
          output_dims, output_num_dims, init_value, output_data);
    }
  }

  switch (reduce_type) {
    case ReduceType::kProd:
      return Reduce<T, T, ProdOp<T>, ProdOp<T>>(
          input_data, input_dims, input_num_dims, axis, num_axis_dimensions,
          output_data, ProdOp<T>(), ProdOp<T>());
    case ReduceType::kSum:
      return Reduce<T, T, SumOp<T>, SumOp<T>>(
          input_data, input_dims, input_num_dims, axis, num_axis_dimensions,
          output_data, SumOp<T>(), SumOp<T>());
    case ReduceType::kMin:
      return Reduce<T, T, MinOp<T>, MinOp<T>>(
          input_data, input_dims, input_num_dims, axis, num_axis_dimensions,
          output_data, MinOp<T>(), MinOp<T>());
    case ReduceType::kMax:
      return Reduce<T, T, MaxOp<T>, MaxOp<T>>(
          input_data, input_dims, input_num_dims, axis, num_axis_dimensions,
          output_data, MaxOp<T>(), MaxOp<T>());
    default:
      return false;
  }
}

template <>
inline bool ReduceDispatcher<bool>(const bool* input_data,
                                   const int* input_dims,
                                   const int input_num_dims,
                                   const int* output_dims, int output_num_dims,
                                   bool* output_data, const int* axis,
                                   const int64_t num_axis_dimensions,
                                   ReduceType reduce_type) {
  bool init_value;
  switch (reduce_type) {
    case ReduceType::kAny:
      init_value = OrOp::kNeutralElement;
      break;
    case ReduceType::kAll:
      init_value = AndOp::kNeutralElement;
      break;
    default:
      return false;
  }
  // Return early when input shape has zero dim. This is done after initializing
  // data for output tensor because there are cases that the input tensor is
  // empty but output tensor is not. In that case, output tensor should be
  // filled with Op::kNeutralElement.
  for (int i = 0; i < input_num_dims; ++i) {
    if (input_dims[i] == 0) {
      return reference_ops::InitTensorDataForReduce(
          output_dims, output_num_dims, init_value, output_data);
    }
  }
  switch (reduce_type) {
    case ReduceType::kAll:
      return Reduce<bool, bool, AndOp, AndOp>(
          input_data, input_dims, input_num_dims, axis, num_axis_dimensions,
          output_data, AndOp(), AndOp());
    case ReduceType::kAny:
      return Reduce<bool, bool, OrOp, OrOp>(
          input_data, input_dims, input_num_dims, axis, num_axis_dimensions,
          output_data, OrOp(), OrOp());
    default:
      return false;
  }
}

// Calculate the reduced product by rescaling each multiplication step to
// avoid an overflow.
template <typename T>
struct ReducerFirst {
  explicit ReducerFirst(int input_zero_point_arg)
      : input_zero_point(input_zero_point_arg) {}
  int32_t operator()(T in) const { return in - input_zero_point; }
  int input_zero_point;
};

template <typename T>
struct ReducerNext {
  ReducerNext(int32_t input_zero_point_arg, int32_t scaling_multiplier_arg,
              int32_t scaling_shift_arg)
      : input_zero_point(input_zero_point_arg),
        scaling_multiplier(scaling_multiplier_arg),
        scaling_shift(scaling_shift_arg) {}
  int32_t operator()(int32_t current, T in) const {
    const int64_t result =
        static_cast<int64_t>(current) * (in - input_zero_point);
    return MultiplyByQuantizedMultiplier(result, scaling_multiplier,
                                         scaling_shift);
  }
  int32_t input_zero_point, scaling_multiplier, scaling_shift;
};

template <typename T>
inline bool QuantizedReduceProd(
    const T* input_data, int32_t input_zero_point,
    const RuntimeShape& input_shape, T* output_data, int32_t output_zero_point,
    const RuntimeShape& output_shape, const int* axis,
    const int64_t num_axis_dimensions, int* resolved_axis, int* normalized_dims,
    int32_t* temp_prod, int32_t scaling_multiplier, int scaling_shift) {
  const int32_t kMinValue = std::numeric_limits<T>::min();
  const int32_t kMaxValue = std::numeric_limits<T>::max();

  // Resolve axis.
  int num_resolved_axis = 0;
  int normalized_num_dims = 0;
  if (!reduce_utils::ResolveAxis(input_shape.DimensionsCount(), axis,
                                 num_axis_dimensions, resolved_axis,
                                 &num_resolved_axis, input_shape.DimsData(),
                                 normalized_dims, &normalized_num_dims)) {
    return false;
  }

  if (!Reduce<T, int32_t, ReducerFirst<T>, ReducerNext<T>>(
          input_data, normalized_dims, normalized_num_dims, resolved_axis,
          num_resolved_axis, temp_prod, ReducerFirst<T>(input_zero_point),
          ReducerNext<T>(input_zero_point, scaling_multiplier,
                         scaling_shift))) {
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

// Computes the generic value (i.e., sum/max/min/prod) of elements across
// dimensions given in axis. It needs to pass in init_value and reducer.
template <typename T>
inline bool ReduceGeneric(const T* input_data, const int* input_dims,
                          const int input_num_dims, T* output_data,
                          const int* output_dims, const int output_num_dims,
                          const int* axis, const int64_t num_axis_dimensions,
                          int* resolved_axis, int* normalized_dims,
                          ReduceType reduce_type) {
  int num_resolved_axis = 0;
  int normalized_num_dims = 0;
  if (!reduce_utils::ResolveAxis(input_num_dims, axis, num_axis_dimensions,
                                 resolved_axis, &num_resolved_axis, input_dims,
                                 normalized_dims, &normalized_num_dims)) {
    return false;
  }
  if (num_resolved_axis == 0) {
    optimized_ops::ReduceIsCopy(input_data, input_dims, input_num_dims,
                                output_data);
    return true;
  }
  return ReduceDispatcher(input_data, normalized_dims, normalized_num_dims,
                          output_dims, output_num_dims, output_data,
                          resolved_axis, num_resolved_axis, reduce_type);
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_OPTIMIZED_OPS_H_
