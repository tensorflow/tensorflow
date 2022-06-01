/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>

#include "third_party/eigen3/Eigen/Core"
#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/add_n.h"
#include "tensorflow/lite/kernels/internal/reference/arg_min_max.h"
#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"
#include "tensorflow/lite/kernels/internal/reference/batch_to_space_nd.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/cast.h"
#include "tensorflow/lite/kernels/internal/reference/ceil.h"
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"
#include "tensorflow/lite/kernels/internal/reference/concatenation.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/depth_to_space.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/div.h"
#include "tensorflow/lite/kernels/internal/reference/elu.h"
#include "tensorflow/lite/kernels/internal/reference/exp.h"
#include "tensorflow/lite/kernels/internal/reference/fill.h"
#include "tensorflow/lite/kernels/internal/reference/floor.h"
#include "tensorflow/lite/kernels/internal/reference/floor_div.h"
#include "tensorflow/lite/kernels/internal/reference/floor_mod.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/gather.h"
#include "tensorflow/lite/kernels/internal/reference/hard_swish.h"
#include "tensorflow/lite/kernels/internal/reference/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/leaky_relu.h"
#include "tensorflow/lite/kernels/internal/reference/log_softmax.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/lstm_cell.h"
#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/neg.h"
#include "tensorflow/lite/kernels/internal/reference/pad.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/reference/resize_bilinear.h"
#include "tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h"
#include "tensorflow/lite/kernels/internal/reference/round.h"
#include "tensorflow/lite/kernels/internal/reference/slice.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/space_to_batch_nd.h"
#include "tensorflow/lite/kernels/internal/reference/space_to_depth.h"
#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"
#include "tensorflow/lite/kernels/internal/reference/string_comparisons.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/reference/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
namespace tflite {

namespace reference_ops {

template <typename T>
inline void Relu(const RuntimeShape& input_shape, const T* input_data,
                 const RuntimeShape& output_shape, T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T lower = 0;
    const T clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

template <typename T>
inline void Relu0To1(const RuntimeShape& input_shape, const T* input_data,
                     const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Relu0To1 (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T upper = 1;
    const T lower = 0;
    const T clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

template <typename T>
inline void Relu1(const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Relu1 (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T upper = 1;
    const T lower = -1;
    const T clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void Relu6(const RuntimeShape& input_shape, const float* input_data,
                  const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Relu6 (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 6;
    const float lower = 0;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

template <typename T>
inline void ReluX(const tflite::ReluParams& params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Quantized ReluX (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const int32 val = static_cast<int32_t>(input_data[i]);
    int32 clamped = params.output_offset +
                    MultiplyByQuantizedMultiplier(val - params.input_offset,
                                                  params.output_multiplier,
                                                  params.output_shift);
    clamped = std::max(params.quantized_activation_min, clamped);
    clamped = std::min(params.quantized_activation_max, clamped);
    output_data[i] = static_cast<T>(clamped);
  }
}

template <typename T>
inline void ReluX(const tflite::ActivationParams& params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Quantized ReluX (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  const T max_value = params.quantized_activation_max;
  const T min_value = params.quantized_activation_min;
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T clamped = val > max_value   ? max_value
                      : val < min_value ? min_value
                                        : val;
    output_data[i] = clamped;
  }
}

// TODO(jiawen): We can implement BroadcastMul on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
inline void BroadcastMulFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8* output_data) {
  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input2_offset = unswitched_params.input1_offset;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const uint8* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const uint8* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise Mul of
  // sections of the arrays.
  uint8* output_data_ptr = output_data;
  const uint8* input1_data_ptr = input1_data;
  const uint8* input2_data_reset = input2_data;
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  for (int i0 = 0; i0 < y0; ++i0) {
    const uint8* input2_data_ptr;
    for (int i1 = 0; i1 < y1; ++i1) {
      input2_data_ptr = input2_data_reset;
      for (int i2 = 0; i2 < y2; ++i2) {
        for (int i3 = 0; i3 < y3; ++i3) {
          MulElementwise(y4, params, input1_data_ptr, input2_data_ptr,
                         output_data_ptr);
          input2_data_ptr += y4;
          output_data_ptr += y4;
        }
        input1_data_ptr += y4;
      }
    }
    input2_data_reset = input2_data_ptr;
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  ruy::profiler::ScopeLabel label("Mul/Int16");

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    output_data[i] = unclamped_result.raw();
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  ruy::profiler::ScopeLabel label("Mul/Int16Uint8");
  int32 output_offset = params.output_offset;
  int32 output_activation_min = params.quantized_activation_min;
  int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    int16 rescaled_result =
        gemmlowp::RoundingDivideByPOT(unclamped_result.raw(), 8);
    int16 clamped_result =
        std::min<int16>(output_activation_max - output_offset, rescaled_result);
    clamped_result =
        std::max<int16>(output_activation_min - output_offset, clamped_result);
    output_data[i] = output_offset + clamped_result;
  }
}

inline void Sub16(const ArithmeticParams& params,
                  const RuntimeShape& input1_shape, const int16_t* input1_data,
                  const RuntimeShape& input2_shape, const int16_t* input2_data,
                  const RuntimeShape& output_shape, int16_t* output_data) {
  ruy::profiler::ScopeLabel label("Sub/Int16");
  const int input1_shift = params.input1_shift;
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  const int16 output_activation_min = params.quantized_activation_min;
  const int16 output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK(input1_shift == 0 || params.input2_shift == 0);
  TFLITE_DCHECK_LE(input1_shift, 0);
  TFLITE_DCHECK_LE(params.input2_shift, 0);
  const int16* not_shift_input = input1_shift == 0 ? input1_data : input2_data;
  const int16* shift_input = input1_shift == 0 ? input2_data : input1_data;
  const int input_right_shift =
      input1_shift == 0 ? -params.input2_shift : -input1_shift;

  if (input1_shift == 0) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    for (int i = 0; i < flat_size; ++i) {
      F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
      F0 scaled_input = F0::FromRaw(
          gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
      F0 result = SaturatingSub(input_ready_scaled, scaled_input);
      const int16 raw_output = result.raw();
      const int16 clamped_output = std::min(
          output_activation_max, std::max(output_activation_min, raw_output));
      output_data[i] = clamped_output;
    }
  } else {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    for (int i = 0; i < flat_size; ++i) {
      F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
      F0 scaled_input = F0::FromRaw(
          gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
      F0 result = SaturatingSub(scaled_input, input_ready_scaled);
      const int16 raw_output = result.raw();
      const int16 clamped_output = std::min(
          output_activation_max, std::max(output_activation_min, raw_output));
      output_data[i] = clamped_output;
    }
  }
}

template <typename Scalar>
void Pack(const PackParams& params, const RuntimeShape* const* input_shapes,
          const Scalar* const* input_data, const RuntimeShape& output_shape,
          Scalar* output_data) {
  ruy::profiler::ScopeLabel label("Pack");
  const int dimensions = output_shape.DimensionsCount();
  int axis = params.axis;
  int inputs_count = params.inputs_count;

  int outer_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= output_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = params.axis + 1; i < dimensions; i++) {
    copy_size *= output_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ((**input_shapes).FlatSize(), copy_size * outer_size);

  for (int i = 0; i < inputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      const Scalar* input_ptr = input_data[i] + copy_size * k;
      int loc = k * inputs_count * copy_size + i * copy_size;
      memcpy(output_data + loc, input_ptr, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar>
void Unpack(const UnpackParams& params, const RuntimeShape& input_shape,
            const Scalar* input_data, const RuntimeShape& output_shape,
            Scalar* const* output_datas) {
  ruy::profiler::ScopeLabel label("Unpack");
  const int dimensions = input_shape.DimensionsCount();
  const int outputs_count = params.num_split;

  int outer_size = 1;
  int axis = params.axis;
  if (axis < 0) {
    axis += dimensions;
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, dimensions);
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; ++i) {
    copy_size *= input_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ(output_shape.FlatSize(), copy_size * outer_size);

  for (int i = 0; i < outputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      Scalar* output_ptr = output_datas[i] + copy_size * k;
      int loc = k * outputs_count * copy_size + i * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar>
void PackWithScaling(const PackParams& params,
                     const RuntimeShape* const* input_shapes,
                     const uint8* const* input_data,
                     const RuntimeShape& output_shape, uint8* output_data) {
  ruy::profiler::ScopeLabel label("PackWithScaling");
  const int dimensions = output_shape.DimensionsCount();
  int axis = params.axis;
  const int32* input_zeropoint = params.input_zeropoint;
  const float* input_scale = params.input_scale;
  int inputs_count = params.inputs_count;
  const int32 output_zeropoint = params.output_zeropoint;
  const float output_scale = params.output_scale;

  int outer_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= output_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; i++) {
    copy_size *= output_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ((**input_shapes).FlatSize(), copy_size * outer_size);

  Scalar* output_ptr = output_data;
  const float inverse_output_scale = 1.f / output_scale;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      if (input_zeropoint[i] == output_zeropoint &&
          input_scale[i] == output_scale) {
        memcpy(output_ptr, input_data[i] + k * copy_size,
               copy_size * sizeof(Scalar));
      } else {
        assert(false);
        const float scale = input_scale[i] * inverse_output_scale;
        const float bias = -input_zeropoint[i] * scale;
        auto input_ptr = input_data[i];
        for (int j = 0; j < copy_size; ++j) {
          const int32_t value =
              static_cast<int32_t>(std::round(input_ptr[j] * scale + bias)) +
              output_zeropoint;
          output_ptr[j] =
              static_cast<uint8_t>(std::max(std::min(255, value), 0));
        }
      }
      output_ptr += copy_size;
    }
  }
}

template <typename Scalar>
void DepthConcatenation(const ConcatenationParams& params,
                        const RuntimeShape* const* input_shapes,
                        const Scalar* const* input_data,
                        const RuntimeShape& output_shape, Scalar* output_data) {
  ruy::profiler::ScopeLabel label("DepthConcatenation");
  auto params_copy = params;
  params_copy.axis = 3;
  Concatenation(params_copy, input_shapes, input_data, output_shape,
                output_data);
}

template <typename Scalar>
void Split(const SplitParams& params, const RuntimeShape& input_shape,
           const Scalar* input_data, const RuntimeShape* const* output_shapes,
           Scalar* const* output_data) {
  ruy::profiler::ScopeLabel label("Split");
  const int split_dimensions = input_shape.DimensionsCount();
  int axis = params.axis < 0 ? params.axis + split_dimensions : params.axis;
  int outputs_count = params.num_split;
  TFLITE_DCHECK_LT(axis, split_dimensions);

  int64_t split_size = 0;
  for (int i = 0; i < outputs_count; i++) {
    TFLITE_DCHECK_EQ(output_shapes[i]->DimensionsCount(), split_dimensions);
    for (int j = 0; j < split_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*output_shapes[i], j, input_shape, j);
      }
    }
    split_size += output_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(split_size, input_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  // For all output arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i) {
    base_inner_size *= input_shape.Dims(i);
  }

  const Scalar* input_ptr = input_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < outputs_count; ++i) {
      const int copy_size = output_shapes[i]->Dims(axis) * base_inner_size;
      memcpy(output_data[i] + k * copy_size, input_ptr,
             copy_size * sizeof(Scalar));
      input_ptr += copy_size;
    }
  }
}

inline int NodeOffset(int b, int h, int w, int height, int width) {
  return (b * height + h) * width + w;
}

inline void LocalResponseNormalization(
    const tflite::LocalResponseNormalizationParams& op_params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    for (int c = 0; c < depth; ++c) {
      const int begin_input_c = std::max(0, c - op_params.range);
      const int end_input_c = std::min(depth, c + op_params.range);
      float accum = 0.f;
      for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
        const float input_val = input_data[i * depth + input_c];
        accum += input_val * input_val;
      }
      const float multiplier =
          std::pow(op_params.bias + op_params.alpha * accum, -op_params.beta);
      output_data[i * depth + c] = input_data[i * depth + c] * multiplier;
    }
  }
}

inline void Dequantize(const RuntimeShape& input_shape,
                       const Eigen::half* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = static_cast<float>(input_data[i]);
  }
}

inline void FakeQuant(const tflite::FakeQuantParams& op_params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("FakeQuant");
  float rmin = op_params.minmax.min;
  float rmax = op_params.minmax.max;
  int num_bits = op_params.num_bits;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_DCHECK_LE(rmin, 0.0f);
  TFLITE_DCHECK_GE(rmax, 0.0f);
  TFLITE_DCHECK_LT(rmin, rmax);

  // Code matches tensorflow's FakeQuantWithMinMaxArgsFunctor.
  int quant_min = 0;
  int quant_max = (1 << num_bits) - 1;
  float nudged_min, nudged_max, nudged_scale;
  NudgeQuantizationRange(rmin, rmax, quant_min, quant_max, &nudged_min,
                         &nudged_max, &nudged_scale);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  FakeQuantizeArray(nudged_scale, nudged_min, nudged_max, input_data,
                    output_data, flat_size);
}

// Common subroutine for both `GatherNd` and `GatherNdString`.
struct GatherNdHelperResult {
  int n_slices;
  int slice_size;
  int indices_nd;
  std::vector<int> dims_to_count;
};

// Returns common values being used on both `GatherNd` and `GatherNdString`.
inline GatherNdHelperResult GatherNdHelper(const RuntimeShape& params_shape,
                                           const RuntimeShape& indices_shape) {
  GatherNdHelperResult ret;
  ret.n_slices = 1;
  ret.slice_size = 1;
  const int indices_dims = indices_shape.DimensionsCount();
  ret.indices_nd = indices_shape.Dims(indices_dims - 1);
  const int params_dims = params_shape.DimensionsCount();
  for (int i = 0; i < indices_dims - 1; ++i) {
    ret.n_slices *= indices_shape.Dims(i);
  }
  if (ret.n_slices == 0) return ret;

  for (int i = ret.indices_nd; i < params_dims; ++i) {
    ret.slice_size *= params_shape.Dims(i);
  }

  int remain_flat_size = params_shape.FlatSize();
  ret.dims_to_count = std::vector<int>(ret.indices_nd, 0);
  for (int i = 0; i < ret.indices_nd; ++i) {
    ret.dims_to_count[i] = remain_flat_size / params_shape.Dims(i);
    remain_flat_size = ret.dims_to_count[i];
  }

  return ret;
}

template <typename ParamsT, typename IndicesT = int32>
inline void GatherNd(const RuntimeShape& params_shape,
                     const ParamsT* params_data,
                     const RuntimeShape& indices_shape,
                     const IndicesT* indices_data,
                     const RuntimeShape& output_shape, ParamsT* output_data) {
  ruy::profiler::ScopeLabel label("GatherNd");

  const GatherNdHelperResult res = GatherNdHelper(params_shape, indices_shape);
  for (int i = 0; i < res.n_slices; ++i) {
    int from_pos = 0;
    for (int j = 0; j < res.indices_nd; ++j) {
      from_pos += indices_data[i * res.indices_nd + j] * res.dims_to_count[j];
    }
    std::memcpy(output_data + i * res.slice_size, params_data + from_pos,
                sizeof(ParamsT) * res.slice_size);
  }
}

#ifndef TF_LITE_STATIC_MEMORY
template <typename IndicesT = int32>
inline void GatherNdString(const RuntimeShape& params_shape,
                           const TfLiteTensor* params_data,
                           const RuntimeShape& indices_shape,
                           const IndicesT* indices_data,
                           const RuntimeShape& output_shape,
                           TfLiteTensor* output_data) {
  ruy::profiler::ScopeLabel label("GatherNdString");

  const GatherNdHelperResult res = GatherNdHelper(params_shape, indices_shape);
  DynamicBuffer buffer;
  for (int i = 0; i < res.n_slices; ++i) {
    int from_pos = 0;
    for (int j = 0; j < res.indices_nd; ++j) {
      from_pos += indices_data[i * res.indices_nd + j] * res.dims_to_count[j];
    }
    for (int j = 0; j < res.slice_size; ++j) {
      buffer.AddString(GetString(params_data, from_pos + j));
    }
  }
  buffer.WriteToTensor(output_data, /*new_shape=*/nullptr);
}
#endif

template <typename IndicesT, typename UpdatesT>
inline void ScatterNd(const RuntimeShape& indices_shape,
                      const IndicesT* indices_data,
                      const RuntimeShape& updates_shape,
                      const UpdatesT* updates_data,
                      const RuntimeShape& output_shape, UpdatesT* output_data) {
  ruy::profiler::ScopeLabel label("ScatterNd");

  int n_slices = 1;
  int slice_size = 1;
  const int outer_dims = indices_shape.DimensionsCount() - 1;
  const int indices_nd = indices_shape.Dims(outer_dims);
  const int updates_dims = updates_shape.DimensionsCount();
  for (int i = 0; i < outer_dims; ++i) {
    n_slices *= indices_shape.Dims(i);
  }
  for (int i = outer_dims; i < updates_dims; ++i) {
    slice_size *= updates_shape.Dims(i);
  }

  int output_flat_size = output_shape.FlatSize();
  int remain_flat_size = output_flat_size;
  std::vector<int> dims_to_count(indices_nd, 0);
  for (int i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / output_shape.Dims(i);
    remain_flat_size = dims_to_count[i];
  }

  memset(output_data, 0, sizeof(UpdatesT) * output_flat_size);
  for (int i = 0; i < n_slices; ++i) {
    int to_pos = 0;
    for (int j = 0; j < indices_nd; ++j) {
      IndicesT idx = indices_data[i * indices_nd + j];
      TFLITE_DCHECK(0 <= idx && idx < output_shape.Dims(j));
      to_pos += idx * dims_to_count[j];
    }
    for (int j = 0; j < slice_size; j++) {
      output_data[to_pos + j] += updates_data[i * slice_size + j];
    }
  }
}

template <typename T>
void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  const int flat_size = MatchingFlatSize(input1_shape, output_shape);

  auto min_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] > min_value ? min_value : input1_data[i];
  }
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Minimum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  const int flat_size = MatchingFlatSize(input1_shape, output_shape);

  auto max_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] < max_value ? max_value : input1_data[i];
  }
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Maximum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T1, typename T2, typename T3>
void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
            const T3* input2_data, const RuntimeShape& output_shape,
            T2* output_data) {
  ArgMinMax(input1_shape, input1_data, input2_data, output_shape, output_data,
            std::greater<T1>());
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T1, typename T2, typename T3>
inline void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
                   const RuntimeShape& input2_shape, const T3* input2_data,
                   const RuntimeShape& output_shape, T2* output_data) {
  // Drop shape of second input: not needed.
  ArgMax(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename D, typename T>
void Select(const RuntimeShape& input_condition_shape,
            const D* input_condition_data, const RuntimeShape& input_x_shape,
            const T* input_x_data, const RuntimeShape& input_y_shape,
            const T* input_y_data, const RuntimeShape& output_shape,
            T* output_data) {
  int64_t flatsize;
  // Allow select operator executions on mixed scalar tensors and one element
  // tensors.
  if (input_condition_shape.FlatSize() == 1 && input_x_shape.FlatSize() == 1 &&
      input_y_shape.FlatSize() == 1 && output_shape.FlatSize() == 1) {
    flatsize = 1;
  } else {
    flatsize = MatchingFlatSize(input_condition_shape, input_x_shape,
                                input_y_shape, output_shape);
  }
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] =
        input_condition_data[i] ? input_x_data[i] : input_y_data[i];
  }
}

template <typename D, typename T>
void RankOneSelect(const RuntimeShape& input_condition_shape,
                   const D* input_condition_data,
                   const RuntimeShape& input_x_shape, const T* input_x_data,
                   const RuntimeShape& input_y_shape, const T* input_y_data,
                   const RuntimeShape& output_shape, T* output_data) {
  const int64_t outer_size = input_condition_shape.FlatSize();
  int64_t inner_size;
  if (input_condition_shape.DimensionsCount() == 0) {
    inner_size = MatchingFlatSize(input_x_shape, input_y_shape, output_shape);
  } else {
    TFLITE_DCHECK_EQ(
        MatchingDim(input_x_shape, 0, input_y_shape, 0, output_shape, 0),
        outer_size);
    inner_size =
        MatchingFlatSizeSkipDim(input_x_shape, 0, input_y_shape, output_shape);
  }

  int64_t offset = 0;
  for (int64_t i = 0; i < outer_size; i++) {
    const T* input_data = input_condition_data[i] ? input_x_data : input_y_data;
    memcpy(output_data + offset, input_data + offset, inner_size * sizeof(T));
    offset += inner_size;
  }
}

template <typename D, typename T>
void BroadcastSelect4DSlow(const RuntimeShape& input_condition_shape,
                           const D* input_condition_data,
                           const RuntimeShape& input_x_shape,
                           const T* input_x_data,
                           const RuntimeShape& input_y_shape,
                           const T* input_y_data,
                           const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_LE(input_condition_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(input_x_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(input_y_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), 4);

  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  NdArrayDesc<4> desc_condition;
  NdArrayDesc<4> desc_x;
  NdArrayDesc<4> desc_y;
  NdArrayDescsForElementwiseBroadcast(input_condition_shape, input_x_shape,
                                      input_y_shape, &desc_condition, &desc_x,
                                      &desc_y);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          const int condition_index =
              SubscriptToIndex(desc_condition, b, y, x, c);
          const int x_index = SubscriptToIndex(desc_x, b, y, x, c);
          const int y_index = SubscriptToIndex(desc_y, b, y, x, c);
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              input_condition_data[condition_index] ? input_x_data[x_index]
                                                    : input_y_data[y_index];
        }
      }
    }
  }
}

template <typename D, typename T>
void SelectTrueCoords(const RuntimeShape& input_condition_shape,
                      const D* input_condition_data, T* output_data) {
  const size_t size = input_condition_shape.FlatSize();
  if (size == 0) {
    // Dimension is zero, in which case we don't need to output.
    return;
  }
  const size_t cond_rank = input_condition_shape.DimensionsCount();

  std::vector<int> dims_to_count(cond_rank, 0);
  int cur_flat_size = size;
  for (int i = 0; i < cond_rank; ++i) {
    dims_to_count[i] = cur_flat_size / input_condition_shape.Dims(i);
    cur_flat_size = dims_to_count[i];
  }

  int output_index = 0;
  for (int i = 0; i < size; ++i) {
    if (input_condition_data[i] != D(0)) {
      // Insert the coordinate of the current item (row major) into output.
      int flat_index = i;
      for (int j = 0; j < cond_rank; ++j) {
        int coord_j = flat_index / dims_to_count[j];
        output_data[output_index * cond_rank + j] = coord_j;
        flat_index %= dims_to_count[j];
      }
      output_index++;
    }
  }
}

// For easy implementation, the indices is always a vector of size-4 vectors.
template <typename T, typename TI>
inline void SparseToDense(const std::vector<std::vector<TI>>& indices,
                          const T* values, T default_value,
                          bool value_is_scalar,
                          const RuntimeShape& unextended_output_shape,
                          T* output_data) {
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int value_count = indices.size();

  // First fill the output_data with default value.
  const int num_elements = output_shape.FlatSize();
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = default_value;
  }

  // Special handle for value is scalar case to avoid checking the boolean
  // condition within the loop every time.
  if (value_is_scalar) {
    for (int i = 0; i < value_count; ++i) {
      const std::vector<TI>& index = indices[i];
      TFLITE_DCHECK_EQ(index.size(), 4);
      const T value = *values;  // just use the first value.
      output_data[Offset(output_shape, index[0], index[1], index[2],
                         index[3])] = value;
    }
    return;
  }

  // Go through the values and indices to fill the sparse values.
  for (int i = 0; i < value_count; ++i) {
    const std::vector<TI>& index = indices[i];
    TFLITE_DCHECK_EQ(index.size(), 4);
    const T value = values[i];
    output_data[Offset(output_shape, index[0], index[1], index[2], index[3])] =
        value;
  }
}

template <typename T>
inline void Pow(const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = std::pow(input1_data[i], input2_data[i]);
  }
}

template <typename T>
inline void BroadcastPow4DSlow(const RuntimeShape& unextended_input1_shape,
                               const T* input1_data,
                               const RuntimeShape& unextended_input2_shape,
                               const T* input2_data,
                               const RuntimeShape& unextended_output_shape,
                               T* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = std::pow(in1_val, in2_val);
        }
      }
    }
  }
}

template <typename Scalar>
void Reverse(int axis, const RuntimeShape& input_shape,
             const Scalar* input_data, const RuntimeShape& output_shape,
             Scalar* output_data) {
  ruy::profiler::ScopeLabel label("Reverse");

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }

  int copy_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i) {
    copy_size *= input_shape.Dims(i);
  }

  const int dims_at_axis = input_shape.Dims(axis);
  for (int i = 0; i < outer_size; ++i) {
    for (int j = 0; j < dims_at_axis; ++j) {
      const int start_pos = (i * dims_at_axis + j) * copy_size;
      Scalar* output_ptr = output_data + start_pos;
      int loc = (i * dims_at_axis + dims_at_axis - j - 1) * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar, typename TS>
void ReverseSequence(const TS* seq_lengths, const int seq_dim,
                     const int batch_dim, const RuntimeShape& input_shape,
                     const Scalar* input_data, const RuntimeShape& output_shape,
                     Scalar* output_data) {
  ruy::profiler::ScopeLabel label("ReverseSequence");

  int outer_size = 1;
  int outer_dim = std::min(batch_dim, seq_dim);
  int medium_dim = std::max(batch_dim, seq_dim);
  for (int i = 0; i < outer_dim; ++i) {
    outer_size *= input_shape.Dims(i);
  }

  int medium_size = 1;
  for (int i = outer_dim + 1; i < medium_dim; ++i) {
    medium_size *= input_shape.Dims(i);
  }

  int copy_size = 1;
  for (int i = medium_dim + 1; i < input_shape.DimensionsCount(); ++i) {
    copy_size *= input_shape.Dims(i);
  }

  const int dims_at_outer_dim = input_shape.Dims(outer_dim);
  const int dims_at_medium_dim = input_shape.Dims(medium_dim);

  Scalar* output_ptr;
  if (batch_dim > seq_dim) {
    for (int i = 0; i < outer_size; ++i) {
      for (int j = 0; j < dims_at_outer_dim; ++j) {
        const int in_pos_base = (i * dims_at_outer_dim + j) * medium_size;
        for (int p = 0; p < medium_size; ++p) {
          for (int q = 0; q < dims_at_medium_dim; ++q) {
            const int in_pos =
                ((in_pos_base + p) * dims_at_medium_dim + q) * copy_size;
            const Scalar* in_ptr = input_data + in_pos;
            int sl = seq_lengths[q] - 1;
            if (j > sl) {
              output_ptr = output_data + in_pos;
            } else {
              const int out_pos_base =
                  (i * dims_at_outer_dim + sl - j) * medium_size;
              const int out_pos =
                  ((out_pos_base + p) * dims_at_medium_dim + q) * copy_size;
              output_ptr = output_data + out_pos;
            }
            memcpy(output_ptr, in_ptr, copy_size * sizeof(Scalar));
          }
        }
      }
    }
  } else if (batch_dim < seq_dim) {
    for (int i = 0; i < outer_size; ++i) {
      for (int j = 0; j < dims_at_outer_dim; ++j) {
        const int in_pos_base = (i * dims_at_outer_dim + j) * medium_size;
        int sl = seq_lengths[j] - 1;
        const int out_pos_base = (i * dims_at_outer_dim + j) * medium_size;
        for (int p = 0; p < medium_size; ++p) {
          for (int q = 0; q < dims_at_medium_dim; ++q) {
            const int in_pos =
                ((in_pos_base + p) * dims_at_medium_dim + q) * copy_size;
            const Scalar* in_ptr = input_data + in_pos;
            if (q > sl) {
              output_ptr = output_data + in_pos;
            } else {
              const int out_pos =
                  ((out_pos_base + p) * dims_at_medium_dim + sl - q) *
                  copy_size;
              output_ptr = output_data + out_pos;
            }
            memcpy(output_ptr, in_ptr, copy_size * sizeof(Scalar));
          }
        }
      }
    }
  }
}

template <typename T>
inline void SegmentSum(const RuntimeShape& input_shape, const T* input_data,
                       const RuntimeShape& segment_ids_shape,
                       const int32_t* segment_ids_data,
                       const RuntimeShape& output_shape, T* output_data) {
  const int segment_flat_size =
      MatchingFlatSizeSkipDim(input_shape, 0, output_shape);

  memset(output_data, 0, sizeof(T) * output_shape.FlatSize());

  for (int i = 0; i < input_shape.Dims(0); i++) {
    int output_index = segment_ids_data[i];
    for (int j = 0; j < segment_flat_size; ++j) {
      output_data[output_index * segment_flat_size + j] +=
          input_data[i * segment_flat_size + j];
    }
  }
}

template <typename T>
inline void UnsortedSegmentProd(const RuntimeShape& input_shape,
                                const T* input_data,
                                const RuntimeShape& segment_ids_shape,
                                const int32_t* segment_ids_data,
                                const int32_t num_segments,
                                const RuntimeShape& output_shape,
                                T* output_data) {
  for (int i = 0; i < output_shape.FlatSize(); ++i) {
    output_data[i] = 1;
  }
  const int segment_flat_size =
      MatchingFlatSizeSkipDim(input_shape, 0, output_shape);
  for (int i = 0; i < input_shape.Dims(0); i++) {
    int output_index = segment_ids_data[i];
    for (int j = 0; j < segment_flat_size; ++j) {
      output_data[output_index * segment_flat_size + j] *=
          input_data[i * segment_flat_size + j];
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_
