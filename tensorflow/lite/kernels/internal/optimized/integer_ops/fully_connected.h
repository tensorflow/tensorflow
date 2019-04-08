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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_FULLY_CONNECTED_H_

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"

namespace tflite {
namespace optimized_integer_ops {

inline void optimized_ops_preload_l1_stream(const int8_t* ptr) {
#ifdef GEMMLOWP_ARM_64
  asm volatile("prfm pldl1strm, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
#else
  gemmlowp::Prefetch(ptr);
#endif
}

inline void optimized_ops_preload_l1_keep(const int8_t* ptr) {
#ifdef GEMMLOWP_ARM_64
  asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
#else
  gemmlowp::Prefetch(ptr);
#endif
}

#ifdef USE_NEON
inline void FullyConnectedAsGEMVWorkerImpl(
    const RuntimeShape& input_shape, const int8_t* input_data,
    int32 input_offset, const RuntimeShape& filter_shape,
    const int8_t* filter_data, int32 filter_offset,
    const RuntimeShape& bias_shape, const int32* bias_data, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, const RuntimeShape& output_shape,
    int8_t* output_data, int row_start, int row_end) {
  gemmlowp::ScopedProfilingLabel label("FullyConnectedAsGEMVInt8/8bit");
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  const int output_dim_count = output_shape.DimensionsCount();
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(output_shape, output_dim_count - 1), 1);
  const int input_size = FlatSizeSkipDim(input_shape, 0);
  static constexpr int kPeel = 4;
  const bool shift_left = (output_shift > 0);
  for (int k = 0; k < input_size; k += 64) {
    optimized_ops_preload_l1_stream(input_data + k);
  }
  for (int k = 0; k < kPeel * input_size; k += 64) {
    optimized_ops_preload_l1_stream(filter_data + k);
  }

  TFLITE_DCHECK_GE(row_end - row_start, kPeel);

  for (int out = row_start; out < row_end; out += kPeel) {
    out = std::min(out, row_end - kPeel);
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = acc0;
    int32x4_t acc2 = acc0;
    int32x4_t acc3 = acc0;
    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    const int16x8_t filter_offset_vec = vdupq_n_s16(filter_offset);
    int in = 0;
    for (; in <= input_size - 16; in += 16) {
      const int8x16_t input_val_s8 = vld1q_s8(input_data + in);
      const int8_t* filter_ptr = filter_data + in + out * input_size;
      int8x16_t filter_val_s8_0 = vld1q_s8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      filter_ptr += input_size;
      int8x16_t filter_val_s8_1 = vld1q_s8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      filter_ptr += input_size;
      int8x16_t filter_val_s8_2 = vld1q_s8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      filter_ptr += input_size;
      int8x16_t filter_val_s8_3 = vld1q_s8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      int16x8_t input_val_0, input_val_1;
      int8x8_t low = vget_low_s8(input_val_s8);
      int8x8_t high = vget_high_s8(input_val_s8);
      input_val_0 = vmovl_s8(low);
      input_val_1 = vmovl_s8(high);
      input_val_0 = vaddq_s16(input_val_0, input_offset_vec);
      input_val_1 = vaddq_s16(input_val_1, input_offset_vec);
      low = vget_low_s8(filter_val_s8_0);
      high = vget_high_s8(filter_val_s8_0);
      int16x8_t filter_val_0_0 = vmovl_s8(low);
      int16x8_t filter_val_0_1 = vmovl_s8(high);
      filter_val_0_0 = vaddq_s16(filter_val_0_0, filter_offset_vec);
      filter_val_0_1 = vaddq_s16(filter_val_0_1, filter_offset_vec);
      low = vget_low_s8(filter_val_s8_1);
      high = vget_high_s8(filter_val_s8_1);
      int16x8_t filter_val_1_0 = vmovl_s8(low);
      int16x8_t filter_val_1_1 = vmovl_s8(high);
      filter_val_1_0 = vaddq_s16(filter_val_1_0, filter_offset_vec);
      filter_val_1_1 = vaddq_s16(filter_val_1_1, filter_offset_vec);
      low = vget_low_s8(filter_val_s8_2);
      high = vget_high_s8(filter_val_s8_2);
      int16x8_t filter_val_2_0 = vmovl_s8(low);
      int16x8_t filter_val_2_1 = vmovl_s8(high);
      filter_val_2_0 = vaddq_s16(filter_val_2_0, filter_offset_vec);
      filter_val_2_1 = vaddq_s16(filter_val_2_1, filter_offset_vec);
      low = vget_low_s8(filter_val_s8_3);
      high = vget_high_s8(filter_val_s8_3);
      int16x8_t filter_val_3_0 = vmovl_s8(low);
      int16x8_t filter_val_3_1 = vmovl_s8(high);
      filter_val_3_0 = vaddq_s16(filter_val_3_0, filter_offset_vec);
      filter_val_3_1 = vaddq_s16(filter_val_3_1, filter_offset_vec);
      acc0 = vmlal_s16(acc0, vget_low_s16(filter_val_0_0),
                       vget_low_s16(input_val_0));
      acc1 = vmlal_s16(acc1, vget_low_s16(filter_val_1_0),
                       vget_low_s16(input_val_0));
      acc2 = vmlal_s16(acc2, vget_low_s16(filter_val_2_0),
                       vget_low_s16(input_val_0));
      acc3 = vmlal_s16(acc3, vget_low_s16(filter_val_3_0),
                       vget_low_s16(input_val_0));
      acc0 = vmlal_s16(acc0, vget_low_s16(filter_val_0_1),
                       vget_low_s16(input_val_1));
      acc1 = vmlal_s16(acc1, vget_low_s16(filter_val_1_1),
                       vget_low_s16(input_val_1));
      acc2 = vmlal_s16(acc2, vget_low_s16(filter_val_2_1),
                       vget_low_s16(input_val_1));
      acc3 = vmlal_s16(acc3, vget_low_s16(filter_val_3_1),
                       vget_low_s16(input_val_1));
      acc0 = vmlal_s16(acc0, vget_high_s16(filter_val_0_0),
                       vget_high_s16(input_val_0));
      acc1 = vmlal_s16(acc1, vget_high_s16(filter_val_1_0),
                       vget_high_s16(input_val_0));
      acc2 = vmlal_s16(acc2, vget_high_s16(filter_val_2_0),
                       vget_high_s16(input_val_0));
      acc3 = vmlal_s16(acc3, vget_high_s16(filter_val_3_0),
                       vget_high_s16(input_val_0));
      acc0 = vmlal_s16(acc0, vget_high_s16(filter_val_0_1),
                       vget_high_s16(input_val_1));
      acc1 = vmlal_s16(acc1, vget_high_s16(filter_val_1_1),
                       vget_high_s16(input_val_1));
      acc2 = vmlal_s16(acc2, vget_high_s16(filter_val_2_1),
                       vget_high_s16(input_val_1));
      acc3 = vmlal_s16(acc3, vget_high_s16(filter_val_3_1),
                       vget_high_s16(input_val_1));
    }
    for (; in <= input_size - 8; in += 8) {
      const int8x8_t input_val_s8 = vld1_s8(input_data + in);
      const int8_t* filter_ptr = filter_data + in + out * input_size;
      int8x8_t filter_val_s8_0 = vld1_s8(filter_ptr);
      filter_ptr += input_size;
      int8x8_t filter_val_s8_1 = vld1_s8(filter_ptr);
      filter_ptr += input_size;
      int8x8_t filter_val_s8_2 = vld1_s8(filter_ptr);
      filter_ptr += input_size;
      int8x8_t filter_val_s8_3 = vld1_s8(filter_ptr);
      int16x8_t input_val = vmovl_s8(input_val_s8);
      input_val = vaddq_s16(input_val, input_offset_vec);
      int16x8_t filter_val_0 = vmovl_s8(filter_val_s8_0);
      filter_val_0 = vaddq_s16(filter_val_0, filter_offset_vec);
      int16x8_t filter_val_1 = vmovl_s8(filter_val_s8_1);
      filter_val_1 = vaddq_s16(filter_val_1, filter_offset_vec);
      int16x8_t filter_val_2 = vmovl_s8(filter_val_s8_2);
      filter_val_2 = vaddq_s16(filter_val_2, filter_offset_vec);
      int16x8_t filter_val_3 = vmovl_s8(filter_val_s8_3);
      filter_val_3 = vaddq_s16(filter_val_3, filter_offset_vec);
      acc0 =
          vmlal_s16(acc0, vget_low_s16(filter_val_0), vget_low_s16(input_val));
      acc1 =
          vmlal_s16(acc1, vget_low_s16(filter_val_1), vget_low_s16(input_val));
      acc2 =
          vmlal_s16(acc2, vget_low_s16(filter_val_2), vget_low_s16(input_val));
      acc3 =
          vmlal_s16(acc3, vget_low_s16(filter_val_3), vget_low_s16(input_val));
      acc0 = vmlal_s16(acc0, vget_high_s16(filter_val_0),
                       vget_high_s16(input_val));
      acc1 = vmlal_s16(acc1, vget_high_s16(filter_val_1),
                       vget_high_s16(input_val));
      acc2 = vmlal_s16(acc2, vget_high_s16(filter_val_2),
                       vget_high_s16(input_val));
      acc3 = vmlal_s16(acc3, vget_high_s16(filter_val_3),
                       vget_high_s16(input_val));
    }
    if (in < input_size) {
      int32 buf[16];
      vst1q_s32(buf + 0, acc0);
      vst1q_s32(buf + 4, acc1);
      vst1q_s32(buf + 8, acc2);
      vst1q_s32(buf + 12, acc3);
      for (; in < input_size; in++) {
        int lane = (in + 8 - input_size) % 4;
        const int32 input_val = input_data[in] + input_offset;
        for (int k = 0; k < kPeel; k++) {
          int32 filter_val =
              filter_data[in + (out + k) * input_size] + filter_offset;
          buf[lane + 4 * k] += filter_val * input_val;
        }
      }
      acc0 = vld1q_s32(buf + 0);
      acc1 = vld1q_s32(buf + 4);
      acc2 = vld1q_s32(buf + 8);
      acc3 = vld1q_s32(buf + 12);
    }

    // Horizontally reduce accumulators
    int32x2_t pairwise_reduced_acc_0 =
        vpadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
    int32x2_t pairwise_reduced_acc_1 =
        vpadd_s32(vget_low_s32(acc1), vget_high_s32(acc1));
    int32x2_t pairwise_reduced_acc_2 =
        vpadd_s32(vget_low_s32(acc2), vget_high_s32(acc2));
    int32x2_t pairwise_reduced_acc_3 =
        vpadd_s32(vget_low_s32(acc3), vget_high_s32(acc3));
    const int32x2_t reduced_lo =
        vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
    const int32x2_t reduced_hi =
        vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
    // Add bias values.
    int32x4_t bias_vec = vld1q_s32(bias_data + out);
    reduced = vaddq_s32(reduced, bias_vec);
    if (shift_left) {
      const int32 multiplier_power_of_two = 1 << output_shift;
      reduced = vmulq_n_s32(reduced, multiplier_power_of_two);
      reduced = vqrdmulhq_n_s32(reduced, output_multiplier);
    } else {
      // Multiply by the fixed-point multiplier.
      reduced = vqrdmulhq_n_s32(reduced, output_multiplier);
      // Rounding-shift-right.
      using gemmlowp::RoundingDivideByPOT;
      reduced = RoundingDivideByPOT(reduced, -output_shift);
    }
    // Add the output offset.
    const int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
    reduced = vaddq_s32(reduced, output_offset_vec);
    // Narrow values down to 16 bit signed.
    const int16x4_t res16 = vqmovn_s32(reduced);
    // Narrow values down to 8 bit signed, saturating.
    int8x8_t res8 = vqmovn_s16(vcombine_s16(res16, res16));
    // Apply the clamping from the activation function
    res8 = vmax_s8(res8, vdup_n_s8(output_activation_min));
    res8 = vmin_s8(res8, vdup_n_s8(output_activation_max));
    // Store results to destination.
    vst1_lane_s8(output_data + out + 0, res8, 0);
    vst1_lane_s8(output_data + out + 1, res8, 1);
    vst1_lane_s8(output_data + out + 2, res8, 2);
    vst1_lane_s8(output_data + out + 3, res8, 3);
  }
}

struct FullyConnectedAsGEMVWorkerTask : public gemmlowp::Task {
  FullyConnectedAsGEMVWorkerTask(
      const RuntimeShape& input_shape, const int8_t* input_data,
      int32 input_offset, const RuntimeShape& filter_shape,
      const int8_t* filter_data, int32 filter_offset,
      const RuntimeShape& bias_shape, const int32* bias_data,
      int32 output_offset, int32 output_multiplier, int output_shift,
      int32 output_activation_min, int32 output_activation_max,
      const RuntimeShape& output_shape, int8_t* output_data, int row_start,
      int row_end)
      : input_shape_(input_shape),
        input_data_(input_data),
        input_offset_(input_offset),
        filter_shape_(filter_shape),
        filter_data_(filter_data),
        filter_offset_(filter_offset),
        bias_shape_(bias_shape),
        bias_data_(bias_data),
        output_offset_(output_offset),
        output_multiplier_(output_multiplier),
        output_shift_(output_shift),
        output_activation_min_(output_activation_min),
        output_activation_max_(output_activation_max),
        output_shape_(output_shape),
        output_data_(output_data),
        row_start_(row_start),
        row_end_(row_end) {}

  void Run() override {
    FullyConnectedAsGEMVWorkerImpl(
        input_shape_, input_data_, input_offset_, filter_shape_, filter_data_,
        filter_offset_, bias_shape_, bias_data_, output_offset_,
        output_multiplier_, output_shift_, output_activation_min_,
        output_activation_max_, output_shape_, output_data_, row_start_,
        row_end_);
  }

  const RuntimeShape& input_shape_;
  const int8_t* input_data_;
  int32 input_offset_;
  const RuntimeShape& filter_shape_;
  const int8_t* filter_data_;
  int32 filter_offset_;
  const RuntimeShape& bias_shape_;
  const int32* bias_data_;
  int32 output_offset_;
  int32 output_multiplier_;
  int output_shift_;
  int32 output_activation_min_;
  int32 output_activation_max_;
  const RuntimeShape& output_shape_;
  int8_t* output_data_;
  gemmlowp::GemmContext* gemm_context_;
  int row_start_;
  int row_end_;
};

inline void FullyConnectedAsGEMV(
    const RuntimeShape& input_shape, const int8_t* input_data,
    int32 input_offset, const RuntimeShape& filter_shape,
    const int8_t* filter_data, int32 filter_offset,
    const RuntimeShape& bias_shape, const int32* bias_data, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, const RuntimeShape& output_shape,
    int8_t* output_data, gemmlowp::GemmContext* gemm_context) {
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  const int input_size = FlatSizeSkipDim(input_shape, 0);
  static constexpr int kKernelRows = 4;
  const int thread_count = gemmlowp::HowManyThreads<kKernelRows>(
      gemm_context->max_num_threads(), output_rows, batches, input_size);
  if (thread_count == 1) {
    // Single-thread case: do the computation on the current thread, don't
    // use a threadpool
    FullyConnectedAsGEMVWorkerImpl(
        input_shape, input_data, input_offset, filter_shape, filter_data,
        filter_offset, bias_shape, bias_data, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_shape, output_data, 0, output_rows);
    return;
  }

  // Multi-threaded case: use the gemmlowp context's threadpool.
  TFLITE_DCHECK_GT(thread_count, 1);
  std::vector<gemmlowp::Task*> tasks(thread_count);
  const int kRowsPerWorker = gemmlowp::RoundUp<kKernelRows>(
      gemmlowp::CeilQuotient(output_rows, thread_count));
  int row_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    int row_end = std::min(output_rows, row_start + kRowsPerWorker);
    tasks[i] = new FullyConnectedAsGEMVWorkerTask(
        input_shape, input_data, input_offset, filter_shape, filter_data,
        filter_offset, bias_shape, bias_data, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_shape, output_data, row_start, row_end);
    row_start = row_end;
  }
  TFLITE_DCHECK_EQ(row_start, output_rows);
  gemm_context->workers_pool()->Execute(tasks);
}
#endif  // USE_NEON

struct GemmlowpOutputPipeline {
  typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
      ColVectorMap;
  typedef std::tuple<gemmlowp::OutputStageBiasAddition<ColVectorMap>,
                     gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent,
                     gemmlowp::OutputStageClamp,
                     gemmlowp::OutputStageSaturatingCastToInt8>
      Pipeline;
  static Pipeline MakeExp(const int32* bias_data, int output_rows,
                          int32 output_offset, int32 output_multiplier,
                          int output_left_shift, int32 output_activation_min,
                          int32 output_activation_max) {
    ColVectorMap bias_vector(bias_data, output_rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_offset;
    quantize_down_stage.result_fixedpoint_multiplier = output_multiplier;
    quantize_down_stage.result_exponent = output_left_shift;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = output_activation_min;
    clamp_stage.max = output_activation_max;
    gemmlowp::OutputStageSaturatingCastToInt8 saturating_cast_stage;
    return std::make_tuple(bias_addition_stage, quantize_down_stage,
                           clamp_stage, saturating_cast_stage);
  }
};

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape, int8* output_data,
    gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label("FullyConnectedInt8/8bit");

#ifdef USE_NEON
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  if (batches == 1) {
    const int output_size = MatchingDim(filter_shape, filter_dim_count - 2,
                                        output_shape, output_dim_count - 1);
    if (output_size >= 4) {
      return FullyConnectedAsGEMV(
          input_shape, input_data, input_offset, filter_shape, filter_data,
          filter_offset, bias_shape, bias_data, output_offset,
          output_multiplier, output_shift, output_activation_min,
          output_activation_max, output_shape, output_data, gemm_context);
    }
  }
#endif  // USE_NEON

#ifdef GEMMLOWP_NEON
  const int filter_rows = filter_shape.Dims(filter_dim_count - 2);
  const int filter_cols = filter_shape.Dims(filter_dim_count - 1);
  TFLITE_DCHECK_EQ(filter_shape.FlatSize(), filter_rows * filter_cols);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);

  gemmlowp::MatrixMap<const int8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, output_rows, filter_cols, filter_cols);
  gemmlowp::MatrixMap<const int8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, filter_cols, batches, filter_cols);
  gemmlowp::MatrixMap<int8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, batches, output_rows);
  const auto& output_pipeline = GemmlowpOutputPipeline::MakeExp(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);

  gemmlowp::GemmWithOutputPipeline<
      int8, int8, gemmlowp::SignedL8R8WithLhsNonzeroBitDepthParams>(
      gemm_context, filter_matrix, input_matrix, &output_matrix, filter_offset,
      input_offset, output_pipeline);
  return;
#endif  // GEMMLOWP_NEON

  // If both GEMMLOWP_NEON && NEON paths are skipped, fallback to reference
  // implementation.
  reference_integer_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data, gemm_context);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_FULLY_CONNECTED_H_
