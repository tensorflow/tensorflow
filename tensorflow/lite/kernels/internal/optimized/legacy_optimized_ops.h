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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Unoptimized reference ops:
using reference_ops::ArgMax;
using reference_ops::ArgMinMax;
using reference_ops::Broadcast4DSlowGreater;
using reference_ops::Broadcast4DSlowGreaterEqual;
using reference_ops::Broadcast4DSlowGreaterEqualWithScaling;
using reference_ops::Broadcast4DSlowGreaterWithScaling;
using reference_ops::Broadcast4DSlowLess;
using reference_ops::Broadcast4DSlowLessEqual;
using reference_ops::Broadcast4DSlowLessEqualWithScaling;
using reference_ops::Broadcast4DSlowLessWithScaling;
using reference_ops::BroadcastAdd4DSlow;
using reference_ops::BroadcastGreater;
using reference_ops::BroadcastGreaterEqual;
using reference_ops::BroadcastLess;
using reference_ops::BroadcastLessEqual;
using reference_ops::BroadcastMul4DSlow;
using reference_ops::BroadcastSub4DSlow;
using reference_ops::Concatenation;
using reference_ops::ConcatenationWithScaling;
using reference_ops::DepthConcatenation;
using reference_ops::Div;
using reference_ops::FakeQuant;
using reference_ops::Gather;
using reference_ops::Greater;
using reference_ops::GreaterEqual;
using reference_ops::GreaterEqualWithScaling;
using reference_ops::GreaterWithScaling;
using reference_ops::Less;
using reference_ops::LessEqual;
using reference_ops::LessEqualWithScaling;
using reference_ops::LessWithScaling;
using reference_ops::Mean;
using reference_ops::RankOneSelect;
using reference_ops::Relu1;
using reference_ops::Relu6;
using reference_ops::ReluX;
using reference_ops::Select;
using reference_ops::SpaceToBatchND;
using reference_ops::Split;
using reference_ops::StridedSlice;
using reference_ops::TensorFlowSplit;

static constexpr int kDepthwiseReverseShift = -1;

template <typename Scalar, int N>
VectorMap<Scalar> MapAsVector(Scalar* data, const Dims<N>& dims) {
  const int size = FlatSize(dims);
  return VectorMap<Scalar>(data, size, 1);
}

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsRows(Scalar* data,
                                                const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsCols(Scalar* data,
                                               const Dims<N>& dims) {
  const int cols = dims.sizes[N - 1];
  int rows = 1;
  for (int d = 0; d < N - 1; d++) {
    rows *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar, int N>
ArrayMap<Scalar> MapAsArrayWithFirstDimAsRows(Scalar* data,
                                              const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return ArrayMap<Scalar>(data, rows, cols);
}

// TODO(b/62193649): this function is only needed as long
// as we have the --variable_batch hack.
template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithGivenNumberOfRows(Scalar* data,
                                                   const Dims<N>& dims,
                                                   int rows) {
  const int flatsize = FlatSize(dims);
  TFLITE_DCHECK((flatsize % rows) == 0);
  const int cols = flatsize / rows;
  return MatrixMap<Scalar>(data, rows, cols);
}

inline bool AreSameDims(const Dims<4>& dims1, const Dims<4>& dims2) {
  for (int i = 0; i < 4; i++) {
    if (dims1.sizes[i] != dims2.sizes[i]) {
      return false;
    }
  }
  return true;
}

inline void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height,
                          int dilation_width_factor, int dilation_height_factor,
                          int pad_width, int pad_height, int depth_multiplier,
                          float output_activation_min,
                          float output_activation_max, float* output_data,
                          const Dims<4>& output_dims) {
  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.depth_multiplier = depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  const RuntimeShape output_shape = DimsToShape(output_dims);
  const int output_height = output_shape.Dims(1);

  DepthwiseConvImpl(op_params, DimsToShape(input_dims), input_data,
                    DimsToShape(filter_dims), filter_data,
                    DimsToShape(bias_dims), bias_data, output_shape,
                    output_data, CpuFlags(), /*thread_start=*/0,
                    /*thread_end=*/output_height, /*thread_dim=*/1);
}

inline void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, int depth_multiplier,
                          float output_activation_min,
                          float output_activation_max, float* output_data,
                          const Dims<4>& output_dims) {
  DepthwiseConv(input_data, input_dims, filter_data, filter_dims, bias_data,
                bias_dims, stride_width, stride_height, 1, 1, pad_width,
                pad_height, depth_multiplier, output_activation_min,
                output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                   const float* filter_data, const Dims<4>& filter_dims,
                   const float* bias_data, const Dims<4>& bias_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int depth_multiplier, float* output_data,
                   const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  DepthwiseConv(input_data, input_dims, filter_data, filter_dims, bias_data,
                bias_dims, stride_width, stride_height, pad_width, pad_height,
                depth_multiplier, output_activation_min, output_activation_max,
                output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                   const float* filter_data, const Dims<4>& filter_dims,
                   const float* bias_data, const Dims<4>& bias_dims, int stride,
                   int pad_width, int pad_height, int depth_multiplier,
                   float* output_data, const Dims<4>& output_dims) {
  DepthwiseConv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
                    bias_dims, stride, stride, pad_width, pad_height,
                    depth_multiplier, output_data, output_dims);
}

template <DepthwiseConvOutputRounding kOutputRounding>
inline void LegacyDepthwiseConvWithRounding(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, int thread_start, int thread_end, int thread_dim) {
  ruy::profiler::ScopeLabel label("DepthwiseConv/8bit");
  const int depth_multiplier = params.depth_multiplier;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_GE(dilation_width_factor, 1);
  TFLITE_DCHECK_GE(dilation_height_factor, 1);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_depth = input_shape.Dims(3);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

// Enable for arm64 except for the Nvidia Linux 4 Tegra (L4T) running on
// Jetson TX-2. This compiler does not support the offsetof() macro.
#if defined(__aarch64__) && !defined(GOOGLE_L4T)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int output_shift = params.output_shift;

  // Call kernel optimized for depthwise convolutions using 3x3 filters if
  // parameters are supported.
  if (depthwise_conv::Fast3x3FilterKernelSupported(
          input_shape, filter_shape, stride_width, stride_height,
          dilation_width_factor, dilation_height_factor, pad_width, pad_height,
          depth_multiplier, output_shape, output_shift)) {
    ruy::profiler::ScopeLabel specialized_label("DepthwiseConv/8bit/3x3");
    depthwise_conv::DepthwiseConv3x3Filter<kOutputRounding>(
        params, input_shape, input_data, filter_shape, filter_data, bias_shape,
        bias_data, output_shape, output_data, thread_start, thread_end,
        thread_dim);
    return;
  }
#endif

  ruy::profiler::ScopeLabel specialized_label("DepthwiseConv/8bit/General");
  depthwise_conv::DepthwiseConvGeneral(params, input_shape, input_data,
                                       filter_shape, filter_data, bias_shape,
                                       bias_data, output_shape, output_data,
                                       thread_start, thread_end, thread_dim);
}

inline void LegacyDepthwiseConvImpl(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, int thread_start, int thread_end, int thread_dim) {
  return LegacyDepthwiseConvWithRounding<
      DepthwiseConvOutputRounding::kAwayFromZero>(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data, thread_start, thread_end,
      thread_dim);
}

inline void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                          int32 input_offset, const uint8* filter_data,
                          const Dims<4>& filter_dims, int32 filter_offset,
                          const int32* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height,
                          int dilation_width_factor, int dilation_height_factor,
                          int pad_width, int pad_height, int depth_multiplier,
                          int32 output_offset, int32 output_multiplier,
                          int output_shift, int32 output_activation_min,
                          int32 output_activation_max, uint8* output_data,
                          const Dims<4>& output_dims) {
  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.depth_multiplier = depth_multiplier;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kDepthwiseReverseShift * output_shift;

  const RuntimeShape output_shape = DimsToShape(output_dims);
  const int output_height = output_shape.Dims(1);

  LegacyDepthwiseConvImpl(
      op_params, DimsToShape(input_dims), input_data, DimsToShape(filter_dims),
      filter_data, DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
      output_data, /*thread_start=*/0,
      /*thread_end=*/output_height, /*thread_dim=*/1);
}

inline void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                          int32 input_offset, const uint8* filter_data,
                          const Dims<4>& filter_dims, int32 filter_offset,
                          const int32* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, int depth_multiplier,
                          int32 output_offset, int32 output_multiplier,
                          int output_shift, int32 output_activation_min,
                          int32 output_activation_max, uint8* output_data,
                          const Dims<4>& output_dims) {
  DepthwiseConv(input_data, input_dims, input_offset, filter_data, filter_dims,
                filter_offset, bias_data, bias_dims, stride_width,
                stride_height, 1, 1, pad_width, pad_height, depth_multiplier,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max, output_data,
                output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                   int32 input_offset, const uint8* filter_data,
                   const Dims<4>& filter_dims, int32 filter_offset,
                   const int32* bias_data, const Dims<4>& bias_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int depth_multiplier, int32 output_offset,
                   int32 output_multiplier, int output_shift,
                   int32 output_activation_min, int32 output_activation_max,
                   uint8* output_data, const Dims<4>& output_dims) {
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  DepthwiseConv(input_data, input_dims, input_offset, filter_data, filter_dims,
                filter_offset, bias_data, bias_dims, stride_width,
                stride_height, pad_width, pad_height, depth_multiplier,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max, output_data,
                output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                   int32 input_offset, const uint8* filter_data,
                   const Dims<4>& filter_dims, int32 filter_offset,
                   const int32* bias_data, const Dims<4>& bias_dims, int stride,
                   int pad_width, int pad_height, int depth_multiplier,
                   int32 output_offset, int32 output_multiplier,
                   int output_shift, int32 output_activation_min,
                   int32 output_activation_max, uint8* output_data,
                   const Dims<4>& output_dims) {
  DepthwiseConv<Ac>(input_data, input_dims, input_offset, filter_data,
                    filter_dims, filter_offset, bias_data, bias_dims, stride,
                    stride, pad_width, pad_height, depth_multiplier,
                    output_offset, output_multiplier, output_shift,
                    output_activation_min, output_activation_max, output_data,
                    output_dims);
}

template <typename T, typename TS>
struct LegacyDepthwiseConvWorkerTask : public gemmlowp::Task {
  LegacyDepthwiseConvWorkerTask(
      const DepthwiseParams& params, const RuntimeShape& input_shape,
      const T* input_data, const RuntimeShape& filter_shape,
      const T* filter_data, const RuntimeShape& bias_shape, const TS* bias_data,
      const RuntimeShape& output_shape, T* output_data, int thread_start,
      int thread_end, int thread_dim)
      : params_(params),
        input_shape_(input_shape),
        input_data_(input_data),
        filter_shape_(filter_shape),
        filter_data_(filter_data),
        bias_shape_(bias_shape),
        bias_data_(bias_data),
        output_shape_(output_shape),
        output_data_(output_data),
        thread_start_(thread_start),
        thread_end_(thread_end),
        thread_dim_(thread_dim) {}

  void Run() override {
    LegacyDepthwiseConvImpl(params_, input_shape_, input_data_, filter_shape_,
                            filter_data_, bias_shape_, bias_data_,
                            output_shape_, output_data_, thread_start_,
                            thread_end_, thread_dim_);
  }

 private:
  const DepthwiseParams& params_;
  const RuntimeShape& input_shape_;
  const T* input_data_;
  const RuntimeShape& filter_shape_;
  const T* filter_data_;
  const RuntimeShape& bias_shape_;
  const TS* bias_data_;
  const RuntimeShape& output_shape_;
  T* output_data_;
  int thread_start_;
  int thread_end_;
  int thread_dim_;
};

inline int HowManyConvThreads(const RuntimeShape& output_shape,
                              const RuntimeShape& filter_shape,
                              int thread_dim) {
  constexpr int kMinMulPerThread = 8;
  const int output_units = output_shape.Dims(thread_dim);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int num_mul_per_unit =
      FlatSizeSkipDim(output_shape, thread_dim) * filter_height * filter_width;
  const int min_units_per_thread = kMinMulPerThread / num_mul_per_unit + 1;
  int thread_count = output_units / min_units_per_thread;
  return thread_count;
}

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, gemmlowp::GemmContext* gemmlowp_context = nullptr) {
  ruy::profiler::ScopeLabel label("DepthwiseConv");

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int output_batches = output_shape.Dims(0);
  const int output_rows = output_shape.Dims(1);
  int thread_count_batch = HowManyConvThreads(output_shape, filter_shape, 0);
  int thread_count_row = HowManyConvThreads(output_shape, filter_shape, 1);
  int thread_dim, thread_count, thread_dim_size;
  if (thread_count_batch > thread_count_row) {
    thread_dim = 0;
    thread_dim_size = output_batches;
    thread_count = thread_count_batch;
  } else {
    thread_dim = 1;
    thread_dim_size = output_rows;
    thread_count = thread_count_row;
  }

  const int max_threads =
      gemmlowp_context ? gemmlowp_context->max_num_threads() : 1;
  thread_count = std::max(1, std::min(thread_count, max_threads));

  if (thread_count == 1) {
    LegacyDepthwiseConvImpl(params, input_shape, input_data, filter_shape,
                            filter_data, bias_shape, bias_data, output_shape,
                            output_data, /*thread_start=*/0,
                            /*thread_end=*/output_rows, /*thread_dim=*/1);
  } else {
    std::vector<gemmlowp::Task*> tasks(thread_count);
    int thread_start = 0;
    for (int i = 0; i < thread_count; ++i) {
      int thread_end =
          thread_start + (thread_dim_size - thread_start) / (thread_count - i);
      tasks[i] = new LegacyDepthwiseConvWorkerTask<uint8, int32>(
          params, input_shape, input_data, filter_shape, filter_data,
          bias_shape, bias_data, output_shape, output_data, thread_start,
          thread_end, thread_dim);
      thread_start = thread_end;
    }
    gemmlowp_context->workers_pool()->LegacyExecuteAndDestroyTasks(tasks);
  }
}

template <typename T, typename TS>
struct LegacyPerChannelDepthwiseConvWorkerTask : public gemmlowp::Task {
  LegacyPerChannelDepthwiseConvWorkerTask(
      const DepthwiseParams& params, const int32* output_multiplier,
      const int32* output_shift, const RuntimeShape& input_shape,
      const T* input_data, const RuntimeShape& filter_shape,
      const T* filter_data, const RuntimeShape& bias_shape, const TS* bias_data,
      const RuntimeShape& output_shape, T* output_data, int thread_start,
      int thread_end, int thread_dim)
      : params_(params),
        output_multiplier_(output_multiplier),
        output_shift_(output_shift),
        input_shape_(input_shape),
        input_data_(input_data),
        filter_shape_(filter_shape),
        filter_data_(filter_data),
        bias_shape_(bias_shape),
        bias_data_(bias_data),
        output_shape_(output_shape),
        output_data_(output_data),
        thread_start_(thread_start),
        thread_end_(thread_end),
        thread_dim_(thread_dim) {}

  void Run() override {
    optimized_integer_ops::DepthwiseConvImpl(
        params_, output_multiplier_, output_shift_, input_shape_, input_data_,
        filter_shape_, filter_data_, bias_shape_, bias_data_, output_shape_,
        output_data_, thread_start_, thread_end_, thread_dim_);
  }

 private:
  const DepthwiseParams& params_;
  const int32* output_multiplier_;
  const int32* output_shift_;
  const RuntimeShape& input_shape_;
  const T* input_data_;
  const RuntimeShape& filter_shape_;
  const T* filter_data_;
  const RuntimeShape& bias_shape_;
  const TS* bias_data_;
  const RuntimeShape& output_shape_;
  T* output_data_;
  int thread_start_;
  int thread_end_;
  int thread_dim_;
};

inline void DepthwiseConvPerChannel(
    const DepthwiseParams& params, const int32* output_multiplier,
    const int32* output_shift, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape, int8* output_data,
    gemmlowp::GemmContext* gemmlowp_context = nullptr) {
  ruy::profiler::ScopeLabel label("DepthwiseConvInt8");

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int output_batches = output_shape.Dims(0);
  const int output_rows = output_shape.Dims(1);
  int thread_count_batch = HowManyConvThreads(output_shape, filter_shape, 0);
  int thread_count_row = HowManyConvThreads(output_shape, filter_shape, 1);
  int thread_dim, thread_count, thread_dim_size;
  if (thread_count_batch > thread_count_row) {
    thread_dim = 0;
    thread_dim_size = output_batches;
    thread_count = thread_count_batch;
  } else {
    thread_dim = 1;
    thread_dim_size = output_rows;
    thread_count = thread_count_row;
  }

  const int max_threads =
      gemmlowp_context ? gemmlowp_context->max_num_threads() : 1;
  thread_count = std::max(1, std::min(thread_count, max_threads));

  if (thread_count == 1) {
    optimized_integer_ops::DepthwiseConvImpl(
        params, output_multiplier, output_shift, input_shape, input_data,
        filter_shape, filter_data, bias_shape, bias_data, output_shape,
        output_data, /*thread_start=*/0,
        /*thread_end=*/output_rows, /*thread_dim=*/1);
  } else {
    std::vector<gemmlowp::Task*> tasks(thread_count);
    int thread_start = 0;
    for (int i = 0; i < thread_count; ++i) {
      int thread_end =
          thread_start + (thread_dim_size - thread_start) / (thread_count - i);
      tasks[i] = new LegacyPerChannelDepthwiseConvWorkerTask<int8, int32>(
          params, output_multiplier, output_shift, input_shape, input_data,
          filter_shape, filter_data, bias_shape, bias_data, output_shape,
          output_data, thread_start, thread_end, thread_dim);
      thread_start = thread_end;
    }
    gemmlowp_context->workers_pool()->LegacyExecuteAndDestroyTasks(tasks);
  }
}

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
  DepthwiseConvImpl(params, input_shape, input_data, filter_shape, filter_data,
                    bias_shape, bias_data, output_shape, output_data,
                    CpuFlags(),
                    /*thread_start=*/0,
                    /*thread_end=*/output_shape.Dims(1), /*thread_dim=*/1);
}

inline void AddBiasAndEvalActivationFunction(const float* bias_data,
                                             const Dims<4>& bias_dims,
                                             float* array_data,
                                             const Dims<4>& array_dims,
                                             float output_activation_min,
                                             float output_activation_max) {
  AddBiasAndEvalActivationFunction(output_activation_min, output_activation_max,
                                   DimsToShape(bias_dims), bias_data,
                                   DimsToShape(array_dims), array_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AddBiasAndEvalActivationFunction(const float* bias_data,
                                      const Dims<4>& bias_dims,
                                      float* array_data,
                                      const Dims<4>& array_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  AddBiasAndEvalActivationFunction(bias_data, bias_dims, array_data, array_dims,
                                   output_activation_min,
                                   output_activation_max);
}

template <typename Lhs, typename Rhs, typename Result>
void Gemm(const Eigen::MatrixBase<Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs,
          Eigen::MatrixBase<Result>* result) {
  if (rhs.cols() == 1) {
    ruy::profiler::ScopeLabel label("GEMV");
    result->col(0).noalias() = lhs * rhs.col(0);
  } else {
    ruy::profiler::ScopeLabel label("GEMM");
    result->noalias() = lhs * rhs;
  }
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& bias_shape,
    const float* optional_bias_data, const RuntimeShape& output_shape,
    float* output_data) {
  ruy::profiler::ScopeLabel label("FullyConnected");
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  // TODO(b/62193649): this convoluted shape computation (determining
  // input_rows from the weights_dims, then MapAsMatrixWithGivenNumberOfRows)
  // is because the current --variable_batch hack consists in overwriting the
  // 3rd dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  // When that is fixed, this should become:
  // const auto input_matrix_map =
  //     MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  const int dims_count = weights_shape.DimensionsCount();
  const int input_rows = weights_shape.Dims(dims_count - 1);
  const auto input_matrix_map =
      MapAsMatrixWithGivenNumberOfRows(input_data, input_shape, input_rows);
  const auto filter_matrix_map =
      MapAsMatrixWithLastDimAsRows(weights_data, weights_shape);
  auto output_matrix_map =
      MapAsMatrixWithLastDimAsRows(output_data, output_shape);

  Gemm(filter_matrix_map.transpose(), input_matrix_map, &output_matrix_map);

  if (optional_bias_data != nullptr) {
    AddBiasAndEvalActivationFunction(
        output_activation_min, output_activation_max, bias_shape,
        optional_bias_data, output_shape, output_data);
  } else {
    const int flat_size = output_shape.FlatSize();
    for (int i = 0; i < flat_size; ++i) {
      output_data[i] = ActivationFunctionWithMinMax(
          output_data[i], output_activation_min, output_activation_max);
    }
  }
}

inline void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                           const float* weights_data,
                           const Dims<4>& weights_dims, const float* bias_data,
                           const Dims<4>& bias_dims,
                           float output_activation_min,
                           float output_activation_max, float* output_data,
                           const Dims<4>& output_dims) {
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(weights_dims), weights_data,
                 DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
                 output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                    const float* weights_data, const Dims<4>& weights_dims,
                    const float* bias_data, const Dims<4>& bias_dims,
                    float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  FullyConnected(input_data, input_dims, weights_data, weights_dims, bias_data,
                 bias_dims, output_activation_min, output_activation_max,
                 output_data, output_dims);
}

struct GemmlowpOutputPipeline {
  typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
      ColVectorMap;
  typedef std::tuple<gemmlowp::OutputStageBiasAddition<ColVectorMap>,
                     gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent,
                     gemmlowp::OutputStageClamp,
                     gemmlowp::OutputStageSaturatingCastToUint8>
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
    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    return std::make_tuple(bias_addition_stage, quantize_down_stage,
                           clamp_stage, saturating_cast_stage);
  }
};

struct GemmlowpOutputPipelineInt8 {
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

#ifdef USE_NEON
inline void LegacyFullyConnectedAsGEMVWorkerImpl(
    const RuntimeShape& input_shape, const uint8* input_data,
    int32 input_offset, const RuntimeShape& filter_shape,
    const uint8* filter_data, int32 filter_offset,
    const RuntimeShape& bias_shape, const int32* bias_data, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, const RuntimeShape& output_shape,
    uint8* output_data, int row_start, int row_end) {
  ruy::profiler::ScopeLabel label("FullyConnectedAsGEMV/8bit");
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
      const uint8x16_t input_val_u8 = vld1q_u8(input_data + in);
      const uint8* filter_ptr = filter_data + in + out * input_size;
      uint8x16_t filter_val_u8_0 = vld1q_u8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      filter_ptr += input_size;
      uint8x16_t filter_val_u8_1 = vld1q_u8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      filter_ptr += input_size;
      uint8x16_t filter_val_u8_2 = vld1q_u8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      filter_ptr += input_size;
      uint8x16_t filter_val_u8_3 = vld1q_u8(filter_ptr);
      optimized_ops_preload_l1_stream(filter_ptr + 64);
      int16x8_t input_val_0, input_val_1;
      uint8x8_t low = vget_low_u8(input_val_u8);
      uint8x8_t high = vget_high_u8(input_val_u8);
      input_val_0 = vreinterpretq_s16_u16(vmovl_u8(low));
      input_val_1 = vreinterpretq_s16_u16(vmovl_u8(high));
      input_val_0 = vaddq_s16(input_val_0, input_offset_vec);
      input_val_1 = vaddq_s16(input_val_1, input_offset_vec);
      low = vget_low_u8(filter_val_u8_0);
      high = vget_high_u8(filter_val_u8_0);
      int16x8_t filter_val_0_0 = vreinterpretq_s16_u16(vmovl_u8(low));
      int16x8_t filter_val_0_1 = vreinterpretq_s16_u16(vmovl_u8(high));
      filter_val_0_0 = vaddq_s16(filter_val_0_0, filter_offset_vec);
      filter_val_0_1 = vaddq_s16(filter_val_0_1, filter_offset_vec);
      low = vget_low_u8(filter_val_u8_1);
      high = vget_high_u8(filter_val_u8_1);
      int16x8_t filter_val_1_0 = vreinterpretq_s16_u16(vmovl_u8(low));
      int16x8_t filter_val_1_1 = vreinterpretq_s16_u16(vmovl_u8(high));
      filter_val_1_0 = vaddq_s16(filter_val_1_0, filter_offset_vec);
      filter_val_1_1 = vaddq_s16(filter_val_1_1, filter_offset_vec);
      low = vget_low_u8(filter_val_u8_2);
      high = vget_high_u8(filter_val_u8_2);
      int16x8_t filter_val_2_0 = vreinterpretq_s16_u16(vmovl_u8(low));
      int16x8_t filter_val_2_1 = vreinterpretq_s16_u16(vmovl_u8(high));
      filter_val_2_0 = vaddq_s16(filter_val_2_0, filter_offset_vec);
      filter_val_2_1 = vaddq_s16(filter_val_2_1, filter_offset_vec);
      low = vget_low_u8(filter_val_u8_3);
      high = vget_high_u8(filter_val_u8_3);
      int16x8_t filter_val_3_0 = vreinterpretq_s16_u16(vmovl_u8(low));
      int16x8_t filter_val_3_1 = vreinterpretq_s16_u16(vmovl_u8(high));
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
      const uint8x8_t input_val_u8 = vld1_u8(input_data + in);
      const uint8* filter_ptr = filter_data + in + out * input_size;
      uint8x8_t filter_val_u8_0 = vld1_u8(filter_ptr);
      filter_ptr += input_size;
      uint8x8_t filter_val_u8_1 = vld1_u8(filter_ptr);
      filter_ptr += input_size;
      uint8x8_t filter_val_u8_2 = vld1_u8(filter_ptr);
      filter_ptr += input_size;
      uint8x8_t filter_val_u8_3 = vld1_u8(filter_ptr);
      int16x8_t input_val = vreinterpretq_s16_u16(vmovl_u8(input_val_u8));
      input_val = vaddq_s16(input_val, input_offset_vec);
      int16x8_t filter_val_0 = vreinterpretq_s16_u16(vmovl_u8(filter_val_u8_0));
      filter_val_0 = vaddq_s16(filter_val_0, filter_offset_vec);
      int16x8_t filter_val_1 = vreinterpretq_s16_u16(vmovl_u8(filter_val_u8_1));
      filter_val_1 = vaddq_s16(filter_val_1, filter_offset_vec);
      int16x8_t filter_val_2 = vreinterpretq_s16_u16(vmovl_u8(filter_val_u8_2));
      filter_val_2 = vaddq_s16(filter_val_2, filter_offset_vec);
      int16x8_t filter_val_3 = vreinterpretq_s16_u16(vmovl_u8(filter_val_u8_3));
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
    // Narrow values down to 8 bit unsigned, saturating.
    uint8x8_t res8 = vqmovun_s16(vcombine_s16(res16, res16));
    // Apply the clamping from the activation function
    res8 = vmax_u8(res8, vdup_n_u8(output_activation_min));
    res8 = vmin_u8(res8, vdup_n_u8(output_activation_max));
    // Store results to destination.
    vst1_lane_u8(output_data + out + 0, res8, 0);
    vst1_lane_u8(output_data + out + 1, res8, 1);
    vst1_lane_u8(output_data + out + 2, res8, 2);
    vst1_lane_u8(output_data + out + 3, res8, 3);
  }
}

struct LegacyFullyConnectedAsGEMVWorkerTask : public gemmlowp::Task {
  LegacyFullyConnectedAsGEMVWorkerTask(
      const RuntimeShape& input_shape, const uint8* input_data,
      int32 input_offset, const RuntimeShape& filter_shape,
      const uint8* filter_data, int32 filter_offset,
      const RuntimeShape& bias_shape, const int32* bias_data,
      int32 output_offset, int32 output_multiplier, int output_shift,
      int32 output_activation_min, int32 output_activation_max,
      const RuntimeShape& output_shape, uint8* output_data, int row_start,
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
    LegacyFullyConnectedAsGEMVWorkerImpl(
        input_shape_, input_data_, input_offset_, filter_shape_, filter_data_,
        filter_offset_, bias_shape_, bias_data_, output_offset_,
        output_multiplier_, output_shift_, output_activation_min_,
        output_activation_max_, output_shape_, output_data_, row_start_,
        row_end_);
  }

  const RuntimeShape& input_shape_;
  const uint8* input_data_;
  int32 input_offset_;
  const RuntimeShape& filter_shape_;
  const uint8* filter_data_;
  int32 filter_offset_;
  const RuntimeShape& bias_shape_;
  const int32* bias_data_;
  int32 output_offset_;
  int32 output_multiplier_;
  int output_shift_;
  int32 output_activation_min_;
  int32 output_activation_max_;
  const RuntimeShape& output_shape_;
  uint8* output_data_;
  int row_start_;
  int row_end_;
};

inline void FullyConnectedAsGEMV(
    const RuntimeShape& input_shape, const uint8* input_data,
    int32 input_offset, const RuntimeShape& filter_shape,
    const uint8* filter_data, int32 filter_offset,
    const RuntimeShape& bias_shape, const int32* bias_data, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, const RuntimeShape& output_shape,
    uint8* output_data, gemmlowp::GemmContext* gemmlowp_context) {
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  const int input_size = FlatSizeSkipDim(input_shape, 0);
  static constexpr int kKernelRows = 4;
  const int thread_count = gemmlowp::HowManyThreads<kKernelRows>(
      gemmlowp_context->max_num_threads(), output_rows, batches, input_size);
  if (thread_count == 1) {
    // Single-thread case: do the computation on the current thread, don't
    // use a threadpool
    LegacyFullyConnectedAsGEMVWorkerImpl(
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
    tasks[i] = new LegacyFullyConnectedAsGEMVWorkerTask(
        input_shape, input_data, input_offset, filter_shape, filter_data,
        filter_offset, bias_shape, bias_data, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_shape, output_data, row_start, row_end);
    row_start = row_end;
  }
  TFLITE_DCHECK_EQ(row_start, output_rows);
  gemmlowp_context->workers_pool()->LegacyExecuteAndDestroyTasks(tasks);
}
#endif  // USE_NEON

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, gemmlowp::GemmContext* gemmlowp_context) {
  ruy::profiler::ScopeLabel label("FullyConnected/8bit");
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
#ifdef USE_NEON
  if (batches == 1) {
    const int output_size = MatchingDim(filter_shape, filter_dim_count - 2,
                                        output_shape, output_dim_count - 1);
    if (output_size >= 4) {
      return FullyConnectedAsGEMV(
          input_shape, input_data, input_offset, filter_shape, filter_data,
          filter_offset, bias_shape, bias_data, output_offset,
          output_multiplier, output_shift, output_activation_min,
          output_activation_max, output_shape, output_data, gemmlowp_context);
    }
  }
#endif  // USE_NEON
  const int filter_rows = filter_shape.Dims(filter_dim_count - 2);
  const int filter_cols = filter_shape.Dims(filter_dim_count - 1);
  TFLITE_DCHECK_EQ(filter_shape.FlatSize(), filter_rows * filter_cols);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);

  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, output_rows, filter_cols, filter_cols);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, filter_cols, batches, filter_cols);
  gemmlowp::MatrixMap<uint8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, batches, output_rows);
  const auto& output_pipeline = GemmlowpOutputPipeline::MakeExp(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);
  gemmlowp::GemmWithOutputPipeline<uint8, uint8,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemmlowp_context, filter_matrix, input_matrix, &output_matrix,
      filter_offset, input_offset, output_pipeline);
}

#ifdef GEMMLOWP_NEON
// In the common case of batch size 1, a fully-connected node degenerates
// to a matrix*vector product. LSTM cells contain a fully-connected node;
// when quantized, this becomes a special type of GEMV operation where
// the output is 16bit-quantized, thus needs its own special path.
inline void GEMVForLstmCell(const RuntimeShape& input_shape,
                            const uint8* input_data,
                            const RuntimeShape& weights_shape,
                            const uint8* weights_data, uint8 weights_zero_point,
                            const RuntimeShape& bias_shape,
                            const int32* bias_data, int32 accum_multiplier,
                            int accum_shift, const RuntimeShape& output_shape,
                            int16* output_data) {
  ruy::profiler::ScopeLabel label("GEMVForLstmCell");
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  const int output_dim_count = output_shape.DimensionsCount();
  const int weights_dim_count = weights_shape.DimensionsCount();
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(output_shape, output_dim_count - 1), 1);
  const int input_size = FlatSizeSkipDim(input_shape, 0);
  const int output_size = MatchingDim(weights_shape, weights_dim_count - 2,
                                      output_shape, output_dim_count - 1);
  // This special fast path for quantized LSTM cells does not try to support
  // odd sizes that we haven't encountered in any LSTM cell, that would
  // require special code (that would go untested until any LSTM cell
  // exercises it). We just guard our assumptions about size evenness with
  // the following assertions.
  TFLITE_DCHECK(!(output_size % 4));
  TFLITE_DCHECK(!(input_size % 8));
  const int32* bias_ptr = bias_data;
  int16* output_ptr = output_data;
  for (int out = 0; out < output_size; out += 4) {
    int32x4_t acc_0 = vdupq_n_s32(0);
    int32x4_t acc_1 = vdupq_n_s32(0);
    int32x4_t acc_2 = vdupq_n_s32(0);
    int32x4_t acc_3 = vdupq_n_s32(0);
    const int16x8_t input_offset_vec = vdupq_n_s16(-128);
    const int16x8_t weights_offset_vec = vdupq_n_s16(-weights_zero_point);
    int in = 0;
    // Handle 16 levels of depth at a time.
    for (; in <= input_size - 16; in += 16) {
      const uint8x16_t input_val_u8 = vld1q_u8(input_data + in);
      const uint8* weights_ptr = weights_data + in + out * input_size;
      uint8x16_t weights_val_u8_0 = vld1q_u8(weights_ptr + 0 * input_size);
      uint8x16_t weights_val_u8_1 = vld1q_u8(weights_ptr + 1 * input_size);
      uint8x16_t weights_val_u8_2 = vld1q_u8(weights_ptr + 2 * input_size);
      uint8x16_t weights_val_u8_3 = vld1q_u8(weights_ptr + 3 * input_size);
      int16x8_t input_val_0, input_val_1;
      const uint8x8_t low = vget_low_u8(input_val_u8);
      const uint8x8_t high = vget_high_u8(input_val_u8);
      input_val_0 = vreinterpretq_s16_u16(vmovl_u8(low));
      input_val_1 = vreinterpretq_s16_u16(vmovl_u8(high));
      input_val_0 = vaddq_s16(input_val_0, input_offset_vec);
      input_val_1 = vaddq_s16(input_val_1, input_offset_vec);
      int16x8_t weights_val_0_0, weights_val_1_0, weights_val_2_0,
          weights_val_3_0;
      int16x8_t weights_val_0_1, weights_val_1_1, weights_val_2_1,
          weights_val_3_1;
      weights_val_0_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_0))),
          weights_offset_vec);
      weights_val_0_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_0))),
          weights_offset_vec);
      weights_val_1_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_1))),
          weights_offset_vec);
      weights_val_1_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_1))),
          weights_offset_vec);
      weights_val_2_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_2))),
          weights_offset_vec);
      weights_val_2_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_2))),
          weights_offset_vec);
      weights_val_3_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_3))),
          weights_offset_vec);
      weights_val_3_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_3))),
          weights_offset_vec);
      acc_0 = vmlal_s16(acc_0, vget_low_s16(weights_val_0_0),
                        vget_low_s16(input_val_0));
      acc_1 = vmlal_s16(acc_1, vget_low_s16(weights_val_1_0),
                        vget_low_s16(input_val_0));
      acc_2 = vmlal_s16(acc_2, vget_low_s16(weights_val_2_0),
                        vget_low_s16(input_val_0));
      acc_3 = vmlal_s16(acc_3, vget_low_s16(weights_val_3_0),
                        vget_low_s16(input_val_0));
      acc_0 = vmlal_s16(acc_0, vget_high_s16(weights_val_0_0),
                        vget_high_s16(input_val_0));
      acc_1 = vmlal_s16(acc_1, vget_high_s16(weights_val_1_0),
                        vget_high_s16(input_val_0));
      acc_2 = vmlal_s16(acc_2, vget_high_s16(weights_val_2_0),
                        vget_high_s16(input_val_0));
      acc_3 = vmlal_s16(acc_3, vget_high_s16(weights_val_3_0),
                        vget_high_s16(input_val_0));
      acc_0 = vmlal_s16(acc_0, vget_low_s16(weights_val_0_1),
                        vget_low_s16(input_val_1));
      acc_1 = vmlal_s16(acc_1, vget_low_s16(weights_val_1_1),
                        vget_low_s16(input_val_1));
      acc_2 = vmlal_s16(acc_2, vget_low_s16(weights_val_2_1),
                        vget_low_s16(input_val_1));
      acc_3 = vmlal_s16(acc_3, vget_low_s16(weights_val_3_1),
                        vget_low_s16(input_val_1));
      acc_0 = vmlal_s16(acc_0, vget_high_s16(weights_val_0_1),
                        vget_high_s16(input_val_1));
      acc_1 = vmlal_s16(acc_1, vget_high_s16(weights_val_1_1),
                        vget_high_s16(input_val_1));
      acc_2 = vmlal_s16(acc_2, vget_high_s16(weights_val_2_1),
                        vget_high_s16(input_val_1));
      acc_3 = vmlal_s16(acc_3, vget_high_s16(weights_val_3_1),
                        vget_high_s16(input_val_1));
    }
    // Handle 8 levels of depth at a time.
    for (; in < input_size; in += 8) {
      const uint8x8_t input_val_u8 = vld1_u8(input_data + in);
      const uint8* weights_ptr = weights_data + in + out * input_size;
      uint8x8_t weights_val_u8_0 = vld1_u8(weights_ptr + 0 * input_size);
      uint8x8_t weights_val_u8_1 = vld1_u8(weights_ptr + 1 * input_size);
      uint8x8_t weights_val_u8_2 = vld1_u8(weights_ptr + 2 * input_size);
      uint8x8_t weights_val_u8_3 = vld1_u8(weights_ptr + 3 * input_size);
      int16x8_t input_val;
      input_val = vreinterpretq_s16_u16(vmovl_u8(input_val_u8));
      input_val = vaddq_s16(input_val, input_offset_vec);
      int16x8_t weights_val_0, weights_val_1, weights_val_2, weights_val_3;
      weights_val_0 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_0)),
                    weights_offset_vec);
      weights_val_1 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_1)),
                    weights_offset_vec);
      weights_val_2 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_2)),
                    weights_offset_vec);
      weights_val_3 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_3)),
                    weights_offset_vec);
      acc_0 = vmlal_s16(acc_0, vget_low_s16(weights_val_0),
                        vget_low_s16(input_val));
      acc_1 = vmlal_s16(acc_1, vget_low_s16(weights_val_1),
                        vget_low_s16(input_val));
      acc_2 = vmlal_s16(acc_2, vget_low_s16(weights_val_2),
                        vget_low_s16(input_val));
      acc_3 = vmlal_s16(acc_3, vget_low_s16(weights_val_3),
                        vget_low_s16(input_val));
      acc_0 = vmlal_s16(acc_0, vget_high_s16(weights_val_0),
                        vget_high_s16(input_val));
      acc_1 = vmlal_s16(acc_1, vget_high_s16(weights_val_1),
                        vget_high_s16(input_val));
      acc_2 = vmlal_s16(acc_2, vget_high_s16(weights_val_2),
                        vget_high_s16(input_val));
      acc_3 = vmlal_s16(acc_3, vget_high_s16(weights_val_3),
                        vget_high_s16(input_val));
    }
    // Horizontally reduce accumulators
    int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,
        pairwise_reduced_acc_2, pairwise_reduced_acc_3;
    pairwise_reduced_acc_0 =
        vpadd_s32(vget_low_s32(acc_0), vget_high_s32(acc_0));
    pairwise_reduced_acc_1 =
        vpadd_s32(vget_low_s32(acc_1), vget_high_s32(acc_1));
    pairwise_reduced_acc_2 =
        vpadd_s32(vget_low_s32(acc_2), vget_high_s32(acc_2));
    pairwise_reduced_acc_3 =
        vpadd_s32(vget_low_s32(acc_3), vget_high_s32(acc_3));
    const int32x2_t reduced_lo =
        vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
    const int32x2_t reduced_hi =
        vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
    // Add bias values.
    int32x4_t bias_vec = vld1q_s32(bias_ptr);
    bias_ptr += 4;
    reduced = vaddq_s32(reduced, bias_vec);
    int left_shift = accum_shift > 0 ? accum_shift : 0;
    int right_shift = accum_shift > 0 ? 0 : -accum_shift;
    reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));
    // Multiply by the fixed-point multiplier.
    reduced = vqrdmulhq_n_s32(reduced, accum_multiplier);
    // Rounding-shift-right.
    using gemmlowp::RoundingDivideByPOT;
    reduced = RoundingDivideByPOT(reduced, right_shift);
    // Narrow values down to 16 bit signed.
    const int16x4_t res16 = vqmovn_s32(reduced);
    vst1_s16(output_ptr, res16);
    output_ptr += 4;
  }
}
#endif

#ifdef GEMMLOWP_NEON
inline void GEMVForLstmCellWithSymmetricRange(
    const RuntimeShape& input_shape, const uint8* input_data,
    const RuntimeShape& weights_shape, const uint8* weights_data,
    const RuntimeShape& bias_shape, const int32* bias_data,
    int32 accum_multiplier, int accum_shift, const RuntimeShape& output_shape,
    int16* output_data) {
  ruy::profiler::ScopeLabel label("GEMVForLstmCellWithSymmetricRange");
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  const int output_dim_count = output_shape.DimensionsCount();
  const int weights_dim_count = weights_shape.DimensionsCount();
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(output_shape, output_dim_count - 1), 1);
  const int input_size = FlatSizeSkipDim(input_shape, 0);
  const int output_size = MatchingDim(weights_shape, weights_dim_count - 2,
                                      output_shape, output_dim_count - 1);
  // This special fast path for quantized LSTM cells does not try to support
  // odd sizes that we haven't encountered in any LSTM cell, that would
  // require special code (that would go untested until any LSTM cell
  // exercises it). We just guard our assumptions about size evenness with
  // the following assertions.
  TFLITE_DCHECK(!(output_size % 4));
  TFLITE_DCHECK(!(input_size % 64));
  const int32* bias_ptr = bias_data;
  int16* output_ptr = output_data;
  const uint8x16_t signbit = vdupq_n_u8(0x80);
  for (int in = 0; in < input_size; in += 32) {
    optimized_ops_preload_l1_keep(input_data + in);
  }
  const int left_shift = accum_shift > 0 ? accum_shift : 0;
  const int right_shift = accum_shift > 0 ? 0 : -accum_shift;
  for (int out = 0; out < output_size; out += 4) {
    // Load the bias values
    int32x4_t bias_vec = vld1q_s32(bias_ptr);
    bias_ptr += 4;

    // Clear accumulators. We use 2 accumulator registers per row,
    // for 4 rows. row_accumRN is the N-th accumulator for row R.
    int32x4_t row_accum00 = vdupq_n_s32(0);
    int32x4_t row_accum01 = vdupq_n_s32(0);
    int32x4_t row_accum10 = vdupq_n_s32(0);
    int32x4_t row_accum11 = vdupq_n_s32(0);
    int32x4_t row_accum20 = vdupq_n_s32(0);
    int32x4_t row_accum21 = vdupq_n_s32(0);
    int32x4_t row_accum30 = vdupq_n_s32(0);
    int32x4_t row_accum31 = vdupq_n_s32(0);

    // kReadAhead parametrizes how far ahead we prefetch weights into L1 cache.
    const int kReadAhead = 512;
    // Prefetch the first weights values.
    for (int k = 0; k < kReadAhead; k += 64) {
      optimized_ops_preload_l1_stream(weights_data + (out + 0) * input_size +
                                      k);
      optimized_ops_preload_l1_stream(weights_data + (out + 1) * input_size +
                                      k);
      optimized_ops_preload_l1_stream(weights_data + (out + 2) * input_size +
                                      k);
      optimized_ops_preload_l1_stream(weights_data + (out + 3) * input_size +
                                      k);
    }
    // Loop along the rows, handling 64 bytes per iteration because that's
    // cache line size on most current ARM-architecture CPUs.
    for (int in = 0; in < input_size; in += 64) {
      // Prefetch some future weights values.
      optimized_ops_preload_l1_stream(weights_data + (out + 0) * input_size +
                                      in + kReadAhead);
      optimized_ops_preload_l1_stream(weights_data + (out + 1) * input_size +
                                      in + kReadAhead);
      optimized_ops_preload_l1_stream(weights_data + (out + 2) * input_size +
                                      in + kReadAhead);
      optimized_ops_preload_l1_stream(weights_data + (out + 3) * input_size +
                                      in + kReadAhead);

      // We will use 2 local 16-bit accumulators per row, for 2 rows.
      // See below (*) for the rationale of processing only 2 rows at a time.
      // local_accumRN is the N-th local accumulator for row R.
      int16x8_t local_accum00;
      int16x8_t local_accum01;
      int16x8_t local_accum10;
      int16x8_t local_accum11;

      // Load 64 bytes of input activations values. Convert to signed int8
      // by flipping the sign bit (i.e. subtracting 128, the required
      // zero_point value).
      int8x16_t input0 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 0)));
      int8x16_t input1 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 1)));
      int8x16_t input2 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 2)));
      int8x16_t input3 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 3)));

      // Beginning of the core accumulation. Notice how while we have 4
      // rows to process, this code is taking care of only 2 rows at a time,
      // thus being divided into two parts looking similar ("Rows 0 and 1" and
      // "Rows 2 and 3").
      //
      // (*) The rationale for handling only 2 rows at a time is to avoid
      // cache aliasing issues on 4-way set-associative L1-cache CPUs, such
      // as Cortex-A53. With sufficiently large, power-of-two matrix dimensions,
      // we may find ourselves in a situation where rows alias each other in
      // the L1 cache, and moreover may also mutually alias with the input
      // activations. If we try to load 4 rows at a time, together with the
      // input activations, that may be 5 mutually-aliasing vectors, resulting
      // in constant mutual eviction from L1 cache. Handling 2 rows at a time
      // here largely mitigates these issues, and seems at least to be very
      // effective on Cortex-A53:
      //                          Before       After
      // big (Cortex-A73)         2.85 ms      2.85 ms
      // little (Cortex-A53)      11.0 ms      5.16 ms

      // Rows 0 and 1:
      // Load 64 bytes of weights values from each row. Convert to signed int8
      // by flipping the sign bit (i.e. subtracting 128, the required
      // zero_point value).
      int8x16_t weights00 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 0)));
      int8x16_t weights01 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 1)));
      int8x16_t weights02 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 2)));
      int8x16_t weights03 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 3)));
      int8x16_t weights10 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 0)));
      int8x16_t weights11 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 1)));
      int8x16_t weights12 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 2)));
      int8x16_t weights13 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 3)));
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights00), vget_low_s8(input0));
      local_accum01 = vmull_s8(vget_low_s8(weights01), vget_low_s8(input1));
      local_accum10 = vmull_s8(vget_low_s8(weights10), vget_low_s8(input0));
      local_accum11 = vmull_s8(vget_low_s8(weights11), vget_low_s8(input1));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights00),
                               vget_high_s8(input0));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights01),
                               vget_high_s8(input1));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights10),
                               vget_high_s8(input0));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights11),
                               vget_high_s8(input1));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum00 = vpadalq_s16(row_accum00, local_accum00);
      row_accum01 = vpadalq_s16(row_accum01, local_accum01);
      row_accum10 = vpadalq_s16(row_accum10, local_accum10);
      row_accum11 = vpadalq_s16(row_accum11, local_accum11);
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights02), vget_low_s8(input2));
      local_accum01 = vmull_s8(vget_low_s8(weights03), vget_low_s8(input3));
      local_accum10 = vmull_s8(vget_low_s8(weights12), vget_low_s8(input2));
      local_accum11 = vmull_s8(vget_low_s8(weights13), vget_low_s8(input3));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights02),
                               vget_high_s8(input2));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights03),
                               vget_high_s8(input3));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights12),
                               vget_high_s8(input2));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights13),
                               vget_high_s8(input3));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum00 = vpadalq_s16(row_accum00, local_accum00);
      row_accum01 = vpadalq_s16(row_accum01, local_accum01);
      row_accum10 = vpadalq_s16(row_accum10, local_accum10);
      row_accum11 = vpadalq_s16(row_accum11, local_accum11);

      // Rows 2 and 3:
      // Load 64 bytes of weights values from each row. Convert to signed int8
      // by flipping the sign bit (i.e. subtracting 128, the required
      // zero_point value).
      weights00 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 0)));
      weights01 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 1)));
      weights02 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 2)));
      weights03 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 3)));
      weights10 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 0)));
      weights11 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 1)));
      weights12 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 2)));
      weights13 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 3)));
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights00), vget_low_s8(input0));
      local_accum01 = vmull_s8(vget_low_s8(weights01), vget_low_s8(input1));
      local_accum10 = vmull_s8(vget_low_s8(weights10), vget_low_s8(input0));
      local_accum11 = vmull_s8(vget_low_s8(weights11), vget_low_s8(input1));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights00),
                               vget_high_s8(input0));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights01),
                               vget_high_s8(input1));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights10),
                               vget_high_s8(input0));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights11),
                               vget_high_s8(input1));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum20 = vpadalq_s16(row_accum20, local_accum00);
      row_accum21 = vpadalq_s16(row_accum21, local_accum01);
      row_accum30 = vpadalq_s16(row_accum30, local_accum10);
      row_accum31 = vpadalq_s16(row_accum31, local_accum11);
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights02), vget_low_s8(input2));
      local_accum01 = vmull_s8(vget_low_s8(weights03), vget_low_s8(input3));
      local_accum10 = vmull_s8(vget_low_s8(weights12), vget_low_s8(input2));
      local_accum11 = vmull_s8(vget_low_s8(weights13), vget_low_s8(input3));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights02),
                               vget_high_s8(input2));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights03),
                               vget_high_s8(input3));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights12),
                               vget_high_s8(input2));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights13),
                               vget_high_s8(input3));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum20 = vpadalq_s16(row_accum20, local_accum00);
      row_accum21 = vpadalq_s16(row_accum21, local_accum01);
      row_accum30 = vpadalq_s16(row_accum30, local_accum10);
      row_accum31 = vpadalq_s16(row_accum31, local_accum11);
    }

    row_accum00 = vaddq_s32(row_accum00, row_accum01);
    row_accum10 = vaddq_s32(row_accum10, row_accum11);
    row_accum20 = vaddq_s32(row_accum20, row_accum21);
    row_accum30 = vaddq_s32(row_accum30, row_accum31);
    // Horizontally reduce accumulators
    int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,
        pairwise_reduced_acc_2, pairwise_reduced_acc_3;
    pairwise_reduced_acc_0 =
        vpadd_s32(vget_low_s32(row_accum00), vget_high_s32(row_accum00));
    pairwise_reduced_acc_1 =
        vpadd_s32(vget_low_s32(row_accum10), vget_high_s32(row_accum10));
    pairwise_reduced_acc_2 =
        vpadd_s32(vget_low_s32(row_accum20), vget_high_s32(row_accum20));
    pairwise_reduced_acc_3 =
        vpadd_s32(vget_low_s32(row_accum30), vget_high_s32(row_accum30));
    const int32x2_t reduced_lo =
        vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
    const int32x2_t reduced_hi =
        vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
    // Add bias values.
    reduced = vaddq_s32(reduced, bias_vec);
    reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));
    // Multiply by the fixed-point multiplier.
    reduced = vqrdmulhq_n_s32(reduced, accum_multiplier);
    // Rounding-shift-right.
    using gemmlowp::RoundingDivideByPOT;
    reduced = RoundingDivideByPOT(reduced, right_shift);
    // Narrow values down to 16 bit signed.
    const int16x4_t res16 = vqmovn_s32(reduced);
    vst1_s16(output_ptr, res16);
    output_ptr += 4;
  }
}
#endif

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data_int32, const RuntimeShape& output_shape,
    int16* output_data, gemmlowp::GemmContext* gemmlowp_context) {
  ruy::profiler::ScopeLabel label("FullyConnected/Uint8Int16");
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.
  (void)gemmlowp_context;  // only used in properly optimized code.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(output_offset, 0);
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
  const int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  // Implementation of the fully connected node suited to the inside of an LSTM
  // cell. The operands are 8-bit integers, the accumulators are internally
  // 32bit integers, and the output is 16-bit fixed-point with 3 integer bits so
  // the output range is [-2^3, 2^3] == [-8, 8]. The rationale for that
  // is explained in the function comment above.
#ifdef GEMMLOWP_NEON
  if (batches == 1 && input_offset == -128 && output_activation_min == -32768 &&
      output_activation_max == 32767) {
    if (filter_offset == -128 && !(output_depth % 4) && !(accum_depth % 64)) {
      GEMVForLstmCellWithSymmetricRange(
          input_shape, input_data, filter_shape, filter_data, bias_shape,
          bias_data_int32, output_multiplier, output_shift, output_shape,
          output_data);
      return;
    }
    if (!(output_depth % 4) && !(accum_depth % 8)) {
      GEMVForLstmCell(input_shape, input_data, filter_shape, filter_data,
                      filter_offset, bias_shape, bias_data_int32,
                      output_multiplier, output_shift, output_shape,
                      output_data);
      return;
    }
  }
#endif
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> weights_matrix(
      filter_data, output_depth, accum_depth);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, accum_depth, batches);
  gemmlowp::MatrixMap<int16, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_depth, batches);
  typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
      ColVectorMap;
  ColVectorMap bias_vector(bias_data_int32, output_depth);
  gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
  bias_addition_stage.bias_vector = bias_vector;
  gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent scale_stage;
  scale_stage.result_offset_after_shift = 0;
  scale_stage.result_fixedpoint_multiplier = output_multiplier;
  // Note that this shift is negated wrt ordinary FC.
  scale_stage.result_exponent = output_shift;
  gemmlowp::OutputStageClamp clamp_stage;
  clamp_stage.min = output_activation_min;
  clamp_stage.max = output_activation_max;
  gemmlowp::OutputStageSaturatingCastToInt16 saturating_cast_int16_stage;
  auto output_pipeline =
      std::make_tuple(bias_addition_stage, scale_stage, clamp_stage,
                      saturating_cast_int16_stage);
  gemmlowp::GemmWithOutputPipeline<uint8, int16,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemmlowp_context, weights_matrix, input_matrix, &output_matrix,
      filter_offset, input_offset, output_pipeline);
}

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, uint8* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemmlowp_context) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data, DimsToShape(output_dims), output_data,
                 gemmlowp_context);
}

inline void FullyConnected(
    const uint8* input_data, const Dims<4>& input_dims, int32 input_offset,
    const uint8* filter_data, const Dims<4>& filter_dims, int32 filter_offset,
    const int32* bias_data_int32, const Dims<4>& bias_dims, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, int16* output_data, const Dims<4>& output_dims,
    gemmlowp::GemmContext* gemmlowp_context) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data_int32, DimsToShape(output_dims), output_data,
                 gemmlowp_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_offset, const uint8* filter_data,
                    const Dims<4>& filter_dims, int32 filter_offset,
                    const int32* bias_data, const Dims<4>& bias_dims,
                    int32 output_offset, int32 output_multiplier,
                    int output_shift, int32 output_activation_min,
                    int32 output_activation_max, uint8* output_data,
                    const Dims<4>& output_dims,
                    gemmlowp::GemmContext* gemmlowp_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  FullyConnected(input_data, input_dims, input_offset, filter_data, filter_dims,
                 filter_offset, bias_data, bias_dims, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data, output_dims,
                 gemmlowp_context);
}

#ifdef USE_NEON
inline void LegacyInt8FullyConnectedAsGEMVWorkerImpl(
    const RuntimeShape& input_shape, const int8_t* input_data,
    int32 input_offset, const RuntimeShape& filter_shape,
    const int8_t* filter_data, int32 filter_offset,
    const RuntimeShape& bias_shape, const int32* bias_data, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, const RuntimeShape& output_shape,
    int8_t* output_data, int row_start, int row_end) {
  ruy::profiler::ScopeLabel label("FullyConnectedAsGEMVInt8/8bit");
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  const int output_dim_count = output_shape.DimensionsCount();
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(output_shape, output_dim_count - 1), 1);
  const int input_size = FlatSizeSkipDim(input_shape, 0);
  static constexpr int kPeel = 4;
  const bool shift_left = (output_shift > 0);
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
      filter_ptr += input_size;
      int8x16_t filter_val_s8_1 = vld1q_s8(filter_ptr);
      filter_ptr += input_size;
      int8x16_t filter_val_s8_2 = vld1q_s8(filter_ptr);
      filter_ptr += input_size;
      int8x16_t filter_val_s8_3 = vld1q_s8(filter_ptr);
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

struct LegacyInt8FullyConnectedAsGEMVWorkerTask : public gemmlowp::Task {
  LegacyInt8FullyConnectedAsGEMVWorkerTask(
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
    LegacyInt8FullyConnectedAsGEMVWorkerImpl(
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
  int row_start_;
  int row_end_;
};

inline void LegacyInt8FullyConnectedAsGEMV(
    const RuntimeShape& input_shape, const int8_t* input_data,
    int32 input_offset, const RuntimeShape& filter_shape,
    const int8_t* filter_data, int32 filter_offset,
    const RuntimeShape& bias_shape, const int32* bias_data, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, const RuntimeShape& output_shape,
    int8_t* output_data, gemmlowp::GemmContext* gemmlowp_context) {
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  const int input_size = FlatSizeSkipDim(input_shape, 0);
  static constexpr int kKernelRows = 4;
  const int thread_count = gemmlowp::HowManyThreads<kKernelRows>(
      gemmlowp_context->max_num_threads(), output_rows, batches, input_size);
  if (thread_count == 1) {
    // Single-thread case: do the computation on the current thread, don't
    // use a threadpool
    LegacyInt8FullyConnectedAsGEMVWorkerImpl(
        input_shape, input_data, input_offset, filter_shape, filter_data,
        filter_offset, bias_shape, bias_data, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max,
        output_shape, output_data, 0, output_rows);
    return;
  }

  // Multi-threaded case: use the gemmlowp context's threadpool.
  TFLITE_DCHECK_GT(thread_count, 1);
  std::vector<LegacyInt8FullyConnectedAsGEMVWorkerTask> tasks;
  // TODO(b/131746020) don't create new heap allocations every time.
  // At least we make it a single heap allocation by using reserve().
  tasks.reserve(thread_count);
  const int kRowsPerWorker = gemmlowp::RoundUp<kKernelRows>(
      gemmlowp::CeilQuotient(output_rows, thread_count));
  int row_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    int row_end = std::min(output_rows, row_start + kRowsPerWorker);
    tasks.emplace_back(input_shape, input_data, input_offset, filter_shape,
                       filter_data, filter_offset, bias_shape, bias_data,
                       output_offset, output_multiplier, output_shift,
                       output_activation_min, output_activation_max,
                       output_shape, output_data, row_start, row_end);
    row_start = row_end;
  }
  TFLITE_DCHECK_EQ(row_start, output_rows);
  gemmlowp_context->workers_pool()->Execute(tasks.size(), tasks.data());
}
#endif  // USE_NEON

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape, int8* output_data,
    gemmlowp::GemmContext* gemmlowp_context) {
  ruy::profiler::ScopeLabel label("FullyConnectedInt8/8bit");

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
      return LegacyInt8FullyConnectedAsGEMV(
          input_shape, input_data, input_offset, filter_shape, filter_data,
          filter_offset, bias_shape, bias_data, output_offset,
          output_multiplier, output_shift, output_activation_min,
          output_activation_max, output_shape, output_data, gemmlowp_context);
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
  const auto& output_pipeline = GemmlowpOutputPipelineInt8::MakeExp(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);

  gemmlowp::GemmWithOutputPipeline<
      int8, int8, gemmlowp::SignedL8R8WithLhsNonzeroBitDepthParams>(
      gemmlowp_context, filter_matrix, input_matrix, &output_matrix,
      filter_offset, input_offset, output_pipeline);
  return;
#endif  // GEMMLOWP_NEON

  // If both GEMMLOWP_NEON && NEON paths are skipped, fallback to reference
  // implementation.
  reference_integer_ops::FullyConnected(params, input_shape, input_data,
                                        filter_shape, filter_data, bias_shape,
                                        bias_data, output_shape, output_data);
}

struct LegacyShuffledFullyConnectedWorkerTask : gemmlowp::Task {
  LegacyShuffledFullyConnectedWorkerTask(const uint8* input_data,
                                         const int8* shuffled_weights_data,
                                         int batches, int output_depth,
                                         int output_stride, int accum_depth,
                                         const int32* bias_data,
                                         int32 output_multiplier,
                                         int output_shift, int16* output_data)
      : input_data_(input_data),
        shuffled_weights_data_(shuffled_weights_data),
        batches_(batches),
        output_depth_(output_depth),
        output_stride_(output_stride),
        accum_depth_(accum_depth),
        bias_data_(bias_data),
        output_multiplier_(output_multiplier),
        output_shift_(output_shift),
        output_data_(output_data) {}

  void Run() override {
    ShuffledFullyConnectedWorkerImpl(
        input_data_, shuffled_weights_data_, batches_, output_depth_,
        output_stride_, accum_depth_, bias_data_, output_multiplier_,
        output_shift_, output_data_);
  }

  const uint8* input_data_;
  const int8* shuffled_weights_data_;
  int batches_;
  int output_depth_;
  int output_stride_;
  int accum_depth_;
  const int32* bias_data_;
  int32 output_multiplier_;
  int output_shift_;
  int16* output_data_;
};

inline void ShuffledFullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& weights_shape,
    const uint8* shuffled_weights_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int16* output_data, uint8* shuffled_input_workspace_data,
    gemmlowp::GemmContext* gemmlowp_context) {
  ruy::profiler::ScopeLabel label("ShuffledFullyConnected/8bit");
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  (void)gemmlowp_context;  // only used in optimized code.
  TFLITE_DCHECK_EQ(output_activation_min, -32768);
  TFLITE_DCHECK_EQ(output_activation_max, 32767);
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dim_count - 1);
  TFLITE_DCHECK((accum_depth % 16) == 0);
  TFLITE_DCHECK((output_depth % 4) == 0);
  // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
  // so that just reinterpreting them as int8 values is equivalent to
  // subtracting 128 from them, thus implementing for free the subtraction of
  // the zero_point value 128.
  const int8* int8_shuffled_weights_data =
      reinterpret_cast<const int8*>(shuffled_weights_data);

  // Shuffling and xoring of input activations into the workspace buffer
  if (batches == 1) {
#ifdef USE_NEON
    const uint8x16_t signbit = vdupq_n_u8(0x80);
    for (int i = 0; i < accum_depth; i += 16) {
      uint8x16_t val = vld1q_u8(input_data + i);
      val = veorq_u8(val, signbit);
      vst1q_u8(shuffled_input_workspace_data + i, val);
    }
#else
    for (int i = 0; i < accum_depth; i++) {
      shuffled_input_workspace_data[i] = input_data[i] ^ 0x80;
    }
#endif
  } else if (batches == 4) {
    uint8* shuffled_input_workspace_ptr = shuffled_input_workspace_data;
    int c = 0;
#ifdef USE_NEON
    const uint8x16_t signbit = vdupq_n_u8(0x80);
    for (c = 0; c < accum_depth; c += 16) {
      const uint8* src_data_ptr = input_data + c;
      uint8x16_t val0 = vld1q_u8(src_data_ptr + 0 * accum_depth);
      uint8x16_t val1 = vld1q_u8(src_data_ptr + 1 * accum_depth);
      uint8x16_t val2 = vld1q_u8(src_data_ptr + 2 * accum_depth);
      uint8x16_t val3 = vld1q_u8(src_data_ptr + 3 * accum_depth);
      val0 = veorq_u8(val0, signbit);
      val1 = veorq_u8(val1, signbit);
      val2 = veorq_u8(val2, signbit);
      val3 = veorq_u8(val3, signbit);
      vst1q_u8(shuffled_input_workspace_ptr + 0, val0);
      vst1q_u8(shuffled_input_workspace_ptr + 16, val1);
      vst1q_u8(shuffled_input_workspace_ptr + 32, val2);
      vst1q_u8(shuffled_input_workspace_ptr + 48, val3);
      shuffled_input_workspace_ptr += 64;
    }
#else
    for (c = 0; c < accum_depth; c += 16) {
      for (int b = 0; b < 4; b++) {
        const uint8* src_data_ptr = input_data + b * accum_depth + c;
        for (int j = 0; j < 16; j++) {
          uint8 src_val = *src_data_ptr++;
          // Flip the sign bit, so that the kernel will only need to
          // reinterpret these uint8 values as int8, getting for free the
          // subtraction of the zero_point value 128.
          uint8 dst_val = src_val ^ 0x80;
          *shuffled_input_workspace_ptr++ = dst_val;
        }
      }
    }
#endif
  } else {
    TFLITE_DCHECK(false);
    return;
  }

  static constexpr int kKernelRows = 4;
  const int thread_count = gemmlowp::HowManyThreads<kKernelRows>(
      gemmlowp_context->max_num_threads(), output_depth, batches, accum_depth);
  if (thread_count == 1) {
    // Single-thread case: do the computation on the current thread, don't
    // use a threadpool
    ShuffledFullyConnectedWorkerImpl(
        shuffled_input_workspace_data, int8_shuffled_weights_data, batches,
        output_depth, output_depth, accum_depth, bias_data, output_multiplier,
        output_shift, output_data);
    return;
  }

  // Multi-threaded case: use the gemmlowp context's threadpool.
  TFLITE_DCHECK_GT(thread_count, 1);
  std::vector<gemmlowp::Task*> tasks(thread_count);
  const int kRowsPerWorker = gemmlowp::RoundUp<kKernelRows>(
      gemmlowp::CeilQuotient(output_depth, thread_count));
  int row_start = 0;
  for (int i = 0; i < thread_count; i++) {
    int row_end = std::min(output_depth, row_start + kRowsPerWorker);
    tasks[i] = new LegacyShuffledFullyConnectedWorkerTask(
        shuffled_input_workspace_data,
        int8_shuffled_weights_data + row_start * accum_depth, batches,
        row_end - row_start, output_depth, accum_depth, bias_data + row_start,
        output_multiplier, output_shift, output_data + row_start);
    row_start = row_end;
  }
  TFLITE_DCHECK_EQ(row_start, output_depth);
  gemmlowp_context->workers_pool()->LegacyExecuteAndDestroyTasks(tasks);
}

inline void ShuffledFullyConnected(
    const uint8* input_data, const Dims<4>& input_dims,
    const uint8* shuffled_weights_data, const Dims<4>& weights_dims,
    const int32* bias_data, const Dims<4>& bias_dims, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    int16* output_data, const Dims<4>& output_dims,
    uint8* shuffled_input_workspace_data,
    gemmlowp::GemmContext* gemmlowp_context) {
  tflite::FullyConnectedParams op_params;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  ShuffledFullyConnected(op_params, DimsToShape(input_dims), input_data,
                         DimsToShape(weights_dims), shuffled_weights_data,
                         DimsToShape(bias_dims), bias_data,
                         DimsToShape(output_dims), output_data,
                         shuffled_input_workspace_data, gemmlowp_context);
}

template <typename T>
inline void ExtractPatchIntoBufferColumn(
    const Dims<4>& input_dims, int w, int h, int b, int kheight, int kwidth,
    int stride_width, int stride_height, int pad_width, int pad_height,
    int in_width, int in_height, int in_depth, int single_buffer_length,
    int buffer_id, const T* in_data, T* conv_buffer_data, uint8 zero_byte) {
  ExtractPatchIntoBufferColumn(
      DimsToShape(input_dims), w, h, b, kheight, kwidth, stride_width,
      stride_height, pad_width, pad_height, in_width, in_height, in_depth,
      single_buffer_length, buffer_id, in_data, conv_buffer_data, zero_byte);
}

template <typename T>
void DilatedIm2col(const T* input_data, const Dims<4>& input_dims,
                   const Dims<4>& filter_dims, int stride_width,
                   int stride_height, int dilation_width_factor,
                   int dilation_height_factor, int pad_width, int pad_height,
                   const Dims<4>& output_dims, uint8 zero_byte,
                   T* im2col_data) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;

  DilatedIm2col(op_params, zero_byte, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), DimsToShape(output_dims),
                im2col_data);
}

template <typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride_width,
            int stride_height, int pad_width, int pad_height, int kheight,
            int kwidth, uint8 zero_byte, T* output_data,
            const Dims<4>& output_dims) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;

  Im2col(op_params, kheight, kwidth, zero_byte, DimsToShape(input_dims),
         input_data, DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int kheight, int kwidth,
            uint8 zero_byte, T* output_data, const Dims<4>& output_dims) {
  Im2col(input_data, input_dims, stride, stride, pad_width, pad_height, kheight,
         kwidth, zero_byte, output_data, output_dims);
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;
  (void)im2col_shape;
  ruy::profiler::ScopeLabel label("Conv");

  // NB: the float 0.0f value is represented by all zero bytes.
  const uint8 float_zero_byte = 0x00;
  const float* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_dilated_im2col) {
    DilatedIm2col(params, float_zero_byte, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    Im2col(params, filter_height, filter_width, float_zero_byte, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    // TODO(aselle): We need to make sure to not send im2col if it is not
    // needed.
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  // The following code computes matrix multiplication c = a * transponse(b)
  // with CBLAS, where:
  // * `a` is a matrix with dimensions (m, k).
  // * `b` is a matrix with dimensions (n, k), so transpose(b) is (k, n).
  // * `c` is a matrix with dimensions (m, n).
  // The naming of variables are aligned with CBLAS specification here.
  const float* a = gemm_input_data;
  const float* b = filter_data;
  float* c = output_data;
  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(3);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);

#if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
  // The stride of matrix a, b and c respectively.
  int stride_a = k;
  int stride_b = k;
  int stride_c = n;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a,
              stride_a, b, stride_b, 0.0f, c, stride_c);
#else
  // When an optimized CBLAS implementation is not available, fall back
  // to using Eigen.
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Matrix;
  typedef Eigen::Map<Matrix> MatrixRef;
  typedef Eigen::Map<const Matrix> ConstMatrixRef;

  MatrixRef matrix_c(c, m, n);
  ConstMatrixRef matrix_a(a, m, k);
  ConstMatrixRef matrix_b(b, n, k);

  // The following special casing for when a or b is a vector is required
  // as Eigen seem to fail to make this optimization on its own.
  if (n == 1) {
    ruy::profiler::ScopeLabel label("GEMV");
    matrix_c.col(0).noalias() = matrix_a * matrix_b.row(0).transpose();
  } else if (m == 1) {
    ruy::profiler::ScopeLabel label("GEMV");
    matrix_c.row(0).noalias() = matrix_a.row(0) * matrix_b.transpose();
  } else {
    ruy::profiler::ScopeLabel label("GEMM");
    matrix_c.noalias() = matrix_a * matrix_b.transpose();
  }

#endif  //  defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)

  optimized_ops::AddBiasAndEvalActivationFunction(
      output_activation_min, output_activation_max, bias_shape, bias_data,
      output_shape, output_data);
}

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  Conv(op_params, DimsToShape(input_dims), input_data, DimsToShape(filter_dims),
       filter_data, DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
       output_data, DimsToShape(im2col_dims), im2col_data);
}

inline void HybridConv(const int8_t* input_data, const Dims<4>& input_dims,
                       const int8_t* filter_data, const Dims<4>& filter_dims,
                       const float* bias_data, const Dims<4>& bias_dims,
                       int stride_width, int stride_height, int pad_width,
                       int pad_height, float* scaling_factors_ptr,
                       float output_activation_min, float output_activation_max,
                       int32_t* scratch_data, const Dims<4>& scratch_dims,
                       float* output_data, const Dims<4>& output_dims,
                       int8_t* im2col_data, const Dims<4>& im2col_dims,
                       CpuBackendContext* context) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  HybridConv(op_params, scaling_factors_ptr, DimsToShape(input_dims),
             input_data, DimsToShape(filter_dims), filter_data,
             DimsToShape(bias_dims), bias_data, DimsToShape(scratch_dims),
             scratch_data, DimsToShape(output_dims), output_data,
             DimsToShape(im2col_dims), im2col_data, context);
}

template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int dilation_width_factor,
          int dilation_height_factor, int pad_width, int pad_height,
          float* output_data, const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, dilation_width_factor,
       dilation_height_factor, pad_width, pad_height, output_activation_min,
       output_activation_max, output_data, output_dims, im2col_data,
       im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, 1, 1, pad_width, pad_height,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  Conv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
           bias_dims, stride, stride, 1, 1, pad_width, pad_height, output_data,
           output_dims, im2col_data, im2col_dims);
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8* input_data, const RuntimeShape& filter_shape,
                 const uint8* filter_data, const RuntimeShape& bias_shape,
                 const int32* bias_data, const RuntimeShape& output_shape,
                 uint8* output_data, const RuntimeShape& im2col_shape,
                 uint8* im2col_data, gemmlowp::GemmContext* gemmlowp_context) {
  ruy::profiler::ScopeLabel label("Conv/8bit");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const uint8* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_dilated_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = -input_offset;
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    DilatedIm2col(params, input_zero_point, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = -input_offset;
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    Im2col(params, filter_height, filter_width, input_zero_point, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int gemm_input_rows = gemm_input_shape->Dims(3);
  // Using FlatSizeSkipDim causes segfault in some contexts (see b/79927784).
  // The root cause has not yet been identified though. Same applies below for
  // the other calls commented out. This is a partial rollback of cl/196819423.
  // const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
  const int gemm_input_cols = gemm_input_shape->Dims(0) *
                              gemm_input_shape->Dims(1) *
                              gemm_input_shape->Dims(2);
  const int filter_rows = filter_shape.Dims(0);
  // See b/79927784.
  // const int filter_cols = FlatSizeSkipDim(filter_shape, 0);
  const int filter_cols =
      filter_shape.Dims(1) * filter_shape.Dims(2) * filter_shape.Dims(3);
  const int output_rows = output_shape.Dims(3);
  // See b/79927784.
  // const int output_cols = FlatSizeSkipDim(output_shape, 3);
  const int output_cols =
      output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, gemm_input_cols);
  TFLITE_DCHECK_EQ(filter_cols, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);

#ifdef USE_NEON
  if (gemm_input_cols == 1 && output_rows >= 4) {
    RuntimeShape fc_filter_shape{
        filter_shape.Dims(0),
        filter_shape.Dims(filter_shape.DimensionsCount() - 1)};

    return FullyConnectedAsGEMV(
        *gemm_input_shape, gemm_input_data, input_offset, fc_filter_shape,
        filter_data, filter_offset, bias_shape, bias_data, output_offset,
        output_multiplier, output_shift, output_activation_min,
        output_activation_max, output_shape, output_data, gemmlowp_context);
  }
#endif

  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, filter_rows, filter_cols);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      gemm_input_data, gemm_input_rows, gemm_input_cols);
  gemmlowp::MatrixMap<uint8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, output_cols);
  const auto& output_pipeline = GemmlowpOutputPipeline::MakeExp(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);
  gemmlowp::GemmWithOutputPipeline<uint8, uint8,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemmlowp_context, filter_matrix, input_matrix, &output_matrix,
      filter_offset, input_offset, output_pipeline);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 int32 output_offset, int32 output_multiplier, int output_shift,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims,
                 uint8* im2col_data, const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemmlowp_context) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Conv(op_params, DimsToShape(input_dims), input_data, DimsToShape(filter_dims),
       filter_data, DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
       output_data, DimsToShape(im2col_dims), im2col_data, gemmlowp_context);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemmlowp_context) {
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height, 1, 1,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemmlowp_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemmlowp_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemmlowp_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const uint8* input_data, const Dims<4>& input_dims,
          int32 input_offset, const uint8* filter_data,
          const Dims<4>& filter_dims, int32 filter_offset,
          const int32* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, int32 output_offset,
          int32 output_multiplier, int output_shift,
          int32 output_activation_min, int32 output_activation_max,
          uint8* output_data, const Dims<4>& output_dims, uint8* im2col_data,
          const Dims<4>& im2col_dims, gemmlowp::GemmContext* gemmlowp_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride, stride, pad_width,
       pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemmlowp_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int kheight, int kwidth,
            uint8 zero_byte, T* output_data, const Dims<4>& output_dims) {
  Im2col(input_data, input_dims, stride, stride, pad_width, pad_height, kheight,
         kwidth, zero_byte, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void ConvAsGemm(const float* input_data, const Dims<4>& input_dims,
                const float* filter_data, const Dims<4>& filter_dims,
                const float* bias_data, const Dims<4>& bias_dims,
                float* output_data, const Dims<4>& output_dims) {
  ruy::profiler::ScopeLabel label("ConvAsGemm");

  const auto input_matrix_map =
      MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  const auto filter_matrix_map =
      MapAsMatrixWithLastDimAsCols(filter_data, filter_dims);
  auto output_matrix_map =
      MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), input_matrix_map, &output_matrix_map);

  AddBiasAndEvalActivationFunction<Ac>(bias_data, bias_dims, output_data,
                                       output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void ConvAsGemm(const uint8* input_data, const Dims<4>& input_dims,
                int32 input_offset, const uint8* filter_data,
                const Dims<4>& filter_dims, int32 filter_offset,
                const int32* bias_data, const Dims<4>& bias_dims,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims,
                gemmlowp::GemmContext* gemmlowp_context) {
  ruy::profiler::ScopeLabel label("ConvAsGemm/8bit");
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  const int input_rows = input_dims.sizes[0];
  const int input_cols = FlatSizeSkipDim(input_dims, 0);
  const int filter_rows = filter_dims.sizes[3];
  const int filter_cols = FlatSizeSkipDim(filter_dims, 3);
  const int output_rows = output_dims.sizes[0];
  const int output_cols = FlatSizeSkipDim(output_dims, 0);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, input_cols);
  TFLITE_DCHECK_EQ(filter_cols, input_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[0], output_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[1], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[2], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[3], 1);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, output_rows, filter_cols, filter_cols);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, filter_cols, output_cols, filter_cols);
  gemmlowp::MatrixMap<uint8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, output_cols, output_rows);
  const auto& output_pipeline = GemmlowpOutputPipeline::MakeExp(
      bias_data, output_rows, output_offset, output_multiplier, -output_shift,
      output_activation_min, output_activation_max);
  gemmlowp::GemmWithOutputPipeline<uint8, uint8,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemmlowp_context, filter_matrix, input_matrix, &output_matrix,
      filter_offset, input_offset, output_pipeline);
}

inline void TransposeConv(
    const ConvParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& output_shape,
    float* output_data, const RuntimeShape& im2col_shape, float* im2col_data) {
  ruy::profiler::ScopeLabel label("TransposeConv");
  // Note we could use transposed weights with forward conv for unstrided
  // cases. But we are already getting good performance with this code as-is.
  TFLITE_DCHECK(im2col_data);
  TransposeIm2col(params, 0, input_shape, input_data, filter_shape,
                  output_shape, im2col_data);

  const auto im2col_matrix_map =
      MapAsMatrixWithLastDimAsRows(im2col_data, im2col_shape);
  const auto filter_matrix_map =
      MapAsMatrixWithFirstDimAsCols(filter_data, filter_shape);
  auto output_matrix_map =
      MapAsMatrixWithLastDimAsRows(output_data, output_shape);

  Gemm(filter_matrix_map.transpose(), im2col_matrix_map, &output_matrix_map);
}

inline void TransposeConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, float* output_data,
                          const Dims<4>& output_dims, float* im2col_data,
                          const Dims<4>& im2col_dims) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;

  TransposeConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data, DimsToShape(output_dims),
                output_data, DimsToShape(im2col_dims), im2col_data);
}

template <typename T>
void TransposeIm2col(const T* input_data, const Dims<4>& input_dims,
                     const Dims<4>& filter_dims, int stride_width,
                     int stride_height, int pad_width, int pad_height,
                     const Dims<4>& output_dims, uint8 zero_byte,
                     T* im2col_data) {
  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;

  TransposeIm2col(op_params, zero_byte, DimsToShape(input_dims), input_data,
                  DimsToShape(filter_dims), DimsToShape(output_dims),
                  im2col_data);
}

inline void LstmCell(
    const LstmCellParams& params, const RuntimeShape& unextended_input_shape,
    const float* input_data, const RuntimeShape& unextended_prev_activ_shape,
    const float* prev_activ_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& unextended_bias_shape,
    const float* bias_data, const RuntimeShape& unextended_prev_state_shape,
    const float* prev_state_data,
    const RuntimeShape& unextended_output_state_shape, float* output_state_data,
    const RuntimeShape& unextended_output_activ_shape, float* output_activ_data,
    const RuntimeShape& unextended_concat_temp_shape, float* concat_temp_data,
    const RuntimeShape& unextended_activ_temp_shape, float* activ_temp_data) {
  ruy::profiler::ScopeLabel label("LstmCell");
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_bias_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_concat_temp_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_activ_temp_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape prev_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_activ_shape);
  const RuntimeShape bias_shape =
      RuntimeShape::ExtendedShape(4, unextended_bias_shape);
  const RuntimeShape prev_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_state_shape);
  const RuntimeShape output_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_state_shape);
  const RuntimeShape output_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_activ_shape);
  const RuntimeShape concat_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_concat_temp_shape);
  const RuntimeShape activ_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_activ_temp_shape);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);

  const int weights_dim_count = weights_shape.DimensionsCount();
  MatchingDim(  // batches
      input_shape, 0, prev_activ_shape, 0, prev_state_shape, 0,
      output_state_shape, 0, output_activ_shape, 0);
  MatchingDim(  // height
      input_shape, 1, prev_activ_shape, 1, prev_state_shape, 1,
      output_state_shape, 1, output_activ_shape, 1);
  MatchingDim(  // width
      input_shape, 2, prev_activ_shape, 2, prev_state_shape, 2,
      output_state_shape, 2, output_activ_shape, 2);
  const int input_depth = input_shape.Dims(3);
  const int prev_activ_depth = prev_activ_shape.Dims(3);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_DCHECK_EQ(weights_shape.Dims(weights_dim_count - 1),
                   total_input_depth);
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(bias_shape, 3), 1);
  const int intern_activ_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, bias_shape, 3);
  TFLITE_DCHECK_EQ(weights_shape.FlatSize(),
                   intern_activ_depth * total_input_depth);
  TFLITE_DCHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingDim(prev_state_shape, 3, prev_activ_shape, 3, output_state_shape,
                  3, output_activ_shape, 3);
  TFLITE_DCHECK_EQ(output_depth, intern_activ_depth / 4);

  // Concatenate prev_activ and input data together
  std::vector<float const*> concat_input_arrays_data;
  std::vector<RuntimeShape const*> concat_input_arrays_shapes;
  concat_input_arrays_data.push_back(input_data);
  concat_input_arrays_data.push_back(prev_activ_data);
  concat_input_arrays_shapes.push_back(&input_shape);
  concat_input_arrays_shapes.push_back(&prev_activ_shape);
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 3;
  concat_params.inputs_count = concat_input_arrays_data.size();
  Concatenation(concat_params, &(concat_input_arrays_shapes[0]),
                &(concat_input_arrays_data[0]), concat_temp_shape,
                concat_temp_data);

  // Fully connected
  tflite::FullyConnectedParams fc_params;
  fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  fc_params.float_activation_max = std::numeric_limits<float>::max();
  FullyConnected(fc_params, concat_temp_shape, concat_temp_data, weights_shape,
                 weights_data, bias_shape, bias_data, activ_temp_shape,
                 activ_temp_data);

  // Map raw arrays to Eigen arrays so we can use Eigen's optimized array
  // operations.
  ArrayMap<float> activ_temp_map =
      MapAsArrayWithLastDimAsRows(activ_temp_data, activ_temp_shape);
  auto input_gate_sm = activ_temp_map.block(0 * output_depth, 0, output_depth,
                                            activ_temp_map.cols());
  auto new_input_sm = activ_temp_map.block(1 * output_depth, 0, output_depth,
                                           activ_temp_map.cols());
  auto forget_gate_sm = activ_temp_map.block(2 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  auto output_gate_sm = activ_temp_map.block(3 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  ArrayMap<const float> prev_state_map =
      MapAsArrayWithLastDimAsRows(prev_state_data, prev_state_shape);
  ArrayMap<float> output_state_map =
      MapAsArrayWithLastDimAsRows(output_state_data, output_state_shape);
  ArrayMap<float> output_activ_map =
      MapAsArrayWithLastDimAsRows(output_activ_data, output_activ_shape);

  // Combined memory state and final output calculation
  ruy::profiler::ScopeLabel label2("MemoryStateAndFinalOutput");
  output_state_map =
      input_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
          new_input_sm.tanh() +
      forget_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
          prev_state_map;
  output_activ_map =
      output_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
      output_state_map.tanh();
}

inline void LstmCell(const float* input_data, const Dims<4>& input_dims,
                     const float* prev_activ_data,
                     const Dims<4>& prev_activ_dims, const float* weights_data,
                     const Dims<4>& weights_dims, const float* bias_data,
                     const Dims<4>& bias_dims, const float* prev_state_data,
                     const Dims<4>& prev_state_dims, float* output_state_data,
                     const Dims<4>& output_state_dims, float* output_activ_data,
                     const Dims<4>& output_activ_dims, float* concat_temp_data,
                     const Dims<4>& concat_temp_dims, float* activ_temp_data,
                     const Dims<4>& activ_temp_dims) {
  tflite::LstmCellParams op_params;
  // Float LSTM cell does not need parameters to be set: leave untouched.

  LstmCell(op_params, DimsToShape(input_dims), input_data,
           DimsToShape(prev_activ_dims), prev_activ_data,
           DimsToShape(weights_dims), weights_data, DimsToShape(bias_dims),
           bias_data, DimsToShape(prev_state_dims), prev_state_data,
           DimsToShape(output_state_dims), output_state_data,
           DimsToShape(output_activ_dims), output_activ_data,
           DimsToShape(concat_temp_dims), concat_temp_data,
           DimsToShape(activ_temp_dims), activ_temp_data);
}

template <int StateIntegerBits>
inline void LstmCell(
    const LstmCellParams& params, const RuntimeShape& unextended_input_shape,
    const uint8* input_data_uint8,
    const RuntimeShape& unextended_prev_activ_shape,
    const uint8* prev_activ_data_uint8, const RuntimeShape& weights_shape,
    const uint8* weights_data_uint8, const RuntimeShape& unextended_bias_shape,
    const int32* bias_data_int32,
    const RuntimeShape& unextended_prev_state_shape,
    const int16* prev_state_data_int16,
    const RuntimeShape& unextended_output_state_shape,
    int16* output_state_data_int16,
    const RuntimeShape& unextended_output_activ_shape,
    uint8* output_activ_data_uint8,
    const RuntimeShape& unextended_concat_temp_shape,
    uint8* concat_temp_data_uint8,
    const RuntimeShape& unextended_activ_temp_shape,
    int16* activ_temp_data_int16, gemmlowp::GemmContext* gemmlowp_context) {
  ruy::profiler::ScopeLabel label(
      "LstmCell/quantized (8bit external, 16bit internal)");
  int32 weights_zero_point = params.weights_zero_point;
  int32 accum_multiplier = params.accum_multiplier;
  int accum_shift = params.accum_shift;
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_bias_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_concat_temp_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_activ_temp_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape prev_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_activ_shape);
  const RuntimeShape bias_shape =
      RuntimeShape::ExtendedShape(4, unextended_bias_shape);
  const RuntimeShape prev_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_state_shape);
  const RuntimeShape output_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_state_shape);
  const RuntimeShape output_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_activ_shape);
  const RuntimeShape concat_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_concat_temp_shape);
  const RuntimeShape activ_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_activ_temp_shape);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);

  // Gather dimensions information, and perform consistency checks.
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int outer_size = MatchingFlatSizeSkipDim(
      input_shape, 3, prev_activ_shape, prev_state_shape, output_state_shape,
      output_activ_shape);
  const int input_depth = input_shape.Dims(3);
  const int prev_activ_depth = prev_activ_shape.Dims(3);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_DCHECK_EQ(weights_shape.Dims(weights_dim_count - 1),
                   total_input_depth);
  const int intern_activ_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, bias_shape, 3);
  TFLITE_DCHECK_EQ(weights_shape.FlatSize(),
                   intern_activ_depth * total_input_depth);
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(bias_shape, 3), 1);
  TFLITE_DCHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingDim(prev_state_shape, 3, prev_activ_shape, 3, output_state_shape,
                  3, output_activ_shape, 3);
  TFLITE_DCHECK_EQ(output_depth, intern_activ_depth / 4);
  const int fc_batches = FlatSizeSkipDim(activ_temp_shape, 3);
  const int fc_output_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, activ_temp_shape, 3);
  const int fc_accum_depth = total_input_depth;
  TFLITE_DCHECK_EQ(fc_output_depth, 4 * output_depth);

  // Depth-concatenate prev_activ and input data together.
  uint8 const* concat_input_arrays_data[2] = {input_data_uint8,
                                              prev_activ_data_uint8};
  const RuntimeShape* concat_input_arrays_shapes[2] = {&input_shape,
                                                       &prev_activ_shape};
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 3;
  concat_params.inputs_count = 2;
  Concatenation(concat_params, concat_input_arrays_shapes,
                concat_input_arrays_data, concat_temp_shape,
                concat_temp_data_uint8);

  // Implementation of the fully connected node inside the LSTM cell.
  // The operands are 8-bit integers, the accumulators are internally 32bit
  // integers, and the output is 16-bit fixed-point with 3 integer bits so
  // the output range is [-2^3, 2^3] == [-8, 8]. The rationale for that
  // is explained in the function comment above.
  bool gemm_already_performed = false;
#ifdef GEMMLOWP_NEON
  if (fc_batches == 1 && !(fc_output_depth % 4) && !(fc_accum_depth % 8)) {
    GEMVForLstmCell(concat_temp_shape, concat_temp_data_uint8, weights_shape,
                    weights_data_uint8, weights_zero_point, bias_shape,
                    bias_data_int32, accum_multiplier, accum_shift,
                    activ_temp_shape, activ_temp_data_int16);
    gemm_already_performed = true;
  }
#endif
  if (!gemm_already_performed) {
    gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor>
        weights_matrix(weights_data_uint8, fc_output_depth, fc_accum_depth);
    gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
        concat_temp_data_uint8, fc_accum_depth, fc_batches);
    gemmlowp::MatrixMap<int16, gemmlowp::MapOrder::ColMajor> output_matrix(
        activ_temp_data_int16, fc_output_depth, fc_batches);
    typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
        ColVectorMap;
    ColVectorMap bias_vector(bias_data_int32, fc_output_depth);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent scale_stage;
    scale_stage.result_offset_after_shift = 0;
    scale_stage.result_fixedpoint_multiplier = accum_multiplier;
    scale_stage.result_exponent = accum_shift;
    gemmlowp::OutputStageSaturatingCastToInt16 saturating_cast_int16_stage;
    auto output_pipeline = std::make_tuple(bias_addition_stage, scale_stage,
                                           saturating_cast_int16_stage);
    gemmlowp::GemmWithOutputPipeline<
        uint8, int16, gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
        gemmlowp_context, weights_matrix, input_matrix, &output_matrix,
        -weights_zero_point, -128, output_pipeline);
  }

  // Rest of the LSTM cell: tanh and logistic math functions, and some adds
  // and muls, all done in 16-bit fixed-point.
  const int16* input_gate_input_ptr = activ_temp_data_int16;
  const int16* input_modulation_gate_input_ptr =
      activ_temp_data_int16 + output_depth;
  const int16* forget_gate_input_ptr = activ_temp_data_int16 + 2 * output_depth;
  const int16* output_gate_input_ptr = activ_temp_data_int16 + 3 * output_depth;
  const int16* prev_state_ptr = prev_state_data_int16;
  int16* output_state_data_ptr = output_state_data_int16;
  uint8* output_activ_data_ptr = output_activ_data_uint8;

  for (int b = 0; b < outer_size; ++b) {
    int c = 0;
#ifdef GEMMLOWP_NEON
    for (; c <= output_depth - 8; c += 8) {
      // Define the fixed-point data types that we will use here. All use
      // int16 as the underlying integer type i.e. all are 16-bit fixed-point.
      // They only differ by the number of integral vs. fractional bits,
      // determining the range of values that they can represent.
      //
      // F0 uses 0 integer bits, range [-1, 1].
      // This is the return type of math functions such as tanh, logistic,
      // whose range is in [-1, 1].
      using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
      // F3 uses 3 integer bits, range [-8, 8].
      // This is the range of the previous fully-connected node's output,
      // which is our input here.
      using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;
      // FS uses StateIntegerBits integer bits, range [-2^StateIntegerBits,
      // 2^StateIntegerBits]. It's used to represent the internal state, whose
      // number of integer bits is currently dictated by the model. See comment
      // on the StateIntegerBits template parameter above.
      using FS = gemmlowp::FixedPoint<int16x8_t, StateIntegerBits>;
      // Implementation of input gate, using fixed-point logistic function.
      F3 input_gate_input = F3::FromRaw(vld1q_s16(input_gate_input_ptr));
      input_gate_input_ptr += 8;
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input =
          F3::FromRaw(vld1q_s16(input_modulation_gate_input_ptr));
      input_modulation_gate_input_ptr += 8;
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(vld1q_s16(forget_gate_input_ptr));
      forget_gate_input_ptr += 8;
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(vld1q_s16(output_gate_input_ptr));
      output_gate_input_ptr += 8;
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(vld1q_s16(prev_state_ptr));
      prev_state_ptr += 8;
      FS prev_state_times_forget_state = forget_gate_output * prev_state;
      // Implementation of internal addition node, saturating.
      FS new_state = gemmlowp::SaturatingAdd(
          gemmlowp::Rescale<StateIntegerBits>(input_times_input_modulation),
          prev_state_times_forget_state);
      // Implementation of last internal Tanh node, still in fixed-point.
      // Since a Tanh fixed-point implementation is specialized for a given
      // number or integer bits, and each specialization can have a substantial
      // code size, and we already used above a Tanh on an input with 3 integer
      // bits, and per the table in the above function comment there is no
      // significant accuracy to be lost by clamping to [-8, +8] for a
      // 3-integer-bits representation, let us just do that. This helps people
      // porting this to targets where code footprint must be minimized.
      F3 new_state_f3 = gemmlowp::Rescale<3>(new_state);
      F0 output_activ_int16 = output_gate_output * gemmlowp::tanh(new_state_f3);
      // Store the new internal state back to memory, as 16-bit integers.
      // Note: here we store the original value with StateIntegerBits, not
      // the rescaled 3-integer-bits value fed to tanh.
      vst1q_s16(output_state_data_ptr, new_state.raw());
      output_state_data_ptr += 8;
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16x8_t rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int8x8_t int8_output_activ = vqmovn_s16(rescaled_output_activ);
      uint8x8_t uint8_output_activ =
          vadd_u8(vdup_n_u8(128), vreinterpret_u8_s8(int8_output_activ));
      vst1_u8(output_activ_data_ptr, uint8_output_activ);
      output_activ_data_ptr += 8;
    }
#endif
    for (; c < output_depth; ++c) {
      // Define the fixed-point data types that we will use here. All use
      // int16 as the underlying integer type i.e. all are 16-bit fixed-point.
      // They only differ by the number of integral vs. fractional bits,
      // determining the range of values that they can represent.
      //
      // F0 uses 0 integer bits, range [-1, 1].
      // This is the return type of math functions such as tanh, logistic,
      // whose range is in [-1, 1].
      using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
      // F3 uses 3 integer bits, range [-8, 8].
      // This is the range of the previous fully-connected node's output,
      // which is our input here.
      using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;
      // FS uses StateIntegerBits integer bits, range [-2^StateIntegerBits,
      // 2^StateIntegerBits]. It's used to represent the internal state, whose
      // number of integer bits is currently dictated by the model. See comment
      // on the StateIntegerBits template parameter above.
      using FS = gemmlowp::FixedPoint<std::int16_t, StateIntegerBits>;
      // Implementation of input gate, using fixed-point logistic function.
      F3 input_gate_input = F3::FromRaw(*input_gate_input_ptr++);
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input =
          F3::FromRaw(*input_modulation_gate_input_ptr++);
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(*forget_gate_input_ptr++);
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(*output_gate_input_ptr++);
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(*prev_state_ptr++);
      FS prev_state_times_forget_state = forget_gate_output * prev_state;
      // Implementation of internal addition node, saturating.
      FS new_state = gemmlowp::SaturatingAdd(
          gemmlowp::Rescale<StateIntegerBits>(input_times_input_modulation),
          prev_state_times_forget_state);
      // Implementation of last internal Tanh node, still in fixed-point.
      // Since a Tanh fixed-point implementation is specialized for a given
      // number or integer bits, and each specialization can have a substantial
      // code size, and we already used above a Tanh on an input with 3 integer
      // bits, and per the table in the above function comment there is no
      // significant accuracy to be lost by clamping to [-8, +8] for a
      // 3-integer-bits representation, let us just do that. This helps people
      // porting this to targets where code footprint must be minimized.
      F3 new_state_f3 = gemmlowp::Rescale<3>(new_state);
      F0 output_activ_int16 = output_gate_output * gemmlowp::tanh(new_state_f3);
      // Store the new internal state back to memory, as 16-bit integers.
      // Note: here we store the original value with StateIntegerBits, not
      // the rescaled 3-integer-bits value fed to tanh.
      *output_state_data_ptr++ = new_state.raw();
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16 rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int16 clamped_output_activ =
          std::max<int16>(-128, std::min<int16>(127, rescaled_output_activ));
      *output_activ_data_ptr++ = 128 + clamped_output_activ;
    }
    input_gate_input_ptr += 3 * output_depth;
    input_modulation_gate_input_ptr += 3 * output_depth;
    forget_gate_input_ptr += 3 * output_depth;
    output_gate_input_ptr += 3 * output_depth;
  }
}

template <int StateIntegerBits>
void LstmCell(const uint8* input_data_uint8, const Dims<4>& input_dims,
              const uint8* prev_activ_data_uint8,
              const Dims<4>& prev_activ_dims, const uint8* weights_data_uint8,
              const Dims<4>& weights_dims, const int32* bias_data_int32,
              const Dims<4>& bias_dims, const int16* prev_state_data_int16,
              const Dims<4>& prev_state_dims, int16* output_state_data_int16,
              const Dims<4>& output_state_dims, uint8* output_activ_data_uint8,
              const Dims<4>& output_activ_dims, uint8* concat_temp_data_uint8,
              const Dims<4>& concat_temp_dims, int16* activ_temp_data_int16,
              const Dims<4>& activ_temp_dims, int32 weights_zero_point,
              int32 accum_multiplier, int accum_shift,
              gemmlowp::GemmContext* gemmlowp_context) {
  tflite::LstmCellParams op_params;
  op_params.weights_zero_point = weights_zero_point;
  op_params.accum_multiplier = accum_multiplier;
  op_params.accum_shift = accum_shift;

  LstmCell<StateIntegerBits>(
      op_params, DimsToShape(input_dims), input_data_uint8,
      DimsToShape(prev_activ_dims), prev_activ_data_uint8,
      DimsToShape(weights_dims), weights_data_uint8, DimsToShape(bias_dims),
      bias_data_int32, DimsToShape(prev_state_dims), prev_state_data_int16,
      DimsToShape(output_state_dims), output_state_data_int16,
      DimsToShape(output_activ_dims), output_activ_data_uint8,
      DimsToShape(concat_temp_dims), concat_temp_data_uint8,
      DimsToShape(activ_temp_dims), activ_temp_data_int16, gemmlowp_context);
}

template <typename T>
void BroadcastDiv(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastDiv4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const RuntimeShape& input_shape,
                     float* output_data, const RuntimeShape& output_shape) {
  static_assert(Ac == FusedActivationFunctionType::kNone, "");
  tflite::L2NormalizationParams op_params;
  // No params need to be set for float, but reserved in signature for future
  // activations.

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

inline void L2Normalization(const uint8* input_data,
                            const RuntimeShape& input_shape,
                            int32 input_zero_point, uint8* output_data,
                            const RuntimeShape& output_shape) {
  tflite::L2NormalizationParams op_params;
  op_params.input_zero_point = input_zero_point;

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  L2Normalization<Ac>(input_data, DimsToShape(input_dims), output_data,
                      DimsToShape(output_dims));
}

inline void L2Normalization(const uint8* input_data, const Dims<4>& input_dims,
                            int32 input_zero_point, uint8* output_data,
                            const Dims<4>& output_dims) {
  L2Normalization(input_data, DimsToShape(input_dims), input_zero_point,
                  output_data, DimsToShape(output_dims));
}

inline void Relu(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  Relu(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Add(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
inline void Add(int left_shift, const uint8* input1_data,
                const Dims<4>& input1_dims, int32 input1_offset,
                int32 input1_multiplier, int input1_shift,
                const uint8* input2_data, const Dims<4>& input2_dims,
                int32 input2_offset, int32 input2_multiplier, int input2_shift,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Add(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  ruy::profiler::ScopeLabel label("Add/int32");
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);

  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = std::numeric_limits<int32>::min();
  op_params.quantized_activation_max = std::numeric_limits<int32>::max();
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  tflite::ArithmeticParams op_params;
  op_params.broadcast_category =
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.broadcast_shape[4] = y0;
  op_params.broadcast_shape[3] = y1;
  op_params.broadcast_shape[2] = y2;
  op_params.broadcast_shape[1] = y3;
  op_params.broadcast_shape[0] = y4;
  BroadcastAddFivefold(op_params, DimsToShape(input1_dims), input1_data,
                       DimsToShape(input2_dims), input2_data,
                       DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastAdd(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

template <FusedActivationFunctionType Ac>
inline void Add(const int16* input1_data, const Dims<4>& input1_dims,
                int input1_shift, const int16* input2_data,
                const Dims<4>& input2_dims, int input2_shift,
                int16 output_activation_min, int16 output_activation_max,
                int16* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, -32768);
    TFLITE_DCHECK_EQ(output_activation_max, 32767);
  }

  tflite::ArithmeticParams op_params;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Sub(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(FusedActivationFunctionType::kNone,
                      &output_activation_min, &output_activation_max);
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void Sub(const T* input1_data, const Dims<4>& input1_dims, const T* input2_data,
         const Dims<4>& input2_dims, T* output_data,
         const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(FusedActivationFunctionType::kNone,
                      &output_activation_min, &output_activation_max);
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
  op_params.input1_offset = input1_offset;
  op_params.input2_offset = input2_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  BroadcastMul(input1_data, input1_dims, input1_offset, input2_data,
               input2_dims, input2_offset, output_offset, output_multiplier,
               output_shift, output_activation_min, output_activation_max,
               output_data, output_dims);
}

inline void AveragePool(const float* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int kwidth, int kheight,
                        float output_activation_min,
                        float output_activation_max, float* output_data,
                        const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  AveragePool(params, DimsToShape(input_dims), input_data,
              DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int kwidth, int kheight, float* output_data,
                 const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, kwidth, kheight, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, float* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_data, output_dims);
}

inline void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int filter_width, int filter_height,
                        int32 output_activation_min,
                        int32 output_activation_max, uint8* output_data,
                        const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  AveragePool(params, DimsToShape(input_dims), input_data,
              DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int filter_width, int filter_height,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_activation_min,
                  output_activation_max, output_data, output_dims);
}

inline void MaxPool(const float* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int kwidth, int kheight,
                    float output_activation_min, float output_activation_max,
                    float* output_data, const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int kwidth, int kheight, float* output_data,
             const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, kwidth, kheight, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             float* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_data, output_dims);
}

inline void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int filter_width, int filter_height,
                    int32 output_activation_min, int32 output_activation_max,
                    uint8* output_data, const Dims<4>& output_dims) {
  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int filter_width, int filter_height, int32 output_activation_min,
             int32 output_activation_max, uint8* output_data,
             const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, filter_width, filter_height, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             int32 output_activation_min, int32 output_activation_max,
             uint8* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

inline void L2Pool(const float* input_data, const Dims<4>& input_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int filter_width, int filter_height,
                   float output_activation_min, float output_activation_max,
                   float* output_data, const Dims<4>& output_dims) {
  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  L2Pool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
         output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims,
            int stride_width, int stride_height, int pad_width, int pad_height,
            int filter_width, int filter_height, float* output_data,
            const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  L2Pool(input_data, input_dims, stride_width, stride_height, pad_width,
         pad_height, filter_width, filter_height, output_activation_min,
         output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int filter_width, int filter_height,
            float* output_data, const Dims<4>& output_dims) {
  L2Pool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
             filter_width, filter_height, output_data, output_dims);
}

inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const uint8* input_data,
                    const RuntimeShape& output_shape, uint8* output_data) {
  const int32 input_beta_multiplier = params.input_multiplier;
  const int32 input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  ruy::profiler::ScopeLabel label("Softmax/8bit");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int b = 0; b < outer_size; ++b) {
    const uint8* input_data_ptr = input_data + b * depth;
    uint8* output_data_ptr = output_data + b * depth;

    // Determine the largest entry in the current row
    uint8 max_in_row = 0;
    {
      int c = 0;
#ifdef USE_NEON
      uint8x16_t max16_0 = vdupq_n_u8(0);
      uint8x16_t max16_1 = vdupq_n_u8(0);
      for (; c <= depth - 32; c += 32) {
        max16_0 = vmaxq_u8(max16_0, vld1q_u8(input_data_ptr + c + 0));
        max16_1 = vmaxq_u8(max16_1, vld1q_u8(input_data_ptr + c + 16));
      }
      uint8x16_t max16 = vmaxq_u8(max16_0, max16_1);
      if (c <= depth - 16) {
        max16 = vmaxq_u8(max16, vld1q_u8(input_data_ptr + c));
        c += 16;
      }
      uint8x8_t max8 = vmax_u8(vget_low_u8(max16), vget_high_u8(max16));
      if (c <= depth - 8) {
        max8 = vmax_u8(max8, vld1_u8(input_data_ptr + c));
        c += 8;
      }
      uint8x8_t max4 = vmax_u8(max8, vext_u8(max8, max8, 4));
      uint8x8_t max2 = vmax_u8(max4, vext_u8(max4, max4, 2));
      uint8x8_t max1 = vpmax_u8(max2, max2);
      max_in_row = vget_lane_u8(max1, 0);
#endif
      for (; c < depth; ++c) {
        max_in_row = std::max(max_in_row, input_data_ptr[c]);
      }
    }

#ifdef USE_NEON
    using FixedPointAccumInt32x4 =
        gemmlowp::FixedPoint<int32x4_t, kAccumulationIntegerBits>;
    using FixedPointScaledDiffInt32x4 =
        gemmlowp::FixedPoint<int32x4_t, kScaledDiffIntegerBits>;
    using FixedPoint0Int32x4 = gemmlowp::FixedPoint<int32x4_t, 0>;
    FixedPoint0Int32x4 input_beta_multiplier_f0 =
        FixedPoint0Int32x4::FromScalarRaw(input_beta_multiplier);
    int16x8_t max_in_row_s16 = vdupq_n_s16(max_in_row);
#endif

    // Compute the sum of exponentials of the differences of entries in the
    // current row from the largest entry in the current row.
    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    {
      int c = 0;
#ifdef USE_NEON
      int32x4_t diff_min_s32 = vdupq_n_s32(diff_min);
      FixedPointAccumInt32x4 sum_of_exps_0 = FixedPointAccumInt32x4::Zero();
      FixedPointAccumInt32x4 sum_of_exps_1 = FixedPointAccumInt32x4::Zero();
      FixedPointAccumInt32x4 zeros = FixedPointAccumInt32x4::Zero();
      for (; c <= depth - 8; c += 8) {
        uint16x8_t input_u16 = vmovl_u8(vld1_u8(input_data_ptr + c));
        int16x8_t input_diff_s16 =
            vsubq_s16(vreinterpretq_s16_u16(input_u16), max_in_row_s16);
        int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
        int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
        int32x4_t mask_0 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_diff_s32_0, diff_min_s32);
        int32x4_t mask_1 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_diff_s32_1, diff_min_s32);
        FixedPointScaledDiffInt32x4 scaled_diff_0 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_0, input_beta_left_shift));
        FixedPointScaledDiffInt32x4 scaled_diff_1 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_1, input_beta_left_shift));
        FixedPointAccumInt32x4 exps_0 =
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(scaled_diff_0));
        FixedPointAccumInt32x4 exps_1 =
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(scaled_diff_1));
        FixedPointAccumInt32x4 masked_exps_0 =
            SelectUsingMask(mask_0, exps_0, zeros);
        FixedPointAccumInt32x4 masked_exps_1 =
            SelectUsingMask(mask_1, exps_1, zeros);
        sum_of_exps_0 = sum_of_exps_0 + masked_exps_0;
        sum_of_exps_1 = sum_of_exps_1 + masked_exps_1;
      }
      int32x4_t sum_of_exps_reduced_4 = (sum_of_exps_0 + sum_of_exps_1).raw();
      int32x2_t sum_of_exps_reduced_2 =
          vadd_s32(vget_low_s32(sum_of_exps_reduced_4),
                   vget_high_s32(sum_of_exps_reduced_4));
      int32x2_t sum_of_exps_reduced_1 =
          vpadd_s32(sum_of_exps_reduced_2, sum_of_exps_reduced_2);
      sum_of_exps =
          FixedPointAccum::FromRaw(vget_lane_s32(sum_of_exps_reduced_1, 0));
#endif
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);
          sum_of_exps =
              sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                exp_on_negative_values(scaled_diff_f8));
        }
      }
    }

    // Compute the fixed-point multiplier and shift that we need to apply to
    // perform a division by the above-computed sum-of-exponentials.
    int num_bits_over_unit = 0;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

    // Compute the quotients of exponentials of differences of entries in the
    // current row from the largest entry, over the previously-computed sum of
    // exponentials.
    {
      int c = 0;
#ifdef USE_NEON
      int16x8_t diff_min_s16 = vdupq_n_s16(diff_min);
      for (; c <= depth - 8; c += 8) {
        uint16x8_t input_u16 = vmovl_u8(vld1_u8(input_data_ptr + c));
        int16x8_t input_diff_s16 =
            vsubq_s16(vreinterpretq_s16_u16(input_u16), max_in_row_s16);
        int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
        int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
        uint8x8_t mask = vmovn_u16(vcgeq_s16(input_diff_s16, diff_min_s16));
        FixedPointScaledDiffInt32x4 scaled_diff_0 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_0, input_beta_left_shift));
        FixedPointScaledDiffInt32x4 scaled_diff_1 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_1, input_beta_left_shift));
        FixedPoint0Int32x4 exp_0 = exp_on_negative_values(scaled_diff_0);
        FixedPoint0Int32x4 exp_1 = exp_on_negative_values(scaled_diff_1);
        int32x4_t output_s32_0 = gemmlowp::RoundingDivideByPOT(
            vqrdmulhq_n_s32(exp_0.raw(), shifted_scale.raw()),
            num_bits_over_unit + 31 - 8);
        int32x4_t output_s32_1 = gemmlowp::RoundingDivideByPOT(
            vqrdmulhq_n_s32(exp_1.raw(), shifted_scale.raw()),
            num_bits_over_unit + 31 - 8);
        int16x8_t output_s16 =
            vcombine_s16(vqmovn_s32(output_s32_0), vqmovn_s32(output_s32_1));
        uint8x8_t output_u8 = vqmovun_s16(output_s16);
        uint8x8_t masked_output = vbsl_u8(mask, output_u8, vdup_n_u8(0));
        vst1_u8(output_data_ptr + c, masked_output);
      }
#endif
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);

          FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
          int32 unsat_output = gemmlowp::RoundingDivideByPOT(
              (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);

          output_data_ptr[c] = std::max(std::min(unsat_output, 255), 0);

        } else {
          output_data_ptr[c] = 0;
        }
      }
    }
  }
}

inline void Softmax(const float* input_data, const RuntimeShape& input_shape,
                    float beta, float* output_data,
                    const RuntimeShape& output_shape) {
  SoftmaxParams params;
  params.beta = beta;
  Softmax(params, input_shape, input_data, output_shape, output_data);
}

inline void Softmax(const float* input_data, const Dims<4>& input_dims,
                    float beta, float* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), beta, output_data,
          DimsToShape(output_dims));
}

inline void Softmax(const uint8* input_data, const RuntimeShape& input_shape,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const RuntimeShape& output_shape) {
  SoftmaxParams params;
  params.input_multiplier = input_beta_multiplier;
  params.input_left_shift = input_beta_left_shift;
  params.diff_min = diff_min;
  Softmax(params, input_shape, input_data, output_shape, output_data);
}
inline void Softmax(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), input_beta_multiplier,
          input_beta_left_shift, diff_min, output_data,
          DimsToShape(output_dims));
}

inline void LogSoftmax(const float* input_data, const RuntimeShape& input_shape,
                       float* output_data, const RuntimeShape& output_shape) {
  SoftmaxParams params;
  // No params currently used for float LogSoftmax.
  LogSoftmax(params, input_shape, input_data, output_shape, output_data);
}

inline void LogSoftmax(const float* input_data, const Dims<4>& input_dims,
                       float* output_data, const Dims<4>& output_dims) {
  LogSoftmax(input_data, DimsToShape(input_dims), output_data,
             DimsToShape(output_dims));
}

inline void LogSoftmax(const uint8* input_data, const RuntimeShape& input_shape,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const RuntimeShape& output_shape) {
  SoftmaxParams params;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  params.reverse_scaling_divisor = reverse_scaling_divisor;
  params.reverse_scaling_right_shift = reverse_scaling_right_shift;
  params.diff_min = diff_min;
  reference_ops::LogSoftmax(params, input_shape, input_data, output_shape,
                            output_data);
}

inline void LogSoftmax(const uint8* input_data, const Dims<4>& input_dims,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const Dims<4>& output_dims) {
  reference_ops::LogSoftmax(
      input_data, DimsToShape(input_dims), input_multiplier, input_left_shift,
      reverse_scaling_divisor, reverse_scaling_right_shift, diff_min,
      output_data, DimsToShape(output_dims));
}

inline void Logistic(const LogisticParams& params,
                     const RuntimeShape& input_shape, const uint8* input_data,
                     const RuntimeShape& output_shape, uint8* output_data) {
  ruy::profiler::ScopeLabel label("Logistic/Uint8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int input_left_shift = params.input_left_shift;
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
#ifdef USE_NEON
  // Handle 16 values at a time
  for (; c <= size - 16; c += 16) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    uint8x16_t input_val_u8 = vld1q_u8(input_data + c);
    int16x8_t input_val_centered_0 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));
    int16x8_t input_val_centered_1 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint16x8_t mask_rightclamp_0 =
        vcgtq_s16(input_val_centered_0, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_rightclamp_1 =
        vcgtq_s16(input_val_centered_1, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_leftclamp_0 =
        vcgeq_s16(input_val_centered_0, vdupq_n_s16(-input_range_radius));
    uint16x8_t mask_leftclamp_1 =
        vcgeq_s16(input_val_centered_1, vdupq_n_s16(-input_range_radius));
    uint8x16_t mask_rightclamp = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                                             vshrn_n_u16(mask_rightclamp_1, 8));
    uint8x16_t mask_leftclamp = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                                            vshrn_n_u16(mask_leftclamp_1, 8));

    // This performs what is expressed in the scalar code as
    // const int32 input_val_rescaled =
    //     MultiplyByQuantizedMultiplierGreaterThanOne(
    //         input_val_centered, input_multiplier, input_left_shift);
    int32x4_t input_val_rescaled_0 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_1 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_2 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_3 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    input_val_rescaled_0 =
        vqrdmulhq_n_s32(input_val_rescaled_0, input_multiplier);
    input_val_rescaled_1 =
        vqrdmulhq_n_s32(input_val_rescaled_1, input_multiplier);
    input_val_rescaled_2 =
        vqrdmulhq_n_s32(input_val_rescaled_2, input_multiplier);
    input_val_rescaled_3 =
        vqrdmulhq_n_s32(input_val_rescaled_3, input_multiplier);

    // Invoke gemmlowp::logistic on FixedPoint wrapping int32x4_t
    using FixedPoint4 = gemmlowp::FixedPoint<int32x4_t, 4>;
    using FixedPoint0 = gemmlowp::FixedPoint<int32x4_t, 0>;
    const FixedPoint4 input_val_f4_0 =
        FixedPoint4::FromRaw(input_val_rescaled_0);
    const FixedPoint4 input_val_f4_1 =
        FixedPoint4::FromRaw(input_val_rescaled_1);
    const FixedPoint4 input_val_f4_2 =
        FixedPoint4::FromRaw(input_val_rescaled_2);
    const FixedPoint4 input_val_f4_3 =
        FixedPoint4::FromRaw(input_val_rescaled_3);
    const FixedPoint0 output_val_f0_0 = gemmlowp::logistic(input_val_f4_0);
    const FixedPoint0 output_val_f0_1 = gemmlowp::logistic(input_val_f4_1);
    const FixedPoint0 output_val_f0_2 = gemmlowp::logistic(input_val_f4_2);
    const FixedPoint0 output_val_f0_3 = gemmlowp::logistic(input_val_f4_3);

    // Divide by 2^23 as in the scalar code
    using gemmlowp::RoundingDivideByPOT;
    int32x4_t output_val_s32_0 = RoundingDivideByPOT(output_val_f0_0.raw(), 23);
    int32x4_t output_val_s32_1 = RoundingDivideByPOT(output_val_f0_1.raw(), 23);
    int32x4_t output_val_s32_2 = RoundingDivideByPOT(output_val_f0_2.raw(), 23);
    int32x4_t output_val_s32_3 = RoundingDivideByPOT(output_val_f0_3.raw(), 23);

    // Cast output values to uint8, saturating
    int16x8_t output_val_s16_0 = vcombine_s16(vqmovn_s32(output_val_s32_0),
                                              vqmovn_s32(output_val_s32_1));
    int16x8_t output_val_s16_1 = vcombine_s16(vqmovn_s32(output_val_s32_2),
                                              vqmovn_s32(output_val_s32_3));
    uint8x16_t output_val_u8 = vcombine_u8(vqmovun_s16(output_val_s16_0),
                                           vqmovun_s16(output_val_s16_1));

    // Perform the bit-masking with the bit masks computed at the beginning,
    // see the comment there.
    output_val_u8 = vorrq_u8(output_val_u8, mask_rightclamp);
    output_val_u8 = vandq_u8(output_val_u8, mask_leftclamp);

    // Store back to memory
    vst1q_u8(output_data + c, output_val_u8);
  }
#endif
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 23);
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s32, 0);
      TFLITE_DCHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8>(output_val_s32);
    }
    output_data[c] = output_val;
  }
}

inline void Logistic(const uint8* input_data, const RuntimeShape& input_shape,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const RuntimeShape& output_shape) {
  LogisticParams params;
  params.input_zero_point = input_zero_point;
  params.input_range_radius = input_range_radius;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  Logistic(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
           output_data);
}

inline void Logistic(const uint8* input_data, const Dims<4>& input_dims,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const Dims<4>& output_dims) {
  Logistic(input_data, DimsToShape(input_dims), input_zero_point,
           input_range_radius, input_multiplier, input_left_shift, output_data,
           DimsToShape(output_dims));
}

inline void Logistic(const RuntimeShape& input_shape, const int16* input_data,
                     const RuntimeShape& output_shape, int16* output_data) {
  LogisticParams params;
  // No params currently needed by int16 Logistic.
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const int16* input_data, const RuntimeShape& input_shape,
                     int16* output_data, const RuntimeShape& output_shape) {
  LogisticParams params;
  // No params currently needed by int16 Logistic.
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const int16* input_data, const Dims<4>& input_dims,
                     int16* output_data, const Dims<4>& output_dims) {
  Logistic(input_data, DimsToShape(input_dims), output_data,
           DimsToShape(output_dims));
}

inline void Tanh(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  Tanh(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Tanh(const TanhParams& params, const RuntimeShape& input_shape,
                 const uint8* input_data, const RuntimeShape& output_shape,
                 uint8* output_data) {
  // Note that this is almost the exact same code as in Logistic().
  ruy::profiler::ScopeLabel label("Tanh");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int input_left_shift = params.input_left_shift;
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  int32_t output_zero_point = 128;
#ifdef USE_NEON
  // Handle 16 values at a time
  for (; c <= size - 16; c += 16) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    uint8x16_t input_val_u8 = vld1q_u8(input_data + c);
    int16x8_t input_val_centered_0 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));
    int16x8_t input_val_centered_1 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint16x8_t mask_rightclamp_0 =
        vcgtq_s16(input_val_centered_0, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_rightclamp_1 =
        vcgtq_s16(input_val_centered_1, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_leftclamp_0 =
        vcgeq_s16(input_val_centered_0, vdupq_n_s16(-input_range_radius));
    uint16x8_t mask_leftclamp_1 =
        vcgeq_s16(input_val_centered_1, vdupq_n_s16(-input_range_radius));
    uint8x16_t mask_rightclamp = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                                             vshrn_n_u16(mask_rightclamp_1, 8));
    uint8x16_t mask_leftclamp = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                                            vshrn_n_u16(mask_leftclamp_1, 8));

    // This performs what is expressed in the scalar code as
    // const int32 input_val_rescaled =
    //     MultiplyByQuantizedMultiplierGreaterThanOne(
    //         input_val_centered, input_multiplier, input_left_shift);
    int32x4_t input_val_rescaled_0 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_1 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_2 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_3 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    input_val_rescaled_0 =
        vqrdmulhq_n_s32(input_val_rescaled_0, input_multiplier);
    input_val_rescaled_1 =
        vqrdmulhq_n_s32(input_val_rescaled_1, input_multiplier);
    input_val_rescaled_2 =
        vqrdmulhq_n_s32(input_val_rescaled_2, input_multiplier);
    input_val_rescaled_3 =
        vqrdmulhq_n_s32(input_val_rescaled_3, input_multiplier);

    // Invoke gemmlowp::tanh on FixedPoint wrapping int32x4_t
    using FixedPoint4 = gemmlowp::FixedPoint<int32x4_t, 4>;
    using FixedPoint0 = gemmlowp::FixedPoint<int32x4_t, 0>;
    const FixedPoint4 input_val_f4_0 =
        FixedPoint4::FromRaw(input_val_rescaled_0);
    const FixedPoint4 input_val_f4_1 =
        FixedPoint4::FromRaw(input_val_rescaled_1);
    const FixedPoint4 input_val_f4_2 =
        FixedPoint4::FromRaw(input_val_rescaled_2);
    const FixedPoint4 input_val_f4_3 =
        FixedPoint4::FromRaw(input_val_rescaled_3);
    const FixedPoint0 output_val_f0_0 = gemmlowp::tanh(input_val_f4_0);
    const FixedPoint0 output_val_f0_1 = gemmlowp::tanh(input_val_f4_1);
    const FixedPoint0 output_val_f0_2 = gemmlowp::tanh(input_val_f4_2);
    const FixedPoint0 output_val_f0_3 = gemmlowp::tanh(input_val_f4_3);

    // Divide by 2^24 as in the scalar code
    using gemmlowp::RoundingDivideByPOT;
    int32x4_t output_val_s32_0 = RoundingDivideByPOT(output_val_f0_0.raw(), 24);
    int32x4_t output_val_s32_1 = RoundingDivideByPOT(output_val_f0_1.raw(), 24);
    int32x4_t output_val_s32_2 = RoundingDivideByPOT(output_val_f0_2.raw(), 24);
    int32x4_t output_val_s32_3 = RoundingDivideByPOT(output_val_f0_3.raw(), 24);

    // Add the output zero point
    int32x4_t output_zero_point_s32 = vdupq_n_s32(output_zero_point);
    output_val_s32_0 = vaddq_s32(output_val_s32_0, output_zero_point_s32);
    output_val_s32_1 = vaddq_s32(output_val_s32_1, output_zero_point_s32);
    output_val_s32_2 = vaddq_s32(output_val_s32_2, output_zero_point_s32);
    output_val_s32_3 = vaddq_s32(output_val_s32_3, output_zero_point_s32);

    // Cast output values to uint8, saturating
    int16x8_t output_val_s16_0 = vcombine_s16(vqmovn_s32(output_val_s32_0),
                                              vqmovn_s32(output_val_s32_1));
    int16x8_t output_val_s16_1 = vcombine_s16(vqmovn_s32(output_val_s32_2),
                                              vqmovn_s32(output_val_s32_3));
    uint8x16_t output_val_u8 = vcombine_u8(vqmovun_s16(output_val_s16_0),
                                           vqmovun_s16(output_val_s16_1));

    // Perform the bit-masking with the bit masks computed at the beginning,
    // see the comment there.
    output_val_u8 = vorrq_u8(output_val_u8, mask_rightclamp);
    output_val_u8 = vandq_u8(output_val_u8, mask_leftclamp);

    // Store back to memory
    vst1q_u8(output_data + c, output_val_u8);
  }
#endif
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 24);
      output_val_s32 += output_zero_point;
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s32, 0);
      TFLITE_DCHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8>(output_val_s32);
    }
    output_data[c] = output_val;
  }
}

inline void Tanh(const uint8* input_data, const RuntimeShape& input_shape,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const RuntimeShape& output_shape) {
  TanhParams params;
  params.input_zero_point = input_zero_point;
  params.input_range_radius = input_range_radius;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  Tanh(params, input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const Dims<4>& output_dims) {
  Tanh(input_data, DimsToShape(input_dims), input_zero_point,
       input_range_radius, input_multiplier, input_left_shift, output_data,
       DimsToShape(output_dims));
}

inline void Tanh(const int16* input_data, const RuntimeShape& input_shape,
                 int input_left_shift, int16* output_data,
                 const RuntimeShape& output_shape) {
  TanhParams params;
  params.input_left_shift = input_left_shift;
  Tanh(params, input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const int16* input_data, const Dims<4>& input_dims,
                 int input_left_shift, int16* output_data,
                 const Dims<4>& output_dims) {
  Tanh(input_data, DimsToShape(input_dims), input_left_shift, output_data,
       DimsToShape(output_dims));
}

template <typename T>
inline void DepthToSpace(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  tflite::DepthToSpaceParams op_params;
  op_params.block_size = block_size;

  DepthToSpace(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void SpaceToDepth(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  tflite::SpaceToDepthParams op_params;
  op_params.block_size = block_size;

  SpaceToDepth(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

inline void Mul(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float output_activation_min, float output_activation_max,
                float* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Mul(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  Mul(input1_data, input1_dims, input2_data, input2_dims, output_activation_min,
      output_activation_max, output_data, output_dims);
}

inline void Mul(const int32* input1_data, const Dims<4>& input1_dims,
                const int32* input2_data, const Dims<4>& input2_dims,
                int32 output_activation_min, int32 output_activation_max,
                int32* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Mul(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);
  tflite::ArithmeticParams op_params;
  // No parameters needed.

  MulNoActivation(op_params, DimsToShape(input1_dims), input1_data,
                  DimsToShape(input2_dims), input2_data,
                  DimsToShape(output_dims), output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int16* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  // No parameters needed.

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int32 output_offset, int32 output_activation_min,
                int32 output_activation_max, uint8* output_data,
                const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.output_offset = output_offset;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// For compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const float* input1_data, const Dims<4>& input1_dims,
                         const float* input2_data, const Dims<4>& input2_dims,
                         float* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  float float_activation_min;
  float float_activation_max;
  GetActivationMinMax(Ac, &float_activation_min, &float_activation_max);
  SetActivationParams(float_activation_min, float_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

inline void LocalResponseNormalization(const float* input_data,
                                       const Dims<4>& input_dims, int range,
                                       float bias, float alpha, float beta,
                                       float* output_data,
                                       const Dims<4>& output_dims) {
  tflite::LocalResponseNormalizationParams op_params;
  op_params.range = range;
  op_params.bias = bias;
  op_params.alpha = alpha;
  op_params.beta = beta;

  LocalResponseNormalization(op_params, DimsToShape(input_dims), input_data,
                             DimsToShape(output_dims), output_data);
}

template <typename SrcT, typename DstT>
void Cast(const SrcT* input_data, const Dims<4>& input_dims, DstT* output_data,
          const Dims<4>& output_dims) {
  Cast(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Floor(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  Floor(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = align_corners;
  op_params.half_pixel_centers = false;
  ResizeBilinear(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(output_size_dims), output_size_data,
                 DimsToShape(output_dims), output_data);
}

inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = align_corners;
  op_params.half_pixel_centers = false;
  ResizeBilinear(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(output_size_dims), output_size_data,
                 DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear(input_data, input_dims, output_size_data, output_size_dims,
                 output_data, output_dims, /*align_corners=*/false);
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear(input_data, input_dims, output_size_data, output_size_dims,
                 output_data, output_dims, /*align_corners=*/false);
}

template <typename T>
inline void BatchToSpaceND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* crops_data, const Dims<4>& crops_dims,
                           T* output_data, const Dims<4>& output_dims) {
  BatchToSpaceND(DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(crops_dims), crops_data, DimsToShape(output_dims),
                 output_data);
}

// Legacy signature, function covered both Pad and PadV2.
template <typename T>
inline void PadV2(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& left_paddings,
                  const std::vector<int>& right_paddings, T* output_data,
                  const Dims<4>& output_dims, const T pad_value) {
  TFLITE_DCHECK_EQ(left_paddings.size(), 4);
  TFLITE_DCHECK_EQ(right_paddings.size(), 4);
  tflite::PadParams op_params;
  op_params.left_padding_count = 4;
  op_params.right_padding_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.left_padding[i] = left_paddings[3 - i];
    op_params.right_padding[i] = right_paddings[3 - i];
  }
  const T pad_value_copy = pad_value;

  Pad(op_params, DimsToShape(input_dims), input_data, &pad_value_copy,
      DimsToShape(output_dims), output_data);
}

// Old Pad that calls legacy PadV2.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims, const int32_t pad_value) {
  const T converted_pad_value = static_cast<T>(pad_value);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, converted_pad_value);
}

// Old Pad that only padded with 0.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
  const T pad_value = static_cast<T>(0);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, pad_value);
}

template <typename T>
inline void Slice(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& begin, const std::vector<int>& size,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::SliceParams op_params;
  op_params.begin_count = 4;
  op_params.size_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.begin[i] = begin[3 - i];
    op_params.size[i] = size[3 - i];
  }

  Slice(op_params, DimsToShape(input_dims), input_data,
        DimsToShape(output_dims), output_data);
}

template <typename T>
void TensorFlowMinimum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  Minimum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

template <typename T>
void TensorFlowMaximum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  Maximum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

inline void Dequantize(const uint8* input_data, const Dims<4>& input_dims,
                       int32 zero_point, double scale, float* output_data,
                       const Dims<4>& output_dims) {
  tflite::DequantizationParams op_params;
  op_params.zero_point = zero_point;
  op_params.scale = scale;

  Dequantize(op_params, DimsToShape(input_dims), input_data,
             DimsToShape(output_dims), output_data);
}

template <typename T>
void Transpose(const T* input, const Dims<4>& input_dims, T* output,
               const Dims<4>& output_dims, const int* permuted_axes) {
  TransposeParams params;
  params.perm_count = 4;
  for (int i = 0; i < 4; ++i) {
    params.perm[i] = 3 - permuted_axes[3 - i];
  }
  Transpose(params, DimsToShape(input_dims), input, DimsToShape(output_dims),
            output);
}

}  // namespace optimized_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_
