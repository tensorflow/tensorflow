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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_DEPTHWISE_CONV_HYBRID_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_DEPTHWISE_CONV_HYBRID_H_

#include <algorithm>
#include <memory>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid_3x3_filter.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {
namespace depthwise_conv {

// Initializes the accumulator buffer with zeros.
inline void DepthwiseConvInitAccBuffer(int num_output_pixels, int output_depth,
                                       int32* acc_buffer) {
  memset(acc_buffer, 0,
         sizeof(acc_buffer[0]) * output_depth * num_output_pixels);
}

// Base DWConv Implementation used with both static and dynamic
// accumulator buffers.
// Initializes the accumulator buffer with bias values.
static void DoDepthwiseConvHybridGeneral(
    const DepthwiseParams& params, const float* input_scales,
    const RuntimeShape& input_shape, const int8* input_data,
    const RuntimeShape& filter_shape, const int8* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scales, const int32_t* input_offsets,
    int thread_start, int thread_end, int thread_dim, int32* acc_buffer,
    int32 acc_buffer_size) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_rows = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  TFLITE_DCHECK_GE(acc_buffer_size, output_depth);
  const int kOutputPixelsInAccBuffer = acc_buffer_size / output_depth;
  const int kAccBufferActualSize = kOutputPixelsInAccBuffer * output_depth;
  TFLITE_DCHECK_LE(kOutputPixelsInAccBuffer * output_depth,
                   kAccBufferActualSize);
  TFLITE_DCHECK_LE(kAccBufferActualSize, acc_buffer_size);
  TFLITE_DCHECK_GE(kOutputPixelsInAccBuffer, 1);
  TFLITE_DCHECK(thread_dim == 0 || thread_dim == 1);

  // row_accum_func will point to the core accumulation function to be used
  // for this DepthwiseConvHybrid op.
  using row_accum_func_t = decltype(&QuantizedDepthwiseConvAccumRowGeneric);
  row_accum_func_t row_accum_func = nullptr;

#define TFMINI_USE_DEPTHWISECONV_KERNEL(ALLOW_STRIDED, FIXED_INPUT_DEPTH, \
                                        FIXED_DEPTH_MULTIPLIER)           \
  if (!row_accum_func && (stride_width == 1 || ALLOW_STRIDED) &&          \
      (input_depth == FIXED_INPUT_DEPTH || FIXED_INPUT_DEPTH == 0) &&     \
      depth_multiplier == FIXED_DEPTH_MULTIPLIER) {                       \
    row_accum_func =                                                      \
        QuantizedDepthwiseConvAccumRow<ALLOW_STRIDED, FIXED_INPUT_DEPTH,  \
                                       FIXED_DEPTH_MULTIPLIER>;           \
  }

#ifdef USE_NEON
  // We go over our list of kernels by decreasing order of preference
  // for the cases where multiple kernels could apply.

  // Start with the fastest kernels: AllowStrided=false, fixed input depth.

  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 1, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 2, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 4, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 1, 4)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 4, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 4, 4)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 8, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 2, 8)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 2, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(false, 12, 1)

  // Next come the strided kernels: AllowStrided=true, fixed input depth.
  // They are a bit less efficient, but allow stride!=1.

  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 8, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 16, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 16)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 20)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 32)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 1, 8)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 8, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 2, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 4, 1)

  // Finally, the kernels allowing a variable input depth,
  // these are the least efficient but most general kernels.

  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 1)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 2)
  TFMINI_USE_DEPTHWISECONV_KERNEL(true, 0, 3)
#endif  // USE_NEON

  // No matching fast kernel found, use slow fallback.
  if (!row_accum_func) {
    row_accum_func = QuantizedDepthwiseConvAccumRowGeneric;
  }

#undef TFMINI_USE_DEPTHWISECONV_KERNEL

  const int input_height_stride = input_shape.Dims(3) * input_shape.Dims(2);
  const int input_batch_stride = input_height_stride * input_shape.Dims(1);
  const int filter_height_stride = filter_shape.Dims(3) * filter_shape.Dims(2);

  // Now that we have determined row_accum_func, we can start work.
  int batch_start = 0;
  int batch_end = batches;
  int row_start = 0;
  int row_end = output_rows;
  int output_ptr_offset = 0;

  switch (thread_dim) {
    case 0:
      TFLITE_DCHECK_GE(thread_start, 0);
      TFLITE_DCHECK_LE(thread_end, batches);
      batch_start = thread_start;
      batch_end = thread_end;
      output_ptr_offset = batch_start * FlatSizeSkipDim(output_shape, 0);
      break;
    case 1:
      TFLITE_DCHECK_GE(thread_start, 0);
      TFLITE_DCHECK_LE(thread_end, output_rows);
      row_start = thread_start;
      row_end = thread_end;
      output_ptr_offset = row_start * output_width * output_depth;
      break;
  }

  float* output_ptr = output_data + output_ptr_offset;
  int batch_step =
      (output_rows + row_start - row_end) * output_width * output_depth;
  for (int b = batch_start; b < batch_end; ++b) {
    float input_scale = input_scales[b];
    int32_t input_offset = input_offsets[b];
    for (int out_y = row_start; out_y < row_end; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      const int filter_y_start =
          std::max(0, (-in_y_origin + dilation_height_factor - 1) /
                          dilation_height_factor);
      const int filter_y_end =
          std::min(filter_height,
                   (input_height - in_y_origin + dilation_height_factor - 1) /
                       dilation_height_factor);
      for (int out_x_buffer_start = 0; out_x_buffer_start < output_width;
           out_x_buffer_start += kOutputPixelsInAccBuffer) {
        const int out_x_buffer_end = std::min(
            output_width, out_x_buffer_start + kOutputPixelsInAccBuffer);
        // We call a 'pixel' a group of activation that share all but the
        // 'depth'/'channel' coordinate. num_output_pixels is the number of
        // output pixels that we will accumulate in this loop iteration.
        const int num_output_pixels = out_x_buffer_end - out_x_buffer_start;
        DepthwiseConvInitAccBuffer(num_output_pixels, output_depth, acc_buffer);

        // Accumulation loop. Most of the time should be spent in here.
        for (int filter_y = filter_y_start; filter_y < filter_y_end;
             ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          row_accum_func(
              stride_width, dilation_width_factor, input_depth, input_width,
              input_data + in_y * input_height_stride + b * input_batch_stride,
              -input_offset, pad_width, depth_multiplier, filter_width,
              filter_data + filter_y * filter_height_stride, out_x_buffer_start,
              out_x_buffer_end, output_depth, acc_buffer);
        }
        // Finished accumulating int32 values. Just store them as float values
        gemmlowp::ScopedProfilingLabel label("store");
        const int num_output_values = output_depth * num_output_pixels;
        int c = 0;
        while (c < output_depth) {
          int target_output_depth = output_depth;

#ifdef USE_NEON
          const float32x4_t output_activation_min_vec =
              vdupq_n_f32(output_activation_min);
          const float32x4_t output_activation_max_vec =
              vdupq_n_f32(output_activation_max);
          const float32x4_t input_scale_32x4 = vdupq_n_f32(input_scale);
          for (; c <= output_depth - 4; c += 4) {
            if ((c + 4) > output_depth) {
              break;
            }
            const float32x4_t channel_scale_32x4 =
                vld1q_f32(per_channel_scales + c);
            const float32x4_t bias_32x4 = vld1q_f32(bias_data + c);
            for (int n = 0; n < num_output_pixels; ++n) {
              int loc = n * output_depth + c;
              int32x4_t acc = vld1q_s32(acc_buffer + loc);
              float32x4_t float_acc = vcvtq_f32_s32(acc);
              float_acc = vmulq_f32(float_acc, channel_scale_32x4);
              float_acc = vmulq_f32(float_acc, input_scale_32x4);
              float_acc = vaddq_f32(float_acc, bias_32x4);
              float_acc = vmaxq_f32(float_acc, output_activation_min_vec);
              float_acc = vminq_f32(float_acc, output_activation_max_vec);
              vst1q_f32(output_ptr + loc, float_acc);
            }
          }
#endif  // USE_NEON

          for (; c < target_output_depth; c++) {
            for (int n = 0; n < num_output_pixels; ++n) {
              int loc = n * output_depth + c;
              int32 acc = acc_buffer[loc];
              float float_acc = acc * input_scale * per_channel_scales[c];
              float_acc += bias_data[c];
              float_acc = std::max(float_acc, output_activation_min);
              float_acc = std::min(float_acc, output_activation_max);
              output_ptr[loc] = float_acc;
            }
          }
        }
        output_ptr += num_output_values;
      }
    }
    output_ptr += batch_step;
  }
}

// Utilize the base implementation of DWConv with a stack allocated accumulator
// buffer. The static allocation limits the number of depthwise channels that
// can be processed to kStaticAccBufferMaxSize.
static void DoDepthwiseConvHybridGeneralStatic(
    const DepthwiseParams& params, const float* input_scales,
    const RuntimeShape& input_shape, const int8* input_data,
    const RuntimeShape& filter_shape, const int8* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scales, const int32_t* input_offsets,
    int thread_start, int thread_end, int thread_dim) {
  static const int kStaticAccBufferMaxSize = 2048;
  int32 stack_acc_buffer[kStaticAccBufferMaxSize];
  DoDepthwiseConvHybridGeneral(
      params, input_scales, input_shape, input_data, filter_shape, filter_data,
      bias_shape, bias_data, output_shape, output_data, per_channel_scales,
      input_offsets, thread_start, thread_end, thread_dim, stack_acc_buffer,
      kStaticAccBufferMaxSize);
}

// This DWConv function uses static memory for accumulation by default for upto
// kStaticAccBufferMaxSize channels. Beyound that, a dynamic buffer is used on
// a per call basis. The function errors out if number of channels is larger
// than kStaticAccBufferMaxSize and TF_LITE_STATIC_MEMORY is defined.
inline void DepthwiseConvHybridGeneral(
    const DepthwiseParams& params, const float* input_scales,
    const RuntimeShape& input_shape, const int8* input_data,
    const RuntimeShape& filter_shape, const int8* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scales, const int32_t* input_offsets,
    int thread_start, int thread_end, int thread_dim) {
#ifndef TF_LITE_STATIC_MEMORY
  static const int kStaticAccBufferMaxSize = 2048;
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

  if (kStaticAccBufferMaxSize < output_depth) {
    std::unique_ptr<int32[]> heap_acc_buffer(new int32[output_depth]);
    DoDepthwiseConvHybridGeneral(
        params, input_scales, input_shape, input_data, filter_shape,
        filter_data, bias_shape, bias_data, output_shape, output_data,
        per_channel_scales, input_offsets, thread_start, thread_end, thread_dim,
        heap_acc_buffer.get(), output_depth);

    return;
  }
#endif

  DoDepthwiseConvHybridGeneralStatic(
      params, input_scales, input_shape, input_data, filter_shape, filter_data,
      bias_shape, bias_data, output_shape, output_data, per_channel_scales,
      input_offsets, thread_start, thread_end, thread_dim);
}

}  // namespace depthwise_conv

template <DepthwiseConvOutputRounding kOutputRounding>
inline void DepthwiseConvHybridWithRounding(
    const DepthwiseParams& params, const float* input_scales,
    const RuntimeShape& input_shape, const int8* input_data,
    const RuntimeShape& filter_shape, const int8* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scales, const int32_t* input_offsets,
    int thread_start, int thread_end, int thread_dim) {
  gemmlowp::ScopedProfilingLabel label("DepthwiseConvHybridInt8/8bit");
  const int depth_multiplier = params.depth_multiplier;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  TFLITE_DCHECK_GE(dilation_width_factor, 1);
  TFLITE_DCHECK_GE(dilation_height_factor, 1);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
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

  // Call kernel optimized for depthwise convolutions using 3x3 filters if
  // parameters are supported.
  if (optimized_ops::depthwise_conv::Fast3x3FilterKernelSupported<
      optimized_ops::depthwise_conv::QuantizationType::kNonPerChannelUint8>(
          input_shape, filter_shape, stride_width, stride_height,
          dilation_width_factor, dilation_height_factor, pad_width, pad_height,
          depth_multiplier, output_shape, 0, nullptr)) {
    gemmlowp::ScopedProfilingLabel specialized_label(
        "DepthwiseConvHybridInt8/8bit/3x3");
    optimized_ops::depthwise_conv::DepthwiseConvHybrid3x3FilterPerChannel<
        DepthwiseConvOutputRounding::kUpward>(
            params, input_scales, input_shape, input_data,
            filter_shape, filter_data, bias_shape, bias_data, output_shape,
            output_data, per_channel_scales, input_offsets,
            thread_start, thread_end, thread_dim);
    return;
  }
#endif

  gemmlowp::ScopedProfilingLabel specialized_label(
      "DepthwiseConvHybridInt8/8bit/General");
  depthwise_conv::DepthwiseConvHybridGeneral(
      params, input_scales, input_shape, input_data,
      filter_shape, filter_data, bias_shape, bias_data, output_shape,
      output_data, per_channel_scales, input_offsets,
      thread_start, thread_end, thread_dim);
}

inline void DepthwiseConvHybridImpl(
    const DepthwiseParams& params, const float* input_scales,
    const RuntimeShape& input_shape, const int8* input_data,
    const RuntimeShape& filter_shape, const int8* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scales, const int32_t* input_offsets,
    int thread_start, int thread_end, int thread_dim) {
  return DepthwiseConvHybridWithRounding<
      DepthwiseConvOutputRounding::kAwayFromZero>(
          params, input_scales, input_shape, input_data,
          filter_shape, filter_data, bias_shape, bias_data, output_shape,
          output_data, per_channel_scales, input_offsets,
          thread_start, thread_end, thread_dim);
}

template <typename T, typename TS>
struct DepthwiseConvHybridWorkerTask : cpu_backend_threadpool::Task {
  DepthwiseConvHybridWorkerTask(const DepthwiseParams& params,
                                const float* input_scales,
                                const RuntimeShape& input_shape,
                                const T* input_data,
                                const RuntimeShape& filter_shape,
                                const T* filter_data,
                                const RuntimeShape& bias_shape,
                                const TS* bias_data,
                                const RuntimeShape& output_shape,
                                float* output_data,
                                const float* per_channel_scales,
                                const int32_t* input_offsets,
                                int thread_start, int thread_end,
                                int thread_dim)
      : params(params),
        input_scales(input_scales),
        input_shape(input_shape),
        input_data(input_data),
        filter_shape(filter_shape),
        filter_data(filter_data),
        bias_shape(bias_shape),
        bias_data(bias_data),
        output_shape(output_shape),
        output_data(output_data),
        per_channel_scales(per_channel_scales),
        input_offsets(input_offsets),
        thread_start(thread_start),
        thread_end(thread_end),
        thread_dim(thread_dim) {}

  void Run() override {
    DepthwiseConvHybridImpl(params, input_scales, input_shape,
                            input_data, filter_shape, filter_data,
                            bias_shape, bias_data, output_shape,
                            output_data, per_channel_scales, input_offsets,
                            thread_start, thread_end, thread_dim);
  }

 private:
  const DepthwiseParams& params;
  const float* input_scales;
  const RuntimeShape& input_shape;
  const T* input_data;
  const RuntimeShape& filter_shape;
  const T* filter_data;
  const RuntimeShape& bias_shape;
  const TS* bias_data;
  const RuntimeShape& output_shape;
  float* output_data;
  const float* per_channel_scales;
  const int32_t* input_offsets;
  int thread_start;
  int thread_end;
  int thread_dim;
};

inline void DepthwiseConvHybridPerChannel(
    const DepthwiseParams& params, const float* input_scales,
    const RuntimeShape& input_shape, const int8* input_data,
    const RuntimeShape& filter_shape, const int8* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scales, int32_t* input_offsets,
    CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("DepthwiseConvHybridInt8");
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

  const int max_threads = cpu_backend_context->max_num_threads();
  thread_count = std::max(1, std::min(thread_count, max_threads));

  if (thread_count == 1) {
    DepthwiseConvHybridImpl(params, input_scales, input_shape,
                            input_data, filter_shape, filter_data, bias_shape,
                            bias_data, output_shape, output_data,
                            per_channel_scales, input_offsets,
                            /*thread_start=*/0, /*thread_end=*/output_rows,
                            /*thread_dim=*/1);
  } else {
    std::vector<DepthwiseConvHybridWorkerTask<int8, float>> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(thread_count);
    int thread_start = 0;
    for (int i = 0; i < thread_count; ++i) {
      int thread_end =
          thread_start + (thread_dim_size - thread_start) / (thread_count - i);
      tasks.emplace_back(params, input_scales, input_shape,
                         input_data, filter_shape, filter_data, bias_shape,
                         bias_data, output_shape, output_data,
                         per_channel_scales, input_offsets, thread_start,
                         thread_end, thread_dim);
      thread_start = thread_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_DEPTHWISE_CONV_HYBRID_H_
