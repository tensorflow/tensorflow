/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_

#include <algorithm>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

inline void FullyConnectedSparseWeight(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("Random Sparse");
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  const int output_elements = output_shape.FlatSize();
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  const int w0_size = sparsity.dim_metadata[0].dense_size;
  const int* w1_segments = sparsity.dim_metadata[1].array_segments->data;
  const int* w1_indices = sparsity.dim_metadata[1].array_indices->data;

  for (int i = 0; i < output_elements; ++i) {
    output_data[i] = 0.f;
  }

  for (int b = 0; b < batches; ++b) {
    for (int idx_0 = 0; idx_0 < w0_size; ++idx_0) {
      for (int pw1 = w1_segments[idx_0]; pw1 < w1_segments[idx_0 + 1]; ++pw1) {
        int idx_1 = w1_indices[pw1];
        output_data[b * output_depth + idx_0] +=
            weights_data[pw1] * input_data[b * accum_depth + idx_1];
      }
    }
  }

  for (int b = 0; b < batches; ++b) {
    for (int i = 0; i < output_depth; ++i) {
      float total = output_data[b * output_depth + i];
      const float bias_value = bias_data ? bias_data[i] : 0;
      output_data[b * output_depth + i] = ActivationFunctionWithMinMax(
          total + bias_value, output_activation_min, output_activation_max);
    }
  }
}

inline void FullyConnectedSparseWeight1x16Impl(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& weights_shape, const int8_t* weights_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data, int thread_start,
    int thread_end, const CpuBackendContext& cpu_backend_context) {
  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("1x16 Block Sparse");

  const int input_dims_count = input_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = thread_end - thread_start;
  const int input_depth = MatchingDim(weights_shape, weights_dims_count - 1,
                                      input_shape, input_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int32_t output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int* w1_segments = sparsity.dim_metadata[1].array_segments->data;
  const int* w1_indices = sparsity.dim_metadata[1].array_indices->data;

  tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate1x16(
      weights_data, w1_segments, w1_indices, weights_shape.Dims(0),
      weights_shape.Dims(1), input_data + thread_start * input_depth, bias_data,
      batches, input_offset, output_multiplier, output_shift, output_offset,
      output_activation_min, output_activation_max,
      output_data + thread_start * output_depth);
}

inline void FullyConnectedSparseWeight1x4Impl(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data, int thread_start,
    int thread_end, const CpuBackendContext& cpu_backend_context) {
  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("1x4 Block Sparse");
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  const int input_dims_count = input_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = thread_end - thread_start;
  const int input_depth = MatchingDim(weights_shape, weights_dims_count - 1,
                                      input_shape, input_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int* w1_segments = sparsity.dim_metadata[1].array_segments->data;
  const int* w1_indices = sparsity.dim_metadata[1].array_indices->data;

  tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate1x4(
      weights_data, w1_segments, w1_indices, weights_shape.Dims(0),
      weights_shape.Dims(1), input_data + thread_start * input_depth, batches,
      output_data + thread_start * output_depth);

  ruy::profiler::ScopeLabel activation_label("activation function");
  for (int b = thread_start; b < thread_end; ++b) {
    for (int i = 0; i < output_depth; ++i) {
      float total = output_data[b * output_depth + i];
      const float bias_value = bias_data ? bias_data[i] : 0;
      output_data[b * output_depth + i] = ActivationFunctionWithMinMax(
          total + bias_value, output_activation_min, output_activation_max);
    }
  }
}

struct FullyConnectedSparseWeight1x4Task : cpu_backend_threadpool::Task {
  FullyConnectedSparseWeight1x4Task(
      const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
      const RuntimeShape& input_shape, const float* input_data,
      const RuntimeShape& weights_shape, const float* weights_data,
      const RuntimeShape& bias_shape, const float* bias_data,
      const RuntimeShape& output_shape, float* output_data, int thread_start,
      int thread_end, const CpuBackendContext& cpu_backend_context_x)
      : sparsity(sparsity),
        params(params),
        input_shape(input_shape),
        input_data(input_data),
        weights_shape(weights_shape),
        weights_data(weights_data),
        bias_shape(bias_shape),
        bias_data(bias_data),
        output_shape(output_shape),
        output_data(output_data),
        thread_start(thread_start),
        thread_end(thread_end),
        cpu_backend_context(cpu_backend_context_x) {}

  void Run() override {
    FullyConnectedSparseWeight1x4Impl(
        sparsity, params, input_shape, input_data, weights_shape, weights_data,
        bias_shape, bias_data, output_shape, output_data, thread_start,
        thread_end, cpu_backend_context);
  }

 private:
  const TfLiteSparsity& sparsity;
  const FullyConnectedParams& params;
  const RuntimeShape& input_shape;
  const float* input_data;
  const RuntimeShape& weights_shape;
  const float* weights_data;
  const RuntimeShape& bias_shape;
  const float* bias_data;
  const RuntimeShape& output_shape;
  float* output_data;
  int thread_start;
  int thread_end;
  const CpuBackendContext& cpu_backend_context;
};

inline void FullyConnectedSparseWeight1x16(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& weights_shape, const int8_t* weights_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    CpuBackendContext* cpu_backend_context) {
  const int output_elements = output_shape.FlatSize();
  memset(output_data, 0, output_elements * sizeof(int8_t));

  const int batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);

  // TODO(b/220851507): Add multi-thread support for quantized sparse kernel.
  return FullyConnectedSparseWeight1x16Impl(
      sparsity, params, input_shape, input_data, weights_shape, weights_data,
      bias_shape, bias_data, output_shape, output_data, 0, batches,
      *cpu_backend_context);
}

// The multi-threaded kernel slices the workload along the batch dimension. If
// there's not enough batches of data, the number of threads used is equal to
// the batch size. We can improve this later with slicing along the row
// dimension of the weight.
inline void FullyConnectedSparseWeight1x4(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    CpuBackendContext* cpu_backend_context) {
  const int output_elements = output_shape.FlatSize();
  memset(output_data, 0, output_elements * sizeof(float));

  const int max_threads = cpu_backend_context->max_num_threads();
  const int batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int thread_count = std::max(1, std::min(batches, max_threads));
  if (thread_count == 1) {
    return FullyConnectedSparseWeight1x4Impl(
        sparsity, params, input_shape, input_data, weights_shape, weights_data,
        bias_shape, bias_data, output_shape, output_data, 0, batches,
        *cpu_backend_context);
  }
  std::vector<FullyConnectedSparseWeight1x4Task> tasks;
  tasks.reserve(thread_count);
  int thread_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    // This makes sure the workload is relatively balanced when batches is not a
    // multiple of thread_count. The first mod(batches, thread_count) tasks need
    // to process one more batch than the rest.
    int thread_end = thread_start + batches / thread_count;
    if (i < batches % thread_count) thread_end++;

    tasks.emplace_back(sparsity, params, input_shape, input_data, weights_shape,
                       weights_data, bias_shape, bias_data, output_shape,
                       output_data, thread_start, thread_end,
                       *cpu_backend_context);
    thread_start = thread_end;
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
}

}  // namespace optimized_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_
