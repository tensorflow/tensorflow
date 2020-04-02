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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_MULTITHREAD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_MULTITHREAD_H_

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"

namespace tflite {
namespace optimized_ops {

// TODO(luwa): add multithread to per-channel depthwise_conv
// DepthwiseConv can run with multi threads on the dim specified by thread_dim.
// Each thread processes output elements on dim, thread_dim, in the range of
// [thread_start, thread_end).
// For example, assume thread_start = 2, thread_end = 6, and thread_dim = 1, it
// means that it will calculate DepthwiseConv for output_data[:, 2:5, :, :].
template <typename T, typename TS>
struct DepthwiseConvWorkerTask : cpu_backend_threadpool::Task {
  DepthwiseConvWorkerTask(const DepthwiseParams& params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const RuntimeShape& filter_shape,
                          const T* filter_data, const RuntimeShape& bias_shape,
                          const TS* bias_data, const RuntimeShape& output_shape,
                          T* output_data, const CpuFlags& cpu_flags,
                          int thread_start, int thread_end, int thread_dim)
      : params_(params),
        input_shape_(input_shape),
        input_data_(input_data),
        filter_shape_(filter_shape),
        filter_data_(filter_data),
        bias_shape_(bias_shape),
        bias_data_(bias_data),
        output_shape_(output_shape),
        output_data_(output_data),
        cpu_flags_(cpu_flags),
        thread_start_(thread_start),
        thread_end_(thread_end),
        thread_dim_(thread_dim) {}

  void Run() override {
    DepthwiseConvImpl(params_, input_shape_, input_data_, filter_shape_,
                      filter_data_, bias_shape_, bias_data_, output_shape_,
                      output_data_, cpu_flags_, thread_start_, thread_end_,
                      thread_dim_);
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
  const CpuFlags& cpu_flags_;
  int thread_start_;
  int thread_end_;
  int thread_dim_;
};

inline int HowManyConvThreads(const RuntimeShape& output_shape,
                              const RuntimeShape& filter_shape) {
  // How many scalar multiplications are needed to make it worth using one
  // more thread
  static constexpr int kMinMulPerThread = 1 << 13;  // 8k
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int num_muls = output_shape.FlatSize() * filter_height * filter_width;
  // Try to avoid real runtime divisions if possible by dividing by a
  // compile-time constant.
  int thread_count = std::max(1, num_muls / kMinMulPerThread);
  return thread_count;
}

inline bool MultithreadAlongBatches(int thread_count, int batches) {
  TFLITE_DCHECK_GE(thread_count, 2);
  // If there are fewer batch entries than the number of threads we want to use,
  // then better do intra-batch-entry multithreading.
  if (batches < thread_count) {
    return false;
  }
  // If there are at least 2 batch entries to be handed to each thread, then
  // it's safe to proceed with batch-wise multithreading: each thread will have
  // approximately equal number of batch entries to handle, so the load
  // balancing will be reasonable, and the amount to which the load is not
  // perfectly balanced will be offset by the inherent advantages of
  // batch-wise multithreading (each thread is more efficient thanks to working
  // on larger buffers with less boundary-handling overhead).
  if (batches >= 2 * thread_count) {
    return true;
  }
  // In the limit case were there are at least 1 but not much more than 1
  // batch entries per thread, it may be a good idea to do per-batch
  // multithreading if the number of batch entries is a multiple of the number
  // of threads, so that each thread will have the same number of batch entries
  // to process.
  return ((batches % thread_count) == 0);
}

template <typename T, typename TS>
inline void DepthwiseConv(const DepthwiseParams& params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const RuntimeShape& filter_shape,
                          const T* filter_data, const RuntimeShape& bias_shape,
                          const TS* bias_data, const RuntimeShape& output_shape,
                          T* output_data,
                          CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("DepthwiseConv");

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  int thread_count = HowManyConvThreads(output_shape, filter_shape);
  const int max_threads = cpu_backend_context->max_num_threads();
  thread_count = std::max(1, std::min(thread_count, max_threads));
#ifndef TFLITE_WITH_RUY
  // Cap the number of threads to 2 for float path to avoid regression in
  // performance (b/132294857).
  if (std::is_floating_point<T>::value) {
    thread_count = std::min(thread_count, 2);
  }
#endif

  const int output_batches = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);

  CpuFlags cpu_flags;
  GetCpuFlags(cpu_backend_context, &cpu_flags);

  if (thread_count == 1) {
    DepthwiseConvImpl(params, input_shape, input_data, filter_shape,
                      filter_data, bias_shape, bias_data, output_shape,
                      output_data, cpu_flags, /*thread_start=*/0,
                      /*thread_end=*/output_height, /*thread_dim=*/1);
    return;
  }

  int thread_dim, thread_dim_size;
  if (MultithreadAlongBatches(thread_count, output_batches)) {
    thread_dim = 0;
    thread_dim_size = output_batches;
  } else {
    thread_dim = 1;
    thread_dim_size = output_height;
  }

  std::vector<DepthwiseConvWorkerTask<T, TS>> tasks;
  // TODO(b/131746020) don't create new heap allocations every time.
  // At least we make it a single heap allocation by using reserve().
  tasks.reserve(thread_count);
  int thread_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    int thread_end =
        thread_start + (thread_dim_size - thread_start) / (thread_count - i);
    tasks.emplace_back(params, input_shape, input_data, filter_shape,
                       filter_data, bias_shape, bias_data, output_shape,
                       output_data, cpu_flags, thread_start, thread_end,
                       thread_dim);
    thread_start = thread_end;
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_MULTITHREAD_H_
