/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/meta_support.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

#if (defined(GEMMLOWP_NEON_32) || defined(GEMMLOWP_NEON_64)) && \
    !defined(TENSORFLOW_DISABLE_META) && !defined(__APPLE__)
#define TENSORFLOW_USE_META (1)
#endif

namespace tensorflow {
namespace meta {

namespace {

int g_num_threads = 0;
bool g_enabled = true;
bool g_use_local_context = false;

#ifdef TENSORFLOW_USE_META

const int kAlignment = 32;
const int kScratchSize = 2048 * 1024 + kAlignment;

class Scratch : public ResourceBase {
 public:
  Scratch() : scratch_(new uint8_t[kScratchSize]) {
    // Make sure scratch is aligned to 32 bytes. Scratch object owns the
    // scratch buffer.
    scratch_32_aligned_ =
        scratch_.get() + kAlignment -
        (reinterpret_cast<uintptr_t>(scratch_.get()) % kAlignment);
  }

  uint8_t* buffer() { return scratch_32_aligned_; }

  string DebugString() { return "MetaGemmScratchResource"; }

 private:
  std::unique_ptr<uint8_t> scratch_;
  uint8_t* scratch_32_aligned_;
};

uint8_t* GetScratch(OpKernelContext* context) {
  Scratch* scratch = nullptr;
  std::function<Status(Scratch**)> creator = [](Scratch** resource) {
    *resource = new Scratch();
    return Status::OK();
  };
  Status s = context->resource_manager()->LookupOrCreate(
      "MetaGemm", "ScratchBuffer", &scratch, creator);
  if (!s.ok()) {
    context->CtxFailureWithWarning(s);
    return nullptr;
  }
  return scratch->buffer();
}

gemmlowp::WorkersPool* GetWorkersPool() {
  static gemmlowp::WorkersPool* pool = new gemmlowp::WorkersPool();
  return pool;
}

mutex& GetMutex() {
  static mutex mu;
  return mu;
}

int GetWorkersCount(OpKernelContext* tf_context) {
  if (g_num_threads == 0) {
    return tf_context->device()->tensorflow_cpu_worker_threads()->num_threads;
  }
  return g_num_threads;
}

typedef gemmlowp::meta::SimpleContext<gemmlowp::WorkersPool> LocalContext;

template <typename Context, typename Params>
void MultiThreadGemm(Context* context, const Params& params) {
  if (params.m <= 4) {
      gemmlowp::meta::MultiThreadGemm<
          Context, gemmlowp::meta::GemmExecutorPackLHSCacheFriendly<>, Params,
          1, 8, 8>(context, params);
  } else {
    if (params.m >= params.n) {
      gemmlowp::meta::MultiThreadGemm<
          Context, gemmlowp::meta::GemmExecutorPackRHSCacheFriendly<>, Params,
          2, 4, 8>(context, params);
    } else {
      gemmlowp::meta::MultiThreadGemm<
          Context, gemmlowp::meta::GemmExecutorPackLHSCacheFriendly<>, Params,
          2, 4, 8>(context, params);
    }
  }
}

template <typename LeftStream, typename RightStream>
void QuantizedGemmImpl(OpKernelContext* tf_context, const quint8* a_data,
                       const quint8* b_data, qint32* c_data, int m, int n,
                       int k, int offset_a, int offset_b, int lda, int ldb,
                       int ldc) {
  typedef gemmlowp::meta::GemmParams<
      uint8_t, int32_t, LeftStream, RightStream,
      gemmlowp::meta::QuantizedStaticPreprocessedAsInt32,
      gemmlowp::meta::RowMajor>
      Params;
  Params params;

  params.m = m;
  params.n = n;
  params.k = k;

  params.lhs = reinterpret_cast<const uint8_t*>(&(a_data->value));
  params.rhs = reinterpret_cast<const uint8_t*>(&(b_data->value));
  params.result = reinterpret_cast<int32_t*>(&(c_data->value));
  params.scratch = CHECK_NOTNULL(GetScratch(tf_context));

  params.left_stream.count = k;
  params.left_stream.stride = lda;
  params.left_stream.multiplicative_sum_offset = offset_b;
  params.left_stream.additive_sum_offset = k * offset_a * offset_b;

  params.right_stream.count = k;
  params.right_stream.stride = ldb;
  params.right_stream.multiplicative_sum_offset = offset_a;
  params.right_stream.additive_sum_offset = 0;

  params.fused_kernel.kernel.count = k;
  params.fused_kernel.output_stream.stride = ldc * sizeof(int32_t);

  if (g_use_local_context) {
    LocalContext local_context(GetWorkersCount(tf_context), GetWorkersPool());
    MultiThreadGemm<LocalContext, Params>(&local_context, params);
  } else {
    auto& workers = *(tf_context->device()->tensorflow_cpu_worker_threads());
    TensorflowGemmContext context(workers.num_threads, workers.workers);
    MultiThreadGemm<TensorflowGemmContext, Params>(&context, params);
  }
}

template <typename Params, int kernel_size>
void MultiThreadTransform1D(OpKernelContext* tf_context, const Params& params) {
  if (g_use_local_context) {
    LocalContext local_context(GetWorkersCount(tf_context), GetWorkersPool());
    gemmlowp::meta::MultiThreadTransform1D<LocalContext, Params, kernel_size>(
        &local_context, params);
  } else {
    auto& workers = *(tf_context->device()->tensorflow_cpu_worker_threads());
    TensorflowGemmContext context(workers.num_threads, workers.workers);
    gemmlowp::meta::MultiThreadTransform1D<TensorflowGemmContext, Params,
                                           kernel_size>(&context, params);
  }
}

template <typename QuantizedType>
double CalculateRangeScale(float min, float max) {
  const int bits = sizeof(QuantizedType) * 8;
  return static_cast<double>(max - min) /
         ((static_cast<int64_t>(1) << bits) - 1);
}

template <typename QuantizedType>
double CalculateOneOverRangeScale(float min, float max) {
  if (min == max) {
    return 0.0;
  }
  const int bits = sizeof(QuantizedType) * 8;
  return static_cast<double>((static_cast<int64_t>(1) << bits) - 1) /
         (max - min);
}

#endif  // TENSORFLOW_USE_META

}  // namespace

void SetNumThreads(int num_threads) { g_num_threads = num_threads; }

int GetNumThreads() { return g_num_threads; }

void SetUseLocalContext(bool use_local_context) {
  g_use_local_context = use_local_context;
}

bool GetUseLocalContext() { return g_use_local_context; }

bool IsSupported() {
#if defined(TENSORFLOW_USE_META)
  return true;
#else
  return false;
#endif
}

bool IsEnabled() { return g_enabled; }

void SetEnabled(bool enabled) { g_enabled = enabled; }

bool IsSupportedAndEnabled() { return IsSupported() && IsEnabled(); }

void QuantizedGemm(OpKernelContext* tf_context, bool transpose_a,
                   bool transpose_b, const quint8* a_data, const quint8* b_data,
                   qint32* c_data, int m, int n, int k, int offset_a,
                   int offset_b, int lda, int ldb, int ldc) {
#ifdef TENSORFLOW_USE_META
  mutex_lock library_lock(GetMutex());
  if (transpose_a) {
    if (transpose_b) {
      QuantizedGemmImpl<gemmlowp::meta::ColumnMajorWithSum,
                        gemmlowp::meta::RowMajorWithSum>(
          tf_context, a_data, b_data, c_data, m, n, k, offset_a, offset_b, lda,
          ldb, ldc);
    } else {
      QuantizedGemmImpl<gemmlowp::meta::ColumnMajorWithSum,
                        gemmlowp::meta::ColumnMajorWithSum>(
          tf_context, a_data, b_data, c_data, m, n, k, offset_a, offset_b, lda,
          ldb, ldc);
    }
  } else {
    if (transpose_b) {
      QuantizedGemmImpl<gemmlowp::meta::RowMajorWithSum,
                        gemmlowp::meta::RowMajorWithSum>(
          tf_context, a_data, b_data, c_data, m, n, k, offset_a, offset_b, lda,
          ldb, ldc);
    } else {
      QuantizedGemmImpl<gemmlowp::meta::RowMajorWithSum,
                        gemmlowp::meta::ColumnMajorWithSum>(
          tf_context, a_data, b_data, c_data, m, n, k, offset_a, offset_b, lda,
          ldb, ldc);
    }
  }
#else
  LOG(FATAL) << "QuantizedGemm: Meta fastpath not supported.";
#endif
}

void Requantize(OpKernelContext* tf_context, const qint32* input, int count,
                float input_min, float input_max, float output_min,
                float output_max, quint8* output) {
#ifdef TENSORFLOW_USE_META
  mutex_lock library_lock(GetMutex());
  typedef gemmlowp::meta::Transform1DParams<int32_t, uint8_t,
                                            gemmlowp::meta::Requantize>
      Params;

  Params params;
  params.input = reinterpret_cast<const int32_t*>(input);
  params.output = reinterpret_cast<uint8_t*>(output);
  params.kernel.count = count;
  params.kernel.input_range_min = input_min;
  params.kernel.output_range_min = output_min;
  params.kernel.input_range_scale =
      CalculateRangeScale<int32_t>(input_min, input_max);
  params.kernel.one_over_output_range_scale =
      CalculateOneOverRangeScale<uint8_t>(output_min, output_max);
  params.kernel.input_range_offset =
      static_cast<float>(std::numeric_limits<int32_t>::lowest());

  // After adding the output_range_offset the value is cast from float to uint.
  // The float to int/uint cast in NEON uses round toward 0. To keep the
  // rounding consistent with Eigen, which uses round toward closest, we can
  // add 0.5f and exploit the fact that we only operate on non negative values.
  // TODO(maciekc): fix the actual kernel in gemmlowp/meta
  params.kernel.output_range_offset =
      static_cast<float>(std::numeric_limits<uint8_t>::lowest()) + 0.5f;

  MultiThreadTransform1D<Params, 16>(tf_context, params);
#else
  LOG(FATAL) << "Requantize: Meta fastpath not supported.";
#endif
}

void Dequantize(OpKernelContext* tf_context, const quint8* input, int count,
                float range_min, float range_max, float* output) {
#ifdef TENSORFLOW_USE_META
  mutex_lock library_lock(GetMutex());
  typedef gemmlowp::meta::Transform1DParams<uint8_t, float,
                                            gemmlowp::meta::Dequantize>
      Params;

  Params params;
  params.input = reinterpret_cast<const uint8_t*>(input);
  params.output = reinterpret_cast<float*>(output);
  params.kernel.count = count;
  params.kernel.range_min = range_min;
  params.kernel.range_scale =
      CalculateRangeScale<uint8_t>(range_min, range_max);
  params.kernel.range_offset =
      static_cast<float>(std::numeric_limits<uint8_t>::lowest());

  MultiThreadTransform1D<Params, 16>(tf_context, params);
#else
  LOG(FATAL) << "Dequantize: Meta fastpath not supported.";
#endif
}

void Quantize(OpKernelContext* tf_context, const float* input, int count,
              float range_min, float range_max, quint8* output) {
#ifdef TENSORFLOW_USE_META
  mutex_lock library_lock(GetMutex());
  typedef gemmlowp::meta::Transform1DParams<float, uint8_t,
                                            gemmlowp::meta::Quantize>
      Params;

  Params params;
  params.input = reinterpret_cast<const float*>(input);
  params.output = reinterpret_cast<uint8_t*>(output);
  params.kernel.count = count;
  params.kernel.range_min = range_min;
  params.kernel.range_scale =
      CalculateOneOverRangeScale<uint8_t>(range_min, range_max);

  // After adding the range_offset the value is cast from float to uint.
  // The float to int/uint cast in NEON uses round toward 0. To keep the
  // rounding consistent with Eigen, which uses round toward closest, we can
  // add 0.5f and exploit the fact that we only operate on non negative values.
  // TODO(maciekc): fix the actual kernel in gemmlowp/meta
  params.kernel.range_offset =
      static_cast<float>(std::numeric_limits<uint8_t>::lowest()) + 0.5f;

  MultiThreadTransform1D<Params, 16>(tf_context, params);
#else
  LOG(FATAL) << "Quantize: Meta fastpath not supported.";
#endif
}

void QuantizedBiasAdd(OpKernelContext* tf_context, const quint8* input,
                      int input_count, const quint8* bias, int bias_count,
                      float input_min, float input_max, float bias_min,
                      float bias_max, float output_min, float output_max,
                      qint32* output) {
#ifdef TENSORFLOW_USE_META
  mutex_lock library_lock(GetMutex());
  typedef gemmlowp::meta::Transform1DParams<uint8_t, int32_t,
                                            gemmlowp::meta::BiasAdd<uint8_t>>
      Params;

  Params params;
  params.input = reinterpret_cast<const uint8_t*>(input);
  params.output = reinterpret_cast<int32_t*>(output);
  params.kernel.bias = reinterpret_cast<const uint8_t*>(bias);
  params.kernel.count = bias_count;
  params.kernel.rows = input_count / bias_count;
  params.kernel.input_range_min = input_min;
  params.kernel.bias_range_min = bias_min;
  params.kernel.input_range_scale =
      CalculateRangeScale<uint8_t>(input_min, input_max);
  params.kernel.bias_range_scale =
      CalculateRangeScale<uint8_t>(bias_min, bias_max);
  params.kernel.input_range_offset = 0;
  params.kernel.bias_range_offset = 0;
  params.kernel.output_range_min = output_min;
  params.kernel.one_over_output_range_scale =
      CalculateOneOverRangeScale<int32_t>(output_min, output_max);
  params.kernel.output_range_offset =
      static_cast<float>(std::numeric_limits<int32_t>::lowest());

  // TODO(maciekc): add multithreading to bias add.
  // Right now this kernel does not support multi threaded execution.
  gemmlowp::meta::Transform1D<Params, 16>(params);
#else
  LOG(FATAL) << "QuantizedBiasAdd: Meta fastpath not supported.";
#endif
}

void Clamp(OpKernelContext* tf_context, const quint8* input, int count,
           quint8 clamp_min, quint8 clamp_max, quint8* output) {
#ifdef TENSORFLOW_USE_META
  mutex_lock library_lock(GetMutex());
  typedef gemmlowp::meta::Transform1DParams<uint8_t, uint8_t,
                                            gemmlowp::meta::MinMax<uint8_t>>
      Params;

  Params params;
  params.input = reinterpret_cast<const uint8_t*>(input);
  params.output = reinterpret_cast<uint8_t*>(output);
  params.kernel.count = count;
  params.kernel.min = clamp_min;
  params.kernel.max = clamp_max;

  MultiThreadTransform1D<Params, 16>(tf_context, params);
#else
  LOG(FATAL) << "Clamp: Meta fastpath not supported.";
#endif
}

}  // namespace meta
}  // namespace tensorflow
