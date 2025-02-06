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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_

#if (defined(__i386) || defined(_M_IX86) || defined(__x86_64__) || \
     defined(_M_X64))
#define TFLITE_X86_PLATFORM
#endif

#include <memory>

#include "public/gemmlowp.h"
#ifdef TFLITE_KERNEL_USE_XNNPACK
#include "pthreadpool.h"  // from @pthreadpool
#endif
#include "ruy/context.h"  // from @ruy
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/external_cpu_backend_context.h"

namespace tflite {

class CpuBackendContext final : public TfLiteInternalBackendContext {
 public:
  static CpuBackendContext* GetFromContext(TfLiteContext* context);

  CpuBackendContext();
  ~CpuBackendContext() override;

  ruy::Context* ruy_context() const { return ruy_context_.get(); }

  gemmlowp::GemmContext* gemmlowp_context() const {
    return gemmlowp_context_.get();
  }

  // Sets the maximum-number-of-threads-to-use parameter, only as a means of
  // passing around this information.
  void SetMaxNumThreads(int max_num_threads) override;

  int max_num_threads() const { return max_num_threads_; }

  void SetUseCaching(bool flag);

  bool use_caching() const { return use_caching_; }

#ifdef TFLITE_KERNEL_USE_XNNPACK
  pthreadpool_t get_xnnpack_threadpool();
#endif

  void ClearCaches() override { ruy_context_->ClearPrepackedCache(); }

  // Gemmlowp on x86 is a deprecated path but some clients may still use
  // this path based on link time dependencies.
  bool PreferGemmlowpOnX86();

 private:
  bool RuyHasAvxOrAbove();

  // Copy the wrapper class for cpuinfo from Ruy.
  class CpuInfo final {
   public:
    CpuInfo() {}
    ~CpuInfo();

    // X86 features
    bool Avx();
    bool Avx2Fma();
    bool Avx512();

   private:
    enum class InitStatus {
      kNotYetAttempted,
      kInitialized,
      kFailed,
    };

    InitStatus init_status_ = InitStatus::kNotYetAttempted;

    bool EnsureInitialized();
    InitStatus Initialize();
    CpuInfo(const CpuInfo&) = delete;
    CpuInfo& operator=(const CpuInfo&) = delete;
  };

  // To enable a smooth transition from the current direct usage
  // of the underlying gemmlowp context to going through abstractions
  // (see :cpu_backend_gemm), for now a CpuBackendContext always
  // stores both a gemmlowp context and a ruy context.
  // TODO(b/131416458): Once call sites all go through abstractions,
  // elide what can be elided based on TFLITE_WITH_RUY.
  const std::unique_ptr<ruy::Context> ruy_context_;
  const std::unique_ptr<gemmlowp::GemmContext> gemmlowp_context_;
  CpuInfo cpuinfo_;

  // The maximum of threads used for parallelizing TfLite ops. However,
  // cpu_backend_threadpool::Execute creates as many threads as it's
  // asked to, regardless of this. Typically a call site would query
  // cpu_backend_context->max_num_threads() and used that to determine
  // the number of tasks to create and to give to
  // cpu_backend_threadpool::Execute.
  //
  // This value also gets propagated to back-ends, where it plays the same
  // information-only role.
  int max_num_threads_;
  // For matrix muliplications with constants parameters (i.e. weights), we can
  // sometimes provide speedups by caching the "prepacked" data, for some
  // additional memory cost. This flag permits the user to route all
  // CpuBackendGem operations to a library that permits such an optimization
  // (currently the Ruy library only).
  bool use_caching_;

#ifdef TFLITE_KERNEL_USE_XNNPACK
  // A smart pointer for the xnnpack threadpool. Is created by a call from the
  // interpreter, and then consumed by xnnpack, possibly via a TFLite kernel.
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
      xnnpack_threadpool_{nullptr, &pthreadpool_destroy};
#endif

  CpuBackendContext(const CpuBackendContext&) = delete;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
