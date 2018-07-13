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

// The ROCM implementation of the StreamExecutorInterface functionality.
// ROCM inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the ROCM streams
// programming model provided by the librocm.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_KERNEL_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_KERNEL_H_

#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/rocm/rocm_driver.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

#ifdef PLATFORMS_GPUS_ROCM_DYNAMIC_LIBROCM_DYNAMIC_LIBROCM_H_
#error \
    "No driver calls in this file, wrap driver functionality in rocm_driver.cc."
#endif

#ifdef __ROCM_RUNTIME_H__
#error \
    "ROCM runtime being included into ROCM GPU executor; should be driver only."
#endif

namespace stream_executor {
namespace rocm {

// Wraps a hipFunction_t to implement the platform-independent KernelInterface.
class ROCMKernel : public internal::KernelInterface {
 public:
  ROCMKernel()
      : rocm_function_(nullptr),
        arity_(0),
        preferred_cache_config_(KernelCacheConfig::kNoPreference) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the ROCMExecutor.
  ~ROCMKernel() override {}

  // As arity cannot be reflected upon using the ROCM API, the arity is
  // explicitly set during the ROCMExecutor::GetKernel initialization process.
  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  // Returns the hipFunction_t value for passing to the ROCM API.
  hipFunction_t AsROCMFunctionValue() const {
    DCHECK(rocm_function_ != nullptr);
    return const_cast<hipFunction_t>(rocm_function_);
  }

  // Returns the slot that the hipFunction_t is stored within for this object,
  // for the ROCM API which wants to load into a hipFunction_t*.
  hipFunction_t *rocm_function_ptr() { return &rocm_function_; }

  // ROCM supports setting the preferred cache configuration of a hipFunction_t
  // (more-or-less equivalent to a ROCMKernel). We support this via the below
  // functions; users can set a preference, and that is applied when the kernel
  // is [lazy-]loaded (in ROCMExecutor::Launch). The alternative would be to
  // load the kernel & set the preference when the user calls the setter below;
  // either approach is valid.
  // Sets the current kernel cache configuration preference.
  void SetPreferredCacheConfig(KernelCacheConfig config) override {
    preferred_cache_config_ = config;
  }

  // Returns the current kernel cache configuration preference.
  KernelCacheConfig GetPreferredCacheConfig() const override {
    return preferred_cache_config_;
  }

  // Returns the current kernel cache configuration preference as a
  // hipFuncCache_t.
  hipFuncCache_t GetROCMCacheConfig() const {
    switch (preferred_cache_config_) {
      case KernelCacheConfig::kNoPreference:
        return hipFuncCachePreferNone;
      case KernelCacheConfig::kPreferShared:
        return hipFuncCachePreferShared;
        ;
      case KernelCacheConfig::kPreferL1:
        return hipFuncCachePreferL1;
        ;
      case KernelCacheConfig::kPreferEqual:
        return hipFuncCachePreferEqual;
      default:
        LOG(FATAL) << "Unknown KernelCacheConfig"
                   << static_cast<int32>(preferred_cache_config_);
    }
  }

 private:
  hipFunction_t rocm_function_;  // Wrapped ROCM kernel handle.
  unsigned arity_;  // Number of formal parameters the kernel takes.

  // Preferred (but not required) cache configuration for this kernel.
  KernelCacheConfig preferred_cache_config_;
};

// Given a platform-independent kernel datatype, returns the (const) internal
// ROCM platform implementation pointer.
inline const ROCMKernel *AsROCMKernel(const KernelBase *kernel) {
  return static_cast<const ROCMKernel *>(kernel->implementation());
}

// Given a platform-independent kernel datatype, returns the (non-const)
// internal ROCM platform implementation pointer.
inline ROCMKernel *AsROCMKernel(KernelBase *kernel) {
  return static_cast<ROCMKernel *>(kernel->implementation());
}
}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_KERNEL_H_
