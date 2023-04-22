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

// Implements C wrappers around the ROCm library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.

#if TENSORFLOW_USE_ROCM

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "rocm/include/hip/hip_runtime.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

#define HIP_REPORT_IF_ERROR_WITH_CTX(expr, context)                           \
  [](hipError_t result, tensorflow::OpKernelContext *ctx) {                   \
    if (!result) return;                                                      \
    const char *name = hipGetErrorName(result);                               \
    if (!name) name = "<unknown>";                                            \
    std::string msg = absl::StrCat("'", #expr, "' failed with '", name, "'"); \
    if (ctx != nullptr) {                                                     \
      ctx->CtxFailureWithWarning(                                             \
          tensorflow::Status{tensorflow::error::INTERNAL, msg});              \
    } else {                                                                  \
      LOG(WARNING) << msg << "\n";                                            \
    }                                                                         \
  }(expr, context)

#define HIP_REPORT_IF_ERROR(expr) HIP_REPORT_IF_ERROR_WITH_CTX(expr, nullptr)

namespace {
// Implements a cache for loading modules. The assumption is that we never
// unload modules again during the lifetime of a tensorflow runtime process.
struct HipRuntimeCache {
 public:
  hipModule_t loadModule(void *data) {
    tensorflow::mutex_lock lock(module_handle_mutex);
    hipModule_t &module = module_handles[data];
    if (!module) {
      HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
    }
    return module;
  }

  // Returns the runtime cache for the context associated with stream.
  static HipRuntimeCache *get(hipStream_t stream) {
    using CacheWithLock =
        std::pair<tensorflow::mutex,
                  absl::flat_hash_map<hipCtx_t, HipRuntimeCache *>>;
    static auto *cache_with_lock = new CacheWithLock();
    tensorflow::mutex_lock lock(cache_with_lock->first);
    hipCtx_t context;
    // HIP does not support getting the context of a stream. Use the current
    // context instead.
    HIP_REPORT_IF_ERROR(hipCtxGetCurrent(&context));
    auto &runtime_cache = cache_with_lock->second[context];
    if (!runtime_cache) {
      runtime_cache = new HipRuntimeCache();
    }
    return runtime_cache;
  }

 private:
  HipRuntimeCache() = default;

  tensorflow::mutex module_handle_mutex;
  absl::flat_hash_map<void *, hipModule_t> module_handles
      TF_GUARDED_BY(module_handle_mutex);
};
}  // namespace

// The wrapper uses intptr_t instead of HIP's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" void tfKernelGenLaunchKernel(tensorflow::OpKernelContext *ctx,
                                        void *module_blob, char *kernel_name,
                                        intptr_t gridX, intptr_t gridY,
                                        intptr_t gridZ, intptr_t blockX,
                                        intptr_t blockY, intptr_t blockZ,
                                        void **params) {
  // For empty grids, we don't need to do anything.
  if (!gridX || !gridY || !gridZ) {
    return;
  }

  stream_executor::Stream *se_stream = ctx->op_device_context()->stream();
  auto stream = reinterpret_cast<hipStream_t>(
      se_stream->implementation()->GpuStreamHack());
  hipModule_t module = HipRuntimeCache::get(stream)->loadModule(module_blob);
  hipFunction_t function;
  HIP_REPORT_IF_ERROR_WITH_CTX(
      hipModuleGetFunction(&function, module, kernel_name), ctx);

  HIP_REPORT_IF_ERROR_WITH_CTX(
      hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                            blockZ,
                            /*sharedMemBytes=*/0, stream, params, nullptr),
      ctx);
}

#endif  // TENSORFLOW_USE_ROCM
