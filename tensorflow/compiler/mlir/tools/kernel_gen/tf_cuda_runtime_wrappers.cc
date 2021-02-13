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

// Implements a C wrapper around the TensorFlow runtime and CUDA that allows
// launching a kernel on the current device and stream from a binary blob for
// the module and function name.

#if GOOGLE_CUDA

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

#define CUDA_REPORT_IF_ERROR_WITH_CTX(expr, context)                          \
  [](CUresult result, tensorflow::OpKernelContext *ctx) {                     \
    if (!result) return;                                                      \
    const char *name = nullptr;                                               \
    cuGetErrorName(result, &name);                                            \
    if (!name) name = "<unknown>";                                            \
    std::string msg = absl::StrCat("'", #expr, "' failed with '", name, "'"); \
    if (ctx != nullptr) {                                                     \
      ctx->CtxFailureWithWarning(                                             \
          tensorflow::Status{tensorflow::error::INTERNAL, msg});              \
    } else {                                                                  \
      LOG(WARNING) << msg << "\n";                                            \
    }                                                                         \
  }(expr, context)

#define CUDA_REPORT_IF_ERROR(expr) CUDA_REPORT_IF_ERROR_WITH_CTX(expr, nullptr)

namespace {
// Implements a cache for loading modules. The assumption is that we never
// unload modules again during the lifetime of a tensorflow runtime process.
struct CudaRuntimeCache {
 public:
  CUmodule loadModule(void *data) {
    tensorflow::mutex_lock lock(module_handle_mutex);
    CUmodule &module = module_handles[data];
    if (!module) {
      CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
    }
    return module;
  }

  // Returns the runtime cache for the context associated with stream.
  static CudaRuntimeCache *get(CUstream stream) {
    using CacheWithLock =
        std::pair<tensorflow::mutex,
                  absl::flat_hash_map<CUcontext, CudaRuntimeCache *>>;
    static auto *cache_with_lock = new CacheWithLock();
    tensorflow::mutex_lock lock(cache_with_lock->first);
    CUcontext context;
    CUDA_REPORT_IF_ERROR(cuStreamGetCtx(stream, &context));
    auto &runtime_cache = cache_with_lock->second[context];
    if (!runtime_cache) {
      runtime_cache = new CudaRuntimeCache();
    }
    return runtime_cache;
  }

 private:
  CudaRuntimeCache() = default;

  tensorflow::mutex module_handle_mutex;
  absl::flat_hash_map<void *, CUmodule> module_handles
      TF_GUARDED_BY(module_handle_mutex);
};
}  // namespace

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
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
  auto stream =
      reinterpret_cast<CUstream>(se_stream->implementation()->GpuStreamHack());
  CUmodule module = CudaRuntimeCache::get(stream)->loadModule(module_blob);
  CUfunction function;
  CUDA_REPORT_IF_ERROR_WITH_CTX(
      cuModuleGetFunction(&function, module, kernel_name), ctx);

  CUDA_REPORT_IF_ERROR_WITH_CTX(
      cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                     /*sharedMemBytes=*/0, stream, params, nullptr),
      ctx);
}

#endif  // GOOGLE_CUDA
