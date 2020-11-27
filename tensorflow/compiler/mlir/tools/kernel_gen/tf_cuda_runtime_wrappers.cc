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

// Implements C wrappers around the CUDA library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.

#include <cassert>
#include <numeric>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"

#define CUDA_REPORT_IF_ERROR(expr)                                      \
  [](CUresult result) {                                                 \
    if (!result) return;                                                \
    const char *name = nullptr;                                         \
    cuGetErrorName(result, &name);                                      \
    if (!name) name = "<unknown>";                                      \
    LOG(WARNING) << "'" << #expr << "' failed with '" << name << "'\n"; \
  }(expr)

namespace {
// Implements a cache for loading modules and creating streams. The assumption
// is that we never unload modules or delete streams again during the lifetime
// of a tensorflow runtime process.
struct CudaRuntimeCache {
 public:
  CUmodule loadModule(void *data) {
    tensorflow::mutex_lock lock(module_handle_mutex);
    auto it = module_handles.find(data);
    if (it != module_handles.end()) {
      return it->second;
    }
    CUmodule module = nullptr;
    CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
    module_handles.insert({data, module});
    return module;
  }

  CUstream createStream() {
    tensorflow::mutex_lock lock(stream_handle_mutex);
    CUstream stream = nullptr;
    if (stream_handles.empty()) {
      CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    } else {
      stream = stream_handles.back();
      stream_handles.pop_back();
    }
    return stream;
  }

  void releaseStream(CUstream stream) {
    tensorflow::mutex_lock lock(stream_handle_mutex);
    stream_handles.push_back(stream);
  }

  static CudaRuntimeCache *get() {
    static auto *instance = new CudaRuntimeCache();
    return instance;
  }

 private:
  CudaRuntimeCache() = default;

  tensorflow::mutex stream_handle_mutex;
  std::vector<CUstream> stream_handles TF_GUARDED_BY(stream_handle_mutex);
  tensorflow::mutex module_handle_mutex;
  absl::flat_hash_map<void *, CUmodule> module_handles
      TF_GUARDED_BY(module_handle_mutex);
};
}  // namespace

extern "C" CUmodule mgpuModuleLoad(void *data) {
  return CudaRuntimeCache::get()->loadModule(data);
}

extern "C" void mgpuModuleUnload(CUmodule module) {
  // We never unload modules.
}

extern "C" CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" void mgpuLaunchKernel(CUfunction function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem, CUstream stream,
                                 void **params, void **extra) {
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}

extern "C" CUstream mgpuStreamCreate() {
  return CudaRuntimeCache::get()->createStream();
}

extern "C" void mgpuStreamDestroy(CUstream stream) {
  return CudaRuntimeCache::get()->releaseStream(stream);
}

extern "C" void mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(CUstream stream, CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" CUevent mgpuEventCreate() {
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" void mgpuEventDestroy(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventDestroy(event));
}

extern "C" void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern "C" void mgpuEventRecord(CUevent event, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

#endif
