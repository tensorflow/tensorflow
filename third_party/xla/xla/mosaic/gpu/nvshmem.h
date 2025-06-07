/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_GPU_COMM_H_
#define JAXLIB_MOSAIC_GPU_COMM_H_

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

#define NVSHMEM_SUCCESS 0

namespace mosaic {
namespace gpu {

#define NVSHMEM_SET_FN(FnName)                                          \
  FnName = reinterpret_cast<decltype(FnName)>(dlsym(library, #FnName)); \
  if (!FnName) {                                                        \
    fprintf(stderr, #FnName " not available in this library.");         \
  }

class NvshmemApi {
 public:
  // Returns a default NvshmemApi for a current process.
  // NvshmemApi follows the Singleton design pattern
  static NvshmemApi& Default(bool assert_ok = true) {
    static NvshmemApi instance;
    if (assert_ok && !instance.is_loaded()) {
      fprintf(stderr, "Failed to load the NVSHMEM library.\n");
      abort();
    }
    return instance;
  }

  int cumodule_init(CUmodule module) {
    std::lock_guard<std::mutex> lock(mutex_);
    return nvshmemx_cumodule_init(module);
  }

  void barrier_all_on_stream(cudaStream_t stream) {
    nvshmemx_barrier_all_on_stream(stream);
  }

  bool is_loaded() {
    return nvshmemx_init_status != nullptr && nvshmemx_init_status() == 2;
  }

  NvshmemApi(NvshmemApi const&) = delete;
  void operator=(NvshmemApi const&) = delete;

 private:
  NvshmemApi() {
    const char* env_value = getenv("MOSAIC_GPU_NVSHMEM_SO_PATH");
    const char* libnvshmem_path =
        env_value && *env_value != 0 ? env_value : nullptr;
    void* library = dlopen(libnvshmem_path, RTLD_LAZY);
    if (library == nullptr) {
      fprintf(stderr, "Failed to open library (from %s): %s",
              libnvshmem_path ? libnvshmem_path : "<in process>", dlerror());
    }

    NVSHMEM_SET_FN(nvshmemx_barrier_all_on_stream)
    NVSHMEM_SET_FN(nvshmemx_cumodule_init)
    NVSHMEM_SET_FN(nvshmemx_init_status)
  }

  int (*nvshmemx_barrier_all_on_stream)(cudaStream_t);
  int (*nvshmemx_cumodule_init)(CUmodule);
  int (*nvshmemx_init_status)();

  std::mutex mutex_;
};

}  // namespace gpu
}  // namespace mosaic

#endif  // JAXLIB_MOSAIC_GPU_COMM_H_
