/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"

// We need to specify "hard" types here because unfortunately cuda_runtime.h
// uses a fake overload for cudaMallocManaged and cudaFree
extern "C" {
  using managed_alloc_t = cudaError_t(*)(void**, size_t);
  using managed_free_t = cudaError_t(*)(void*);
};

namespace tensorflow {
#ifdef GOOGLE_CUDA
namespace dynload {

#define GPU_LIBCUDART_WRAP(__name, __cast)                                   \
  struct DynLoadShim__##__name {                                             \
    static const char* kName;                                                \
    static void* GetDsoHandle() {                                            \
      auto s =                                                               \
          stream_executor::internal::CachedDsoLoader::GetCudartDsoHandle();  \
      return s.ValueOrDie();                                                 \
    }                                                                        \
    static __cast LoadOrDie() {                                              \
      void* f;                                                               \
      auto s = stream_executor::port::Env::Default()->GetSymbolFromLibrary(  \
          GetDsoHandle(), kName, &f);                                        \
      CHECK(s.ok()) << "could not find " << kName                            \
                    << " in libcuda DSO; dlerror: " << s.error_message();    \
      return reinterpret_cast<__cast>(f);                                    \
    }                                                                        \
    static __cast DynLoad() {                                                \
      static __cast f = LoadOrDie();                                         \
      return f;                                                              \
    }                                                                        \
    template <typename... Args>                                              \
    cudaError_t operator()(Args... args) {                                   \
      return DynLoad()(args...);                                             \
    }                                                                        \
  } __name;                                                                  \
  const char* DynLoadShim__##__name::kName = #__name;

GPU_LIBCUDART_WRAP(cudaMallocManaged, managed_alloc_t)
GPU_LIBCUDART_WRAP(cudaFree, managed_free_t)
}  // namespace dynload
#endif  // GOOGLE_CUDA

void* GpuManagedAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* ptr = nullptr;
#ifdef GOOGLE_CUDA
  CHECK_EQ(dynload::cudaMallocManaged(&ptr, num_bytes), cudaSuccess);
#endif
  CHECK(!(reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)));
  return ptr;
}

void GpuManagedAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
  CHECK_EQ(dynload::cudaFree(ptr), cudaSuccess);
#endif
}

}  // namespace tensorflow
