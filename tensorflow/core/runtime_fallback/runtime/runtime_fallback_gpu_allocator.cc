/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.h"

#include "llvm/Support/Errc.h"
#include "tensorflow/core/platform/mutex.h"
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {

class RuntimeFallbackGpuAllocator : public tfrt::gpu::GpuAllocator {
 public:
  explicit RuntimeFallbackGpuAllocator(
      tensorflow::Allocator* tf_gpu_allocator,
      const tfrt::gpu::wrapper::Context& context)
      : tf_gpu_allocator_(tf_gpu_allocator), context_(context) {}
  ~RuntimeFallbackGpuAllocator() override;

  llvm::Expected<tfrt::gpu::GpuPointer> Allocate(
      size_t size, tfrt::gpu::wrapper::Stream stream) override;

  llvm::Error Deallocate(tfrt::gpu::GpuPointer pointer,
                         tfrt::gpu::wrapper::Stream stream) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RuntimeFallbackGpuAllocator);

  // Structures immutable after construction

  // Does not own tf_gpu_allocator.
  tensorflow::Allocator* tf_gpu_allocator_;

  tfrt::gpu::wrapper::Context context_;

  // Structures mutable after construction
  mutable tensorflow::mutex mu_;

  // Because we don't support multiple streams, stream_ is the stream
  // for all allocations. All allocation requests on a different stream will be
  // denied.
  // We can't easily support "stream transitioning" now because:
  //  - we need to synchronize the former stream when we transition to the new
  //  stream.
  //  - the allocator is not notified when the stream is destroyed. So, the
  //  synchronization can happen after the stream is destroyed causing
  //  segfault.
  tfrt::gpu::wrapper::Stream stream_ TF_GUARDED_BY(mu_);
};

RuntimeFallbackGpuAllocator::~RuntimeFallbackGpuAllocator() {}

llvm::Expected<tfrt::gpu::GpuPointer> RuntimeFallbackGpuAllocator::Allocate(
    size_t size, tfrt::gpu::wrapper::Stream stream) {
  {
    tensorflow::mutex_lock lock(mu_);
    if (stream_ == nullptr) {
      stream_ = stream;
    } else if (stream != stream_) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "RuntimeFallbackGpuAllocator does not support multiple streams");
    }
  }
  // tfrt::gpu::GpuAllocator::kAlignment is the minimum alignment. AllocateRaw
  // adjusts alignment internally as needed.
  void* gpu_ptr =
      tf_gpu_allocator_->AllocateRaw(tfrt::gpu::GpuAllocator::kAlignment, size);

  // TODO(zhangqiaorjc): AllocateRaw does LOG(WARNING) for different errors, it
  // should return llvm::Error instead.
  if (gpu_ptr == nullptr)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        tfrt::StrCat("errors trying to allocate ", size));

  return tfrt::gpu::wrapper::Pointer<void>(gpu_ptr, context_.platform());
}

llvm::Error RuntimeFallbackGpuAllocator::Deallocate(
    tfrt::gpu::GpuPointer pointer, tfrt::gpu::wrapper::Stream stream) {
  tf_gpu_allocator_->DeallocateRaw(pointer.raw());
  return llvm::Error::success();
}

tfrt::gpu::GpuAllocatorFactory CreateRuntimeFallbackGpuAllocatorFactory(
    tensorflow::Allocator* tf_gpu_allocator) {
  return [tf_gpu_allocator](const tfrt::gpu::wrapper::Context& context) {
    return std::make_unique<RuntimeFallbackGpuAllocator>(tf_gpu_allocator,
                                                         context);
  };
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
