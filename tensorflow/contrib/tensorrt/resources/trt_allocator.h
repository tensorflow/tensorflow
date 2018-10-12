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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_ALLOCATOR_H_
#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_ALLOCATOR_H_

#include <unordered_map>

#include "tensorflow/core/framework/allocator.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"
#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace tensorrt {
// std::align is not supported, so this function mimic its behavior.
void* Align(uint64_t alignment, uint64_t size, void*& ptr, uint64_t& space);
}  // namespace tensorrt
}  // namespace tensorflow

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#if NV_TENSORRT_MAJOR == 3
// Define interface here temporarily until TRT 4.0 is released
namespace nvinfer1 {
class IGpuAllocator {
 public:
  virtual void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) = 0;
  virtual void free(void* memory) = 0;
};
}  // namespace nvinfer1
#endif

namespace tensorflow {
namespace tensorrt {

class TRTBaseAllocator : public nvinfer1::IGpuAllocator {
  // Base allocator class so we can have a virtual destructor;
 public:
  // python wrapper seems to be not happy with an pure virtual destructor;
  virtual ~TRTBaseAllocator() = default;
};

class TRTCudaAllocator : public TRTBaseAllocator {
  // Allocator implementation that is using cuda allocator instead of device
  // allocator in case we can't get device allocator from TF.
 public:
  TRTCudaAllocator() {}
  virtual ~TRTCudaAllocator() {}
  void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) override;
  void free(void* memory) override;
};

class TRTDeviceAllocator : public TRTBaseAllocator {
  // Allocator implementation wrapping TF device allocators.
 public:
  TRTDeviceAllocator(tensorflow::Allocator* allocator);

  // TODO(aaroey): base class doesn't have a virtual destructor, work with
  // Nvidia to fix it.
  virtual ~TRTDeviceAllocator() {
    VLOG(1) << "Destroying allocator attached to " << allocator_->Name();
  }
  void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) override;
  void free(void* memory) override;

 private:
  tensorflow::Allocator* allocator_;

  // supporting alignment from allocation request requires a map to free;
  std::unordered_map<void*, void*> mem_map_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_ALLOCATOR_H_
