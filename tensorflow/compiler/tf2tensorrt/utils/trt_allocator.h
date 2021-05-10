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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ALLOCATOR_H_

#include <unordered_map>

#include "tensorflow/core/framework/allocator.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
// std::align is not supported, so this function mimic its behavior.
void* Align(uint64_t alignment, uint64_t size, void*& ptr, uint64_t& space);
}  // namespace tensorrt
}  // namespace tensorflow

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

class TRTBaseAllocator : public nvinfer1::IGpuAllocator {
  // Base allocator class so we can have a virtual destructor;
 public:
  // python wrapper seems to be not happy with an pure virtual destructor;
  virtual ~TRTBaseAllocator() = default;
};

class TRTDeviceAllocator : public TRTBaseAllocator {
  // Allocator implementation wrapping TF device allocators.
 public:
  TRTDeviceAllocator(Allocator* allocator);

  // TODO(aaroey): base class doesn't have a virtual destructor, work with
  // Nvidia to fix it.
  virtual ~TRTDeviceAllocator() {
    VLOG(1) << "Destroying allocator attached to " << allocator_->Name();
  }
  void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) override;
  void free(void* memory) override;

 private:
  Allocator* allocator_;

  // supporting alignment from allocation request requires a map to free;
  std::unordered_map<void*, void*> mem_map_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TRT_ALLOCATOR_H_
