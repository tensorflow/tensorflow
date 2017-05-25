/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if !TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_COMMON_RUNTIME_SYCL_SYCL_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_SYCL_SYCL_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

class SYCLAllocator : public Allocator {
 public:
  SYCLAllocator(Eigen::QueueInterface *device) : device_(device) {}
  virtual ~SYCLAllocator() override;
  string Name() override;
  void *AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void *ptr) override;

  void EnterLameDuckMode();
  virtual bool ShouldAllocateEmptyTensors() override final { return true; }

 private:
  Eigen::QueueInterface *device_;  // not owned
  TF_DISALLOW_COPY_AND_ASSIGN(SYCLAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_SYCL_SYCL_ALLOCATOR_H_
