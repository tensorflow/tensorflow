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

#ifndef XLA_STREAM_EXECUTOR_MODULE_SPEC_H_
#define XLA_STREAM_EXECUTOR_MODULE_SPEC_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xla/stream_executor/platform/port.h"
#include "tsl/platform/logging.h"

namespace stream_executor {

//===----------------------------------------------------------------------===//
// ModuleHandle
//===----------------------------------------------------------------------===//

// An opaque handle to a loaded module.
//
// An instance of this is returned from StreamExecutor::GetModule.
class ModuleHandle {
 public:
  explicit ModuleHandle(void* id = nullptr) : id_(id) {}

  // A ModuleHandle with id() == nullptr is an invalid module handle, akin to a
  // null pointer.
  void* id() const { return id_; }

  explicit operator bool() const { return id() != nullptr; }

 private:
  void* id_;
};

//===----------------------------------------------------------------------===//
// MultiModuleLoaderSpec
//===----------------------------------------------------------------------===//

// Describes how to load a module on a target platform.
//
// The exact meaning of a "module" may differ from platform to platform but
// loosely speaking a module a collection of kernels and global variables.  It
// corresponds to CUmodule when running on CUDA.
class MultiModuleLoaderSpec {
 public:
  bool has_cuda_cubin_in_memory() const { return has_cuda_cubin_in_memory_; }
  absl::Span<const uint8_t> cuda_cubin_in_memory() const {
    CHECK(has_cuda_cubin_in_memory());
    return {cuda_cubin_in_memory_.data(), cuda_cubin_in_memory_.size()};
  }

  bool has_cuda_ptx_in_memory() const { return has_cuda_ptx_in_memory_; }
  const char* cuda_ptx_in_memory() const {
    CHECK(has_cuda_ptx_in_memory());
    return cuda_ptx_in_memory_;
  }

  void AddCudaCubinInMemory(absl::Span<const uint8_t> cubin_bytes) {
    CHECK(!cubin_bytes.empty());
    has_cuda_cubin_in_memory_ = true;
    cuda_cubin_in_memory_ = cubin_bytes;
  }

  void AddCudaPtxInMemory(const char* ptx) {
    has_cuda_ptx_in_memory_ = true;
    // The CUDA driver does not like getting an empty string as PTX.
    cuda_ptx_in_memory_ = *ptx ? ptx : nullptr;
  }

 private:
  absl::Span<const uint8_t> cuda_cubin_in_memory_;
  bool has_cuda_cubin_in_memory_ = false;
  const char* cuda_ptx_in_memory_;
  bool has_cuda_ptx_in_memory_ = false;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MODULE_SPEC_H_
