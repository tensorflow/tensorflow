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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_MODULE_SPEC_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_MODULE_SPEC_H_

#include "tensorflow/compiler/xla/stream_executor/lib/array_slice.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"

namespace stream_executor {

// Describes how to load a module on a target platform.
//
// The exact meaning of a "module" may differ from platform to platform but
// loosely speaking a module a collection of kernels and global variables.  It
// corresponds to CUmodule when running on CUDA.
class MultiModuleLoaderSpec {
 public:
  bool has_cuda_cubin_in_memory() const { return has_cuda_cubin_in_memory_; }
  port::ArraySlice<const uint8> cuda_cubin_in_memory() const {  // non-absl ok
    CHECK(has_cuda_cubin_in_memory());
    return {cuda_cubin_in_memory_.data(), cuda_cubin_in_memory_.size()};
  }

  bool has_cuda_ptx_in_memory() const { return has_cuda_ptx_in_memory_; }
  const char* cuda_ptx_in_memory() const {
    CHECK(has_cuda_ptx_in_memory());
    return cuda_ptx_in_memory_;
  }

  void AddCudaCubinInMemory(
      port::ArraySlice<const uint8> cubin_bytes) {  // non-absl ok
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
  port::ArraySlice<const uint8> cuda_cubin_in_memory_;  // non-absl ok
  bool has_cuda_cubin_in_memory_ = false;
  const char* cuda_ptx_in_memory_;
  bool has_cuda_ptx_in_memory_ = false;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_MODULE_SPEC_H_
