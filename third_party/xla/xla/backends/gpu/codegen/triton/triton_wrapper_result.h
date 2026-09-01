/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TRITON_WRAPPER_RESULT_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TRITON_WRAPPER_RESULT_H_

#include <cstdint>
#include <vector>

#include "llvm/IR/Metadata.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {

struct TritonWrapperResult {
  int64_t shmem_bytes = 0;
  int64_t global_scratch_memory_size = 0;
  stream_executor::gpu::TmaMetadata tma_metadata;
  stream_executor::ThreadDim thread_dims;
  bool use_pdl = false;

  // The captured nvvm.annotations from the lowest level LLVM IR coming from
  // Triton. We need to propagate them because we later create the kernel and
  // splice the impl_fn into it.
  std::vector<llvm::Metadata*> nvvm_annotations;
  LlvmKernelSource kernel_source;
};

std::ostream& operator<<(std::ostream& os, const TritonWrapperResult& result);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TRITON_WRAPPER_RESULT_H_
