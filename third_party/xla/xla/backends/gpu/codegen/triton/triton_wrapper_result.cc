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

#include "xla/backends/gpu/codegen/triton/triton_wrapper_result.h"

namespace xla::gpu {

std::ostream& operator<<(std::ostream& os, const TritonWrapperResult& result) {
  os << "\nTritonWrapperResult: " << "\n";
  os << "  shmem_bytes: " << result.shmem_bytes << "\n";
  auto tma_metadata = result.tma_metadata.ToProto();
  os << "  tma_metadata: {\n";
  for (const auto& tma_entry : tma_metadata.arg_index_to_tma_info()) {
    os << "    " << tma_entry.first << " : " << tma_entry.second.DebugString()
       << "\n";
  }
  os << "  }\n";
  os << "  thread_dims: " << result.thread_dims.ToString() << "\n";
  os << "  nvvm_annotations: " << result.nvvm_annotations.size() << "\n";
  os << "  llvm_module: " << result.kernel_source.ToString() << "\n";
  return os;
}

}  // namespace xla::gpu
