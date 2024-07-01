/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRITON_SPARSE_EXTENSIONS_H_
#define XLA_SERVICE_GPU_TRITON_SPARSE_EXTENSIONS_H_

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace xla::gpu {

std::unique_ptr<mlir::Pass> createAddSparseDotEncodingPass(
    int32_t num_warps, int32_t threads_per_warp, int32_t num_ctas);
std::unique_ptr<mlir::Pass> createSparseBlockedToMMAPass();
std::unique_ptr<mlir::Pass> createSparseLocalLoadOpToLLVMPass();
std::unique_ptr<mlir::Pass> createSparseDotOpToLLVMPass();

void registerSparsePasses();

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRITON_SPARSE_EXTENSIONS_H_
