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

#include "xla/backends/gpu/codegen/tensor_ir/custom_kernel.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "tensor_ir/Artifact/TensorIRArtifact.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::tensor_ir {

absl::StatusOr<KernelReuseCache::Entry> MakeKernelCacheEntry(
    const ::tensor_ir::rt::TensorIRArtifact& artifact) {
  llvm::ArrayRef<char> device_code =
      std::visit([](const auto& b) { return llvm::ArrayRef<char>(b.bytes); },
                 artifact.getBinary());
  if (device_code.empty()) {
    return absl::InternalError(absl::StrCat(
        "TensorIR kernel '", artifact.getKernelName(), "' has no device code"));
  }
  if (std::holds_alternative<::tensor_ir::rt::TileIRBytecode>(
          artifact.getBinary())) {
    // The buffer holds TileIR bytecode rather than an assembled cubin. That is
    // fine as far as loading goes: `cuModuleLoadData`, which StreamExecutor
    // calls to load a cubin, documents "Tile IR data" alongside cubin, fatbin
    // and PTX as an accepted image, and JITs it. But that JIT lives in the
    // driver and needs r610 or newer, so say so once per process -- this is a
    // property of the installation, not of any one kernel.
    LOG_FIRST_N(WARNING, 1)
        << "TensorIR kernel '" << artifact.getKernelName()
        << "' is TileIR bytecode rather than an assembled cubin, because no "
           "'tileiras' assembler was found; CUDA 13.3 or newer ships one. The "
           "CUDA driver will JIT the bytecode when the kernel is loaded, which "
           "needs a native r610 or newer driver.";
  }

  // The launch grid is fixed at compile time for everything XLA emits: the
  // compiler only installs a runtime grid computer for dynamic shapes, and a
  // fusion never has any.
  const std::optional<std::array<int32_t, 3>>& grid = artifact.getStaticGrid();
  if (!grid.has_value()) {
    return absl::InternalError(
        absl::StrCat("TensorIR kernel '", artifact.getKernelName(),
                     "' has a launch grid that is only known at run time"));
  }

  // CudaTile's launch ABI, from `CudaTileRuntimeKernel::launch`: one thread per
  // block, no shared memory, and -- since XLA only produces static shapes, for
  // which the compiler installs a `PointerOnlyArgPacker` -- one device pointer
  // per argument, in order. That is exactly what
  // `CreateSharedCubinCustomKernel` builds.
  //
  // The one thing lost is the `CU_CLUSTER_SCHEDULING_POLICY_SPREAD` launch
  // attribute that `launch` sets. It is a scheduling hint for thread block
  // clusters, and these kernels are launched without a cluster dimension.
  KernelReuseCache::Entry entry;
  entry.kernel_name = artifact.getFuncName();
  entry.launch_dimensions = LaunchDimensions(
      se::BlockDim((*grid)[0], (*grid)[1], (*grid)[2]), se::ThreadDim(1, 1, 1));
  entry.shmem_bytes = 0;
  entry.binary = std::make_shared<const std::vector<uint8_t>>(
      device_code.begin(), device_code.end());
  return entry;
}

}  // namespace xla::gpu::tensor_ir
