/* Copyright 2025 The OpenXLA Authors.
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

#include "xla/stream_executor/sycl/sycl_kernel.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::sycl {

// TODO(intel-tf): Implement this feature in SYCL
absl::StatusOr<int32_t> SyclKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  return absl::UnimplementedError(
      "GetMaxOccupiedBlocksPerCore is unimplemented for SYCL platform.");
}

// TODO(intel-tf): Implement this feature in SYCL
absl::StatusOr<KernelMetadata> SyclKernel::GetKernelMetadata() {
  return absl::UnimplementedError(
      "GetKernelMetadata is unimplemented for SYCL platform.");
}

absl::Status SyclKernel::Launch(const ThreadDim& thread_dims,
                                const BlockDim& block_dims,
                                const std::optional<ClusterDim>& cluster_dims,
                                Stream* stream, const KernelArgs& args) {
  VLOG(2) << thread_dims.ToString() << ", " << block_dims.ToString() << ", "
          << (cluster_dims.has_value() ? cluster_dims.value().ToString()
                                       : "ClusterDim{std::nullopt}");

  if (cluster_dims.has_value()) {
    VLOG(2) << cluster_dims.value().ToString();
  }

  ::sycl::kernel* function = gpu_function();
  if (function == nullptr) {
    return absl::InternalError("SYCL kernel function is not set");
  }

  // Launch kernels with packed arguments.
  const auto launch = [this, &cluster_dims, &thread_dims, &block_dims,
                       &function,
                       stream](const KernelArgsPackedArrayBase& packed) {
    // Use an extra argument when using shared memory.
    bool has_shared_memory = packed.number_of_shared_bytes() > 0;
    int32_t expected_number_of_arguments =
        Arity() + (has_shared_memory ? 1 : 0);

    CHECK_EQ(expected_number_of_arguments, packed.number_of_arguments())
        << "Kernel " << name() << " has " << packed.number_of_arguments()
        << " arguments, but expected " << expected_number_of_arguments
        << "; arity=" << Arity() << "; has_shared_memory=" << has_shared_memory
        << "; number_of_shared_bytes=" << packed.number_of_shared_bytes();

    std::vector<void*> kernel_args;
    const auto& arg_addrs = packed.argument_addresses();
    // If there are no arguments to pass (e.g., when using shared memory),
    // we can skip packing the kernel arguments.
    if (arg_addrs.empty() && packed.number_of_arguments() == 0) {
      return stream->LaunchKernel(thread_dims, block_dims, cluster_dims,
                                  function, name(), nullptr,
                                  packed.number_of_shared_bytes());
    } else {
      kernel_args.reserve(arg_addrs.size());
      for (const void* const arg : arg_addrs) {
        kernel_args.push_back(
            reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg)));
      }
      size_t kernel_args_size = kernel_args.size();
      std::array<void*, 2> config = {kernel_args.data(), &kernel_args_size};
      return stream->LaunchKernel(thread_dims, block_dims, cluster_dims,
                                  function, name(),
                                  reinterpret_cast<void**>(config.data()),
                                  packed.number_of_shared_bytes());
    }
  };

  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return launch(*packed);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceAddressArray>(&args)) {
    auto& pack = args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(*this, *device_mem));
    return launch(*packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

}  // namespace stream_executor::sycl
