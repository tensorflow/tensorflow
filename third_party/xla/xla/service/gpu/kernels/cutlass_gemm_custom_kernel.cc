/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/kernels/cutlass_gemm_custom_kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::gemm_universal {

static constexpr auto Default = Arch::kDefault;  // NOLINT
static constexpr auto Sm80 = Arch::kSm80;        // NOLINT
static constexpr auto Sm90 = Arch::kSm90;        // NOLINT

// Each individual CUTLASS kernel adaptor will be compiled in a separate
// cuda_library and linked into the `cutlass_gemm_custom_kernels` target. We use
// this approach for a few reasons:
//
//   - It enables parallel compilation of CUTLASS templates which in practice
//     becomes quite expensive for any non-trivial GEMM.
//
//   - We do not include any of the CUTLASS headers in our custom kernel
//     library which would require converting it to a cuda_library, and we
//     want to minimize the number of headers included in .cu.cc files as NVCC
//     does not particularly like templates defined in ABSL.
//
extern template struct Adaptor<F32xF32ToF32<Default>>;
extern template struct DeviceKernel<F32xF32ToF32<Default>>;

extern template struct Adaptor<Bf16xBf16ToBf16<Default>>;
extern template struct DeviceKernel<Bf16xBf16ToBf16<Default>>;

extern template struct Adaptor<Bf16xBf16ToBf16<Sm80>>;
extern template struct DeviceKernel<Bf16xBf16ToBf16<Sm80>>;

extern template struct Adaptor<Bf16xBf16ToBf16<Sm90>>;
extern template struct DeviceKernel<Bf16xBf16ToBf16<Sm90>>;

//===----------------------------------------------------------------------===//
// CUTLASS kernel arguments packing
//===----------------------------------------------------------------------===//

using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

template <typename Dim>
static Dim As(Dim3 dim3) {
  return Dim(dim3.x, dim3.y, dim3.z);
}

template <typename Dim>
static std::optional<Dim> As(std::optional<Dim3> dim3) {
  if (dim3.has_value()) return Dim(dim3->x, dim3->y, dim3->z);
  return std::nullopt;
}

// Returns a pointer to device memory holding a slice offset.
static int32_t* SlicePtr(const se::KernelArgsDeviceMemoryArray* args,
                         int64_t index) {
  const void* opaque = args->device_memory_ptr(index);
  return static_cast<int32_t*>(const_cast<void*>(opaque));
}

template <typename Tag>
KernelArgsPacking ArgsPacking(int32_t m, int32_t n, int32_t k,
                              const ArgsIndices& indices,
                              const DynamicSliceIndices& slices,
                              int32_t device_sms, Adaptor<Tag> adaptor) {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;

  // TODO(ezhulenev): CUTLASS kernel Params struct not necessarily trivially
  // destructible or even trivially copyable, we have to own the life time of an
  // object constructed in the storage. For now we ignore it, and it's textbook
  // definition of UB, but for CUTLASS kernels we use today it's perfectly safe.
  struct Params {
    alignas(64) std::byte storage[1024];
  };

  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

    Arguments arguments = {m, n, k};
    arguments.lhs = const_cast<void*>(mem_args->device_memory_ptr(indices.lhs));
    arguments.rhs = const_cast<void*>(mem_args->device_memory_ptr(indices.rhs));
    arguments.out = const_cast<void*>(mem_args->device_memory_ptr(indices.out));

    // Workspace argument always passed as the last one (if passed at all).
    if (indices.has_workspace) {
      size_t num_mem_args = mem_args->device_memory_args().size();
      arguments.workspace =
          const_cast<void*>(mem_args->device_memory_ptr(num_mem_args - 1));
    } else {
      arguments.workspace = nullptr;
    }

    // Set up dynamic slices if they are available.
    if (slices.out.has_value()) {
      arguments.slices.out = SlicePtr(mem_args, *slices.out);
    }

    if (!adaptor.CanImplement(arguments)) {
      return absl::InternalError(absl::StrCat(
          "CUTLASS kernel can not implement gemm for a given problem size",
          ": m=", m, ", n=", n, ", k=", k));
    }

    auto threads = As<se::ThreadDim>(adaptor.ThreadDim());
    auto shmem_bytes = adaptor.SharedMemoryBytes();

    // We keep max_occupancy in a static variable as currently for all
    // practical purposes all stream executors in the process have identical
    // underlying devices, and there is no need to repeatedly query this
    // property.
    static int32_t sm_occupancy =
        kernel.GetMaxOccupiedBlocksPerCore(threads, shmem_bytes).value_or(1);

    // TODO(ezhulenev): In theory when sm_occupancy is 0 we should not be able
    // to run kernels, and we could return error here, however in practice
    // it's not true, and kernels with 0 occupancy run just fine! Figure out
    // where is the problem, and how we can reliably use sm occupancy numbers.
    //
    // TODO(ezhulenev): We need to set kernel dynamic shmem limit before asking
    // for sm occupancy, it's likely why we get 0 today.
    if (sm_occupancy == 0) {
      LOG_FIRST_N(WARNING, 1)
          << "CUTLASS gemm kernel reported 0 occupancy: threads_per_block="
          << (threads.x * threads.y * threads.z)
          << ", dynamic_shared_memory_bytes=" << shmem_bytes;
    }

    // Initialize parameters storage using adaptor.
    Params params;
    adaptor.Initialize(&params, arguments, device_sms, sm_occupancy);

    // TODO(ezhulenev): We need to support EmplaceKernelArgs with inplace
    // construction to avoid copying 1kb of byte storage.
    //
    // TODO(ezhulenev): Remove `DynamicSliceArguments` once we encode
    // dynamic slice offsets in kernel parameters.
    return se::PackKernelArgs<Params, DynamicSliceArguments>(
        args.number_of_shared_bytes(), params, arguments.slices);
  };
}
//===----------------------------------------------------------------------===//

template <typename Tag>
static absl::StatusOr<CustomKernel> Load(std::string name, int32_t m, int32_t n,
                                         int32_t k, const ArgsIndices& indices,
                                         const DynamicSliceIndices& slices,
                                         const se::DeviceDescription& device,
                                         Adaptor<Tag> adaptor = {},
                                         DeviceKernel<Tag> kernel = {}) {
  // Get the dispatch grid size and shared memory requirements.
  auto cluster_dim = As<se::ClusterDim>(adaptor.ClusterDim());
  auto block_dim = As<se::BlockDim>(adaptor.BlockDim(m, n, k));
  auto thread_dim = As<se::ThreadDim>(adaptor.ThreadDim());
  auto shared_memory_bytes = adaptor.SharedMemoryBytes();

  auto packing =
      ArgsPacking<Tag>(m, n, k, indices, slices, device.core_count(), adaptor);

  se::MultiKernelLoaderSpec spec(/*arity=*/2, std::move(packing));
  spec.AddInProcessSymbol(kernel.symbol(), name);

  if (cluster_dim.has_value()) {
    return CustomKernel(std::move(name), std::move(spec), block_dim, thread_dim,
                        *cluster_dim, shared_memory_bytes);
  } else {
    return CustomKernel(std::move(name), std::move(spec), block_dim, thread_dim,
                        shared_memory_bytes);
  }
}

absl::StatusOr<CustomKernel> GetCutlassGemmKernel(
    std::string name, PrimitiveType dtype, int32_t m, int32_t n, int32_t k,
    const ArgsIndices& indices, const DynamicSliceIndices& slices,
    const se::DeviceDescription& device) {
  auto& cuda_cc =
      std::get<se::CudaComputeCapability>(device.gpu_compute_capability());

  switch (dtype) {
    case PrimitiveType::F32:
      return Load<F32xF32ToF32<Default>>(std::move(name), m, n, k, indices,
                                         slices, device);
    case PrimitiveType::BF16:
      if (cuda_cc.IsAtLeastHopper()) {
        return Load<Bf16xBf16ToBf16<Sm90>>(std::move(name), m, n, k, indices,
                                           slices, device);
      } else if (cuda_cc.IsAtLeastAmpere()) {
        return Load<Bf16xBf16ToBf16<Sm80>>(std::move(name), m, n, k, indices,
                                           slices, device);
      }
      return Load<Bf16xBf16ToBf16<Default>>(std::move(name), m, n, k, indices,
                                            slices, device);

    default:
      return absl::InvalidArgumentError("Unsupported CUTLASS gemm data type");
  }
}

absl::StatusOr<CustomKernel> LoadCutlassGemmKernel(
    std::string name, const std::string& library_path, PrimitiveType dtype,
    int32_t m, int32_t n, int32_t k, const ArgsIndices& indices,
    const DynamicSliceIndices& slices, const se::DeviceDescription& device) {
  auto adaptor = Adaptor<DlOpenedKernel>::Load(library_path);
  if (!adaptor.has_value()) {
    return absl::InternalError(
        absl::StrCat("Failed to load CUTLASS adaptor from a shared library: ",
                     library_path));
  }

  auto kernel = DeviceKernel<DlOpenedKernel>::Load(library_path);
  if (!kernel.has_value()) {
    return absl::InternalError(absl::StrCat(
        "Failed to load CUTLASS kernel from a shared library: ", library_path));
  }

  return Load<DlOpenedKernel>(std::move(name), m, n, k, indices, slices, device,
                              *adaptor, *kernel);
}

}  // namespace xla::gpu::kernel::gemm_universal
