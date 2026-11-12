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

#include "xla/backends/gpu/runtime/ffi_kernel_registry.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/kernel_registry_api.h"
#include "xla/ffi/api/kernel_registry_c_api.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

// Concrete definition of the opaque kernel handle declared in the FFI C API.
struct XLA_FFI_Kernel {
  struct DeviceKernel {
    std::unique_ptr<stream_executor::Kernel> owned;
    std::atomic<stream_executor::Kernel*> published{nullptr};
  };

  XLA_FFI_Kernel(const xla::gpu::FfiKernelRegistry* absl_nonnull owner,
                 std::string name, std::string binary,
                 XLA_FFI_SourceFormat format, bool use_pdl, int num_devices)
      : owner(owner),
        name(std::move(name)),
        binary(std::move(binary)),
        format(format),
        use_pdl(use_pdl),
        per_device_kernel(num_devices) {}

  const xla::gpu::FfiKernelRegistry* owner;
  std::string name;
  // Declared before `per_device_kernel` so kernels unload before bytes free.
  std::string binary;
  XLA_FFI_SourceFormat format;
  bool use_pdl;
  xla::gpu::PerDeviceState<DeviceKernel> per_device_kernel;
};

namespace xla::gpu {

struct FfiKernelRegistry::Impl {
  // Identifies a registered kernel by content. Views in stored entries point
  // into the owned `XLA_FFI_Kernel`; lookup keys view the caller's input.
  struct KernelKey {
    absl::string_view name;
    absl::string_view binary;
    size_t binary_hash;
    XLA_FFI_SourceFormat format;
    bool use_pdl;

    bool operator==(const KernelKey& other) const {
      return binary_hash == other.binary_hash && format == other.format &&
             use_pdl == other.use_pdl && name == other.name &&
             binary == other.binary;
    }

    template <typename H>
    friend H AbslHashValue(H h, const KernelKey& key) {
      return H::combine(std::move(h), key.name, key.binary_hash, key.format,
                        key.use_pdl);
    }
  };

  explicit Impl(int num_devices) : num_devices(num_devices) {}

  int num_devices;
  absl::Mutex mutex;
  absl::flat_hash_map<KernelKey, std::unique_ptr<XLA_FFI_Kernel>> kernels
      ABSL_GUARDED_BY(mutex);
};

namespace {

// Checks that a C API struct is present and at least `min_size` bytes.
template <typename T>
absl::Status CheckStruct(const T* absl_nullable value, size_t min_size,
                         absl::string_view name) {
  if (value == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s cannot be null", name));
  }
  if (value->struct_size < min_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unexpected %s size: expected %zu, got %zu", name,
                        min_size, value->struct_size));
  }
  return absl::OkStatus();
}

stream_executor::Kernel* absl_nullable LookupPublishedKernel(
    const XLA_FFI_Kernel& kernel, int device_ordinal) {
  const XLA_FFI_Kernel::DeviceKernel* slot =
      kernel.per_device_kernel.Find(device_ordinal);
  return slot != nullptr ? slot->published.load(std::memory_order_acquire)
                         : nullptr;
}

absl::Status LaunchKernelImpl(XLA_FFI_KernelRegistry_Launch_Args* args) {
  ABSL_RETURN_IF_ERROR(CheckStruct(args,
                              XLA_FFI_KernelRegistry_Launch_Args_STRUCT_SIZE,
                              "XLA_FFI_KernelRegistry_Launch_Args"));
  // NOLINTNEXTLINE(*-reinterpret-cast): `state` encodes `Stream*` directly.
  auto* stream = reinterpret_cast<stream_executor::Stream*>(args->state);
  if (stream == nullptr) {
    return absl::FailedPreconditionError(
        "Kernel launch is only available in the Execute stage");
  }
  const XLA_FFI_KernelLaunchParams* params = args->launch_params;
  ABSL_RETURN_IF_ERROR(CheckStruct(params, XLA_FFI_KernelLaunchParams_STRUCT_SIZE,
                              "XLA_FFI_KernelLaunchParams"));
  ABSL_ASSIGN_OR_RETURN(
      stream_executor::Kernel * se_kernel,
      FindDeviceKernel(args->kernel, stream->parent()->device_ordinal()));
  ABSL_ASSIGN_OR_RETURN(FfiLaunchDims dims,
                   ConvertFfiLaunchDims(params->launch_dims));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<stream_executor::KernelArgsPackedArray> packed_args,
      PackKernelArgs(params->args, params->shared_mem_bytes));
  return se_kernel->Launch(dims.threads, dims.blocks, dims.cluster, stream,
                           *packed_args);
}

}  // namespace

absl::StatusOr<std::unique_ptr<stream_executor::KernelArgsPackedArray>>
PackKernelArgs(const XLA_FFI_KernelArgs* absl_nullable args,
               uint32_t shared_mem_bytes) {
  static constexpr XLA_FFI_KernelArgs kEmptyArgs = {nullptr, 0};
  if (args == nullptr) {
    args = &kEmptyArgs;
  }
  if (args->num_args < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Kernel argument count must not be negative, got %d", args->num_args));
  }
  if (args->num_args > 0 && args->args == nullptr) {
    return absl::InvalidArgumentError(
        "Kernel argument array cannot be null when num_args > 0");
  }
  auto packed_args =
      std::make_unique<stream_executor::KernelArgsPackedArray>(args->num_args);
  packed_args->add_shared_bytes(shared_mem_bytes);
  for (int64_t i = 0; i < args->num_args; ++i) {
    const XLA_FFI_KernelArg& arg = args->args[i];
    if (arg.type == XLA_FFI_KernelArgType_HostValue) {
      if (arg.size <= 0 || arg.arg_address == nullptr) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Host value argument %d must have a non-null address and a "
            "positive size",
            i));
      }
      packed_args->add_argument(stream_executor::PackedKernelArg(
          arg.size, [&arg](absl::Span<char> packed) {
            std::memcpy(packed.data(), arg.arg_address, arg.size);
          }));
    } else if (arg.type == XLA_FFI_KernelArgType_DevicePtr) {
      if (arg.size != 0) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Device pointer argument %d must have size 0", i));
      }
      // DeviceAddressBase needs void*; kernel launches never mutate it on host.
      // NOLINTNEXTLINE(*-readability-const-cast): see comment above.
      void* device_ptr = const_cast<void*>(arg.arg_address);
      packed_args->add_argument(stream_executor::DeviceAddressBase(device_ptr));
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported kernel argument type %d for argument %d", arg.type, i));
    }
  }
  return packed_args;
}

absl::StatusOr<FfiLaunchDims> ConvertFfiLaunchDims(
    const XLA_FFI_LaunchDims& launch_dims) {
  auto positive = [](const XLA_FFI_Dim3& d) {
    return d.x > 0 && d.y > 0 && d.z > 0;
  };
  const XLA_FFI_Dim3& grid = launch_dims.grid;
  const XLA_FFI_Dim3& block = launch_dims.block;
  const XLA_FFI_Dim3& cluster = launch_dims.cluster;
  const bool no_cluster = cluster.x == 0 && cluster.y == 0 && cluster.z == 0;
  if (!positive(grid) || !positive(block) ||
      (!no_cluster && !positive(cluster))) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Grid and block dimensions must be positive and the cluster must be "
        "all zero or all positive, got grid=(%d, %d, %d) block=(%d, %d, %d) "
        "cluster=(%d, %d, %d)",
        grid.x, grid.y, grid.z, block.x, block.y, block.z, cluster.x, cluster.y,
        cluster.z));
  }
  std::optional<stream_executor::ClusterDim> cluster_dim;
  if (!no_cluster) {
    cluster_dim = stream_executor::ClusterDim(cluster.x, cluster.y, cluster.z);
  }
  return FfiLaunchDims{stream_executor::ThreadDim(block.x, block.y, block.z),
                       stream_executor::BlockDim(grid.x, grid.y, grid.z),
                       cluster_dim};
}

absl::StatusOr<stream_executor::Kernel*> FindDeviceKernel(
    const XLA_FFI_Kernel* absl_nullable kernel, int device_ordinal) {
  if (kernel == nullptr) {
    return absl::InvalidArgumentError("XLA_FFI_Kernel cannot be null");
  }
  if (stream_executor::Kernel* loaded =
          LookupPublishedKernel(*kernel, device_ordinal);
      loaded != nullptr) {
    return loaded;
  }
  return absl::NotFoundError(
      absl::StrFormat("Kernel '%s' is not registered on device %d",
                      kernel->name, device_ordinal));
}

FfiKernelRegistry::FfiKernelRegistry(int num_devices)
    : impl_(std::make_unique<Impl>(num_devices)) {}

FfiKernelRegistry::~FfiKernelRegistry() = default;

bool FfiKernelRegistry::Contains(
    int device_ordinal, const XLA_FFI_Kernel* absl_nullable kernel) const {
  return kernel != nullptr && kernel->owner == this &&
         LookupPublishedKernel(*kernel, device_ordinal) != nullptr;
}

absl::StatusOr<const XLA_FFI_Kernel*> FfiKernelRegistry::Register(
    stream_executor::StreamExecutor* absl_nullable executor,
    absl::string_view name, const void* absl_nullable data, size_t size,
    XLA_FFI_SourceFormat format, bool use_pdl) {
  if (executor == nullptr) {
    return absl::InvalidArgumentError("StreamExecutor cannot be null");
  }
  if (name.empty()) {
    return absl::InvalidArgumentError("Kernel name cannot be empty");
  }
  if (data == nullptr || size == 0) {
    return absl::InvalidArgumentError("Kernel data cannot be empty");
  }
  if (format != XLA_FFI_SourceFormat_PTX &&
      format != XLA_FFI_SourceFormat_CUBIN) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FfiKernelRegistry only supports PTX and CUBIN kernels, got format %d",
        format));
  }
  const absl::string_view binary(static_cast<const char*>(data), size);
  const size_t binary_hash = absl::HashOf(binary);
  const Impl::KernelKey lookup_key{name, binary, binary_hash, format, use_pdl};

  XLA_FFI_Kernel* kernel = nullptr;
  {
    absl::ReaderMutexLock reader_lock(impl_->mutex);
    if (auto it = impl_->kernels.find(lookup_key); it != impl_->kernels.end()) {
      kernel = it->second.get();
    }
  }
  if (kernel == nullptr) {
    absl::MutexLock writer_lock(impl_->mutex);
    auto it = impl_->kernels.find(lookup_key);
    if (it == impl_->kernels.end()) {
      auto owned = std::make_unique<XLA_FFI_Kernel>(
          this, std::string(name), std::string(binary), format, use_pdl,
          impl_->num_devices);
      // The stored key views the owned copy, which lives as long as the entry.
      Impl::KernelKey owned_key{owned->name, owned->binary, binary_hash, format,
                                use_pdl};
      it = impl_->kernels.emplace(owned_key, std::move(owned)).first;
    }
    kernel = it->second.get();
  }

  const int device_ordinal = executor->device_ordinal();
  ABSL_RETURN_IF_ERROR(kernel->per_device_kernel.GetOrCreateAndInitialize(
      device_ordinal,
      [executor, kernel](XLA_FFI_Kernel::DeviceKernel* slot) -> absl::Status {
        auto spec =
            kernel->format == XLA_FFI_SourceFormat_PTX
                ? stream_executor::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                      kernel->binary, kernel->name, /*arity=*/0)
                : stream_executor::KernelLoaderSpec::
                      CreateCudaCubinInMemorySpec(
                          absl::Span<const uint8_t>(
                              reinterpret_cast<const uint8_t*>(
                                  kernel->binary.data()),
                              kernel->binary.size()),
                          kernel->name, /*arity=*/0);
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<stream_executor::Kernel> loaded,
                         executor->LoadKernel(spec));
        // Set once before publication so concurrent launches only read it.
        loaded->set_use_pdl(kernel->use_pdl);
        slot->owned = std::move(loaded);
        slot->published.store(slot->owned.get(), std::memory_order_release);
        return absl::OkStatus();
      }));
  return kernel;
}

absl::Status KernelRegistrationExtension::RegisterImpl(
    XLA_FFI_KernelRegistry_Register_Args* args) {
  ABSL_RETURN_IF_ERROR(CheckStruct(args,
                              XLA_FFI_KernelRegistry_Register_Args_STRUCT_SIZE,
                              "XLA_FFI_KernelRegistry_Register_Args"));
  args->kernel = nullptr;
  // NOLINTNEXTLINE(*-reinterpret-cast): `state` points to `State`.
  const auto* state = reinterpret_cast<const State*>(args->state);
  if (state == nullptr || state->registry == nullptr ||
      state->executor == nullptr) {
    return absl::FailedPreconditionError(
        "Kernel registration is only available in Prepare and Initialize");
  }
  const XLA_FFI_KernelSpec* spec = args->kernel_spec;
  ABSL_RETURN_IF_ERROR(
      CheckStruct(spec, XLA_FFI_KernelSpec_STRUCT_SIZE, "XLA_FFI_KernelSpec"));
  if (spec->name.ptr == nullptr && spec->name.len > 0) {
    return absl::InvalidArgumentError("Kernel name pointer cannot be null");
  }
  const absl::string_view name(spec->name.ptr, spec->name.len);
  const bool use_pdl = args->uses_pdl != 0;
  if (const XLA_FFI_Kernel* cached = args->cached_kernel;
      cached != nullptr &&
      state->registry->Contains(state->executor->device_ordinal(), cached) &&
      cached->name == name && cached->binary.size() == spec->size &&
      cached->format == spec->format && cached->use_pdl == use_pdl) {
    args->kernel = cached;
    return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(args->kernel, state->registry->Register(
                                     state->executor, name, spec->data,
                                     spec->size, spec->format, use_pdl));
  return absl::OkStatus();
}

XLA_FFI_Error* KernelRegistrationExtension::Register(
    XLA_FFI_KernelRegistry_Register_Args* args) {
  return ffi::CreateError(RegisterImpl(args));
}

KernelRegistrationExtension::KernelRegistrationExtension(
    FfiKernelRegistry* absl_nonnull registry,
    stream_executor::StreamExecutor* absl_nonnull executor)
    : state_{registry, executor} {
  static constexpr XLA_FFI_KernelRegistryApi kRegistrationApi = {
      /*register_kernel=*/&KernelRegistrationExtension::Register,
      /*launch_kernel=*/nullptr,
  };
  extension_ = XLA_FFI_KernelRegistry_Extension{
      ffi::MakeExtensionHeader<
          ffi::internal::KernelRegistryExtensionBase<void>>(),
      // NOLINTNEXTLINE(*-reinterpret-cast): opaque C handle to `state_`.
      reinterpret_cast<XLA_FFI_KernelRegistryState*>(&state_),
      &kRegistrationApi,
  };
}

XLA_FFI_KernelRegistry_Extension MakeKernelLaunchExtension(
    stream_executor::Stream* absl_nonnull stream) {
  static constexpr XLA_FFI_KernelRegistryApi kLaunchApi = {
      /*register_kernel=*/nullptr,
      /*launch_kernel=*/
      +[](XLA_FFI_KernelRegistry_Launch_Args* args) {
        return ffi::CreateError(LaunchKernelImpl(args));
      },
  };
  return XLA_FFI_KernelRegistry_Extension{
      ffi::MakeExtensionHeader<
          ffi::internal::KernelRegistryExtensionBase<void>>(),
      // NOLINTNEXTLINE(*-reinterpret-cast): `state` encodes `Stream*` directly.
      reinterpret_cast<XLA_FFI_KernelRegistryState*>(stream),
      &kLaunchApi,
  };
}

}  // namespace xla::gpu
