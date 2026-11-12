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

#ifndef XLA_BACKENDS_GPU_RUNTIME_FFI_KERNEL_REGISTRY_H_
#define XLA_BACKENDS_GPU_RUNTIME_FFI_KERNEL_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/kernel_registry_c_api.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// Validates FFI kernel args and converts them into StreamExecutor packed args.
// Shared by kernel launches via the registry and by RecordContext launches.
absl::StatusOr<std::unique_ptr<stream_executor::KernelArgsPackedArray>>
PackKernelArgs(const XLA_FFI_KernelArgs* absl_nullable args,
               uint32_t shared_mem_bytes);

// Launch dimensions converted from `XLA_FFI_LaunchDims`.
struct FfiLaunchDims {
  stream_executor::ThreadDim threads;
  stream_executor::BlockDim blocks;
  std::optional<stream_executor::ClusterDim> cluster;
};

// Validates and converts `XLA_FFI_LaunchDims`. Grid and block dimensions must
// be positive. The cluster is either all zeros (no cluster) or all positive.
absl::StatusOr<FfiLaunchDims> ConvertFfiLaunchDims(
    const XLA_FFI_LaunchDims& launch_dims);

// Returns the kernel loaded on `device_ordinal` for `kernel`, or errors if
// `kernel` is null or not registered on `device_ordinal`. Lock-free.
absl::StatusOr<stream_executor::Kernel*> FindDeviceKernel(
    const XLA_FFI_Kernel* absl_nullable kernel, int device_ordinal);

// Executable-scoped registry that owns the kernels FFI handlers register.
class FfiKernelRegistry {
 public:
  explicit FfiKernelRegistry(int num_devices);
  ~FfiKernelRegistry();

  FfiKernelRegistry(const FfiKernelRegistry&) = delete;
  FfiKernelRegistry& operator=(const FfiKernelRegistry&) = delete;

  // Registers (or reuses) a kernel handle and loads it on `executor`'s device.
  // `data` is copied when needed; matching (name, binary, format, `use_pdl`)
  // share one handle.
  absl::StatusOr<const XLA_FFI_Kernel*> Register(
      stream_executor::StreamExecutor* absl_nullable executor,
      absl::string_view name, const void* absl_nullable data, size_t size,
      XLA_FFI_SourceFormat format, bool use_pdl);

  // Returns true if `kernel` has been registered on `device_ordinal` in this
  // registry. Lock-free and thread-safe.
  bool Contains(int device_ordinal,
                const XLA_FFI_Kernel* absl_nullable kernel) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Scoped KernelRegistry FFI extension for Prepare and Initialize.
class KernelRegistrationExtension {
 public:
  KernelRegistrationExtension(
      FfiKernelRegistry* absl_nonnull registry,
      stream_executor::StreamExecutor* absl_nonnull executor);

  KernelRegistrationExtension(const KernelRegistrationExtension&) = delete;
  KernelRegistrationExtension& operator=(const KernelRegistrationExtension&) =
      delete;

  const XLA_FFI_KernelRegistry_Extension& raw() const { return extension_; }
  const XLA_FFI_Extension* base() const { return &extension_.extension_base; }

 private:
  static absl::Status RegisterImpl(XLA_FFI_KernelRegistry_Register_Args* args);
  static XLA_FFI_Error* Register(XLA_FFI_KernelRegistry_Register_Args* args);

  struct State {
    FfiKernelRegistry* registry;
    stream_executor::StreamExecutor* executor;
  };
  State state_;
  XLA_FFI_KernelRegistry_Extension extension_;
};

// KernelRegistry FFI extension for Execute. Borrows `stream` for the call.
XLA_FFI_KernelRegistry_Extension MakeKernelLaunchExtension(
    stream_executor::Stream* absl_nonnull stream);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_FFI_KERNEL_REGISTRY_H_
