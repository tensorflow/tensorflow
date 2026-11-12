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

#ifndef XLA_FFI_API_KERNEL_REGISTRY_API_H_
#define XLA_FFI_API_KERNEL_REGISTRY_API_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <type_traits>
#include <vector>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/kernel_registry_c_api.h"  // IWYU pragma: export

// C++ wrapper for the XLA FFI Kernel Registry API.
namespace xla::ffi {
namespace internal {

// C++ wrapper for the XLA FFI KernelRegistry extension API.
// Unified implementation for statically linked and dynamically linked FFI
// modules, parameterized by `ExecutionStage` so that `Register` is only
// available in `kPrepare`/`kInitialize` and `Launch` is only available in
// `kExecute`.
template <typename ErrorPolicy, ExecutionStage stage>
class KernelRegistryBase {
 public:
  using Status = typename ErrorPolicy::Status;
  template <typename T>
  using StatusOr = typename ErrorPolicy::template StatusOr<T>;

  KernelRegistryBase(const XLA_FFI_Api* api,
                     const XLA_FFI_KernelRegistry_Extension* extension)
      : api_(api), extension_(extension) {}

  // Registers a device kernel (PTX or CUBIN) for the current device and returns
  // an opaque `XLA_FFI_Kernel*` handle owned by the executable. Pass a
  // previously returned `cached_kernel` from the same executable to ensure it
  // is loaded on the current device without rehashing `data`.
  // Enabled only in `ExecutionStage::kPrepare` and
  // `ExecutionStage::kInitialize`.
  template <ExecutionStage s = stage,
            std::enable_if_t<s == ExecutionStage::kPrepare ||
                             s == ExecutionStage::kInitialize>* = nullptr>
  StatusOr<const XLA_FFI_Kernel*> Register(std::string_view name,
                                           const void* data, size_t size,
                                           SourceFormat format, bool uses_pdl) {
    return Register(name, data, size, format, uses_pdl,
                    /*cached_kernel=*/nullptr);
  }

  template <ExecutionStage s = stage,
            std::enable_if_t<s == ExecutionStage::kPrepare ||
                             s == ExecutionStage::kInitialize>* = nullptr>
  StatusOr<const XLA_FFI_Kernel*> Register(
      std::string_view name, const void* data, size_t size, SourceFormat format,
      bool uses_pdl, const XLA_FFI_Kernel* cached_kernel) {
    if (data == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INVALID_ARGUMENT,
          "Kernel data must not be null during kernel registration.");
    }
    if (extension_->api == nullptr ||
        extension_->api->register_kernel == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_UNIMPLEMENTED,
          "Kernel registration is not supported by this KernelRegistry "
          "extension.");
    }
    XLA_FFI_KernelSpec kernel_spec = {
        XLA_FFI_KernelSpec_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        XLA_FFI_ByteSpan{name.data(), name.size()},
        data,
        size,
        static_cast<XLA_FFI_SourceFormat>(format),
    };
    XLA_FFI_KernelRegistry_Register_Args args{};
    args.struct_size = XLA_FFI_KernelRegistry_Register_Args_STRUCT_SIZE;
    args.state = extension_->state;
    args.kernel_spec = &kernel_spec;
    args.uses_pdl = static_cast<int32_t>(uses_pdl);
    args.cached_kernel = cached_kernel;

    if (XLA_FFI_Error* err = extension_->api->register_kernel(&args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    if (args.kernel == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INTERNAL,
          "Kernel handle is null but no error was returned during "
          "kernel registration.");
    }
    return args.kernel;
  }

  // Launches a previously registered `XLA_FFI_Kernel*` on the execution stream.
  // Supports both `Span<const KernelArg>` and `Span<const void* const>` for
  // `args`. Enabled only in `ExecutionStage::kExecute`.
  template <typename KernelArgSpan = std::initializer_list<KernelArg>,
            ExecutionStage s = stage,
            std::enable_if_t<s == ExecutionStage::kExecute>* = nullptr>
  Status Launch(const XLA_FFI_Kernel* kernel, XLA_FFI_LaunchDims launch_dims,
                uint32_t shared_mem_bytes, bool uses_pdl,
                const KernelArgSpan& args = {}) {
    if (kernel == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INVALID_ARGUMENT,
          "Kernel handle must not be null during kernel launch.");
    }
    if (extension_->api == nullptr ||
        extension_->api->launch_kernel == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_UNIMPLEMENTED,
          "Kernel launch is not supported by this KernelRegistry extension.");
    }
    std::vector<XLA_FFI_KernelArg> raw_args = ConvertKernelArgs(args);
    XLA_FFI_KernelArgs ffi_args{raw_args.data(),
                                static_cast<int64_t>(raw_args.size())};
    XLA_FFI_KernelLaunchParams launch_params = {
        XLA_FFI_KernelLaunchParams_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        launch_dims,
        shared_mem_bytes,
        static_cast<int32_t>(uses_pdl),
        &ffi_args,
    };

    XLA_FFI_KernelRegistry_Launch_Args launch_args{};
    launch_args.struct_size = XLA_FFI_KernelRegistry_Launch_Args_STRUCT_SIZE;
    launch_args.state = extension_->state;
    launch_args.kernel = kernel;
    launch_args.launch_params = &launch_params;

    if (XLA_FFI_Error* err = extension_->api->launch_kernel(&launch_args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

 private:
  const XLA_FFI_Api* api_;
  const XLA_FFI_KernelRegistry_Extension* extension_;
};

// Restricts `KernelRegistryBase` binding to matching `kPrepare`, `kInitialize`,
// and `kExecute` stages.
template <typename ErrorPolicy, ExecutionStage registry_stage,
          ExecutionStage handler_stage>
struct IsStageCompatibleCtx<KernelRegistryBase<ErrorPolicy, registry_stage>,
                            handler_stage>
    : std::bool_constant<registry_stage == handler_stage &&
                         (handler_stage == ExecutionStage::kPrepare ||
                          handler_stage == ExecutionStage::kInitialize ||
                          handler_stage == ExecutionStage::kExecute)> {};

// Common base struct for static and dynamic KernelRegistry extensions.
// Defines traits for CtxDecoding<Extension<KernelRegistryExtensionBase<T>>>.
template <typename KernelRegistryT>
struct KernelRegistryExtensionBase {
  using Type = KernelRegistryT;
  using CExtension = XLA_FFI_KernelRegistry_Extension;

  static constexpr auto kName = "KernelRegistryExtension";
  static constexpr int32_t kExtensionType = XLA_FFI_Extension_KernelRegistry;
  static constexpr int32_t kMajorVersion =
      XLA_FFI_Extension_KernelRegistry_MajorVersion;
  static constexpr int32_t kMinorVersion =
      XLA_FFI_Extension_KernelRegistry_MinorVersion;

  static Type Create(const XLA_FFI_Api* api, const CExtension* ext) {
    return Type(api, ext);
  }
};

}  // namespace internal

template <typename ErrorPolicy, ExecutionStage stage>
struct CtxBinding<internal::KernelRegistryBase<ErrorPolicy, stage>> {
  using Ctx = internal::KernelRegistryBase<ErrorPolicy, stage>;
};

template <typename ErrorPolicy, ExecutionStage stage>
struct CtxDecoding<internal::KernelRegistryBase<ErrorPolicy, stage>>
    : CtxDecoding<Extension<internal::KernelRegistryExtensionBase<
          internal::KernelRegistryBase<ErrorPolicy, stage>>>> {};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_KERNEL_REGISTRY_API_H_
