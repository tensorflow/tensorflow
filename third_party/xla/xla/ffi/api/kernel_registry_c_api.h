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

#ifndef XLA_FFI_API_KERNEL_REGISTRY_C_API_H_
#define XLA_FFI_API_KERNEL_REGISTRY_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// XLA FFI Kernel Registry API
// The Kernel Registry API lets FFI handlers register device kernels (PTX or
// CUBIN) with the XLA runtime during the Prepare or Initialize stages and
// launch them by handle during the Execute stage (or record them in the Record
// stage via RecordContext::CreateLaunch).
// Loading kernel modules upfront during initialization avoids driver lock
// contention and synchronization overhead on the hot execution path.
// NOTE: This API is still under development. Breaking changes may happen
// frequently.
//==============================================================================

//==============================================================================
// Version macros for the kernel registry extension.
//==============================================================================
#define XLA_FFI_Extension_KernelRegistry 130
#define XLA_FFI_Extension_KernelRegistry_MajorVersion 0
#define XLA_FFI_Extension_KernelRegistry_MinorVersion 1

//===----------------------------------------------------------------------===//
// Kernel Registry Extension API
//===----------------------------------------------------------------------===//

// Opaque handle to backend-specific kernel registry state.
// Used to maintain per-invocation registration and execution context between
// XLA and the client.
typedef struct XLA_FFI_KernelRegistryState XLA_FFI_KernelRegistryState;

// Arguments for registering a device kernel with the XLA runtime.
// `kernel_spec` describes the kernel name, binary payload (PTX or CUBIN), size,
// and format. `uses_pdl` is a flag (0 = off, non-zero = on) that says whether
// the kernel is launched with Programmatic Dependent Launch (PDL).
struct XLA_FFI_KernelRegistry_Register_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_KernelRegistryState* state;
  const XLA_FFI_KernelSpec* kernel_spec;
  int32_t uses_pdl;
  const XLA_FFI_Kernel* cached_kernel;  // in, optional
  const XLA_FFI_Kernel* kernel;         // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_KernelRegistry_Register_Args, kernel);

// Registers a device kernel (PTX or CUBIN) for the current device and returns
// an opaque `XLA_FFI_Kernel*` handle in `args->kernel`. Call on every Prepare
// or Initialize so each device loads the module; pass `cached_kernel` (from a
// prior call on the same executable) to skip binary hashing and map lookup.
// Valid stages: Prepare, Initialize.
// Lifetime:
// - `kernel_spec` (including `name` and `data`) is only read during the call
//   and does not need to outlive it; binary bytes are not re-checked when
//   `cached_kernel` is non-null.
// - The returned `XLA_FFI_Kernel*` is owned by the XLA executable and remains
//   valid until the executable is destroyed. Passing a handle from a destroyed
//   executable is undefined behavior.
// - Repeated registrations of the same kernel (same name, binary contents,
//   format, `uses_pdl`, and executable PDL setting) reuse one handle and load
//   the module at most once per device.
typedef XLA_FFI_Error* XLA_FFI_KernelRegistry_Register(
    XLA_FFI_KernelRegistry_Register_Args* args);

// Arguments for launching a previously registered `XLA_FFI_Kernel*` on the
// execution stream.
struct XLA_FFI_KernelRegistry_Launch_Args {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_KernelRegistryState* state;
  const XLA_FFI_Kernel* kernel;
  const XLA_FFI_KernelLaunchParams* launch_params;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_KernelRegistry_Launch_Args, launch_params);

// Launches a previously registered `XLA_FFI_Kernel*` on the current execution
// stream using `args->launch_params`. `launch_params->uses_pdl` must match and
// will be removed in the future.
// Valid stages: Execute.
// Lifetime:
// - `launch_params` and its `args` array are copied before `launch_kernel`
//   returns and do not need to outlive the call.
// - Device pointers passed in `launch_params->args` must remain valid until
//   kernel execution completes on the stream.
typedef XLA_FFI_Error* XLA_FFI_KernelRegistry_Launch(
    XLA_FFI_KernelRegistry_Launch_Args* args);

// API for registering and launching device kernels for FFI clients.
// Usage:
// 1. Bind `ffi::KernelRegistry` in the Initialize and Execute handlers:
// ```
// XLA_FFI_DEFINE_HANDLER(kMyInit, MyInit,
//                        ffi::Ffi::BindInitialize()
//                            .Ctx<ffi::KernelRegistry>()
//                            .Ctx<ffi::State<MyState>>());
//
// XLA_FFI_DEFINE_HANDLER(kMyExecute, MyExecute,
//                        ffi::Ffi::Bind()
//                            .Ctx<ffi::KernelRegistry>()
//                            .RemainingArgs()
//                            .Ctx<ffi::State<MyState>>());
// ```
// 2. Register the kernel during Initialize and launch it during Execute:
// ```
// struct MyState {
//   std::atomic<const XLA_FFI_Kernel*> kernel{nullptr};
// };
//
// absl::Status MyInit(ffi::InitializeKernelRegistry registry, MyState* state) {
//   const XLA_FFI_Kernel* cached =
//       state->kernel.load(std::memory_order_acquire);
//   ABSL_ASSIGN_OR_RETURN(
//       const XLA_FFI_Kernel* kernel,
//       registry.Register("my_kernel", kPtxData, kPtxSize,
//                         ffi::SourceFormat::kPtx, /*uses_pdl=*/false,
//                         cached));
//   if (cached == nullptr) {
//     state->kernel.store(kernel, std::memory_order_release);
//   }
//   return absl::OkStatus();
// }
//
// absl::Status MyExecute(ffi::ExecuteKernelRegistry registry,
//                        ffi::RemainingArgs args, MyState* state) {
//   return registry.Launch(state->kernel.load(std::memory_order_acquire),
//                          launch_dims, shared_mem_bytes, /*uses_pdl=*/false,
//                          kernel_args);
// }
// ```
typedef struct XLA_FFI_KernelRegistryApi {
  XLA_FFI_KernelRegistry_Register* register_kernel;
  XLA_FFI_KernelRegistry_Launch* launch_kernel;
} XLA_FFI_KernelRegistryApi;

// Main extension struct for the kernel registry API.
typedef struct XLA_FFI_KernelRegistry_Extension {
  XLA_FFI_Extension extension_base;
  XLA_FFI_KernelRegistryState* state;
  const XLA_FFI_KernelRegistryApi* api;
} XLA_FFI_KernelRegistry_Extension;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_KernelRegistry_Extension, api);

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_KERNEL_REGISTRY_C_API_H_
