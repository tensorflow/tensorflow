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

#ifndef XLA_FFI_API_RECORD_API_H_
#define XLA_FFI_API_RECORD_API_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <vector>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/record_c_api.h"  // IWYU pragma: export

// C++ wrapper for the XLA FFI Record API.
namespace xla::ffi {

enum class RecordAction {
  kCreate = XLA_FFI_RecordAction_Create,
  kUpdate = XLA_FFI_RecordAction_Update,
};

namespace internal {
template <typename ErrorPolicy>
class RecordContextBase;
}  // namespace internal

// Wrapper for accessing a pointers to opaque command pointers for recording.
// The container is capped to the maximum number of commands as specified
// during construction.
// This is a just a wrapper to mutate the underlying command vector.
// All arguments to the constructor are non-owning and will be modified in place
// by the wrapper.
class BoundedCommandVector {
 public:
  BoundedCommandVector(const XLA_FFI_Command** commands, int64_t* num_commands,
                       int64_t max_commands)
      : commands_(commands),
        num_commands_(num_commands),
        max_commands_(max_commands) {}
  // Read element by index
  const XLA_FFI_Command* operator[](size_t index) const {
    return commands_[index];
  }

  bool has_storage() const { return commands_ != nullptr; }

  int64_t size() const { return *num_commands_; }
  int64_t capacity() const { return max_commands_; }
  bool is_full() const { return !has_storage() || size() >= max_commands_; }

 private:
  template <typename T>
  friend class internal::RecordContextBase;

  bool push_back(const XLA_FFI_Command* command) {
    if (is_full()) {
      return false;
    }
    commands_[(*num_commands_)++] = command;
    return true;
  }

  const XLA_FFI_Command** commands_;
  int64_t* num_commands_;
  const int64_t max_commands_;
};

namespace internal {

// C++ wrapper for the XLA FFI Record extension API.
// Unified implementation for statically linked and dynamically linked FFI
// modules.
template <typename ErrorPolicy>
class RecordContextBase {
 public:
  using Status = typename ErrorPolicy::Status;
  template <typename T>
  using StatusOr = typename ErrorPolicy::template StatusOr<T>;

  RecordContextBase(const XLA_FFI_Api* api, const XLA_FFI_RecordFrame* frame)
      : api_(api), frame_(frame) {}

  RecordAction action() const {
    return static_cast<RecordAction>(frame_->action);
  }

  // Instantiate a user-facing command wrapper for convenience.
  BoundedCommandVector commands() const {
    return BoundedCommandVector(frame_->commands, frame_->num_commands,
                                frame_->max_commands);
  }

  // Creates a launch command for a kernel registered through the
  // KernelRegistry extension. `args` can be a span of `KernelArg` or of
  // `void* const`.
  template <typename KernelArgSpan = std::initializer_list<KernelArg>>
  StatusOr<const XLA_FFI_Command*> CreateLaunch(const XLA_FFI_Kernel* kernel,
                                                XLA_FFI_LaunchDims launch_dims,
                                                uint32_t shared_mem_bytes,
                                                const KernelArgSpan& args) {
    static_assert(
        !std::is_same_v<KernelArgSpan, bool>,
        "CreateLaunch(XLA_FFI_Kernel*, ...) does not take uses_pdl; PDL is "
        "fixed at registration time.");
    return CreateLaunch(kernel, launch_dims, shared_mem_bytes, args,
                        std::initializer_list<const XLA_FFI_Command*>{});
  }

  template <typename KernelArgSpan = std::initializer_list<KernelArg>,
            typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  StatusOr<const XLA_FFI_Command*> CreateLaunch(const XLA_FFI_Kernel* kernel,
                                                XLA_FFI_LaunchDims launch_dims,
                                                uint32_t shared_mem_bytes,
                                                const KernelArgSpan& args,
                                                const DepSpan& dependencies) {
    static_assert(
        !std::is_same_v<KernelArgSpan, bool>,
        "CreateLaunch(XLA_FFI_Kernel*, ...) does not take uses_pdl; PDL is "
        "fixed at registration time.");
    if (kernel == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INVALID_ARGUMENT,
          "kernel must not be null during create_launch.");
    }
    XLA_FFI_RecordKernel record_kernel = {
        XLA_FFI_RecordKernel_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        kernel,
        /*function=*/nullptr,
        /*function_name=*/XLA_FFI_ByteSpan{nullptr, 0},
    };
    return CreateRecordKernelLaunch(record_kernel, launch_dims,
                                    shared_mem_bytes, /*uses_pdl=*/false, args,
                                    dependencies);
  }

  // TODO(b/578838426): Remove once internal runtime callers (e.g.
  // multi_gpu_barrier.cc) migrate off raw `void*` handles.
  // Deprecated, will be removed: creates a launch command for a device
  // function handle (e.g. CUfunction) loaded by the caller. Prefer registering
  // the kernel through the KernelRegistry extension.
  template <typename KernelArgSpan,
            typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  StatusOr<const XLA_FFI_Command*> CreateLaunch(
      std::string_view function_name, void* function,
      XLA_FFI_LaunchDims launch_dims, uint32_t shared_mem_bytes, bool uses_pdl,
      const KernelArgSpan& args, const DepSpan& dependencies = {}) {
    if (function == nullptr) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INVALID_ARGUMENT,
          "function must not be null during create_launch.");
    }
    XLA_FFI_RecordKernel record_kernel = {
        XLA_FFI_RecordKernel_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        /*kernel=*/nullptr,
        function,
        XLA_FFI_ByteSpan{function_name.data(), function_name.size()},
    };
    return CreateRecordKernelLaunch(record_kernel, launch_dims,
                                    shared_mem_bytes, uses_pdl, args,
                                    dependencies);
  }

  // Deprecated, will be removed: same as the overload above, kept so callers
  // that still pass `SourceFormat::kFunctionPtr` keep compiling.
  template <typename KernelArgSpan,
            typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  StatusOr<const XLA_FFI_Command*> CreateLaunch(
      std::string_view kernel_name, const void* kernel_data, size_t kernel_size,
      SourceFormat format, XLA_FFI_LaunchDims launch_dims,
      uint32_t shared_mem_bytes, bool uses_pdl, const KernelArgSpan& args,
      const DepSpan& dependencies = {}) {
    if (format != SourceFormat::kFunctionPtr || kernel_size != 0) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INVALID_ARGUMENT,
          "create_launch only accepts kFunctionPtr with kernel_size 0.");
    }
    // The FFI API passes the handle as const void*, but driver function
    // handles (e.g. CUfunction) are non-const opaque pointers.
    // NOLINTNEXTLINE(*-readability-const-cast): see comment above.
    return CreateLaunch(kernel_name, const_cast<void*>(kernel_data),
                        launch_dims, shared_mem_bytes, uses_pdl, args,
                        dependencies);
  }

  template <typename KernelArgSpan>
  Status UpdateLaunch(const XLA_FFI_Command* command,
                      const KernelArgSpan& args) {
    std::vector<XLA_FFI_KernelArg> raw_args = ConvertKernelArgs(args);
    XLA_FFI_KernelArgs ffi_args{raw_args.data(),
                                static_cast<int64_t>(raw_args.size())};
    XLA_FFI_Error* err =
        frame_->api->update_launch(frame_->record_ctx, command, &ffi_args);
    if (err) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

  template <typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  StatusOr<const XLA_FFI_Command*> CreateEmptyCommand(
      DepSpan dependencies = {}) {
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err = frame_->api->create_empty_command(
        frame_->record_ctx, std::data(dependencies), std::size(dependencies),
        &out_command);
    if (err) {
      return ErrorPolicy::TakeError(api_, err);
    }
    if (!out_command) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INTERNAL,
          "out_command is null but no error was returned during "
          "create_empty_command.");
    }
    if (!commands().push_back(out_command)) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_RESOURCE_EXHAUSTED,
          "Maximum number of commands that can be recorded per FFI::Record "
          "call has been reached.");
    }
    return out_command;
  }

  Status RequestStreamCapture() {
    XLA_FFI_Error* err =
        frame_->api->request_stream_capture(frame_->record_ctx);
    if (err) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

  template <typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  StatusOr<const XLA_FFI_Command*> CreateMemcpyD2D(void* dst, void* src,
                                                   size_t size,
                                                   DepSpan dependencies = {}) {
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err = frame_->api->create_memcpy_d2d(
        frame_->record_ctx, dst, src, size, std::data(dependencies),
        std::size(dependencies), &out_command);

    if (err) {
      return ErrorPolicy::TakeError(api_, err);
    }
    if (!out_command) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INTERNAL,
          "out_command is null but no error was returned during "
          "create_memcpy_d2d.");
    }
    if (!commands().push_back(out_command)) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_RESOURCE_EXHAUSTED,
          "Maximum number of commands that can be recorded per FFI::Record "
          "call has been reached.");
    }
    return out_command;
  }

  Status UpdateMemcpyD2D(const XLA_FFI_Command* command, void* dst, void* src,
                         size_t size) {
    XLA_FFI_Error* err = frame_->api->update_memcpy_d2d(
        frame_->record_ctx, command, dst, src, size);
    if (err) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

 private:
  template <typename KernelArgSpan, typename DepSpan>
  StatusOr<const XLA_FFI_Command*> CreateRecordKernelLaunch(
      const XLA_FFI_RecordKernel& record_kernel, XLA_FFI_LaunchDims launch_dims,
      uint32_t shared_mem_bytes, bool uses_pdl, const KernelArgSpan& args,
      const DepSpan& dependencies) {
    // Check for space first, so we never create a command we cannot track.
    BoundedCommandVector recorded = commands();
    if (recorded.is_full()) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_RESOURCE_EXHAUSTED,
          "Maximum number of commands that can be recorded per FFI::Record "
          "call has been reached.");
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
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err = frame_->api->create_launch(
        frame_->record_ctx, &record_kernel, &launch_params,
        std::data(dependencies), std::size(dependencies), &out_command);
    if (err) {
      return ErrorPolicy::TakeError(api_, err);
    }
    if (!out_command) {
      return ErrorPolicy::FromErrorCode(
          XLA_FFI_Error_Code_INTERNAL,
          "out_command is null but no error was returned during "
          "create_launch.");
    }
    recorded.push_back(out_command);
    return out_command;
  }

  const XLA_FFI_Api* api_;
  const XLA_FFI_RecordFrame* frame_;
};

// Common base struct for static and dynamic Record extensions.
// Defines traits for CtxDecoding<Extension<RecordExtension>>.
template <typename RecordContextT>
struct RecordExtensionBase {
  using Type = RecordContextT;
  using CExtension = XLA_FFI_Record_Extension;

  static constexpr auto kName = "RecordExtension";
  static constexpr int32_t kExtensionType = XLA_FFI_Extension_Record;
  static constexpr int32_t kMajorVersion =
      XLA_FFI_Extension_Record_MajorVersion;
  static constexpr int32_t kMinorVersion =
      XLA_FFI_Extension_Record_MinorVersion;

  // Builds a context from the extension.
  static RecordContextT Create(const XLA_FFI_Api* api, const CExtension* ext) {
    return RecordContextT(api, ext->record_frame);
  }
};

}  // namespace internal

inline XLA_FFI_Record_Extension BuildRecordCExtension(
    XLA_FFI_RecordFrame* frame) {
  XLA_FFI_Record_Extension ext;
  // context type is not important for constructing the extension since we are
  // not decoding the context here.
  ext.extension_base =
      MakeExtensionHeader<internal::RecordExtensionBase<void>>();
  ext.record_frame = frame;
  return ext;
}

}  // namespace xla::ffi

#endif  // XLA_FFI_API_RECORD_API_H_
