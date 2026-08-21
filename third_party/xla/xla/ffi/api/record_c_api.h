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

#ifndef XLA_FFI_API_RECORD_C_API_H_
#define XLA_FFI_API_RECORD_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/ffi/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// XLA FFI Record API
// Record API is used to record commands for FFI handlers.
// Instead of directly invoking the commands, the handler can record a sequence
// of commands along with its dependencies and execute them at a later time.
// Knowing the commands upront allows the XLA runtime to perform launch time
// optmizations for backends that support it.
// NOTE: This API is still under development. Breaking changes may happen
// frequently.
//==============================================================================

//==============================================================================
// Version macros for the record extension.
//==============================================================================
#define XLA_FFI_Extension_Record 128
#define XLA_FFI_Extension_Record_MajorVersion 0
#define XLA_FFI_Extension_Record_MinorVersion 1

//===----------------------------------------------------------------------===//
// Command Buffer Recording API (FFI Record)
//===----------------------------------------------------------------------===//

// Opaque structs for the record context and command.
// Contexts are used to maintain recording state between XLA and the client.
// Implementations are backend specific.
typedef struct XLA_FFI_RecordContext XLA_FFI_RecordContext;
// When a command is recorded using the record API, a command object is
// created. This command object can be used to specify dependencies for other
// commands. From the client's perspective, the command object is opaque.
typedef struct XLA_FFI_Command XLA_FFI_Command;

typedef enum XLA_FFI_RecordAction {
  XLA_FFI_RecordAction_Create = 0,
  XLA_FFI_RecordAction_Update = 1,
} XLA_FFI_RecordAction;

typedef enum XLA_FFI_SourceFormat {
  XLA_FFI_SourceFormat_PTX = 0,
  XLA_FFI_SourceFormat_CUBIN = 1,
} XLA_FFI_SourceFormat;

typedef struct XLA_FFI_Dim3 {
  int32_t x;
  int32_t y;
  int32_t z;
} XLA_FFI_Dim3;

typedef struct XLA_FFI_LaunchDims {
  XLA_FFI_Dim3 grid;
  XLA_FFI_Dim3 block;
  // Cluster dims are optional, 0, 0, 0 is considered as no cluster dims.
  XLA_FFI_Dim3 cluster;
} XLA_FFI_LaunchDims;

typedef enum XLA_FFI_KernelArgType {
  XLA_FFI_KernelArgType_DevicePtr = 0,
  XLA_FFI_KernelArgType_HostValue = 1,
} XLA_FFI_KernelArgType;

// A kernel argument is either a pointer to a device buffer or a host value.
// For a pointer to a device buffer, the size must be 0.
// For a host value, the size must be non-zero. In this case, XLA will pass the
// argument by value to the kernel.
typedef struct XLA_FFI_KernelArg {
  const void* arg_address;
  int64_t size;
  XLA_FFI_KernelArgType type;
} XLA_FFI_KernelArg;

typedef struct XLA_FFI_KernelArgs {
  const XLA_FFI_KernelArg* args;
  int64_t num_args;
} XLA_FFI_KernelArgs;

// API for recording kernel invocations for FFI clients.
// Usage:
// 1. The client registers a C++ function using standard XLA_FFI_DEFINE_HANDLER.
// ```
// XLA_FFI_DEFINE_HANDLER(kMyRecordHandler, MyRecordHandler,
//                        ffi::Ffi::BindRecord()
//                            .Ctx<ffi::Stream>()
//                            .Ctx<ffi::RecordContext>()
//                            .Ctx<ffi::RecordAction>()
//                            .Ctx<ffi::CommandVector>()
//                            .RemainingArgs()
//                            .RemainingRets()
//                            .Attrs()
//                            .Ctx<xla::ffi::State<MyState>>());
// ```
// 2. Then call the appropriate C++ API function to record/update commands.
// ```
//   absl::Status MyRecordHandler(
//     se::Stream* stream,
//     xla::ffi::RecordContext record_ctx,
//     xla::ffi::RecordAction action,
//     xla::ffi::CommandVector commands,
//     xla::ffi::RemainingArgs inputs,
//     xla::ffi::RemainingRets results,
//     xla::ffi::Attributes attrs,
//     MyState* state
//   ) {
//     if (CanRecord()) {
//       record_ctx.RequestStreamCapture();
//       return absl::OkStatus();
//     }
//     if (action == xla::ffi::RecordAction::kCreate) {
//       // Create a launch command.
//       record_ctx.CreateLaunch(...);
//     } else if (action == xla::ffi::RecordAction::kUpdate) {
//       // Update the launch command.
//       record_ctx.UpdateLaunch(...);
//     }
//     return absl::OkStatus();
//   }
// ```
typedef struct XLA_FFI_RecordApi {
  // Creates a launch command for a kernel with the given name, data and size.
  // kernel_data is the binary of the kernel, and format is the format of the
  // kernel data. Since recording implies creation of a graph, dependencies
  // argument is used to specify dependencies on other commands.
  // For eg:
  // XLA_FFI_Command* cmd1 = api->create_launch(...);
  // XLA_FFI_Command* cmd2 = api->create_launch(..., {&cmd1}, 1);
  // And so on.
  // Lifetime of the out_command is the same as the lifetime of the record
  // context.
  // Device pointers themselves need to be alive until the command is executed.
  // However, the host values will be copied by the XLA runtime and does not
  // need to be kept alive by the client. The args array itself can also be
  // destroyed after the call to create_launch since implementations must copy
  // this internally before returning.
  XLA_FFI_Error* (*create_launch)(
      XLA_FFI_RecordContext* ctx,
      const char* kernel_name,                     //
      const void* kernel_data,                     //
      int64_t kernel_size,                         //
      XLA_FFI_SourceFormat format,                 //
      XLA_FFI_LaunchDims launch_dims,              //
      uint32_t shared_mem_bytes,                   //
      const XLA_FFI_KernelArgs* args,              //
      const XLA_FFI_Command* const* dependencies,  //
      uint32_t num_dependencies,                   //
      const XLA_FFI_Command** out_command          //
  );

  // Updates the arguments for a launch command.
  XLA_FFI_Error* (*update_launch)(XLA_FFI_RecordContext* ctx,
                                  const XLA_FFI_Command* command,
                                  const XLA_FFI_KernelArgs* args);

  // Creates a memcpy D2D command.
  XLA_FFI_Error* (*create_memcpy_d2d)(
      XLA_FFI_RecordContext* ctx, void* dst, void* src, int64_t size,
      const XLA_FFI_Command* const* dependencies, uint32_t num_dependencies,
      const XLA_FFI_Command** out_command);

  // Updates the arguments for a memcpy D2D command.
  XLA_FFI_Error* (*update_memcpy_d2d)(XLA_FFI_RecordContext* ctx,
                                      const XLA_FFI_Command* command, void* dst,
                                      void* src, int64_t size);

  // Requests the XLA runtime to fallback to stream capture mode and capture
  // the current stream.
  XLA_FFI_Error* (*request_stream_capture)(XLA_FFI_RecordContext* ctx);

  // Used to join nodes into a single node that consumers can depend on.
  XLA_FFI_Error* (*create_empty_command)(
      XLA_FFI_RecordContext* ctx, const XLA_FFI_Command* const* dependencies,
      uint32_t num_dependencies, const XLA_FFI_Command** out_command);
} XLA_FFI_RecordApi;

// A record frame struct that wraps all record related data for the record
// extension. This is just convenience for builders so extensions can be
// constructed using:
// XLA_FFI_RecordFrame frame = { .api = &kNmyRecordApi, .record_ctx = ..., };
// XLA_FFI_Record_Extension ext = BuildRecordCExtension(&frame);
typedef struct XLA_FFI_RecordFrame {
  XLA_FFI_RecordContext* record_ctx;
  const XLA_FFI_RecordApi* api;
  XLA_FFI_RecordAction action;
  const XLA_FFI_Command** commands;
  int64_t* num_commands;
  int64_t max_commands;
} XLA_FFI_RecordFrame;

// Main extension struct for the record API.
typedef struct XLA_FFI_Record_Extension {
  XLA_FFI_Extension extension_base;
  XLA_FFI_RecordFrame* record_frame;
} XLA_FFI_Record_Extension;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Record_Extension, record_frame);

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_RECORD_C_API_H_
