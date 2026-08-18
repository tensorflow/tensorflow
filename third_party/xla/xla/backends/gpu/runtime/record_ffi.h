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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RECORD_FFI_H_
#define XLA_BACKENDS_GPU_RUNTIME_RECORD_FFI_H_

#include <vector>

#include "xla/ffi/api/record_c_api.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream_executor.h"

// Concrete definition of the opaque type declared in the FFI C API.
struct XLA_FFI_RecordContext {
  stream_executor::CommandBuffer* command_buffer;
  stream_executor::StreamExecutor* executor;
  std::vector<const stream_executor::CommandBuffer::Command*> dependencies;
  bool use_pdl = false;
  bool stream_capture_requested = false;
};

namespace xla::gpu {

// Returns the singleton GPU implementation of the XLA FFI Record API.
const XLA_FFI_RecordApi* GetXlaFfiRecordApi();

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_RECORD_FFI_H_
