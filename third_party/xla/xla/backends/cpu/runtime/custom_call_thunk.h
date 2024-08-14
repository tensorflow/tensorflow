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

#ifndef XLA_BACKENDS_CPU_RUNTIME_CUSTOM_CALL_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_CUSTOM_CALL_THUNK_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/ffi/call_frame.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Handles XLA custom calls.
class CustomCallThunk final : public Thunk {
 public:
  // Buffer allocation slices and shapes to fill FFI arguments.
  struct OpBuffers {
    std::vector<BufferAllocation::Slice> arguments_buffers;
    std::vector<Shape> arguments_shapes;

    std::vector<BufferAllocation::Slice> results_buffers;
    std::vector<Shape> results_shapes;
    bool is_tuple_result;
  };

  static absl::StatusOr<std::unique_ptr<CustomCallThunk>> Create(
      Info info, absl::string_view target_name, OpBuffers op_buffers,
      absl::string_view backend_config, CustomCallApiVersion api_version);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 private:
  CustomCallThunk(Info info, absl::string_view target_name,
                  OpBuffers op_buffers, CustomCallApiVersion api_version,
                  absl::string_view backend_config,
                  std::optional<ffi::CallFrame> call_frame);

  // Handles typed-FFI custom calls (API v4).
  tsl::AsyncValueRef<ExecuteEvent> CallTypedFFI(const ExecuteParams& params);

  // Handles legacy, untyped custom calls (API v1-v3).
  tsl::AsyncValueRef<ExecuteEvent> CallUntypedAPI(const ExecuteParams& params);

  // Function signature for legacy untyped API.
  using CustomCallTarget = std::function<void(void*, const void**, const char*,
                                              size_t, XlaCustomCallStatus*)>;

  std::string target_name_;
  OpBuffers op_buffers_;
  CustomCallApiVersion api_version_;
  std::string backend_config_;
  std::optional<ffi::CallFrame> call_frame_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_CUSTOM_CALL_THUNK_H_
