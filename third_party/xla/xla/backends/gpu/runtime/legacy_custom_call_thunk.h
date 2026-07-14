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

#ifndef XLA_BACKENDS_GPU_RUNTIME_LEGACY_CUSTOM_CALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_LEGACY_CUSTOM_CALL_THUNK_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Thunk to run a legacy (non-FFI) GPU custom call.
//
// This thunk is DEPRECATED. All new custom calls should use the XLA FFI
// mechanism via CustomCallThunk instead. This class exists only to support
// legacy custom calls that have not yet migrated to FFI.
//
// For the FFI-based custom call thunk, see custom_call_thunk.h.
//
// Also implements TracedCommand so it can be recorded directly into command
// buffers; the default TracedCommand::Record() traces ExecuteOnStream() on the
// command-buffer trace stream.
class LegacyCustomCallThunk : public TracedCommand {
 public:
  using CustomCallTarget =
      std::function<void(stream_executor::Stream*, void**, const char*, size_t,
                         XlaCustomCallStatus*)>;

  // Creates a serializable legacy custom call thunk. The callback is resolved
  // using the legacy CustomCallTargetRegistry.
  static absl::StatusOr<std::unique_ptr<LegacyCustomCallThunk>> Create(
      ThunkInfo thunk_info, std::string target_name,
      std::vector<NullableShapedSlice> operands,
      std::vector<NullableShapedSlice> results, std::string opaque,
      CustomCallApiVersion api_version, absl::string_view platform_name);

  // Creates a legacy custom call thunk from a given call target. A thunk
  // created this way cannot be serialized to a proto. This overload is only
  // permitted for unit testing code.
  static absl::StatusOr<std::unique_ptr<LegacyCustomCallThunk>> Create(
      ThunkInfo thunk_info, std::string target_name,
      CustomCallTarget call_target, std::vector<NullableShapedSlice> operands,
      std::vector<NullableShapedSlice> results, std::string opaque);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const std::string& target_name() const { return target_name_; }
  CustomCallTarget call_target() const { return call_target_; }

  const std::vector<NullableShapedSlice>& operands() const { return operands_; }
  const std::vector<NullableShapedSlice>& results() const { return results_; }

  absl::string_view opaque() const { return opaque_; }

  BufferUses buffer_uses() const override {
    BufferUses res;
    res.reserve(operands_.size() + results_.size());
    for (const NullableShapedSlice& shaped_slice : operands_) {
      if (!shaped_slice.has_value()) {
        continue;
      }
      res.push_back(BufferUse::Read(shaped_slice->slice, shaped_slice->shape));
    }
    for (const NullableShapedSlice& shaped_slice : results_) {
      if (!shaped_slice.has_value()) {
        continue;
      }
      res.push_back(BufferUse::Write(shaped_slice->slice, shaped_slice->shape));
    }
    return res;
  }

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<LegacyCustomCallThunk>> FromProto(
      ThunkInfo thunk_info, const CustomCallThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      absl::string_view platform_name);

 private:
  LegacyCustomCallThunk(ThunkInfo thunk_info, std::string target_name,
                        std::vector<NullableShapedSlice> operands,
                        std::vector<NullableShapedSlice> results,
                        std::string opaque, CustomCallTarget call_target,
                        const std::optional<CustomCallApiVersion>& api_version);

  // API version of the custom call. If not set, the thunk was created from a
  // non-registered function pointer and cannot be serialized.
  std::optional<CustomCallApiVersion> api_version_;
  std::string target_name_;

  std::vector<NullableShapedSlice> operands_;
  std::vector<NullableShapedSlice> results_;

  CustomCallTarget call_target_;
  std::string opaque_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_LEGACY_CUSTOM_CALL_THUNK_H_
