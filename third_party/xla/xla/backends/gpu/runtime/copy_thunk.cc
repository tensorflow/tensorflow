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

#include "xla/backends/gpu/runtime/copy_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::Status CopyThunk::AsyncEvents::Emplace(se::StreamExecutor* executor,
                                             int64_t instr_id,
                                             std::unique_ptr<se::Event> event) {
  Key key = {executor, instr_id};
  absl::MutexLock lock(&mutex_);
  VLOG(3) << "Emplace event " << event.get();
  if (auto [it, inserted] = events_.try_emplace(key, std::move(event));
      inserted) {
    return absl::OkStatus();
  }
  return absl::InternalError("Async copy event already exists!");
}

absl::StatusOr<std::unique_ptr<se::Event>> CopyThunk::AsyncEvents::Extract(
    se::StreamExecutor* executor, int64_t instr_id) {
  Key key = {executor, instr_id};
  absl::MutexLock lock(mutex_);
  if (auto event = events_.extract(key)) {
    VLOG(3) << "Extract event " << event.mapped().get();
    return std::move(event.mapped());
  }
  return absl::InternalError("Async copy event was not found!");
}

CopyThunk::CopyThunk(ThunkInfo thunk_info, const ShapedSlice& source_buffer,
                     const ShapedSlice& destination_buffer, int64_t mem_size)
    : Thunk(Kind::kCopy, std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {
  CHECK_EQ(ShapeUtil::ByteSizeOfElements(source_buffer_.shape),
           ShapeUtil::ByteSizeOfElements(destination_buffer_.shape));

  CHECK_GE(source_buffer_.slice.size(), mem_size);
  CHECK_GE(destination_buffer_.slice.size(), mem_size);
}

absl::Status CopyThunk::ExecuteOnStream(const ExecuteParams& params) {
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> CopyThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CopyThunkProto* copy_thunk_proto = proto.mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source_buffer_.ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination_buffer_.ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
}

absl::StatusOr<std::unique_ptr<CopyThunk>> CopyThunk::FromProto(
    ThunkInfo thunk_info, const CopyThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      ShapedSlice src_slice,
      ShapedSlice::FromProto(thunk_proto.source_buffer(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(ShapedSlice dst_slice,
                      ShapedSlice::FromProto(thunk_proto.destination_buffer(),
                                             buffer_allocations));
  if (ShapeUtil::ByteSizeOfElements(src_slice.shape) !=
      ShapeUtil::ByteSizeOfElements(dst_slice.shape)) {
    return absl::FailedPreconditionError(
        "DeviceToDeviceCopyThunkProto with incompatible shapes.");
  }

  return std::make_unique<CopyThunk>(std::move(thunk_info), src_slice,
                                     dst_slice, thunk_proto.mem_size());
}

}  // namespace gpu
}  // namespace xla
