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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_GROUP_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_GROUP_THUNK_H_

#include <memory>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace gpu {

// Collective group thunk fuses together a set of arbitrary collective
// operations into a single group call in order for them to be dispatched
// together. Implementation is backend-specific and might not be supported by
// all collective implementations.
class CollectiveGroupThunk : public Thunk {
 public:
  CollectiveGroupThunk(
      ThunkInfo thunk_info, Thunk::Kind kind,
      std::vector<std::unique_ptr<Thunk>> thunks,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events =
          std::make_shared<CollectiveThunk::AsyncEvents>());
  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status ExecuteOnStream(const Thunk::ExecuteParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  void ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const override;
  void ForAllThunksMutable(absl::FunctionRef<void(Thunk*)> fn) override;
  absl::Status TransformAllNestedThunks(
      absl::FunctionRef<
          absl::StatusOr<std::unique_ptr<Thunk>>(std::unique_ptr<Thunk>)>
          fn) override;

  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events() const {
    return async_events_;
  }

  static absl::StatusOr<std::unique_ptr<CollectiveGroupThunk>> FromProto(
      ThunkInfo thunk_info, const CollectiveGroupThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      CollectiveThunk::AsyncEventsMap& async_events_map,
      const Deserializer& deserializer);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  ThunkSequence thunks_;
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_GROUP_THUNK_H_
