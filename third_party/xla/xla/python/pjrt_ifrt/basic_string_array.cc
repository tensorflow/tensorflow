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

#include "xla/python/pjrt_ifrt/basic_string_array.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char BasicStringArray::ID = 0;

absl::StatusOr<tsl::RCReference<BasicStringArray>> BasicStringArray::Create(
    Client* client, Shape shape, std::shared_ptr<const Sharding> sharding,
    Future<Buffers> buffers, OnDoneWithBuffer on_done_with_buffer) {
  if (!buffers.IsValid()) {
    return absl::InvalidArgumentError("Got buffers_ future is invalid");
  }

  auto buffers_promise = Future<Buffers>::CreatePromise();
  auto buffers_future = Future<Buffers>(buffers_promise);

  auto ready_promise = Future<>::CreatePromise();
  auto ready_future = Future<>(ready_promise);

  // Buffers when the become ready must be consistent with the sharding. For
  // instance, Buffers.size() (the number of per-shard spans of string_views)
  // and the devices in the sharding that was used to create an array must
  // match. If they do not, the array's ready future and buffers future should
  // become ready with an appropriate error status.

  auto buffer_validator =
      [buffers_promise = std::move(buffers_promise),
       ready_promise = std::move(ready_promise),
       sharding = sharding](absl::StatusOr<Buffers> buffers) mutable {
        if (!buffers.ok()) {
          buffers_promise.Set(buffers.status());
          ready_promise.Set(buffers.status());
          return;
        }

        if (sharding->devices().size() != (*buffers).size()) {
          auto error = absl::FailedPreconditionError(absl::StrCat(
              "Number of buffers: ", (*buffers).size(),
              " does not match the number of devices in sharding: ",
              sharding->devices().size()));
          buffers_promise.Set(error);
          ready_promise.Set(error);
          return;
        }

        buffers_promise.Set(std::move(buffers));
        ready_promise.Set(absl::OkStatus());
      };

  buffers.OnReady(std::move(buffer_validator));

  return tsl::MakeRef<BasicStringArray>(
      client, std::move(shape), std::move(sharding), std::move(buffers_future),
      std::move(ready_future), std::move(on_done_with_buffer));
}

BasicStringArray::BasicStringArray(Client* client, Shape shape,
                                   std::shared_ptr<const Sharding> sharding,
                                   Future<Buffers> buffers,
                                   Future<> ready_future,
                                   OnDoneWithBuffer on_done_with_buffer)
    : client_(client),
      shape_(std::move(shape)),
      sharding_(std::move(sharding)),
      buffers_(std::move(buffers)),
      ready_future_(std::move(ready_future)),
      on_done_with_buffer_(std::move(on_done_with_buffer)) {}

BasicStringArray::~BasicStringArray() { DeleteInternal(); }

Future<> BasicStringArray::Delete() {
  DeleteInternal();
  return Future<>(absl::OkStatus());
}

bool BasicStringArray::IsDeleted() const {
  absl::MutexLock lock(&mu_);
  return is_deleted_;
}

void BasicStringArray::DeleteInternal() {
  absl::MutexLock lock(&mu_);
  if (is_deleted_) {
    return;
  }
  if (on_done_with_buffer_) {
    std::move(on_done_with_buffer_)();
  }
  is_deleted_ = true;
}

Future<> BasicStringArray::GetReadyFuture() const {
  DCHECK(this);
  absl::MutexLock lock(&mu_);
  if (is_deleted_) {
    return Future<>(
        absl::FailedPreconditionError("Array has already been deleted"));
  }
  return ready_future_;
}

absl::StatusOr<std::vector<tsl::RCReference<Array>>>
BasicStringArray::DisassembleIntoSingleDeviceArrays(
    ArrayCopySemantics semantics) {
  DCHECK(this);
  return absl::UnimplementedError("Not implemented");
}

Future<> BasicStringArray::CopyToHostBuffer(
    void* data, std::optional<absl::Span<const int64_t>> byte_strides,
    ArrayCopySemantics semantics) {
  DCHECK(this);
  return Future<>(absl::UnimplementedError("Not implemented"));
}

absl::StatusOr<tsl::RCReference<Array>> BasicStringArray::Reshard(
    std::shared_ptr<const Sharding> new_sharding,
    ArrayCopySemantics semantics) {
  DCHECK(this);
  return absl::UnimplementedError("Not implemented");
}

absl::StatusOr<tsl::RCReference<Array>> BasicStringArray::FullyReplicatedShard(
    ArrayCopySemantics semantics) {
  // Make a single sharded BasicStringArray from the first shard.
  return absl::UnimplementedError("Not implemented");
}

absl::StatusOr<std::unique_ptr<PjRtLayout>> BasicStringArray::layout() const {
  return absl::UnimplementedError("Not implemented");
}

std::string BasicStringArray::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "BasicStringArray(shape=%s; sharding=%s; layout=major-to-minor-dense)",
      shape_.DebugString(), sharding_->DebugString());
}

}  // namespace ifrt
}  // namespace xla
