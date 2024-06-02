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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

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
  absl::MutexLock lock(&mu_);
  if (is_deleted_) {
    return absl::FailedPreconditionError("Array has already been deleted");
  }

  int num_shards = sharding_->devices().size();

  // For each single device array we are going to pre-make:
  //   (1) a Promise-Future pair for passing the buffers,
  //
  //   (2) a Per-shard buffer backing store and the corresponding
  //   on-done-with-buffer callback.
  //
  //   (3) shape and sharding by disassembing the source array's sharding.
  //
  // The Futures, the on-done-with-host-buffer callbacks, shapes and shardings
  // are used to make the arrays. The promises and the buffer backing stores are
  // passed onto the OnReady callback that populates them when the buffers of
  // the source array become ready.
  std::vector<Promise<Buffers>> buffer_promises;
  buffer_promises.reserve(num_shards);
  std::vector<Future<Buffers>> buffer_futures;
  buffer_futures.reserve(num_shards);

  struct PerShardBufferBackingStore {  // Data (strings) for a single shard.
    void CopyFrom(absl::Span<const absl::string_view> input_buffer) {
      strings.reserve(input_buffer.size());
      string_views.reserve(input_buffer.size());
      for (absl::string_view buf : input_buffer) {
        strings.push_back(std::string(buf.data(), buf.size()));
        string_views.push_back(strings.back());
      }
    }
    std::vector<std::string> strings;
    std::vector<absl::string_view> string_views;
  };
  std::vector<std::shared_ptr<PerShardBufferBackingStore>>
      per_shard_buffer_backing_stores;
  per_shard_buffer_backing_stores.reserve(num_shards);
  std::vector<OnDoneWithBuffer> on_done_with_buffer_callbacks;
  on_done_with_buffer_callbacks.reserve(num_shards);

  for (int i = 0; i < num_shards; ++i) {
    buffer_promises.push_back(Future<Buffers>::CreatePromise());
    buffer_futures.push_back(Future<Buffers>(buffer_promises.back()));

    auto backing_store = std::make_shared<PerShardBufferBackingStore>();
    per_shard_buffer_backing_stores.push_back(backing_store);
    on_done_with_buffer_callbacks.push_back(
        [backing_store = std::move(backing_store)]() {});
  }

  // Copy each of the per-shard data into the its per-shard buffer backing
  // store, make a Buffers object and set the corresponding promise.
  buffers_.OnReady([buffer_promises = std::move(buffer_promises),
                    per_shard_buffer_backing_stores =
                        std::move(per_shard_buffer_backing_stores)](
                       absl::StatusOr<Buffers> buffers) mutable {
    if (!buffers.ok()) {
      for (auto& promise : buffer_promises) {
        promise.Set(buffers.status());
      }
      per_shard_buffer_backing_stores.clear();
      return;
    }
    auto num_shards = buffers->size();
    for (int i = 0; i < num_shards; ++i) {
      per_shard_buffer_backing_stores[i]->CopyFrom((*buffers)[i]);
      Buffers buffers;
      buffers.push_back(per_shard_buffer_backing_stores[i]->string_views);
      buffer_promises[i].Set(std::move(buffers));
    }
  });

  // Make and return the individual single device arrays. These will become
  // ready when the this (source) array becomes ready and the callback we set up
  // above runs.
  TF_ASSIGN_OR_RETURN(auto shapes_and_shadings, sharding_->Disassemble(shape_));

  std::vector<tsl::RCReference<Array>> arrays;
  arrays.reserve(num_shards);
  for (int i = 0; i < num_shards; ++i) {
    TF_ASSIGN_OR_RETURN(auto array,
                        BasicStringArray::Create(
                            client_, std::move(shapes_and_shadings[i].first),
                            std::move(shapes_and_shadings[i].second),
                            std::move(buffer_futures[i]),
                            std::move(on_done_with_buffer_callbacks[i])));
    arrays.push_back(array);
  }
  return arrays;
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
