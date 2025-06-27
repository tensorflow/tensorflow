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
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

// TODO(jmudigonda): Several BasicStringArray operations such as
// DisassembleIntoSingleDeviceArrays, Reshard, FullyReplicatedShard,
// CopyToHostBuffer and AssembleFromSingleDeviceArrays share a common pattern
// that waits for the source array(s) buffers to become ready and then copies
// the data into a new array's buffer. Factor out the common
// pattern into a helper function.

namespace xla {
namespace ifrt {

/////////////////////////////////////////////////////////////////////////////
//
// BasicStringArray
//

char BasicStringArray::ID = 0;

absl::StatusOr<tsl::RCReference<BasicStringArray>> BasicStringArray::Create(
    Client* client, Shape shape, ShardingRef sharding, Future<Buffers> buffers,
    OnDoneWithBuffer on_done_with_buffer) {
  if (!buffers.IsValid()) {
    return absl::InvalidArgumentError("Got buffers_ future is invalid");
  }

  auto buffers_promise = Future<Buffers>::CreatePromise();
  auto buffers_future = Future<Buffers>(buffers_promise);

  auto ready_promise = Future<>::CreatePromise();
  auto ready_future = Future<>(ready_promise);

  // Buffers when the become ready must be consistent with the sharding. For
  // instance, Buffers.size() (the number of per-shard spans of absl::Cords)
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

        const int64_t num_addressable_devices =
            sharding->devices()->AddressableDeviceList()->size();
        if (num_addressable_devices != (*buffers).size()) {
          auto error = absl::FailedPreconditionError(absl::StrCat(
              "Number of buffers: ", (*buffers).size(),
              " does not match the number of addressable devices in sharding: ",
              num_addressable_devices));
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
                                   ShardingRef sharding,
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

absl::StatusOr<std::vector<ArrayRef>>
BasicStringArray::DisassembleIntoSingleDeviceArrays(
    ArrayCopySemantics semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  DCHECK(this);
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      !sharding_->devices()->IsFullyAddressable()) {
    return InvalidArgument(
        "All shards are requested but the sharding has non-addressable "
        "devices: %v",
        *sharding_->devices());
  }

  absl::MutexLock lock(&mu_);
  if (is_deleted_) {
    return absl::FailedPreconditionError("Array has already been deleted");
  }

  TF_ASSIGN_OR_RETURN(
      auto shapes_and_shadings,
      sharding_->Disassemble(shape_, single_device_shard_semantics));
  const int num_shards = shapes_and_shadings.size();

  // For each single device array we are going to pre-make:
  //   (1) a Promise-Future pair for passing the buffers,
  //
  //   (2) a Per-shard data store and the corresponding on-done-with-buffer
  //   callback.
  //
  //   (3) shape and sharding by disassembing the source array's sharding.
  //
  // The Futures, the on-done-with-host-buffer callbacks, shapes and shardings
  // are used to make the arrays. The promises and the per-shard stores
  // are passed onto the OnReady callback that populates them when the buffers
  // of the source array become ready.
  std::vector<Promise<Buffers>> buffer_promises;
  buffer_promises.reserve(num_shards);
  std::vector<Future<Buffers>> buffer_futures;
  buffer_futures.reserve(num_shards);

  struct PerShardStringStore {  // Data (strings) for a single shard.
    void CopyFrom(absl::Span<const absl::Cord> input_buffer) {
      strings.reserve(input_buffer.size());
      for (const auto& input_string : input_buffer) {
        strings.push_back(input_string);
      }
    }
    std::vector<absl::Cord> strings;
  };

  std::vector<std::shared_ptr<PerShardStringStore>> per_shard_strings;
  per_shard_strings.reserve(num_shards);
  std::vector<OnDoneWithBuffer> on_done_with_buffer_callbacks;
  on_done_with_buffer_callbacks.reserve(num_shards);

  for (int i = 0; i < num_shards; ++i) {
    buffer_promises.push_back(Future<Buffers>::CreatePromise());
    buffer_futures.push_back(Future<Buffers>(buffer_promises.back()));

    auto current_shard_strings = std::make_shared<PerShardStringStore>();
    per_shard_strings.push_back(current_shard_strings);
    on_done_with_buffer_callbacks.push_back(
        [data = std::move(current_shard_strings)]() {});
  }

  // When the buffers become ready, copy each of the per-shard data into the
  // buffer of the corresponding single-device array.
  buffers_.OnReady([buffer_promises = std::move(buffer_promises),
                    per_shard_data = std::move(per_shard_strings)](
                       absl::StatusOr<Buffers> buffers) mutable {
    if (!buffers.ok()) {
      for (auto& promise : buffer_promises) {
        promise.Set(buffers.status());
      }
      per_shard_data.clear();
      return;
    }
    auto num_shards = buffers->size();
    for (int i = 0; i < num_shards; ++i) {
      per_shard_data[i]->CopyFrom((*buffers)[i]);
      Buffers buffers;
      buffers.push_back(absl::MakeConstSpan(per_shard_data[i]->strings));
      buffer_promises[i].Set(std::move(buffers));
    }
  });

  // Make and return the individual single device arrays. These will become
  // ready when the this (source) array becomes ready and the callback we set
  // up above runs.
  std::vector<ArrayRef> arrays;
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
  absl::MutexLock lock(&mu_);
  if (is_deleted_) {
    return Future<>(
        absl::FailedPreconditionError("Array has already been deleted"));
  }

  if (sharding_->devices()->size() != 1) {
    return Future<>(absl::InvalidArgumentError(absl::StrFormat(
        "CopyToHostBuffer only supports single device string arrays. This "
        "array has been sharded over %d devices.",
        sharding_->devices()->size())));
  }

  auto copy_completion_promise = Future<>::CreatePromise();
  auto copy_completion_future = Future<>(copy_completion_promise);

  buffers_.OnReady(
      [copy_completion_promise = std::move(copy_completion_promise),
       host_buffer = static_cast<absl::Cord*>(data)](
          absl::StatusOr<Buffers> input_buffers) mutable {
        if (!input_buffers.ok()) {
          copy_completion_promise.Set(input_buffers.status());
          return;
        }
        const absl::Span<const absl::Cord>& input_buffer = (*input_buffers)[0];
        for (int i = 0; i < input_buffer.size(); ++i) {
          host_buffer[i] = input_buffer[i];
        }
        copy_completion_promise.Set(absl::OkStatus());
      });
  return copy_completion_future;
}

absl::StatusOr<ArrayRef> BasicStringArray::Copy(
    std::optional<xla::ifrt::DeviceListRef> devices,
    std::optional<xla::ifrt::MemoryKind> memory_kind,
    ArrayCopySemantics semantics) {
  DCHECK(this);
  absl::MutexLock lock(&mu_);
  if (is_deleted_) {
    return absl::FailedPreconditionError("Array has already been deleted");
  }

  TF_ASSIGN_OR_RETURN(auto new_sharding, sharding().WithDeviceAssignment(
                                             std::move(devices), memory_kind));
  if (new_sharding->devices()->size() != sharding_->devices()->size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of devices in new sharding: ", new_sharding->devices()->size(),
        " does not match the number of devices in the current sharding: ",
        sharding_->devices()->size()));
  }

  struct StringStore {
    void AddShardData(absl::Span<const absl::Cord> input_buffer) {
      auto& shard_strings = strings.emplace_back();
      shard_strings.reserve(input_buffer.size());

      for (const auto& input_string : input_buffer) {
        shard_strings.push_back(input_string);
      }
    }
    std::vector<std::vector<absl::Cord>> strings;
  };

  auto string_store = std::make_shared<StringStore>();
  auto on_done_with_buffer = [string_store]() {};
  auto buffers_promise = Future<Buffers>::CreatePromise();
  auto buffers_future = Future<Buffers>(buffers_promise);

  auto copier = [string_store = std::move(string_store),
                 buffers_promise = std::move(buffers_promise)](
                    absl::StatusOr<Buffers> input_buffers) mutable {
    if (!input_buffers.ok()) {
      buffers_promise.Set(input_buffers.status());
      return;
    }
    Buffers buffers;
    buffers.reserve(input_buffers->size());
    for (auto& input_buffer : *input_buffers) {
      string_store->AddShardData(input_buffer);
      buffers.push_back(string_store->strings.back());
    }
    buffers_promise.Set(std::move(buffers));
  };
  buffers_.OnReady(std::move(copier));
  return BasicStringArray::Create(client_, shape_, std::move(new_sharding),
                                  std::move(buffers_future),
                                  std::move(on_done_with_buffer));
}

// Makes a single sharded BasicStringArray from the first shard.
absl::StatusOr<ArrayRef> BasicStringArray::FullyReplicatedShard(
    ArrayCopySemantics semantics) {
  absl::MutexLock lock(&mu_);
  if (is_deleted_) {
    return absl::FailedPreconditionError("Array has already been deleted");
  }

  // Consider a check here to make sure that the first shard contains the full
  // array - i.e., this indeed is a fully replicated array. Checking the shading
  // object may not be sufficient since currently IFRT users (e.g., JAX) can
  // sometimes use ConcreteSharding even for single device arrays, and
  // ConcreteSharding is currently hardcoded to be non-fully-replicated.

  struct StringStore {  // Data (strings) for a single shard.
    void CopyFrom(absl::Span<const absl::Cord> input_buffer) {
      strings.reserve(input_buffer.size());
      for (const auto& input_strings : input_buffer) {
        strings.push_back(input_strings);
      }
    }
    std::vector<absl::Cord> strings;
  };

  auto string_store = std::make_shared<StringStore>();
  auto on_done_with_buffer = [string_store]() {};
  auto buffers_promise = Future<Buffers>::CreatePromise();
  auto buffers_future = Future<Buffers>(buffers_promise);

  auto copier = [string_store = std::move(string_store),
                 buffers_promise = std::move(buffers_promise)](
                    absl::StatusOr<Buffers> input_buffers) mutable {
    if (!input_buffers.ok()) {
      buffers_promise.Set(input_buffers.status());
      return;
    }

    // No need to check the size of input_buffers. The consistency checks that
    // were run when the source array's buffers became ready would have
    // ensured that the input_buffers have at least one shard's worth of data.
    auto& input_buffer = (*input_buffers)[0];
    string_store->CopyFrom(input_buffer);

    Buffers buffers;
    buffers.push_back(string_store->strings);
    buffers_promise.Set(std::move(buffers));
  };
  buffers_.OnReady(std::move(copier));

  return BasicStringArray::Create(
      client_, shape_,
      SingleDeviceSharding::Create(sharding_->devices()->devices().front(),
                                   MemoryKind()),
      std::move(buffers_future), std::move(on_done_with_buffer));
}

absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>
BasicStringArray::pjrt_layout() const {
  return absl::UnimplementedError("String arrays do not support PjRtLayout");
}

std::string BasicStringArray::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "BasicStringArray(shape=%s; sharding=%s; layout=major-to-minor-dense)",
      shape_.DebugString(), sharding_->DebugString());
}

}  // namespace ifrt
}  // namespace xla
