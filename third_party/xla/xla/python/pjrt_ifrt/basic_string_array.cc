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
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/status.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char BasicStringArray::ID = 0;

absl::StatusOr<tsl::RCReference<BasicStringArray>> BasicStringArray::Create(
    Client* client, Shape shape, std::shared_ptr<const Sharding> sharding,
    Future<Buffers> buffers, OnDoneWithBuffer on_done_with_buffer) {
  return tsl::MakeRef<BasicStringArray>(client, std::move(shape),
                                        std::move(sharding), std::move(buffers),
                                        std::move(on_done_with_buffer));
}

absl::StatusOr<tsl::RCReference<Array>> BasicStringArray::FullyReplicatedShard(
    ArrayCopySemantics semantics) {
  // Make a single sharded BasicStringArray from the first shard.
  return absl::UnimplementedError("Not implemented");
}

BasicStringArray::BasicStringArray(Client* client, Shape shape,
                                   std::shared_ptr<const Sharding> sharding,
                                   Future<Buffers> buffers,
                                   OnDoneWithBuffer on_done_with_buffer)
    : client_(client),
      shape_(std::move(shape)),
      sharding_(std::move(sharding)),
      buffers_(std::move(buffers)),
      on_done_with_buffer_(std::move(on_done_with_buffer)) {}

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

Future<> BasicStringArray::GetReadyFuture() const {
  DCHECK(this);
  return Future<>(absl::UnimplementedError("Not implemented"));
}

Future<> BasicStringArray::Delete() {
  DCHECK(this);
  return Future<>(absl::UnimplementedError("Not implemented"));
}

bool BasicStringArray::IsDeleted() const {
  DCHECK(this);
  return false;
}

std::string BasicStringArray::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "BasicStringArray(shape=%s; sharding=%s; layout=major-to-minor-dense)",
      shape_.DebugString(), sharding_->DebugString());
}

}  // namespace ifrt
}  // namespace xla
