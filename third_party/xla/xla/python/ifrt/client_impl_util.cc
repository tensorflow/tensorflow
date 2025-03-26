/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/client_impl_util.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

// Validates if the host buffer metadata is consistent with the array spec and a
// computed shard shape (if available).
absl::Status CheckHostBuffer(
    const Client::MakeArraysFromHostBufferShardsSpec& spec,
    const Client::HostBuffer& host_buffer,
    const std::optional<xla::ifrt::Shape>& shard_shape) {
  if (spec.array_spec.dtype != host_buffer.dtype) {
    return absl::InvalidArgumentError(
        absl::StrCat("Host buffer dtype does not match array spec dtype: ",
                     host_buffer.dtype, " vs. ", spec.array_spec.dtype));
  }
  if (shard_shape.has_value() && *shard_shape != host_buffer.shape) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Host buffer shape does not match shard shape: ", host_buffer.shape,
        " vs. ", *shard_shape));
  }
  return absl::OkStatus();
}

// Tries to get the shard shape from the sharding. Returns std::nullopt if the
// sharding does not support `GetShardShape()`.
std::optional<xla::ifrt::Shape> TryToGetShardShape(
    const ArraySpec& array_spec) {
  if (auto shard_shape_from_sharding =
          array_spec.sharding->GetShardShape(array_spec.shape);
      shard_shape_from_sharding.ok()) {
    return *std::move(shard_shape_from_sharding);
  }
  return std::nullopt;
}

// Checks if the given `MakeArraysFromHostBufferShardsSpec` can be handled by a
// single call to `MakeArrayFromHostBuffer`.
bool CanUseMakeArrayFromHostBuffer(
    const Client::MakeArraysFromHostBufferShardsSpec& spec) {
  if (spec.buffers.size() == 1) {
    const auto& [addressable_shard_indices, _] = spec.buffers.front();
    if (addressable_shard_indices.size() == spec.array_spec.sharding->devices()
                                                ->AddressableDeviceList()
                                                ->size() &&
        spec.array_spec.sharding->IsFullyReplicated()) {
      return true;
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<std::vector<tsl::RCReference<Array>>>
ClientMakeArraysFromHostBufferShards(
    Client* client,
    absl::Span<Client::MakeArraysFromHostBufferShardsSpec> specs,
    Client::HostBufferSemantics semantics) {
  std::vector<tsl::RCReference<Array>> arrays;
  arrays.reserve(specs.size());
  for (Client::MakeArraysFromHostBufferShardsSpec& spec : specs) {
    std::optional<xla::ifrt::Shape> shard_shape =
        TryToGetShardShape(spec.array_spec);

    if (CanUseMakeArrayFromHostBuffer(spec)) {
      // Fast-path for fully replicated arrays. Assumes that
      // `MakeArrayFromHostBuffer` can handle fully replicated array creation.
      auto& [addressable_shard_indices, host_buffer] = spec.buffers.front();
      TF_RETURN_IF_ERROR(CheckHostBuffer(spec, host_buffer, shard_shape));

      TF_ASSIGN_OR_RETURN(
          tsl::RCReference<Array> array,
          client->MakeArrayFromHostBuffer(
              host_buffer.data, host_buffer.dtype, std::move(host_buffer.shape),
              std::move(host_buffer.byte_strides),
              std::move(spec.array_spec.sharding), semantics,
              std::move(host_buffer.on_done)));
      arrays.push_back(std::move(array));
      continue;
    }

    absl::Span<xla::ifrt::Device* const> addressable_devices =
        spec.array_spec.sharding->devices()->AddressableDeviceList()->devices();

    std::vector<tsl::RCReference<Array>> addressable_shards;
    addressable_shards.resize(addressable_devices.size());
    int64_t num_processed_shards = 0;

    // Note that `host_buffer` is const reference. We cannot move any member
    // from it because the same instance may be used multiple times if the same
    // index domain shows up in `addressable_index_domains` multiple times.
    for (const auto& [addressable_shard_indices, host_buffer] : spec.buffers) {
      TF_RETURN_IF_ERROR(CheckHostBuffer(spec, host_buffer, shard_shape));

      std::function<void()> on_done_with_host_buffer_per_device;
      if (host_buffer.on_done != nullptr) {
        auto count = std::make_shared<std::atomic<int64_t>>(
            addressable_shard_indices.size());
        on_done_with_host_buffer_per_device =
            [on_done = std::move(host_buffer.on_done), count]() mutable {
              if (count->fetch_sub(1, std::memory_order_relaxed) == 1) {
                on_done();
              }
              on_done = nullptr;
            };
      }

      for (int64_t addressable_shard_index : addressable_shard_indices) {
        if (addressable_shard_index < 0 ||
            addressable_shard_index >= addressable_devices.size()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Invalid addressable shard index: ", addressable_shard_index,
              "; expected: [0, ", addressable_devices.size(), ")"));
        }
        tsl::RCReference<Array>& shard =
            addressable_shards[addressable_shard_index];
        if (shard != nullptr) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Duplicate addressable shard index: ", addressable_shard_index));
        }
        auto sharding = xla::ifrt::SingleDeviceSharding::Create(
            addressable_devices[addressable_shard_index],
            spec.array_spec.sharding->memory_kind());
        TF_ASSIGN_OR_RETURN(
            shard, client->MakeArrayFromHostBuffer(
                       host_buffer.data, host_buffer.dtype, host_buffer.shape,
                       host_buffer.byte_strides, std::move(sharding), semantics,
                       on_done_with_host_buffer_per_device));
      }
      num_processed_shards += addressable_shard_indices.size();
    }
    if (num_processed_shards != addressable_devices.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Number of processed shards does not match the number of addressable "
          "devices: ",
          num_processed_shards, " vs. ", addressable_devices.size()));
    }

    TF_ASSIGN_OR_RETURN(
        tsl::RCReference<Array> array,
        client->AssembleArrayFromSingleDeviceArrays(
            spec.array_spec.dtype, std::move(spec.array_spec.shape),
            std::move(spec.array_spec.sharding),
            absl::MakeSpan(addressable_shards),
            ArrayCopySemantics::kDonateInput,
            SingleDeviceShardSemantics::kAddressableShards));
    arrays.push_back(std::move(array));
  }
  return arrays;
}

}  // namespace ifrt
}  // namespace xla
