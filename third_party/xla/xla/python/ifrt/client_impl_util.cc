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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/bundle.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt/value_util.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

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

absl::StatusOr<std::vector<ArrayRef>> ClientMakeArraysFromHostBufferShards(
    Client* client,
    absl::Span<Client::MakeArraysFromHostBufferShardsSpec> specs,
    Client::HostBufferSemantics semantics) {
  for (int i = 1; i < specs.size(); ++i) {
    const Client::MakeArraysFromHostBufferShardsSpec& spec = specs[i];
    if (specs[0].array_spec.sharding->devices() !=
        spec.array_spec.sharding->devices()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "All arrays in MakeArraysFromHostBufferShards must have the "
          "same device list, but got ",
          specs[0].array_spec.sharding->devices(), " vs. ",
          spec.array_spec.sharding->devices()));
    }
    if (specs[0].array_spec.sharding->memory_kind() !=
        spec.array_spec.sharding->memory_kind()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "All arrays in MakeArraysFromHostBufferShards must have the "
          "same memory kind, but got ",
          specs[0].array_spec.sharding->memory_kind(), " vs. ",
          spec.array_spec.sharding->memory_kind()));
    }
  }

  std::vector<ArrayRef> arrays;
  arrays.reserve(specs.size());
  for (Client::MakeArraysFromHostBufferShardsSpec& spec : specs) {
    std::optional<xla::ifrt::Shape> shard_shape =
        TryToGetShardShape(spec.array_spec);

    if (CanUseMakeArrayFromHostBuffer(spec)) {
      // Fast-path for fully replicated arrays. Assumes that
      // `MakeArrayFromHostBuffer` can handle fully replicated array creation.
      auto& [addressable_shard_indices, host_buffer] = spec.buffers.front();
      ABSL_RETURN_IF_ERROR(CheckHostBuffer(spec, host_buffer, shard_shape));

      ABSL_ASSIGN_OR_RETURN(
          ArrayRef array,
          client->MakeArrayFromHostBuffer(
              host_buffer.data, host_buffer.dtype, std::move(host_buffer.shape),
              host_buffer.byte_strides, std::move(spec.array_spec.sharding),
              /*layout=*/nullptr, semantics, std::move(host_buffer.on_done)));
      arrays.push_back(std::move(array));
      continue;
    }

    absl::Span<xla::ifrt::Device* const> addressable_devices =
        spec.array_spec.sharding->devices()->AddressableDeviceList()->devices();

    std::vector<ArrayRef> addressable_shards;
    addressable_shards.resize(addressable_devices.size());
    int64_t num_processed_shards = 0;

    // Note that `host_buffer` is const reference. We cannot move any member
    // from it because the same instance may be used multiple times if the same
    // index domain shows up in `addressable_index_domains` multiple times.
    for (const auto& [addressable_shard_indices, host_buffer] : spec.buffers) {
      ABSL_RETURN_IF_ERROR(CheckHostBuffer(spec, host_buffer, shard_shape));

      std::function<void()> on_done_with_host_buffer_per_device;
      if (host_buffer.on_done != nullptr) {
        auto shared_on_done = std::shared_ptr<void>(
            nullptr,
            [on_done = std::move(host_buffer.on_done)](void*) { on_done(); });
        on_done_with_host_buffer_per_device =
            [shared_on_done = std::move(shared_on_done)]() mutable {
              shared_on_done.reset();
            };
      }

      for (int64_t addressable_shard_index : addressable_shard_indices) {
        if (addressable_shard_index < 0 ||
            addressable_shard_index >= addressable_devices.size()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Invalid addressable shard index: ", addressable_shard_index,
              "; expected: [0, ", addressable_devices.size(), ")"));
        }
        ArrayRef& shard = addressable_shards[addressable_shard_index];
        if (shard != nullptr) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Duplicate addressable shard index: ", addressable_shard_index));
        }
        auto sharding = xla::ifrt::SingleDeviceSharding::Create(
            addressable_devices[addressable_shard_index],
            spec.array_spec.sharding->memory_kind());
        LayoutRef layout;
        if (spec.array_spec.layout != nullptr) {
          layout = PjRtLayout::Create(spec.array_spec.layout);  // NOLINT
        }
        ABSL_ASSIGN_OR_RETURN(
            shard, client->MakeArrayFromHostBuffer(
                       host_buffer.data, host_buffer.dtype, host_buffer.shape,
                       host_buffer.byte_strides, std::move(sharding),
                       std::move(layout), semantics,
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

    ABSL_ASSIGN_OR_RETURN(
        ArrayRef array,
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

absl::StatusOr<std::vector<tsl::Future<>>> ClientCopyArraysToHostBufferShards(
    Client* client, absl::Span<Client::CopyArraysToHostBufferShardsSpec> specs,
    ArrayCopySemantics semantics) {
  for (int i = 1; i < specs.size(); ++i) {
    if (specs[0].array != nullptr && specs[i].array != nullptr &&
        specs[0].array->sharding().devices() !=
            specs[i].array->sharding().devices()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "All arrays in CopyArraysToHostBufferShards must have the "
          "same device list, but got ",
          specs[0].array->sharding().devices(), " vs. ",
          specs[i].array->sharding().devices()));
    }
  }

  std::vector<tsl::Future<>> result;
  result.reserve(specs.size());
  for (Client::CopyArraysToHostBufferShardsSpec& spec : specs) {
    if (spec.buffers.empty()) {  // Nothing to copy.
      result.push_back(absl::OkStatus());
      continue;
    }
    if (spec.array == nullptr) {
      return absl::InvalidArgumentError(
          "CopyArraysToHostBufferShards called with a null array.");
    }
    if (!spec.array->sharding().devices()->IsFullyAddressable()) {
      return absl::InvalidArgumentError(
          "CopyArraysToHostBufferShards called with an array with some "
          "non-addressable devices.");
    }
    // Split the array into single-device arrays.
    ABSL_ASSIGN_OR_RETURN(std::vector<ArrayRef> single_device_arrays,
                     spec.array->DisassembleIntoSingleDeviceArrays(
                         semantics, SingleDeviceShardSemantics::kAllShards));

    absl::InlinedVector<tsl::Future<>, 4> buffer_futures;
    buffer_futures.reserve(spec.buffers.size());
    for (auto& buffer : spec.buffers) {
      Client::CopyArraysToHostBufferShardsSpec::ShardIndices& shard_indices =
          buffer.first;
      Client::MutableHostBuffer& host_buffer = buffer.second;
      if (shard_indices.empty()) {
        return absl::InvalidArgumentError(
            "No source shard indices specified for a host buffer in "
            "CopyArraysToHostBufferShards.");
      }
      // If multiple array source shards are specified, pick the first one to
      // copy from.
      const int64_t shard_idx = shard_indices.front();
      if (shard_idx < 0 || shard_idx >= single_device_arrays.size()) {
        return absl::OutOfRangeError(
            absl::StrCat("Shard index ", shard_idx, " out of range [0, ",
                         single_device_arrays.size(), ")"));
      }
      std::optional<absl::Span<const int64_t>> byte_strides;
      if (host_buffer.byte_strides.has_value()) {
        byte_strides = absl::MakeConstSpan(*host_buffer.byte_strides);
      }
      buffer_futures.push_back(
          single_device_arrays[shard_idx]->CopyToHostBuffer(
              host_buffer.data, byte_strides, semantics));
    }
    CHECK(!buffer_futures.empty());
    result.push_back(buffer_futures.size() == 1
                         ? std::move(buffer_futures.front())
                         : tsl::JoinFutures(buffer_futures));
  }
  return result;
}

absl::StatusOr<LoadedExecutable::ExecuteBundleResult>
LoadedExecutableExecuteBundle(
    LoadedExecutable* executable, absl::Span<BundleRef> args,
    const ExecuteOptions& options,
    const std::optional<std::vector<int>>& outputs_bundle_slice_sizes) {
  int total_args_size = 0;
  for (const BundleRef& bundle : args) {
    total_args_size += bundle->num_values();
  }
  std::vector<ArrayRef> arg_arrays;
  arg_arrays.reserve(total_args_size);

  for (const BundleRef& bundle : args) {
    if (bundle->IsDeleted()) {
      return absl::FailedPreconditionError("Bundle is deleted or donated.");
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<ValueRef> values,
                     bundle->GetValues(ArrayCopySemantics::kReuseInput));
    for (const ValueRef& value : values) {
      if (auto* array = dyn_cast<Array>(value.get())) {
        arg_arrays.push_back(tsl::FormRef(array));
      } else {
        return absl::InvalidArgumentError(
            "`LoadedExecutableExecuteBundle` only supports `Bundle`s "
            "containing `Array`s.");
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(LoadedExecutable::ExecuteResult result,
                   executable->Execute(absl::MakeSpan(arg_arrays), options,
                                       /*devices=*/std::nullopt));

  Client* client = executable->client();

  std::vector<BundleRef> output_bundles;
  if (outputs_bundle_slice_sizes.has_value()) {
    int total_output_size = 0;
    for (int size : *outputs_bundle_slice_sizes) {
      total_output_size += size;
    }
    if (total_output_size != result.outputs.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Output bundle slice sizes do not match the number of "
                       "outputs: ",
                       total_output_size, " vs. ", result.outputs.size()));
    }
    output_bundles.reserve(outputs_bundle_slice_sizes->size());

    int offset = 0;
    for (int size : *outputs_bundle_slice_sizes) {
      std::vector<ValueRef> values;
      values.reserve(size);
      for (int j = 0; j < size; ++j) {
        values.push_back(std::move(result.outputs[offset]));
        ++offset;
      }

      ABSL_ASSIGN_OR_RETURN(BundleRef bundle,
                       client->Bundle(absl::MakeSpan(values),
                                      ArrayCopySemantics::kDonateInput));
      output_bundles.push_back(std::move(bundle));
    }
  } else {
    std::vector<ValueRef> output_values =
        ToValues(absl::MakeSpan(result.outputs));
    ABSL_ASSIGN_OR_RETURN(BundleRef output_bundle,
                     client->Bundle(absl::MakeSpan(output_values),
                                    ArrayCopySemantics::kDonateInput));
    output_bundles.push_back(std::move(output_bundle));
  }

  return LoadedExecutable::ExecuteBundleResult{
      /*outputs=*/std::move(output_bundles),
      /*status=*/std::move(result.status),
      /*cancellation_handle=*/std::move(result.cancellation_handle)};
}

}  // namespace ifrt
}  // namespace xla
