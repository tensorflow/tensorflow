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

#ifndef XLA_PYTHON_IFRT_REMAP_PLAN_H_
#define XLA_PYTHON_IFRT_REMAP_PLAN_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/remap_plan.pb.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

class Client;

// Remap plan that describes how the shards from input `Array`s are mapped to
// the shards of output `Array`s.
//
// * All input (or output) `Array`s must have the same dtype and per-shard
// shape.
// * An input shard can be used at most once.
// * Every output shard must have exactly one input shard mapped.
//
// There is no API-level constraint on their global shapes and shardings.
class RemapPlan {
 public:
  // List of devices that are used as the source shards for a given input array
  // contributing to a given output array.
  struct InputDeviceRange {
    int in_array;
    DeviceListRef input_devices;

    bool operator==(const InputDeviceRange& other) const {
      return in_array == other.in_array && input_devices == other.input_devices;
    }

    template <typename H>
    friend H AbslHashValue(H h, const InputDeviceRange& input_device_range) {
      return H::combine(std::move(h), input_device_range.in_array,
                        input_device_range.input_devices);
    }
  };

  RemapPlan() : rep_(std::make_shared<Rep>()) {}

  RemapPlan(std::vector<ArraySpec> input_specs,
            std::vector<ArraySpec> output_specs,
            absl::flat_hash_map<int, std::vector<InputDeviceRange>>
                input_devices_for_output_map)
      : rep_(std::make_shared<Rep>(std::move(input_specs),
                                   std::move(output_specs),
                                   std::move(input_devices_for_output_map))) {}

  absl::Span<const ArraySpec> input_specs() const { return rep_->input_specs; }

  absl::Span<const ArraySpec> output_specs() const {
    return rep_->output_specs;
  }

  const absl::flat_hash_map<int, std::vector<InputDeviceRange>>&
  input_devices_for_output_map() const {
    return rep_->input_devices_for_output_map;
  }

  // Validates array-level consistency (dtype, shard shape, memory kind, and
  // layout) between input and output array pairs. The result will be cached
  // within the plan. `Client::RemapArrays` implementations should at least do
  // this validation.
  absl::Status ValidateArraySpecs() const;

  // Validates this plan against all requirements, including array-level
  // consistency (via `ValidateArraySpecs()`) and shard-level consistency (input
  // array shards are correctly mapped to output array shards). This is a slow
  // operation. The result will be cached within the plan. The users building a
  // complex `RemapPlan` are strongly encouraged to call this method.
  absl::Status Validate() const;

  // Constructs `RemapPlan` from `RemapPlanProto`. Devices are looked up
  // using `lookup_device`. Device ids in the proto must be consistent with
  // the devices returned by `lookup_device`.
  static absl::StatusOr<RemapPlan> FromProto(Client* client,
                                             const RemapPlanProto& proto);

  // Converts this plan to a protobuf.
  absl::Status ToProto(
      RemapPlanProto& proto,
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  // Returns a `RemapPlanProto` representation.
  absl::StatusOr<RemapPlanProto> ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const {
    RemapPlanProto proto;
    ABSL_RETURN_IF_ERROR(ToProto(proto, version));
    return proto;
  }

  std::string DebugString() const;

  // Checks whether the RemapPlan is valid with `semantics`.
  absl::Status CheckArrayCopySemantics(
      xla::ifrt::ArrayCopySemantics semantics) const;

  bool operator==(const RemapPlan& other) const {
    return rep_ == other.rep_ ||
           (absl::HashOf(*this) == absl::HashOf(other) &&
            rep_->input_specs == other.rep_->input_specs &&
            rep_->output_specs == other.rep_->output_specs &&
            rep_->input_devices_for_output_map ==
                other.rep_->input_devices_for_output_map);
  }

  template <typename H>
  friend H AbslHashValue(H h, const RemapPlan& plan) {
    plan.Hash(absl::HashState::Create(&h));
    return std::move(h);
  }

 private:
  void Hash(absl::HashState state) const;

  // Validates array-level consistency (dtype, shard shape, memory kind, and
  // layout) between input and output array pairs.
  absl::Status ValidateArraySpecsUncached() const;

  // Validates shard-level consistency (input array shards are correctly mapped
  // to output array shards).
  //
  // Prerequisite: `ValidateArraySpecsUncached()` must have succeeded on this
  // plan. This method assumes that array-level consistency, non-empty inputs,
  // and array index bounds are already validated.
  absl::Status ValidateArrayShardMappingsUncached() const;

  struct Rep {
    // Specification of inputs.
    std::vector<ArraySpec> input_specs;

    // Specification of outputs.
    std::vector<ArraySpec> output_specs;

    // If a key K is present in `input_devices_for_output_map` then it describes
    // all the inputs that contribute to the output with index K.
    //
    // The value lists all the input array indices that contribute to output K,
    // and for each input array I a device list containing all of the devices
    // that hold shards coming from I.
    absl::flat_hash_map<int, std::vector<InputDeviceRange>>
        input_devices_for_output_map;

    // Cached hash. 0 indicates the hash needs to be computed and cached. May be
    // written multiple times with the same non-zero value.
    static constexpr uint64_t kUnsetHash = 0;
    mutable std::atomic<uint64_t> hash = kUnsetHash;

    mutable absl::once_flag validate_array_specs_once;
    mutable absl::Status validate_array_specs_status;

    mutable absl::once_flag validate_array_shard_mappings_once;
    mutable absl::Status validate_array_shard_mappings_status;

    Rep() = default;

    Rep(std::vector<ArraySpec> input_specs, std::vector<ArraySpec> output_specs,
        absl::flat_hash_map<int, std::vector<InputDeviceRange>>
            input_devices_for_output_map)
        : input_specs(std::move(input_specs)),
          output_specs(std::move(output_specs)),
          input_devices_for_output_map(
              std::move(input_devices_for_output_map)) {}

    // `operator==` is more efficient with shallow copies.
    Rep(const Rep&) = delete;
    Rep& operator=(const Rep&) = delete;
  };

  absl_nonnull std::shared_ptr<const Rep> rep_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_REMAP_PLAN_H_
