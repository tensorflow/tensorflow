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
  // Half-open interval with optional skips. Represents elements at offset
  // `[start, start + step, start + step * 2, ..., end)` (`end` is excluded).
  // Using the Python slice representation, it corresponds to
  // `[start:end:step]`. `start` and `end` must be zero or positive. `step`
  // must be positive (reverse iteration is disallowed for simplicity).
  struct Interval {
    int64_t start;
    int64_t end;
    int64_t step;

    bool operator==(const Interval& other) const {
      return start == other.start && end == other.end && step == other.step;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Interval& interval) {
      return H::combine(std::move(h), interval.start, interval.end,
                        interval.step);
    }

    std::string DebugString() const;
  };

  // Mapping of shards from an input array to an output array. The shards whose
  // index is chosen by `from` in `arrays[in_array]` will be used for the shards
  // whose index is chosen by `to` in `out_arrays[out_array]`. `from` and `to`
  // must contain the same number of `Interval`s, and each corresponding pair of
  // `Interval` from `from` and `to` must represent the same number of shards.
  struct Mapping {
    int in_array;
    int out_array;
    std::vector<Interval> from;
    std::vector<Interval> to;

    bool operator==(const Mapping& other) const {
      return in_array == other.in_array && out_array == other.out_array &&
             from == other.from && to == other.to;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Mapping& mapping) {
      return H::combine(std::move(h), mapping.in_array, mapping.out_array,
                        mapping.from, mapping.to);
    }

    std::string DebugString() const;
  };

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

  RemapPlan(std::vector<ArraySpec> input_specs,
            std::vector<ArraySpec> output_specs, std::vector<Mapping> mappings)
      : rep_(std::make_shared<Rep>(std::move(input_specs),
                                   std::move(output_specs),
                                   std::move(mappings))) {}

  ABSL_DEPRECATED(
      "Use the constructor that takes `input_devices_for_output_map` without "
      "`mappings` instead.")
  RemapPlan(std::vector<ArraySpec> input_specs,
            std::vector<ArraySpec> output_specs, std::vector<Mapping> mappings,
            absl::flat_hash_map<int, std::vector<InputDeviceRange>>
                input_devices_for_output_map)
      : rep_(std::make_shared<Rep>(std::move(input_specs),
                                   std::move(output_specs), std::move(mappings),
                                   std::move(input_devices_for_output_map))) {}

  // A convenience method that calculates `input_devices_for_output_map`,
  // creates a `RemapPlan`, and validates it. Users who create remap plans with
  // mappings once and reuse them should prefer this over constructors.
  static absl::StatusOr<RemapPlan> CreateOptimized(
      Client* client, std::vector<ArraySpec> input_specs,
      std::vector<ArraySpec> output_specs, std::vector<Mapping> mappings);

  absl::Span<const ArraySpec> input_specs() const { return rep_->input_specs; }

  absl::Span<const ArraySpec> output_specs() const {
    return rep_->output_specs;
  }

  absl::Span<const Mapping> mappings() const { return rep_->mappings; }

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
            rep_->mappings == other.rep_->mappings &&
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

    // Mappings.
    std::vector<Mapping> mappings;

    // If a key K is present in `input_devices_for_output_map` then it describes
    // all the inputs that contribute to the output with index K.
    //
    // The value lists all the input array indices that contribute to output K,
    // and for each input array I a device list containing all of the devices
    // that hold shards coming from I.
    //
    // If `mappings` is not empty, information must be consistent with the
    // information in `mappings`, i.e., `input_devices_for_output_map` must
    // duplicate, not replace, information in `mappings`.
    //
    // Entries in `input_devices_for_output_map` are strictly optional, but
    // their presence may allow some implementations to be more efficient since
    // the implementation need not construct the device lists at execution time.
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
        std::vector<Mapping> mappings)
        : input_specs(std::move(input_specs)),
          output_specs(std::move(output_specs)),
          mappings(std::move(mappings)) {}

    Rep(std::vector<ArraySpec> input_specs, std::vector<ArraySpec> output_specs,
        std::vector<Mapping> mappings,
        absl::flat_hash_map<int, std::vector<InputDeviceRange>>
            input_devices_for_output_map)
        : input_specs(std::move(input_specs)),
          output_specs(std::move(output_specs)),
          mappings(std::move(mappings)),
          input_devices_for_output_map(
              std::move(input_devices_for_output_map)) {}

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
