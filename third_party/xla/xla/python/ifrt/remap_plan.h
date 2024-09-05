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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/remap_plan.pb.h"

namespace xla {
namespace ifrt {

// Remap plan that describes how the shards from input `Array`s are mapped to
// the shards of output `Array`s.
//
// * All input (or output) `Array`s must have the same dtype and per-shard
// shape.
// * An input shard can be used at most once.
// * Every output shard must have exactly one input shard mapped.
//
// There is no API-level constraint on their global shapes and shardings.
struct RemapPlan {
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

    std::string DebugString() const;
  };

  // Specification of inputs.
  std::vector<ArraySpec> input_specs;

  // Specification of outputs.
  std::vector<ArraySpec> output_specs;

  // Mappings.
  std::shared_ptr<std::vector<Mapping>> mappings;

  // Validates this plan against the requirements (see `RemapPlan` comment).
  // This is a slow operation. It should not be performed repeatedly.
  // Implementations of `Client::RemapArrays()` may bypass runtime checks on a
  // plan's validity, delegating the role to this method.
  absl::Status Validate() const;

  // Constructs `RemapPlan` from `RemapPlanProto`. Devices are looked up
  // using `lookup_device`. Device ids in the proto must be consistent with
  // the devices returned by `lookup_device`.
  static absl::StatusOr<RemapPlan> FromProto(
      DeviceList::LookupDeviceFunc lookup_device, const RemapPlanProto& proto);

  // Returns a `RemapPlanProto` representation.
  absl::StatusOr<RemapPlanProto> ToProto() const;

  std::string DebugString() const;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_REMAP_PLAN_H_
