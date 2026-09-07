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

#include "xla/python/ifrt/remap_plan.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/remap_plan.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/status_macros.h"

namespace xla {
namespace ifrt {

namespace {

// Deserializes `RemapPlanProto::MappingProto` into `RemapPlan::Mapping`.
absl::StatusOr<RemapPlan::Mapping> MappingFromProto(
    const RemapPlanProto::MappingProto& mapping_proto) {
  RemapPlan::Mapping mapping;

  mapping.in_array = mapping_proto.in_array();
  mapping.out_array = mapping_proto.out_array();

  const int64_t num_intervals = mapping_proto.from_start_size();
  TF_RET_CHECK(mapping_proto.from_end_size() == num_intervals);
  TF_RET_CHECK(mapping_proto.from_step_size() == num_intervals);
  TF_RET_CHECK(mapping_proto.to_start_size() == num_intervals);
  TF_RET_CHECK(mapping_proto.to_end_size() == num_intervals);
  TF_RET_CHECK(mapping_proto.to_step_size() == num_intervals);

  mapping.from.reserve(num_intervals);
  mapping.to.reserve(num_intervals);
  for (int64_t i = 0; i < num_intervals; ++i) {
    mapping.from.push_back(
        RemapPlan::Interval{/*start=*/mapping_proto.from_start(i),
                            /*end=*/mapping_proto.from_end(i),
                            /*step=*/mapping_proto.from_step(i)});
    mapping.to.push_back(
        RemapPlan::Interval{/*start=*/mapping_proto.to_start(i),
                            /*end=*/mapping_proto.to_end(i),
                            /*step=*/mapping_proto.to_step(i)});
  }
  return mapping;
}

// Serializes `RemapPlan::Mapping` into `RemapPlanProto::MappingProto`.
absl::Status MappingToProto(const RemapPlan::Mapping& mapping,
                            RemapPlanProto::MappingProto& proto) {
  TF_RET_CHECK(mapping.from.size() == mapping.to.size());

  proto.set_in_array(mapping.in_array);
  proto.set_out_array(mapping.out_array);

  const int64_t num_intervals = mapping.from.size();
  proto.mutable_from_start()->Reserve(num_intervals);
  proto.mutable_from_end()->Reserve(num_intervals);
  proto.mutable_from_step()->Reserve(num_intervals);
  proto.mutable_to_start()->Reserve(num_intervals);
  proto.mutable_to_end()->Reserve(num_intervals);
  proto.mutable_to_step()->Reserve(num_intervals);
  for (int64_t i = 0; i < mapping.from.size(); ++i) {
    proto.add_from_start(mapping.from[i].start);
    proto.add_from_end(mapping.from[i].end);
    proto.add_from_step(mapping.from[i].step);
    proto.add_to_start(mapping.to[i].start);
    proto.add_to_end(mapping.to[i].end);
    proto.add_to_step(mapping.to[i].step);
  }
  return absl::OkStatus();
}

absl::StatusOr<RemapPlan::InputDeviceRange> InputDeviceRangeFromProto(
    Client* client, const RemapPlanProto::InputDevices& proto) {
  RemapPlan::InputDeviceRange range;
  range.in_array = proto.in_array();
  ABSL_ASSIGN_OR_RETURN(range.input_devices,
                   DeviceList::FromProto(client, proto.device_list()));
  return range;
}

void InputDeviceToOutputToProto(
    SerDesVersion version, int out_array,
    absl::Span<const RemapPlan::InputDeviceRange> input_devices,
    RemapPlanProto::InputDevicesForOutput& proto) {
  proto.set_out_array(out_array);
  for (const RemapPlan::InputDeviceRange& input : input_devices) {
    RemapPlanProto::InputDevices* input_proto = proto.add_input_devices();
    input_proto->set_in_array(input.in_array);
    input.input_devices->ToProto(*input_proto->mutable_device_list(), version);
  }
}

// Checks if `interval` is in a valid range for the given number of shards.
absl::Status CheckRange(int64_t num_shards,
                        const RemapPlan::Interval& interval) {
  if (interval.start < 0 || interval.start > num_shards - 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "start must be in [0, %d], but is %d", num_shards - 1, interval.start));
  }
  if (interval.step <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("step must be positive, but is %d", interval.step));
  }
  if (interval.end < 0 || interval.end > num_shards + interval.step - 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "end must be in [0, %d] if step is %d, but is %d",
        num_shards + interval.step - 1, interval.step, interval.end));
  }
  // The `end` bound above is necessary but not sufficient: with a large `step`,
  // the last stepped index can exceed `num_shards` while still satisfying
  // `index < end`, which would lead to out-of-bounds indexing of the per-shard
  // buffers. Verify the last stepped index explicitly.
  if (interval.end > interval.start) {
    const int64_t last_index =
        interval.end - 1 - (interval.end - 1 - interval.start) % interval.step;
    if (last_index >= num_shards) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "interval addresses shard %d, which is out of range [0, %d)",
          last_index, num_shards));
    }
  }
  return absl::OkStatus();
}

// Returns the number of steps in `interval`.
int64_t GetNumberOfSteps(const RemapPlan::Interval& interval) {
  return (interval.end - interval.start + interval.step - 1) / interval.step;
}

bool CheckOneInputForOneOutput(const xla::ifrt::RemapPlan& plan) {
  const auto& mappings = plan.mappings();
  if (mappings.empty()) {
    for (const auto& [out_array, inputs] :
         // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
         plan.input_devices_for_output_map()) {
      int first_in_array = -1;
      for (const auto& input : inputs) {
        if (first_in_array == -1) {
          first_in_array = input.in_array;
        } else if (first_in_array != input.in_array) {
          return false;
        }
      }
    }
    return true;
  }

  absl::flat_hash_map<int, int> output_to_input;

  for (const auto& mapping : mappings) {
    int in_array = mapping.in_array;
    int out_array = mapping.out_array;

    const auto [it, inserted] = output_to_input.insert({out_array, in_array});
    if (!inserted && it->second != in_array) {
      return false;
    }
  }

  return true;
}

absl::StatusOr<DeviceListRef> ComputeDeviceListFromIntervals(
    Client* client, const DeviceListRef& device_list, int64_t count,
    absl::Span<const RemapPlan::Interval> intervals) {
  TF_RET_CHECK(count >= 0);
  std::vector<Device*> devices;
  devices.reserve(count);
  for (const RemapPlan::Interval& interval : intervals) {
    if (interval.step <= 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("step must be positive, but is %d", interval.step));
    }
    int64_t index = interval.start;
    while (index < interval.end) {
      TF_RET_CHECK(index >= 0 && index < device_list->size());
      devices.push_back(device_list->devices()[index]);
      index += interval.step;
    }
  }
  return client->MakeDeviceList(devices);
}

}  // namespace

std::string RemapPlan::Interval::DebugString() const {
  return absl::StrCat("[", start, ":", end, ":", step, "]");
}

std::string RemapPlan::Mapping::DebugString() const {
  auto format_intervals = [](absl::Span<const RemapPlan::Interval> intervals) {
    return absl::StrCat(
        "[",
        absl::StrJoin(
            intervals, ",",
            [](std::string* out, const RemapPlan::Interval& interval) {
              absl::StrAppend(out, interval.DebugString());
            }),
        "]");
  };
  return absl::StrCat("Mapping(in_array=", in_array, ",",
                      "out_array=", out_array, ",from=", format_intervals(from),
                      ",to=", format_intervals(to), ")");
}

namespace {

absl::StatusOr<
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>>
ComputeInputDevicesForOutputMap(Client* client,
                                absl::Span<const ArraySpec> input_specs,
                                absl::Span<const ArraySpec> output_specs,
                                absl::Span<const RemapPlan::Mapping> mappings) {
  // A list of intervals along with the sum of entries across all the intervals.
  struct IntervalsAndCount {
    std::vector<RemapPlan::Interval> intervals;
    int64_t count = 0;
  };

  // Map from output array index to all its input contributors.
  //
  // The value is a map from input array index to the intervals of that input
  // array that contribute to the given output.
  absl::flat_hash_map<int, absl::flat_hash_map<int, IntervalsAndCount>>
      output_to_inputs_and_intervals;
  for (int64_t i = 0; i < mappings.size(); ++i) {
    const RemapPlan::Mapping& mapping = mappings[i];
    if (mapping.in_array < 0 || mapping.in_array >= input_specs.size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("mappings[%d].in_array must be in [0, %d], but is %d",
                          i, input_specs.size() - 1, mapping.in_array));
    }
    if (mapping.out_array < 0 || mapping.out_array >= output_specs.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "mappings[%d].out_array must be in [0, %d], but is %d", i,
          output_specs.size() - 1, mapping.out_array));
    }
    if (mapping.from.size() != mapping.to.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "mappings[%d].from and mappings[%d].to must have the same number of "
          "intervals, but has %d and %d intervals",
          i, i, mapping.from.size(), mapping.to.size()));
    }
    const int64_t in_shards_count = input_specs[mapping.in_array]
                                        .sharding->devices()
                                        ->AddressableDeviceList()
                                        ->size();
    const int64_t out_shards_count = output_specs[mapping.out_array]
                                         .sharding->devices()
                                         ->AddressableDeviceList()
                                         ->size();
    IntervalsAndCount& intervals =
        output_to_inputs_and_intervals[mapping.out_array][mapping.in_array];
    for (int s = 0; s < mapping.from.size(); ++s) {
      ABSL_RETURN_IF_ERROR(CheckRange(in_shards_count, mapping.from[s]));
      ABSL_RETURN_IF_ERROR(CheckRange(out_shards_count, mapping.to[s]));
      intervals.intervals.push_back(mapping.from[s]);
      intervals.count += GetNumberOfSteps(mapping.from[s]);
    }
  }

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  for (const auto& [out_array, input_intervals] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       output_to_inputs_and_intervals) {
    TF_RET_CHECK(out_array >= 0 && out_array < output_specs.size());
    const DeviceListRef& out_devices =
        output_specs[out_array].sharding->devices();
    auto [it, inserted] = input_devices_for_output_map.insert({out_array, {}});
    TF_RET_CHECK(inserted);
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [in_array, intervals] : input_intervals) {
      TF_RET_CHECK(in_array >= 0 && in_array < input_specs.size());
      const DeviceListRef& in_devices =
          input_specs[in_array].sharding->devices();
      TF_RET_CHECK(intervals.count >= 0 &&
                   intervals.count <= out_devices->size());
      TF_RET_CHECK(intervals.count >= 0 &&
                   intervals.count <= in_devices->size());
      DeviceListRef interval_device_list;
      if (intervals.count == in_devices->size()) {
        interval_device_list = in_devices;
      } else if (intervals.count == out_devices->size()) {
        interval_device_list = out_devices;
      } else {
        ABSL_ASSIGN_OR_RETURN(
            interval_device_list,
            ComputeDeviceListFromIntervals(client, in_devices, intervals.count,
                                           intervals.intervals));
      }
      it->second.push_back({in_array, interval_device_list});
    }
  }
  return input_devices_for_output_map;
}

}  // namespace

absl::StatusOr<RemapPlan> RemapPlan::CreateOptimized(
    Client* client, std::vector<ArraySpec> input_specs,
    std::vector<ArraySpec> output_specs, std::vector<Mapping> mappings) {
  ABSL_ASSIGN_OR_RETURN(auto input_devices_for_output_map,
                   ComputeInputDevicesForOutputMap(client, input_specs,
                                                   output_specs, mappings));
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(mappings), std::move(input_devices_for_output_map));
  ABSL_RETURN_IF_ERROR(plan.Validate());
  return plan;
}

namespace {

// Validates array-level consistency between an input array spec and an output
// array spec.
absl::Status CheckArraySpecConsistency(int in_array, const ArraySpec& in_spec,
                                       int out_array,
                                       const ArraySpec& out_spec) {
  if (in_spec.dtype != out_spec.dtype) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same dtype: %v (input %d) vs. %v "
        "(output %d)",
        in_spec.dtype, in_array, out_spec.dtype, out_array));
  }

  if (in_spec.sharding == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input array %d has null sharding", in_array));
  }
  if (out_spec.sharding == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Output array %d has null sharding", out_array));
  }

  ABSL_ASSIGN_OR_RETURN(const Shape in_shard_shape,
                   in_spec.sharding->GetShardShape(in_spec.shape));
  ABSL_ASSIGN_OR_RETURN(const Shape out_shard_shape,
                   out_spec.sharding->GetShardShape(out_spec.shape));
  if (in_shard_shape != out_shard_shape) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same shard shape: %v (input %d) vs. %v "
        "(output %d)",
        in_shard_shape, in_array, out_shard_shape, out_array));
  }

  if (in_spec.sharding->memory_kind() != out_spec.sharding->memory_kind()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same memory kind: %v (input %d) vs. %v "
        "(output %d)",
        in_spec.sharding->memory_kind(), in_array,
        out_spec.sharding->memory_kind(), out_array));
  }

  const std::shared_ptr<const xla::PjRtLayout>& in_layout = in_spec.layout;
  const std::shared_ptr<const xla::PjRtLayout>& out_layout = out_spec.layout;
  if (in_layout != out_layout &&
      (in_layout == nullptr || out_layout == nullptr ||
       *in_layout != *out_layout)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input and output must have the same layout: %s (input %d) vs. %s "
        "(output %d)",
        in_layout != nullptr ? in_layout->ToString() : "<nullptr>", in_array,
        out_layout != nullptr ? out_layout->ToString() : "<nullptr>",
        out_array));
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status RemapPlan::ValidateArraySpecsUncached() const {
  const int num_inputs = rep_->input_specs.size();
  if (num_inputs == 0) {
    return absl::InvalidArgumentError("Must have at least one input");
  }

  const int num_outputs = rep_->output_specs.size();
  if (num_outputs == 0) {
    return absl::InvalidArgumentError("Must have at least one output");
  }

  for (int i = 0; i < num_inputs; ++i) {
    if (rep_->input_specs[i].sharding == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input array %d has null sharding", i));
    }
  }
  for (int i = 0; i < num_outputs; ++i) {
    if (rep_->output_specs[i].sharding == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Output array %d has null sharding", i));
    }
  }

  if (rep_->mappings.empty() && rep_->input_devices_for_output_map.empty()) {
    return absl::InvalidArgumentError(
        "Must have at least one mapping or input_devices_for_output_map");
  }

  absl::flat_hash_set<std::pair<int, int>> checked_pairs;
  for (int64_t i = 0; i < rep_->mappings.size(); ++i) {
    const RemapPlan::Mapping& mapping = rep_->mappings[i];
    if (mapping.in_array < 0 || mapping.in_array >= num_inputs) {
      return absl::InvalidArgumentError(
          absl::StrFormat("mappings[%d].in_array must be in [0, %d], but is %d",
                          i, num_inputs - 1, mapping.in_array));
    }
    if (mapping.out_array < 0 || mapping.out_array >= num_outputs) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "mappings[%d].out_array must be in [0, %d], but is %d", i,
          num_outputs - 1, mapping.out_array));
    }
    if (checked_pairs.insert({mapping.in_array, mapping.out_array}).second) {
      ABSL_RETURN_IF_ERROR(CheckArraySpecConsistency(
          mapping.in_array, rep_->input_specs[mapping.in_array],
          mapping.out_array, rep_->output_specs[mapping.out_array]));
    }
  }

  if (!rep_->input_devices_for_output_map.empty()) {
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [out_array, inputs] : rep_->input_devices_for_output_map) {
      if (out_array < 0 || out_array >= num_outputs) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Output buffer index %d in `input_devices_for_output_map` is out "
            "of range [0, %d]",
            out_array, num_outputs - 1));
      }
      for (const InputDeviceRange& range : inputs) {
        if (range.in_array < 0 || range.in_array >= num_inputs) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Input buffer index %d in `input_devices_for_output_map` is out "
              "of range [0, %d]",
              range.in_array, num_inputs - 1));
        }
        if (checked_pairs.insert({range.in_array, out_array}).second) {
          ABSL_RETURN_IF_ERROR(CheckArraySpecConsistency(
              range.in_array, rep_->input_specs[range.in_array], out_array,
              rep_->output_specs[out_array]));
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::Status RemapPlan::ValidateArrayShardMappingsUncached() const {
  const int num_inputs = rep_->input_specs.size();
  const int num_outputs = rep_->output_specs.size();
  TF_RET_CHECK(num_inputs > 0);
  TF_RET_CHECK(num_outputs > 0);

  std::vector<std::vector<bool>> in_used_buffers_list;
  std::vector<absl::InlinedVector<Device*, 1>> out_assigned_devices_list;
  absl::flat_hash_map<int,
                      absl::flat_hash_map<int, absl::flat_hash_set<Device*>>>
      out_buffer_to_in_buffer_and_devices;

  if (!rep_->mappings.empty()) {
    in_used_buffers_list.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      in_used_buffers_list[i].resize(
          /*count=*/rep_->input_specs[i]
              .sharding->devices()
              ->AddressableDeviceList()
              ->size(),
          /*value=*/false);
    }

    out_assigned_devices_list.resize(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      out_assigned_devices_list[i].resize(
          /*n=*/rep_->output_specs[i]
              .sharding->devices()
              ->AddressableDeviceList()
              ->size(),
          /*v=*/nullptr);
    }
    for (int64_t i = 0; i < rep_->mappings.size(); ++i) {
      const RemapPlan::Mapping& mapping = rep_->mappings[i];
      TF_RET_CHECK(mapping.in_array >= 0 && mapping.in_array < num_inputs);
      TF_RET_CHECK(mapping.out_array >= 0 && mapping.out_array < num_outputs);
      absl::flat_hash_set<Device*>* in_device_set =
          rep_->input_devices_for_output_map.contains(mapping.out_array)
              ? &out_buffer_to_in_buffer_and_devices[mapping.out_array]
                                                    [mapping.in_array]
              : nullptr;
      if (mapping.from.size() != mapping.to.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "mappings[%d].from and mappings[%d].to must have the same number "
            "of intervals, but has %d and %d intervals",
            i, i, mapping.from.size(), mapping.to.size()));
      }

      std::vector<bool>& in_used_buffers =
          in_used_buffers_list[mapping.in_array];
      absl::Span<Device* const> in_devices = rep_->input_specs[mapping.in_array]
                                                 .sharding->devices()
                                                 ->AddressableDeviceList()
                                                 ->devices();
      absl::InlinedVector<Device*, 1>& out_assigned_devices =
          out_assigned_devices_list[mapping.out_array];
      const int64_t in_shards_count = in_used_buffers.size();
      const int64_t out_shards_count = out_assigned_devices.size();

      for (int s = 0; s < mapping.from.size(); ++s) {
        const RemapPlan::Interval& in_interval = mapping.from[s];
        const RemapPlan::Interval& out_interval = mapping.to[s];

        ABSL_RETURN_IF_ERROR(CheckRange(in_shards_count, in_interval));
        ABSL_RETURN_IF_ERROR(CheckRange(out_shards_count, out_interval));
        if (GetNumberOfSteps(in_interval) != GetNumberOfSteps(out_interval)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "mappings[%d].from[%d] and mappings[%d].to[%d] must have the "
              "same number of steps, but were %d and %d (%s vs. %s)",
              i, s, i, s, GetNumberOfSteps(in_interval),
              GetNumberOfSteps(out_interval), in_interval.DebugString(),
              out_interval.DebugString()));
        }

        int64_t in_shard = in_interval.start;
        int64_t out_shard = out_interval.start;
        while (in_shard < in_interval.end) {
          TF_RET_CHECK(in_shard >= 0 && in_shard < in_shards_count);
          TF_RET_CHECK(out_shard >= 0 && out_shard < out_shards_count);
          if (in_used_buffers[in_shard]) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Input array %d addressable shard %d is already used",
                mapping.in_array, in_shard));
          }
          in_used_buffers[in_shard] = true;

          if (in_device_set) {
            if (!in_device_set->insert(in_devices[in_shard]).second) {
              return absl::InvalidArgumentError(absl::StrFormat(
                  "Input device %s used more than once in mappings from input "
                  "array %d to output array %d",
                  in_devices[in_shard]->DebugString(), mapping.in_array,
                  mapping.out_array));
            }
          }
          if (out_assigned_devices[out_shard] != nullptr) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Output array %d addressable shard %d is already assigned",
                mapping.out_array, out_shard));
          }
          out_assigned_devices[out_shard] = in_devices[in_shard];

          in_shard += in_interval.step;
          out_shard += out_interval.step;
        }
      }
    }

    for (int i = 0; i < num_outputs; ++i) {
      xla::ifrt::DeviceList* devices =
          rep_->output_specs[i].sharding->devices()->AddressableDeviceList();
      for (int out_shard = 0; out_shard < devices->size(); ++out_shard) {
        if (out_assigned_devices_list[i][out_shard] == nullptr) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output array %d addressable shard %d is unassigned", i,
              out_shard));
        }
      }
      if (out_assigned_devices_list[i] != devices->devices()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Output array %d addressable devices and sharding devices do not "
            "match: Expected %v, but got [%s]",
            i, *devices,
            absl::StrJoin(out_assigned_devices_list[i], ", ",
                          [](std::string* s, Device* d) {
                            absl::StrAppend(s, d->ToString());
                          })));
      }
    }
  }

  if (!rep_->input_devices_for_output_map.empty()) {
    if (rep_->input_devices_for_output_map.size() != num_outputs) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "`input_devices_for_output_map` has %d outputs, but expected %d "
          "outputs",
          rep_->input_devices_for_output_map.size(), num_outputs));
    }
    std::vector<absl::flat_hash_set<Device*>> in_device_sets;
    in_device_sets.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const xla::ifrt::DeviceList* in_devices =
          rep_->input_specs[i].sharding->devices()->AddressableDeviceList();
      in_device_sets.push_back(absl::flat_hash_set<Device*>(
          in_devices->devices().begin(), in_devices->devices().end()));
    }
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [out_array, inputs] : rep_->input_devices_for_output_map) {
      TF_RET_CHECK(out_array >= 0 && out_array < num_outputs);
      for (const InputDeviceRange& range : inputs) {
        TF_RET_CHECK(range.in_array >= 0 && range.in_array < num_inputs);
        if (range.input_devices == nullptr) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output buffer index %d in `input_devices_for_output_map` has "
              "null input_devices for input array %d",
              out_array, range.in_array));
        }

        const absl::flat_hash_set<Device*>& in_device_set =
            in_device_sets[range.in_array];
        for (Device* device :
             range.input_devices->AddressableDeviceList()->devices()) {
          if (!in_device_set.contains(device)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Output buffer index %d in `input_devices_for_output_map` "
                "references device %s from input array %d that is not in the "
                "input array's addressable device list",
                out_array, device->DebugString(), range.in_array));
          }
        }
      }
    }
  }

  if (!rep_->mappings.empty() && !rep_->input_devices_for_output_map.empty()) {
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [out_array, inputs] : rep_->input_devices_for_output_map) {
      const auto out_it = out_buffer_to_in_buffer_and_devices.find(out_array);
      if (out_it == out_buffer_to_in_buffer_and_devices.end()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Output buffer index %d in `input_devices_for_output_map` but not "
            "in `mappings`",
            out_array));
      }
      if (inputs.size() != out_it->second.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Output buffer index %d in `input_devices_for_output_map` has %d "
            "inputs, but `mappings` reference %d inputs",
            out_array, inputs.size(), out_it->second.size()));
      }
      for (const InputDeviceRange& range : inputs) {
        const auto in_it = out_it->second.find(range.in_array);
        if (in_it == out_it->second.end()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output buffer index %d in `input_devices_for_output_map` "
              "references input array %d that is not present in `mappings`",
              out_array, range.in_array));
        }
        if (in_it->second.size() !=
            range.input_devices->AddressableDeviceList()->size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Output buffer index %d in `input_devices_for_output_map` "
              "uses %d addressable devices from input array %d, but `mappings` "
              "contains %d addressable devices",
              out_array, range.input_devices->AddressableDeviceList()->size(),
              range.in_array, in_it->second.size()));
        }
        for (const Device* const device :
             range.input_devices->AddressableDeviceList()->devices()) {
          if (!in_it->second.contains(device)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Output buffer index %d in `input_devices_for_output_map` "
                "references device %s from input array %d, but `mappings` does "
                "not reference that device",
                out_array, device->DebugString(), range.in_array));
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::Status RemapPlan::ValidateArraySpecs() const {
  absl::call_once(rep_->validate_array_specs_once, [this]() {
    rep_->validate_array_specs_status = ValidateArraySpecsUncached();
  });
  return rep_->validate_array_specs_status;
}

absl::Status RemapPlan::Validate() const {
  ABSL_RETURN_IF_ERROR(ValidateArraySpecs());
  absl::call_once(rep_->validate_array_shard_mappings_once, [this]() {
    rep_->validate_array_shard_mappings_status =
        ValidateArrayShardMappingsUncached();
  });
  return rep_->validate_array_shard_mappings_status;
}

absl::StatusOr<RemapPlan> RemapPlan::FromProto(Client* client,
                                               const RemapPlanProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for RemapPlan deserialization"));
  }

  std::vector<ArraySpec> input_specs;
  input_specs.reserve(proto.input_specs_size());
  for (const auto& input_spec_proto : proto.input_specs()) {
    ABSL_ASSIGN_OR_RETURN(ArraySpec input_spec,
                     ArraySpec::FromProto(client, input_spec_proto));
    input_specs.push_back(std::move(input_spec));
  }

  std::vector<ArraySpec> output_specs;
  output_specs.reserve(proto.output_specs_size());
  for (const auto& output_spec_proto : proto.output_specs()) {
    ABSL_ASSIGN_OR_RETURN(ArraySpec output_spec,
                     ArraySpec::FromProto(client, output_spec_proto));
    output_specs.push_back(std::move(output_spec));
  }

  std::vector<Mapping> mappings;
  mappings.reserve(proto.mappings_size());
  for (const auto& mapping_proto : proto.mappings()) {
    ABSL_ASSIGN_OR_RETURN(auto mapping, MappingFromProto(mapping_proto));
    mappings.push_back(std::move(mapping));
  }

  absl::flat_hash_map<int, std::vector<InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.reserve(proto.input_devices_for_output_size());
  for (const auto& inputs_for_output_proto : proto.input_devices_for_output()) {
    std::vector<InputDeviceRange>& input_ranges =
        input_devices_for_output_map[inputs_for_output_proto.out_array()];
    for (const auto& inputs_range_proto :
         inputs_for_output_proto.input_devices()) {
      ABSL_ASSIGN_OR_RETURN(auto devices,
                       InputDeviceRangeFromProto(client, inputs_range_proto));
      input_ranges.push_back(std::move(devices));
    }
  }

  if (mappings.empty()) {
    return RemapPlan(std::move(input_specs), std::move(output_specs),
                     std::move(input_devices_for_output_map));
  }
  if (input_devices_for_output_map.empty()) {
    return RemapPlan(std::move(input_specs), std::move(output_specs),
                     std::move(mappings));
  }
  return RemapPlan(std::move(input_specs), std::move(output_specs),
                   std::move(mappings),
                   std::move(input_devices_for_output_map));
}

absl::Status RemapPlan::ToProto(RemapPlanProto& proto,
                                SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for RemapPlan serialization"));
  }

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());

  proto.mutable_input_specs()->Reserve(rep_->input_specs.size());
  for (const auto& input_spec : rep_->input_specs) {
    ABSL_RETURN_IF_ERROR(input_spec.ToProto(*proto.add_input_specs(), version));
  }
  proto.mutable_output_specs()->Reserve(rep_->output_specs.size());
  for (const auto& output_spec : rep_->output_specs) {
    ABSL_RETURN_IF_ERROR(output_spec.ToProto(*proto.add_output_specs(), version));
  }

  proto.mutable_mappings()->Reserve(rep_->mappings.size());
  for (const auto& mapping : rep_->mappings) {
    ABSL_RETURN_IF_ERROR(MappingToProto(mapping, *proto.add_mappings()));
  }

  proto.mutable_input_devices_for_output()->Reserve(
      rep_->input_devices_for_output_map.size());
  for (const auto& [out_array, input_devices] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       rep_->input_devices_for_output_map) {
    InputDeviceToOutputToProto(version, out_array, input_devices,
                               *proto.add_input_devices_for_output());
  }

  return absl::OkStatus();
}

std::string RemapPlan::DebugString() const {
  auto format_array_specs = [](absl::Span<const ArraySpec> array_specs) {
    return absl::StrCat("[", absl::StrJoin(array_specs, ","), "]");
  };
  auto format_mappings = [](absl::Span<const Mapping> mappings) {
    return absl::StrCat(
        "[",
        absl::StrJoin(mappings, ",",
                      [](std::string* out, const Mapping& mapping) {
                        absl::StrAppend(out, mapping.DebugString());
                      }),
        "]");
  };
  auto format_output_to_inputs =
      [](const absl::flat_hash_map<int, std::vector<InputDeviceRange>>&
             output_to_inputs) {
        return absl::StrCat(
            "[",
            absl::StrJoin(
                output_to_inputs, ",",
                [](std::string* out, const auto& output_to_inputs) {
                  const auto& [out_array, input_devices] = output_to_inputs;
                  absl::StrAppend(
                      out, "o", out_array, ":{",
                      absl::StrJoin(
                          input_devices, ",",
                          [](std::string* out, const InputDeviceRange& range) {
                            absl::StrAppend(out, "i", range.in_array, ":#",
                                            range.input_devices->size());
                          }),
                      "}");
                }),
            "]");
      };
  return absl::StrCat(
      "RemapPlan(input_specs=", format_array_specs(rep_->input_specs),
      ",output_specs=", format_array_specs(rep_->output_specs), ",",
      "mappings=", format_mappings(rep_->mappings), ",output_map=",
      format_output_to_inputs(rep_->input_devices_for_output_map), ")");
}

absl::Status RemapPlan::CheckArrayCopySemantics(
    xla::ifrt::ArrayCopySemantics semantics) const {
  if (semantics != xla::ifrt::ArrayCopySemantics::kDonateInput) {
    if (!CheckOneInputForOneOutput(*this)) {
      return absl::InvalidArgumentError(
          "kDonateInput is required if multiple inputs are mapped to one "
          "output");
    }
  }
  return absl::OkStatus();
}

void RemapPlan::Hash(absl::HashState state) const {
  uint64_t hash = rep_->hash.load(std::memory_order_relaxed);
  if (hash == Rep::kUnsetHash) {
    hash = absl::HashOf(rep_->input_specs, rep_->output_specs, rep_->mappings,
                        rep_->input_devices_for_output_map);
    if (ABSL_PREDICT_FALSE(hash == Rep::kUnsetHash)) {
      ++hash;
    }
    rep_->hash.store(hash, std::memory_order_relaxed);
  }
  absl::HashState::combine(std::move(state), hash);
}

}  // namespace ifrt
}  // namespace xla
