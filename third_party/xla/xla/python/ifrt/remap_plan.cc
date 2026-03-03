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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

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
  TF_ASSIGN_OR_RETURN(range.input_devices,
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
    return InvalidArgument("start must be in [0, %d], but is %d",
                           num_shards - 1, interval.start);
  }
  if (interval.step <= 0) {
    return InvalidArgument("step must be positive, but is %d", interval.step);
  }
  if (interval.end < 0 || interval.end > num_shards + interval.step - 1) {
    return InvalidArgument("end must be in [0, %d] if step is %d, but is %d",
                           num_shards + interval.step - 1, interval.step,
                           interval.end);
  }
  return absl::OkStatus();
}

// Returns the number of steps in `interval`.
int64_t GetNumberOfSteps(const RemapPlan::Interval& interval) {
  return (interval.end - interval.start + interval.step - 1) / interval.step;
}

bool CheckOneInputForOneOutput(const xla::ifrt::RemapPlan& plan) {
  if (!plan.mappings) return false;
  const auto& mappings = *plan.mappings;
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
  std::vector<Device*> devices;
  devices.reserve(count);
  for (const RemapPlan::Interval& interval : intervals) {
    int64_t index = interval.start;
    while (index < interval.end) {
      TF_RET_CHECK(index < device_list->size());
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

absl::Status RemapPlan::ComputeInputDevicesForOutputMap(Client* client) {
  TF_RET_CHECK(mappings);
  TF_RET_CHECK(input_devices_for_output_map.empty());
  // A list of intervals along with the sum of entries across all the intervals.
  struct IntervalsAndCount {
    std::vector<Interval> intervals;
    int64_t count = 0;
  };

  // Map from output array index to all its input contributors.
  //
  // The value is a map fron input array index to the intervals of that input
  // array that contribute to the given output.
  absl::flat_hash_map<int, absl::flat_hash_map<int, IntervalsAndCount>>
      output_to_inputs_and_intervals;
  for (const Mapping& mapping : *mappings) {
    IntervalsAndCount& intervals =
        output_to_inputs_and_intervals[mapping.out_array][mapping.in_array];
    for (const Interval& interval : mapping.from) {
      intervals.intervals.push_back(interval);
      intervals.count += GetNumberOfSteps(interval);
    }
  }

  for (const auto& [out_array, input_intervals] :
       output_to_inputs_and_intervals) {
    TF_RET_CHECK(out_array < output_specs.size());
    const DeviceListRef& out_devices =
        output_specs[out_array].sharding->devices();
    auto [it, inserted] = input_devices_for_output_map.insert({out_array, {}});
    TF_RET_CHECK(inserted);
    for (const auto& [in_array, intervals] : input_intervals) {
      TF_RET_CHECK(in_array < input_specs.size());
      const DeviceListRef& in_devices =
          input_specs[in_array].sharding->devices();
      TF_RET_CHECK(intervals.count <= out_devices->size());
      TF_RET_CHECK(intervals.count <= in_devices->size());
      DeviceListRef interval_device_list;
      if (intervals.count == in_devices->size()) {
        interval_device_list = in_devices;
      } else if (intervals.count == out_devices->size()) {
        interval_device_list = out_devices;
      } else {
        TF_ASSIGN_OR_RETURN(
            interval_device_list,
            ComputeDeviceListFromIntervals(client, in_devices, intervals.count,
                                           intervals.intervals));
      }
      it->second.push_back({in_array, interval_device_list});
    }
  }
  return absl::OkStatus();
}

namespace {

// A utility class that calculates the shard shape from an array spec.
class ShardShapeVector {
 public:
  static absl::StatusOr<ShardShapeVector> Create(const ArraySpec& spec) {
    // Fast path for even shardings.
    if (absl::StatusOr<Shape> s = spec.sharding->GetShardShape(spec.shape);
        s.ok()) {
      return ShardShapeVector(*std::move(s));
    }

    TF_ASSIGN_OR_RETURN(
        auto shards, spec.sharding->Disassemble(
                         spec.shape, SingleDeviceShardSemantics::kAllShards));
    std::vector<Shape> shapes;
    shapes.reserve(shards.size());
    for (auto& shard : shards) {
      shapes.push_back(std::move(shard.first));
    }
    return ShardShapeVector(std::move(shapes));
  }

  // Returns the shard shape of `index`-th shard.
  const Shape& shard(int index) const {
    if (auto* shape = std::get_if<Shape>(&shapes_)) {
      return *shape;
    }
    if (auto* shapes = std::get_if<std::vector<Shape>>(&shapes_)) {
      return (*shapes)[index];
    }
    LOG(FATAL) << "Unexpected shapes variant: " << shapes_.index();
  }

 private:
  explicit ShardShapeVector(Shape shape) : shapes_(std::move(shape)) {}

  explicit ShardShapeVector(std::vector<Shape> shapes)
      : shapes_(std::move(shapes)) {}

  std::variant<Shape, std::vector<Shape>> shapes_;
};

}  // namespace

absl::Status RemapPlan::Validate() const {
  const int num_inputs = input_specs.size();
  if (num_inputs == 0) {
    return InvalidArgument("Must have at least one input");
  }

  std::vector<std::vector<bool>> in_used_buffers_list(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    in_used_buffers_list[i].resize(
        /*count=*/input_specs[i]
            .sharding->devices()
            ->AddressableDeviceList()
            ->size(),
        /*value=*/false);
  }

  const int num_outputs = output_specs.size();
  std::vector<absl::InlinedVector<Device*, 1>> out_assigned_devices_list(
      num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    out_assigned_devices_list[i].resize(
        /*n=*/output_specs[i]
            .sharding->devices()
            ->AddressableDeviceList()
            ->size(),
        /*v=*/nullptr);
  }

  if (!mappings || mappings->empty()) {
    return InvalidArgument("Must have at least one mapping");
  }

  absl::flat_hash_map<int,
                      absl::flat_hash_map<int, absl::flat_hash_set<Device*>>>
      out_buffer_to_in_buffer_and_devices;
  for (int64_t i = 0; i < mappings->size(); ++i) {
    const RemapPlan::Mapping& mapping = (*mappings)[i];
    absl::flat_hash_set<Device*>* in_device_set =
        input_devices_for_output_map.contains(mapping.out_array)
            ? &out_buffer_to_in_buffer_and_devices[mapping.out_array]
                                                  [mapping.in_array]
            : nullptr;
    if (mapping.in_array < 0 || mapping.in_array >= num_inputs) {
      return InvalidArgument(
          "mappings[%d].in_array must be in [0, %d], but is %d", i,
          num_inputs - 1, mapping.in_array);
    }
    if (mapping.out_array < 0 || mapping.out_array >= num_outputs) {
      return InvalidArgument(
          "mappings[%d].out_array must be in [0, %d], but is %d", i,
          num_outputs - 1, mapping.out_array);
    }
    if (mapping.from.size() != mapping.to.size()) {
      return InvalidArgument(
          "mappings[%d].from and mappings[%d].to must have the same number of "
          "intervals, but has %d and %d intervals",
          i, i, mapping.from.size(), mapping.to.size());
    }

    const ArraySpec& input_spec = input_specs[mapping.in_array];
    const ArraySpec& output_spec = output_specs[mapping.out_array];

    if (input_spec.dtype != output_spec.dtype) {
      return InvalidArgument(
          "Input and output must have the same dtype: %v (input %d) vs. %v "
          "(output %d)",
          input_spec.dtype, mapping.in_array, output_spec.dtype,
          mapping.out_array);
    }

    const std::shared_ptr<const xla::PjRtLayout>& in_layout = input_spec.layout;
    const std::shared_ptr<const xla::PjRtLayout>& out_layout =
        output_spec.layout;
    if (in_layout != out_layout) {
      return InvalidArgument(
          "Input and output must have the same layout: %s (input %d) vs. %s "
          "(output %d)",
          in_layout != nullptr ? in_layout->ToString() : "<nullptr>",
          mapping.in_array,
          out_layout != nullptr ? out_layout->ToString() : "<nullptr>",
          mapping.out_array);
    }

    TF_ASSIGN_OR_RETURN(const auto input_shard_shapes,
                        ShardShapeVector::Create(input_spec));
    TF_ASSIGN_OR_RETURN(const auto output_shard_shapes,
                        ShardShapeVector::Create(output_spec));

    std::vector<bool>& in_used_buffers = in_used_buffers_list[mapping.in_array];
    absl::Span<Device* const> in_devices = input_specs[mapping.in_array]
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

      TF_RETURN_IF_ERROR(CheckRange(in_shards_count, in_interval));
      TF_RETURN_IF_ERROR(CheckRange(out_shards_count, out_interval));
      if (GetNumberOfSteps(in_interval) != GetNumberOfSteps(out_interval)) {
        return InvalidArgument(
            "mappings[%d].from[%d] and mappings[%d].to[%d] must have the same "
            "number of steps, but were %d and %d "
            "(%s vs. %s)",
            i, s, i, s, GetNumberOfSteps(in_interval),
            GetNumberOfSteps(out_interval), in_interval.DebugString(),
            out_interval.DebugString());
      }

      int64_t in_shard = in_interval.start;
      int64_t out_shard = out_interval.start;
      while (in_shard < in_interval.end) {
        if (in_used_buffers[in_shard]) {
          return InvalidArgument(
              "Input array %d addressable shard %d is already used",
              mapping.in_array, in_shard);
        }
        in_used_buffers[in_shard] = true;

        if (in_device_set) {
          if (!in_device_set->insert(in_devices[in_shard]).second) {
            return InvalidArgument(
                "Input device %s used more than once in mappings from input "
                "array %d to output array %d",
                in_devices[in_shard]->DebugString(), mapping.in_array,
                mapping.out_array);
          }
        }
        if (out_assigned_devices[out_shard] != nullptr) {
          return InvalidArgument(
              "Output array %d addressable shard %d is already assigned",
              mapping.out_array, out_shard);
        }
        out_assigned_devices[out_shard] = in_devices[in_shard];

        if (input_shard_shapes.shard(in_shard) !=
            output_shard_shapes.shard(out_shard)) {
          return InvalidArgument(
              "Output array %d addressable shard %d has a different shard "
              "shape from the corresponding input shard: %v -> %v",
              mapping.out_array, out_shard, input_shard_shapes.shard(in_shard),
              output_shard_shapes.shard(out_shard));
        }

        in_shard += in_interval.step;
        out_shard += out_interval.step;
      }
    }
  }

  for (const auto& [out_array, inputs] : input_devices_for_output_map) {
    const auto out_it = out_buffer_to_in_buffer_and_devices.find(out_array);
    if (out_it == out_buffer_to_in_buffer_and_devices.end()) {
      return InvalidArgument(
          "Output buffer index %d in `input_devices_for_output_map` but not in "
          "`mappings`",
          out_array);
    }
    if (inputs.size() != out_it->second.size()) {
      return InvalidArgument(
          "Output buffer index %d in `input_devices_for_output_map` has %d "
          "inputs, but `mappings` reference %d inputs",
          out_array, inputs.size(), out_it->second.size());
    }
    for (const InputDeviceRange& range : inputs) {
      const auto in_it = out_it->second.find(range.in_array);
      if (in_it == out_it->second.end()) {
        return InvalidArgument(
            "Output buffer index %d in `input_devices_for_output_map` "
            "references input array %d that is not present in `mappings`",
            out_array, range.in_array);
      }
      if (in_it->second.size() !=
          range.input_devices->AddressableDeviceList()->size()) {
        return InvalidArgument(
            "Output buffer index %d in `input_devices_for_output_map` "
            "uses %d addressable devices from input array %d, but `mappings` "
            "contains %d addressable devices",
            out_array, range.input_devices->AddressableDeviceList()->size(),
            range.in_array, in_it->second.size());
      }
      for (const Device* const device :
           range.input_devices->AddressableDeviceList()->devices()) {
        if (!in_it->second.contains(device)) {
          return InvalidArgument(
              "Output buffer index %d in `input_devices_for_output_map` "
              "references device %s from input array %d, but `mappings` does "
              "not reference that device",
              out_array, device->DebugString(), range.in_array);
        }
      }
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    xla::ifrt::DeviceList* devices =
        output_specs[i].sharding->devices()->AddressableDeviceList();
    for (int out_shard = 0; out_shard < devices->size(); ++out_shard) {
      if (out_assigned_devices_list[i][out_shard] == nullptr) {
        return InvalidArgument(
            "Output array %d addressable shard %d is unassigned", i, out_shard);
      }
    }
    if (out_assigned_devices_list[i] != devices->devices()) {
      return InvalidArgument(
          "Output array %d addressable devices and sharding devices do not "
          "match: Expected %v, but got [%s]",
          i, *devices,
          absl::StrJoin(out_assigned_devices_list[i], ", ",
                        [](std::string* s, Device* d) {
                          absl::StrAppend(s, d->ToString());
                        }));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<RemapPlan> RemapPlan::FromProto(Client* client,
                                               const RemapPlanProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for RemapPlan deserialization"));
  }

  RemapPlan plan;

  plan.input_specs.reserve(proto.input_specs_size());
  for (const auto& input_spec_proto : proto.input_specs()) {
    TF_ASSIGN_OR_RETURN(ArraySpec input_spec,
                        ArraySpec::FromProto(client, input_spec_proto));
    plan.input_specs.push_back(std::move(input_spec));
  }

  plan.output_specs.reserve(proto.output_specs_size());
  for (const auto& output_spec_proto : proto.output_specs()) {
    TF_ASSIGN_OR_RETURN(ArraySpec output_spec,
                        ArraySpec::FromProto(client, output_spec_proto));
    plan.output_specs.push_back(std::move(output_spec));
  }

  plan.mappings = std::make_shared<std::vector<Mapping>>();
  plan.mappings->reserve(proto.mappings_size());
  for (const auto& mapping_proto : proto.mappings()) {
    TF_ASSIGN_OR_RETURN(auto mapping, MappingFromProto(mapping_proto));
    plan.mappings->push_back(std::move(mapping));
  }

  plan.input_devices_for_output_map.reserve(
      proto.input_devices_for_output_size());
  for (const auto& inputs_for_output_proto : proto.input_devices_for_output()) {
    std::vector<InputDeviceRange>& input_ranges =
        plan.input_devices_for_output_map[inputs_for_output_proto.out_array()];
    for (const auto& inputs_range_proto :
         inputs_for_output_proto.input_devices()) {
      TF_ASSIGN_OR_RETURN(
          auto devices, InputDeviceRangeFromProto(client, inputs_range_proto));
      input_ranges.push_back(std::move(devices));
    }
  }

  return plan;
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

  proto.mutable_input_specs()->Reserve(input_specs.size());
  for (const auto& input_spec : input_specs) {
    TF_RETURN_IF_ERROR(input_spec.ToProto(*proto.add_input_specs(), version));
  }
  proto.mutable_output_specs()->Reserve(output_specs.size());
  for (const auto& output_spec : output_specs) {
    TF_RETURN_IF_ERROR(output_spec.ToProto(*proto.add_output_specs(), version));
  }

  proto.mutable_mappings()->Reserve(mappings->size());
  for (const auto& mapping : *mappings) {
    TF_RETURN_IF_ERROR(MappingToProto(mapping, *proto.add_mappings()));
  }

  proto.mutable_input_devices_for_output()->Reserve(
      input_devices_for_output_map.size());
  for (const auto& [out_array, input_devices] : input_devices_for_output_map) {
    InputDeviceToOutputToProto(version, out_array, input_devices,
                               *proto.add_input_devices_for_output());
  }

  return absl::OkStatus();
}

std::string RemapPlan::DebugString() const {
  auto format_array_specs = [](absl::Span<const ArraySpec> array_specs) {
    return absl::StrCat(
        "[",
        absl::StrJoin(array_specs, ",",
                      [](std::string* out, const ArraySpec& spec) {
                        absl::StrAppend(out, spec.DebugString());
                      }),
        "]");
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
  return absl::StrCat("RemapPlan(input_specs=", format_array_specs(input_specs),
                      ",output_specs=", format_array_specs(output_specs), ",",
                      "mappings=", format_mappings(*mappings), ",output_map=",
                      format_output_to_inputs(input_devices_for_output_map),
                      ")");
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

}  // namespace ifrt
}  // namespace xla
