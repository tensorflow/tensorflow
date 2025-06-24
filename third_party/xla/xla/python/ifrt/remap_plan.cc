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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/remap_plan.pb.h"
#include "xla/python/ifrt/serdes_version.h"
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
absl::StatusOr<RemapPlanProto::MappingProto> MappingToProto(
    const RemapPlan::Mapping& mapping) {
  TF_RET_CHECK(mapping.from.size() == mapping.to.size());

  RemapPlanProto::MappingProto proto;

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
  return proto;
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

absl::Status RemapPlan::Validate() const {
  const int num_inputs = input_specs.size();
  if (num_inputs == 0) {
    return InvalidArgument("Must have at least one input");
  }

  std::vector<std::vector<bool>> in_used_buffers_list(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    in_used_buffers_list[i].resize(
        /*count=*/input_specs[i].sharding->devices()->size(),
        /*value=*/false);
  }

  const int num_outputs = output_specs.size();
  std::vector<absl::InlinedVector<Device*, 1>> out_assigned_devices_list(
      num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    out_assigned_devices_list[i].resize(
        /*n=*/output_specs[i].sharding->devices()->size(),
        /*v=*/nullptr);
  }

  if (!mappings || mappings->empty()) {
    return InvalidArgument("Must have at least one mapping");
  }

  for (int64_t i = 0; i < mappings->size(); ++i) {
    const RemapPlan::Mapping& mapping = (*mappings)[i];
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

    if (input_specs[mapping.in_array].dtype !=
        output_specs[mapping.out_array].dtype) {
      return InvalidArgument(
          "Input and output must have the same dtype: %v (input %d) vs. %v "
          "(output %d)",
          input_specs[mapping.in_array].dtype, mapping.in_array,
          output_specs[mapping.out_array].dtype, mapping.out_array);
    }

    const std::shared_ptr<const xla::PjRtLayout>& in_layout =
        input_specs[mapping.in_array].layout;
    const std::shared_ptr<const xla::PjRtLayout>& out_layout =
        output_specs[mapping.out_array].layout;
    if (in_layout != out_layout) {
      return InvalidArgument(
          "Input and output must have the same layout: %s (input %d) vs. %s "
          "(output %d)",
          in_layout != nullptr ? in_layout->ToString() : "<nullptr>",
          mapping.in_array,
          out_layout != nullptr ? out_layout->ToString() : "<nullptr>",
          mapping.out_array);
    }

    std::vector<bool>& in_used_buffers = in_used_buffers_list[mapping.in_array];
    absl::Span<Device* const> in_devices =
        input_specs[mapping.in_array].sharding->devices()->devices();
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
          return InvalidArgument("Input array %d shard %d is already used",
                                 mapping.in_array, in_shard);
        }
        in_used_buffers[in_shard] = true;
        if (out_assigned_devices[out_shard] != nullptr) {
          return InvalidArgument("Output array %d shard %d is already assigned",
                                 mapping.out_array, out_shard);
        }
        out_assigned_devices[out_shard] = in_devices[in_shard];
        in_shard += in_interval.step;
        out_shard += out_interval.step;
      }
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    for (int out_shard = 0;
         out_shard < output_specs[i].sharding->devices()->size(); ++out_shard) {
      if (out_assigned_devices_list[i][out_shard] == nullptr) {
        return InvalidArgument("Output array %d shard %d is unassigned", i,
                               out_shard);
      }
    }
    if (out_assigned_devices_list[i] !=
        output_specs[i].sharding->devices()->devices()) {
      return InvalidArgument(
          "Output array %d devices and sharding devices do not match: "
          "Expected %v, but got [%s]",
          i, *output_specs[i].sharding->devices(),
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

  return plan;
}

absl::StatusOr<RemapPlanProto> RemapPlan::ToProto(SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for RemapPlan serialization"));
  }

  RemapPlanProto proto;
  proto.set_version_number(SerDesVersionNumber(0).value());

  proto.mutable_input_specs()->Reserve(input_specs.size());
  for (const auto& input_spec : input_specs) {
    TF_ASSIGN_OR_RETURN(*proto.add_input_specs(), input_spec.ToProto(version));
  }
  proto.mutable_output_specs()->Reserve(output_specs.size());
  for (const auto& output_spec : output_specs) {
    TF_ASSIGN_OR_RETURN(*proto.add_output_specs(),
                        output_spec.ToProto(version));
  }

  proto.mutable_mappings()->Reserve(mappings->size());
  for (const auto& mapping : *mappings) {
    TF_ASSIGN_OR_RETURN(*proto.add_mappings(), MappingToProto(mapping));
  }

  return proto;
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
  return absl::StrCat("RemapPlan(input_specs=", format_array_specs(input_specs),
                      ",output_specs=", format_array_specs(output_specs), ",",
                      "mappings=", format_mappings(*mappings), ")");
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
