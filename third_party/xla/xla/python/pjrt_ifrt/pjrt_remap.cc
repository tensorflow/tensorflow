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

#include "xla/python/pjrt_ifrt/pjrt_remap.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
PjRtCompatibleClientRemapArrays(PjRtCompatibleClient* client,
                                const RemapPlan& plan,
                                absl::Span<xla::ifrt::ArrayRef> arrays,
                                ArrayCopySemantics semantics) {
  ABSL_RETURN_IF_ERROR(plan.CheckArrayCopySemantics(semantics));
  const int num_inputs = plan.input_specs().size();
  const int num_actual_inputs = arrays.size();
  const int num_outputs = plan.output_specs().size();
  if (num_inputs != num_actual_inputs) {
    return InvalidArgument("RemapArrays expects %d input arrays, but got %d",
                           num_inputs, num_actual_inputs);
  }
  for (int i = 0; i < num_inputs; ++i) {
    if (!isa<PjRtCompatibleArray>(arrays[i].get())) {
      return InvalidArgument(
          "Only PjRtCompatibleArray is supported, but input#%d is %s", i,
          arrays[i]->DebugString());
    }

    if (plan.input_specs()[i].dtype != arrays[i]->dtype()) {
      return InvalidArgument(
          "RemapArrays expects input #%d to have dtype %v, but got %v", i,
          plan.input_specs()[i].dtype, arrays[i]->dtype());
    }
    if (plan.input_specs()[i].shape != arrays[i]->shape()) {
      return InvalidArgument(
          "RemapArrays expects input #%d to have shape %v, but got %v", i,
          plan.input_specs()[i].shape, arrays[i]->shape());
    }
    // Skip xla::ifrt::Sharding::HasSamePartitioning() check because RemapArrays
    // is currently called with input arrays with implicit sharding
    // reinterpretation. Such patterns should be fixed before enabling stricter
    // checking to avoid false positives.
    if (*plan.input_specs()[i].sharding->devices() !=
            *arrays[i]->sharding().devices() ||
        plan.input_specs()[i].sharding->memory_kind() !=
            arrays[i]->sharding().memory_kind()) {
      return InvalidArgument(
          "RemapArrays expects input #%d to be on %v with "
          "%v, but is on %v with %v",
          i, *plan.input_specs()[i].sharding->devices(),
          plan.input_specs()[i].sharding->memory_kind(),
          *arrays[i]->sharding().devices(),
          arrays[i]->sharding().memory_kind());
    }
  }

  std::vector<PjRtArray::PjRtBuffers> out_buffers_list(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    out_buffers_list[i].resize(plan.output_specs()[i]
                                   .sharding->devices()
                                   ->AddressableDeviceList()
                                   ->size());
  }

  // Handle outputs using `input_devices_for_output_map` when specified.
  if (!plan.input_devices_for_output_map().empty()) {
    std::vector<std::optional<absl::flat_hash_map<Device*, int>>>
        in_device_to_shard(num_inputs);

    for (const auto& [out_array, input_ranges] :
         // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
         plan.input_devices_for_output_map()) {
      if (out_array < 0 || out_array >= num_outputs) {
        return InvalidArgument("out_array must be in [0, %d), but is %d",
                               num_outputs, out_array);
      }
      absl::Span<Device* const> out_devices = plan.output_specs()[out_array]
                                                  .sharding->devices()
                                                  ->AddressableDeviceList()
                                                  ->devices();
      // Skip outputs with no addressable shards on this controller.
      if (out_devices.empty()) {
        continue;
      }

      absl::flat_hash_map<Device*, int> device_to_out_shard;
      device_to_out_shard.reserve(out_devices.size());
      for (int out_shard = 0; out_shard < out_devices.size(); ++out_shard) {
        device_to_out_shard.insert({out_devices[out_shard], out_shard});
      }

      PjRtArray::PjRtBuffers& out_buffers = out_buffers_list[out_array];

      for (const RemapPlan::InputDeviceRange& input_range : input_ranges) {
        absl::Span<Device* const> range_devices =
            input_range.input_devices->AddressableDeviceList()->devices();
        // Skip input ranges with no addressable shards on this controller.
        if (range_devices.empty()) {
          continue;
        }

        int in_array = input_range.in_array;
        if (in_array < 0 || in_array >= num_inputs) {
          return InvalidArgument("in_array must be in [0, %d), but is %d",
                                 num_inputs, in_array);
        }
        ABSL_ASSIGN_OR_RETURN(
            absl::Span<std::shared_ptr<xla::PjRtBuffer>> in_buffers,
            static_cast<PjRtCompatibleArray*>(arrays[in_array].get())
                ->mutable_pjrt_buffers());

        if (!in_device_to_shard[in_array].has_value()) {
          absl::Span<Device* const> in_devices = arrays[in_array]
                                                     ->sharding()
                                                     .devices()
                                                     ->AddressableDeviceList()
                                                     ->devices();
          auto& map = in_device_to_shard[in_array].emplace();
          map.reserve(in_devices.size());
          for (int in_shard = 0; in_shard < in_devices.size(); ++in_shard) {
            map.insert({in_devices[in_shard], in_shard});
          }
        }

        for (Device* const device : range_devices) {
          auto in_it = in_device_to_shard[in_array]->find(device);
          if (in_it == in_device_to_shard[in_array]->end()) {
            return InvalidArgument("Device %s not found in input array %d",
                                   device->DebugString(), in_array);
          }
          int in_shard = in_it->second;

          auto out_it = device_to_out_shard.find(device);
          if (out_it == device_to_out_shard.end()) {
            return InvalidArgument("Device %s not found in output array %d",
                                   device->DebugString(), out_array);
          }
          int out_shard = out_it->second;

          if (out_buffers[out_shard] != nullptr) {
            return InvalidArgument(
                "Output array %d shard on device %s is assigned more than once",
                out_array, device->DebugString());
          }

          switch (semantics) {
            case ArrayCopySemantics::kReuseInput:
              out_buffers[out_shard] = in_buffers[in_shard];
              break;
            case ArrayCopySemantics::kDonateInput:
              if (in_buffers[in_shard] == nullptr) {
                return InvalidArgument(
                    "Input array %d shard on device %s is used more than once",
                    in_array, device->DebugString());
              }
              out_buffers[out_shard] = std::move(in_buffers[in_shard]);
              break;
            default:
              return InvalidArgument("Invalid ArrayCopySemantics: %d",
                                     semantics);
          }
        }
      }
    }
  } else {
    for (const RemapPlan::Mapping& mapping : plan.mappings()) {
      ABSL_ASSIGN_OR_RETURN(
          absl::Span<std::shared_ptr<xla::PjRtBuffer>> in_buffers,
          static_cast<PjRtCompatibleArray*>(arrays[mapping.in_array].get())
              ->mutable_pjrt_buffers());
      PjRtArray::PjRtBuffers& out_buffers = out_buffers_list[mapping.out_array];
      for (int s = 0; s < mapping.from.size(); ++s) {
        const RemapPlan::Interval& in_interval = mapping.from[s];
        const RemapPlan::Interval& out_interval = mapping.to[s];
        int64_t in_shard = in_interval.start;
        int64_t out_shard = out_interval.start;
        switch (semantics) {
          case ArrayCopySemantics::kReuseInput:
            while (in_shard < in_interval.end) {
              if (out_buffers[out_shard] != nullptr) {
                return InvalidArgument(
                    "Output array %d shard %d is assigned more than once",
                    mapping.out_array, out_shard);
              }
              out_buffers[out_shard] = in_buffers[in_shard];
              in_shard += in_interval.step;
              out_shard += out_interval.step;
            }
            break;
          case ArrayCopySemantics::kDonateInput:
            while (in_shard < in_interval.end) {
              if (out_buffers[out_shard] != nullptr) {
                return InvalidArgument(
                    "Output array %d shard %d is assigned more than once",
                    mapping.out_array, out_shard);
              }
              if (in_buffers[in_shard] == nullptr) {
                return InvalidArgument(
                    "Input array %d shard %d is used more than once",
                    mapping.in_array, in_shard);
              }
              out_buffers[out_shard] = std::move(in_buffers[in_shard]);
              in_shard += in_interval.step;
              out_shard += out_interval.step;
            }
            break;
          default:
            return InvalidArgument("Invalid ArrayCopySemantics: %d", semantics);
        }
      }
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    for (int s = 0; s < out_buffers_list[i].size(); ++s) {
      if (out_buffers_list[i][s] == nullptr) {
        return InvalidArgument(
            "Output array %d addressable shard %d is unassigned", i, s);
      }
    }
  }

  std::vector<xla::ifrt::ArrayRef> output_arrays;
  output_arrays.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    CHECK_GE(out_buffers_list[i].size(), 1);
    std::shared_ptr<const xla::PjRtLayout> layout =
        out_buffers_list[i].front()->layout();
    ABSL_ASSIGN_OR_RETURN(
        auto output_array,
        PjRtArray::Create(client, plan.output_specs()[i].dtype,
                          plan.output_specs()[i].shape,
                          plan.output_specs()[i].sharding,
                          std::move(out_buffers_list[i]), std::move(layout)));
    output_arrays.push_back(std::move(output_array));
  }
  return output_arrays;
}

}  // namespace ifrt
}  // namespace xla
