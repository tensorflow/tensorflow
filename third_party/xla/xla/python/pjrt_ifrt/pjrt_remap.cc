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
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
PjRtCompatibleClientRemapArrays(
    PjRtCompatibleClient* client, const RemapPlan& plan,
    absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
    ArrayCopySemantics semantics) {
  TF_RETURN_IF_ERROR(plan.CheckArrayCopySemantics(semantics));
  const int num_inputs = plan.input_specs.size();
  const int num_actual_inputs = arrays.size();
  const int num_outputs = plan.output_specs.size();
  if (num_inputs != num_actual_inputs) {
    return InvalidArgument("RemapArrays expects %d input arrays, but got %d",
                           num_inputs, num_actual_inputs);
  }
  for (int i = 0; i < num_inputs; ++i) {
    if (!llvm::isa<PjRtCompatibleArray>(arrays[i].get())) {
      return InvalidArgument(
          "Only PjRtCompatibleArray is supported, but input#%d is %s", i,
          arrays[i]->DebugString());
    }

    if (plan.input_specs[i].dtype != arrays[i]->dtype()) {
      return InvalidArgument(
          "RemapArrays expects input #%d to have dtype %v, but got %v", i,
          plan.input_specs[i].dtype, arrays[i]->dtype());
    }
    if (plan.input_specs[i].shape != arrays[i]->shape()) {
      return InvalidArgument(
          "RemapArrays expects input #%d to have shape %v, but got %v", i,
          plan.input_specs[i].shape, arrays[i]->shape().DebugString());
    }
    // Skip xla::ifrt::Sharding::HasSamePartitioning() check because RemapArrays
    // is currently called with input arrays with implicit sharding
    // reinterpretation. Such patterns should be fixed before enabling stricter
    // checking to avoid false positives.
    if (*plan.input_specs[i].sharding->devices() !=
            *arrays[i]->sharding().devices() ||
        plan.input_specs[i].sharding->memory_kind() !=
            arrays[i]->sharding().memory_kind()) {
      return InvalidArgument(
          "RemapArrays expects input #%d to be on %v with "
          "%v, but is on %v with %v",
          i, *plan.input_specs[i].sharding->devices(),
          plan.input_specs[i].sharding->memory_kind(),
          *arrays[i]->sharding().devices(),
          arrays[i]->sharding().memory_kind());
    }
  }

  std::vector<PjRtArray::PjRtBuffers> out_buffers_list(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    out_buffers_list[i].resize(plan.output_specs[i]
                                   .sharding->devices()
                                   ->AddressableDeviceList()
                                   ->size());
  }

  for (const RemapPlan::Mapping& mapping : *plan.mappings) {
    TF_ASSIGN_OR_RETURN(
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
            out_buffers[out_shard] = in_buffers[in_shard];
            in_shard += in_interval.step;
            out_shard += out_interval.step;
          }
          break;
        case ArrayCopySemantics::kDonateInput:
          while (in_shard < in_interval.end) {
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

  std::vector<tsl::RCReference<xla::ifrt::Array>> output_arrays;
  output_arrays.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    CHECK_GE(out_buffers_list[i].size(), 1);
    std::shared_ptr<const xla::PjRtLayout> layout =
        out_buffers_list[i].front()->layout();
    TF_ASSIGN_OR_RETURN(
        auto output_array,
        PjRtArray::Create(client, plan.output_specs[i].dtype,
                          plan.output_specs[i].shape,
                          plan.output_specs[i].sharding,
                          std::move(out_buffers_list[i]), std::move(layout)));
    output_arrays.push_back(std::move(output_array));
  }
  return output_arrays;
}

}  // namespace ifrt
}  // namespace xla
