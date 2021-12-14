/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/profiler/protobuf/memory_viewer_preprocess.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using absl::StrFormat;
using ::xla::BufferAllocationProto;
using ::xla::HloProto;
using ::xla::LayoutUtil;
using ::xla::LogicalBufferProto;
using ::xla::HloInstructionProto;
using ::xla::Shape;
using ::xla::ShapeUtil;

double BytesToMiB(int64_t bytes) {
  return static_cast<double>(bytes) / tensorflow::MathUtil::IPow(2, 20);
}

HeapObject MakeHeapObjectCommon(std::string label, int logical_buffer_id,
                                int64_t logical_buffer_size_bytes,
                                int64_t unpadded_shape_bytes) {
  HeapObject result;
  result.set_label(std::move(label));
  result.set_logical_buffer_id(logical_buffer_id);
  result.set_logical_buffer_size_mib(BytesToMiB(logical_buffer_size_bytes));
  result.set_unpadded_shape_mib(BytesToMiB(unpadded_shape_bytes));
  return result;
}

HeapObject MakeHeapObject(int color, std::string label, int logical_buffer_id,
                          int64_t logical_buffer_size_bytes,
                          int64_t unpadded_shape_bytes) {
  HeapObject result =
      MakeHeapObjectCommon(std::move(label), logical_buffer_id,
                           logical_buffer_size_bytes, unpadded_shape_bytes);
  result.set_numbered(color);
  return result;
}

HeapObject MakeHeapObject(std::string color, std::string label,
                          int logical_buffer_id,
                          int64_t logical_buffer_size_bytes,
                          int64_t unpadded_shape_bytes) {
  HeapObject result =
      MakeHeapObjectCommon(std::move(label), logical_buffer_id,
                           logical_buffer_size_bytes, unpadded_shape_bytes);
  result.set_named(std::move(color));
  return result;
}

BufferSpan MakeBufferSpan(int32 start, int32 limit) {
  BufferSpan result;
  result.set_start(start);
  result.set_limit(limit);
  return result;
}

const Shape* ResolveShapeIndex(const Shape* shape,
                               absl::Span<const int64_t> shape_index) {
  for (int64_t value : shape_index) {
    shape = &shape->tuple_shapes(value);
  }
  return shape;
}

// A wrapper around ShapeUtil::ByteSizeOf that clears out the layout/padding,
// since that is considered in the ByteSizeOf calculation.
int64_t UnpaddedSize(Shape shape) {
  // Ensure the layout has no padding by making it the default layout.
  LayoutUtil::SetToDefaultLayout(&shape);
  // Note: we make a simplifying assumption here that a "minimal" size for a
  // tuple member would be the size of a `void*` -- there may be even fancier
  // ways of doing things, but this should give a good enough approximation of
  // what a minimal tuple size is.
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/sizeof(void*));
}

void Convert(const xla::BufferAllocationProto_Assigned& assigned,
             const absl::flat_hash_map<int64_t, const LogicalBufferProto*>&
                 id_to_logical_buffer,
             const absl::node_hash_map<std::string, const HloInstructionProto*>&
                 name_to_hlo,
             LogicalBuffer* result) {
  result->set_id(assigned.logical_buffer_id()),
      result->set_size_mib(BytesToMiB(assigned.size()));
  const LogicalBufferProto* proto =
      id_to_logical_buffer.at(assigned.logical_buffer_id());
  const std::string& instruction_name = proto->defined_at().instruction_name();
  result->set_hlo_name(instruction_name);
  result->mutable_shape_index()->CopyFrom(proto->defined_at().shape_index());
  const Shape top_level_shape(name_to_hlo.at(instruction_name)->shape());
  const Shape* shape =
      ResolveShapeIndex(&top_level_shape, proto->defined_at().shape_index());
  result->set_shape(ShapeUtil::HumanStringWithLayout(*shape));
}

bool IsReusable(const BufferAllocationProto& buffer_allocation) {
  return !buffer_allocation.is_thread_local() && !buffer_allocation.is_tuple();
}

void Convert(const BufferAllocationProto& proto,
             const absl::flat_hash_map<int64_t, const LogicalBufferProto*>&
                 id_to_logical_buffer,
             const absl::node_hash_map<std::string, const HloInstructionProto*>&
                 name_to_hlo,
             BufferAllocation* result) {
  result->set_id(proto.index());
  result->set_size_mib(BytesToMiB(proto.size()));
  if (proto.is_entry_computation_parameter()) {
    result->add_attributes("entry computation parameter");
  }
  if (proto.maybe_live_out()) {
    result->add_attributes("may-be live out");
  }
  if (IsReusable(proto)) {
    result->add_attributes("reusable");
  }
  for (const auto& assigned : proto.assigned()) {
    Convert(assigned, id_to_logical_buffer, name_to_hlo,
            result->add_logical_buffers());
  }
  // Check whether all logical buffers for this buffer allocation have a common
  // shape.
  if (!result->logical_buffers().empty()) {
    std::string common_shape = result->logical_buffers(0).shape();
    for (int64_t i = 1; i < result->logical_buffers_size(); ++i) {
      if (result->logical_buffers(i).shape() != common_shape) {
        common_shape = "";
        break;
      }
    }
    if (!common_shape.empty()) {
      result->set_common_shape(common_shape);
    }
  }
}

void NoteSpecialAllocations(
    const absl::flat_hash_set<const BufferAllocationProto*>&
        all_buffer_allocations,
    const absl::flat_hash_map<int64_t, const LogicalBufferProto*>&
        id_to_logical_buffer,

    const absl::node_hash_map<std::string, const HloInstructionProto*>&
        name_to_hlo,
    int64_t small_buffer_size, PreprocessResult* result) {
  int64_t entry_parameters_bytes = 0;
  int64_t non_reusable_bytes = 0;
  int64_t maybe_live_out_bytes = 0;
  for (const BufferAllocationProto* buffer_allocation :
       all_buffer_allocations) {
    if (buffer_allocation->is_entry_computation_parameter()) {
      entry_parameters_bytes += buffer_allocation->size();
    }
    if (!IsReusable(*buffer_allocation)) {
      non_reusable_bytes += buffer_allocation->size();
    }
    if (buffer_allocation->maybe_live_out()) {
      if (buffer_allocation->size() > small_buffer_size) {
        VLOG(1) << "Maybe live out buffer allocation: "
                << buffer_allocation->size()
                << " bytes :: " << buffer_allocation->ShortDebugString();
      }
      maybe_live_out_bytes += buffer_allocation->size();
    }
    Convert(*buffer_allocation, id_to_logical_buffer, name_to_hlo,
            result->add_indefinite_lifetimes());
  }

  result->set_entry_computation_parameters_mib(
      BytesToMiB(entry_parameters_bytes));
  result->set_non_reusable_mib(BytesToMiB(non_reusable_bytes));
  result->set_maybe_live_out_mib(BytesToMiB(maybe_live_out_bytes));
}

}  // namespace

absl::StatusOr<PreprocessResult> ConvertHloProtoToPreprocessResult(
    const HloProto& hlo_proto, int64_t small_buffer_size,
    int64_t heap_simulator_trace_id) {
  // Construct a mapping from name to HLO proto.
  absl::node_hash_map<std::string, const HloInstructionProto*> name_to_hlo;
  for (const auto& computation : hlo_proto.hlo_module().computations()) {
    for (const auto& instruction : computation.instructions()) {
      name_to_hlo[instruction.name()] = &instruction;
      VLOG(1) << "HLO: " << instruction.ShortDebugString();
    }
  }

  // Mapping from logical buffer ID to logical buffer, and set of all logical
  // buffer protos.
  absl::flat_hash_map<int64_t, const LogicalBufferProto*> id_to_logical_buffer;
  absl::flat_hash_set<const LogicalBufferProto*> all_logical_buffers;
  for (const auto& logical_buffer :
       hlo_proto.buffer_assignment().logical_buffers()) {
    VLOG(1) << "Logical buffer: " << logical_buffer.ShortDebugString();
    id_to_logical_buffer[logical_buffer.id()] = &logical_buffer;
    all_logical_buffers.insert(&logical_buffer);
  }

  // Mapping from logocal buffer proto to the buffer allocation that it exists
  // inside (there must be only one).
  //
  // Also a reverse mapping from buffer allocation proto to the set of logical
  // buffer protos that exist inside of it.
  absl::flat_hash_map<const LogicalBufferProto*, const BufferAllocationProto*>
      logical_buffer_to_buffer_allocation;
  absl::node_hash_map<const BufferAllocationProto*,
                      absl::flat_hash_set<const LogicalBufferProto*>>
      buffer_allocation_to_logical_buffers;
  absl::flat_hash_set<const BufferAllocationProto*> all_buffer_allocations;
  for (const BufferAllocationProto& buffer_allocation :
       hlo_proto.buffer_assignment().buffer_allocations()) {
    all_buffer_allocations.insert(&buffer_allocation);
    for (const xla::BufferAllocationProto_Assigned& assigned :
         buffer_allocation.assigned()) {
      const LogicalBufferProto* logical_buffer =
          id_to_logical_buffer.at(assigned.logical_buffer_id());
      buffer_allocation_to_logical_buffers[&buffer_allocation].insert(
          logical_buffer);
      auto insert_result = logical_buffer_to_buffer_allocation.insert(
          {logical_buffer, &buffer_allocation});
      if (!insert_result.second) {
        return absl::InvalidArgumentError(
            "A logical buffer appears to be associated with multiple buffer "
            "allocations.");
      }
    }
  }

  std::vector<int64_t> logical_buffers;
  std::vector<int64_t> peak_logical_buffers;

  int64_t heap_size_bytes = 0;
  int64_t unpadded_heap_size_bytes = 0;

  int64_t peak_heap_size_bytes = 0;
  int64_t unpadded_peak_heap_size_bytes = 0;  // Unpadded size at peak.
  int64_t peak_heap_size_position = 0;
  std::vector<double> heap_sizes;
  std::vector<double> unpadded_heap_sizes;

  absl::node_hash_map<int64_t, std::pair<int64_t, absl::optional<int64_t>>>
      logical_buffer_spans;
  absl::flat_hash_set<const LogicalBufferProto*> seen;
  absl::flat_hash_set<const BufferAllocationProto*> seen_buffer_allocations;

  // Run through all the simulator events in the given trace, and simulate the
  // heap in order to find the point of peak memory usage and record its
  // associated metadata.
  if (heap_simulator_trace_id >= 0 &&
      heap_simulator_trace_id <
          hlo_proto.buffer_assignment().heap_simulator_traces_size()) {
    const auto& simulator_events =
        hlo_proto.buffer_assignment()
            .heap_simulator_traces(heap_simulator_trace_id)
            .events();
    for (const auto& event : simulator_events) {
      heap_sizes.push_back(BytesToMiB(heap_size_bytes));
      unpadded_heap_sizes.push_back(BytesToMiB(unpadded_heap_size_bytes));
      const auto* logical_buffer = id_to_logical_buffer.at(event.buffer_id());
      seen.insert(logical_buffer);
      seen_buffer_allocations.insert(
          logical_buffer_to_buffer_allocation.at(logical_buffer));
      const auto& instruction_name =
          logical_buffer->defined_at().instruction_name();
      const Shape top_level_shape(name_to_hlo.at(instruction_name)->shape());
      const Shape* shape = ResolveShapeIndex(
          &top_level_shape, logical_buffer->defined_at().shape_index());
      if (event.kind() == xla::HeapSimulatorTrace_Event::ALLOC ||
          event.kind() == xla::HeapSimulatorTrace_Event::SHARE_WITH) {
        logical_buffers.push_back(event.buffer_id());
        heap_size_bytes += logical_buffer->size();
        unpadded_heap_size_bytes += UnpaddedSize(*shape);
        // Initialize the buffer span from the current event to the last event.
        logical_buffer_spans[event.buffer_id()] = {heap_sizes.size() - 1,
                                                   simulator_events.size() - 1};
        int64_t prior_peak_heap_size_bytes = peak_heap_size_bytes;
        peak_heap_size_bytes = std::max(peak_heap_size_bytes, heap_size_bytes);
        if (prior_peak_heap_size_bytes != peak_heap_size_bytes) {
          peak_heap_size_position = heap_sizes.size() - 1;
          unpadded_peak_heap_size_bytes = unpadded_heap_size_bytes;
          VLOG(1) << StrFormat("New peak heap size on %d: %s :: %d bytes",
                               peak_heap_size_position, instruction_name,
                               peak_heap_size_bytes);
          peak_logical_buffers = logical_buffers;
        }
      } else if (event.kind() == xla::HeapSimulatorTrace_Event::FREE) {
        logical_buffers.erase(
            std::remove(logical_buffers.begin(), logical_buffers.end(),
                        event.buffer_id()),
            logical_buffers.end());
        heap_size_bytes -= logical_buffer->size();
        unpadded_heap_size_bytes -= UnpaddedSize(*shape);
        logical_buffer_spans[event.buffer_id()].second = heap_sizes.size() - 1;
        if (heap_size_bytes < 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "heap_size_bytes should be non-negative: ", heap_size_bytes));
        }
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Unhandled event kind: ", event.kind()));
      }
    }

    if (seen_buffer_allocations.size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("All heap simulation should work out of a single buffer "
                       "allocation, actual seen_buffer_allocations.size():",
                       seen_buffer_allocations.size()));
    }
  }

  VLOG(1) << "Found " << peak_logical_buffers.size()
          << " logical buffers alive at point of peak heap usage.";

  VLOG(1) << "Peak logical buffers: ["
          << absl::StrJoin(peak_logical_buffers, ",") << "]";

  int64_t indefinite_memory_usage_bytes = 0;
  std::vector<HeapObject> max_heap;
  int colorno = 0;
  int64_t rest = 0;

  // Helper lambda that adds the logical buffer as an element in the "max heap"
  // view with constitutent logical buffers.
  auto add_heap_object = [&](const LogicalBufferProto* logical_buffer) {
    if (logical_buffer->size() <= small_buffer_size) {
      rest += logical_buffer->size();
      return;
    }
    const std::string& instruction_name =
        logical_buffer->defined_at().instruction_name();
    const Shape top_level_shape(name_to_hlo.at(instruction_name)->shape());
    const Shape* shape = ResolveShapeIndex(
        &top_level_shape, logical_buffer->defined_at().shape_index());
    std::string shape_string = ShapeUtil::HumanStringWithLayout(*shape);
    int64 unpadded_shape_bytes = UnpaddedSize(*shape);
    const std::string& metadata =
        name_to_hlo.at(instruction_name)->metadata().op_name();
    std::string label =
        StrFormat("%s: %s # %s", instruction_name, shape_string, metadata);
    max_heap.push_back(
        MakeHeapObject(colorno++, std::move(label), logical_buffer->id(),
                       logical_buffer->size(), unpadded_shape_bytes));
  };

  // Now look for all logical buffers which have not been seen, and assume they
  // have indefinite lifetime if they are not in thread-local buffer
  // allocations.
  absl::flat_hash_set<const LogicalBufferProto*> unseen;
  for (const LogicalBufferProto* logical_buffer : all_logical_buffers) {
    if (!seen.contains(logical_buffer)) {
      unseen.insert(logical_buffer);
    }
  }
  for (const LogicalBufferProto* logical_buffer : unseen) {
    const BufferAllocationProto* buffer_allocation =
        logical_buffer_to_buffer_allocation.at(logical_buffer);
    if (buffer_allocation->is_thread_local()) {
      continue;
    }
    // Clear out the assigned logical buffers when stringifying the buffer
    // allocation, as it can be a long list.
    auto to_string = [](const BufferAllocationProto* p) {
      BufferAllocationProto copy = *p;
      copy.mutable_assigned()->Clear();
      return copy.ShortDebugString();
    };
    if (seen_buffer_allocations.insert(buffer_allocation).second) {
      indefinite_memory_usage_bytes += buffer_allocation->size();
      const auto& logical_buffers =
          buffer_allocation_to_logical_buffers.at(buffer_allocation);
      if (logical_buffers.size() == 1) {
        add_heap_object(*logical_buffers.begin());
      } else {
        VLOG(1) << "Indefinite lifetime, no heap object shown due to "
                   "multiple logical buffers in buffer allocation: "
                << logical_buffer->ShortDebugString()
                << " :: " << to_string(buffer_allocation) << std::endl;
      }
      if (buffer_allocation->size() > small_buffer_size) {
        VLOG(1) << "Indefinite memory usage now: "
                << indefinite_memory_usage_bytes << " bytes (+"
                << buffer_allocation->size() << " bytes)";
      }
    }
  }

  // For the buffers that have indefinite lifetime (that is, lifetime not
  // reflected by the heap simulation) add it to the peak values and the vectors
  // of heap sizes.
  peak_heap_size_bytes += indefinite_memory_usage_bytes;
  unpadded_peak_heap_size_bytes += indefinite_memory_usage_bytes;
  double addend = BytesToMiB(indefinite_memory_usage_bytes);
  for (int i = 0; i < heap_sizes.size(); ++i) {
    heap_sizes[i] += addend;
    unpadded_heap_sizes[i] += addend;
  }

  // Accumulate data for use in a stacked bar plot.
  //
  // We accumulate it in "program order" -- the order in which it was placed
  // into the logical_buffers sequence above was program order, and we iterate
  // that order to create data points.
  for (int logical_buffer_id : peak_logical_buffers) {
    const auto* logical_buffer = id_to_logical_buffer.at(logical_buffer_id);
    add_heap_object(logical_buffer);
  }
  if (rest != 0) {
    max_heap.push_back(MakeHeapObject(
        "gray", StrFormat("small (<%d bytes)", small_buffer_size), -1, rest,
        0));
  }

  std::vector<const HeapObject*> max_heap_by_size;
  max_heap_by_size.reserve(max_heap.size());
  for (const auto& object : max_heap) {
    max_heap_by_size.push_back(&object);
  }
  std::sort(max_heap_by_size.begin(), max_heap_by_size.end(),
            [](const HeapObject* a, const HeapObject* b) {
              return a->logical_buffer_size_mib() >
                     b->logical_buffer_size_mib();
            });

  std::vector<int> max_heap_to_by_size;
  max_heap_to_by_size.reserve(max_heap.size());
  for (const auto& object : max_heap) {
    auto it =
        std::find(max_heap_by_size.begin(), max_heap_by_size.end(), &object);
    int index = std::distance(max_heap_by_size.begin(), it);
    max_heap_to_by_size.push_back(index);
  }

  std::vector<int> by_size_to_max_heap;
  for (const auto* object : max_heap_by_size) {
    int index = object - &max_heap[0];
    by_size_to_max_heap.push_back(index);
  }

  PreprocessResult result;
  result.set_module_name(hlo_proto.hlo_module().name());
  result.set_entry_computation_name(
      hlo_proto.hlo_module().entry_computation_name());
  *result.mutable_heap_sizes() = {heap_sizes.begin(), heap_sizes.end()};
  *result.mutable_unpadded_heap_sizes() = {unpadded_heap_sizes.begin(),
                                           unpadded_heap_sizes.end()};
  *result.mutable_max_heap() = {max_heap.begin(), max_heap.end()};
  for (const HeapObject* o : max_heap_by_size) {
    *result.add_max_heap_by_size() = *o;
  }
  *result.mutable_max_heap_to_by_size() = {max_heap_to_by_size.begin(),
                                           max_heap_to_by_size.end()};
  *result.mutable_by_size_to_max_heap() = {by_size_to_max_heap.begin(),
                                           by_size_to_max_heap.end()};
  result.set_peak_heap_mib(BytesToMiB(peak_heap_size_bytes));
  result.set_peak_unpadded_heap_mib(BytesToMiB(unpadded_peak_heap_size_bytes));
  result.set_peak_heap_size_position(peak_heap_size_position);

  for (const auto& item : logical_buffer_spans) {
    (*result.mutable_logical_buffer_spans())[item.first] =
        MakeBufferSpan(item.second.first, item.second.second.value());
  }

  NoteSpecialAllocations(all_buffer_allocations, id_to_logical_buffer,
                         name_to_hlo, small_buffer_size, &result);
  return result;
}

// From a list of heap simulator traces, identify the one that has the largest
// number of HBM (color = 0) memory events.
// If unable to find the heap simulator trace, return -1, and
// ConvertHloProtoToPreprocessResult will not consider heap_simulator_traces
// during preprocess.
int64_t GetHeapSimulatorTraceIdFromEvents(const HloProto& proto) {
  absl::flat_hash_map<int64_t, const xla::LogicalBufferProto*>
      id_to_logical_buffer;
  for (const auto& logical_buffer :
       proto.buffer_assignment().logical_buffers()) {
    id_to_logical_buffer[logical_buffer.id()] = &logical_buffer;
  }
  int64_t best_index = -1;
  int64_t best_event_count = 0;
  for (int64_t i = 0;
       i < proto.buffer_assignment().heap_simulator_traces_size(); i++) {
    const auto& heap_simulator_trace =
        proto.buffer_assignment().heap_simulator_traces(i);
    int64_t event_count = 0;
    for (const auto& event : heap_simulator_trace.events()) {
      const auto iter = id_to_logical_buffer.find(event.buffer_id());
      if (iter == id_to_logical_buffer.end()) {
        continue;
      }
      // TODO(tianrun): Add a "memory space color" query parameter.
      if (iter->second->color() == 0) {
        event_count++;
      }
    }
    if (event_count > best_event_count) {
      best_index = i;
      best_event_count = event_count;
    }
  }

  return best_index;
}

// Tries to get the correct heap simulator trace based on
// buffer_allocation_index.
int64_t GetHeapSimulatorTraceIdFromBufferAllocationIndex(
    const HloProto& proto) {
  absl::flat_hash_map<int64_t, const xla::BufferAllocationProto*>
      id_to_buffer_allocation;
  for (const auto& buffer_allocation :
       proto.buffer_assignment().buffer_allocations()) {
    id_to_buffer_allocation[buffer_allocation.index()] = &buffer_allocation;
  }
  for (int64_t i = 0;
       i < proto.buffer_assignment().heap_simulator_traces_size(); ++i) {
    int64_t buffer_allocation_index = proto.buffer_assignment()
                                          .heap_simulator_traces(i)
                                          .buffer_allocation_index();
    const auto iter = id_to_buffer_allocation.find(buffer_allocation_index);
    if (buffer_allocation_index && iter != id_to_buffer_allocation.end()) {
      // TODO(tianrun): Add a "memory space color" query parameter.
      // Find the heap simulator trace that corresponds to the HLO temporaries
      // buffer allocation, where is_thread_local,
      // is_entry_computation_parameter, is_constant, and maybe_live_out will
      // all be false.
      const auto* buffer_allocation = iter->second;
      if (buffer_allocation->color() == 0 &&
          !buffer_allocation->is_thread_local() &&
          !buffer_allocation->is_entry_computation_parameter() &&
          !buffer_allocation->is_constant() &&
          !buffer_allocation->maybe_live_out()) {
        return i;
      }
    }
  }
  return -1;
}

int64_t GetHeapSimulatorTraceId(const HloProto& proto) {
  int64_t id = GetHeapSimulatorTraceIdFromBufferAllocationIndex(proto);
  if (id != -1) {
    return id;
  }
  return GetHeapSimulatorTraceIdFromEvents(proto);
}

}  // namespace profiler
}  // namespace tensorflow
