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
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/protobuf/memory_viewer_preprocess.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using absl::StrFormat;
using ::xla::BufferAllocationProto;
using ::xla::HeapSimulatorTrace;
using ::xla::HloInstructionProto;
using ::xla::HloProto;
using ::xla::LayoutUtil;
using ::xla::LogicalBufferProto;
using ::xla::Shape;
using ::xla::ShapeUtil;

const Shape* ResolveShapeIndex(const Shape* shape,
                               absl::Span<const int64_t> shape_index) {
  for (int64_t value : shape_index) {
    shape = &shape->tuple_shapes(value);
  }
  return shape;
}

// A wrapper of HLO BufferAssignment, with lookup maps for logical buffers and
// buffer allocations.
class HloProtoBufferWrapper {
 public:
  explicit HloProtoBufferWrapper(const ::xla::HloProto& hlo_proto)
      : hlo_proto_(hlo_proto) {
    Init();
  }

  // Get the heap simulator trace ID using memory color.
  // If unable to find the heap simulator trace, return -1.
  int64_t GetHeapSimulatorTraceId(const int64_t memory_color) const {
    int64_t id = GetHeapSimulatorTraceIdFromBufferAllocationIndex(memory_color);
    if (id != -1) {
      return id;
    }
    return GetHeapSimulatorTraceIdFromEvents(memory_color);
  }

  // Get the raw HLO proto.
  const ::xla::HloProto& GetHloProto() const { return hlo_proto_; }

  // Helper functions to get LogicalBuffer and BufferAllocation.
  // We use map.at() directly in these function assuming the HLO proto is
  // invalid.
  const xla::LogicalBufferProto& GetLogicalBuffer(
      int64_t logical_buffer_id) const {
    return *id_to_logical_buffer_.at(logical_buffer_id);
  }

  const xla::BufferAllocationProto& GetBufferAllocation(
      const xla::LogicalBufferProto& logical_buffer) const {
    return *logical_buffer_to_buffer_allocation_.at(&logical_buffer);
  }

  const xla::HloInstructionProto& GetHloInstruction(
      const xla::LogicalBufferProto& logical_buffer) const {
    return *name_to_hlo_.at(logical_buffer.defined_at().instruction_name());
  }

  const std::vector<const ::xla::LogicalBufferProto*>&
  GetLogicalBuffersFromBufferAllocation(
      const xla::BufferAllocationProto& buffer_allocation) const {
    return buffer_allocation_to_logical_buffers_.at(&buffer_allocation);
  }

  const xla::Shape& GetLogicalBufferShape(
      const xla::LogicalBufferProto& logical_buffer) const {
    return logical_buffer_to_shape_.at(&logical_buffer);
  }

 private:
  // Initialize the mappings of logical buffers and buffer allocations.
  void Init() {
    for (const auto& computation : hlo_proto_.hlo_module().computations()) {
      for (const auto& instruction : computation.instructions()) {
        name_to_hlo_[instruction.name()] = &instruction;
      }
    }
    for (const auto& logical_buffer :
         hlo_proto_.buffer_assignment().logical_buffers()) {
      id_to_logical_buffer_[logical_buffer.id()] = &logical_buffer;
      // Get shape of logical buffer.
      const auto& instruction_name =
          logical_buffer.defined_at().instruction_name();
      const Shape top_level_shape(name_to_hlo_.at(instruction_name)->shape());
      const Shape* shape = ResolveShapeIndex(
          &top_level_shape, logical_buffer.defined_at().shape_index());
      logical_buffer_to_shape_[&logical_buffer] = *shape;
    }
    for (const auto& buffer_allocation :
         hlo_proto_.buffer_assignment().buffer_allocations()) {
      for (const auto& assigned : buffer_allocation.assigned()) {
        const auto* logical_buffer =
            id_to_logical_buffer_.at(assigned.logical_buffer_id());
        buffer_allocation_to_logical_buffers_[&buffer_allocation].push_back(
            logical_buffer);
        logical_buffer_to_buffer_allocation_[logical_buffer] =
            &buffer_allocation;
      }
    }
  }

  // From a list of heap simulator traces, identify the one that has the largest
  // number of memory events with color <memory_color>.
  int64_t GetHeapSimulatorTraceIdFromEvents(const int64_t memory_color) const {
    int64_t best_index = -1;
    int64_t best_event_count = 0;
    for (int64_t i = 0;
         i < hlo_proto_.buffer_assignment().heap_simulator_traces_size(); i++) {
      const auto& heap_simulator_trace =
          hlo_proto_.buffer_assignment().heap_simulator_traces(i);
      int64_t event_count = 0;
      for (const auto& event : heap_simulator_trace.events()) {
        const LogicalBufferProto& logical_buffer =
            GetLogicalBuffer(event.buffer_id());
        if (logical_buffer.color() == memory_color) {
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

  // Tries to get heap simulator trace based on buffer_allocation_index.
  int64_t GetHeapSimulatorTraceIdFromBufferAllocationIndex(
      const int64_t memory_color) const {
    absl::flat_hash_map<int64_t, const BufferAllocationProto*>
        id_to_buffer_allocation;
    for (const auto& buffer_allocation :
         hlo_proto_.buffer_assignment().buffer_allocations()) {
      id_to_buffer_allocation[buffer_allocation.index()] = &buffer_allocation;
    }
    for (int64_t i = 0;
         i < hlo_proto_.buffer_assignment().heap_simulator_traces_size(); i++) {
      int64_t buffer_allocation_index = hlo_proto_.buffer_assignment()
                                            .heap_simulator_traces(i)
                                            .buffer_allocation_index();
      const auto iter = id_to_buffer_allocation.find(buffer_allocation_index);
      if (buffer_allocation_index && iter != id_to_buffer_allocation.end()) {
        // Find the heap simulator trace that corresponds to the HLO temporaries
        // buffer allocation, where is_thread_local,
        // is_entry_computation_parameter, is_constant, and maybe_live_out will
        // all be false.
        const auto* buffer_allocation = iter->second;
        if (buffer_allocation->color() == memory_color &&
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

  // Reference to the original HLO proto.
  const ::xla::HloProto& hlo_proto_;

  // A mapping from name to HLO instruction.
  absl::flat_hash_map<absl::string_view, const ::xla::HloInstructionProto*>
      name_to_hlo_;

  // A mapping from logical buffer ID to logical buffer.
  absl::flat_hash_map<int64_t, const ::xla::LogicalBufferProto*>
      id_to_logical_buffer_;

  // A mapping from logical buffer to the buffer allocation it exists inside
  // (there must be only one).
  absl::flat_hash_map<const ::xla::LogicalBufferProto*,
                      const ::xla::BufferAllocationProto*>
      logical_buffer_to_buffer_allocation_;

  // The reverse mapping from buffer allocation to all the logical buffers that
  // exist inside it.
  absl::flat_hash_map<const ::xla::BufferAllocationProto*,
                      std::vector<const ::xla::LogicalBufferProto*>>
      buffer_allocation_to_logical_buffers_;

  // A mapping from logical buffer to its shape.
  absl::flat_hash_map<const ::xla::LogicalBufferProto*, xla::Shape>
      logical_buffer_to_shape_;
};

double BytesToMiB(int64_t bytes) {
  return static_cast<double>(bytes) / tensorflow::MathUtil::IPow(2, 20);
}

// Get buffer allocation property.
std::string GetAllocationGroupName(
    const BufferAllocationProto& buffer_allocation) {
  if (buffer_allocation.is_entry_computation_parameter()) {
    return "Parameter";
  } else if (buffer_allocation.maybe_live_out()) {
    return "Output";
  } else if (buffer_allocation.is_thread_local()) {
    return "Thread-local";
  } else {
    return "Temporary";
  }
}

// Get the instruction name with shape index for a logical buffer.
std::string GetInstructionNameWithShapeIndex(
    const LogicalBufferProto& logical_buffer) {
  if (logical_buffer.defined_at().shape_index().empty()) {
    return logical_buffer.defined_at().instruction_name();
  } else {
    return absl::StrCat(
        logical_buffer.defined_at().instruction_name(), "{",
        absl::StrJoin(logical_buffer.defined_at().shape_index(), ""), "}");
  }
}

// A wrapper around ShapeUtil::ByteSizeOf that clears out the layout/padding,
// since that is considered in the ByteSizeOf calculation.
int64_t ShapeUnpaddedSize(Shape shape) {
  // Ensure the layout has no padding by making it the default layout.
  LayoutUtil::SetToDefaultLayout(&shape);
  // Note: we make a simplifying assumption here that a "minimal" size for a
  // tuple member would be the size of a `void*` -- there may be even fancier
  // ways of doing things, but this should give a good enough approximation of
  // what a minimal tuple size is.
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/sizeof(void*));
}

std::string ShapeDescription(const Shape& shape) {
  return ShapeUtil::HumanStringWithLayout(shape);
}

HeapObject MakeHeapObjectCommon(std::string label, int32_t color,
                                int64_t logical_buffer_id,
                                int64_t logical_buffer_size_bytes,
                                int64_t unpadded_shape_bytes) {
  HeapObject result;
  result.set_numbered(color);
  result.set_label(std::move(label));
  result.set_logical_buffer_id(logical_buffer_id);
  result.set_logical_buffer_size_mib(BytesToMiB(logical_buffer_size_bytes));
  result.set_unpadded_shape_mib(BytesToMiB(unpadded_shape_bytes));
  return result;
}

HeapObject MakeHeapObject(const HloProtoBufferWrapper& wrapper,
                          const LogicalBufferProto& logical_buffer,
                          const BufferAllocationProto& buffer_allocation,
                          int32_t color) {
  const Shape& shape = wrapper.GetLogicalBufferShape(logical_buffer);
  const HloInstructionProto& hlo_instruction =
      wrapper.GetHloInstruction(logical_buffer);
  std::string shape_string = ShapeDescription(shape);
  int64_t unpadded_shape_bytes = ShapeUnpaddedSize(shape);
  std::string label = absl::StrFormat(
      "%s: %s # %s", logical_buffer.defined_at().instruction_name(),
      shape_string, hlo_instruction.metadata().op_name());
  HeapObject result =
      MakeHeapObjectCommon(std::move(label), color, logical_buffer.id(),
                           logical_buffer.size(), unpadded_shape_bytes);
  result.set_numbered(color);
  result.set_instruction_name(GetInstructionNameWithShapeIndex(logical_buffer));
  result.set_group_name(GetAllocationGroupName(buffer_allocation));
  result.set_tf_op_name(hlo_instruction.metadata().op_name());
  result.set_shape_string(shape_string);
  result.set_op_code(hlo_instruction.opcode());
  return result;
}

BufferSpan MakeBufferSpan(int32 start, int32 limit) {
  BufferSpan result;
  result.set_start(start);
  result.set_limit(limit);
  return result;
}

std::string BufferAllocationDescription(
    const BufferAllocationProto& buffer_allocation) {
  // Clear out the assigned logical buffers when stringifying the buffer
  // allocation, as it can be a long list.
  auto copy = buffer_allocation;
  copy.mutable_assigned()->Clear();
  return copy.ShortDebugString();
}

void Convert(const xla::BufferAllocationProto_Assigned& assigned,
             const HloProtoBufferWrapper& wrapper, LogicalBuffer* result) {
  result->set_id(assigned.logical_buffer_id()),
      result->set_size_mib(BytesToMiB(assigned.size()));
  const LogicalBufferProto& proto =
      wrapper.GetLogicalBuffer(assigned.logical_buffer_id());
  const std::string& instruction_name = proto.defined_at().instruction_name();
  result->set_hlo_name(instruction_name);
  result->mutable_shape_index()->CopyFrom(proto.defined_at().shape_index());
  const Shape& shape = wrapper.GetLogicalBufferShape(proto);
  result->set_shape(ShapeDescription(shape));
}

bool IsReusable(const BufferAllocationProto& buffer_allocation) {
  return !buffer_allocation.is_thread_local() && !buffer_allocation.is_tuple();
}

void Convert(const BufferAllocationProto& proto,
             const HloProtoBufferWrapper& wrapper, BufferAllocation* result) {
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
    Convert(assigned, wrapper, result->add_logical_buffers());
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

void NoteSpecialAllocations(const HloProtoBufferWrapper& wrapper,
                            int64_t small_buffer_size,
                            PreprocessResult* result) {
  int64_t entry_parameters_bytes = 0;
  int64_t non_reusable_bytes = 0;
  int64_t maybe_live_out_bytes = 0;
  for (const BufferAllocationProto& buffer_allocation :
       wrapper.GetHloProto().buffer_assignment().buffer_allocations()) {
    if (buffer_allocation.is_entry_computation_parameter()) {
      entry_parameters_bytes += buffer_allocation.size();
    }
    if (!IsReusable(buffer_allocation)) {
      non_reusable_bytes += buffer_allocation.size();
    }
    if (buffer_allocation.maybe_live_out()) {
      if (buffer_allocation.size() > small_buffer_size) {
        VLOG(1) << "Maybe live out buffer allocation: "
                << buffer_allocation.size()
                << " bytes :: " << buffer_allocation.ShortDebugString();
      }
      maybe_live_out_bytes += buffer_allocation.size();
    }
    Convert(buffer_allocation, wrapper, result->add_indefinite_lifetimes());
  }

  result->set_entry_computation_parameters_mib(
      BytesToMiB(entry_parameters_bytes));
  result->set_non_reusable_mib(BytesToMiB(non_reusable_bytes));
  result->set_maybe_live_out_mib(BytesToMiB(maybe_live_out_bytes));
}

// Memory usage statistics collected from heap simulator trace.
struct HeapSimulatorStats {
  explicit HeapSimulatorStats(const HloProtoBufferWrapper& wrapper)
      : wrapper(wrapper) {}

  void SetSimulatorTraceEventSize(int64_t size) {
    simulator_trace_event_size = size;
  }

  // Update stats for general simulator event.
  void UpdateOnSimulatorEvent(const HeapSimulatorTrace::Event& event) {
    // Update memory timelines and seen buffers.
    heap_size_bytes_timeline.push_back(heap_size_bytes);
    unpadded_heap_size_bytes_timeline.push_back(unpadded_heap_size_bytes);
    const auto& logical_buffer = wrapper.GetLogicalBuffer(event.buffer_id());
    seen_logical_buffers.insert(&logical_buffer);
    seen_buffer_allocations.insert(
        &wrapper.GetBufferAllocation(logical_buffer));
  }

  // Update stats when memory usage increase.
  void IncreaseMemoryUsage(const LogicalBufferProto& canonical_logical_buffer,
                           bool init_buffer_span) {
    logical_buffers.push_back(canonical_logical_buffer.id());
    heap_size_bytes += canonical_logical_buffer.size();
    const Shape& shape =
        wrapper.GetLogicalBufferShape(canonical_logical_buffer);
    unpadded_heap_size_bytes += ShapeUnpaddedSize(shape);

    // Increase peak memory usage if needed.
    int64_t prior_peak_heap_size_bytes = peak_heap_size_bytes;
    peak_heap_size_bytes = std::max(peak_heap_size_bytes, heap_size_bytes);
    if (prior_peak_heap_size_bytes != peak_heap_size_bytes) {
      peak_heap_size_position = heap_size_bytes_timeline.size() - 1;
      peak_unpadded_heap_size_bytes = unpadded_heap_size_bytes;
      VLOG(1) << absl::StrFormat("New peak heap size on %d :: %d bytes",
                                 peak_heap_size_position, peak_heap_size_bytes);
      peak_logical_buffers = logical_buffers;
    }
    // Initialize the buffer lifespan if needed.
    if (init_buffer_span) {
      // Initialize the buffer span from the current event to the last event in
      // heap simulator trace.
      logical_buffer_spans[canonical_logical_buffer.id()] = {
          heap_size_bytes_timeline.size() - 1, simulator_trace_event_size - 1};
    }
  }

  // Update stats when memory usage decrease.
  Status DecreaseMemoryUsage(
      const LogicalBufferProto& canonical_logical_buffer) {
    int64_t canonical_buffer_id = canonical_logical_buffer.id();
    logical_buffers.erase(
        std::remove(logical_buffers.begin(), logical_buffers.end(),
                    canonical_buffer_id),
        logical_buffers.end());
    heap_size_bytes -= canonical_logical_buffer.size();
    if (heap_size_bytes < 0) {
      return errors::InvalidArgument(absl::StrCat(
          "Heap size should be non-negative, but get: ", heap_size_bytes));
    }
    const Shape& shape =
        wrapper.GetLogicalBufferShape(canonical_logical_buffer);
    unpadded_heap_size_bytes -= ShapeUnpaddedSize(shape);
    // Mark the end of this buffer.
    logical_buffer_spans[canonical_buffer_id].second =
        heap_size_bytes_timeline.size() - 1;
    return OkStatus();
  }

  // Finalize the memory usage stats from heap simulator trace.
  Status FinalizeMemoryUsage() {
    // Add the final heap size after simulating the entire heap trace.
    heap_size_bytes_timeline.push_back(heap_size_bytes);
    unpadded_heap_size_bytes_timeline.push_back(unpadded_heap_size_bytes);

    if (seen_buffer_allocations.size() != 1) {
      return errors::InvalidArgument(
          absl::StrCat("All heap simulation should work out of a single buffer "
                       "allocation, actual seen_buffer_allocations.size():",
                       seen_buffer_allocations.size()));
    }

    // Log stats.
    VLOG(1) << "Found " << peak_logical_buffers.size()
            << " logical buffers alive at point of peak heap usage.";

    VLOG(1) << "Peak logical buffers: ["
            << absl::StrJoin(peak_logical_buffers, ", ") << "]";

    return OkStatus();
  }

  // Get the logical buffers with indefinite lifetime (buffers that do not
  // appear in heap simulator trace, so the profiler does not know its exact
  // lifetime span). They will be handled separately by
  // ProcessIndefiniteLifetimeBuffers.
  std::vector<const LogicalBufferProto*> LogicalBuffersWithIndefiniteLifetime()
      const {
    std::vector<const LogicalBufferProto*> indefinite_logical_buffers;
    for (const auto& logical_buffer :
         wrapper.GetHloProto().buffer_assignment().logical_buffers()) {
      if (!seen_logical_buffers.contains(&logical_buffer)) {
        indefinite_logical_buffers.push_back(&logical_buffer);
      }
    }
    return indefinite_logical_buffers;
  }

  // Keep track of memory usage when iterating through heap simulator trace
  // events.
  int64_t heap_size_bytes = 0;
  int64_t unpadded_heap_size_bytes = 0;
  // Memory usage at peak.
  int64_t peak_heap_size_bytes = 0;
  int64_t peak_unpadded_heap_size_bytes = 0;

  // Keep track of logical buffer IDs when iterating through heap simulator
  // trace events.
  std::vector<int64_t> logical_buffers;
  // Logical buffer IDs at peak.
  std::vector<int64_t> peak_logical_buffers;

  // Heap size timeline.
  std::vector<int64_t> heap_size_bytes_timeline;
  std::vector<int64_t> unpadded_heap_size_bytes_timeline;

  // Position of peak memory usage in the timeline.
  int64_t peak_heap_size_position = 0;

  // Logical buffers and buffer allocations that exists in heap simulator trace.
  absl::flat_hash_set<const LogicalBufferProto*> seen_logical_buffers;
  absl::flat_hash_set<const BufferAllocationProto*> seen_buffer_allocations;

  // Lifetime span of logical buffer.
  absl::flat_hash_map<int64_t, std::pair<int64_t, int64_t>>
      logical_buffer_spans;

  // Constants while iterating through heap simulator trace.
  const HloProtoBufferWrapper& wrapper;
  int64_t simulator_trace_event_size;
};

// Tracker for logical buffer sharing.
class LogicalBufferShareTracker {
 public:
  // Canonical logical buffer ID and its ref count.
  struct BufferRefCount {
    int64_t canonical_buffer_id;
    int32_t ref_count;
    BufferRefCount(int64_t id, int32_t count)
        : canonical_buffer_id(id), ref_count(count) {}
  };

  // Process ALLOC event.
  void ProcessAllocEvent(const HeapSimulatorTrace::Event& event) {
    // The first time a canonical buffer is allocated.
    canonical_buffer_ref_count_[event.buffer_id()] = 1;
  }

  // Process FREE event, return canonical buffer and its ref count.
  StatusOr<BufferRefCount> ProcessFreeEvent(
      const HeapSimulatorTrace::Event& event) {
    // Get the canonical buffer ID of this free event.
    int64_t canonical_buffer_id = event.buffer_id();
    if (const int64_t* canonical_id =
            gtl::FindOrNull(share_with_to_canonical_, event.buffer_id())) {
      canonical_buffer_id = *canonical_id;
    }
    // Decrease the ref count of canonical buffer.
    int32_t& ref_count = canonical_buffer_ref_count_[canonical_buffer_id];
    --ref_count;
    if (ref_count < 0) {
      return errors::InvalidArgument(absl::StrCat(
          "Buffer ", canonical_buffer_id, "is freed multiple times."));
    }
    return BufferRefCount(canonical_buffer_id, ref_count);
  }

  // Process SHARE_WITH event, return canonical buffer and its ref count.
  BufferRefCount ProcessShareWithEvent(const HeapSimulatorTrace::Event& event) {
    int64_t canonical_buffer_id = event.share_with_canonical_id();
    share_with_to_canonical_[event.buffer_id()] = canonical_buffer_id;
    // Increase the ref count of canonical buffer.
    int32_t& ref_count = canonical_buffer_ref_count_[canonical_buffer_id];
    ++ref_count;
    return BufferRefCount(canonical_buffer_id, ref_count);
  }

 private:
  // Map from the logical buffer ID of the SHARE_WITH buffer to the logical
  // buffer ID of the canonical buffer being shared.
  absl::flat_hash_map<int64_t, int64_t> share_with_to_canonical_;
  // Number of times a canonical buffer is referenced.
  absl::flat_hash_map<int64_t, int32_t> canonical_buffer_ref_count_;
};

Status ProcessHeapSimulatorTrace(const HloProtoBufferWrapper& wrapper,
                                 const int64_t memory_color,
                                 int64_t heap_simulator_trace_id,
                                 HeapSimulatorStats* stats) {
  // If heap simulator trace id is not explicitly set by user, the profiler will
  // try to infer the heap simulator trace id from <memory_color>.
  if (heap_simulator_trace_id == -1) {
    heap_simulator_trace_id = wrapper.GetHeapSimulatorTraceId(memory_color);
  }
  // If still unable to get a valid heap simulator trace id, skip heap simulator
  // trace and process the rest of the buffers.
  if (heap_simulator_trace_id < 0 ||
      heap_simulator_trace_id >= wrapper.GetHloProto()
                                     .buffer_assignment()
                                     .heap_simulator_traces_size()) {
    return OkStatus();
  }

  // Run through all the simulator events in the given trace, and simulate the
  // heap in order to find the point of peak memory usage and record its
  // associated metadata.
  const auto& trace =
      wrapper.GetHloProto().buffer_assignment().heap_simulator_traces(
          heap_simulator_trace_id);

  LogicalBufferShareTracker share_tracker;
  stats->SetSimulatorTraceEventSize(trace.events_size());
  for (const auto& event : trace.events()) {
    stats->UpdateOnSimulatorEvent(event);
    if (event.kind() == HeapSimulatorTrace::Event::ALLOC) {
      share_tracker.ProcessAllocEvent(event);
      // ALLOC event increases memory usage and initializes the buffer lifetime
      // span.
      const LogicalBufferProto& logical_buffer =
          wrapper.GetLogicalBuffer(event.buffer_id());
      stats->IncreaseMemoryUsage(logical_buffer,
                                 /*init_buffer_span=*/true);
    } else if (event.kind() == HeapSimulatorTrace::Event::FREE) {
      TF_ASSIGN_OR_RETURN(auto buffer_ref_count,
                          share_tracker.ProcessFreeEvent(event));
      if (buffer_ref_count.ref_count == 0) {
        // There is no more reference to the canonical buffer, the canonical
        // buffer is finally freed. Update memory usage and memory timespan
        // using the metadata of canonical buffer.
        const LogicalBufferProto& canonical_buffer =
            wrapper.GetLogicalBuffer(buffer_ref_count.canonical_buffer_id);
        TF_RETURN_IF_ERROR(stats->DecreaseMemoryUsage(canonical_buffer));
      }
    } else if (event.kind() == HeapSimulatorTrace::Event::SHARE_WITH) {
      auto buffer_ref_count = share_tracker.ProcessShareWithEvent(event);
      if (buffer_ref_count.ref_count == 1) {
        // SHARE_WITH happens after the FREE of a canonical buffer.
        const LogicalBufferProto& canonical_buffer =
            wrapper.GetLogicalBuffer(buffer_ref_count.canonical_buffer_id);
        // SHARE_WITH event does not initialize buffer lifetime span, it was
        // initialized by ALLOC event using the canonical logical buffer.
        stats->IncreaseMemoryUsage(canonical_buffer,
                                   /*init_buffer_span=*/false);
      }
    } else {
      return errors::InvalidArgument(
          absl::StrCat("Unhandled event kind: ", event.kind()));
    }
  }
  TF_RETURN_IF_ERROR(stats->FinalizeMemoryUsage());
  return OkStatus();
}

// The stats when processing buffer allocations and logical buffers.
struct BufferStats {
  BufferStats(const HloProtoBufferWrapper& wrapper,
              const HeapSimulatorStats& simulator_stats,
              int64_t small_buffer_size)
      : wrapper(wrapper),
        simulator_stats(simulator_stats),
        small_buffer_size(small_buffer_size) {}

  // Add a HeapObject derived from logical buffer and buffer allocation.
  void AddHeapObject(const LogicalBufferProto& logical_buffer,
                     const BufferAllocationProto& buffer_allocation) {
    if (logical_buffer.size() < small_buffer_size) {
      // Accumulate small buffers, don't make a HeapObject.
      total_small_buffer_size_bytes += logical_buffer.size();
    } else {
      // Make a new HeapObject, assign a new color to visualize it.
      max_heap_objects.push_back(MakeHeapObject(wrapper, logical_buffer,
                                                buffer_allocation, colorno++));
    }
  }

  void FinalizeBufferUsage() {
    // Buffers from HeapSimulatorTrace.
    for (const int64_t logical_buffer_id :
         simulator_stats.peak_logical_buffers) {
      const LogicalBufferProto& logical_buffer =
          wrapper.GetLogicalBuffer(logical_buffer_id);
      const BufferAllocationProto& buffer_allocation =
          wrapper.GetBufferAllocation(logical_buffer);
      AddHeapObject(logical_buffer, buffer_allocation);
    }

    // Make a single HeapObject out of all the small buffers.
    if (total_small_buffer_size_bytes != 0) {
      max_heap_objects.push_back(MakeHeapObjectCommon(
          absl::StrFormat("small (<%d bytes)", small_buffer_size), colorno++,
          /*logical_buffer_id=*/-1, total_small_buffer_size_bytes,
          /*unpadded_shape_bytes=*/0));
    }
  }

  // All the HeapObjects at peak memory time.
  std::vector<HeapObject> max_heap_objects;
  // The total size of all memory buffers with indefinite lifetime.
  int64_t indefinite_memory_usage_bytes = 0;
  // The accumulated size of all small buffers.
  int64_t total_small_buffer_size_bytes = 0;
  // Tracker of memory viewer color.
  int32_t colorno = 0;

  const HloProtoBufferWrapper& wrapper;
  const HeapSimulatorStats& simulator_stats;
  const int64_t small_buffer_size;
};

void ProcessIndefiniteLifetimeBuffers(const HloProtoBufferWrapper& wrapper,
                                      const HeapSimulatorStats& simulator_stats,
                                      int64_t memory_color,
                                      BufferStats* buffer_stats) {
  absl::flat_hash_set<const BufferAllocationProto*> seen_buffer_allocations =
      simulator_stats.seen_buffer_allocations;
  for (const LogicalBufferProto* logical_buffer :
       simulator_stats.LogicalBuffersWithIndefiniteLifetime()) {
    const BufferAllocationProto& buffer_allocation =
        wrapper.GetBufferAllocation(*logical_buffer);
    if (buffer_allocation.is_thread_local() ||
        logical_buffer->color() != memory_color) {
      continue;
    }
    if (seen_buffer_allocations.insert(&buffer_allocation).second) {
      buffer_stats->indefinite_memory_usage_bytes += buffer_allocation.size();
      const auto& logical_buffers =
          wrapper.GetLogicalBuffersFromBufferAllocation(buffer_allocation);
      if (logical_buffers.size() == 1) {
        buffer_stats->AddHeapObject(*logical_buffer, buffer_allocation);
      } else {
        VLOG(1) << "Indefinite lifetime, no heap object shown due to "
                << "multiple logical buffers in buffer allocation: "
                << logical_buffer->ShortDebugString()
                << " :: " << BufferAllocationDescription(buffer_allocation);
      }
      if (buffer_allocation.size() < buffer_stats->small_buffer_size) {
        VLOG(1) << "Indefinite memory usage now: "
                << buffer_stats->indefinite_memory_usage_bytes << " bytes (+"
                << buffer_allocation.size() << " bytes)";
      }
    }
  }

  buffer_stats->FinalizeBufferUsage();
}

void GeneratePreprocessResult(const HloProtoBufferWrapper& wrapper,
                              const HeapSimulatorStats& simulator_stats,
                              const BufferStats& buffer_stats,
                              PreprocessResult* result) {
  // Module info.
  result->set_module_name(wrapper.GetHloProto().hlo_module().name());
  result->set_entry_computation_name(
      wrapper.GetHloProto().hlo_module().entry_computation_name());

  // Build HeapObjects and index.
  std::vector<const HeapObject*> max_heap_by_size;
  max_heap_by_size.reserve(buffer_stats.max_heap_objects.size());
  for (const auto& object : buffer_stats.max_heap_objects) {
    max_heap_by_size.push_back(&object);
  }
  std::sort(max_heap_by_size.begin(), max_heap_by_size.end(),
            [](const HeapObject* a, const HeapObject* b) {
              return a->logical_buffer_size_mib() >
                     b->logical_buffer_size_mib();
            });

  std::vector<int> max_heap_to_by_size;
  max_heap_to_by_size.reserve(max_heap_by_size.size());
  for (const auto& object : buffer_stats.max_heap_objects) {
    auto it =
        std::find(max_heap_by_size.begin(), max_heap_by_size.end(), &object);
    int index = std::distance(max_heap_by_size.begin(), it);
    max_heap_to_by_size.push_back(index);
  }

  std::vector<int> by_size_to_max_heap;
  for (const auto* object : max_heap_by_size) {
    int index = object - &buffer_stats.max_heap_objects[0];
    by_size_to_max_heap.push_back(index);
  }

  *result->mutable_max_heap() = {buffer_stats.max_heap_objects.begin(),
                                 buffer_stats.max_heap_objects.end()};
  result->mutable_max_heap_by_size()->Reserve(max_heap_by_size.size());
  for (const HeapObject* o : max_heap_by_size) {
    *result->add_max_heap_by_size() = *o;
  }
  *result->mutable_max_heap_to_by_size() = {max_heap_to_by_size.begin(),
                                            max_heap_to_by_size.end()};
  *result->mutable_by_size_to_max_heap() = {by_size_to_max_heap.begin(),
                                            by_size_to_max_heap.end()};

  // For the buffers that have indefinite lifetime (that is, lifetime not
  // reflected by the heap simulation) add it to the peak values and the vectors
  // of heap sizes.
  size_t timeline_size = simulator_stats.heap_size_bytes_timeline.size();
  double add_mib = BytesToMiB(buffer_stats.indefinite_memory_usage_bytes);
  result->mutable_heap_sizes()->Reserve(timeline_size);
  result->mutable_unpadded_heap_sizes()->Reserve(timeline_size);
  for (size_t i = 0; i < timeline_size; i++) {
    result->add_heap_sizes(
        BytesToMiB(simulator_stats.heap_size_bytes_timeline[i]) + add_mib);
    result->add_unpadded_heap_sizes(
        BytesToMiB(simulator_stats.unpadded_heap_size_bytes_timeline[i]) +
        add_mib);
  }

  result->set_peak_heap_mib(BytesToMiB(simulator_stats.peak_heap_size_bytes) +
                            add_mib);
  result->set_peak_unpadded_heap_mib(
      BytesToMiB(simulator_stats.peak_unpadded_heap_size_bytes) + add_mib);
  result->set_peak_heap_size_position(simulator_stats.peak_heap_size_position);

  // Build buffer lifespan.
  for (const auto& item : simulator_stats.logical_buffer_spans) {
    (*result->mutable_logical_buffer_spans())[item.first] =
        MakeBufferSpan(item.second.first, item.second.second);
  }

  NoteSpecialAllocations(wrapper, buffer_stats.small_buffer_size, result);
}

}  // namespace

absl::StatusOr<PreprocessResult> ConvertHloProtoToPreprocessResult(
    const HloProto& hlo_proto, int64_t small_buffer_size,
    int64_t heap_simulator_trace_id, int64_t memory_color) {
  HloProtoBufferWrapper wrapper(hlo_proto);

  // Process heap simulator trace.
  HeapSimulatorStats simulator_stats(wrapper);
  auto status = ProcessHeapSimulatorTrace(
      wrapper, memory_color, heap_simulator_trace_id, &simulator_stats);
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to process heap simulator trace: ", status.error_message()));
  }

  // Process buffers with indefinite lifetime.
  BufferStats buffer_stats(wrapper, simulator_stats, small_buffer_size);
  ProcessIndefiniteLifetimeBuffers(wrapper, simulator_stats, memory_color,
                                   &buffer_stats);

  PreprocessResult result;
  GeneratePreprocessResult(wrapper, simulator_stats, buffer_stats, &result);
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
