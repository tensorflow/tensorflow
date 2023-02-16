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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/protobuf/memory_viewer_preprocess.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

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

std::string ShapeDescription(const Shape& shape) {
  return ShapeUtil::HumanStringWithLayout(shape);
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

class BufferAllocationStruct {
 public:
  explicit BufferAllocationStruct(const BufferAllocationProto& proto)
      : buffer_allocation_((proto)) {}
  bool IsIndefinite() const {
    return buffer_allocation_.is_thread_local() ||
           buffer_allocation_.is_entry_computation_parameter() ||
           buffer_allocation_.is_constant() ||
           buffer_allocation_.maybe_live_out();
  }
  const BufferAllocationProto& proto() const { return buffer_allocation_; }
  size_t size() const { return buffer_allocation_.size(); }
  int64_t color() const { return buffer_allocation_.color(); }
  int64_t index() const { return buffer_allocation_.index(); }
  std::optional<int64_t> heap_simulator_trace_id() const {
    return heap_simulator_trace_id_;
  }
  void set_heap_simulator_trace_id(int64_t id) {
    heap_simulator_trace_id_ = id;
  }

  // Get buffer allocation category.
  std::string category() const {
    if (buffer_allocation_.is_entry_computation_parameter()) {
      return "Parameter";
    } else if (buffer_allocation_.maybe_live_out()) {
      return "Output";
    } else if (buffer_allocation_.is_thread_local()) {
      return "Thread-local";
    } else if (buffer_allocation_.is_constant()) {
      return "Constant";
    } else {
      return "Temporary";
    }
  }

  std::string description() const {
    return absl::StrFormat(
        "buffer_allocation_id:%d\nsize:%d\nbuffer_counts:%d\n",
        buffer_allocation_.index(), size(), buffer_allocation_.assigned_size());
  }

 private:
  const BufferAllocationProto& buffer_allocation_;
  std::optional<int64_t> heap_simulator_trace_id_;
};

struct LogicalBufferStruct {
  LogicalBufferStruct(const LogicalBufferProto& p,
                      const BufferAllocationStruct& b,
                      const ::xla::HloInstructionProto& i, uint64_t offset)
      : proto(p), buffer_allocation(b), hlo_instruction(i), offset(offset) {
    // Get shape of logical buffer.
    const Shape top_level_shape(hlo_instruction.shape());
    shape =
        *ResolveShapeIndex(&top_level_shape, proto.defined_at().shape_index());
  }

  absl::string_view instruction_name() const { return hlo_instruction.name(); }

  int64_t color() const { return proto.color(); }
  size_t size() const { return proto.size(); }
  size_t unpadded_size() const { return ShapeUnpaddedSize(shape); }

  // reference counting related
  int64_t inc() {
    if (canonical_buffer) return canonical_buffer->inc();
    return ++ref_count;
  }
  int64_t dec() {
    if (canonical_buffer) return canonical_buffer->dec();
    return --ref_count;
  }
  int64_t share_with(LogicalBufferStruct* buffer) {
    canonical_buffer = buffer;
    return canonical_buffer->inc();
  }
  LogicalBufferStruct* get_canonical_buffer() {
    return canonical_buffer ? canonical_buffer->get_canonical_buffer() : this;
  }

  // Get the instruction name with shape index for a logical buffer.
  std::string GetInstructionNameWithShapeIndex() const {
    if (proto.defined_at().shape_index().empty()) {
      return std::string(instruction_name());
    } else {
      return absl::StrCat(instruction_name(), "{",
                          absl::StrJoin(proto.defined_at().shape_index(), ","),
                          "}");
    }
  }

  std::string description() const {
    return absl::StrFormat(
        "buffer_id:%d\nhlo_op:%s\nshape:%s\nsize:%d\nunpadded_size:%d\n"
        "offset:%d\nspan:(%lld,%lld)",
        proto.id(), instruction_name(), ShapeDescription(shape), size(),
        unpadded_size(), offset, span ? span->first : -1,
        span ? span->second : -1);
  }

  const LogicalBufferProto& proto;
  const BufferAllocationStruct& buffer_allocation;
  const ::xla::HloInstructionProto& hlo_instruction;
  uint64_t offset;  // within the buffer allocation;
  // Span within the specific simulator trace.
  std::optional<std::pair<uint64_t, uint64_t>> span;
  xla::Shape shape;
  int64_t ref_count = 0;
  LogicalBufferStruct* canonical_buffer = nullptr;
};

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

  const BufferAllocationStruct& GetBufferAllocation(
      int64_t buffer_allocation_id) const {
    return *id_to_buffer_allocation_.at(buffer_allocation_id);
  }

  std::vector<const BufferAllocationStruct*> GetBufferAllocations(
      int64_t memory_color) const {
    std::vector<const BufferAllocationStruct*> buffer_allocations;
    for (const auto& iter : id_to_buffer_allocation_) {
      if (iter.second->proto().color() != memory_color) continue;
      buffer_allocations.push_back(iter.second.get());
    }
    return buffer_allocations;
  }

  LogicalBufferStruct& GetLogicalBuffer(int64_t logical_buffer_id) const {
    return *id_to_logical_buffer_.at(logical_buffer_id);
  }

  // Get the logical buffers with indefinite lifetime (excluding thread_local).
  std::vector<const LogicalBufferStruct*> LogicalBuffersWithIndefiniteLifetime(
      int64_t memory_color) const {
    std::vector<const LogicalBufferStruct*> indefinite_logical_buffers;

    for (const auto& buffer_assignment : GetBufferAllocations(memory_color)) {
      if (!buffer_assignment->IsIndefinite()) continue;
      if (buffer_assignment->proto().is_thread_local()) continue;
      // A indefinite buffer allocation will contain multiple logical buffers.
      // None of them have a offset, and may have different size than the buffer
      // allocation's size. In most cases, if not all cases, one of the logical
      // buffer will have the size equal to buffer allocation's size. We will
      // pick the biggest logical buffer.
      const LogicalBufferStruct* best_logical_buffer = nullptr;
      size_t best_size = 0;
      for (const auto& assigned : buffer_assignment->proto().assigned()) {
        const auto& logical_buffer_struct =
            GetLogicalBuffer(assigned.logical_buffer_id());
        if (logical_buffer_struct.size() > best_size) {
          best_size = logical_buffer_struct.size();
          best_logical_buffer = &logical_buffer_struct;
        }
      }
      if (best_logical_buffer) {
        indefinite_logical_buffers.push_back(best_logical_buffer);
      }
    }
    return indefinite_logical_buffers;
  }

 private:
  // Initialize the mappings of logical buffers and buffer allocations.
  void Init() {
    // A mapping from name to HLO instruction.
    absl::flat_hash_map<absl::string_view, const ::xla::HloInstructionProto*>
        name_to_hlo;
    absl::flat_hash_map<uint64_t, const ::xla::HloInstructionProto*>
        unique_id_to_hlo;

    for (const auto& computation : hlo_proto_.hlo_module().computations()) {
      for (const auto& instruction : computation.instructions()) {
        name_to_hlo[instruction.name()] = &instruction;
        unique_id_to_hlo[instruction.id()] = &instruction;
      }
    }

    absl::flat_hash_map<int64_t, const LogicalBufferProto*>
        id_to_logical_buffer_proto;
    for (const auto& logical_buffer :
         hlo_proto_.buffer_assignment().logical_buffers()) {
      id_to_logical_buffer_proto[logical_buffer.id()] = &logical_buffer;
    }

    for (const auto& buffer_allocation :
         hlo_proto_.buffer_assignment().buffer_allocations()) {
      auto& buffer_allocation_s =
          id_to_buffer_allocation_[buffer_allocation.index()];
      buffer_allocation_s =
          std::make_unique<BufferAllocationStruct>(buffer_allocation);
      for (const auto& assigned : buffer_allocation.assigned()) {
        const auto id = assigned.logical_buffer_id();
        const auto* logical_buffer = id_to_logical_buffer_proto.at(id);
        const auto& instruction_name =
            logical_buffer->defined_at().instruction_name();
        const auto* instruction =
            instruction_name.empty()
                ? unique_id_to_hlo.at(
                      logical_buffer->defined_at().instruction_id())
                : name_to_hlo.at(instruction_name);
        id_to_logical_buffer_[id] = std::make_unique<LogicalBufferStruct>(
            *logical_buffer, *buffer_allocation_s, *instruction,
            assigned.offset());
      }
    }

    const auto& heap_simulator_traces =
        hlo_proto_.buffer_assignment().heap_simulator_traces();
    for (int64_t i = 0; i < heap_simulator_traces.size(); i++) {
      // The trace's buffer_allocation_index is not trustful, so we are trying
      // to obtain the buffer allocation index ourselves.
      if (heap_simulator_traces[i].events().empty()) continue;
      int logical_buffer_id = heap_simulator_traces[i].events(0).buffer_id();
      auto* logical_buffer = id_to_logical_buffer_[logical_buffer_id].get();
      auto buffer_allocation_index = logical_buffer->buffer_allocation.index();
      id_to_buffer_allocation_[buffer_allocation_index]
          ->set_heap_simulator_trace_id(i);
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
        const auto& logical_buffer =
            id_to_logical_buffer_.at(event.buffer_id());
        if (logical_buffer->color() == memory_color) {
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
    auto buffer_allocations = GetBufferAllocations(memory_color);
    for (const auto* buffer_allocation : buffer_allocations) {
      if (buffer_allocation->IsIndefinite()) continue;
      // TODO(xprof): handle multiple temporary buffer allocations for the same
      // color.
      if (buffer_allocation->heap_simulator_trace_id()) {
        return *buffer_allocation->heap_simulator_trace_id();
      }
    }
    return -1;
  }

  // Reference to the original HLO proto.
  const ::xla::HloProto& hlo_proto_;

  // A mapping from logical buffer ID to logical buffer.
  absl::flat_hash_map<int64_t, std::unique_ptr<LogicalBufferStruct>>
      id_to_logical_buffer_;

  // A mapping from buffer allocation ID to BufferAllocationProto.
  absl::flat_hash_map<int64_t, std::unique_ptr<BufferAllocationStruct>>
      id_to_buffer_allocation_;
};

double BytesToMiB(int64_t bytes) {
  return static_cast<double>(bytes) / (1ULL << 20);
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

HeapObject MakeHeapObject(const LogicalBufferStruct& logical_buffer,
                          int32_t color) {
  const HloInstructionProto& hlo_instruction = logical_buffer.hlo_instruction;
  std::string shape_string = ShapeDescription(logical_buffer.shape);
  std::string label =
      absl::StrFormat("%s: %s # %s", logical_buffer.instruction_name(),
                      shape_string, hlo_instruction.metadata().op_name());
  HeapObject result = MakeHeapObjectCommon(
      std::move(label), color, logical_buffer.proto.id(), logical_buffer.size(),
      logical_buffer.unpadded_size());
  result.set_instruction_name(
      logical_buffer.GetInstructionNameWithShapeIndex());
  result.set_group_name(logical_buffer.buffer_allocation.category());
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

void Convert(const xla::BufferAllocationProto_Assigned& assigned,
             const HloProtoBufferWrapper& wrapper, LogicalBuffer* result) {
  result->set_id(assigned.logical_buffer_id()),
      result->set_size_mib(BytesToMiB(assigned.size()));
  const auto& logical_buffer =
      wrapper.GetLogicalBuffer(assigned.logical_buffer_id());
  result->set_hlo_name(std::string(logical_buffer.instruction_name()));
  result->mutable_shape_index()->CopyFrom(
      logical_buffer.proto.defined_at().shape_index());
  result->set_shape(ShapeDescription(logical_buffer.shape));
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
    seen_buffer_allocations.insert(&logical_buffer.buffer_allocation.proto());
  }

  // Update stats when memory usage increase.
  void IncreaseMemoryUsage(LogicalBufferStruct* canonical_logical_buffer,
                           bool init_buffer_span) {
    logical_buffers.push_back(canonical_logical_buffer->proto.id());
    heap_size_bytes += canonical_logical_buffer->size();
    unpadded_heap_size_bytes += canonical_logical_buffer->unpadded_size();

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
      canonical_logical_buffer->span.emplace(
          heap_size_bytes_timeline.size() - 1, simulator_trace_event_size - 1);
    }
  }

  // Update stats when memory usage decrease.
  Status DecreaseMemoryUsage(LogicalBufferStruct* canonical_logical_buffer) {
    int64_t canonical_buffer_id = canonical_logical_buffer->proto.id();
    logical_buffers.remove(canonical_buffer_id);
    heap_size_bytes -= canonical_logical_buffer->size();
    if (heap_size_bytes < 0) {
      return errors::InvalidArgument(absl::StrCat(
          "Heap size should be non-negative, but get: ", heap_size_bytes));
    }
    unpadded_heap_size_bytes -= canonical_logical_buffer->unpadded_size();
    // Mark the end of this buffer.
    if (canonical_logical_buffer->span) {
      canonical_logical_buffer->span->second =
          heap_size_bytes_timeline.size() - 1;
    }
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

  // Keep track of memory usage when iterating through heap simulator trace
  // events.
  int64_t heap_size_bytes = 0;
  int64_t unpadded_heap_size_bytes = 0;
  // Memory usage at peak.
  int64_t peak_heap_size_bytes = 0;
  int64_t peak_unpadded_heap_size_bytes = 0;

  // Keep track of logical buffer IDs when iterating through heap simulator
  // trace events. It is important this is in "program order", i.e. heap
  // simulator's order.
  std::list<int64_t> logical_buffers;
  // Logical buffer IDs at peak.
  std::list<int64_t> peak_logical_buffers;

  // Heap size timeline.
  std::vector<int64_t> heap_size_bytes_timeline;
  std::vector<int64_t> unpadded_heap_size_bytes_timeline;

  // Position of peak memory usage in the timeline.
  int64_t peak_heap_size_position = 0;

  // Logical buffers and buffer allocations that exists in heap simulator trace.
  absl::flat_hash_set<const LogicalBufferStruct*> seen_logical_buffers;
  absl::flat_hash_set<const BufferAllocationProto*> seen_buffer_allocations;

  // Constants while iterating through heap simulator trace.
  const HloProtoBufferWrapper& wrapper;
  int64_t simulator_trace_event_size;
};

Status ProcessHeapSimulatorTrace(const HloProtoBufferWrapper& wrapper,
                                 const int64_t memory_color,
                                 HeapSimulatorStats* stats) {
  int64_t heap_simulator_trace_id =
      wrapper.GetHeapSimulatorTraceId(memory_color);

  // If unable to get a valid heap simulator trace id, skip heap simulator
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

  stats->SetSimulatorTraceEventSize(trace.events_size());
  for (const auto& event : trace.events()) {
    stats->UpdateOnSimulatorEvent(event);
    auto& logical_buffer = wrapper.GetLogicalBuffer(event.buffer_id());
    if (event.kind() == HeapSimulatorTrace::Event::ALLOC) {
      // ALLOC event increases memory usage and initializes the buffer lifetime
      // span.
      logical_buffer.inc();
      stats->IncreaseMemoryUsage(&logical_buffer,
                                 /*init_buffer_span=*/true);
    } else if (event.kind() == HeapSimulatorTrace::Event::FREE) {
      auto ref_count = logical_buffer.dec();
      if (ref_count < 0) {
        return errors::InvalidArgument(absl::StrCat(
            "Buffer ", logical_buffer.proto.id(), "is freed multiple times."));
      }
      if (ref_count == 0) {
        // There is no more reference to the canonical buffer, the canonical
        // buffer is finally freed. Update memory usage and memory timespan
        // using the metadata of canonical buffer.
        auto& canonical_buffer = *logical_buffer.get_canonical_buffer();
        TF_RETURN_IF_ERROR(stats->DecreaseMemoryUsage(&canonical_buffer));
      }
    } else if (event.kind() == HeapSimulatorTrace::Event::SHARE_WITH) {
      int64_t canonical_buffer_id = event.share_with_canonical_id();
      auto& canonical_buffer = wrapper.GetLogicalBuffer(canonical_buffer_id);
      auto ref_count = logical_buffer.share_with(&canonical_buffer);

      if (ref_count == 1) {
        // SHARE_WITH happens after the FREE of a canonical buffer.
        // SHARE_WITH event does not initialize buffer lifetime span, it was
        // initialized by ALLOC event using the canonical logical buffer.
        stats->IncreaseMemoryUsage(&canonical_buffer,
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
struct PeakUsageSnapshot {
  PeakUsageSnapshot(const HloProtoBufferWrapper& wrapper,
                    const HeapSimulatorStats& simulator_stats,
                    int64_t small_buffer_size)
      : wrapper(wrapper),
        simulator_stats(simulator_stats),
        small_buffer_size(small_buffer_size) {}

  // Add a HeapObject derived from logical buffer and buffer allocation.
  void AddHeapObject(const LogicalBufferStruct& logical_buffer) {
    if (logical_buffer.size() < small_buffer_size) {
      // Accumulate small buffers, don't make a HeapObject.
      total_small_buffer_size_bytes += logical_buffer.size();
    } else {
      // Make a new HeapObject, assign a new color to visualize it.
      max_heap_objects.push_back(MakeHeapObject(logical_buffer, colorno++));
    }
  }

  void FinalizeBufferUsage() {
    // Buffers from HeapSimulatorTrace.
    for (const int64_t logical_buffer_id :
         simulator_stats.peak_logical_buffers) {
      const auto& logical_buffer = wrapper.GetLogicalBuffer(logical_buffer_id);
      AddHeapObject(logical_buffer);
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

void CreatePeakUsageSnapshot(const HloProtoBufferWrapper& wrapper,
                             int64_t memory_color,
                             PeakUsageSnapshot* peak_snapshot) {
  // Add indefinite (global) buffers to peak usage snapshot.
  for (const auto* logical_buffer :
       wrapper.LogicalBuffersWithIndefiniteLifetime(memory_color)) {
    const auto& buffer_allocation = logical_buffer->buffer_allocation;
    peak_snapshot->indefinite_memory_usage_bytes += buffer_allocation.size();
    peak_snapshot->AddHeapObject(*logical_buffer);
  }

  // Add temporary buffers (traced by heap simulator) to peak usage snapshot.
  peak_snapshot->FinalizeBufferUsage();
}

void ConvertAllocationTimeline(const HloProtoBufferWrapper& wrapper,
                               const HeapSimulatorStats& simulator_stats,
                               const int64_t memory_color,
                               PreprocessResult* result) {
  // The color constants from https://graphviz.org/doc/info/colors.html.
  const char* lb_colors[] = {
      "antiquewhite3",
      "aqua",
      "aquamarine",
      "bisque",
      "blanchedalmond",
      "blue",
      "blueviolet",
      "brown",
      "burlywood",
      "cadetblue",
      "chartreuse",
      "chocolate",
      "coral",
      "cornflowerblue",
      "crimson",
      "cyan",
      "darkblue",
      "darkcyan",
      "darkgoldenrod",
      "darkgray",
      "darkgreen",
      "darkkhaki",
      "darkmagenta",
      "darkolivegreen",
      "darkorange",
      "darkorchid",
      "darkred",
      "darksalmon",
      "darkseagreen",
      "darkslateblue",
      "darkslategray",
      "darkturquoise",
      "darkviolet",
      "deeppink",
      "deepskyblue",
      "dimgray",
      "dodgerblue",
      "firebrick",
      "floralwhite",
      "forestgreen",
      "fuchsia",
      "gainsboro",
      "gold",
      "goldenrod",
      "green",
      "greenyellow",
      "goldenrod",
      "greenyellow",
      "honeydew",
      "hotpink",
      "indianred",
      "indigo",
      "ivory3",
      "khaki",
      "lavender",
      "lavenderblush",
      "lawngreen",
      "lemonchiffon",
      "lightblue",
      "lightcoral",
      "lightcyan",
      "lightpink",
      "limegreen",
      "lightsalmon",
      "lightseagreen",
      "lightskyblue",
      "lime",
      "magenta",
      "maroon",
      "mediumaquamarine",
      "mediumblue",
      "mediumorchid",
      "mediumpurple",
      "midnightblue",
      "mediumvioletred",
      "mistyrose",
      "moccasin",
      "olive",
      "orange",
      "orangered",
      "orchid",
      "palegoldenrod"
      "palegreen",
      "paleturquoise",
      "palevioletred",
      "papayawhip",
      "peachpuff",
      "peachpuff",
      "pink",
      "plum",
      "powderblue",
      "purple",
      "rebeccapurple",
      "red",
      "rosybrown",
      "royalblue",
      "salmon",
      "sandybrown",
      "seagreen",
      "seashell",
      "sienna",
      "skyblue",
      "tan",
      "teal",
      "turquoise",
      "tomato",
      "violet",
      "violetred",
      "yellow",
  };

  struct RenderOptions {
    size_t graph_width = 2048;
    size_t graph_height = 2048;
  } render_options;

  const char* ba_colors[] = {
      "azure",
      "beige",
      "cornsilk",
  };

  int num_lb_colors = sizeof(lb_colors) / sizeof(lb_colors[0]);
  int num_ba_colors = sizeof(ba_colors) / sizeof(ba_colors[0]);
  std::vector<size_t> buffer_allocation_offsets;
  size_t total_y_size = 0;  // Range of y dimension.
  size_t total_x_size = 0;  // Range of x dimension.
  std::vector<std::string> rects;
  auto buffer_allocations = wrapper.GetBufferAllocations(memory_color);
  const auto& heap_simulator_traces =
      wrapper.GetHloProto().buffer_assignment().heap_simulator_traces();
  for (const auto& buffer_allocation : buffer_allocations) {
    // Exclude BAs for "global variables". The timeline provides little value.
    if (buffer_allocation->IsIndefinite()) continue;
    auto heap_simulator_trace_id = buffer_allocation->heap_simulator_trace_id();
    if (!heap_simulator_trace_id) continue;
    buffer_allocation_offsets.push_back(total_y_size);
    total_y_size += buffer_allocation->size();
    total_x_size = std::max<size_t>(
        total_x_size,
        heap_simulator_traces.at(*heap_simulator_trace_id).events_size());
  }
  if (!total_y_size || !total_x_size) return;
  double scale_x =
      static_cast<double>(render_options.graph_width) / total_x_size;
  double scale_y =
      static_cast<double>(render_options.graph_height) / total_y_size;

  int node_id = 0;
  auto add_rect = [&](size_t x, size_t y, size_t width, size_t height,
                      const string& description, const char* color) {
    size_t center_x = x + (width >> 1);
    size_t center_y = y + (height >> 1);
    int pos_x = center_x * scale_x;
    int pos_y = center_y * scale_y;
    int rect_w = width * scale_x;
    int rect_h = height * scale_y;
    // Skip when block size is smaller than half a pixel in output size.
    if (height * scale_y < 0.5) return;
    rect_h = std::max(rect_h, 1);  // Rounding up.
    std::string rect = absl::StrFormat(
        R"("%d" [tooltip="%s", pos="%d,%d!", width="%d!", height="%d!", color=%s];)",
        node_id++, description, pos_x, pos_y, rect_w, rect_h, color);
    rects.push_back(rect);
  };
  int buffer_id = 0;
  for (const auto& buffer_allocation : buffer_allocations) {
    // Exclude BAs for "global variables". The timeline provides little value.
    if (buffer_allocation->IsIndefinite()) continue;
    auto buffer_allocation_offset = buffer_allocation_offsets[buffer_id++];
    add_rect(0, buffer_allocation_offset, total_x_size,
             buffer_allocation->size(), buffer_allocation->description(),
             ba_colors[buffer_id % num_ba_colors]);

    for (const auto& assigned : buffer_allocation->proto().assigned()) {
      const LogicalBufferStruct& logical_buffer =
          wrapper.GetLogicalBuffer(assigned.logical_buffer_id());
      // Exclude non-canonical logical buffers.
      if (!logical_buffer.span || logical_buffer.canonical_buffer) continue;
      size_t width = logical_buffer.span->second - logical_buffer.span->first;
      size_t height = buffer_allocation_offset + logical_buffer.size();
      add_rect(logical_buffer.span->first, logical_buffer.offset, width, height,
               logical_buffer.description(),
               lb_colors[node_id % num_lb_colors]);
    }
  }
  VLOG(1) << "rects:" << rects.size();
  result->set_allocation_timeline(
      absl::StrFormat("graph G {\n node [shape=box,style=filled];\n %s\n}",
                      absl::StrJoin(rects, "\n")));
}

void GeneratePreprocessResult(const HloProtoBufferWrapper& wrapper,
                              const HeapSimulatorStats& simulator_stats,
                              const PeakUsageSnapshot& peak_snapshot,
                              const int64_t memory_color,
                              PreprocessResult* result) {
  // Module info.
  result->set_module_name(wrapper.GetHloProto().hlo_module().name());
  result->set_entry_computation_name(
      wrapper.GetHloProto().hlo_module().entry_computation_name());

  // Build HeapObjects and index.
  std::vector<const HeapObject*> max_heap_by_size;
  max_heap_by_size.reserve(peak_snapshot.max_heap_objects.size());
  for (const auto& object : peak_snapshot.max_heap_objects) {
    max_heap_by_size.push_back(&object);
  }
  std::sort(max_heap_by_size.begin(), max_heap_by_size.end(),
            [](const HeapObject* a, const HeapObject* b) {
              return a->logical_buffer_size_mib() >
                     b->logical_buffer_size_mib();
            });

  std::vector<int> max_heap_to_by_size;
  max_heap_to_by_size.reserve(max_heap_by_size.size());
  for (const auto& object : peak_snapshot.max_heap_objects) {
    auto it =
        std::find(max_heap_by_size.begin(), max_heap_by_size.end(), &object);
    int index = std::distance(max_heap_by_size.begin(), it);
    max_heap_to_by_size.push_back(index);
  }

  std::vector<int> by_size_to_max_heap;
  for (const auto* object : max_heap_by_size) {
    int index = object - &peak_snapshot.max_heap_objects[0];
    by_size_to_max_heap.push_back(index);
  }

  *result->mutable_max_heap() = {peak_snapshot.max_heap_objects.begin(),
                                 peak_snapshot.max_heap_objects.end()};
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
  double add_mib = BytesToMiB(peak_snapshot.indefinite_memory_usage_bytes);
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
  for (const auto* logical_buffer : simulator_stats.seen_logical_buffers) {
    if (!logical_buffer->span) continue;
    (*result->mutable_logical_buffer_spans())[logical_buffer->proto.id()] =
        MakeBufferSpan(logical_buffer->span->first,
                       logical_buffer->span->second);
  }

  NoteSpecialAllocations(wrapper, peak_snapshot.small_buffer_size, result);

  ConvertAllocationTimeline(wrapper, simulator_stats, memory_color, result);
}

}  // namespace

absl::StatusOr<PreprocessResult> ConvertHloProtoToPreprocessResult(
    const HloProto& hlo_proto, int64_t small_buffer_size,
    int64_t memory_color) {
  HloProtoBufferWrapper wrapper(hlo_proto);

  // Process heap simulator trace.
  HeapSimulatorStats simulator_stats(wrapper);
  auto status =
      ProcessHeapSimulatorTrace(wrapper, memory_color, &simulator_stats);
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to process heap simulator trace: ", status.error_message()));
  }

  // Process buffers with indefinite lifetime.
  PeakUsageSnapshot peak_snapshot(wrapper, simulator_stats, small_buffer_size);
  CreatePeakUsageSnapshot(wrapper, memory_color, &peak_snapshot);

  PreprocessResult result;
  GeneratePreprocessResult(wrapper, simulator_stats, peak_snapshot,
                           memory_color, &result);
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
