/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {

// Index of the time-sorted memory_profile_snapshots list, and the
// MemoryActivityMetadata proto it contains.
using IndexMetaPair = std::pair<int64 /*index*/, const MemoryActivityMetadata*>;

// Aggregated memory stats from an allocator. Temporary container to fill
// MemoryAggregationStats.
struct AggregationStats {
  int64 bytes_reserved = 0;
  int64 bytes_allocated = 0;
  int64 bytes_available = 0;
  double fragmentation = 0;
  int64 peak_bytes_in_use = 0;
};

// Metadata associated with each memory allocation/deallocation activity.
// Temporary container to fill MemoryActivityMetadata.
struct ActivityMetadata {
  int64 requested_bytes = 0;
  int64 allocation_bytes = 0;
  uint64 address = 0;
  absl::string_view tf_op_name;
  int64 step_id = -1;
  absl::string_view region_type;
  int64 data_type = 0;
  absl::string_view tensor_shape;
};

bool IsMemoryAllocation(int64 event_type) {
  return event_type == HostEventType::kMemoryAllocation;
}

bool IsMemoryDeallocation(int64 event_type) {
  return event_type == HostEventType::kMemoryDeallocation;
}

void FillAggregationStats(const AggregationStats& src,
                          MemoryAggregationStats* dst) {
  dst->set_stack_reserved_bytes(src.bytes_reserved);
  dst->set_heap_allocated_bytes(src.bytes_allocated);
  dst->set_free_memory_bytes(src.bytes_available);
  dst->set_fragmentation(src.fragmentation);
  dst->set_peak_bytes_in_use(src.peak_bytes_in_use);
}

void FillActivityMetadata(int64 event_type, const ActivityMetadata& src,
                          MemoryActivityMetadata* dst) {
  if (IsMemoryAllocation(event_type)) {
    dst->set_memory_activity(ALLOCATION);
  } else if (IsMemoryDeallocation(event_type)) {
    dst->set_memory_activity(DEALLOCATION);
  }
  dst->set_requested_bytes(src.requested_bytes);
  dst->set_allocation_bytes(src.allocation_bytes);
  dst->set_address(src.address);
  dst->set_tf_op_name(std::string(src.tf_op_name));
  dst->set_step_id(src.step_id);
  dst->set_region_type(std::string(src.region_type));
  dst->set_data_type(tensorflow::DataTypeString(
      static_cast<tensorflow::DataType>(src.data_type)));
  dst->set_tensor_shape(std::string(src.tensor_shape));
}

void UpdateProfileSummary(const AggregationStats& stats, int64 time_offset_ps,
                          MemoryProfileSummary* summary) {
  // Update the peak memory usage over allocator's lifetime.
  summary->set_peak_bytes_usage_lifetime(stats.peak_bytes_in_use);
  MemoryAggregationStats* peak_stats = summary->mutable_peak_stats();
  // If we reach (or stay at) peak memory usage within the profiling window,
  // update memory profile summary.
  if (stats.bytes_reserved + stats.bytes_allocated >=
      peak_stats->peak_bytes_in_use()) {
    peak_stats->set_peak_bytes_in_use(stats.bytes_reserved +
                                      stats.bytes_allocated);
    peak_stats->set_stack_reserved_bytes(stats.bytes_reserved);
    peak_stats->set_heap_allocated_bytes(stats.bytes_allocated);
    peak_stats->set_free_memory_bytes(stats.bytes_available);
    peak_stats->set_fragmentation(stats.fragmentation);
    summary->set_peak_stats_time_ps(time_offset_ps);
    summary->set_memory_capacity(stats.bytes_reserved + stats.bytes_allocated +
                                 stats.bytes_available);
  }
}

// Generate memory profile proto by processing host trace XPlane.
MemoryProfile GenerateMemoryProfile(const XPlane* host_trace) {
  XPlaneVisitor plane = CreateTfXPlaneVisitor(host_trace);
  MemoryProfile memory_profile;
  auto* step_count = memory_profile.mutable_step_count();
  // Iterate over all XEvents in the XPlane, and add the XStats to a new
  // MemoryProfileSnapshot if the EventType is kMemoryAllocation or
  // kMemoryDeallocation.
  plane.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      int64 event_type = event.Type().value_or(kUnknownHostEventType);
      if (!(IsMemoryAllocation(event_type) ||
            IsMemoryDeallocation(event_type))) {
        return;
      }

      AggregationStats stats;
      ActivityMetadata metadata;
      std::string memory_id;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (!stat.Type().has_value()) return;
        switch (stat.Type().value()) {
          case StatType::kIndexOnHost:
          case StatType::kDeviceOrdinal:
            memory_id = absl::StrFormat("%d", stat.IntValue());
            break;
          case StatType::kAllocatorName:
            memory_id = std::string(stat.StrOrRefValue());
            break;
          case StatType::kBytesReserved:
            stats.bytes_reserved = stat.IntValue();
            break;
          case StatType::kBytesAllocated:
            stats.bytes_allocated = stat.IntValue();
            break;
          case StatType::kBytesAvailable:
            stats.bytes_available = stat.IntValue();
            break;
          case StatType::kFragmentation:
            stats.fragmentation = stat.DoubleValue();
            break;
          case StatType::kPeakBytesInUse:
            stats.peak_bytes_in_use = stat.IntValue();
            break;
          case StatType::kRequestedBytes:
            metadata.requested_bytes = stat.IntValue();
            break;
          case StatType::kAllocationBytes:
            metadata.allocation_bytes = stat.IntValue();
            break;
          case StatType::kAddress:
            metadata.address = stat.IntValue();
            break;
          case StatType::kTfOp:
            metadata.tf_op_name = stat.StrOrRefValue();
            break;
          case StatType::kStepId:
            metadata.step_id = stat.IntValue();
            if (metadata.step_id != 0) (*step_count)[metadata.step_id]++;
            break;
          case StatType::kRegionType:
            metadata.region_type = stat.StrOrRefValue();
            break;
          case StatType::kDataType:
            metadata.data_type = stat.IntValue();
            break;
          case StatType::kTensorShapes:
            metadata.tensor_shape = stat.StrOrRefValue();
            break;
        }
      });

      MemoryProfileSnapshot* snapshot =
          (*memory_profile.mutable_memory_profile_per_allocator())[memory_id]
              .add_memory_profile_snapshots();
      snapshot->set_time_offset_ps(event.OffsetPs());
      FillAggregationStats(stats, snapshot->mutable_aggregation_stats());
      FillActivityMetadata(event_type, metadata,
                           snapshot->mutable_activity_metadata());

      MemoryProfileSummary* summary =
          (*memory_profile.mutable_memory_profile_per_allocator())[memory_id]
              .mutable_profile_summary();
      UpdateProfileSummary(stats, event.OffsetPs(), summary);
    });
  });
  return memory_profile;
}

// Sequentialize step ids for the memory profile.
void UpdateStepId(const tensorflow::protobuf::Map<
                      tensorflow::protobuf_int64 /*orig_step_id*/,
                      tensorflow::protobuf_int64 /*count*/>& step_count,
                  PerAllocatorMemoryProfile* memory_profile) {
  // Map from original random step id to sequential step id.
  absl::flat_hash_map<int64 /*orig_step_id*/, int64 /*step_id*/> step_map;
  constexpr int kUnknownStep = -2;
  constexpr double kStepFilterRatio = 0.1;  // Magic number for filtering.
  tensorflow::protobuf_int64 max_step_count = 0;
  for (const auto& step_and_count : step_count) {
    max_step_count = std::max(max_step_count, step_and_count.second);
  }
  // Filter out noisy and incomplete original step ids.
  for (const auto& step_and_count : step_count) {
    if (static_cast<double>(step_and_count.second) / max_step_count >
        kStepFilterRatio) {
      step_map[step_and_count.first] = kUnknownStep;
    }
  }

  // Update the step ids in memory_profile for this allocator.
  int64 step_id = -1;
  for (auto& snapshot : *memory_profile->mutable_memory_profile_snapshots()) {
    DCHECK(snapshot.has_activity_metadata());
    // Convert the random step id to sequential step id.
    int64 orig_step_id = snapshot.activity_metadata().step_id();
    if (step_map.contains(orig_step_id) &&
        step_map[orig_step_id] == kUnknownStep) {
      step_map[orig_step_id] = ++step_id;
    }
    snapshot.mutable_activity_metadata()->set_step_id(step_id);
  }
  VLOG(2) << "Max sequential step id in profile: " << step_id;
}

// Update the MemoryActivityMetadata for each deallocation event by copying from
// matching allocation.
void UpdateDeallocation(PerAllocatorMemoryProfile* memory_profile) {
  absl::flat_hash_map<uint64 /*address*/, const MemoryActivityMetadata*>
      addr_metadata_map;
  for (auto& snapshot : *memory_profile->mutable_memory_profile_snapshots()) {
    // Match the deallocation with previous allocation based on address.
    uint64 address = snapshot.activity_metadata().address();
    if (snapshot.activity_metadata().memory_activity() == DEALLOCATION) {
      if (addr_metadata_map.contains(address)) {
        const MemoryActivityMetadata* alloc_meta = addr_metadata_map[address];
        snapshot.mutable_activity_metadata()->set_tf_op_name(
            alloc_meta->tf_op_name());
        snapshot.mutable_activity_metadata()->set_region_type(
            alloc_meta->region_type());
        snapshot.mutable_activity_metadata()->set_data_type(
            alloc_meta->data_type());
        snapshot.mutable_activity_metadata()->set_tensor_shape(
            alloc_meta->tensor_shape());
        // In case of following (unexpected) deallocations to the same chunk
        // address, leave the metadata as it is (empty or already captured).
        addr_metadata_map.erase(address);
      } else {
        VLOG(2)
            << "Can't find matching memory allocation for this deallocation: "
            << snapshot.DebugString();
      }
    } else if (!addr_metadata_map.contains(address)) {  // Allocation.
      addr_metadata_map[address] = &snapshot.activity_metadata();
    } else {
      VLOG(2) << "There are two allocations recorded for the same address: "
              << address
              << ". The later allocation event is: " << snapshot.DebugString();
    }
  }
  VLOG(2) << "Number of allocations that cannot find matching dealloctions: "
          << addr_metadata_map.size();
}

// Return the step id for the peak memory usage data point.
int64 GetPeakMemoryStep(int64 peak_bytes_profile,
                        const PerAllocatorMemoryProfile* memory_profile) {
  int64 peak_bytes_profile_step_id = 0;
  for (const auto& snapshot : memory_profile->memory_profile_snapshots()) {
    // Get the step id of the peak memory usage.
    if (peak_bytes_profile ==
        snapshot.aggregation_stats().heap_allocated_bytes() +
            snapshot.aggregation_stats().stack_reserved_bytes()) {
      DCHECK(snapshot.has_activity_metadata());
      peak_bytes_profile_step_id = snapshot.activity_metadata().step_id();
    }
  }
  return peak_bytes_profile_step_id;
}

// Functor that compares (index, metadata) pair to sort in the order of
// allocation bytes and requested bytes (descending), as well as TF Op name,
// region type, data type, and tensor shape (ascending).
struct MetadataComparator {
  bool operator()(const IndexMetaPair& a, const IndexMetaPair& b) const {
    const MemoryActivityMetadata* a_meta = a.second;
    const MemoryActivityMetadata* b_meta = b.second;
    DCHECK_NE(a_meta, nullptr);
    DCHECK_NE(b_meta, nullptr);

    auto lhs =
        std::make_tuple(-a_meta->allocation_bytes(), -a_meta->requested_bytes(),
                        a_meta->tf_op_name(), a_meta->region_type(),
                        a_meta->data_type(), a_meta->tensor_shape());
    auto rhs =
        std::make_tuple(-b_meta->allocation_bytes(), -b_meta->requested_bytes(),
                        b_meta->tf_op_name(), b_meta->region_type(),
                        b_meta->data_type(), b_meta->tensor_shape());
    return lhs < rhs;
  }
};

// If applicable, add items into active_allocs vector and special_allocations
// proto for the unmapped memory usage (in heap) and stack reservation at peak.
void InsertSpecialAllocations(int64 unmapped_allocation_bytes, int64 step_id,
                              PerAllocatorMemoryProfile* memory_profile,
                              std::vector<IndexMetaPair>* active_allocs) {
  int index = 0;
  if (unmapped_allocation_bytes > 0) {
    MemoryActivityMetadata* special_allocation =
        memory_profile->add_special_allocations();
    FillActivityMetadata(
        HostEventType::kMemoryAllocation,
        {unmapped_allocation_bytes, unmapped_allocation_bytes, 0,
         "preallocated/unknown", step_id, "persist/dynamic", 0, "unknown"},
        special_allocation);
    active_allocs->push_back({--index, special_allocation});
  }
  int64 stack_bytes =
      memory_profile->profile_summary().peak_stats().stack_reserved_bytes();
  if (stack_bytes > 0) {
    MemoryActivityMetadata* special_allocation =
        memory_profile->add_special_allocations();
    FillActivityMetadata(
        HostEventType::kMemoryAllocation,
        {stack_bytes, stack_bytes, 0, "stack", step_id, "stack", 0, "unknown"},
        special_allocation);
    active_allocs->push_back({--index, special_allocation});
  }
}

bool operator==(const IndexMetaPair& a, const IndexMetaPair& b) {
  const MemoryActivityMetadata* a_meta = a.second;
  const MemoryActivityMetadata* b_meta = b.second;
  return a_meta->allocation_bytes() == b_meta->allocation_bytes() &&
         a_meta->requested_bytes() == b_meta->requested_bytes() &&
         a_meta->tf_op_name() == b_meta->tf_op_name() &&
         a_meta->region_type() == b_meta->region_type() &&
         a_meta->data_type() == b_meta->data_type() &&
         a_meta->tensor_shape() == b_meta->tensor_shape();
}

// Generate the memory breakdown table of active allocations at the peak usage
// (within profiling window) and fill each ActiveAllocation proto (i.e. a row).
void ProcessActiveAllocations(int64 peak_bytes_profile_step_id,
                              PerAllocatorMemoryProfile* memory_profile) {
  int64 unmapped_allocation_bytes =
      memory_profile->profile_summary().peak_stats().heap_allocated_bytes();
  int64 unmapped_deallocation_bytes = 0;
  absl::flat_hash_map<int64 /*address*/, IndexMetaPair> active_alloc_map;
  // Only account for the memory activities in the step that includes peak
  // memory usage.
  for (int i = 0; i < memory_profile->memory_profile_snapshots_size(); i++) {
    const auto& snapshot = memory_profile->memory_profile_snapshots().at(i);
    DCHECK(snapshot.has_activity_metadata());
    const MemoryActivityMetadata& metadata = snapshot.activity_metadata();
    if (snapshot.time_offset_ps() >
        memory_profile->profile_summary().peak_stats_time_ps())
      break;
    if (metadata.step_id() != peak_bytes_profile_step_id) continue;

    if (metadata.memory_activity() == ALLOCATION) {
      active_alloc_map[metadata.address()] = {i, &metadata};
      unmapped_allocation_bytes -= metadata.allocation_bytes();
    } else {
      DCHECK_EQ(metadata.memory_activity(), DEALLOCATION);
      if (active_alloc_map.contains(metadata.address())) {
        active_alloc_map.erase(metadata.address());
      } else {
        unmapped_deallocation_bytes += metadata.allocation_bytes();
      }
      unmapped_allocation_bytes += metadata.allocation_bytes();
    }
  }
  // This separates the persistent memory from the freed memory from last step's
  // allocations.
  unmapped_allocation_bytes -= unmapped_deallocation_bytes;

  VLOG(2) << "unmapped_allocation_bytes=" << unmapped_allocation_bytes
          << ", unmapped_deallocation_bytes=" << unmapped_deallocation_bytes;

  // Using pair of (index, MemoryActivityMetadata*) so that we can sort by the
  // metadata, and fetch metadata by indexing the time-sorted snapshots at
  // frontend.
  std::vector<IndexMetaPair> active_allocs;
  for (const auto& address_and_index_meta : active_alloc_map) {
    active_allocs.push_back(address_and_index_meta.second);
  }

  InsertSpecialAllocations(unmapped_allocation_bytes,
                           peak_bytes_profile_step_id, memory_profile,
                           &active_allocs);

  std::sort(active_allocs.begin(), active_allocs.end(), MetadataComparator());

  // Fill the sorted active_allocations proto messages at peak memory usage.
  // Merge identical allocations and show occurrences.
  for (int i = 0; i < active_allocs.size(); i++) {
    ActiveAllocation* allocation = memory_profile->add_active_allocations();
    allocation->set_snapshot_index(active_allocs[i].first);
    if (active_allocs[i].first < 0) {
      allocation->set_special_index(-active_allocs[i].first - 1);
    } else {
      allocation->set_special_index(-1);
    }
    allocation->set_num_occurrences(1);
    while (i < active_allocs.size() - 1 &&
           active_allocs[i] == active_allocs[i + 1]) {
      allocation->set_num_occurrences(allocation->num_occurrences() + 1);
      i++;
    }
  }

  VLOG(2) << "Distinctive active allocation count="
          << memory_profile->active_allocations_size();
}

// Post-process the memory profile to correctly update proto fields, and break
// down peak memory usage for each allocator.
void ProcessMemoryProfileProto(MemoryProfile* memory_profile) {
  memory_profile->set_num_hosts(1);
  // Add sorted memory ids within memory profile data to the selection list.
  for (const auto& id_and_allocator_profile :
       memory_profile->memory_profile_per_allocator()) {
    if (!id_and_allocator_profile.second.memory_profile_snapshots().empty()) {
      memory_profile->add_memory_ids(id_and_allocator_profile.first);
    }
  }
  absl::c_sort(*memory_profile->mutable_memory_ids());

  for (auto& id_and_allocator_profile :
       *memory_profile->mutable_memory_profile_per_allocator()) {
    PerAllocatorMemoryProfile* allocator_memory_profile =
        &id_and_allocator_profile.second;
    // Sort the memory_profile_snapshots by time_offset_ps (ascending) in proto.
    absl::c_sort(
        *allocator_memory_profile->mutable_memory_profile_snapshots(),
        [](const MemoryProfileSnapshot& a, const MemoryProfileSnapshot& b) {
          return a.time_offset_ps() < b.time_offset_ps();
        });

    UpdateStepId(memory_profile->step_count(), allocator_memory_profile);
    UpdateDeallocation(allocator_memory_profile);

    int64 peak_bytes_profile = allocator_memory_profile->profile_summary()
                                   .peak_stats()
                                   .peak_bytes_in_use();
    int64 peak_step_id =
        GetPeakMemoryStep(peak_bytes_profile, allocator_memory_profile);
    ProcessActiveAllocations(peak_step_id, allocator_memory_profile);
  }
}

}  // namespace

MemoryProfile ConvertXPlaneToMemoryProfile(const XPlane& host_plane) {
  MemoryProfile memory_profile = GenerateMemoryProfile(&host_plane);
  ProcessMemoryProfileProto(&memory_profile);
  return memory_profile;
}

}  // namespace profiler
}  // namespace tensorflow
