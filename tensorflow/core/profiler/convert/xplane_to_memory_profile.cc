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
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {

constexpr int64_t kInvalidStepId = -1;

// Index of the time-sorted memory_profile_snapshots list, and the
// MemoryActivityMetadata proto it contains.
using IndexMetaPair =
    std::pair<int64_t /*index*/, const MemoryActivityMetadata*>;

bool IsMemoryAllocation(int64_t event_type) {
  return event_type == HostEventType::kMemoryAllocation;
}

bool IsMemoryDeallocation(int64_t event_type) {
  return event_type == HostEventType::kMemoryDeallocation;
}

void UpdateProfileSummary(const MemoryAggregationStats& stats,
                          int64_t time_offset_ps,
                          MemoryProfileSummary* summary) {
  // Update the peak memory usage over allocator's lifetime.
  summary->set_peak_bytes_usage_lifetime(stats.peak_bytes_in_use());
  MemoryAggregationStats* peak_stats = summary->mutable_peak_stats();
  // If we reach (or stay at) peak memory usage within the profiling window,
  // update memory profile summary.
  if (stats.stack_reserved_bytes() + stats.heap_allocated_bytes() >=
      peak_stats->peak_bytes_in_use()) {
    *peak_stats = stats;
    peak_stats->set_peak_bytes_in_use(stats.stack_reserved_bytes() +
                                      stats.heap_allocated_bytes());
    summary->set_peak_stats_time_ps(time_offset_ps);
    summary->set_memory_capacity(stats.stack_reserved_bytes() +
                                 stats.heap_allocated_bytes() +
                                 stats.free_memory_bytes());
  }
}

// Generate memory profile proto by processing host trace XPlane.
MemoryProfile GenerateMemoryProfile(const XPlane* host_trace) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  MemoryProfile memory_profile;
  // Iterate over all XEvents in the XPlane, and add the XStats to a new
  // MemoryProfileSnapshot if the EventType is kMemoryAllocation or
  // kMemoryDeallocation.
  plane.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      int64_t event_type =
          event.Type().value_or(HostEventType::kUnknownHostEventType);
      if (!(IsMemoryAllocation(event_type) ||
            IsMemoryDeallocation(event_type))) {
        return;
      }

      MemoryAggregationStats stats;
      MemoryActivityMetadata metadata;
      if (IsMemoryAllocation(event_type)) {
        metadata.set_memory_activity(ALLOCATION);
      } else if (IsMemoryDeallocation(event_type)) {
        metadata.set_memory_activity(DEALLOCATION);
      }
      metadata.set_step_id(kInvalidStepId);

      std::string memory_id;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (!stat.Type().has_value()) return;
        switch (stat.Type().value()) {
          case StatType::kIndexOnHost:
          case StatType::kDeviceOrdinal:
            memory_id = absl::StrCat(stat.IntValue());
            break;
          case StatType::kAllocatorName:
            memory_id = std::string(stat.StrOrRefValue());
            break;
          case StatType::kBytesReserved:
            stats.set_stack_reserved_bytes(stat.IntValue());
            break;
          case StatType::kBytesAllocated:
            stats.set_heap_allocated_bytes(stat.IntValue());
            break;
          case StatType::kBytesAvailable:
            stats.set_free_memory_bytes(stat.IntValue());
            break;
          case StatType::kFragmentation:
            stats.set_fragmentation(stat.DoubleValue());
            break;
          case StatType::kPeakBytesInUse:
            stats.set_peak_bytes_in_use(stat.IntValue());
            break;
          case StatType::kRequestedBytes:
            metadata.set_requested_bytes(stat.IntValue());
            break;
          case StatType::kAllocationBytes:
            metadata.set_allocation_bytes(stat.IntValue());
            break;
          case StatType::kAddress:
            metadata.set_address(stat.IntValue());
            break;
          case StatType::kTfOp:
            metadata.set_tf_op_name(std::string(stat.StrOrRefValue()));
            break;
          case StatType::kGroupId:
            metadata.set_step_id(stat.IntValue());
            break;
          case StatType::kRegionType:
            metadata.set_region_type(std::string(stat.StrOrRefValue()));
            break;
          case StatType::kDataType:
            metadata.set_data_type(tensorflow::DataTypeString(
                static_cast<tensorflow::DataType>(stat.IntValue())));
            break;
          case StatType::kTensorShapes:
            metadata.set_tensor_shape(std::string(stat.StrOrRefValue()));
            break;
        }
      });

      MemoryProfileSummary* summary =
          (*memory_profile.mutable_memory_profile_per_allocator())[memory_id]
              .mutable_profile_summary();
      UpdateProfileSummary(stats, event.OffsetPs(), summary);

      MemoryProfileSnapshot* snapshot =
          (*memory_profile.mutable_memory_profile_per_allocator())[memory_id]
              .add_memory_profile_snapshots();
      snapshot->set_time_offset_ps(event.OffsetPs());
      *snapshot->mutable_aggregation_stats() = std::move(stats);
      *snapshot->mutable_activity_metadata() = std::move(metadata);
    });
  });
  return memory_profile;
}

// Fix invalid step ids of snapshots at the beginning/end of the profile or at
// the step boundaries. The snapshots with invalid step ids at the beginning get
// 0 for their step ids. Those at the step boundaries or at the end get the
// previous snapshot's step id + 1.
void UpdateStepId(PerAllocatorMemoryProfile* memory_profile) {
  int64_t last_valid_step_id = -1;
  // Snapshots are already sorted in time.
  for (auto& snapshot : *memory_profile->mutable_memory_profile_snapshots()) {
    DCHECK(snapshot.has_activity_metadata());
    if (snapshot.mutable_activity_metadata()->step_id() == kInvalidStepId) {
      snapshot.mutable_activity_metadata()->set_step_id(last_valid_step_id + 1);
    } else {
      last_valid_step_id = snapshot.mutable_activity_metadata()->step_id();
    }
  }
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
int64_t GetPeakMemoryStep(int64_t peak_bytes_profile,
                          const PerAllocatorMemoryProfile* memory_profile) {
  int64_t peak_bytes_profile_step_id = 0;
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
void InsertSpecialAllocations(int64_t unmapped_allocation_bytes,
                              int64_t step_id,
                              PerAllocatorMemoryProfile* memory_profile,
                              std::vector<IndexMetaPair>* active_allocs) {
  int index = 0;
  if (unmapped_allocation_bytes > 0) {
    MemoryActivityMetadata* special_allocation =
        memory_profile->add_special_allocations();
    special_allocation->set_memory_activity(ALLOCATION);
    special_allocation->set_requested_bytes(unmapped_allocation_bytes);
    special_allocation->set_allocation_bytes(unmapped_allocation_bytes);
    special_allocation->set_address(0);
    special_allocation->set_tf_op_name("unused preallocated device memory");
    special_allocation->set_step_id(step_id);
    special_allocation->set_region_type("persist/dynamic");
    special_allocation->set_data_type(
        tensorflow::DataTypeString(static_cast<tensorflow::DataType>(0)));
    special_allocation->set_tensor_shape("unknown");
    active_allocs->push_back({--index, special_allocation});
  }
  int64_t stack_bytes =
      memory_profile->profile_summary().peak_stats().stack_reserved_bytes();
  if (stack_bytes > 0) {
    MemoryActivityMetadata* special_allocation =
        memory_profile->add_special_allocations();
    special_allocation->set_memory_activity(ALLOCATION);
    special_allocation->set_requested_bytes(stack_bytes);
    special_allocation->set_allocation_bytes(stack_bytes);
    special_allocation->set_address(0);
    special_allocation->set_tf_op_name("stack");
    special_allocation->set_step_id(step_id);
    special_allocation->set_region_type("stack");
    special_allocation->set_data_type(
        tensorflow::DataTypeString(static_cast<tensorflow::DataType>(0)));
    special_allocation->set_tensor_shape("unknown");
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
void ProcessActiveAllocations(int64_t peak_bytes_profile_step_id,
                              PerAllocatorMemoryProfile* memory_profile) {
  int64_t unmapped_allocation_bytes =
      memory_profile->profile_summary().peak_stats().heap_allocated_bytes();
  int64_t unmapped_deallocation_bytes = 0;
  absl::flat_hash_map<int64_t /*address*/, IndexMetaPair> active_alloc_map;
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
  for (int i = 0, end = active_allocs.size(); i < end; i++) {
    ActiveAllocation* allocation = memory_profile->add_active_allocations();
    allocation->set_snapshot_index(active_allocs[i].first);
    if (active_allocs[i].first < 0) {
      allocation->set_special_index(-active_allocs[i].first - 1);
    } else {
      allocation->set_special_index(-1);
    }
    allocation->set_num_occurrences(1);
    const int last_alloc = active_allocs.size() - 1;
    while (i < last_alloc && active_allocs[i] == active_allocs[i + 1]) {
      allocation->set_num_occurrences(allocation->num_occurrences() + 1);
      i++;
    }
  }

  VLOG(2) << "Distinctive active allocation count="
          << memory_profile->active_allocations_size();
}

// This function saves the MemoryProfileSnapshots referenced by
// <active_allocations> max_num_snapshots.
void SaveActiveAllocationSnapshots(
    protobuf::RepeatedPtrField<MemoryProfileSnapshot>* snapshots,
    protobuf::RepeatedPtrField<ActiveAllocation>* active_allocations) {
  std::vector<MemoryProfileSnapshot*> samples;
  // Puts the snapshots referenced by active_allocations in <samples>.
  for (const auto& allocation : *active_allocations) {
    auto orig_index = allocation.snapshot_index();
    if (orig_index < 0) continue;
    samples.push_back(&(*snapshots)[orig_index]);
  }

  // Change the reference index in <active_allocations>.
  int new_index = 0;
  for (auto& allocation : *active_allocations) {
    int64_t origin_index = allocation.snapshot_index();
    if (origin_index < 0) continue;
    allocation.set_snapshot_index(new_index);
    new_index++;
  }

  protobuf::RepeatedPtrField<MemoryProfileSnapshot> new_snapshots;
  new_snapshots.Reserve(samples.size());
  for (const auto& sample : samples) {
    *new_snapshots.Add() = std::move(*sample);
  }
  *snapshots = std::move(new_snapshots);
}

// Sample <max_num_snapshots> memory profile snapshots from the original memory
// profile data.
void SampleMemoryProfileTimeline(int64_t max_num_snapshots,
                                 PerAllocatorMemoryProfile* memory_profile) {
  const protobuf::RepeatedPtrField<MemoryProfileSnapshot>& original_snapshots =
      memory_profile->memory_profile_snapshots();
  protobuf::RepeatedPtrField<MemoryProfileSnapshot>* timeline_snapshots =
      memory_profile->mutable_sampled_timeline_snapshots();
  int64_t snapshot_count = original_snapshots.size();
  if (snapshot_count > max_num_snapshots) {
    // When there are more memory profile data than <max_num_snapshots>, we
    // sample the origin data using a max box filter. Filter width is
    // <filter_width>, collect <count> samples starting from the <start> index
    // in the original snapshots.
    auto max_box_filter = [&](int filter_width, int count, int start) {
      for (int i = 0; i < count; i++) {
        // Use a max function to get the MemoryProfileSnapshot with the largest
        // memory usage in the box filter.
        const MemoryProfileSnapshot* max_snapshot =
            &original_snapshots[start + filter_width * i];
        int64_t max_bytes =
            max_snapshot->aggregation_stats().heap_allocated_bytes() +
            max_snapshot->aggregation_stats().stack_reserved_bytes();
        for (int index = start + filter_width * i + 1;
             index < start + filter_width * (i + 1); index++) {
          int64_t bytes = original_snapshots[index]
                              .aggregation_stats()
                              .heap_allocated_bytes() +
                          original_snapshots[index]
                              .aggregation_stats()
                              .stack_reserved_bytes();
          if (bytes > max_bytes) {
            max_snapshot = &original_snapshots[index];
            max_bytes = bytes;
          }
        }
        *timeline_snapshots->Add() = *max_snapshot;
      }
    };

    int width = snapshot_count / max_num_snapshots;
    int count1 = max_num_snapshots * (width + 1) - snapshot_count;
    int count2 = max_num_snapshots - count1;

    // Collect <count1> samples with box filter width <width>, then collect
    // <count2> samples with box filter width <width+1>, the total number of
    // samples collected will be <max_num_snapshot>.
    max_box_filter(width, count1, 0);
    max_box_filter(width + 1, count2, width * count1);
  } else {
    // When the number of original snapshots are smaller than
    // <max_num_snapshots>, just copy all the data points to the timeline.
    *timeline_snapshots = original_snapshots;
  }
}

// Post-process the memory profile to correctly update proto fields, and break
// down peak memory usage for each allocator.
void ProcessMemoryProfileProto(int64_t max_num_snapshots,
                               MemoryProfile* memory_profile) {
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
    protobuf::RepeatedPtrField<MemoryProfileSnapshot>* snapshots =
        allocator_memory_profile->mutable_memory_profile_snapshots();
    // Sort the memory_profile_snapshots by time_offset_ps (ascending) in proto.
    absl::c_sort(*snapshots, [](const MemoryProfileSnapshot& a,
                                const MemoryProfileSnapshot& b) {
      return a.time_offset_ps() < b.time_offset_ps();
    });

    UpdateStepId(allocator_memory_profile);
    UpdateDeallocation(allocator_memory_profile);

    // Sample a subset of MemoryProfileSnapshots to display in the frontend
    // memory timeline graph.
    SampleMemoryProfileTimeline(max_num_snapshots, allocator_memory_profile);

    int64_t peak_step_id =
        GetPeakMemoryStep(allocator_memory_profile->profile_summary()
                              .peak_stats()
                              .peak_bytes_in_use(),
                          allocator_memory_profile);
    ProcessActiveAllocations(peak_step_id, allocator_memory_profile);
    SaveActiveAllocationSnapshots(
        snapshots, allocator_memory_profile->mutable_active_allocations());
  }
}

template <typename Proto>
absl::Status ConvertProtoToJson(const Proto& proto_output,
                                std::string* json_output) {
  protobuf::util::JsonPrintOptions json_options;
  json_options.always_print_primitive_fields = true;
  auto status = protobuf::util::MessageToJsonString(proto_output, json_output,
                                                    json_options);
  if (!status.ok()) {
    // Convert error_msg google::protobuf::StringPiece (or absl::string_view) to
    // tensorflow::StringPiece.
    auto error_msg = status.message();
    return errors::Internal(
        "Could not convert proto to JSON string: ",
        absl::string_view(error_msg.data(), error_msg.length()));
  }
  return absl::OkStatus();
}

}  // namespace

MemoryProfile ConvertXPlaneToMemoryProfile(const XPlane& host_plane,
                                           int64_t max_num_snapshots) {
  MemoryProfile memory_profile = GenerateMemoryProfile(&host_plane);
  ProcessMemoryProfileProto(max_num_snapshots, &memory_profile);
  // Default version number is 0, set version number to 1 here due to the new
  // memory profile sampling algorithm.
  memory_profile.set_version(1);
  return memory_profile;
}

absl::Status ConvertXSpaceToMemoryProfileJson(const XSpace& xspace,
                                              std::string* json_output) {
  if (const XPlane* host_plane =
          FindPlaneWithName(xspace, kHostThreadsPlaneName)) {
    MemoryProfile memory_profile = ConvertXPlaneToMemoryProfile(*host_plane);
    TF_RETURN_IF_ERROR(ConvertProtoToJson(memory_profile, json_output));
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
