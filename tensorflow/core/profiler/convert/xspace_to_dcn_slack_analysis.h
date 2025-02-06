/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XSPACE_TO_DCN_SLACK_ANALYSIS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XSPACE_TO_DCN_SLACK_ANALYSIS_H_

#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/profiler/protobuf/dcn_collective_info.pb.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"
#include "tensorflow/core/profiler/protobuf/topology.pb.h"
#include "tensorflow/core/profiler/utils/hlo_proto_map.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

using tensorflow::profiler::DcnSlackAnalysis;

namespace dcn_analysis_internal {

struct DcnOpState {
  uint64_t start_time = 0;
  uint64_t end_time = 0;

  // Duration of containing send/send-done/recv/recv-done ops that needs to be
  // subtracted from the total duration
  uint64_t overlapping_duration = 0;
  std::string rendezvous_name;
  std::string transfer_type;
  uint64_t stall_duration_ns = 0;
  std::string send_op_name;
  int replica_group_size = 0;

  OpInstance send;
  OpInstance send_done;
  OpInstance recv;
  OpInstance recv_done;
};

// Structure to extract and store the DcnHostEvents.
struct DcnHostEvent {
  std::string rendezvous_name;
  tsl::profiler::Timespan timespan;
  int multi_slice_device_id;
};

// When visiting DcnHostEvents from the megascale planes, The events are stored
// in separate lines in an ascending (by time) order. The List allows insertion
// of multiple arrays of sorted events.
class DcnHostEventList {
 public:
  // Insert the event into the sorted list.
  void insert(DcnHostEvent event);

  // Pop the events from the front that is included within the timestamp when
  // available.
  std::optional<DcnHostEvent> pop(const tsl::profiler::Timespan& timespan);

  // Number of events.
  int size() const { return events_.size(); }

 private:
  std::list<DcnHostEvent> events_;
  std::list<DcnHostEvent>::iterator iter_ = events_.begin();
};

struct InstrMetadata {
  xla::HloOpcode opcode;
  uint64_t channel_id;
  std::optional<std::string> rendezvous_name;
  int64_t size = 0;
  std::optional<std::string> transfer_type;
};

class DcnTracker {
 public:
  explicit DcnTracker(const tensorflow::profiler::HloProtoMap& hlo_proto_map,
                      bool is_megacore)
      : hlo_proto_map_(hlo_proto_map), is_megacore_(is_megacore) {}

  absl::StatusOr<InstrMetadata> GetInstructionMetadata(std::string_view module,
                                                       std::string_view instr);

  DcnSlackAnalysis Finalize();

  void DebugString();

  void VisitOp(const InstrMetadata& instr,
               const tsl::profiler::XEventVisitor& visitor);

  void VisitHostEvent(const DcnHostEvent& event);

  void ProcessTopology(const tensorflow::profiler::Topology& topology);

 private:
  DcnSlackAnalysis slack_analysis_;
  absl::flat_hash_map<std::string, DcnOpState> rendezvous_to_op_map_;
  absl::flat_hash_map<uint64_t, std::string> channel_id_to_rendezvous_map_;
  absl::flat_hash_map<std::string, InstrMetadata> instruction_metadata_map_;
  absl::flat_hash_map<std::string, DcnHostEventList> core_id_to_host_event_map_;
  const tensorflow::profiler::HloProtoMap& hlo_proto_map_;
  absl::flat_hash_map<int, int> global_chip_id_to_local_index_map_;
  absl::flat_hash_map<std::string, std::unique_ptr<xla::HloModule>>
      hlo_module_cache_;
  absl::flat_hash_map<std::string, int> rendezvous_to_replica_group_size_map_;
  bool is_megacore_ = true;

  absl::StatusOr<InstrMetadata> GetInstrMetadataFromHloModule(
      std::string_view module, std::string_view instr);

  void UpdateActiveOps(uint64_t duration);

  void SummarizeDcnSlackAnalysis();

  std::optional<DcnHostEvent> GetCollectiveHostEvent(
      int core_id, std::string_view rendezvous_name,
      tsl::profiler::Timespan timespan);

  // GetLocalIndex when available, else return the global_device_id itself.
  int GetLocalIndex(int dcn_device_id);

  // Get number of replica group
  int GetReplicaGroupSize(const std::string& rendezvous_name,
                          const tsl::profiler::XEventVisitor& visitor);

  // Compute data transmitted size based on number of replica groups
  uint64_t ComputeTransmittedDataSize(int64_t buffer_size, int group_size,
                                      const std::string& transfer_type);
};

}  // namespace dcn_analysis_internal

// Convert Hlo Events in XSpace to Dcn Slack analysis.
DcnSlackAnalysis ConvertXSpaceToDcnSlackAnalysis(
    const tensorflow::profiler::XSpace& xspace,
    const tensorflow::profiler::XPlane* dcn_host_plane,
    const tensorflow::profiler::Topology* topology, bool is_megacore = true);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XSPACE_TO_DCN_SLACK_ANALYSIS_H_
