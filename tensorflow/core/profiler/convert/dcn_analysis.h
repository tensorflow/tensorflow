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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_DCN_ANALYSIS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_DCN_ANALYSIS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/profiler/convert/dcn_utils.h"

namespace tensorflow {
namespace profiler {

// Structure representing a DcnMessage using two entries:
// One for the start of the message and one for the end.
struct TimestampEvent {
  uint64_t timestamp_ns;  // TraceMe logging timestamp
  uint64_t duration_ns;   // 0 for start of message, duration for end of message
  int32_t message_diff;   // +1/-1 for start/end of message.
                          // Makes handling 0-sized messages easier and is
                          // convenient for the burst generation algorithm.
  size_t size_diff;       // +size/-size for start/end of message.
  int32_t src_slice_id;   // Source slice for message, used for stragglers
};

// We use an multi map since TimestampEvents will be ordered and we
// need separate entries for possible events happening at exactly the
// same time.
typedef std::multimap<uint64_t, std::shared_ptr<TimestampEvent>> TimestampMap;
typedef absl::flat_hash_map<std::string, TimestampMap> CollectiveTimestampMap;

// Straggler messages. These are shown at the end of the bursts they belong to.
struct Straggler {
  uint64_t duration_ns;       // Message duration in ns
  uint64_t end_timestamp_ns;  // End of the message. For the last straggler
                              // this will be the end of the burst
  size_t size_bytes;          // Size of the message in bytes
  int32_t src_slice_id;       // Source slice of the message
                              // TODO(emizan) Add host info.
};

static constexpr uint32_t kMaxStragglersPerBurst = 4;

// DCN Burst description.
// A burst is defined as a period of time during which there is at least one
// message in the network. Since DCN traffic is bursty this structure is
// convenient to summarize 100K+ messages in a few 10s of bursts.
// Burst scope is flexible. In this analysis we have per-host bursts, which
// include messages arriving on a single host independent of sender/target TPU/
// and collective. We also have per collective/TPU bursts which include messages
// for a single collective+TPU combination.
struct DcnBurst {
  uint64_t start_timestamp_ns;        // Beginning of burst in ns
  uint64_t end_timestamp_ns;          // End of burst in ns
  uint64_t burst_size_bytes;          // Total number of bytes in burst
  uint64_t num_messages;              // Messages in burst
  uint64_t max_overlapping_messages;  // Max overlapping messages in burst
  // Buffer of stragglers in a bursts. Contains the last few messages in a burst
  std::array<Straggler, kMaxStragglersPerBurst> stragglers;
};

// Class with functionality to generate DcnBursts out of TimestampEvents.
// Burst creation is a non-trivial state machine
class DcnBurstManager {
 public:
  DcnBurstManager() = default;
  uint64_t TotalLatency() const { return total_latency_; }
  void SetToDisplay(bool to_display) { to_display_ = to_display; }
  bool ToDisplay() const { return to_display_; }
  const std::vector<DcnBurst> &GetBursts() const { return bursts_; }

  // Run burst state machine creation out of timestamp map.
  void CreateBursts(const TimestampMap &tm_events);
  // For debugging purposes.
  void PrintBursts() {
    for (const auto &burst : bursts_) {
      LOG(INFO) << burst.start_timestamp_ns << " " << burst.end_timestamp_ns
                << " " << burst.num_messages << " " << burst.burst_size_bytes
                << " " << burst.max_overlapping_messages;
    }
  }

 private:
  std::vector<DcnBurst> bursts_;  // Bursts created by this manager
  uint64_t total_latency_ = 0;    // Total latency of all bursts created
                                  // Used to see if bursts will be displayed
  bool to_display_ = false;       // Set to true to enable burst display

  int32_t active_burst_messages_;  // Used by burst creation state machine.
  DcnBurst active_burst_;          // Active burst in creation
  uint32_t straggler_idx_;

  // Initializes state machine when new burst is detected.
  void ResetBurstState();
};

typedef absl::flat_hash_map<std::string, DcnBurstManager>
    CollectiveBurstManager;

class DcnEventsProcessor {
 public:
  DcnEventsProcessor() = delete;
  DcnEventsProcessor(uint32_t num_tpu_tensor_cores, bool is_megacore);

  uint32_t NumTpuTensorCores() const { return num_tpu_tensor_cores_; }
  bool IsMegacore() const { return is_megacore_; }

  // Populates available megascale messages from event metadata.
  void SetupMessageInfo(const tsl::profiler::XPlaneVisitor &plane);

  std::optional<int32_t> MegaScaleMessageId(absl::string_view msg_name) const {
    auto iter = megascale_msg_.find(msg_name);
    if (iter != megascale_msg_.end()) {
      return iter->second;
    }
    return std::nullopt;
  }

  uint32_t NumReceivedMessages() const { return received_messages_.size(); }
  const tensorflow::profiler::DcnMessage &GetMessage(uint32_t i) const {
    return received_messages_[i];
  }

  // Checks if messages with msg event name have been found in event metadata.
  bool HasDcnMessages(absl::string_view msg_name) const {
    return (megascale_msg_.find(msg_name) != megascale_msg_.end());
  }

  const TimestampMap &HostTsMap() const { return host_ts_map_; }
  const std::vector<DcnBurst> &GetHostBursts() const {
    return host_dcn_bursts_.GetBursts();
  }

  // Main function to process receive messages, and call other functions
  // to generate timestamp events and bursts.
  void ProcessReceiveMessages(const tsl::profiler::XPlaneVisitor &plane);

  // Update XPlanes using DCN traffic info
  void AddHostDcnTrafficToXPlane(tsl::profiler::XPlane *host_xplane);
  void AddTpuCollectiveDcnTrafficToXPlane(tsl::profiler::XPlane *device_xplane);

 private:
  // Tensor cores and megacore flag for this host. DCN messages are sent to a
  // TPU chip, so we need to know the number of tensor cores and whether
  // megacore is used to map DCN traffic to the proper tensor core.
  const uint32_t num_tpu_tensor_cores_;
  const bool is_megacore_;

  // Used for visualization of BW and computation of BW utilization.
  static constexpr float kLimitLowHostDcnBw = 4.17;
  static constexpr float kLimitMedHostDcnBw = 8.34;
  static constexpr float kMaxHostDcnBw = 12.5;

  std::vector<absl::string_view> registered_dcn_messages_;

  // Available megascale messages for this trace.
  absl::flat_hash_map<absl::string_view, int32_t> megascale_msg_;

  std::vector<tensorflow::profiler::DcnMessage> received_messages_;

  // TimestampMaps for messages that arrive to this host
  // and for messages of distinct collectives going to different TPUs.
  TimestampMap host_ts_map_;
  std::vector<CollectiveTimestampMap> tpu_collective_ts_map_;

  // DcnBurstManagers for bursts that arrive to this host
  // and for burst from distinct collectives going to different TPUs.
  DcnBurstManager host_dcn_bursts_;
  std::vector<CollectiveBurstManager> tpu_collective_bursts_;

  // Find the TPU index a DCN message goes to.
  uint32_t FindTpuIdx(int tpu);

  // Generates BW info to display in the trace viewer.
  // This included trace event BW level string, mean BW per burst and
  // utilization.
  absl::string_view GetBwInfo(bool is_per_tpu, const DcnBurst &burst,
                              float &burst_mean_bw,
                              float &burst_bw_utilization);

  // Qualify collectives to display on trace viewer.
  // Qualified collectives are given a dedicated line, while for the rest
  // we share a single line for their stragglers.
  uint32_t NumCollectivesQualified(const std::vector<uint64_t> &latencies);
  void QualifyCollectives();
  // Export collective DCN activity to trace viewer.
  void AddQualifiedCollectivesToXPlane(
      tsl::profiler::XPlaneBuilder &plane_builder, uint32_t tpu_idx);
  void AddUnqualifiedCollectivesToXPlane(
      tsl::profiler::XPlaneBuilder &plane_builder, uint32_t tpu_idx);

  // Create timestamp events for every message
  void GenerateTimestampEvents(
      const tensorflow::profiler::DcnMessage &dcn_message);
  // For debugging purposes
  void PrintTimestampEvents();
  // Generate bursts (host and TPU/collective) from timestamp events.
  void GenerateBursts();
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_DCN_ANALYSIS_H_
