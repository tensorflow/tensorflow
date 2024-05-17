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
#include "tensorflow/core/profiler/convert/dcn_analysis.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/convert/dcn_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tsl/profiler/utils/math_utils.h"
#include "tsl/profiler/utils/tpu_xplane_utils.h"
#include "tsl/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::kMaxCollectivesToDisplay;
using tsl::profiler::kMegaScaleDcnReceive;
using tsl::profiler::LineIdType;
using tsl::profiler::MicroToNano;

void DcnBurstManager::ResetBurstState() {
  active_burst_messages_ = 0;
  straggler_idx_ = 0;
  active_burst_.num_messages = 0;
  active_burst_.max_overlapping_messages = 0;
  active_burst_.start_timestamp_ns = 0;
  active_burst_.end_timestamp_ns = 0;
  active_burst_.burst_size_bytes = 0;
}

void DcnBurstManager::CreateBursts(const TimestampMap& tm_events) {
  ResetBurstState();
  for (const auto& tm_event : tm_events) {
    if (active_burst_messages_ < 0) {
      LOG_FIRST_N(WARNING, 10)
          << "Negative messages in burst, bursts will be incorrect.";
    }
    if (active_burst_messages_ == 0) {
      // When no messages are active, next event starts a new burst
      active_burst_.start_timestamp_ns = tm_event.first;
    }
    active_burst_messages_ += tm_event.second->message_diff;
    if (tm_event.second->message_diff > 0) {
      // On beginning of message increase messages and bytes
      active_burst_.num_messages += tm_event.second->message_diff;
      active_burst_.burst_size_bytes += tm_event.second->size_diff;
    } else {
      // On end of message, register straggler
      Straggler straggler = {tm_event.second->duration_ns,   // duration_ns
                             tm_event.second->timestamp_ns,  // end_timestamp_ns
                             tm_event.second->size_diff * (-1),  // size_bytes
                             tm_event.second->src_slice_id};     // src_slice_id
      active_burst_.stragglers[straggler_idx_] = straggler;
      straggler_idx_ = (straggler_idx_ + 1) % kMaxStragglersPerBurst;
    }
    active_burst_.max_overlapping_messages =
        std::max(active_burst_.max_overlapping_messages,
                 static_cast<uint64_t>(active_burst_messages_));
    // If we are back at 0 messages, the burst has finished and can be added
    // to the bursts_ vector.
    if (active_burst_messages_ == 0) {
      active_burst_.end_timestamp_ns = tm_event.first;
      total_latency_ +=
          (active_burst_.end_timestamp_ns - active_burst_.start_timestamp_ns);
      bursts_.emplace_back(std::move(active_burst_));
      ResetBurstState();
    }
  }
}

DcnEventsProcessor::DcnEventsProcessor(uint32_t num_tpu_tensor_cores,
                                       bool is_megacore)
    : num_tpu_tensor_cores_(num_tpu_tensor_cores), is_megacore_(is_megacore) {
  // Register all MSXLA messages we may need to analyze. Currently only
  // receive messages are processed.
  registered_dcn_messages_.push_back(kMegaScaleDcnReceive);
  tpu_collective_ts_map_.resize(num_tpu_tensor_cores_);
  tpu_collective_bursts_.resize(num_tpu_tensor_cores_);
}

// Sets up map between registered Megascale messages and their event metadata
// so they can be captured from host events.
void DcnEventsProcessor::SetupMessageInfo(const XPlaneVisitor& plane) {
  plane.ForEachEventMetadata([&](const XEventMetadataVisitor& event_metadata) {
    if (std::find(registered_dcn_messages_.begin(),
                  registered_dcn_messages_.end(),
                  event_metadata.Name()) != registered_dcn_messages_.end()) {
      megascale_msg_[event_metadata.Name()] = event_metadata.Id();
    }
  });
}

// If we use megacore, collective traffic goes to even TPU tensor cores.
// Odd ones are woken up from their even pair (e.g. 0 wakes up 1).
uint32_t DcnEventsProcessor::FindTpuIdx(int tpu) {
  uint32_t num_tpus = num_tpu_tensor_cores_;
  if (is_megacore_) {
    num_tpus /= 2;
  }
  uint32_t tpu_idx = tpu % num_tpus;
  if (is_megacore_) {
    tpu_idx = tpu_idx * 2;
  }
  return tpu_idx;
}

void DcnEventsProcessor::GenerateTimestampEvents(
    const DcnMessage& dcn_message) {
  // Create one event for the beginning and one for the end of the message
  std::shared_ptr<TimestampEvent> start_event(
      new TimestampEvent{dcn_message.start_timestamp_ns, 0, 1,
                         dcn_message.size_bytes, dcn_message.slice_src});
  std::shared_ptr<TimestampEvent> end_event(new TimestampEvent{
      dcn_message.end_timestamp_ns,
      static_cast<uint64_t>(MicroToNano(dcn_message.duration_us)), -1,
      -1 * dcn_message.size_bytes, dcn_message.slice_src});

  // Add messages to host timestamp event map
  std::pair<uint64_t, std::shared_ptr<TimestampEvent>> start_event_entry =
      std::make_pair(dcn_message.start_timestamp_ns, start_event);
  std::pair<uint64_t, std::shared_ptr<TimestampEvent>> end_event_entry =
      std::make_pair(dcn_message.end_timestamp_ns, end_event);
  host_ts_map_.insert(start_event_entry);
  host_ts_map_.insert(end_event_entry);

  // Add messages to the proper TPU collective timestamp event map.
  const std::string& collective_name = dcn_message.collective_name;
  uint32_t tpu_idx = FindTpuIdx(dcn_message.tpu_dst);
  auto& m = tpu_collective_ts_map_[tpu_idx][collective_name];
  m.insert(start_event_entry);
  m.insert(end_event_entry);
}

void DcnEventsProcessor::PrintTimestampEvents() {
  for (const auto& host_ts : host_ts_map_) {
    LOG(INFO) << host_ts.first << ": " << host_ts.second->timestamp_ns << " "
              << host_ts.second->duration_ns << " "
              << host_ts.second->message_diff << " "
              << host_ts.second->size_diff << " "
              << host_ts.second->src_slice_id;
  }
  for (uint32_t tpu_idx = 0; tpu_idx < num_tpu_tensor_cores_; tpu_idx++) {
    LOG(INFO) << "TPU: " << tpu_idx;
    for (const auto& col_id : tpu_collective_ts_map_[tpu_idx]) {
      LOG(INFO) << col_id.first;
      for (const auto& tpu_col_ts :
           tpu_collective_ts_map_[tpu_idx][col_id.first]) {
        LOG(INFO) << tpu_col_ts.first << ": " << tpu_col_ts.second->timestamp_ns
                  << " " << tpu_col_ts.second->duration_ns << " "
                  << tpu_col_ts.second->message_diff << " "
                  << tpu_col_ts.second->size_diff << " "
                  << tpu_col_ts.second->src_slice_id;
      }
    }
  }
}

// Uses heuristics to qualify a good enough amount of collectives.
// kMaxCollectivesToDisplay - 1 are displayed.
// Collectives with < 5% of total host BW time are never qualified
// Collectives with < 20% of total host BW time are qualified if less than 4
//   collectives  have already been qualified.
// Top 8 collectives with > 20% of total host BW time are qualified
uint32_t DcnEventsProcessor::NumCollectivesQualified(
    const std::vector<uint64_t>& latencies) {
  uint32_t num_collectives_qualified = 0;
  // Allow for 1 line to display stragglers of non-qualified collectives.
  uint32_t max_collectives = kMaxCollectivesToDisplay - 1;
  for (const auto& lat : latencies) {
    if (lat < host_dcn_bursts_.TotalLatency() * 0.05) {
      return num_collectives_qualified;
    } else if (lat < host_dcn_bursts_.TotalLatency() * 0.2 &&
               num_collectives_qualified >= (max_collectives / 2)) {
      return num_collectives_qualified;
    } else if (num_collectives_qualified >= max_collectives) {
      return num_collectives_qualified;
    } else {
      num_collectives_qualified++;
    }
  }
  return latencies.size();
}

// Find which collectives you are going to display in details (dedicated line)
// and which not (shared line for stragglers).
// Order collectives based on burst latency -- then qualify the top ones based
// on NumCollectivesQualified function.
void DcnEventsProcessor::QualifyCollectives() {
  for (auto tpu_idx = 0; tpu_idx < num_tpu_tensor_cores_; tpu_idx++) {
    std::vector<uint64_t> latency_to_order;
    latency_to_order.reserve(tpu_collective_bursts_[tpu_idx].size());
    for (const auto& col_info : tpu_collective_bursts_[tpu_idx]) {
      latency_to_order.emplace_back(col_info.second.TotalLatency());
    }
    std::sort(latency_to_order.begin(), latency_to_order.end(),
              std::greater<uint64_t>());
    uint32_t num_collectives_qualified =
        NumCollectivesQualified(latency_to_order);
    if (num_collectives_qualified > 0) {
      uint32_t min_latency_to_qualify =
          latency_to_order[num_collectives_qualified - 1];
      uint32_t col_num = 0;
      for (auto& col_info : tpu_collective_bursts_[tpu_idx]) {
        if (col_info.second.TotalLatency() >= min_latency_to_qualify) {
          col_info.second.SetToDisplay(true);
          if (++col_num == kMaxCollectivesToDisplay - 1) break;
        }
      }
    }
  }
}

void DcnEventsProcessor::GenerateBursts() {
  host_dcn_bursts_.CreateBursts(host_ts_map_);
  host_dcn_bursts_.SetToDisplay(true);

  for (auto tpu_idx = 0; tpu_idx < num_tpu_tensor_cores_; tpu_idx++) {
    for (const auto& col_info : tpu_collective_ts_map_[tpu_idx]) {
      tpu_collective_bursts_[tpu_idx][col_info.first].CreateBursts(
          tpu_collective_ts_map_[tpu_idx][col_info.first]);
    }
  }
  QualifyCollectives();
}

void DcnEventsProcessor::ProcessReceiveMessages(const XPlaneVisitor& plane) {
  plane.ForEachLine([&](const XLineVisitor& line) {
    uint32_t recv_msg_id = megascale_msg_[kMegaScaleDcnReceive];
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Id() == recv_msg_id) {
        DcnMessage dcn_message = GetDcnMessageFromXEvent(event);
        // TODO(emizan): Report invalid and clock skew messages somehow.
        // TODO(emizan): Bring back loopback messages when MSXLA fixes them.
        if (dcn_message.validity_info == DCN_MESSAGE_VALID) {
          GenerateTimestampEvents(dcn_message);
        }
        received_messages_.emplace_back(std::move(dcn_message));
      }
    });
  });
  GenerateBursts();
}

absl::string_view DcnEventsProcessor::GetBwInfo(bool is_per_tpu,
                                                const DcnBurst& burst,
                                                float& burst_mean_bw,
                                                float& burst_bw_utilization) {
  absl::string_view bw_level;
  uint32_t bw_divider = 1;
  burst_mean_bw = static_cast<float>(burst.burst_size_bytes) /
                  (burst.end_timestamp_ns - burst.start_timestamp_ns);
  if (is_per_tpu) {
    bw_divider = num_tpu_tensor_cores_;
    if (is_megacore_) {
      bw_divider /= 2;
    }
  }
  // Have 3 BW categories (low/med/high) to limit the amount of colors in the
  // trace viewer
  if (burst_mean_bw < kLimitLowHostDcnBw / bw_divider) {
    bw_level = "Low BW";
  } else if (burst_mean_bw < kLimitMedHostDcnBw / bw_divider) {
    bw_level = "Med BW";
  } else {
    bw_level = "High BW";
  }
  burst_bw_utilization = burst_mean_bw / (kMaxHostDcnBw / bw_divider);
  return bw_level;
}

void DcnEventsProcessor::AddHostDcnTrafficToXPlane(XPlane* host_xplane) {
  if (!host_dcn_bursts_.ToDisplay()) return;
  XPlaneBuilder plane_builder(host_xplane);
  XLineBuilder line =
      plane_builder.GetOrCreateLine(LineIdType::kDcnHostTraffic);
  line.SetNameIfEmpty("DCN Host Bandwidth");
  line.SetTimestampNs(0);
  XStatMetadata* bw_stat_metadata =
      plane_builder.GetOrCreateStatMetadata("Bandwidth (GBytes/sec)");
  XStatMetadata* bw_util_stat_metadata =
      plane_builder.GetOrCreateStatMetadata("Bandwidth Utilization");
  XStatMetadata* num_msg_stat_metadata =
      plane_builder.GetOrCreateStatMetadata("Total Messages");
  XStatMetadata* max_overlap_msg_stat_metadata =
      plane_builder.GetOrCreateStatMetadata("Max Overlapping Messages");
  XStatMetadata* avg_msg_size_stat_metadata =
      plane_builder.GetOrCreateStatMetadata("Average Message Size (Bytes)");
  for (const auto& host_burst : host_dcn_bursts_.GetBursts()) {
    float burst_mean_bw, bw_utilization;
    absl::string_view bw_level =
        GetBwInfo(false, host_burst, burst_mean_bw, bw_utilization);
    XEventMetadata* event_metadata =
        plane_builder.GetOrCreateEventMetadata(bw_level);
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetNs(host_burst.start_timestamp_ns);
    event.SetDurationNs(host_burst.end_timestamp_ns -
                        host_burst.start_timestamp_ns);

    // Using std::string to limit number of decimals.
    event.ParseAndAddStatValue(*bw_stat_metadata,
                               std::to_string(burst_mean_bw));
    event.ParseAndAddStatValue(*bw_util_stat_metadata,
                               std::to_string(bw_utilization));
    event.AddStatValue(*num_msg_stat_metadata, host_burst.num_messages);
    event.AddStatValue(*max_overlap_msg_stat_metadata,
                       host_burst.max_overlapping_messages);
    uint32_t avg_message_size =
        host_burst.burst_size_bytes / host_burst.num_messages;
    event.AddStatValue(*avg_msg_size_stat_metadata, avg_message_size);
  }
}

void DcnEventsProcessor::AddUnqualifiedCollectivesToXPlane(
    XPlaneBuilder& plane_builder, uint32_t tpu_idx) {
  XLineBuilder line =
      plane_builder.GetOrCreateLine(LineIdType::kDcnCollectiveTrafficMax);
  line.SetNameIfEmpty("Remaining collectives");
  line.SetTimestampNs(0);
  for (const auto& col_item : tpu_collective_bursts_[tpu_idx]) {
    if (col_item.second.ToDisplay()) continue;
    for (const auto& col_burst : col_item.second.GetBursts()) {
      XEventMetadata* straggler_event_metadata =
          plane_builder.GetOrCreateEventMetadata(col_item.first);
      uint32_t stragglers_processed = 0;
      XStatMetadata* straggler_src_slice_stat_metadata =
          plane_builder.GetOrCreateStatMetadata("Source slice");
      XStatMetadata* straggler_duration_ns_stat_metadata =
          plane_builder.GetOrCreateStatMetadata("Duration ns");
      XStatMetadata* straggler_send_time_ns_stat_metadata =
          plane_builder.GetOrCreateStatMetadata("Send timestamp ns");
      XStatMetadata* straggler_recv_time_ns_stat_metadata =
          plane_builder.GetOrCreateStatMetadata("Recv timestamp ns");
      for (const auto& straggler : col_burst.stragglers) {
        XEventBuilder straggler_event =
            line.AddEvent(*straggler_event_metadata);
        straggler_event.SetOffsetNs(straggler.end_timestamp_ns - 10000);
        straggler_event.SetDurationNs(10000);
        straggler_event.AddStatValue(*straggler_src_slice_stat_metadata,
                                     straggler.src_slice_id);
        straggler_event.AddStatValue(*straggler_duration_ns_stat_metadata,
                                     straggler.duration_ns);
        straggler_event.AddStatValue(
            *straggler_send_time_ns_stat_metadata,
            straggler.end_timestamp_ns - straggler.duration_ns);
        straggler_event.AddStatValue(*straggler_recv_time_ns_stat_metadata,
                                     straggler.end_timestamp_ns);
        if (++stragglers_processed >= col_burst.num_messages) break;
      }
    }
  }
}

void DcnEventsProcessor::AddQualifiedCollectivesToXPlane(
    XPlaneBuilder& plane_builder, uint32_t tpu_idx) {
  uint32_t total_collectives = 0;
  for (const auto& col_item : tpu_collective_bursts_[tpu_idx]) {
    // Skip collectives not enabled for display.
    if (!col_item.second.ToDisplay()) continue;
    const std::string& col_name = col_item.first;
    XLineBuilder line = plane_builder.GetOrCreateLine(
        LineIdType::kDcnCollectiveTraffic + total_collectives++);
    line.SetNameIfEmpty(col_name);
    line.SetTimestampNs(0);
    XStatMetadata* bw_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Bandwidth (GBytes/sec)");
    XStatMetadata* bw_util_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Bandwidth Utilization");
    XStatMetadata* num_msg_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Total Messages");
    XStatMetadata* max_overlap_msg_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Max Overlapping Messages");
    XStatMetadata* avg_msg_size_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Average Message Size (Bytes)");
    XStatMetadata* straggler_details_metadata =
        plane_builder.GetOrCreateStatMetadata("Straggler info:");
    XStatMetadata* straggler_src_slice_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Source slice");
    XStatMetadata* straggler_duration_ns_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Duration ns");
    XStatMetadata* straggler_send_time_ns_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Send timestamp ns");
    XStatMetadata* straggler_recv_time_ns_stat_metadata =
        plane_builder.GetOrCreateStatMetadata("Recv timestamp ns");
    for (const auto& col_burst : col_item.second.GetBursts()) {
      float burst_mean_bw, bw_utilization;
      absl::string_view bw_level =
          GetBwInfo(true, col_burst, burst_mean_bw, bw_utilization);
      XEventMetadata* event_metadata =
          plane_builder.GetOrCreateEventMetadata(bw_level);
      XEventBuilder event = line.AddEvent(*event_metadata);
      event.SetOffsetNs(col_burst.start_timestamp_ns);
      event.SetDurationNs(col_burst.end_timestamp_ns -
                          col_burst.start_timestamp_ns);
      event.ParseAndAddStatValue(*bw_stat_metadata,
                                 std::to_string(burst_mean_bw));
      event.ParseAndAddStatValue(*bw_util_stat_metadata,
                                 std::to_string(bw_utilization));
      event.AddStatValue(*num_msg_stat_metadata, col_burst.num_messages);
      event.AddStatValue(*max_overlap_msg_stat_metadata,
                         col_burst.max_overlapping_messages);
      event.AddStatValue(*avg_msg_size_stat_metadata,
                         col_burst.burst_size_bytes / col_burst.num_messages);
      // Add straggler info.
      XEventMetadata* straggler_event_metadata =
          plane_builder.GetOrCreateEventMetadata("Straggler");
      uint32_t stragglers_processed = 0;
      std::string straggler_details = "Stragglers:\n";
      for (const auto& straggler : col_burst.stragglers) {
        // Add an event for the last straggler
        if (straggler.end_timestamp_ns == col_burst.end_timestamp_ns) {
          XEventBuilder straggler_event =
              line.AddEvent(*straggler_event_metadata);
          straggler_event.SetOffsetNs(straggler.end_timestamp_ns -
                                      straggler.duration_ns);
          straggler_event.SetDurationNs(straggler.duration_ns);
          straggler_event.AddStatValue(*straggler_src_slice_stat_metadata,
                                       straggler.src_slice_id);
          straggler_event.AddStatValue(*straggler_duration_ns_stat_metadata,
                                       straggler.duration_ns);
          straggler_event.AddStatValue(
              *straggler_send_time_ns_stat_metadata,
              straggler.end_timestamp_ns - straggler.duration_ns);
          straggler_event.AddStatValue(*straggler_recv_time_ns_stat_metadata,
                                       straggler.end_timestamp_ns);
        }
        // Add text metadata for all stragglers.
        straggler_details +=
            "  Src slice: " + std::to_string(straggler.src_slice_id) +
            " -- Duration (ns): " + std::to_string(straggler.duration_ns) +
            " -- [Send Timestamp, Recv Timestamp]: [" +
            std::to_string(straggler.end_timestamp_ns - straggler.duration_ns) +
            ", " + std::to_string(straggler.end_timestamp_ns) + "]\n";
        if (++stragglers_processed >= col_burst.num_messages) break;
      }
      event.AddStatValue(*straggler_details_metadata, straggler_details);
    }
  }
}

void DcnEventsProcessor::AddTpuCollectiveDcnTrafficToXPlane(
    XPlane* device_xplane) {
  XPlaneBuilder plane_builder(device_xplane);
  auto tpu = tsl::profiler::GetTensorCoreId(plane_builder.Name());
  if (!tpu.has_value()) return;
  uint32_t tpu_idx = FindTpuIdx(tpu.value());
  AddQualifiedCollectivesToXPlane(plane_builder, tpu_idx);
  AddUnqualifiedCollectivesToXPlane(plane_builder, tpu_idx);
}
}  // namespace profiler
}  // namespace tensorflow
