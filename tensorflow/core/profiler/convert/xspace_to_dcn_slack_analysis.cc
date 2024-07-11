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
#include "tensorflow/core/profiler/convert/xspace_to_dcn_slack_analysis.h"

#include <sys/types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/profiler/protobuf/dcn_collective_info.pb.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"
#include "tensorflow/core/profiler/protobuf/topology.pb.h"
#include "tensorflow/core/profiler/utils/hlo_module_utils.h"
#include "tensorflow/core/profiler/utils/hlo_proto_map.h"
#include "tensorflow/core/profiler/utils/hlo_proto_to_module.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tsl/platform/regexp.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/utils/math_utils.h"
#include "tsl/profiler/utils/tf_xplane_visitor.h"
#include "tsl/profiler/utils/timespan.h"
#include "tsl/profiler/utils/tpu_xplane_utils.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

using tensorflow::profiler::DcnSlackSummary;
using tensorflow::profiler::Topology;
using tsl::profiler::CreateTfXPlaneVisitor;
using tsl::profiler::FindLineWithName;
using tsl::profiler::kXlaOpLineName;
using tsl::profiler::NanoToMicro;
using tsl::profiler::PicoToMicro;
using tsl::profiler::SafeDivide;
using tsl::profiler::StatType;
using tsl::profiler::Timespan;
using tsl::profiler::XEventContextTracker;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;
using xla::HloOpcode;

// TODO: Identify mechanism to maintain consistency between producer and
// consumer here.
const char kHostEventRegex[] = {
    "device_[0-9]+([0-9][0-9][0-9][0-9][0-9])_gid_(.*)"};

std::optional<std::string> GetAttributeFromInstr(
    const xla::HloInstruction* instr, std::string_view attribute) {
  std::optional<std::string> attribute_value;
  if (instr->frontend_attributes().IsInitialized() &&
      !instr->frontend_attributes().map().empty() &&
      instr->frontend_attributes().map().contains(attribute)) {
    attribute_value = instr->frontend_attributes().map().at(attribute);
  }
  return attribute_value;
}
std::optional<std::string> GetRendezvous(const xla::HloInstruction* instr) {
  return GetAttributeFromInstr(instr, xla::kXlaHostTransferRendezvousNameAttr);
}

dcn_analysis_internal::DcnHostEvent ParseDcnHostEvent(
    const XEventVisitor& visitor) {
  dcn_analysis_internal::DcnHostEvent event;
  static const LazyRE2 re = {kHostEventRegex};
  RE2::FullMatch(visitor.Name(), *re, &event.multi_slice_device_id,
                 &event.rendezvous_name);

  event.timespan = visitor.GetTimespan();
  return event;
}

std::optional<std::string> GetTransferType(const xla::HloInstruction* instr) {
  return GetAttributeFromInstr(instr, "_xla_megascale_transfer_type");
}

std::string HostCollectiveKey(int index_on_host,
                              std::string_view rendezvous_name) {
  return absl::StrCat(index_on_host, "_", rendezvous_name);
}

DcnCollectiveInfoProto GetDcnCollectiveInfoProto(const XEventVisitor& xevent) {
  DcnCollectiveInfoProto dcn_collective_info;
  xevent.Metadata().ForEachStat([&](const XStatVisitor& xstat) {
    if (static_cast<StatType>(*xstat.Type()) == StatType::kDcnCollectiveInfo) {
      absl::string_view byte_value = xstat.BytesValue();
      if (!dcn_collective_info.ParseFromArray(byte_value.data(),
                                              byte_value.size())) {
        LOG(WARNING) << "Could not parse DcnCollectiveInfoProto from metadata.";
      }
    }
  });

  return dcn_collective_info;
}

}  // namespace

namespace dcn_analysis_internal {

void DcnHostEventList::insert(DcnHostEvent event) {
  if (iter_ != events_.end() && event.timespan < iter_->timespan) {
    // The event being inserted is from a new line, Reset iterator to the
    // beginning.
    iter_ = events_.begin();
  }
  while (iter_ != events_.end() && iter_->timespan < event.timespan) {
    iter_++;
  }
  iter_ = events_.insert(iter_, event);
}

std::optional<DcnHostEvent> DcnHostEventList::pop(const Timespan& timespan) {
  while (!events_.empty() && events_.front().timespan < timespan) {
    events_.pop_front();
  }

  if (!events_.empty() &&
      (timespan.Includes(events_.front().timespan.begin_ps()) ||
       events_.front().timespan.Includes(timespan.begin_ps()))) {
    DcnHostEvent front = events_.front();
    events_.pop_front();
    return front;
  } else {
    return std::nullopt;
  }
}

absl::StatusOr<InstrMetadata> DcnTracker::GetInstrMetadataFromHloModule(
    std::string_view module_name, std::string_view instr_name) {
  if (!hlo_module_cache_.contains(module_name)) {
    TF_ASSIGN_OR_RETURN(auto hlo_proto,
                        hlo_proto_map_.GetHloProtoByModuleName(module_name));
    TF_ASSIGN_OR_RETURN(auto module, ConvertHloProtoToModule(*hlo_proto));
    hlo_module_cache_[module_name] = std::move(module);
  }
  const auto& hlo_module = hlo_module_cache_[module_name];
  dcn_analysis_internal::InstrMetadata instr_metadata;
  auto instr = FindInstruction(*hlo_module, std::string(instr_name));

  instr_metadata.opcode = instr->opcode();
  instr_metadata.channel_id = instr->channel_id().value();
  instr_metadata.rendezvous_name = GetRendezvous(instr);
  instr_metadata.transfer_type = GetTransferType(instr);
  instr_metadata.size = 0;
  if (instr->shape().IsArray()) {
    instr_metadata.size = xla::ShapeUtil::ByteSizeOfElements(instr->shape());
  } else if (instr->shape().IsTuple()) {
    for (const auto& shape : instr->shape().tuple_shapes()) {
      instr_metadata.size += xla::ShapeUtil::ByteSizeOf(shape);
    }
  }
  return instr_metadata;
}

absl::StatusOr<InstrMetadata> DcnTracker::GetInstructionMetadata(
    std::string_view module, std::string_view instr) {
  std::string key = absl::StrCat(module, "_", instr);
  if (const auto& it = instruction_metadata_map_.find(key);
      it != instruction_metadata_map_.end()) {
    return it->second;
  }

  absl::StatusOr<InstrMetadata> instr_metadata =
      GetInstrMetadataFromHloModule(module, instr);
  if (instr_metadata.ok()) {
    instruction_metadata_map_[key] = *instr_metadata;
  }

  return instr_metadata;
}

DcnSlackAnalysis DcnTracker::Finalize() {
  SummarizeDcnSlackAnalysis();
  return slack_analysis_;
}

void DcnTracker::DebugString() {
  for (const DcnSlack& analysis : slack_analysis_.dcn_slack()) {
    LOG(INFO) << analysis.rendezvous() << " : " << analysis.slack_us();
  }
}

void DcnTracker::UpdateActiveOps(uint64_t duration) {
  for (auto& [rendezvous, opState] : rendezvous_to_op_map_) {
    opState.overlapping_duration += duration;
  }
}

int DcnTracker::GetReplicaGroupSize(const std::string& rendezvous_name,
                                    const XEventVisitor& visitor) {
  if (rendezvous_to_replica_group_size_map_.contains(rendezvous_name)) {
    return rendezvous_to_replica_group_size_map_[rendezvous_name];
  }

  DcnCollectiveInfoProto dcn_collective_info =
      GetDcnCollectiveInfoProto(visitor);

  if (dcn_collective_info.one_to_one_groups_size() != 0) {
    // OneToOneGroup has a source and a destination, which is one replica group
    rendezvous_to_replica_group_size_map_[rendezvous_name] = 1;
  } else if (dcn_collective_info.endpoint_groups_size() != 0) {
    rendezvous_to_replica_group_size_map_[rendezvous_name] =
        dcn_collective_info.endpoint_groups(0).endpoints().size();
  } else {
    rendezvous_to_replica_group_size_map_[rendezvous_name] = 0;
  }

  return rendezvous_to_replica_group_size_map_[rendezvous_name];
}

// ComputeTransmittedDataSize is called with the buffer_size for recv-done.
uint64_t DcnTracker::ComputeTransmittedDataSize(
    const int64_t recv_buffer_size, const int group_size,
    const std::string& transfer_type) {
  uint64_t transmitted_bytes = 0;
  if (group_size == 0) {
    LOG(ERROR) << "Replica group size is 0.";
    return transmitted_bytes;
  }

  if (transfer_type == "ONE_TO_ONE") {
    transmitted_bytes = group_size * recv_buffer_size;
  } else if (transfer_type == "ALL_GATHER") {
    transmitted_bytes =
        SafeDivide((group_size - 1) * recv_buffer_size, group_size);
  } else if (transfer_type == "ALL_REDUCE") {
    // Since the reduced buffer now has to be sent back to the replicas,
    // the total bytes transmitted over the network is 2x the shape of the op.
    transmitted_bytes =
        2 * SafeDivide(group_size - 1, group_size) * recv_buffer_size;
  } else if (transfer_type == "ALL_TO_ALL") {
    transmitted_bytes =
        SafeDivide(group_size - 1, group_size) * recv_buffer_size;
  } else if (transfer_type == "REDUCE_SCATTER") {
    transmitted_bytes = recv_buffer_size * (group_size - 1);
  } else {
    LOG(ERROR) << "Unsupported transfer type: " << transfer_type;
  }
  return transmitted_bytes;
}

void DcnTracker::VisitOp(const InstrMetadata& instr,
                         const XEventVisitor& visitor) {
  std::string rendezvous_name;
  if (instr.rendezvous_name.has_value()) {
    rendezvous_name = *instr.rendezvous_name;
    channel_id_to_rendezvous_map_[instr.channel_id] = rendezvous_name;
  } else {
    if (auto it = channel_id_to_rendezvous_map_.find(instr.channel_id);
        it != channel_id_to_rendezvous_map_.end()) {
      rendezvous_name = it->second;
    } else {
      // Ignore ops as we have not seen the corresponding send/recv.
      return;
    }
  }

  DcnOpState& opState = rendezvous_to_op_map_[rendezvous_name];
  opState.stall_duration_ns += visitor.DurationNs();

  switch (instr.opcode) {
    case HloOpcode::kSend:
      opState.start_time = visitor.TimestampNs();
      opState.rendezvous_name = rendezvous_name;
      opState.transfer_type =
          instr.transfer_type.has_value() ? *instr.transfer_type : "";
      opState.overlapping_duration = 0;
      opState.stall_duration_ns = visitor.DurationNs();
      opState.send_op_name = visitor.DisplayName();
      opState.send.set_duration_ps(visitor.DurationPs());
      opState.send.set_start_time_ps(visitor.TimestampPs());
      opState.replica_group_size =
          GetReplicaGroupSize(rendezvous_name, visitor);
      break;
    case HloOpcode::kRecv:
      opState.recv.set_duration_ps(visitor.DurationPs());
      opState.recv.set_start_time_ps(visitor.TimestampPs());
      break;
    case HloOpcode::kSendDone:
      opState.send_done.set_duration_ps(visitor.DurationPs());
      opState.send_done.set_start_time_ps(visitor.TimestampPs());
      break;
    case HloOpcode::kRecvDone: {
      opState.recv_done.set_duration_ps(visitor.DurationPs());
      opState.recv_done.set_start_time_ps(visitor.TimestampPs());
      if (opState.start_time != 0) {
        DcnSlack* analysis = slack_analysis_.add_dcn_slack();
        analysis->set_rendezvous(rendezvous_name);
        analysis->set_transfer_type(opState.transfer_type);
        analysis->set_send_start_time_us(NanoToMicro(opState.start_time));
        analysis->set_recv_done_end_time_us(
            NanoToMicro(visitor.EndTimestampNs()));
        analysis->set_slack_us(NanoToMicro(visitor.TimestampNs() -
                                           opState.start_time -
                                           opState.overlapping_duration));
        analysis->set_bytes_transmitted_over_network(ComputeTransmittedDataSize(
            instr.size, opState.replica_group_size, opState.transfer_type));
        analysis->set_stall_duration_us(NanoToMicro(opState.stall_duration_ns));
        analysis->set_recv_op_name(std::string(visitor.DisplayName()));
        analysis->set_send_op_name(opState.send_op_name);
        *analysis->mutable_send() = opState.send;
        *analysis->mutable_recv() = opState.recv;
        *analysis->mutable_send_done() = opState.send_done;
        *analysis->mutable_recv_done() = opState.recv_done;
      }

      break;
    }
    default:
      LOG(ERROR) << "Received unexpected op";
  }
  UpdateActiveOps(visitor.DurationNs());
}

std::optional<DcnHostEvent> DcnTracker::GetCollectiveHostEvent(
    int core_id, std::string_view rendezvous, Timespan timespan) {
  return core_id_to_host_event_map_[HostCollectiveKey(core_id, rendezvous)].pop(
      timespan);
}

void DcnTracker::SummarizeDcnSlackAnalysis() {
  absl::flat_hash_map<std::string_view, DcnSlackSummary> summary;
  // TODO(b/302596260) : Expand to process all cores.
  int core_id = 0;
  for (DcnSlack& analysis : *slack_analysis_.mutable_dcn_slack()) {
    DcnSlackSummary& s = summary[analysis.rendezvous()];
    s.set_slack_us(s.slack_us() + analysis.slack_us());
    s.set_occurrences(s.occurrences() + 1);
    s.set_rendezvous(analysis.rendezvous());
    s.set_transfer_type(analysis.transfer_type());
    s.set_bytes_transmitted_over_network(
        analysis.bytes_transmitted_over_network());
    s.set_stall_duration_us(s.stall_duration_us() +
                            analysis.stall_duration_us());
    s.set_observed_duration_us(s.observed_duration_us() +
                               analysis.recv_done_end_time_us() -
                               analysis.send_start_time_us());
    s.set_recv_op_name(analysis.recv_op_name());
    s.set_send_op_name(analysis.send_op_name());
    s.set_send_duration_us(s.send_duration_us() +
                           PicoToMicro(analysis.send().duration_ps()));
    s.set_recv_duration_us(s.recv_duration_us() +
                           PicoToMicro(analysis.recv().duration_ps()) / 1E6);
    s.set_send_done_duration_us(
        s.send_done_duration_us() +
        PicoToMicro(analysis.send_done().duration_ps()));
    s.set_recv_done_duration_us(
        s.recv_done_duration_us() +
        PicoToMicro(analysis.recv_done().duration_ps()));

    // Populate Host summary to DcnSlackSummary
    std::optional<DcnHostEvent> host_event = GetCollectiveHostEvent(
        core_id, analysis.rendezvous(),
        Timespan::FromEndPoints(analysis.send().start_time_ps(),
                                analysis.recv_done().start_time_ps() +
                                    analysis.recv_done().duration_ps()));
    if (host_event.has_value()) {
      OpInstance* host_graph_execution =
          analysis.mutable_host_graph_execution();
      host_graph_execution->set_start_time_ps(host_event->timespan.begin_ps());
      host_graph_execution->set_duration_ps(host_event->timespan.duration_ps());
      s.set_host_stall_us(s.host_stall_us() +
                          (((int64_t)host_event->timespan.end_ps() -
                            (int64_t)analysis.recv_done().start_time_ps()) /
                           1E6));
      s.set_host_events_count(s.host_events_count() + 1);
    }
  }

  for (auto& [_, s] : summary) {
    s.set_slack_us(SafeDivide(s.slack_us(), s.occurrences()));
    s.set_stall_duration_us(SafeDivide(s.stall_duration_us(), s.occurrences()));
    s.set_observed_duration_us(
        SafeDivide(s.observed_duration_us(), s.occurrences()));
    s.set_send_done_duration_us(
        SafeDivide(s.send_done_duration_us(), s.occurrences()));
    s.set_recv_done_duration_us(
        SafeDivide(s.recv_done_duration_us(), s.occurrences()));
    s.set_send_duration_us(SafeDivide(s.send_duration_us(), s.occurrences()));
    s.set_recv_duration_us(SafeDivide(s.recv_duration_us(), s.occurrences()));
    s.set_host_stall_us(SafeDivide(s.host_stall_us(), s.host_events_count()));
    *slack_analysis_.add_dcn_slack_summary() = s;
  }
}

void DcnTracker::ProcessTopology(const Topology& topology) {
  for (const auto& mesh_location : topology.mesh_location()) {
    global_chip_id_to_local_index_map_[mesh_location.global_id()] =
        mesh_location.index_on_host();
  }
}

int DcnTracker::GetLocalIndex(int dcn_device_id) {
  /* Based on if megacore was present or not, the LocalIndex calculation will
   * differ,
   * dcn device id would use the global index in cases of megacore, and use
   * 2*global_index (+1) for non megacore instances
   * TODO(b/302145703): Identify if transformation can be obtained from the
   * TpuTopology directly
   */
  int global_device_id = dcn_device_id;
  if (!is_megacore_) {
    if (global_chip_id_to_local_index_map_.contains(global_device_id)) {
      return global_chip_id_to_local_index_map_[dcn_device_id / 2] +
             dcn_device_id % 2;
    }
  }
  if (global_chip_id_to_local_index_map_.contains(global_device_id)) {
    return global_chip_id_to_local_index_map_[global_device_id];
  }
  LOG(WARNING) << "Could not map dcn_device_id to Local index, Using "
                  "dcn_device_id : "
               << global_device_id;
  return global_device_id;
}

void DcnTracker::VisitHostEvent(const DcnHostEvent& event) {
  std::string key = HostCollectiveKey(
      GetLocalIndex(event.multi_slice_device_id), event.rendezvous_name);
  if (event.rendezvous_name.empty()) return;
  core_id_to_host_event_map_[key].insert(event);
}

void ProcessDcnTraces(const XPlane& xplane, DcnTracker& dcn_tracker) {
  XPlaneVisitor xplane_visitor = CreateTfXPlaneVisitor(&xplane);
  HloProtoMap hlo_proto_map;
  xplane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      dcn_tracker.VisitHostEvent(ParseDcnHostEvent(event));
    });
  });
}

}  // namespace dcn_analysis_internal

DcnSlackAnalysis ConvertXSpaceToDcnSlackAnalysis(const XSpace& xspace,
                                                 const XPlane* dcn_host_plane,
                                                 const Topology* topology,
                                                 bool is_megacore) {
  int num_cores = tsl::profiler::FindTensorCorePlanes(xspace).size();
  if (num_cores == 0) return DcnSlackAnalysis();
  const XPlane* xplane =
      FindPlaneWithName(xspace, tsl::profiler::TpuPlaneName(0));
  XPlaneVisitor xplane_visitor = CreateTfXPlaneVisitor(xplane);
  HloProtoMap hlo_proto_map;
  hlo_proto_map.AddHloProtosFromXSpace(xspace);
  dcn_analysis_internal::DcnTracker dcn_tracker(hlo_proto_map, is_megacore);
  XEventContextTracker hlo_module_context(
      &xplane_visitor,
      FindLineWithName(*xplane, tsl::profiler::kXlaModuleLineName));
  xplane_visitor.ForEachLine([&](const XLineVisitor& xline) {
    if (xline.Name() == kXlaOpLineName) {
      xline.ForEachEvent([&](const XEventVisitor& xevent) {
        std::string_view hlo_category;

        xevent.Metadata().ForEachStat([&](const XStatVisitor& xstat) {
          switch (static_cast<StatType>(*xstat.Type())) {
            case StatType::kHloCategory:
              hlo_category = xstat.StrOrRefValue();
              break;
            default:
              break;
          }
        });
        auto module =
            hlo_module_context.GetContainingEvent(xevent.GetTimespan());
        if (!module.has_value()) return;
        if (absl::StrContains(hlo_category, "host send") ||
            absl::StrContains(hlo_category, "host recv")) {
          // All Dcn send/send-done/recv/recv-done ops.
          auto instr = dcn_tracker.GetInstructionMetadata(module->Name(),
                                                          xevent.DisplayName());
          if (instr.ok()) {
            dcn_tracker.VisitOp(*instr, xevent);
          }
        }
      });
    }
  });

  if (dcn_host_plane != nullptr) {
    VLOG(1) << "Processing host traces.";
    if (topology != nullptr) {
      dcn_tracker.ProcessTopology(*topology);
    }
    ProcessDcnTraces(*dcn_host_plane, dcn_tracker);
  }
  return dcn_tracker.Finalize();
}

}  // namespace profiler
}  // namespace tensorflow
