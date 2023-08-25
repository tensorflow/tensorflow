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
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/side_effect_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"
#include "tensorflow/core/profiler/utils/hlo_module_utils.h"
#include "tensorflow/core/profiler/utils/hlo_proto_map.h"
#include "tensorflow/core/profiler/utils/hlo_proto_to_module.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/math_utils.h"
#include "tensorflow/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/tsl/profiler/utils/xplane_utils.h"
#include "tensorflow/tsl/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

using tensorflow::profiler::DcnSlackSummary;
using tsl::profiler::CreateTfXPlaneVisitor;
using tsl::profiler::FindLineWithName;
using tsl::profiler::FindPlanesWithPrefix;
using tsl::profiler::kTpuPlanePrefix;
using tsl::profiler::kXlaOpLineName;
using tsl::profiler::NanoToMicro;
using tsl::profiler::SafeDivide;
using tsl::profiler::StatType;
using tsl::profiler::XEventContextTracker;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;
using xla::HloOpcode;

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

}  // namespace

namespace dcn_analysis_internal {

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
      opState.overlapping_duration = 0;
      opState.stall_duration_ns = visitor.DurationNs();
      opState.send_op_name = visitor.DisplayName();
      break;
    case HloOpcode::kRecv:
    case HloOpcode::kSendDone:
      break;
    case HloOpcode::kRecvDone: {
      if (opState.start_time != 0) {
        DcnSlack* analysis = slack_analysis_.add_dcn_slack();
        analysis->set_rendezvous(rendezvous_name);
        analysis->set_send_start_time_us(NanoToMicro(opState.start_time));
        analysis->set_recv_done_end_time_us(
            NanoToMicro(visitor.EndTimestampNs()));
        analysis->set_slack_us(NanoToMicro(visitor.TimestampNs() -
                                           opState.start_time -
                                           opState.overlapping_duration));
        // TODO(b/294584919): The current transmitted bytes measures the
        // buffer size at the recv-done. This could include bytes that were not
        // received over the network. Fix the calculation to improve accuracy.
        analysis->set_bytes_transmitted_over_network(instr.size);
        analysis->set_stall_duration_us(NanoToMicro(opState.stall_duration_ns));
        analysis->set_recv_op_name(std::string(visitor.DisplayName()));
        analysis->set_send_op_name(opState.send_op_name);
      }

      break;
    }
    default:
      LOG(ERROR) << "Received unexpected op";
  }
  UpdateActiveOps(visitor.DurationNs());
}

void DcnTracker::SummarizeDcnSlackAnalysis() {
  absl::flat_hash_map<std::string_view, DcnSlackSummary> summary;
  for (const DcnSlack& analysis : slack_analysis_.dcn_slack()) {
    DcnSlackSummary& s = summary[analysis.rendezvous()];
    s.set_slack_us(s.slack_us() + analysis.slack_us());
    s.set_occurrences(s.occurrences() + 1);
    s.set_rendezvous(analysis.rendezvous());
    s.set_bytes_transmitted_over_network(
        analysis.bytes_transmitted_over_network());
    s.set_stall_duration_us(s.stall_duration_us() +
                            analysis.stall_duration_us());
    s.set_observed_duration_us(s.observed_duration_us() +
                               analysis.recv_done_end_time_us() -
                               analysis.send_start_time_us());
    s.set_recv_op_name(analysis.recv_op_name());
    s.set_send_op_name(analysis.send_op_name());
  }

  for (auto& [_, s] : summary) {
    s.set_slack_us(SafeDivide(s.slack_us(), s.occurrences()));
    s.set_stall_duration_us(SafeDivide(s.stall_duration_us(), s.occurrences()));
    s.set_observed_duration_us(
        SafeDivide(s.observed_duration_us(), s.occurrences()));
    *slack_analysis_.add_dcn_slack_summary() = s;
  }
}

}  // namespace dcn_analysis_internal

DcnSlackAnalysis ConvertXSpaceToDcnSlackAnalysis(const XSpace& xspace) {
  const auto& xplanes = FindPlanesWithPrefix(xspace, kTpuPlanePrefix);
  if (xplanes.empty()) return DcnSlackAnalysis();
  const XPlane* xplane = xplanes.at(0);
  XPlaneVisitor xplane_visitor = CreateTfXPlaneVisitor(xplane);
  HloProtoMap hlo_proto_map;
  hlo_proto_map.AddHloProtosFromXSpace(xspace);
  dcn_analysis_internal::DcnTracker dcn_tracker(hlo_proto_map);
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
          // All Megascale send/send-done/recv/recv-done ops.
          auto instr = dcn_tracker.GetInstructionMetadata(module->Name(),
                                                          xevent.DisplayName());
          if (instr.ok()) {
            dcn_tracker.VisitOp(*instr, xevent);
          }
        }
      });
    }
  });
  return dcn_tracker.Finalize();
}

}  // namespace profiler
}  // namespace tensorflow
