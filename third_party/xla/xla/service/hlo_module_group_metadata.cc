/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/hlo_module_group_metadata.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/tuple_points_to_analysis.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {

std::string HloModuleGroupMetadata::TrackedInstruction::ToString() const {
  std::string repr =
      (instruction_ != nullptr) ? instruction_->ToShortString() : "NULL";
  switch (kind_) {
    case ComputationKind::kInvalid:
      repr += ":INVALID";
      break;
    case ComputationKind::kWhileCondition:
      repr += ":WHILE_CONDITION";
      break;
    case ComputationKind::kWhileBody:
      repr += ":WHILE_BODY";
      break;
    case ComputationKind::kConditionalBranch:
      repr += absl::StrCat(":CONDITIONAL_BRANCH_", index_);
      break;
    case ComputationKind::kCallFunction:
      repr += ":CALL";
      break;
  }
  return repr;
}

/* static */ absl::StatusOr<std::unique_ptr<HloModuleGroupMetadata>>
HloModuleGroupMetadata::Build(absl::Span<HloModule* const> modules) {
  auto metadata = std::make_unique<HloModuleGroupMetadata>(modules);
  TF_RETURN_IF_ERROR(metadata->Build());
  return std::move(metadata);
}

Status HloModuleGroupMetadata::Build() {
  TF_RETURN_IF_ERROR(RecordInstructions());
  TF_RETURN_IF_ERROR(VerifyChannelInstructions());

  // Record all companion while instructions.
  const auto visitor = [this](HloInstruction* hlo) -> Status {
    // We only need to process if the instruction is within the computation
    // of a companion instruction, like in the condition or body computation
    // of a While.
    const TrackedInstruction* tracked = GetTrackedInstruction(hlo->parent());
    if (tracked == nullptr) {
      return OkStatus();
    }

    if (IsChannelInstruction(hlo) || IsNonSpmdCrossModuleAllReduce(hlo)) {
      std::vector<HloComputation*> peers;
      if (IsChannelInstruction(hlo)) {
        peers.push_back(PeerComputation(hlo));
      } else if (IsNonSpmdCrossModuleAllReduce(hlo)) {
        for (HloInstruction* instr : GetAllReduceGroup(*hlo->channel_id())) {
          if (instr == hlo) {
            continue;
          }
          peers.push_back(instr->parent());
        }
      }

      // Add the parent computation of this channel (or all-reduce) instruction
      // and its peer computation(s) (both must be while computations) as
      // companions.
      for (HloComputation* peer_computation : peers) {
        const TrackedInstruction* peer_tracked =
            GetTrackedInstruction(peer_computation);
        if (peer_tracked == nullptr) {
          continue;
        }
        TF_RET_CHECK(*tracked == *peer_tracked)
            << "Peer instruction does not match the computation kind";
        TF_RETURN_IF_ERROR(
            AddCompanion(tracked->instruction(), peer_tracked->instruction()));
        tracked_instructions_comms_[tracked->instruction()].push_back(hlo);
      }
    } else if (IsCompanionInstruction(hlo)) {
      // Add the parents of companion instructions (they must be all of the same
      // kind of instructions, opcode wise) as companions.
      for (HloInstruction* companion : Companions(hlo)) {
        const TrackedInstruction* companion_tracked =
            GetTrackedInstruction(companion->parent());
        TF_RET_CHECK(companion_tracked != nullptr);
        TF_RET_CHECK(*tracked == *companion_tracked);
        TF_RETURN_IF_ERROR(AddCompanion(tracked->instruction(),
                                        companion_tracked->instruction()));
      }
    }

    return OkStatus();
  };

  // Visit the computations in postorder so that the companion information grows
  // from inner computations to outer ones.
  for (HloModule* module : modules_) {
    FunctionVisitor function_visitor(visitor);
    for (HloComputation* computation : module->MakeComputationPostOrder()) {
      TF_RETURN_IF_ERROR(computation->Accept(&function_visitor));
    }
  }

  // While building the companion sets, initial sets may be removed by inserting
  // nullptr in companion_sets_. Prune those removed sets to compact.
  std::vector<std::unique_ptr<std::vector<HloInstruction*>>> sets;
  for (int64_t i = 0; i < companion_sets_.size(); ++i) {
    if (companion_sets_[i] == nullptr) {
      continue;
    }
    sets.push_back(std::move(companion_sets_[i]));
    for (HloInstruction* hlo : *sets.back()) {
      companion_set_index_[hlo] = sets.size() - 1;
    }
  }
  companion_sets_ = std::move(sets);

  TF_RETURN_IF_ERROR(VerifyCompanionSets());
  if (VLOG_IS_ON(4)) {
    DumpCollectedStats();
  }

  for (HloModule* module : modules_) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                        HloAliasAnalysis::Run(module));
    alias_analyses_[module] = std::move(alias_analysis);
  }

  return OkStatus();
}

Status HloModuleGroupMetadata::VerifyCompanionSets() const {
  for (const auto& companions : companion_sets_) {
    // A companion set must be composed at most of an instruction per
    // device/module.
    absl::flat_hash_set<int64_t> devices;
    for (HloInstruction* instruction : *companions) {
      // Go through all the communicating instructions (send, recv) of the given
      // companion, and record their device.
      auto it = tracked_instructions_comms_.find(instruction);
      if (it == tracked_instructions_comms_.end()) {
        // Companions can be added even if they have no communicating
        // instructions, if they are parent of companions.
        continue;
      }
      absl::flat_hash_set<int64_t> comm_devices;
      for (HloInstruction* comm_instruction : it->second) {
        auto device = GetInstructionDevice(*comm_instruction);
        TF_RET_CHECK(device) << "Instruction " << comm_instruction->ToString()
                             << " does not have a device";
        comm_devices.insert(*device);
      }
      for (int64_t device : comm_devices) {
        if (!devices.insert(device).second) {
          std::stringstream ss;
          ss << "Companion set:" << std::endl;
          for (HloInstruction* hlo : *companions) {
            ss << "  " << hlo->name() << std::endl;
          }
          ss << "has multiple instructions on the same device";
          return FailedPrecondition("%s", ss.str());
        }
      }
    }
  }
  return OkStatus();
}

bool HloModuleGroupMetadata::IsChannelInstruction(
    const HloInstruction* instruction) const {
  switch (instruction->opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone: {
      const HloSendRecvInstruction* send_recv_instr =
          DynCast<HloSendRecvInstruction>(instruction);
      CHECK(send_recv_instr != nullptr);
      return !send_recv_instr->is_host_transfer();
    }
    default:
      return false;
  }
}

bool HloModuleGroupMetadata::IsCompanionInstruction(HloInstruction* hlo) const {
  return companion_set_index_.contains(hlo);
}

bool HloModuleGroupMetadata::IsNonSpmdCrossModuleAllReduce(
    HloInstruction* hlo) const {
  return hlo->IsCrossModuleAllReduce() &&
         !hlo->GetModule()->config().use_spmd_partitioning();
}

bool HloModuleGroupMetadata::InstructionCommunicates(
    HloInstruction* hlo) const {
  return IsChannelInstruction(hlo) || IsCompanionInstruction(hlo) ||
         IsNonSpmdCrossModuleAllReduce(hlo);
}

const HloModuleGroupMetadata::Channel& HloModuleGroupMetadata::GetChannel(
    int64_t channel_id) const {
  CHECK(channel_id_map_.find(channel_id) != channel_id_map_.end());
  return channels_[channel_id_map_.at(channel_id)];
}

bool HloModuleGroupMetadata::HasChannel(int64_t channel_id) const {
  return channel_id_map_.find(channel_id) != channel_id_map_.end();
}

HloComputation* HloModuleGroupMetadata::PeerComputation(
    const HloInstruction* instruction) const {
  CHECK(IsChannelInstruction(instruction));
  const Channel& channel = GetChannel(*instruction->channel_id());
  switch (instruction->opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
      return channel.recv->parent();
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
      return channel.send->parent();
    default:
      LOG(FATAL) << "opcode not supported";
  }
}

const std::vector<HloInstruction*>& HloModuleGroupMetadata::GetAllReduceGroup(
    int64_t channel_id) const {
  auto it = all_reduce_map_.find(channel_id);
  CHECK(it != all_reduce_map_.end());
  return it->second;
}

std::vector<HloModuleGroupMetadata::TrackedInstruction>
HloModuleGroupMetadata::GetCompanionsPath(const HloInstruction* hlo) const {
  std::vector<TrackedInstruction> path;
  const HloComputation* parent = hlo->parent();
  const TrackedInstruction* companion;
  while ((companion = GetTrackedInstruction(parent)) != nullptr) {
    parent = companion->instruction()->parent();
    path.push_back(*companion);
  }
  return path;
}

bool HloModuleGroupMetadata::CheckCompanionPathsCompatibility(
    const std::vector<TrackedInstruction>& path0,
    const std::vector<TrackedInstruction>& path1) const {
  if (path0.size() != path1.size()) {
    VLOG(5) << "Companion path size do not match: " << path0.size()
            << " != " << path1.size();
    return false;
  }
  for (int64_t i = 0; i < path0.size(); ++i) {
    if (path0[i] != path1[i]) {
      VLOG(5) << "Companion instructions at path index " << i
              << " do not have the same opcode: " << path0[i].ToString()
              << " vs " << path1[i].ToString();
      return false;
    }
  }
  return true;
}

int64_t HloModuleGroupMetadata::GetModuleId(const HloModule* module) const {
  for (int64_t i = 0; i < modules_.size(); ++i) {
    if (modules_[i] == module) {
      return i;
    }
  }
  LOG(FATAL) << "unknown module";
}

std::optional<int64_t> HloModuleGroupMetadata::GetInstructionDevice(
    const HloInstruction& instruction) const {
  // The module group metadata can be created in both "single module, multiple
  // devices" and "multiple modules, no explicit devices" fashions.
  // The API returns an optional even though the current implementation always
  // returns a device, to account for cases where we cannot guess a device.
  // In such cases the VerifyChannelInstructions() will return proper errors.
  std::optional<int64_t> device = instruction.sharding_unique_device();
  if (!device) {
    device = GetModuleId(instruction.GetModule());
  }
  return device;
}

int64_t HloModuleGroupMetadata::GetDeviceModulesCount() const {
  return modules_.size();
}

Status HloModuleGroupMetadata::RecordInstructions() {
  const auto visitor = [this](HloInstruction* hlo) -> Status {
    if (hlo->opcode() == HloOpcode::kWhile) {
      tracked_instructions_[hlo->while_condition()] =
          TrackedInstruction(hlo, ComputationKind::kWhileCondition);
      tracked_instructions_[hlo->while_body()] =
          TrackedInstruction(hlo, ComputationKind::kWhileBody);
    } else if (hlo->opcode() == HloOpcode::kConditional) {
      for (int b = 0; b < hlo->branch_count(); ++b) {
        tracked_instructions_[hlo->branch_computation(b)] =
            TrackedInstruction(hlo, ComputationKind::kConditionalBranch, b);
      }
    } else if (hlo->opcode() == HloOpcode::kCall) {
      tracked_instructions_[hlo->to_apply()] =
          TrackedInstruction(hlo, ComputationKind::kCallFunction);
    }

    // Group cross module all-reduce instructions by the channel id.
    if (IsNonSpmdCrossModuleAllReduce(hlo)) {
      TF_RET_CHECK(channel_id_map_.find(*hlo->channel_id()) ==
                   channel_id_map_.end())
          << "channel_id " << *hlo->channel_id()
          << " is already used by a send/recv instruction";
      all_reduce_map_[*hlo->channel_id()].push_back(hlo);
      max_channel_id_ = std::max(max_channel_id_, *hlo->channel_id());
      return OkStatus();
    }

    if (!IsChannelInstruction(hlo)) {
      return OkStatus();
    }

    TF_RET_CHECK(all_reduce_map_.find(*hlo->channel_id()) ==
                 all_reduce_map_.end())
        << "channel id " << *hlo->channel_id()
        << " is already used by an all-reduce instruction";

    // Add a new channel if needed.
    if (channel_id_map_.find(*hlo->channel_id()) == channel_id_map_.end()) {
      channels_.emplace_back();
      channels_.back().id = *hlo->channel_id();
      channel_id_map_[*hlo->channel_id()] = channels_.size() - 1;
      max_channel_id_ = std::max(max_channel_id_, *hlo->channel_id());
    }
    Channel& channel = channels_[channel_id_map_[*hlo->channel_id()]];

    if (hlo->opcode() == HloOpcode::kSend) {
      TF_RET_CHECK(channel.send == nullptr)
          << "channel id " << *hlo->channel_id()
          << " is used by multiple send instructions";
      channel.send = hlo;
    }
    if (hlo->opcode() == HloOpcode::kRecv) {
      TF_RET_CHECK(channel.recv == nullptr)
          << "channel id " << *hlo->channel_id()
          << " is used by multiple recv instructions";
      channel.recv = hlo;
    }
    if (hlo->opcode() == HloOpcode::kSendDone) {
      TF_RET_CHECK(channel.send_done == nullptr)
          << "channel id " << *hlo->channel_id()
          << " is used by multiple send-done instructions";
      channel.send_done = hlo;
    }
    if (hlo->opcode() == HloOpcode::kRecvDone) {
      TF_RET_CHECK(channel.recv_done == nullptr)
          << "channel id " << *hlo->channel_id()
          << " is used by multiple recv-done instructions";
      channel.recv_done = hlo;
    }
    return OkStatus();
  };

  for (HloModule* module : modules_) {
    FunctionVisitor function_visitor(visitor);
    for (auto* computation : module->computations()) {
      TF_RETURN_IF_ERROR(computation->Accept(&function_visitor));
    }
  }
  VLOG(2) << "Created " << channels_.size() << " channels";
  VLOG(2) << "Created " << all_reduce_map_.size() << " all-reduce groups";
  return OkStatus();
}

Status HloModuleGroupMetadata::AddCompanion(HloInstruction* instruction1,
                                            HloInstruction* instruction2) {
  TF_RET_CHECK(instruction1->opcode() == HloOpcode::kWhile ||
               instruction1->opcode() == HloOpcode::kConditional ||
               instruction1->opcode() == HloOpcode::kCall);
  VLOG(2) << "adding as companions:" << instruction1->ToString() << " and "
          << instruction2->ToString();
  if (instruction1 == instruction2) {
    return OkStatus();
  } else if (!ContainsKey(companion_set_index_, instruction1) &&
             !ContainsKey(companion_set_index_, instruction2)) {
    companion_sets_.push_back(std::make_unique<std::vector<HloInstruction*>>());
    auto companion_set = companion_sets_.back().get();
    companion_set->push_back(instruction1);
    companion_set->push_back(instruction2);
    companion_set_index_[instruction1] = companion_sets_.size() - 1;
    companion_set_index_[instruction2] = companion_sets_.size() - 1;
  } else if (!ContainsKey(companion_set_index_, instruction1)) {
    companion_sets_[companion_set_index_[instruction2]]->push_back(
        instruction1);
    companion_set_index_[instruction1] = companion_set_index_[instruction2];
  } else if (!ContainsKey(companion_set_index_, instruction2)) {
    companion_sets_[companion_set_index_[instruction1]]->push_back(
        instruction2);
    companion_set_index_[instruction2] = companion_set_index_[instruction1];
  } else if (companion_set_index_[instruction1] !=
             companion_set_index_[instruction2]) {
    // At any point while building the companion sets, each instruction belongs
    // to at most 1 companion set, so the union of two companion sets is
    // concatenating two disjoint sets.
    absl::c_copy(Companions(instruction2),
                 std::back_inserter(
                     *companion_sets_[companion_set_index_[instruction1]]));
    int64_t index_to_remove = companion_set_index_[instruction2];
    for (HloInstruction* hlo : Companions(instruction2)) {
      companion_set_index_[hlo] = companion_set_index_[instruction1];
    }
    // We can't remove the set from the vector because companion_set_index_
    // references sets by their index in this vector, so we reset to nullptr
    // instead.
    companion_sets_[index_to_remove].reset(nullptr);
  }
  return OkStatus();
}

Status HloModuleGroupMetadata::VerifyChannelInstructions() {
  for (const Channel& channel : channels_) {
    if (channel.send == nullptr) {
      return FailedPrecondition("missing send for id : %d", channel.id);
    }
    if (channel.recv == nullptr) {
      return FailedPrecondition("missing recv for id : %d", channel.id);
    }
    if (channel.send_done == nullptr) {
      return FailedPrecondition("missing send-done for id : %d", channel.id);
    }
    if (channel.recv_done == nullptr) {
      return FailedPrecondition("missing recv-done for id : %d", channel.id);
    }
  }

  // Check if the shapes match for each channel.
  for (const Channel& channel : channels_) {
    const Shape& send_shape = channel.send->operand(0)->shape();
    const Shape& recv_shape =
        ShapeUtil::GetTupleElementShape(channel.recv_done->shape(), 0);
    if (!ShapeUtil::Compatible(send_shape, recv_shape)) {
      return FailedPrecondition("send/recv shapes do not match");
    }
    auto send_device = GetInstructionDevice(*channel.send);
    auto send_done_device = GetInstructionDevice(*channel.send_done);
    if (!send_device) {
      return FailedPrecondition("send instruction must have a device: %s",
                                channel.send->ToString());
    }
    if (!send_done_device) {
      return FailedPrecondition("send_done instruction must have a device: %s",
                                channel.send_done->ToString());
    }
    if (*send_device != *send_done_device) {
      return FailedPrecondition(
          "send and send-done (channel=%d) must be on the same device: %d "
          "vs. %d",
          channel.id, *send_device, *send_done_device);
    }
    auto recv_device = GetInstructionDevice(*channel.recv);
    auto recv_done_device = GetInstructionDevice(*channel.recv_done);
    if (!recv_done_device) {
      return FailedPrecondition("recv_done instruction must have a device: %s",
                                channel.recv_done->ToString());
    }
    if (*recv_device != *recv_done_device) {
      return FailedPrecondition(
          "recv and recv-done (channel=%d) must be on the same device: %d "
          "vs. %d",
          channel.id, *recv_device, *recv_done_device);
    }
    if (*send_device == *recv_device) {
      return FailedPrecondition(
          "send and recv (channel=%d) must be on different devices: %d",
          channel.id, *send_device);
    }
  }

  for (const Channel& channel : channels_) {
    TF_RETURN_IF_ERROR(CheckCommunicatingInstruction(channel.send));
    TF_RETURN_IF_ERROR(CheckCommunicatingInstruction(channel.send_done));
    TF_RETURN_IF_ERROR(CheckCommunicatingInstruction(channel.recv));
    TF_RETURN_IF_ERROR(CheckCommunicatingInstruction(channel.recv_done));
  }
  // Check if the nest levels match for each channel.
  for (const Channel& channel : channels_) {
    std::vector<TrackedInstruction> path = GetCompanionsPath(channel.send);
    if (!CheckCompanionPathsCompatibility(
            path, GetCompanionsPath(channel.send_done)) ||
        !CheckCompanionPathsCompatibility(path,
                                          GetCompanionsPath(channel.recv)) ||
        !CheckCompanionPathsCompatibility(
            path, GetCompanionsPath(channel.recv_done))) {
      return FailedPrecondition(
          "Nest companion paths do not match for channel %d", channel.id);
    }
  }
  return OkStatus();
}

Status HloModuleGroupMetadata::CheckCommunicatingInstruction(
    HloInstruction* instruction) const {
  HloComputation* computation = instruction->parent();
  const HloModule* module = computation->parent();
  if (module->entry_computation() == computation ||
      tracked_instructions_.contains(computation)) {
    return OkStatus();
  }
  return FailedPrecondition("channel is used in disallowed computation");
}

void HloModuleGroupMetadata::DumpCollectedStats() const {
  std::map<std::pair<int64_t, int64_t>, int64_t> communication_histogram;
  for (auto& channel : channels_) {
    auto from_device = GetInstructionDevice(*channel.send);
    auto to_device = GetInstructionDevice(*channel.recv);
    LOG(INFO) << "Channel " << channel.id << ": from_device=" << *from_device
              << " to_device=" << *to_device << " send=" << channel.send->name()
              << " send_done=" << channel.send_done->name()
              << " recv=" << channel.recv->name()
              << " recv_done=" << channel.recv_done->name();
    communication_histogram[std::pair<int64_t, int64_t>(*from_device,
                                                        *to_device)] += 1;
  }
  for (auto& fromto_count : communication_histogram) {
    LOG(INFO) << "From " << fromto_count.first.first << " to "
              << fromto_count.first.second << ": " << fromto_count.second;
  }
  for (auto& companion_set : companion_sets_) {
    LOG(INFO) << "Companion set:";
    for (HloInstruction* instruction : *companion_set) {
      LOG(INFO) << "  " << instruction->name();
    }
  }
  for (auto& instruction_comm : tracked_instructions_comms_) {
    LOG(INFO) << "Communicating instruction " << instruction_comm.first->name();
    for (HloInstruction* instruction : instruction_comm.second) {
      auto device = GetInstructionDevice(*instruction);
      LOG(INFO) << "  " << instruction->name() << " on device " << *device;
    }
  }
}

}  // namespace xla
