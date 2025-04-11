/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/model/collective_interpolator.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/model/collective_interpolator_data.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/gpu/model/interpolator.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

struct InterpolationSpecification {
  HloOpcode opcode;
  GPUCommunicationType comm;
  int64_t num_devices;
  int64_t transfer_size;
};

// Returns number of participating devices in an input `device_list`. Supports
// only `iota_replica_group_list`.
absl::StatusOr<int> GetNumParticipatingDevices(
    const CollectiveDeviceList& device_list) {
  auto iota = device_list.iota_replica_group_list();
  if (!iota.has_value()) {
    return absl::FailedPreconditionError(
        "Only iota device assignment is supported.");
  }
  return iota->num_devices_per_group();
}

absl::StatusOr<InterpolationSpecification> Spec(
    const HloInstructionProfile& profile,
    const se::DeviceDescription& device_info) {
  auto module = CollectiveInterpolator::ConstructModule(profile);
  if (module == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot construct module from profile: ", profile.DebugString()));
  }
  auto instr = module->entry_computation()->root_instruction();
  auto collective = Cast<HloCollectiveInstruction>(instr);

  GpuHloCostAnalysis analysis(GpuHloCostAnalysis::Options(), device_info);
  TF_RETURN_IF_ERROR(collective->Accept(&analysis));
  int64_t bytes_transferred = analysis.BytesTransferred(*collective);

  TF_ASSIGN_OR_RETURN(
      auto comm,
      CommunicationType(*collective, device_info.gpu_compute_capability()));
  TF_ASSIGN_OR_RETURN(int num_devices,
                      GetNumParticipatingDevices(collective->device_list()));

  return InterpolationSpecification{
      /*opcode=*/collective->opcode(),
      /*comm=*/comm,
      /*num_devices=*/num_devices,
      /*transfer_size=*/bytes_transferred,
  };
}

std::unique_ptr<HloModule> AllReduceModule(
    const HloInstructionProfile& profile) {
  HloModuleConfig config;
  auto module = std::make_unique<HloModule>("m", config);
  Shape shape(profile.instruction().shape());

  HloComputation::Builder wrapped_computation("wrapped_computation");
  HloComputation::Builder entry_builder("entry");
  Shape s(shape.element_type(), {}, {});
  HloInstruction* a = wrapped_computation.AddInstruction(
      HloInstruction::CreateParameter(0, s, "p0.1"));
  HloInstruction* b = wrapped_computation.AddInstruction(
      HloInstruction::CreateParameter(1, s, "p0.2"));
  // We are just assuming some wrapped computation, we do not save this
  // information with performance tables for now so it does not matter what
  // to_apply op we construct here.
  wrapped_computation.AddInstruction(
      HloInstruction::CreateBinary(s, HloOpcode::kAdd, a, b));

  CollectiveDeviceList collective_device_list(
      IotaReplicaGroupList::FromProto(profile.instruction()
                                          .collective_device_list()
                                          .iota_replica_group_list()));

  HloComputation* subcomp =
      module->AddEmbeddedComputation(wrapped_computation.Build());
  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateAllReduce(
      shape, {p0}, subcomp, collective_device_list,
      profile.instruction().constrain_layout(),
      profile.instruction().channel_id(),
      profile.instruction().use_global_device_ids()));
  module->AddEntryComputation(entry_builder.Build());
  return module;
}

std::unique_ptr<HloModule> ReduceScatterModule(
    const HloInstructionProfile& profile) {
  HloModuleConfig config;
  auto module = std::make_unique<HloModule>("m", config);
  Shape shape(profile.instruction().shape());

  HloComputation::Builder wrapped_computation("wrapped_computation");
  HloComputation::Builder entry_builder("entry");
  Shape s(shape.element_type(), {}, {});
  HloInstruction* a = wrapped_computation.AddInstruction(
      HloInstruction::CreateParameter(0, s, "p0.1"));
  HloInstruction* b = wrapped_computation.AddInstruction(
      HloInstruction::CreateParameter(1, s, "p0.2"));
  // We are just assuming some wrapped computation, we do not save this
  // information with performance tables for now so it does not matter what
  // to_apply op we construct here.
  wrapped_computation.AddInstruction(
      HloInstruction::CreateBinary(s, HloOpcode::kAdd, a, b));

  CollectiveDeviceList collective_device_list(
      IotaReplicaGroupList::FromProto(profile.instruction()
                                          .collective_device_list()
                                          .iota_replica_group_list()));

  HloComputation* subcomp =
      module->AddEmbeddedComputation(wrapped_computation.Build());
  if (shape.dimensions().size() != 1) {
    VLOG(1) << "Unsupported number of dimensions: " << profile.DebugString();
    return nullptr;
  }
  std::vector<int64_t> new_dims(shape.dimensions().begin(),
                                shape.dimensions().end());
  auto num_participating_devices =
      GetNumParticipatingDevices(collective_device_list);
  if (!num_participating_devices.ok()) {
    VLOG(1) << "Cannot get num participating devices: "
            << profile.DebugString();
    return nullptr;
  }
  new_dims[0] = new_dims.front() * *num_participating_devices;
  Shape p_shape = ShapeUtil::MakeShape(shape.element_type(), new_dims);
  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, p_shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateReduceScatter(
      shape, {p0}, subcomp, collective_device_list,
      profile.instruction().constrain_layout(),
      profile.instruction().channel_id(),
      profile.instruction().use_global_device_ids(),
      /*scatter_dimension=*/0));
  module->AddEntryComputation(entry_builder.Build());
  return module;
}

std::unique_ptr<HloModule> AllGatherModule(
    const HloInstructionProfile& profile) {
  HloModuleConfig config;
  auto module = std::make_unique<HloModule>("m", config);
  Shape shape(profile.instruction().shape());

  HloComputation::Builder entry_builder("entry");

  CollectiveDeviceList collective_device_list(
      IotaReplicaGroupList::FromProto(profile.instruction()
                                          .collective_device_list()
                                          .iota_replica_group_list()));

  if (shape.dimensions().size() != 1) {
    VLOG(1) << "Unsupported number of dimensions: " << profile.DebugString();
    return nullptr;
  }
  auto num_participating_devices =
      GetNumParticipatingDevices(collective_device_list);
  if (!num_participating_devices.ok()) {
    VLOG(1) << "Cannot get num participating devices: "
            << profile.DebugString();
    return nullptr;
  }
  std::vector<int64_t> new_dims(shape.dimensions().begin(),
                                shape.dimensions().end());
  new_dims[0] = new_dims.front() / *num_participating_devices;
  Shape p_shape = ShapeUtil::MakeShape(shape.element_type(), new_dims);
  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, p_shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateAllGather(
      shape, {p0}, /*all_gather_dimension=*/0, collective_device_list,
      profile.instruction().constrain_layout(),
      profile.instruction().channel_id(),
      profile.instruction().use_global_device_ids()));
  module->AddEntryComputation(entry_builder.Build());
  return module;
}

HloOpcode AsyncToSyncOpcode(const HloCollectiveInstruction& instr) {
  HloOpcode opcode = instr.opcode();
  switch (opcode) {
    case HloOpcode::kAllGatherStart:
      return HloOpcode::kAllGather;
    case HloOpcode::kAllReduceStart:
      return HloOpcode::kAllReduce;
    case HloOpcode::kAsyncStart:
      if (instr.async_wrapped_opcode() == HloOpcode::kReduceScatter) {
        return HloOpcode::kReduceScatter;
      };
      break;
    default:
      break;
  }
  return opcode;
}

absl::StatusOr<HloInstructionProfileList> ReadDefaultProfiles(
    const se::DeviceDescription& device_info) {
  DeviceHloInstructionProfiles profile;

  if (!tsl::protobuf::TextFormat::ParseFromString(kDefaultCollectivePTable,
                                                  &profile)) {
    return absl::FailedPreconditionError("Cannot parse a default profile.");
  }
  std::string key = HloOpProfiles::GetProfileName(device_info);

  if (!profile.entries().contains(key)) {
    return absl::NotFoundError(absl::StrCat("Cannot find key: ", key));
  }
  return profile.entries().at(key);
}

}  // namespace

/*static*/ absl::StatusOr<std::unique_ptr<CollectiveInterpolator>>
CollectiveInterpolator::Create(const se::DeviceDescription& device_info) {
  auto interpolators = std::make_unique<absl::flat_hash_map<
      InterpolatorKey, std::unique_ptr<InterpolatorBase<int64_t, 2>>>>();

  TF_ASSIGN_OR_RETURN(HloInstructionProfileList profiles,
                      ReadDefaultProfiles(device_info));
  for (auto& profile : profiles.entries()) {
    TF_ASSIGN_OR_RETURN(InterpolationSpecification spec,
                        Spec(profile, device_info));
    CollectiveInterpolator::InterpolatorKey key{
        /*opcode=*/spec.opcode,
        /*communication_type=*/spec.comm,
    };
    auto it = interpolators->find(key);
    if (it == interpolators->end()) {
      auto interpolator =
          std::make_unique<EuclideanComplementInterpolator<int64_t, 2>>(
              /*next_context=*/std::array<int64_t, 2>{-1, -1},
              /*next_power_context=*/std::array<int64_t, 2>{1, 1},
              /*max_context=*/std::array<int64_t, 2>{1 << 30, 8},
              /*min_context=*/std::array<int64_t, 2>{1 << 10, 8});

      (*interpolators)[key] = std::move(interpolator);
    }
    std::array<int64_t, 2> point = {spec.transfer_size, spec.num_devices};
    interpolators->at(key)->Add(point,
                                profile.network_throughput_bytes_per_sec());
  }
  return std::unique_ptr<CollectiveInterpolator>(
      new CollectiveInterpolator(std::move(interpolators), device_info));
}

/*static*/ absl::StatusOr<std::unique_ptr<CollectiveInterpolator>>
CollectiveInterpolator::Create(const HloInstructionProfileList& profiles,
                               const se::DeviceDescription& device_info) {
  auto interpolators = std::make_unique<absl::flat_hash_map<
      InterpolatorKey, std::unique_ptr<InterpolatorBase<int64_t, 2>>>>();

  for (auto& profile : profiles.entries()) {
    TF_ASSIGN_OR_RETURN(InterpolationSpecification spec,
                        Spec(profile, device_info));
    CollectiveInterpolator::InterpolatorKey key{
        /*opcode=*/spec.opcode,
        /*communication_type=*/spec.comm,
    };
    auto it = interpolators->find(key);
    if (it == interpolators->end()) {
      auto interpolator =
          std::make_unique<EuclideanNNInterpolator<int64_t, 2>>();
      (*interpolators)[key] = std::move(interpolator);
    }
    std::array<int64_t, 2> point = {spec.transfer_size, spec.num_devices};
    interpolators->at(key)->Add(point,
                                profile.network_throughput_bytes_per_sec());
  }
  return std::unique_ptr<CollectiveInterpolator>(
      new CollectiveInterpolator(std::move(interpolators), device_info));
}

std::optional<absl::Duration> CollectiveInterpolator::EstimatedRuntime(
    const HloCollectiveInstruction& instr) const {
  GpuHloCostAnalysis analysis(GpuHloCostAnalysis::Options(), device_info_);
  CHECK_OK(instr.Accept(&analysis));
  int64_t bytes_transferred = analysis.BytesTransferred(instr);
  auto comm = CommunicationType(instr, device_info_.gpu_compute_capability());
  if (!comm.ok()) {
    return std::nullopt;
  }
  auto num_devices = GetReplicaGroupCountAndSize(&instr);
  if (!num_devices.ok()) {
    return std::nullopt;
  }
  std::array<int64_t, 2> point({bytes_transferred, (*num_devices)->second});
  CollectiveInterpolator::InterpolatorKey key{
      /*opcode=*/AsyncToSyncOpcode(instr),
      /*communication_type=*/*comm,
  };
  if (!interpolators_->contains(key)) {
    VLOG(1) << "Cannot find key for instr: " << instr.ToString();
    return std::nullopt;
  }
  return absl::Seconds(1.0 * bytes_transferred /
                       interpolators_->at(key)->Eval(point));
}

/*static*/ std::unique_ptr<HloModule> CollectiveInterpolator::ConstructModule(
    const HloInstructionProfile& profile) {
  switch (*StringToHloOpcode(profile.instruction().opcode())) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
      return AllReduceModule(profile);
    case HloOpcode::kReduceScatter:
      return ReduceScatterModule(profile);
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
      return AllGatherModule(profile);
    default:
      LOG(FATAL) << "Unsupported profile instruction: "
                 << profile.DebugString();
  }
  return nullptr;
}

}  // namespace xla::gpu
