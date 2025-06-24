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
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

constexpr int64_t kMaxDefaultTransferSizeBytes = 1 << 30;

constexpr int64_t kMinDefaultTransferSizeBytes = 1 << 10;

constexpr int64_t kMaxDefaultNumberOfParticipatingDevices = 8;

constexpr int64_t kMinDefaultNumberOfParticipatingDevices = 2;

struct InterpolationSpecification {
  HloOpcode opcode;
  GPUCommunicationType comm;
  int64_t num_devices;
  int64_t transfer_size;
  CollectiveDeviceList device_list;
  PrimitiveType data_type;
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
    int num_devices_per_host, const HloInstructionProfile& profile,
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

  TF_ASSIGN_OR_RETURN(auto comm,
                      CommunicationType(num_devices_per_host, *collective,
                                        device_info.gpu_compute_capability()));
  TF_ASSIGN_OR_RETURN(int num_devices,
                      GetNumParticipatingDevices(collective->device_list()));

  return InterpolationSpecification{
      /*opcode=*/collective->opcode(),
      /*comm=*/comm,
      /*num_devices=*/num_devices,
      /*transfer_size=*/bytes_transferred,
      /*device_list=*/collective->device_list(),
      /*data_type=*/collective->shape().element_type(),
  };
}

std::unique_ptr<HloModule> AllReduceModule(
    const HloInstructionProfile& profile) {
  HloModuleConfig config;
  auto module = std::make_unique<HloModule>("m", config);
  auto shape = Shape::FromProto(profile.instruction().shape());
  if (!shape.ok()) {
    VLOG(1) << "Cannot parse shape: " << profile.DebugString();
    return nullptr;
  }

  HloComputation::Builder wrapped_computation("wrapped_computation");
  HloComputation::Builder entry_builder("entry");
  Shape s(shape->element_type(), /*dimensions=*/{});
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
      HloInstruction::CreateParameter(0, *shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateAllReduce(
      *shape, {p0}, subcomp, collective_device_list,
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
  auto shape = Shape::FromProto(profile.instruction().shape());
  if (!shape.ok()) {
    VLOG(1) << "Cannot parse shape: " << profile.DebugString();
    return nullptr;
  }

  HloComputation::Builder wrapped_computation("wrapped_computation");
  HloComputation::Builder entry_builder("entry");
  Shape s(shape->element_type(), /*dimensions=*/{});
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
  if (shape->dimensions().size() != 1) {
    VLOG(1) << "Unsupported number of dimensions: " << profile.DebugString();
    return nullptr;
  }
  std::vector<int64_t> new_dims(shape->dimensions().begin(),
                                shape->dimensions().end());
  auto num_participating_devices =
      GetNumParticipatingDevices(collective_device_list);
  if (!num_participating_devices.ok()) {
    VLOG(1) << "Cannot get num participating devices: "
            << profile.DebugString();
    return nullptr;
  }
  new_dims[0] = new_dims.front() * *num_participating_devices;
  Shape p_shape = ShapeUtil::MakeShape(shape->element_type(), new_dims);
  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, p_shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateReduceScatter(
      *shape, {p0}, subcomp, collective_device_list,
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
  auto shape = Shape::FromProto(profile.instruction().shape());
  if (!shape.ok()) {
    VLOG(1) << "Cannot parse shape: " << profile.DebugString();
    return nullptr;
  }

  HloComputation::Builder entry_builder("entry");

  CollectiveDeviceList collective_device_list(
      IotaReplicaGroupList::FromProto(profile.instruction()
                                          .collective_device_list()
                                          .iota_replica_group_list()));

  if (shape->dimensions().size() != 1) {
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
  std::vector<int64_t> new_dims(shape->dimensions().begin(),
                                shape->dimensions().end());
  new_dims[0] = new_dims.front() / *num_participating_devices;
  Shape p_shape = ShapeUtil::MakeShape(shape->element_type(), new_dims);
  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, p_shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateAllGather(
      *shape, {p0}, /*all_gather_dimension=*/0, collective_device_list,
      profile.instruction().constrain_layout(),
      profile.instruction().channel_id(),
      profile.instruction().use_global_device_ids()));
  module->AddEntryComputation(entry_builder.Build());
  return module;
}

std::unique_ptr<HloModule> AllToAllModule(
    const HloInstructionProfile& profile) {
  HloModuleConfig config;
  auto module = std::make_unique<HloModule>("m", config);
  auto shape = Shape::FromProto(profile.instruction().shape());
  if (!shape.ok()) {
    VLOG(1) << "Cannot parse shape: " << profile.DebugString();
    return nullptr;
  }

  HloComputation::Builder entry_builder("entry");
  CollectiveDeviceList collective_device_list(
      IotaReplicaGroupList::FromProto(profile.instruction()
                                          .collective_device_list()
                                          .iota_replica_group_list()));

  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, *shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateAllToAll(
      *shape, {p0}, collective_device_list,
      profile.instruction().constrain_layout(),
      profile.instruction().channel_id(),
      profile.instruction().use_global_device_ids()));
  module->AddEntryComputation(entry_builder.Build());
  return module;
}

std::optional<CollectiveDeviceList> CanonicalDeviceList(
    const HloCollectiveInstruction& instr) {
  if (instr.device_list().iota_replica_group_list().has_value()) {
    return instr.device_list();
  }
  auto num_groups_and_devices = GetReplicaGroupCountAndSize(&instr);
  if (!num_groups_and_devices.ok() || !num_groups_and_devices->has_value()) {
    VLOG(1) << "Failed to determine a number of devices participating in "
               "the collective: "
            << instr.ToString();
    return std::nullopt;
  }

  IotaReplicaGroupList iota((*num_groups_and_devices)->first,
                            (*num_groups_and_devices)->second);
  return CollectiveDeviceList(iota);
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

int64_t GetBytesTransferred(const HloInstruction& instr,
                            const se::DeviceDescription& device_info,
                            const GpuHloCostAnalysis* analysis) {
  if (analysis != nullptr) {
    return analysis->BytesTransferred(instr);
  }
  GpuHloCostAnalysis adhoc(GpuHloCostAnalysis::Options(), device_info);
  CHECK_OK(instr.Accept(&adhoc));
  return adhoc.BytesTransferred(instr);
}

bool RequiresAccumulation(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<std::unique_ptr<
    absl::flat_hash_map<CollectiveInterpolator::ExactInterpolatorKey,
                        std::unique_ptr<InterpolatorBase<int64_t, 1>>>>>
ConstructExactInterpolators(int num_devices_per_host,
                            const HloInstructionProfileList& profiles,
                            const se::DeviceDescription& device_info) {
  auto exact_interpolators = std::make_unique<
      absl::flat_hash_map<CollectiveInterpolator::ExactInterpolatorKey,
                          std::unique_ptr<InterpolatorBase<int64_t, 1>>>>();

  for (auto& profile : profiles.entries()) {
    TF_ASSIGN_OR_RETURN(InterpolationSpecification spec,
                        Spec(num_devices_per_host, profile, device_info));
    // Construct exact interpolators.
    CollectiveInterpolator::ExactInterpolatorKey exact_key{
        /*opcode=*/spec.opcode,
        /*device_list=*/spec.device_list,
        /*data_type=*/
        RequiresAccumulation(spec.opcode) ? std::make_optional(spec.data_type)
                                          : std::nullopt,
    };
    auto exact_it = exact_interpolators->find(exact_key);
    if (exact_it == exact_interpolators->end()) {
      auto interpolator = std::make_unique<
          EuclideanComplementInterpolator<int64_t, 1>>(
          /*next_context=*/std::array<int64_t, 1>{-1},
          /*next_power_context=*/std::array<int64_t, 1>{1},
          /*max_context=*/std::array<int64_t, 1>{kMaxDefaultTransferSizeBytes},
          /*min_context=*/std::array<int64_t, 1>{kMinDefaultTransferSizeBytes});

      (*exact_interpolators)[exact_key] = std::move(interpolator);
    }
    std::array<int64_t, 1> exact_point = {spec.transfer_size};
    exact_interpolators->at(exact_key)->Add(
        exact_point, profile.network_throughput_bytes_per_sec());
  }
  return exact_interpolators;
}

absl::StatusOr<std::unique_ptr<
    absl::flat_hash_map<CollectiveInterpolator::ExactInterpolatorKey,
                        std::unique_ptr<InterpolatorBase<int64_t, 1>>>>>
ConstructExactNNInterpolators(int num_devices_per_host,
                              const HloInstructionProfileList& profiles,
                              const se::DeviceDescription& device_info) {
  auto exact_interpolators = std::make_unique<
      absl::flat_hash_map<CollectiveInterpolator::ExactInterpolatorKey,
                          std::unique_ptr<InterpolatorBase<int64_t, 1>>>>();

  for (auto& profile : profiles.entries()) {
    TF_ASSIGN_OR_RETURN(InterpolationSpecification spec,
                        Spec(num_devices_per_host, profile, device_info));
    // Construct exact interpolators.
    CollectiveInterpolator::ExactInterpolatorKey exact_key{
        /*opcode=*/spec.opcode,
        /*device_list=*/spec.device_list,
        /*data_type=*/spec.data_type,
    };
    auto exact_it = exact_interpolators->find(exact_key);
    if (exact_it == exact_interpolators->end()) {
      auto interpolator =
          std::make_unique<EuclideanNNInterpolator<int64_t, 1>>();
      (*exact_interpolators)[exact_key] = std::move(interpolator);
    }
    std::array<int64_t, 1> exact_point = {spec.transfer_size};
    exact_interpolators->at(exact_key)->Add(
        exact_point, profile.network_throughput_bytes_per_sec());
  }
  return exact_interpolators;
}

absl::StatusOr<std::unique_ptr<
    absl::flat_hash_map<CollectiveInterpolator::FallbackInterpolatorKey,
                        std::unique_ptr<InterpolatorBase<int64_t, 2>>>>>
ConstructFallbackInterpolators(int num_devices_per_host,
                               const HloInstructionProfileList& profiles,
                               const se::DeviceDescription& device_info) {
  auto fallback_interpolators = std::make_unique<
      absl::flat_hash_map<CollectiveInterpolator::FallbackInterpolatorKey,
                          std::unique_ptr<InterpolatorBase<int64_t, 2>>>>();

  for (auto& profile : profiles.entries()) {
    TF_ASSIGN_OR_RETURN(InterpolationSpecification spec,
                        Spec(num_devices_per_host, profile, device_info));
    CollectiveInterpolator::FallbackInterpolatorKey key{
        /*opcode=*/spec.opcode,
        /*communication_type=*/spec.comm,
    };
    auto it = fallback_interpolators->find(key);
    if (it == fallback_interpolators->end()) {
      auto interpolator =
          std::make_unique<EuclideanComplementInterpolator<int64_t, 2>>(
              /*next_context=*/std::array<int64_t, 2>{-1, -1},
              /*next_power_context=*/std::array<int64_t, 2>{1, 1},
              /*max_context=*/
              std::array<int64_t, 2>{kMaxDefaultTransferSizeBytes,
                                     kMaxDefaultNumberOfParticipatingDevices},
              /*min_context=*/
              std::array<int64_t, 2>{kMinDefaultTransferSizeBytes,
                                     kMinDefaultNumberOfParticipatingDevices});

      (*fallback_interpolators)[key] = std::move(interpolator);
    }
    std::array<int64_t, 2> point = {spec.transfer_size, spec.num_devices};
    fallback_interpolators->at(key)->Add(
        point, profile.network_throughput_bytes_per_sec());
  }
  return fallback_interpolators;
}

absl::StatusOr<std::unique_ptr<
    absl::flat_hash_map<CollectiveInterpolator::FallbackInterpolatorKey,
                        std::unique_ptr<InterpolatorBase<int64_t, 2>>>>>
ConstructFallbackNNInterpolators(int num_devices_per_host,
                                 const HloInstructionProfileList& profiles,
                                 const se::DeviceDescription& device_info) {
  auto fallback_interpolators = std::make_unique<
      absl::flat_hash_map<CollectiveInterpolator::FallbackInterpolatorKey,
                          std::unique_ptr<InterpolatorBase<int64_t, 2>>>>();

  for (auto& profile : profiles.entries()) {
    TF_ASSIGN_OR_RETURN(InterpolationSpecification spec,
                        Spec(num_devices_per_host, profile, device_info));
    CollectiveInterpolator::FallbackInterpolatorKey key{
        /*opcode=*/spec.opcode,
        /*communication_type=*/spec.comm,
    };
    auto it = fallback_interpolators->find(key);
    if (it == fallback_interpolators->end()) {
      auto interpolator =
          std::make_unique<EuclideanNNInterpolator<int64_t, 2>>();

      (*fallback_interpolators)[key] = std::move(interpolator);
    }
    std::array<int64_t, 2> point = {spec.transfer_size, spec.num_devices};
    fallback_interpolators->at(key)->Add(
        point, profile.network_throughput_bytes_per_sec());
  }
  return fallback_interpolators;
}

}  // namespace

// We can get rid of `analysis` being nullptr once we get rid of stats
// collection passes.
/*static*/ absl::StatusOr<std::unique_ptr<CollectiveInterpolator>>
CollectiveInterpolator::Create(int num_devices_per_host,
                               const se::DeviceDescription& device_info,
                               const GpuHloCostAnalysis* analysis) {
  TF_ASSIGN_OR_RETURN(HloInstructionProfileList profiles,
                      ReadDefaultProfiles(device_info));

  TF_ASSIGN_OR_RETURN(
      auto exact_interpolators,
      ConstructExactInterpolators(num_devices_per_host, profiles, device_info));

  TF_ASSIGN_OR_RETURN(auto fallback_interpolators,
                      ConstructFallbackInterpolators(num_devices_per_host,
                                                     profiles, device_info));

  return std::unique_ptr<CollectiveInterpolator>(new CollectiveInterpolator(
      std::move(exact_interpolators), std::move(fallback_interpolators),
      device_info, num_devices_per_host, analysis));
}

/*static*/ absl::StatusOr<std::unique_ptr<CollectiveInterpolator>>
CollectiveInterpolator::Create(int num_devices_per_host,
                               const HloInstructionProfileList& profiles,
                               const se::DeviceDescription& device_info,
                               const GpuHloCostAnalysis* analysis) {
  TF_ASSIGN_OR_RETURN(auto exact_interpolators,
                      ConstructExactNNInterpolators(num_devices_per_host,
                                                    profiles, device_info));

  TF_ASSIGN_OR_RETURN(auto fallback_interpolators,
                      ConstructFallbackNNInterpolators(num_devices_per_host,
                                                       profiles, device_info));

  return std::unique_ptr<CollectiveInterpolator>(new CollectiveInterpolator(
      std::move(exact_interpolators), std::move(fallback_interpolators),
      device_info, num_devices_per_host, analysis));
}

std::optional<absl::Duration> CollectiveInterpolator::EstimatedRuntime(
    const HloCollectiveInstruction& instr) const {
  // Exact interpolation.
  int64_t bytes_transferred =
      GetBytesTransferred(instr, device_info_, analysis_);

  std::optional<CollectiveDeviceList> devices = CanonicalDeviceList(instr);
  if (devices.has_value()) {
    ExactInterpolatorKey exact_key{
        /*opcode=*/instr.opcode(),
        /*device_list=*/*devices,
        /*data_type=*/
        RequiresAccumulation(instr.opcode())
            ? std::make_optional(instr.shape().element_type())
            : std::nullopt,
    };

    if (exact_interpolators_->contains(exact_key)) {
      std::array<int64_t, 1> point({bytes_transferred});
      return absl::Seconds(1.0 * bytes_transferred /
                           exact_interpolators_->at(exact_key)->Eval(point));
    }
  }
  // Fallback interpolation.
  auto comm = CommunicationType(num_devices_per_host_, instr,
                                device_info_.gpu_compute_capability());
  if (!comm.ok()) {
    return std::nullopt;
  }
  auto num_devices = GetReplicaGroupCountAndSize(&instr);
  if (!num_devices.ok()) {
    return std::nullopt;
  }
  std::array<int64_t, 2> point({bytes_transferred, (*num_devices)->second});
  CollectiveInterpolator::FallbackInterpolatorKey key{
      /*opcode=*/AsyncToSyncOpcode(instr),
      /*communication_type=*/*comm,
  };
  if (!fallback_interpolators_->contains(key)) {
    VLOG(1) << "Cannot find key for instr: " << instr.ToString();
    return std::nullopt;
  }
  return absl::Seconds(1.0 * bytes_transferred /
                       fallback_interpolators_->at(key)->Eval(point));
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
    case HloOpcode::kAllToAll:
      return AllToAllModule(profile);
    default:
      LOG(FATAL) << "Unsupported profile instruction: "
                 << profile.DebugString();
  }
  return nullptr;
}

}  // namespace xla::gpu
