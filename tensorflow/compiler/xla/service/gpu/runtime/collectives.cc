/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/collectives.h"

#include <string>
#include <string_view>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::Executable;

namespace {
struct CollectivePermute {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          std::string_view replica_group_offsets,
                          std::string_view replica_group_values,
                          std::string_view source_peers,
                          std::string_view target_peers) const;
  static CollectivePermute Handler() { return CollectivePermute(); }
};
}  // namespace

absl::Status CollectivePermute::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    std::string_view replica_group_offsets,
    std::string_view replica_group_values, std::string_view source_peers,
    std::string_view target_peers) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running CollectivePermute";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");
  if (device_buffers->size() != 1) {
    return absl::InternalError(absl::StrFormat(
        "Expected device buffer size: 1, got %d", device_buffers->size()));
  }

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return ToAbslStatus(global_device_id.status());

  StatusOr<DeviceAssignment::LogicalID> current_logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!current_logical_id.ok())
    return ToAbslStatus(current_logical_id.status());

  const int64_t current_id = static_cast<CollectiveOpGroupMode>(group_mode) ==
                                     CollectiveOpGroupMode::kCrossReplica
                                 ? current_logical_id.value().replica_id
                                 : current_logical_id.value().computation_id;
  std::string device_string = NcclCollectiveThunk::GetDeviceString(params);

  NcclCollectivePermuteConfig::IdToSourceTargetMap id_to_source_target;
  for (int i = 0; i < source_peers.size(); ++i) {
    id_to_source_target.insert({target_peers[i], {}}).first->second.source =
        source_peers[i];
    id_to_source_target.insert({source_peers[i], {}}).first->second.target =
        target_peers[i];
  }
  const NcclCollectivePermuteConfig::SourceTargetMapEntry source_target =
      NcclCollectivePermuteConfig::GetSourceTarget(id_to_source_target,
                                                   current_id);

  auto executed =
      RunCollectivePermute(source_target, (*device_buffers)[0], *stream, **comm,
                           device_string, current_id);
  if (!executed.ok()) return ToAbslStatus(executed);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool CollectivePermute(runtime::ExecutionContext* ctx, void** args,
                              void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.collective_permute")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<std::string_view>("replica_group_offsets")
          .Attr<std::string_view>("replica_group_values")
          .Attr<std::string_view>("source_peers")
          .Attr<std::string_view>("target_peers")
          .To<checks>(CollectivePermute::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void RegisterCollectiveCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.collective_permute", &xla::gpu::CollectivePermute);
}

}  // namespace gpu
}  // namespace xla
