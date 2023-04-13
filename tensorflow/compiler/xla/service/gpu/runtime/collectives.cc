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
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::FlatMemrefView;
using xla::runtime::StridedMemrefView;

namespace {

Status RunRepeated(int32_t count, absl::FunctionRef<Status()> to_run) {
  if (count != 0) {
    VLOG(3) << "Running each collective " << count << " times\n";
  }
  for (int32_t i = 0; i < count; ++i) {
    TF_RETURN_IF_ERROR(to_run());
  }
  return OkStatus();
}

#if XLA_ENABLE_XCCL
StatusOr<NcclComm::Lock> GetNcclComm(
    const NcclExecuteParams& params, int64_t group_mode, int64_t op_id,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values) {
  // TODO(b/233930690): Pass the attribute below as a nested array.
  // Pass an array of arrays using two vectors; one specifying all the values
  // and another specifying the (ending) offsets of each array in the other
  // vector. Example: [ [10, 20, 30, 40], [50, 60], [70, 80, 90] ] turns into
  // offsets=[4, 6, 9] values=[10, 20, 30, 40, 50, 60, 70, 80, 90].
  std::vector<ReplicaGroup> replica_groups;
  int i = 0;
  for (int64_t replica_group_end : replica_group_offsets) {
    ReplicaGroup replica_group;
    while (i < replica_group_end)
      replica_group.add_replica_ids(replica_group_values[i++]);
    replica_groups.push_back(replica_group);
  }

  return LockNcclComm(params, replica_groups,
                      static_cast<CollectiveOpGroupMode>(group_mode), op_id);
}
#endif  // XLA_ENABLE_XCCL

StatusOr<std::vector<DeviceBufferPair>> GetDeviceBufferPairs(
    CustomCall::RemainingArgs& args) {
  // Add MemRef arguments as buffer arguments.
  const int buffer_pairs = args.size() / 2;
  std::vector<DeviceBufferPair> device_buffers;
  device_buffers.reserve(buffer_pairs);
  for (int i = 0; i < buffer_pairs; ++i) {
    auto source = args.get<StridedMemrefView>(i);
    auto destination = args.get<StridedMemrefView>(i + buffer_pairs);
    if (failed(source) || failed(destination)) {
      return InvalidArgument("Unsupported device buffer pair type");
    }

    int64_t element_count = 1;
    for (int64_t size : source->sizes) element_count *= size;
    device_buffers.emplace_back(DeviceBufferPair{
        source->dtype, element_count, GetDeviceAddress(*source),
        GetDeviceAddress(*destination)});
  }
  return device_buffers;
}

absl::Status AsyncDoneImpl(const ServiceExecutableRunOptions* run_options,
                           CollectivesSupport* collectives,
                           AsyncCollectivesSupport* async_collectives,
                           const char* op_name, int32_t uid) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running " << op_name;
  se::Stream* stream = run_options->stream();

  auto event = async_collectives->PopEvent(uid);
  if (!event.ok()) return event.status();
  stream->ThenWaitFor(&*event);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  return collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

//===----------------------------------------------------------------------===//
// CollectivePermute.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status CollectivePermuteImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values,
    absl::Span<const int64_t> source_peers,
    absl::Span<const int64_t> target_peers, int32_t repeat_count = 1) {
  NcclExecuteParams params(*run_options, stream->parent());

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

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
    id_to_source_target[target_peers[i]].source = source_peers[i];
    id_to_source_target[source_peers[i]].target = target_peers[i];
  }
  const NcclCollectivePermuteConfig::SourceTargetMapEntry source_target =
      NcclCollectivePermuteConfig::GetSourceTarget(id_to_source_target,
                                                   current_id);

  return ToAbslStatus(
      RunRepeated(debug_options->xla_gpu_collective_inflation_factor(), [&]() {
        return RunCollectivePermute(source_target, (*device_buffers)[0],
                                    *stream, **comm, device_string, current_id);
      }));
}
#endif  // XLA_ENABLE_XCCL

absl::Status CollectivePermuteImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, CollectivesSupport* collectives,
    CustomCall::RemainingArgs args, int32_t uid, int64_t group_mode,
    int64_t op_id, absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values,
    absl::Span<const int64_t> source_peers,
    absl::Span<const int64_t> target_peers) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running CollectivePermute";
  se::Stream* stream = run_options->stream();
  auto status = CollectivePermuteImplCommon(
      run_options, debug_options, stream, args, group_mode, op_id,
      replica_group_offsets, replica_group_values, source_peers, target_peers);
  if (!status.ok()) return status;

  int32_t device_ordinal = stream->parent()->device_ordinal();
  return collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CollectivePermute, FunctionWrapper<CollectivePermuteImpl>(), checks,
    CustomCall::Bind("xla.gpu.collective_permute")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values")
        .Attr<absl::Span<const int64_t>>("source_peers")
        .Attr<absl::Span<const int64_t>>("target_peers"));

//===----------------------------------------------------------------------===//
// CollectivePermuteStart.
//===----------------------------------------------------------------------===//

absl::Status CollectivePermuteStartImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options,
    AsyncCollectivesSupport* async_collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values,
    absl::Span<const int64_t> source_peers,
    absl::Span<const int64_t> target_peers) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running CollectivePermuteStart";
  se::Stream* stream = run_options->stream();
  se::Stream* async_stream = async_collectives->async_comm_stream();

  // Wait until compute inputs are ready.
  async_stream->ThenWaitFor(stream);

  auto status = CollectivePermuteImplCommon(
      run_options, debug_options, async_stream, args, group_mode, op_id,
      replica_group_offsets, replica_group_values, source_peers, target_peers);
  if (!status.ok()) return status;

  return async_collectives->RecordEvent(uid);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CollectivePermuteStart, FunctionWrapper<CollectivePermuteStartImpl>(),
    checks,
    CustomCall::Bind("xla.gpu.collective_permute_start")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values")
        .Attr<absl::Span<const int64_t>>("source_peers")
        .Attr<absl::Span<const int64_t>>("target_peers"));

//===----------------------------------------------------------------------===//
// CollectivePermuteDone.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CollectivePermuteDone, FunctionWrapper<AsyncDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.collective_permute_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .Value("CollectivePermuteDone")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// AllGather.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status AllGatherImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values) {
  NcclExecuteParams params(*run_options, stream->parent());

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  return ToAbslStatus(RunRepeated(
      debug_options->xla_gpu_collective_inflation_factor(),
      [&]() { return RunAllGather(*device_buffers, *stream, **comm); }));
}
#endif  // XLA_ENABLE_XCCL

absl::Status AllGatherImpl(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           CollectivesSupport* collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id,
                           absl::Span<const int64_t> replica_group_offsets,
                           absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllGather";
  se::Stream* stream = run_options->stream();
  auto status =
      AllGatherImplCommon(run_options, debug_options, stream, args, group_mode,
                          op_id, replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;

  int32_t device_ordinal = stream->parent()->device_ordinal();
  return collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL diasbled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllGather, FunctionWrapper<AllGatherImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_gather")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllGatherStart.
//===----------------------------------------------------------------------===//

absl::Status AllGatherStartImpl(const ServiceExecutableRunOptions* run_options,
                                const DebugOptions* debug_options,
                                AsyncCollectivesSupport* async_collectives,
                                CustomCall::RemainingArgs args,
                                int64_t group_mode, int64_t op_id,
                                absl::Span<const int64_t> replica_group_offsets,
                                absl::Span<const int64_t> replica_group_values,
                                int32_t uid) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllGatherStart";
  se::Stream* stream = run_options->stream();
  se::Stream* async_stream = async_collectives->async_comm_stream();

  // Wait until compute inputs are ready.
  async_stream->ThenWaitFor(stream);

  auto status = AllGatherImplCommon(
      run_options, debug_options, async_stream, args, group_mode, op_id,
      replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;

  return async_collectives->RecordEvent(uid);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllGatherStart, FunctionWrapper<AllGatherStartImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_gather_start")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()              // args
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// AllGatherDone.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllGatherDone, FunctionWrapper<AsyncDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_gather_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .Value("AllGatherDone")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// AllReduce.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status AllReduceImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values) {
  NcclExecuteParams params(*run_options, stream->parent());

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  return ToAbslStatus(
      RunRepeated(debug_options->xla_gpu_collective_inflation_factor(), [&]() {
        return RunAllReduce(static_cast<ReductionKind>(reduction_kind),
                            *device_buffers, *stream, **comm);
      }));
}
#endif  // XLA_ENABLE_XCCL

absl::Status AllReduceImpl(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           CollectivesSupport* collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id,
                           int64_t reduction_kind,
                           absl::Span<const int64_t> replica_group_offsets,
                           absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduce";
  se::Stream* stream = run_options->stream();
  auto status = AllReduceImplCommon(
      run_options, debug_options, stream, args, group_mode, op_id,
      reduction_kind, replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;

  int32_t device_ordinal = stream->parent()->device_ordinal();
  return collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllReduce, FunctionWrapper<AllReduceImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_reduce")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllReduceStart.
//===----------------------------------------------------------------------===//

absl::Status AllReduceStartImpl(const ServiceExecutableRunOptions* run_options,
                                const DebugOptions* debug_options,
                                AsyncCollectivesSupport* async_collectives,
                                CustomCall::RemainingArgs args,
                                int64_t group_mode, int64_t op_id,
                                int64_t reduction_kind,
                                absl::Span<const int64_t> replica_group_offsets,
                                absl::Span<const int64_t> replica_group_values,
                                int32_t uid) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceStart";
  se::Stream* stream = run_options->stream();
  se::Stream* async_stream = async_collectives->async_comm_stream();

  // Wait until compute inputs are ready.
  async_stream->ThenWaitFor(stream);

  auto status = AllReduceImplCommon(
      run_options, debug_options, async_stream, args, group_mode, op_id,
      reduction_kind, replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;

  return async_collectives->RecordEvent(uid);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllReduceStart, FunctionWrapper<AllReduceStartImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_reduce_start")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()              // args
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// AllReduceDone.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllReduceDone, FunctionWrapper<AsyncDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_reduce_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .Value("AllReduceDone")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// AllToAll.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status AllToAllImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode,
    bool has_split_dimension, int64_t op_id,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values) {
  NcclExecuteParams params(*run_options, stream->parent());

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  return ToAbslStatus(
      RunRepeated(debug_options->xla_gpu_collective_inflation_factor(), [&]() {
        return RunAllToAll(has_split_dimension, *device_buffers, *stream,
                           **comm);
      }));
}
#endif  // XLA_ENABLE_XCCL

absl::Status AllToAllImpl(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          CollectivesSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, bool has_split_dimension,
                          int64_t op_id,
                          absl::Span<const int64_t> replica_group_offsets,
                          absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllToAll";
  se::Stream* stream = run_options->stream();
  auto status = AllToAllImplCommon(run_options, debug_options, stream, args,
                                   group_mode, has_split_dimension, op_id,
                                   replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;

  int32_t device_ordinal = stream->parent()->device_ordinal();
  return collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllToAll, FunctionWrapper<AllToAllImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_to_all")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<bool>("has_split_dimension")
        .Attr<int64_t>("op_id")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllToAllStart.
//===----------------------------------------------------------------------===//
absl::Status AllToAllStartImpl(const ServiceExecutableRunOptions* run_options,
                               const DebugOptions* debug_options,
                               AsyncCollectivesSupport* async_collectives,
                               CustomCall::RemainingArgs args, int32_t uid,
                               int64_t group_mode, bool has_split_dimension,
                               int64_t op_id,
                               absl::Span<const int64_t> replica_group_offsets,
                               absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllToAllStart";
  se::Stream* stream = run_options->stream();
  se::Stream* async_stream = async_collectives->async_comm_stream();

  // Wait until compute inputs are ready.
  async_stream->ThenWaitFor(stream);

  auto status = AllToAllImplCommon(run_options, debug_options, async_stream,
                                   args, group_mode, has_split_dimension, op_id,
                                   replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;
  return async_collectives->RecordEvent(uid);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllToAllStart, FunctionWrapper<AllToAllStartImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_to_all_start")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<bool>("has_split_dimension")
        .Attr<int64_t>("op_id")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllToAllDone.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllToAllDone, FunctionWrapper<AsyncDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.all_to_all_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .Value("AllToAllDone")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// ReduceScatter.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status ReduceScatterImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values) {
  NcclExecuteParams params(*run_options, stream->parent());

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  return ToAbslStatus(
      RunRepeated(debug_options->xla_gpu_collective_inflation_factor(), [&]() {
        return RunReduceScatter(static_cast<ReductionKind>(reduction_kind),
                                *device_buffers, *stream, **comm);
      }));
}
#endif  // XLA_ENABLE_XCCL

absl::Status ReduceScatterImpl(const ServiceExecutableRunOptions* run_options,
                               const DebugOptions* debug_options,
                               CollectivesSupport* collectives,
                               CustomCall::RemainingArgs args, int32_t uid,
                               int64_t group_mode, int64_t op_id,
                               int64_t reduction_kind,
                               absl::Span<const int64_t> replica_group_offsets,
                               absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running ReduceScatter";
  se::Stream* stream = run_options->stream();
  auto status = ReduceScatterImplCommon(
      run_options, debug_options, stream, args, group_mode, op_id,
      reduction_kind, replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;

  int32_t device_ordinal = stream->parent()->device_ordinal();
  return collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ReduceScatter, FunctionWrapper<ReduceScatterImpl>(), checks,
    CustomCall::Bind("xla.gpu.reduce_scatter")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// ReduceScatterStart.
//===----------------------------------------------------------------------===//

absl::Status ReduceScatterStartImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options,
    AsyncCollectivesSupport* async_collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running ReduceScatterStart";
  se::Stream* stream = run_options->stream();
  se::Stream* async_stream = async_collectives->async_comm_stream();

  // Wait until compute inputs are ready.
  async_stream->ThenWaitFor(stream);

  auto status = ReduceScatterImplCommon(
      run_options, debug_options, async_stream, args, group_mode, op_id,
      reduction_kind, replica_group_offsets, replica_group_values);
  if (!status.ok()) return status;

  return async_collectives->RecordEvent(uid);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ReduceScatterStart, FunctionWrapper<ReduceScatterStartImpl>(), checks,
    CustomCall::Bind("xla.gpu.reduce_scatter_start")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// ReduceScatterDone.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ReduceScatterDone, FunctionWrapper<AsyncDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.reduce_scatter_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .Value("ReduceScatterDone")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// ReplicaId.
//===----------------------------------------------------------------------===//

absl::Status ReplicaPartitionIdImpl(
    const ServiceExecutableRunOptions* run_options, FlatMemrefView result,
    bool is_replica_id) {
  VLOG(3) << "Running " << (is_replica_id ? "ReplicaId" : "PartitionId");
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream->parent());

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return ToAbslStatus(global_device_id.status());

  StatusOr<DeviceAssignment::LogicalID> logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!logical_id.ok()) return ToAbslStatus(logical_id.status());

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  const uint32_t id =
      is_replica_id ? logical_id->replica_id : logical_id->computation_id;
  stream->ThenMemset32(&result_data, id, /*size=*/4);
  return absl::OkStatus();
}

absl::Status ReplicaIdImpl(const ServiceExecutableRunOptions* run_options,
                           FlatMemrefView result) {
  return ReplicaPartitionIdImpl(run_options, result, /*is_replica_id=*/true);
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ReplicaId, FunctionWrapper<ReplicaIdImpl>(), checks,
    CustomCall::Bind("xla.gpu.replica_id")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<FlatMemrefView>());

//===----------------------------------------------------------------------===//
// PartitionId.
//===----------------------------------------------------------------------===//

absl::Status PartitionIdImpl(const ServiceExecutableRunOptions* run_options,
                             FlatMemrefView result) {
  return ReplicaPartitionIdImpl(run_options, result, /*is_replica_id=*/false);
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    PartitionId, FunctionWrapper<PartitionIdImpl>(), checks,
    CustomCall::Bind("xla.gpu.partition_id")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<FlatMemrefView>());

//===----------------------------------------------------------------------===//

int64_t Key(int32_t uid, int32_t device_ordinal) {
  return static_cast<int64_t>(uid) << 32 | device_ordinal;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Collectives support library.
//===----------------------------------------------------------------------===//

absl::Status CollectivesSupport::MaybeBlockAfterFirstRun(int32_t uid,
                                                         int32_t device_ordinal,
                                                         se::Stream* stream) {
  bool block = [&] {
    absl::MutexLock lock(&mutex_);
    return executed_.insert(Key(uid, device_ordinal)).second;
  }();
  return block ? ToAbslStatus(stream->BlockHostUntilDone()) : absl::OkStatus();
}

AsyncCollectivesSupport::AsyncCollectivesSupport(se::Stream* async_comm_stream)
    : async_comm_stream_(async_comm_stream) {}

absl::Status AsyncCollectivesSupport::RecordEvent(int32_t uid) {
  // Create an event on the async stream for the completion of the collective.
  se::Event done_event(async_comm_stream_->parent());
  if (!done_event.Init()) return absl::InternalError("Failed to create event");
  async_comm_stream_->ThenRecordEvent(&done_event);

  absl::MutexLock lock(&mutex_);
  auto [_, was_inserted] = done_events_.insert({uid, std::move(done_event)});
  if (!was_inserted) {
    return absl::InternalError(absl::StrFormat(
        "Async done event has not been consumed (uid=%d, device_ordinal=%d)",
        uid, async_comm_stream_->parent()->device_ordinal()));
  }
  return absl::OkStatus();
}

absl::StatusOr<se::Event> AsyncCollectivesSupport::PopEvent(int32_t uid) {
  absl::MutexLock lock(&mutex_);
  auto done_event = done_events_.extract(uid);
  if (!done_event) {
    return absl::InternalError(absl::StrFormat(
        "Async done event was not found (uid=%d, device_ordinal=%d)", uid,
        async_comm_stream_->parent()->device_ordinal()));
  }
  return std::move(done_event.mapped());
}

void RegisterCollectiveCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.collective_permute", CollectivePermute);
  registry.Register("xla.gpu.collective_permute_done", CollectivePermuteDone);
  registry.Register("xla.gpu.collective_permute_start", CollectivePermuteStart);
  registry.Register("xla.gpu.all_gather", AllGather);
  registry.Register("xla.gpu.all_gather_done", AllGatherDone);
  registry.Register("xla.gpu.all_gather_start", AllGatherStart);
  registry.Register("xla.gpu.all_reduce", AllReduce);
  registry.Register("xla.gpu.all_reduce_done", AllReduceDone);
  registry.Register("xla.gpu.all_reduce_start", AllReduceStart);
  registry.Register("xla.gpu.all_to_all", AllToAll);
  registry.Register("xla.gpu.all_to_all_start", AllToAllStart);
  registry.Register("xla.gpu.all_to_all_done", AllToAllDone);
  registry.Register("xla.gpu.reduce_scatter", ReduceScatter);
  registry.Register("xla.gpu.reduce_scatter_start", ReduceScatterStart);
  registry.Register("xla.gpu.reduce_scatter_done", ReduceScatterDone);
  registry.Register("xla.gpu.partition_id", PartitionId);
  registry.Register("xla.gpu.replica_id", ReplicaId);
}

}  // namespace gpu
}  // namespace xla
