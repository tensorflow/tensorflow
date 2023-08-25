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
#include "tensorflow/compiler/xla/service/gpu/nccl_recv_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_send_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
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

// Helper function to run a collective either synchronously on main stream or
// asynchronously on the async stream.
absl::Status RunSyncOrAsync(
    const ServiceExecutableRunOptions* run_options,
    CollectivesSupport* collectives, AsyncCollectivesSupport* async_collectives,
    int32_t uid, bool is_async,
    absl::FunctionRef<absl::Status(se::Stream*)> to_run,
    AsyncStreamKind stream_kind = kAsyncStreamCollective) {
  se::Stream* main_stream = run_options->stream();
  se::Stream* async_stream =
      is_async ? async_collectives->async_comm_stream(stream_kind) : nullptr;
  if (is_async) {
    // Wait until compute inputs are ready.
    async_stream->ThenWaitFor(main_stream);
  }

  // Launch the collective on either the main or async stream.
  se::Stream* stream = is_async ? async_stream : main_stream;
  TF_RETURN_IF_ERROR(to_run(stream));

  if (is_async) {
    TF_RETURN_IF_ERROR(async_collectives->RecordEvent(uid, stream_kind));
  }
  int32_t device_ordinal = main_stream->parent()->device_ordinal();
  return collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, main_stream);
}

#if XLA_ENABLE_XCCL
StatusOr<NcclComm::Lock> GetNcclComm(
    const NcclExecuteParams& params, int64_t group_mode, int64_t op_id,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values, int64_t stream_id) {
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
                      static_cast<CollectiveOpGroupMode>(group_mode), op_id,
                      stream_id);
}
#endif  // XLA_ENABLE_XCCL

StatusOr<std::vector<DeviceBufferPair>> GetDeviceBufferPairs(
    CustomCall::RemainingArgs& args) {
  // Add MemRef arguments as buffer arguments.
  TF_RET_CHECK(args.size() % 2 == 0);
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

// Expects a single argument, and returns a device buffer pair with that
// argument replicated in both source and destination buffer.
StatusOr<std::vector<DeviceBufferPair>> GetSingleArgAsDeviceBufferPair(
    CustomCall::RemainingArgs& args) {
  TF_RET_CHECK(args.size() == 1);
  auto buffer = args.get<StridedMemrefView>(0);
  if (failed(buffer)) {
    return InvalidArgument("Unsupported device buffer type");
  }
  int64_t element_count = 1;
  for (int64_t size : buffer->sizes) element_count *= size;
  return std::vector<DeviceBufferPair>{
      DeviceBufferPair{buffer->dtype, element_count, GetDeviceAddress(*buffer),
                       GetDeviceAddress(*buffer)}};
}

absl::Status AsyncDoneImpl(const ServiceExecutableRunOptions* run_options,
                           AsyncCollectivesSupport* async_collectives,
                           int32_t uid, std::string_view done_type) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running " << done_type;
  se::Stream* stream = run_options->stream();

  TF_ASSIGN_OR_RETURN(se::Event event, async_collectives->PopEvent(uid));
  stream->ThenWaitFor(&event);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

//===----------------------------------------------------------------------===//
// CollectivePermute.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
using NcclP2PRunner = absl::FunctionRef<absl::Status(
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, ncclComm_t comm, absl::string_view device_string,
    int64_t current_id)>;

using DeviceBuffersGetter =
    absl::FunctionRef<StatusOr<std::vector<DeviceBufferPair>>(
        CustomCall::RemainingArgs& args)>;

absl::Status P2PImplCommon(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           se::Stream* stream, CustomCall::RemainingArgs args,
                           int64_t group_mode, int64_t op_id,
                           absl::Span<const int64_t> replica_group_offsets,
                           absl::Span<const int64_t> replica_group_values,
                           absl::Span<const int64_t> source_peers,
                           absl::Span<const int64_t> target_peers,
                           NcclP2PRunner runner,
                           DeviceBuffersGetter device_buffers_getter,
                           uint64_t stream_id) {
  NcclExecuteParams params(*run_options, stream->parent());

  const std::string device_string =
      NcclCollectiveThunk::GetDeviceString(params);
  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values, stream_id);
  if (!comm.ok()) return comm.status();

  auto device_buffers = device_buffers_getter(args);
  if (!device_buffers.ok()) return device_buffers.status();
  if (device_buffers->size() != 1) {
    return absl::InternalError(absl::StrFormat(
        "Expected device buffer size: 1, got %d", device_buffers->size()));
  }

  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(DeviceAssignment::LogicalID current_logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));

  const int64_t current_id = static_cast<CollectiveOpGroupMode>(group_mode) ==
                                     CollectiveOpGroupMode::kCrossReplica
                                 ? current_logical_id.replica_id
                                 : current_logical_id.computation_id;

  NcclP2PConfig::IdToSourceTargetMap id_to_source_target;
  for (int i = 0; i < source_peers.size(); ++i) {
    id_to_source_target[target_peers[i]].source = source_peers[i];
    id_to_source_target[source_peers[i]].target = target_peers[i];
  }
  const NcclP2PConfig::SourceTargetMapEntry source_target =
      NcclP2PConfig::GetSourceTarget(id_to_source_target, current_id);

  return RunRepeated(
      debug_options->xla_gpu_collective_inflation_factor(), [&]() -> Status {
        return runner(source_target, (*device_buffers)[0], *stream, **comm,
                      device_string, current_id);
      });
}
#endif  // XLA_ENABLE_XCCL

absl::Status CollectivePermuteImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, CollectivesSupport* collectives,
    AsyncCollectivesSupport* async_collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, bool is_async,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values,
    absl::Span<const int64_t> source_peers,
    absl::Span<const int64_t> target_peers) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running CollectivePermute " << (is_async ? "(Async)" : "(Sync)");
  return RunSyncOrAsync(run_options, collectives, async_collectives, uid,
                        is_async, [&](se::Stream* stream) {
                          return P2PImplCommon(
                              run_options, debug_options, stream, args,
                              group_mode, op_id, replica_group_offsets,
                              replica_group_values, source_peers, target_peers,
                              RunCollectivePermute, GetDeviceBufferPairs,
                              GetStreamId(is_async));
                        });
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
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<bool>("is_async")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values")
        .Attr<absl::Span<const int64_t>>("source_peers")
        .Attr<absl::Span<const int64_t>>("target_peers"));

//===----------------------------------------------------------------------===//
// Send.
//===----------------------------------------------------------------------===//

static absl::Status P2PSendImpl(const ServiceExecutableRunOptions* run_options,
                                const DebugOptions* debug_options,
                                CollectivesSupport* collectives,
                                AsyncCollectivesSupport* async_collectives,
                                CustomCall::RemainingArgs args, int32_t uid,
                                int64_t group_mode, int64_t op_id,
                                bool is_async,
                                absl::Span<const int64_t> replica_group_offsets,
                                absl::Span<const int64_t> replica_group_values,
                                absl::Span<const int64_t> source_peers,
                                absl::Span<const int64_t> target_peers) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running Send";
  TF_RET_CHECK(is_async);
  return RunSyncOrAsync(
      run_options, collectives, async_collectives, uid, is_async,
      [&](se::Stream* stream) {
        return P2PImplCommon(run_options, debug_options, stream, args,
                             group_mode, op_id, replica_group_offsets,
                             replica_group_values, source_peers, target_peers,
                             RunSend, GetSingleArgAsDeviceBufferPair,
                             GetStreamId(is_async, kAsyncStreamP2P));
      },
      kAsyncStreamP2P);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    P2PSend, FunctionWrapper<P2PSendImpl>(), checks,
    CustomCall::Bind("xla.gpu.send")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<bool>("is_async")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values")
        .Attr<absl::Span<const int64_t>>("source_peers")
        .Attr<absl::Span<const int64_t>>("target_peers"));

//===----------------------------------------------------------------------===//
// Recv.
//===----------------------------------------------------------------------===//

static absl::Status P2PRecvImpl(const ServiceExecutableRunOptions* run_options,
                                const DebugOptions* debug_options,
                                CollectivesSupport* collectives,
                                AsyncCollectivesSupport* async_collectives,
                                CustomCall::RemainingArgs args, int32_t uid,
                                int64_t group_mode, int64_t op_id,
                                bool is_async,
                                absl::Span<const int64_t> replica_group_offsets,
                                absl::Span<const int64_t> replica_group_values,
                                absl::Span<const int64_t> source_peers,
                                absl::Span<const int64_t> target_peers) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running Recv";
  TF_RET_CHECK(is_async);
  return RunSyncOrAsync(
      run_options, collectives, async_collectives, uid, is_async,
      [&](se::Stream* stream) {
        return P2PImplCommon(run_options, debug_options, stream, args,
                             group_mode, op_id, replica_group_offsets,
                             replica_group_values, source_peers, target_peers,
                             RunRecv, GetSingleArgAsDeviceBufferPair,
                             GetStreamId(is_async, kAsyncStreamP2P));
      },
      kAsyncStreamP2P);
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    P2PRecv, FunctionWrapper<P2PRecvImpl>(), checks,
    CustomCall::Bind("xla.gpu.recv")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<bool>("is_async")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values")
        .Attr<absl::Span<const int64_t>>("source_peers")
        .Attr<absl::Span<const int64_t>>("target_peers"));

//===----------------------------------------------------------------------===//
// AllGather.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status AllGatherImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values, bool is_async) {
  NcclExecuteParams params(*run_options, stream->parent());

  TF_ASSIGN_OR_RETURN(
      auto comm, GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                             replica_group_values, GetStreamId(is_async)));

  TF_ASSIGN_OR_RETURN(auto device_buffers, GetDeviceBufferPairs(args));

  return RunRepeated(
      debug_options->xla_gpu_collective_inflation_factor(),
      [&]() { return RunAllGather(device_buffers, *stream, *comm); });
}
#endif  // XLA_ENABLE_XCCL

absl::Status AllGatherImpl(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           CollectivesSupport* collectives,
                           AsyncCollectivesSupport* async_collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id, bool is_async,
                           absl::Span<const int64_t> replica_group_offsets,
                           absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllGather " << (is_async ? "(Async)" : "(Sync)");
  return RunSyncOrAsync(run_options, collectives, async_collectives, uid,
                        is_async, [&](se::Stream* stream) {
                          return AllGatherImplCommon(
                              run_options, debug_options, stream, args,
                              group_mode, op_id, replica_group_offsets,
                              replica_group_values, is_async);
                        });
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
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<bool>("is_async")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllReduce.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status AllReduceImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values, bool is_async) {
  NcclExecuteParams params(*run_options, stream->parent());

  TF_ASSIGN_OR_RETURN(
      auto comm, GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                             replica_group_values, GetStreamId(is_async)));

  TF_ASSIGN_OR_RETURN(auto device_buffers, GetDeviceBufferPairs(args));

  return RunRepeated(
      debug_options->xla_gpu_collective_inflation_factor(), [&]() {
        return RunAllReduce(static_cast<ReductionKind>(reduction_kind),
                            device_buffers, *stream, *comm);
      });
}
#endif  // XLA_ENABLE_XCCL

absl::Status AllReduceImpl(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           CollectivesSupport* collectives,
                           AsyncCollectivesSupport* async_collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id, bool is_async,
                           int64_t reduction_kind,
                           absl::Span<const int64_t> replica_group_offsets,
                           absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduce " << (is_async ? "(Async)" : "(Sync)");
  return RunSyncOrAsync(run_options, collectives, async_collectives, uid,
                        is_async, [&](se::Stream* stream) {
                          return AllReduceImplCommon(
                              run_options, debug_options, stream, args,
                              group_mode, op_id, reduction_kind,
                              replica_group_offsets, replica_group_values,
                              is_async);
                        });
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
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<bool>("is_async")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllToAll.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status AllToAllImplCommon(const ServiceExecutableRunOptions* run_options,
                                const DebugOptions* debug_options,
                                se::Stream* stream,
                                CustomCall::RemainingArgs args,
                                int64_t group_mode, bool has_split_dimension,
                                int64_t op_id,
                                absl::Span<const int64_t> replica_group_offsets,
                                absl::Span<const int64_t> replica_group_values,
                                bool is_async) {
  NcclExecuteParams params(*run_options, stream->parent());

  TF_ASSIGN_OR_RETURN(
      auto comm, GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                             replica_group_values, GetStreamId(is_async)));

  TF_ASSIGN_OR_RETURN(auto device_buffers, GetDeviceBufferPairs(args));

  return RunRepeated(
      debug_options->xla_gpu_collective_inflation_factor(), [&]() {
        return RunAllToAll(has_split_dimension, device_buffers, *stream, *comm);
      });
}
#endif  // XLA_ENABLE_XCCL

absl::Status AllToAllImpl(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          CollectivesSupport* collectives,
                          AsyncCollectivesSupport* async_collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, bool has_split_dimension,
                          int64_t op_id, bool is_async,
                          absl::Span<const int64_t> replica_group_offsets,
                          absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllToAll " << (is_async ? "(Async)" : "(Sync)");
  return RunSyncOrAsync(run_options, collectives, async_collectives, uid,
                        is_async, [&](se::Stream* stream) {
                          return AllToAllImplCommon(
                              run_options, debug_options, stream, args,
                              group_mode, has_split_dimension, op_id,
                              replica_group_offsets, replica_group_values,
                              is_async);
                        });
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
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<bool>("has_split_dimension")
        .Attr<int64_t>("op_id")
        .Attr<bool>("is_async")
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// ReduceScatter.
//===----------------------------------------------------------------------===//

#if XLA_ENABLE_XCCL
absl::Status ReduceScatterImplCommon(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, se::Stream* stream,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, absl::Span<const int64_t> replica_group_offsets,
    absl::Span<const int64_t> replica_group_values, bool is_async) {
  NcclExecuteParams params(*run_options, stream->parent());

  TF_ASSIGN_OR_RETURN(
      auto comm, GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                             replica_group_values, GetStreamId(is_async)));

  TF_ASSIGN_OR_RETURN(auto device_buffers, GetDeviceBufferPairs(args));

  return RunRepeated(
      debug_options->xla_gpu_collective_inflation_factor(), [&]() {
        return RunReduceScatter(static_cast<ReductionKind>(reduction_kind),
                                device_buffers, *stream, *comm);
      });
}
#endif  // XLA_ENABLE_XCCL

absl::Status ReduceScatterImpl(const ServiceExecutableRunOptions* run_options,
                               const DebugOptions* debug_options,
                               CollectivesSupport* collectives,
                               AsyncCollectivesSupport* async_collectives,
                               CustomCall::RemainingArgs args, int32_t uid,
                               int64_t group_mode, int64_t op_id, bool is_async,
                               int64_t reduction_kind,
                               absl::Span<const int64_t> replica_group_offsets,
                               absl::Span<const int64_t> replica_group_values) {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running ReduceScatter " << (is_async ? "(Async)" : "(Sync)");
  return RunSyncOrAsync(run_options, collectives, async_collectives, uid,
                        is_async, [&](se::Stream* stream) {
                          return ReduceScatterImplCommon(
                              run_options, debug_options, stream, args,
                              group_mode, op_id, reduction_kind,
                              replica_group_offsets, replica_group_values,
                              is_async);
                        });
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
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<bool>("is_async")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<absl::Span<const int64_t>>("replica_group_offsets")
        .Attr<absl::Span<const int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AsyncDone.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AsyncDone, FunctionWrapper<AsyncDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.async_collective_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<AsyncCollectivesSupport*>()
        .Attr<int32_t>("uid")
        .Attr<std::string_view>("done_type"));

//===----------------------------------------------------------------------===//
// ReplicaId.
//===----------------------------------------------------------------------===//

absl::Status ReplicaPartitionIdImpl(
    const ServiceExecutableRunOptions* run_options, FlatMemrefView result,
    bool is_replica_id) {
  VLOG(3) << "Running " << (is_replica_id ? "ReplicaId" : "PartitionId");
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream->parent());

  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  const uint32_t id =
      is_replica_id ? logical_id.replica_id : logical_id.computation_id;
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
  return block ? stream->BlockHostUntilDone() : absl::OkStatus();
}

AsyncCollectivesSupport::AsyncCollectivesSupport(
    absl::Span<se::Stream* const> async_streams)
    : async_comm_streams_(async_streams.begin(), async_streams.end()) {}

absl::Status AsyncCollectivesSupport::RecordEvent(
    int32_t uid, gpu::AsyncStreamKind async_stream_kind) {
  // Create an event on the async stream for the completion of the collective.
  se::Event done_event(async_comm_stream(async_stream_kind)->parent());
  if (!done_event.Init()) return absl::InternalError("Failed to create event");
  async_comm_stream(async_stream_kind)->ThenRecordEvent(&done_event);

  absl::MutexLock lock(&mutex_);
  auto [_, was_inserted] = done_events_.insert({uid, std::move(done_event)});
  if (!was_inserted) {
    return absl::InternalError(absl::StrFormat(
        "Async done event has not been consumed (uid=%d, device_ordinal=%d)",
        uid, async_comm_stream(async_stream_kind)->parent()->device_ordinal()));
  }
  return absl::OkStatus();
}

absl::StatusOr<se::Event> AsyncCollectivesSupport::PopEvent(int32_t uid) {
  absl::MutexLock lock(&mutex_);
  auto done_event = done_events_.extract(uid);
  if (!done_event) {
    return absl::InternalError(
        absl::StrFormat("Async done event was not found (uid=%d)", uid));
  }
  return std::move(done_event.mapped());
}

void RegisterCollectiveCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.collective_permute", CollectivePermute);
  registry.Register("xla.gpu.send", P2PSend);
  registry.Register("xla.gpu.recv", P2PRecv);
  registry.Register("xla.gpu.all_gather", AllGather);
  registry.Register("xla.gpu.all_reduce", AllReduce);
  registry.Register("xla.gpu.all_to_all", AllToAll);
  registry.Register("xla.gpu.reduce_scatter", ReduceScatter);

  registry.Register("xla.gpu.collective_done", AsyncDone);

  registry.Register("xla.gpu.partition_id", PartitionId);
  registry.Register("xla.gpu.replica_id", ReplicaId);
}

}  // namespace gpu
}  // namespace xla
