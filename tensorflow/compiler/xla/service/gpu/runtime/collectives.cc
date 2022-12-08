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

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::FlatMemrefView;
using xla::runtime::StridedMemrefView;

using llvm::ArrayRef;

#if XLA_ENABLE_XCCL
StatusOr<NcclComm::Lock> GetNcclComm(const NcclExecuteParams& params,
                                     int64_t group_mode, int64_t op_id,
                                     ArrayRef<int64_t> replica_group_offsets,
                                     ArrayRef<int64_t> replica_group_values) {
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

    int element_count = 1;
    for (int size : source->sizes) element_count *= size;
    device_buffers.emplace_back(DeviceBufferPair{
        source->dtype, element_count, GetDeviceAddress(*source),
        GetDeviceAddress(*destination)});
  }
  return device_buffers;
}

//===----------------------------------------------------------------------===//
// Collectives support library.
//===----------------------------------------------------------------------===//

static int64_t Key(int32_t uid, int32_t device_ordinal) {
  return static_cast<int64_t>(uid) << 32 | device_ordinal;
}

AsyncCollectivesSupport::AsyncCollectivesSupport(se::Stream* async_comm_stream)
    : async_comm_stream_(async_comm_stream) {}

Status CollectivesSupport::MaybeBlockAfterFirstRun(int32_t uid,
                                                   int32_t device_ordinal,
                                                   se::Stream* stream) {
  bool block = [&] {
    absl::MutexLock lock(&mutex_);
    return executed_.try_emplace(Key(uid, device_ordinal), true).second;
  }();
  return block ? stream->BlockHostUntilDone() : OkStatus();
}

StatusOr<se::Event> AsyncCollectivesSupport::PopEvent(int32_t uid,
                                                      int32_t device_ordinal) {
  absl::MutexLock lock(&mutex_);
  auto it = done_events_.find(Key(uid, device_ordinal));
  if (it == done_events_.end())
    return Internal(
        "Async collective event was not found uid=%d and device_ordinal=%d",
        uid, device_ordinal);

  se::Event done_event = std::move(it->second);
  done_events_.erase(it);
  return done_event;
}

Status AsyncCollectivesSupport::PushEvent(int32_t uid, int32_t device_ordinal,
                                          se::Event done_event) {
  absl::MutexLock lock(&mutex_);
  auto emplaced =
      done_events_.try_emplace(Key(uid, device_ordinal), std::move(done_event));
  if (!emplaced.second) return Internal("Done event has not been consumed");

  return OkStatus();
}

//===----------------------------------------------------------------------===//
// CollectivePermute.
//===----------------------------------------------------------------------===//

namespace {
struct CollectivePermute {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          CollectivesSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values,
                          ArrayRef<int64_t> source_peers,
                          ArrayRef<int64_t> target_peers) const;
  static CollectivePermute Handler() { return CollectivePermute(); }
};
}  // namespace

absl::Status CollectivePermute::operator()(
    const ServiceExecutableRunOptions* run_options,
    CollectivesSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values, ArrayRef<int64_t> source_peers,
    ArrayRef<int64_t> target_peers) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running CollectivePermute";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

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

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CollectivePermute, CollectivePermute::Handler(), checks,
    CustomCall::Bind("xla.gpu.collective_permute")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<ArrayRef<int64_t>>("replica_group_offsets")
        .Attr<ArrayRef<int64_t>>("replica_group_values")
        .Attr<ArrayRef<int64_t>>("source_peers")
        .Attr<ArrayRef<int64_t>>("target_peers"));

//===----------------------------------------------------------------------===//
// AllGather.
//===----------------------------------------------------------------------===//

namespace {
struct AllGather {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          CollectivesSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static AllGather Handler() { return AllGather(); }
};
}  // namespace

absl::Status AllGather::operator()(
    const ServiceExecutableRunOptions* run_options,
    CollectivesSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllGather";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  auto st = RunAllGather(*device_buffers, *stream, **comm);
  if (!st.ok()) return ToAbslStatus(st);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL diasbled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllGather, AllGather::Handler(), checks,
    CustomCall::Bind("xla.gpu.all_gather")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<ArrayRef<int64_t>>("replica_group_offsets")
        .Attr<ArrayRef<int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllReduce.
//===----------------------------------------------------------------------===//

namespace {
struct AllReduce {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          CollectivesSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          int64_t reduction_kind,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static AllReduce Handler() { return AllReduce(); }
};
}  // namespace

absl::Status AllReduce::operator()(
    const ServiceExecutableRunOptions* run_options,
    CollectivesSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduce";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  auto executed = RunAllReduce(static_cast<ReductionKind>(reduction_kind),
                               *device_buffers, *stream, **comm);
  if (!executed.ok()) return ToAbslStatus(executed);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllReduce, AllReduce::Handler(), checks,
    CustomCall::Bind("xla.gpu.all_reduce")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<ArrayRef<int64_t>>("replica_group_offsets")
        .Attr<ArrayRef<int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// AllReduceStart.
//===----------------------------------------------------------------------===//

namespace {
struct AllReduceStart {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          AsyncCollectivesSupport* async_collectives,
                          CustomCall::RemainingArgs args, int64_t group_mode,
                          int64_t op_id, int64_t reduction_kind,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values,
                          int32_t uid) const;
  static AllReduceStart Handler() { return AllReduceStart(); }
};
}  // namespace

absl::Status AllReduceStart::operator()(
    const ServiceExecutableRunOptions* run_options,
    AsyncCollectivesSupport* async_collectives, CustomCall::RemainingArgs args,
    int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceStart";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  // Wait until compute inputs are ready.
  async_collectives->async_comm_stream()->ThenWaitFor(params.stream);

  auto executed =
      RunAllReduce(static_cast<ReductionKind>(reduction_kind), *device_buffers,
                   *async_collectives->async_comm_stream(), **comm);
  if (!executed.ok()) return ToAbslStatus(executed);

  // Create an event on the async stream for the completion of the all-reduce.
  se::Event done_event(async_collectives->async_comm_stream()->parent());
  if (!done_event.Init()) return absl::InternalError("Failed to create event");
  async_collectives->async_comm_stream()->ThenRecordEvent(&done_event);

  auto pushed = async_collectives->PushEvent(
      uid, stream->parent()->device_ordinal(), std::move(done_event));
  if (!pushed.ok())
    return absl::InternalError(
        absl::StrFormat("Failed to push event to async collectives: %s",
                        pushed.error_message()));

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllReduceStart, AllReduceStart::Handler(), checks,
    CustomCall::Bind("xla.gpu.all_reduce_start")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()              // args
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<ArrayRef<int64_t>>("replica_group_offsets")
        .Attr<ArrayRef<int64_t>>("replica_group_values")
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// AllReduceDone.
//===----------------------------------------------------------------------===//

namespace {
struct AllReduceDone {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          CollectivesSupport* collectives,
                          AsyncCollectivesSupport* async_collectives,
                          CustomCall::RemainingArgs args, int32_t uid) const;
  static AllReduceDone Handler() { return AllReduceDone(); }
};
}  // namespace

absl::Status AllReduceDone::operator()(
    const ServiceExecutableRunOptions* run_options,
    CollectivesSupport* collectives, AsyncCollectivesSupport* async_collectives,
    CustomCall::RemainingArgs args, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceDone";
  se::Stream* stream = run_options->stream();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto event = async_collectives->PopEvent(uid, device_ordinal);
  if (!event.ok())
    return absl::InternalError(absl::StrFormat("Failed to pop event: %s",
                                               event.status().error_message()));

  stream->ThenWaitFor(&*event);

  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return absl::InternalError("Failed to block host");

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllReduceDone, AllReduceDone::Handler(), checks,
    CustomCall::Bind("xla.gpu.all_reduce_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .UserData<AsyncCollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid"));

//===----------------------------------------------------------------------===//
// AllToAll.
//===----------------------------------------------------------------------===//

namespace {
struct AllToAll {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          CollectivesSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, bool has_split_dimension,
                          int64_t op_id,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static AllToAll Handler() { return AllToAll(); }
};
}  // namespace

absl::Status AllToAll::operator()(
    const ServiceExecutableRunOptions* run_options,
    CollectivesSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, bool has_split_dimension, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllToAll";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  auto st = RunAllToAll(has_split_dimension, *device_buffers, *stream, **comm);
  if (!st.ok()) return ToAbslStatus(st);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  st = collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream);
  if (!st.ok()) return ToAbslStatus(st);

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AllToAll, AllToAll::Handler(), checks,
    CustomCall::Bind("xla.gpu.all_to_all")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<bool>("has_split_dimension")
        .Attr<int64_t>("op_id")
        .Attr<ArrayRef<int64_t>>("replica_group_offsets")
        .Attr<ArrayRef<int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// ReduceScatter.
//===----------------------------------------------------------------------===//

namespace {
struct ReduceScatter {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          CollectivesSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          int64_t reduction_kind,
                          ArrayRef<int64_t> replica_group_offsets,
                          ArrayRef<int64_t> replica_group_values) const;
  static ReduceScatter Handler() { return ReduceScatter(); }
};
}  // namespace

absl::Status ReduceScatter::operator()(
    const ServiceExecutableRunOptions* run_options,
    CollectivesSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running ReduceScatter";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (!comm.ok()) return ToAbslStatus(comm.status());

  auto device_buffers = GetDeviceBufferPairs(args);
  if (!device_buffers.ok()) return ToAbslStatus(device_buffers.status());

  auto executed = RunReduceScatter(static_cast<ReductionKind>(reduction_kind),
                                   *device_buffers, *stream, **comm);
  if (!executed.ok()) return ToAbslStatus(executed);

  int32_t device_ordinal = stream->parent()->device_ordinal();
  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return absl::InternalError("Failed to block host");

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ReduceScatter, ReduceScatter::Handler(), checks,
    CustomCall::Bind("xla.gpu.reduce_scatter")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<CollectivesSupport*>()
        .RemainingArgs()  // args
        .Attr<int32_t>("uid")
        .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
        .Attr<int64_t>("op_id")
        .Attr<int64_t>("reduction_kind")  // ReductionKind
        .Attr<ArrayRef<int64_t>>("replica_group_offsets")
        .Attr<ArrayRef<int64_t>>("replica_group_values"));

//===----------------------------------------------------------------------===//
// ReplicaId.
//===----------------------------------------------------------------------===//

namespace {
struct ReplicaId {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          FlatMemrefView result) const;
  static ReplicaId Handler() { return ReplicaId(); }
};
}  // namespace

absl::Status ReplicaId::operator()(
    const ServiceExecutableRunOptions* run_options,
    FlatMemrefView result) const {
  VLOG(3) << "Running ReplicaId";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return ToAbslStatus(global_device_id.status());

  StatusOr<DeviceAssignment::LogicalID> logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!logical_id.ok()) return ToAbslStatus(logical_id.status());

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  params.stream->ThenMemset32(&result_data, logical_id.value().replica_id,
                              /*size=*/4);

  return absl::OkStatus();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ReplicaId, ReplicaId::Handler(), checks,
    CustomCall::Bind("xla.gpu.replica_id")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<FlatMemrefView>());

//===----------------------------------------------------------------------===//
// PartitionId.
//===----------------------------------------------------------------------===//

namespace {
struct PartitionId {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          FlatMemrefView result) const;
  static PartitionId Handler() { return PartitionId(); }
};
}  // namespace

absl::Status PartitionId::operator()(
    const ServiceExecutableRunOptions* run_options,
    FlatMemrefView result) const {
  VLOG(3) << "Running PartitionId";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return ToAbslStatus(global_device_id.status());

  StatusOr<DeviceAssignment::LogicalID> logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!logical_id.ok()) return ToAbslStatus(logical_id.status());

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  params.stream->ThenMemset32(&result_data, logical_id.value().computation_id,
                              /*size=*/4);

  return absl::OkStatus();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    PartitionId, PartitionId::Handler(), checks,
    CustomCall::Bind("xla.gpu.partition_id")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<FlatMemrefView>());

//===----------------------------------------------------------------------===//

void RegisterCollectiveCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.collective_permute", &xla::gpu::CollectivePermute);
  registry.Register("xla.gpu.all_gather", &xla::gpu::AllGather);
  registry.Register("xla.gpu.all_reduce", &xla::gpu::AllReduce);
  registry.Register("xla.gpu.all_reduce_done", &xla::gpu::AllReduceDone);
  registry.Register("xla.gpu.all_reduce_start", &xla::gpu::AllReduceStart);
  registry.Register("xla.gpu.all_to_all", &xla::gpu::AllToAll);
  registry.Register("xla.gpu.reduce_scatter", &xla::gpu::ReduceScatter);
  registry.Register("xla.gpu.partition_id", &xla::gpu::PartitionId);
  registry.Register("xla.gpu.replica_id", &xla::gpu::ReplicaId);
}

}  // namespace gpu
}  // namespace xla
