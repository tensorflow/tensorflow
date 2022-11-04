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

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::Executable;

using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::succeeded;
using mlir::success;

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

namespace {
struct AllGather {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          std::string_view replica_group_offsets,
                          std::string_view replica_group_values) const;
  static AllGather Handler() { return AllGather(); }
};
}  // namespace

absl::Status AllGather::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    std::string_view replica_group_offsets,
    std::string_view replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllGather";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NCCL comm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

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

static bool AllGather(runtime::ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_gather")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<std::string_view>("replica_group_offsets")
          .Attr<std::string_view>("replica_group_values")
          .To<checks>(AllGather::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

JitRtAsyncCollectiveSupport::JitRtAsyncCollectiveSupport(
    se::Stream* async_comm_stream)
    : async_comm_stream_(async_comm_stream) {}

Status JitRtCollectiveSupport::MaybeBlockAfterFirstRun(int32_t uid,
                                                       int32_t device_ordinal,
                                                       se::Stream* stream) {
  bool block = [&] {
    absl::MutexLock lock(&mutex_);
    return executed_.try_emplace(Key(uid, device_ordinal), true).second;
  }();
  return block ? stream->BlockHostUntilDone() : OkStatus();
}

FailureOr<se::Event> JitRtAsyncCollectiveSupport::PopEvent(
    int32_t uid, int32_t device_ordinal) {
  const int64_t key = EventKey(uid, device_ordinal);

  absl::MutexLock lock(&mutex_);
  auto it = done_events_.find(key);
  if (it == done_events_.end()) return failure();

  se::Event done_event = std::move(it->second);
  done_events_.erase(it);
  return done_event;
}

LogicalResult JitRtAsyncCollectiveSupport::PushEvent(int32_t uid,
                                                     int32_t device_ordinal,
                                                     se::Event done_event) {
  const int64_t key = EventKey(uid, device_ordinal);

  absl::MutexLock lock(&mutex_);
  auto result = done_events_.try_emplace(key, std::move(done_event));
  if (!result.second) return failure();  // done event has not been consumed

  return success();
}
// ------------------------------------------------------------------------- //

namespace {
struct AllReduce {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          int64_t reduction_kind,
                          std::string_view replica_group_offsets,
                          std::string_view replica_group_values) const;
  static AllReduce Handler() { return AllReduce(); }
};
}  // namespace

absl::Status AllReduce::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    std::string_view replica_group_offsets,
    std::string_view replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduce";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

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

static bool AllReduce(runtime::ExecutionContext* ctx, void** args, void** attrs,
                      void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_reduce")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<std::string_view>("replica_group_offsets")
          .Attr<std::string_view>("replica_group_values")
          .To<checks>(AllReduce::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduceStart {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtAsyncCollectiveSupport* async_collectives,
                          CustomCall::RemainingArgs args, int64_t group_mode,
                          int64_t op_id, int64_t reduction_kind,
                          std::string_view replica_group_offsets,
                          std::string_view replica_group_values,
                          int32_t uid) const;
  static AllReduceStart Handler() { return AllReduceStart(); }
};
}  // namespace

absl::Status AllReduceStart::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtAsyncCollectiveSupport* async_collectives,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, std::string_view replica_group_offsets,
    std::string_view replica_group_values, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceStart";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

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

  if (failed(async_collectives->PushEvent(
          uid, stream->parent()->device_ordinal(), std::move(done_event))))
    return absl::InternalError("Failed to push event to async collectives");

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduceStart(runtime::ExecutionContext* ctx, void** args,
                           void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_reduce_start")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtAsyncCollectiveSupport*>()
          .RemainingArgs()              // args
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<std::string_view>("replica_group_offsets")
          .Attr<std::string_view>("replica_group_values")
          .Attr<int32_t>("uid")
          .To<checks>(AllReduceStart::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduceDone {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          JitRtAsyncCollectiveSupport* async_collectives,
                          CustomCall::RemainingArgs args, int32_t uid) const;
  static AllReduceDone Handler() { return AllReduceDone(); }
};
}  // namespace

absl::Status AllReduceDone::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives,
    JitRtAsyncCollectiveSupport* async_collectives,
    CustomCall::RemainingArgs args, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceDone";
  se::Stream* stream = run_options->stream();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto event = async_collectives->PopEvent(uid, device_ordinal);
  if (failed(event)) return absl::InternalError("Failed to pop event");

  stream->ThenWaitFor(&*event);

  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return absl::InternalError("Failed to block host");

  return absl::OkStatus();
#else   // XLA_ENABLE_XCCL
  return absl::InternalError("NCCL disabled");
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduceDone(runtime::ExecutionContext* ctx, void** args,
                          void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.all_reduce_done")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<JitRtCollectiveSupport*>()
                             .UserData<JitRtAsyncCollectiveSupport*>()
                             .RemainingArgs()  // args
                             .Attr<int32_t>("uid")
                             .To<checks>(AllReduceDone::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct AllToAll {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, bool has_split_dimension,
                          int64_t op_id, std::string_view replica_group_offsets,
                          std::string_view replica_group_values) const;
  static AllToAll Handler() { return AllToAll(); }
};
}  // namespace

absl::Status AllToAll::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, bool has_split_dimension, int64_t op_id,
    std::string_view replica_group_offsets,
    std::string_view replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllToAll";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NCCL comm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

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

static bool AllToAll(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_to_all")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<bool>("has_split_dimension")
          .Attr<int64_t>("op_id")
          .Attr<std::string_view>("replica_group_offsets")
          .Attr<std::string_view>("replica_group_values")
          .To<checks>(AllToAll::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct ReduceScatter {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          JitRtCollectiveSupport* collectives,
                          CustomCall::RemainingArgs args, int32_t uid,
                          int64_t group_mode, int64_t op_id,
                          int64_t reduction_kind,
                          std::string_view replica_group_offsets,
                          std::string_view replica_group_values) const;
  static ReduceScatter Handler() { return ReduceScatter(); }
};
}  // namespace

absl::Status ReduceScatter::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    std::string_view replica_group_offsets,
    std::string_view replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running ReduceScatter";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return absl::InternalError("Failed to get NcclComm");

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers))
    return absl::InternalError("Failed to get device buffers");

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

static bool ReduceScatter(runtime::ExecutionContext* ctx, void** args,
                          void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.reduce_scatter")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<std::string_view>("replica_group_offsets")
          .Attr<std::string_view>("replica_group_values")
          .To<checks>(ReduceScatter::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void RegisterCollectiveCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.collective_permute", &xla::gpu::CollectivePermute);
  registry.Register("xla.gpu.all_gather", &xla::gpu::AllGather);
  registry.Register("xla.gpu.all_reduce", &xla::gpu::AllReduce);
  registry.Register("xla.gpu.all_reduce_done", &xla::gpu::AllReduceDone);
  registry.Register("xla.gpu.all_reduce_start", &xla::gpu::AllReduceStart);
  registry.Register("xla.gpu.all_to_all", &xla::gpu::AllToAll);
  registry.Register("xla.gpu.reduce_scatter", &xla::gpu::ReduceScatter);
}

}  // namespace gpu
}  // namespace xla
