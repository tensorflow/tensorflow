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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/backends/gpu/tests/collective_ops_ffi_kernels.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/runtime/device_id.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
using ::testing::Values;

struct SynchronizationSignals {
  absl::Mutex mutex;
  absl::BlockingCounter finished_kernels_counter;

  explicit SynchronizationSignals(int num_expected_kernels)
      : finished_kernels_counter(num_expected_kernels) {}

  void IncrementFinishedKernels() {
    absl::MutexLock lock(mutex);
    finished_kernels_counter.DecrementCount();
  }
};

absl::NoDestructor<std::unique_ptr<SynchronizationSignals>> global_signals;

class CollectiveOpsTestFFI : public CollectiveOpsE2ETestBase {
 public:
  CollectiveOpsTestFFI()
      : CollectiveOpsE2ETestBase(/*memory_size=*/32 * kMB,
                                 /*collectives_memory_size=*/32 * kMB) {}
  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    *global_signals =
        std::make_unique<SynchronizationSignals>(/*num_expected_kernels=*/2);
  }

  void TearDown() override {
    CollectiveOpsE2ETestBase::TearDown();
    global_signals->reset();
  }
};

static constexpr int64_t kNumReplicas = 2;

// In this test we execute all collective operations across all devices.
static ReplicaGroup AllDevices() {
  ReplicaGroup group;
  for (int64_t i = 0; i < kNumReplicas; ++i) {
    group.add_replica_ids(i);
  }
  return group;
}

// This is a prepare handler that tells XLA:GPU runtime what collective cliques
// should be acquired before the execution starts. All collective operations
// must let XLA:GPU runtime know what cliques they need ahead of time.
static absl::Status PrepareAllReduce(
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests) {
  TF_RET_CHECK(collective_params && clique_requests);

  // Request a clique that covers all devices (this test runs on 2 gpus).
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID, false));

  std::vector<GlobalDeviceId> all_device_groups;
  for (int i = 0; i < kNumReplicas; ++i) {
    all_device_groups.push_back(GlobalDeviceId(i));
  }

  // Ask XLA:GPU runtime to acquire a clique for this key. Later we will be
  // able to get access to it from the execute handler.
  TF_RETURN_IF_ERROR(clique_requests->RequestClique(
      clique_key, /*device_groups=*/{all_device_groups}));

  return absl::OkStatus();
}

// This is a prepare handler for device-initiated collective operation which
// in addition to the clique asks for device comms and symmetric memory.
static absl::Status PrepareDeviceAllReduce(
    ffi::BufferR0<U32> src, ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests,
    CollectiveMemoryRequests* memory_requests) {
  TF_RET_CHECK(collective_params && clique_requests);

  // Request a clique that covers all devices (this test runs on 2 gpus).
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
          AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  // Ask for a device communicator with 8 lsa barriers.
  CollectiveCliqueRequests::CliqueRequirements requirements;
  requirements.dev_comm = GpuDeviceCommunicator::Requirements{8};
  std::vector<GlobalDeviceId> all_device_groups;
  for (int i = 0; i < kNumReplicas; ++i) {
    all_device_groups.push_back(GlobalDeviceId(i));
  }
  // Request XLA:GPU runtime to acquire a clique for this key. Later we will be
  // able to get access to it from the execute handler.
  TF_RETURN_IF_ERROR(clique_requests->RequestClique(
      clique_key, /*device_groups=*/{all_device_groups}, requirements));

  // Request src and dst buffers to be symmetric on the given clique.
  TF_RETURN_IF_ERROR(memory_requests->RequestSymmetricAddress(
      clique_key, src.device_memory()));
  TF_RETURN_IF_ERROR(memory_requests->RequestSymmetricAddress(
      clique_key, dst->device_memory()));

  return absl::OkStatus();
}

// This is a prepare handler for device-initiated collective operation which
// uses collective multimem to access peer devices.
static absl::Status PrepareMulticastAllReduce(
    ffi::BufferR0<U32> src, ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    CollectiveCliqueRequests* clique_requests,
    CollectiveMemoryRequests* memory_requests) {
  TF_RET_CHECK(collective_params && memory_requests);

  // Request a clique that covers all devices (this test runs on 2 gpus).
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
          AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  std::vector<GlobalDeviceId> all_device_groups;
  for (int i = 0; i < kNumReplicas; ++i) {
    all_device_groups.push_back(GlobalDeviceId(i));
  }
  TF_RETURN_IF_ERROR(clique_requests->RequestClique(
      clique_key, /*device_groups=*/{all_device_groups}));

  // Request src buffer to be mapped to multimem on the given clique.
  //
  // IMPORTANT: We don't request the clique itself, because multimem addresses
  // accessible directly to kernels without a need for support from the
  // underlying collective library.
  TF_RETURN_IF_ERROR(memory_requests->RequestMulticastAddress(
      clique_key, src.device_memory()));

  return absl::OkStatus();
}

// This is a prepare handler for device-initiated collective operation which
// uses collective peer memory to access peer devices.
static absl::Status PreparePeerAllReduce(
    ffi::BufferR0<U32> src, ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    CollectiveMemoryRequests* memory_requests) {
  TF_RET_CHECK(collective_params && memory_requests);

  // Request a clique that covers all devices (this test runs on 2 gpus).
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
          AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  // Request src buffer from all peers in the given clique.
  TF_RETURN_IF_ERROR(
      memory_requests->RequestPeerAddress(clique_key, src.device_memory()));

  return absl::OkStatus();
}

// FFI handler that uses XLA:GPU collectives API to perform an all reduce. This
// is just a test that demonstrates how to use XLA:GPU collectives API in an FFI
// handler, builtin all-reduce is a much better option. This version
// demonstrates requesting a communication stream and synchronizing it with the
// main stream.
static absl::Status AllReduce(se::Stream* stream, se::Stream* comm_stream,
                              ffi::BufferR0<U32> src,
                              ffi::Result<ffi::BufferR0<U32>> dst,
                              const CollectiveParams* collective_params,
                              const CollectiveCliques* collective_cliques) {
  TF_RET_CHECK(collective_params && collective_cliques);

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID, false));

  // Get the communicator for the requested clique.
  TF_ASSIGN_OR_RETURN(Communicator * comm,
                      collective_cliques->GetComm(
                          clique_key, collective_params->global_device_id));

  // Synchronize communication stream with the main stream: make the
  // communication stream wait for all prior work on the main stream.
  TF_RETURN_IF_ERROR(comm_stream->WaitFor(stream));

  // Launch all-reduce on the communication stream.
  Future<> future =
      comm->AllReduce(src.device_memory(), dst->device_memory(),
                      src.element_type(), src.element_count(),
                      ReductionKind::SUM, GpuCollectives::On(*comm_stream));
  TF_RETURN_IF_ERROR(future.Await());

  // Synchronize main stream with the communication stream: make the main
  // stream wait for the all-reduce to complete.
  TF_RETURN_IF_ERROR(stream->WaitFor(comm_stream));

  return absl::OkStatus();
}

// FFI handler that launches device kernel that does all-reduce using NCCL
// device-side APIs.
static absl::Status DeviceAllReduce(se::Stream* stream, ffi::BufferR0<U32> src,
                                    ffi::Result<ffi::BufferR0<U32>> dst,
                                    const CollectiveParams* collective_params,
                                    const CollectiveCliques* collective_cliques,
                                    const CollectiveMemory* collective_memory) {
  TF_RET_CHECK(collective_params && collective_cliques && collective_memory);

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID, false));

  // Find collective memory for src and dst buffers.
  auto [sym_src, src_offset] =
      collective_memory->FindSymmetricMemory(clique_key, src.device_memory());
  auto [sym_dst, dst_offset] =
      collective_memory->FindSymmetricMemory(clique_key, dst->device_memory());
  TF_RET_CHECK(sym_src && sym_dst);

  // Get requested device communicator for a given clique.
  auto rank = clique_key.rank(collective_params->global_device_id);
  TF_ASSIGN_OR_RETURN(
      GpuDeviceCommunicator * dev_comm,
      collective_cliques->GetDeviceComm(
          clique_key, *rank, GpuDeviceCommunicator::Requirements{8}));

  // Load custom kernel that does device-initiated collectives.
  TF_ASSIGN_OR_RETURN(
      auto kernel,
      se::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<SymmetricAllReduce>(collective_params->executor));

  se::BlockDim block_dims(1);
  se::ThreadDim thread_dims(8);

  TF_RETURN_IF_ERROR(kernel.Launch(thread_dims, block_dims, stream, dev_comm,
                                   sym_src, sym_dst, src_offset, dst_offset,
                                   src.element_count()));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  SynchronizationSignals* signals = global_signals->get();
  signals->IncrementFinishedKernels();
  return absl::OkStatus();
}

static absl::Status BlockedDeviceAllReduce(
    se::Stream* stream, ffi::BufferR0<U32> src,
    ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques,
    const CollectiveMemory* collective_memory) {
  TF_RETURN_IF_ERROR(DeviceAllReduce(stream, src, dst, collective_params,
                                     collective_cliques, collective_memory));
  return stream->BlockHostUntilDone();
}

// FFI handler that launches device kernel that does all-reduce using NCCL
// device-side APIs.
static absl::Status DelayedDeviceAllReduce(
    se::Stream* stream, ffi::BufferR0<U32> src,
    ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveCliques* collective_cliques,
    const CollectiveMemory* collective_memory) {
  TF_RETURN_IF_ERROR(
      stream->DoHostCallback([]() { absl::SleepFor(absl::Seconds(1)); }));
  TF_RETURN_IF_ERROR(DeviceAllReduce(stream, src, dst, collective_params,
                                     collective_cliques, collective_memory));
  return absl::OkStatus();
}

static absl::Status MulticastAllReduce(
    se::Stream* stream, ffi::BufferR0<U32> src,
    ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveMemory* collective_memory) {
  TF_RET_CHECK(collective_params && collective_memory);

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
          AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  auto [src_mmem, src_offset] =
      collective_memory->FindMultimemAddress(clique_key, src.device_memory());

  TF_RET_CHECK(src_mmem != nullptr);

  // Load custom kernel that does device-initiated collectives.
  TF_ASSIGN_OR_RETURN(
      auto kernel,
      se::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<MultimemAllReduce>(collective_params->executor));

  // Create device addresses from multimem pointer.
  auto src_addr =
      se::DeviceAddress<uint32_t>::MakeFromByteSize(src_mmem, src.size_bytes());

  // Because we launch a trivial kernel we use a device-side rendezvous to make
  // sure that both devices will execute the kernel together after inputs become
  // ready on both devices. Any real kernel must use device-side barriers.
  static constexpr int32_t kKey = 0;
  const int32_t* key = &kKey;
  TF_RETURN_IF_ERROR(Rendezvous<const int32_t*>(
      "MulticastAllReduce", key, 2, absl::Seconds(1), absl::Seconds(5)));

  se::BlockDim block_dims(1);
  se::ThreadDim thread_dims(8);

  TF_RETURN_IF_ERROR(kernel.Launch(thread_dims, block_dims, stream, src_addr,
                                   dst->device_memory(), src_offset,
                                   src.element_count()));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  SynchronizationSignals* signals = global_signals->get();
  signals->IncrementFinishedKernels();
  return absl::OkStatus();
}

// FFI handler that launches device kernel that does all-reduce using multicast
// memory access.
static absl::Status DelayedMulticastAllReduce(
    se::Stream* stream, ffi::BufferR0<U32> src,
    ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveMemory* collective_memory) {
  TF_RETURN_IF_ERROR(
      stream->DoHostCallback([]() { absl::SleepFor(absl::Seconds(1)); }));
  TF_RETURN_IF_ERROR(MulticastAllReduce(stream, src, dst, collective_params,
                                        collective_memory));
  return absl::OkStatus();
}

// FFI handler that launches device kernel that does all-reduce using multicast
// memory access.
static absl::Status BlockedMulticastAllReduce(
    se::Stream* stream, ffi::BufferR0<U32> src,
    ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveMemory* collective_memory) {
  TF_RETURN_IF_ERROR(MulticastAllReduce(stream, src, dst, collective_params,
                                        collective_memory));
  return stream->BlockHostUntilDone();
}

// FFI handler that launches device kernel that does all-reduce using peer
// memory access.
static absl::Status PeerAllReduce(se::Stream* stream, ffi::BufferR0<U32> src,
                                  ffi::Result<ffi::BufferR0<U32>> dst,
                                  const CollectiveParams* collective_params,
                                  const CollectiveMemory* collective_memory) {
  TF_RET_CHECK(collective_params && collective_memory);

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(
          *collective_params, {AllDevices()},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
          AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE));

  auto src0 = collective_memory->FindPeerAddress(clique_key, RankId(0),
                                                 src.device_memory());
  auto src1 = collective_memory->FindPeerAddress(clique_key, RankId(1),
                                                 src.device_memory());

  TF_RET_CHECK(src0 && src1);

  // Load custom kernel that does device-initiated collectives.
  TF_ASSIGN_OR_RETURN(
      auto kernel,
      se::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<Peer2AllReduce>(collective_params->executor));

  // Because we launch a trivial kernel we use a device-side rendezvous to make
  // sure that both devices will execute the kernel together after inputs become
  // ready on both devices. Any real kernel must use device-side barriers.
  static constexpr int32_t kKey = 0;
  const int32_t* key = &kKey;
  TF_RETURN_IF_ERROR(Rendezvous<const int32_t*>(
      "PeerAllReduce", key, 2, absl::Seconds(1), absl::Seconds(5)));

  se::BlockDim block_dims(1);
  se::ThreadDim thread_dims(8);

  TF_RETURN_IF_ERROR(kernel.Launch(thread_dims, block_dims, stream, *src0,
                                   *src1, dst->device_memory(),
                                   src.element_count()));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  SynchronizationSignals* signals = global_signals->get();
  signals->IncrementFinishedKernels();
  return absl::OkStatus();
}

static absl::Status BlockedPeerAllReduce(
    se::Stream* stream, ffi::BufferR0<U32> src,
    ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveMemory* collective_memory) {
  TF_RETURN_IF_ERROR(
      PeerAllReduce(stream, src, dst, collective_params, collective_memory));
  return stream->BlockHostUntilDone();
}

static absl::Status DelayedPeerAllReduce(
    se::Stream* stream, ffi::BufferR0<U32> src,
    ffi::Result<ffi::BufferR0<U32>> dst,
    const CollectiveParams* collective_params,
    const CollectiveMemory* collective_memory) {
  TF_RETURN_IF_ERROR(
      PeerAllReduce(stream, src, dst, collective_params, collective_memory));
  TF_RETURN_IF_ERROR(
      stream->DoHostCallback([]() { absl::SleepFor(absl::Seconds(2)); }));
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kPrepareAllReduce, PrepareAllReduce,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER(kAllReduce, AllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::CommunicationStream<0>>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliques>());

XLA_FFI_DEFINE_HANDLER(kPrepareDeviceAllReduce, PrepareDeviceAllReduce,
                       ffi::Ffi::BindPrepare()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliqueRequests>()
                           .Ctx<ffi::CollectiveMemoryRequests>());

XLA_FFI_DEFINE_HANDLER(kDeviceAllReduce, BlockedDeviceAllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliques>()
                           .Ctx<ffi::CollectiveMemory>());

XLA_FFI_DEFINE_HANDLER(kDelayedDeviceAllReduce, DelayedDeviceAllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliques>()
                           .Ctx<ffi::CollectiveMemory>());

XLA_FFI_DEFINE_HANDLER(kPrepareMulticastAllReduce, PrepareMulticastAllReduce,
                       ffi::Ffi::BindPrepare()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliqueRequests>()
                           .Ctx<ffi::CollectiveMemoryRequests>());

XLA_FFI_DEFINE_HANDLER(kMulticastAllReduce, BlockedMulticastAllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveMemory>());

XLA_FFI_DEFINE_HANDLER(kDelayedMulticastAllReduce, DelayedMulticastAllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveMemory>());

XLA_FFI_DEFINE_HANDLER(kPreparePeerAllReduce, PreparePeerAllReduce,
                       ffi::Ffi::BindPrepare()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveMemoryRequests>());

XLA_FFI_DEFINE_HANDLER(kPeerAllReduce, BlockedPeerAllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveMemory>());

XLA_FFI_DEFINE_HANDLER(kDelayedPeerAllReduce, DelayedPeerAllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveMemory>());

// Register handler bundle for the custom all-reduce operation.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPrepareAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kAllReduce,
                         });

// Register handler bundle for the custom all-reduce operation with
// device-initiated collective kernels that use multimem addresses.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test_blocked_multimem_all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPrepareMulticastAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kMulticastAllReduce,
                         });

// Register handler bundle for the custom all-reduce operation with
// device-initiated collective kernels that use multimem addresses.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test_delayed_multimem_all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPrepareMulticastAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kDelayedMulticastAllReduce,
                         });

// Register handler bundle for the custom all-reduce operation with
// device-initiated collective kernels that use peer addresses.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test_blocked_peer_all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPreparePeerAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kPeerAllReduce,
                         });

// Register handler bundle for the custom all-reduce operation with
// device-initiated collective kernels that use peer addresses.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test_delayed_peer_all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPreparePeerAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kDelayedPeerAllReduce,
                         });

// Register handler bundle for the custom all-reduce operation with
// device-initiated collective kernels that use blocked execution.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test_blocked_device_all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPrepareDeviceAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kDeviceAllReduce,
                         });

// Register handler bundle for the custom all-reduce operation with
// device-initiated collective kernels that use delayed execution.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test_delayed_device_all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPrepareDeviceAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kDelayedDeviceAllReduce,
                         });

TEST_F(CollectiveOpsTestFFI, AllReduce) {
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  if (!IsHopperAndHigher()) {
    GTEST_SKIP() << "NCCL symmetric memory requires Hopper+";
  }

  constexpr absl::string_view hlo_string = R"(
      HloModule m, replica_count=2

      ENTRY test_computation {
        id = u32[] replica-id()
        ROOT all-reduce = u32[] custom-call(id),
          custom_call_target="__xla_test$$all_reduce",
          api_version=API_VERSION_TYPED_FFI
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>(),
                        /*run_hlo_passes=*/false));

  absl::Span<const Literal> results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

class AllReduceTest : public CollectiveOpsTestFFI,
                      public ::testing::WithParamInterface<absl::string_view> {
};

TEST_P(AllReduceTest, DeviceAllReduce) {
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  if (!IsHopperAndHigher()) {
    GTEST_SKIP() << "NCCL symmetric memory requires Hopper+";
  }

  GpuCollectives* collectives = GpuCollectives::Default("CUDA");
  if (!collectives || !collectives->SupportsDeviceComm()) {
    GTEST_SKIP() << "GPU collectives do not support device communication";
  }

  std::string hlo_string = absl::Substitute(R"(
      HloModule m, replica_count=2

      ENTRY test_computation {
        id = u32[] replica-id()
        in = u32[]{:S(1)} copy(id)
        all-reduce = u32[]{:S(1)} custom-call(in),
          custom_call_target="__xla_test_$0_device_all_reduce",
          api_version=API_VERSION_TYPED_FFI
        ROOT out = u32[] copy(all-reduce)
      }
    )",
                                            GetParam());

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>(),
                        /*run_hlo_passes=*/false));
  SynchronizationSignals* signals = global_signals->get();
  signals->finished_kernels_counter.Wait();

  absl::Span<const Literal> results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

TEST_P(AllReduceTest, PeerAllReduce) {
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  if (!IsHopperAndHigher()) {
    GTEST_SKIP() << "Test requires Hopper+ since on a previous platforms there "
                    "are no guarantess that GPUs have direct peer access";
  }

  std::string hlo_string = absl::Substitute(R"(
      HloModule m, replica_count=2

      ENTRY test_computation {
        id = u32[] replica-id()
        ROOT all-reduce = u32[] custom-call(id),
          custom_call_target="__xla_test_$0_peer_all_reduce",
          api_version=API_VERSION_TYPED_FFI
      }
    )",
                                            GetParam());

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>(),
                        /*run_hlo_passes=*/false));
  SynchronizationSignals* signals = global_signals->get();
  signals->finished_kernels_counter.Wait();

  absl::Span<const Literal> results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

TEST_P(AllReduceTest, MulticastAllReduce) {
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  if (!IsHopperAndHigher()) {
    GTEST_SKIP() << "Test requires Hopper+";
  }

  std::string hlo_string = absl::Substitute(R"(
      HloModule m, replica_count=2

      ENTRY test_computation {
        c0 = u32[] constant(1)
        in = u32[]{:S(1)} copy(c0)
        ROOT all-reduce = u32[] custom-call(in),
          custom_call_target="__xla_test_$0_multimem_all_reduce",
          api_version=API_VERSION_TYPED_FFI
      }
    )",
                                            GetParam());

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>(),
                        /*run_hlo_passes=*/false));
  SynchronizationSignals* signals = global_signals->get();
  signals->finished_kernels_counter.Wait();

  absl::Span<const Literal> results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  const uint32_t expected = 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceTests, AllReduceTest, Values("blocked", "delayed"),
    [](const ::testing::TestParamInfo<absl::string_view>& info) {
      return std::string(info.param);
    });

// Same as DeviceAllReduce, but uses frontend_attributes to specify memory
// spaces instead of hardcoded S(1).
TEST_F(CollectiveOpsTestFFI, DeviceAllReduceWithFrontendAttributes) {
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  if (!IsHopperAndHigher()) {
    GTEST_SKIP() << "NCCL symmetric memory requires Hopper+";
  }

  GpuCollectives* collectives = GpuCollectives::Default("CUDA");
  if (!collectives || !collectives->SupportsDeviceComm()) {
    GTEST_SKIP() << "GPU collectives do not support device communication";
  }

  constexpr absl::string_view hlo_string = R"(
      HloModule m, replica_count=2

      ENTRY test_computation {
        id = u32[] replica-id()
        all-reduce = u32[] custom-call(id),
          custom_call_target="__xla_test_blocked_device_all_reduce",
          api_version=API_VERSION_TYPED_FFI,
          frontend_attributes={
            operands_memory_spaces="{0:1}",
            results_memory_spaces="{0:1}"
          }
        ROOT out = u32[] copy(all-reduce)
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>(),
                        /*run_hlo_passes=*/true));
  SynchronizationSignals* signals = global_signals->get();
  signals->finished_kernels_counter.Wait();

  absl::Span<const Literal> results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

}  // namespace xla::gpu
