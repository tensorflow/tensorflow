/* Copyright 2026 The OpenXLA Authors.

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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/collectives/collectives_test_util.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/debug_options_flags.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);

struct CapturedGraphInfo {
  std::vector<std::string> kernel_names;
  int num_memcpy_nodes = 0;
  int num_batch_mem_op_nodes = 0;
};

absl::StatusOr<CapturedGraphInfo> CaptureCollectiveGraph(
    se::StreamExecutor* executor, se::Stream* stream,
    absl::FunctionRef<absl::Status()> collective_fn) {
  auto activate = executor->Activate();
  CUstream cu_stream =
      reinterpret_cast<CUstream>(stream->platform_specific_handle().stream);
  if (CUresult res =
          cuStreamBeginCapture(cu_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
      res != CUDA_SUCCESS) {
    return absl::InternalError(
        absl::StrFormat("cuStreamBeginCapture failed: %d", res));
  }
  absl::Status status = collective_fn();
  CUgraph graph = nullptr;
  if (CUresult res = cuStreamEndCapture(cu_stream, &graph);
      res != CUDA_SUCCESS) {
    return absl::InternalError(
        absl::StrFormat("cuStreamEndCapture failed: %d", res));
  }
  absl::Cleanup destroy_graph = [graph] {
    if (graph != nullptr) {
      cuGraphDestroy(graph);
    }
  };
  ABSL_RETURN_IF_ERROR(status);

  size_t num_nodes = 0;
  if (CUresult res = cuGraphGetNodes(graph, nullptr, &num_nodes);
      res != CUDA_SUCCESS) {
    return absl::InternalError(
        absl::StrFormat("cuGraphGetNodes failed: %d", res));
  }
  std::vector<CUgraphNode> nodes(num_nodes);
  if (CUresult res = cuGraphGetNodes(graph, nodes.data(), &num_nodes);
      res != CUDA_SUCCESS) {
    return absl::InternalError(
        absl::StrFormat("cuGraphGetNodes failed: %d", res));
  }

  CapturedGraphInfo info;
  for (CUgraphNode node : nodes) {
    CUgraphNodeType type;
    if (CUresult res = cuGraphNodeGetType(node, &type); res != CUDA_SUCCESS) {
      return absl::InternalError(
          absl::StrFormat("cuGraphNodeGetType failed: %d", res));
    }
    if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params;
      if (CUresult res = cuGraphKernelNodeGetParams(node, &params);
          res != CUDA_SUCCESS) {
        return absl::InternalError(
            absl::StrFormat("cuGraphKernelNodeGetParams failed: %d", res));
      }
      const char* name = nullptr;
      if (CUresult res = cuFuncGetName(&name, params.func);
          res != CUDA_SUCCESS) {
        return absl::InternalError(
            absl::StrFormat("cuFuncGetName failed: %d", res));
      }
      if (name != nullptr) {
        info.kernel_names.emplace_back(name);
      }
    } else if (type == CU_GRAPH_NODE_TYPE_MEMCPY) {
      ++info.num_memcpy_nodes;
    } else if (type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
      ++info.num_batch_mem_op_nodes;
    }
  }
  return info;
}

// Test checks that there are no deadlocks when NCCL API is used concurrently
// from multiple threads.
// Test launches two parallel threads which should use the same NCCL
// communicator asynchronously. The first thread should perform memory
// registration and the second thread should perform AllReduce.
// Memory registration operations should wait for the AllReduce to complete
// using a barrier.
TEST(NcclCommunicatorTest, AsyncApiCalls) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));

  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto sym_allocs, Allocate(allocators, 1024));
  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, 1024));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, 1024));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumThreads);

  constexpr int kNumIterations = 1000;
  std::vector<std::unique_ptr<absl::Barrier>> registration_barriers;
  registration_barriers.reserve(kNumIterations);
  std::vector<std::unique_ptr<absl::Barrier>> all_reduce_barriers;
  all_reduce_barriers.reserve(kNumIterations);
  for (int i = 0; i < kNumIterations; ++i) {
    registration_barriers.push_back(
        std::make_unique<absl::Barrier>(kNumThreads));
    all_reduce_barriers.push_back(std::make_unique<absl::Barrier>(kNumThreads));
  }

  // Register memory, synchronize with barrier, unregister memory
  // asynchronously.
  std::function<absl::Status(int)> memory_registration_fn =
      [&](int rank) -> absl::Status {
    for (int i = 0; i < kNumIterations; ++i) {
      if (i > 0) {
        all_reduce_barriers[i - 1]->Block();
      }
      ABSL_ASSIGN_OR_RETURN(auto sym_mem, comms[rank]->CreateSymmetricMemory(
                                         sym_allocs[rank]->address()));
      registration_barriers[i]->Block();
    }
    all_reduce_barriers[kNumIterations - 1]->Block();
    return absl::OkStatus();
  };

  // Perform AllReduce.
  std::function<absl::Status(int)> all_reduce_fn =
      [&](int rank) -> absl::Status {
    se::Stream* stream = rank == 0 ? stream0.get() : stream1.get();
    GpuCollectives::Executor gpu_exec(stream);
    for (int i = 0; i < kNumIterations; ++i) {
      registration_barriers[i]->Block();
      ABSL_RETURN_IF_ERROR(comms[rank]
                          ->AllReduce(send_allocs[rank]->address(),
                                      recv_allocs[rank]->address(), F32, 256,
                                      ReductionKind::SUM, gpu_exec)
                          .Await());
      all_reduce_barriers[i]->Block();
    }
    return absl::OkStatus();
  };

  tsl::Executor& exec = *pool.AsExecutor();
  std::vector<Future<>> futures;
  futures.reserve(kNumThreads);
  futures.push_back(
      MakeFutureOn(exec, [&]() { return memory_registration_fn(0); }));
  futures.push_back(
      MakeFutureOn(exec, [&]() { return memory_registration_fn(1); }));
  futures.push_back(MakeFutureOn(exec, [&]() { return all_reduce_fn(0); }));
  futures.push_back(MakeFutureOn(exec, [&]() { return all_reduce_fn(1); }));

  for (auto& f : futures) {
    EXPECT_OK(f.Await());
  }
}

TEST(NcclCommunicatorTest, CliqueSemaphore) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));

  // 1. Initially, clique semaphore should not be initialized on either
  // communicator.
  EXPECT_FALSE(comms[0]->IsCliqueSemaphoreInitialized());
  EXPECT_FALSE(comms[1]->IsCliqueSemaphoreInitialized());

  // Querying before initialization should return NotFoundError.
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));

  // 2. Allocate collective memory for semaphores on both devices.
  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(
      auto sem_allocs,
      Allocate(allocators, sizeof(uint32_t) *
                               GpuCommunicator::kMaxCollectiveIds *
                               GpuCommunicator::kMaxCliqueSemaphores));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());
  se::DeviceAddressBase addr0 = sem_allocs[0]->address();
  se::DeviceAddressBase addr1 = sem_allocs[1]->address();
  ASSERT_OK(stream0->MemZero(&addr0, addr0.size()));
  ASSERT_OK(stream1->MemZero(&addr1, addr1.size()));
  ASSERT_OK(stream0->BlockHostUntilDone());
  ASSERT_OK(stream1->BlockHostUntilDone());

  // 3. Create symmetric memory concurrently across both communicators.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumDevices);
  tsl::Executor& exec = *pool.AsExecutor();

  auto symm_futures = CreateSymmetricMemory(exec, comms, sem_allocs);
  ASSERT_OK_AND_ASSIGN(auto sym_mems,
                       AwaitSymmetricMemory(std::move(symm_futures)));

  // 4. Tie and initialize clique semaphores in a scoped TiedAny container.
  {
    tsl::TiedAny tied;
    auto tied_sem0 = tied.Tie(std::move(sem_allocs[0]));
    auto tied_sym0 = tied.Tie(std::move(sym_mems[0]));
    comms[0]->InitializeCliqueSemaphore(std::move(tied_sem0),
                                        std::move(tied_sym0));

    auto tied_sem1 = tied.Tie(std::move(sem_allocs[1]));
    auto tied_sym1 = tied.Tie(std::move(sym_mems[1]));
    comms[1]->InitializeCliqueSemaphore(std::move(tied_sem1),
                                        std::move(tied_sym1));

    // After initialization, both communicators report initialized.
    EXPECT_TRUE(comms[0]->IsCliqueSemaphoreInitialized());
    EXPECT_TRUE(comms[1]->IsCliqueSemaphoreInitialized());

    constexpr uint64_t kSemaphoreSliceBytes =
        GpuCommunicator::kMaxCliqueSemaphores * sizeof(uint32_t);

    // 5. Query valid peer ranks (both self and peer) and collective IDs.
    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_for_rank0_c0,
        comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0));
    EXPECT_NE(sem0_for_rank0_c0.opaque(), nullptr);
    EXPECT_EQ(sem0_for_rank0_c0.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_for_rank0_c1,
        comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/1));
    EXPECT_EQ(
        static_cast<char*>(sem0_for_rank0_c1.opaque()),
        static_cast<char*>(sem0_for_rank0_c0.opaque()) + kSemaphoreSliceBytes);
    EXPECT_EQ(sem0_for_rank0_c1.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_for_rank1,
        comms[0]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0));
    EXPECT_NE(sem0_for_rank1.opaque(), nullptr);
    EXPECT_EQ(sem0_for_rank1.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem1_for_rank0,
        comms[1]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0));
    EXPECT_NE(sem1_for_rank0.opaque(), nullptr);
    EXPECT_EQ(sem1_for_rank0.size(), kSemaphoreSliceBytes);

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem1_for_rank1,
        comms[1]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0));
    EXPECT_NE(sem1_for_rank1.opaque(), nullptr);
    EXPECT_EQ(sem1_for_rank1.size(), kSemaphoreSliceBytes);

    // 6. Query invalid rank out of bounds or out-of-range collective_id.
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(999), /*collective_id=*/0),
                Not(IsOk()));
    EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(999), /*collective_id=*/0),
                Not(IsOk()));
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(
                    RankId(0),
                    /*collective_id=*/GpuCommunicator::kMaxCollectiveIds),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/-1),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 7. After TiedAny is destroyed, the tied references expire and
  // GetCliqueSemaphore should return NotFoundError.
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(comms[1]->GetCliqueSemaphore(RankId(1), /*collective_id=*/0),
              StatusIs(absl::StatusCode::kNotFound));

  // 8. Verify bounds check when the symmetric memory buffer is smaller than
  // required for higher collective IDs (including partial slice overflow).
  constexpr uint64_t kSemaphoreSliceBytes =
      GpuCommunicator::kMaxCliqueSemaphores * sizeof(uint32_t);
  constexpr uint64_t kSmallBufferSize = kSemaphoreSliceBytes + sizeof(uint32_t);
  ASSERT_OK_AND_ASSIGN(auto small_sem_allocs,
                       Allocate(allocators, kSmallBufferSize));
  std::vector<Future<std::unique_ptr<SymmetricMemory>>> small_symm_futures;
  small_symm_futures.reserve(kNumDevices);
  for (int i = 0; i < kNumDevices; ++i) {
    se::DeviceAddressBase small_addr(small_sem_allocs[i]->address().opaque(),
                                     kSmallBufferSize);
    small_symm_futures.push_back(MakeFutureOn(exec, [&, i, small_addr] {
      return comms[i]->CreateSymmetricMemory(small_addr);
    }));
  }
  ASSERT_OK_AND_ASSIGN(auto small_sym_mems,
                       AwaitSymmetricMemory(std::move(small_symm_futures)));
  {
    tsl::TiedAny tied;
    comms[0]->InitializeCliqueSemaphore(
        tied.Tie(std::move(small_sem_allocs[0])),
        tied.Tie(std::move(small_sym_mems[0])));
    comms[1]->InitializeCliqueSemaphore(
        tied.Tie(std::move(small_sem_allocs[1])),
        tied.Tie(std::move(small_sym_mems[1])));

    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase sem0_c0,
        comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/0));
    EXPECT_EQ(sem0_c0.size(), kSemaphoreSliceBytes);

    // collective_id=1 starts within the allocation (offset < size), but the
    // full slice exceeds the allocation size.
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/1),
                StatusIs(absl::StatusCode::kInternal));
    // collective_id=2 starts past the end of the allocation.
    EXPECT_THAT(comms[0]->GetCliqueSemaphore(RankId(0), /*collective_id=*/2),
                StatusIs(absl::StatusCode::kInternal));
  }
}

class NcclCommunicatorDestructorTest : public ::testing::TestWithParam<bool> {};

TEST_P(NcclCommunicatorDestructorTest, DestroyWithVlogEnabled) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, 1));
  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(executors, {kD0}, GetParam()));
  ASSERT_EQ(comms.size(), 1);

  const int old_vlog_level =
      absl::SetVLogLevel("nccl_communicator", /*log_level=*/1);

  auto destroyed = std::make_shared<absl::Notification>();
  std::shared_ptr<GpuCommunicator> comm = std::move(comms[0]);
  std::thread destroyer([destroyed, comm = std::move(comm)]() mutable {
    comm.reset();
    destroyed->Notify();
  });

  const bool finished =
      destroyed->WaitForNotificationWithTimeout(absl::Seconds(60));
  absl::SetVLogLevel("nccl_communicator", old_vlog_level);
  if (finished) {
    destroyer.join();
  } else {
    destroyer.detach();
  }
  EXPECT_TRUE(finished) << "NcclCommunicator destructor deadlocked";
}

INSTANTIATE_TEST_SUITE_P(Blocking, NcclCommunicatorDestructorTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Blocking" : "Async";
                         });

bool IsAtLeastNccl23200() {
  if (NCCL_VERSION_CODE < 23200) {
    return false;
  }
  int runtime_version = 0;
  return ncclGetVersion(&runtime_version) == ncclSuccess &&
         runtime_version >= 23200;
}

TEST(NcclCommunicatorTest, VariadicAllReduceBatchedWithoutSymmetricWindow) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));

  constexpr size_t kCount = 256;
  constexpr size_t kAllocBytes = 4096;
  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto send_allocs0, Allocate(allocators, kAllocBytes));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs0, Allocate(allocators, kAllocBytes));
  ASSERT_OK_AND_ASSIGN(auto send_allocs1, Allocate(allocators, kAllocBytes));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs1, Allocate(allocators, kAllocBytes));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumDevices);
  tsl::Executor& exec = *pool.AsExecutor();

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());
  std::vector<se::Stream*> streams = {stream0.get(), stream1.get()};

  auto capture_grouped_all_reduces =
      [&]() -> absl::StatusOr<std::vector<CapturedGraphInfo>> {
    std::vector<Future<CapturedGraphInfo>> futures;
    futures.reserve(kNumDevices);
    for (int rank = 0; rank < kNumDevices; ++rank) {
      futures.push_back(MakeFutureOn(exec, [&, rank] {
        GpuCollectives::Executor gpu_exec(streams[rank]);
        return CaptureCollectiveGraph(
            executors[rank], streams[rank], [&]() -> absl::Status {
              return comms[rank]
                  ->GroupExecute([&]() -> absl::Status {
                    ABSL_RETURN_IF_ERROR(comms[rank]->LaunchAllReduce(
                        send_allocs0[rank]->address(),
                        recv_allocs0[rank]->address(), F32, kCount,
                        ReductionKind::SUM, gpu_exec, /*is_rma=*/false));
                    return comms[rank]->LaunchAllReduce(
                        send_allocs1[rank]->address(),
                        recv_allocs1[rank]->address(), F32, kCount,
                        ReductionKind::SUM, gpu_exec, /*is_rma=*/false);
                  })
                  .Await();
            });
      }));
    }
    return JoinFutures(absl::MakeSpan(futures)).Await();
  };

  // Without any NCCL_WIN_COLL_SYMMETRIC window registered on `comms`, grouped
  // non-RMA collectives must not force task isolation and should batch into a
  // single NCCL kernel launch.
  ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> before_window,
                       capture_grouped_all_reduces());
  for (const CapturedGraphInfo& info : before_window) {
    EXPECT_EQ(info.kernel_names.size(), 1);
  }

  // Registering and then destroying a symmetric window must restore batching.
  if (executors[0]->CanEnablePeerAccessTo(executors[1]) &&
      executors[0]
          ->GetDeviceDescription()
          .cuda_compute_capability()
          .IsAtLeastHopper() &&
      comms[0]->LsaSize() == kNumDevices) {
    {
      ASSERT_OK_AND_ASSIGN(
          auto sym_mems,
          AwaitSymmetricMemory(CreateSymmetricMemory(
              exec, comms, send_allocs0, /*symmetric_window=*/true)));
    }
    ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> after_window_destroy,
                         capture_grouped_all_reduces());
    for (const CapturedGraphInfo& info : after_window_destroy) {
      EXPECT_EQ(info.kernel_names.size(), 1);
    }
  }
}

TEST(NcclCommunicatorTest, SymmetricKernelsGuardedByIsRma) {
  if (!IsAtLeastNccl23200()) {
    GTEST_SKIP() << "Test requires NCCL 2.32 or newer";
  }

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executors, {kD0, kD1},
                                                       /*blocking=*/true));
  if (comms[0]->LsaSize() != kNumDevices) {
    GTEST_SKIP() << "Test requires all ranks in the same LSA team (NVLink)";
  }

  constexpr size_t kCount = 256;
  constexpr size_t kAllocBytes = 4096;
  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, kAllocBytes));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, kAllocBytes));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumDevices);
  tsl::Executor& exec = *pool.AsExecutor();

  ASSERT_OK_AND_ASSIGN(
      auto send_sym_mems,
      AwaitSymmetricMemory(CreateSymmetricMemory(exec, comms, send_allocs,
                                                 /*symmetric_window=*/true)));
  ASSERT_OK_AND_ASSIGN(
      auto recv_sym_mems,
      AwaitSymmetricMemory(CreateSymmetricMemory(exec, comms, recv_allocs,
                                                 /*symmetric_window=*/true)));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());
  std::vector<se::Stream*> streams = {stream0.get(), stream1.get()};

  auto capture_all_reduce =
      [&](bool is_rma) -> absl::StatusOr<std::vector<CapturedGraphInfo>> {
    std::vector<Future<CapturedGraphInfo>> futures;
    futures.reserve(kNumDevices);
    for (int rank = 0; rank < kNumDevices; ++rank) {
      futures.push_back(MakeFutureOn(exec, [&, rank, is_rma] {
        GpuCollectives::Executor gpu_exec(streams[rank]);
        return CaptureCollectiveGraph(
            executors[rank], streams[rank], [&]() -> absl::Status {
              return comms[rank]
                  ->GroupExecute([&]() {
                    return comms[rank]->LaunchAllReduce(
                        send_allocs[rank]->address(),
                        recv_allocs[rank]->address(), F32, kCount,
                        ReductionKind::SUM, gpu_exec, is_rma);
                  })
                  .Await();
            });
      }));
    }
    return JoinFutures(absl::MakeSpan(futures)).Await();
  };

  ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> ar_no_rma,
                       capture_all_reduce(/*is_rma=*/false));
  for (const CapturedGraphInfo& info : ar_no_rma) {
    EXPECT_THAT(info.kernel_names, Not(IsEmpty()));
    EXPECT_THAT(info.kernel_names, Each(HasSubstr("ncclDevKernel_")));
    EXPECT_THAT(info.kernel_names,
                Not(Contains(HasSubstr("ncclSymkDevKernel_"))));
  }

  ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> ar_with_rma,
                       capture_all_reduce(/*is_rma=*/true));
  for (const CapturedGraphInfo& info : ar_with_rma) {
    EXPECT_THAT(info.kernel_names, Contains(HasSubstr("ncclSymkDevKernel_")));
  }

  auto capture_all_gather =
      [&](bool is_rma) -> absl::StatusOr<std::vector<CapturedGraphInfo>> {
    std::vector<Future<CapturedGraphInfo>> futures;
    futures.reserve(kNumDevices);
    for (int rank = 0; rank < kNumDevices; ++rank) {
      futures.push_back(MakeFutureOn(exec, [&, rank, is_rma] {
        GpuCollectives::Executor gpu_exec(streams[rank]);
        return CaptureCollectiveGraph(
            executors[rank], streams[rank], [&]() -> absl::Status {
              return comms[rank]
                  ->GroupExecute([&]() {
                    return comms[rank]->LaunchAllGather(
                        send_allocs[rank]->address(),
                        recv_allocs[rank]->address(), F32, kCount, gpu_exec,
                        is_rma);
                  })
                  .Await();
            });
      }));
    }
    return JoinFutures(absl::MakeSpan(futures)).Await();
  };

  ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> ag_no_rma,
                       capture_all_gather(/*is_rma=*/false));
  for (const CapturedGraphInfo& info : ag_no_rma) {
    EXPECT_THAT(info.kernel_names, Not(IsEmpty()));
    EXPECT_THAT(info.kernel_names, Each(HasSubstr("ncclDevKernel_")));
    EXPECT_THAT(info.kernel_names,
                Not(Contains(HasSubstr("ncclSymkDevKernel_"))));
  }

  ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> ag_with_rma,
                       capture_all_gather(/*is_rma=*/true));
  for (const CapturedGraphInfo& info : ag_with_rma) {
    EXPECT_THAT(info.kernel_names, Contains(HasSubstr("ncclSymkDevKernel_")));
  }
}

TEST(NcclCommunicatorTest, AllGatherCopyEngineGuardedByIsRma) {
  if (!IsAtLeastNccl23200()) {
    GTEST_SKIP() << "Test requires NCCL 2.32 or newer";
  }

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  constexpr int kNumDevices = 2;
  ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                       CreateExecutors(platform, kNumDevices));

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires peer access between devices";
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  std::optional<std::string> original_xla_flags;
  if (const char* env = std::getenv("XLA_FLAGS")) {
    original_xla_flags = env;
  }
  const std::string test_xla_flags =
      original_xla_flags.has_value() && !original_xla_flags->empty()
          ? absl::StrCat(
                *original_xla_flags,
                " --xla_gpu_experimental_enable_nccl_symmetric_buffers=true")
          : "--xla_gpu_experimental_enable_nccl_symmetric_buffers=true";
  (void)GetDebugOptionsFromFlags();
  tsl::setenv("XLA_FLAGS", test_xla_flags.c_str(), /*overwrite=*/1);
  ResetFlagValues();
  ParseDebugOptionFlagsFromEnv(/*reset_envvar=*/true);
  absl::Cleanup restore_xla_flags = [&] {
    if (original_xla_flags.has_value()) {
      tsl::setenv("XLA_FLAGS", original_xla_flags->c_str(), /*overwrite=*/1);
    } else {
      tsl::unsetenv("XLA_FLAGS");
    }
    ResetFlagValues();
    ParseDebugOptionFlagsFromEnv(/*reset_envvar=*/true);
  };

  ASSERT_OK_AND_ASSIGN(
      auto comms, CreateCommunicators(executors, {kD0, kD1}, /*blocking=*/true,
                                      /*num_ids=*/1, /*custom_backend=*/nullptr,
                                      /*use_minimal_resource=*/true));
  if (comms[0]->LsaSize() != kNumDevices) {
    GTEST_SKIP() << "Test requires all ranks in the same LSA team (NVLink)";
  }

  constexpr size_t kCount = 256;
  constexpr size_t kAllocBytes = 4096;
  ASSERT_OK_AND_ASSIGN(auto allocators, CreateMemoryAllocators(executors));
  ASSERT_OK_AND_ASSIGN(auto send_allocs, Allocate(allocators, kAllocBytes));
  ASSERT_OK_AND_ASSIGN(auto recv_allocs, Allocate(allocators, kAllocBytes));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl_test", kNumDevices);
  tsl::Executor& exec = *pool.AsExecutor();

  ASSERT_OK_AND_ASSIGN(
      auto send_sym_mems,
      AwaitSymmetricMemory(CreateSymmetricMemory(exec, comms, send_allocs,
                                                 /*symmetric_window=*/true)));
  ASSERT_OK_AND_ASSIGN(
      auto recv_sym_mems,
      AwaitSymmetricMemory(CreateSymmetricMemory(exec, comms, recv_allocs,
                                                 /*symmetric_window=*/true)));

  ASSERT_OK_AND_ASSIGN(auto stream0, executors[0]->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto stream1, executors[1]->CreateStream());
  std::vector<se::Stream*> streams = {stream0.get(), stream1.get()};

  auto capture_all_gather =
      [&](bool is_rma) -> absl::StatusOr<std::vector<CapturedGraphInfo>> {
    std::vector<Future<CapturedGraphInfo>> futures;
    futures.reserve(kNumDevices);
    for (int rank = 0; rank < kNumDevices; ++rank) {
      futures.push_back(MakeFutureOn(exec, [&, rank, is_rma] {
        GpuCollectives::Executor gpu_exec(streams[rank]);
        return CaptureCollectiveGraph(
            executors[rank], streams[rank], [&]() -> absl::Status {
              return comms[rank]
                  ->GroupExecute([&]() {
                    return comms[rank]->LaunchAllGather(
                        send_allocs[rank]->address(),
                        recv_allocs[rank]->address(), F32, kCount, gpu_exec,
                        is_rma);
                  })
                  .Await();
            });
      }));
    }
    return JoinFutures(absl::MakeSpan(futures)).Await();
  };

  ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> ag_no_rma,
                       capture_all_gather(/*is_rma=*/false));
  for (const CapturedGraphInfo& info : ag_no_rma) {
    EXPECT_THAT(info.kernel_names, Not(IsEmpty()));
    EXPECT_THAT(info.kernel_names, Each(HasSubstr("ncclDevKernel_")));
    EXPECT_THAT(info.kernel_names,
                Not(Contains(HasSubstr("ncclSymkDevKernel_"))));
    EXPECT_EQ(info.num_memcpy_nodes, 0);
    EXPECT_EQ(info.num_batch_mem_op_nodes, 0);
  }

  ASSERT_OK_AND_ASSIGN(std::vector<CapturedGraphInfo> ag_with_rma,
                       capture_all_gather(/*is_rma=*/true));
  for (const CapturedGraphInfo& info : ag_with_rma) {
    EXPECT_THAT(info.kernel_names, IsEmpty());
    EXPECT_THAT(info.num_memcpy_nodes, Gt(0));
    EXPECT_THAT(info.num_batch_mem_op_nodes, Gt(0));
  }
}

}  // namespace
}  // namespace xla::gpu
