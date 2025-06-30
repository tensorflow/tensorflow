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

#include "xla/backends/gpu/collectives/nccl_communicator.h"

#include <cstddef>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

constexpr absl::string_view kCudaError = "unhandled cuda error";

void AssertAborted(absl::Status s) {
  ASSERT_THAT(
      s, StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("aborted")));
};

void AssertEventAborted(tsl::AsyncValueRef<Communicator::Event> event) {
  tsl::BlockUntilReady(event);
  ASSERT_TRUE(event.IsError());
  ASSERT_THAT(event.GetError(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("aborted")));
};

// Creates a non-blocking NCCL communicator.
absl::StatusOr<std::unique_ptr<NcclCommunicator>> CreateCommunicator(
    bool blocking) {
  auto f = [blocking]() -> absl::StatusOr<ncclComm_t> {
    // Create a unique NCCL Id.
    ncclUniqueId id;
    TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclGetUniqueId(&id)));

    // Initialize a communicator.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = blocking ? 1 : 0;
    ncclComm_t comm;
    ncclResult_t r =
        ncclCommInitRankConfig(&comm, /*nranks=*/1, id, /*rank=*/0, &config);
    if (r == ncclUnhandledCudaError) {
      // If this test runs on a machine without any CUDA-capable devices
      // available, we get a ncclUnhandledCudaError. We return a specific error
      // and skip the test.
      LOG(ERROR) << XLA_NCCL_STATUS(r);
      return absl::FailedPreconditionError(kCudaError);
    }
    if (r != ncclSuccess && r != ncclInProgress) {
      return XLA_NCCL_STATUS(r);
    }

    // Wait for the communicator to finish initializing.
    ncclResult_t state = ncclInProgress;
    while (state == ncclInProgress) {
      TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclCommGetAsyncError(comm, &state)));
    }
    TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(state));
    return comm;
  };
  bool is_async = !blocking;
  return NcclCommunicator::Create(f, is_async);
}

TEST(NcclCommunicator, AbortSucceeds) {
  for (const bool blocking : {true, false}) {
    absl::StatusOr<std::unique_ptr<NcclCommunicator>> comm =
        CreateCommunicator(blocking);
    if (comm.status().message() == kCudaError) {
      GTEST_SKIP() << "unhandled cuda error";
    }
    ASSERT_THAT(comm, IsOk());
    ASSERT_THAT((*comm)->Abort(), IsOk());
  }
}

TEST(NcclCommunicator, DoubleAbortFails) {
  for (const bool blocking : {true, false}) {
    absl::StatusOr<std::unique_ptr<NcclCommunicator>> comm =
        CreateCommunicator(blocking);
    if (comm.status().message() == kCudaError) {
      GTEST_SKIP() << "unhandled cuda error";
    }
    ASSERT_THAT(comm.status(), IsOk());
    ASSERT_THAT((*comm)->Abort(), IsOk());
    ASSERT_THAT(
        (*comm)->Abort(),
        StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("aborted")));
  }
}

TEST(NcclCommunicator, OperationsFailAfterAbort) {
  for (const bool blocking : {true, false}) {
    // Declare placeholder variables to make the operations below compile.
    se::DeviceMemoryBase buf;
    PrimitiveType dtype = PrimitiveType::U64;
    size_t count = 0;
    ReductionKind rk = ReductionKind::SUM;
    GpuCollectives::Executor executor(nullptr);

    // Execute NcclCommunicator operations. They should all immediately fail
    // because the communicator has been aborted.
    absl::StatusOr<std::unique_ptr<NcclCommunicator>> comm =
        CreateCommunicator(blocking);
    if (comm.status().message() == kCudaError) {
      GTEST_SKIP() << "unhandled cuda error";
    }
    ASSERT_THAT(comm.status(), IsOk());
    ASSERT_THAT((*comm)->Abort(), IsOk());
    AssertAborted((*comm)->HealthCheck());
    AssertAborted((*comm)->NumRanks().status());
    AssertAborted((*comm)->RegisterBuffer(buf).status());
    AssertEventAborted(
        (*comm)->AllReduce(buf, buf, dtype, count, rk, executor));
    AssertEventAborted(
        (*comm)->Broadcast(buf, buf, dtype, count, RankId(0), executor));
    AssertEventAborted(
        (*comm)->ReduceScatter(buf, buf, dtype, count, rk, executor));
    AssertEventAborted((*comm)->AllGather(buf, buf, dtype, count, executor));
    AssertEventAborted((*comm)->AllToAll({}, {}, dtype, count, executor));
    AssertEventAborted(
        (*comm)->CollectivePermute(buf, buf, dtype, count, {}, {}, executor));
    AssertEventAborted((*comm)->Send(buf, dtype, count, RankId(0), executor));
    AssertEventAborted((*comm)->Recv(buf, dtype, count, RankId(0), executor));
  }
}

}  // namespace
}  // namespace xla::gpu
