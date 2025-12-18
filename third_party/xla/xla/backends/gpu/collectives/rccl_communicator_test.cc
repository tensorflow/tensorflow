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

#include "xla/backends/gpu/collectives/rccl_communicator.h"

#include <cstddef>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/rccl_errors.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/errors.h"

#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;

constexpr absl::string_view kCudaError = "unhandled cuda error";

void AssertAborted(absl::Status s) {
  ASSERT_THAT(s, absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                        HasSubstr("aborted")));
};

void AssertEventAborted(Future<> future) {
  ASSERT_THAT(future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     HasSubstr("aborted")));
};

// Creates a non-blocking NCCL communicator.
absl::StatusOr<std::unique_ptr<RcclCommunicator>> CreateCommunicator(
    bool blocking) {
  auto f = [blocking]() -> absl::StatusOr<ncclComm_t> {
    // Create a unique NCCL Id.
    ncclUniqueId id;
    TF_RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclGetUniqueId(&id)));

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
      LOG(ERROR) << XLA_RCCL_STATUS(r);
      return absl::FailedPreconditionError(kCudaError);
    }
    if (r != ncclSuccess && r != ncclInProgress) {
      return XLA_RCCL_STATUS(r);
    }

    // Wait for the communicator to finish initializing.
    ncclResult_t state = ncclInProgress;
    while (state == ncclInProgress) {
      TF_RETURN_IF_ERROR(XLA_RCCL_STATUS(ncclCommGetAsyncError(comm, &state)));
    }
    TF_RETURN_IF_ERROR(XLA_RCCL_STATUS(state));
    return comm;
  };
  bool is_async = !blocking;
  return RcclCommunicator::Create(f, is_async);
}

TEST(RcclCommunicator, AbortSucceeds) {
  for (const bool blocking : {true, false}) {
    absl::StatusOr<std::unique_ptr<RcclCommunicator>> comm =
        CreateCommunicator(blocking);
    if (comm.status().message() == kCudaError) {
      GTEST_SKIP() << "unhandled cuda error";
    }
    ASSERT_THAT(comm, absl_testing::IsOk());
    ASSERT_THAT((*comm)->Abort(), absl_testing::IsOk());
  }
}

TEST(RcclCommunicator, DoubleAbortFails) {
  for (const bool blocking : {true, false}) {
    absl::StatusOr<std::unique_ptr<RcclCommunicator>> comm =
        CreateCommunicator(blocking);
    if (comm.status().message() == kCudaError) {
      GTEST_SKIP() << "unhandled cuda error";
    }
    ASSERT_THAT(comm.status(), absl_testing::IsOk());
    ASSERT_THAT((*comm)->Abort(), absl_testing::IsOk());
    ASSERT_THAT((*comm)->Abort(),
                absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                       HasSubstr("aborted")));
  }
}

TEST(RcclCommunicator, OperationsFailAfterAbort) {
  for (const bool blocking : {true, false}) {
    // Declare placeholder variables to make the operations below compile.
    se::DeviceAddressBase buf;
    PrimitiveType dtype = PrimitiveType::U64;
    size_t count = 0;
    ReductionKind rk = ReductionKind::SUM;
    GpuCollectives::Executor executor(nullptr);

    // Execute RcclCommunicator operations. They should all immediately fail
    // because the communicator has been aborted.
    absl::StatusOr<std::unique_ptr<RcclCommunicator>> comm =
        CreateCommunicator(blocking);
    if (comm.status().message() == kCudaError) {
      GTEST_SKIP() << "unhandled cuda error";
    }
    ASSERT_THAT(comm.status(), absl_testing::IsOk());
    ASSERT_THAT((*comm)->Abort(), absl_testing::IsOk());
    AssertAborted((*comm)->HealthCheck());
    AssertAborted((*comm)->NumRanks().status());
    AssertAborted((*comm)->RegisterBufferOnce(buf, 0, false));
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
