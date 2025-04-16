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
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/utility/utility.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"

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
using ::tsl::testing::StatusIs;

constexpr absl::string_view kCudaError = "unhandled cuda error";

// Creates a non-blocking NCCL communicator.
absl::StatusOr<NcclCommunicator> CreateNonBlockingCommunicator() {
  // Create a unique NCCL Id.
  ncclUniqueId id;
  TF_RETURN_IF_ERROR(XLA_NCCL_STATUS(ncclGetUniqueId(&id)));

  // Initialize a communicator.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;  // non-blocking
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

  // Wrap and return the communicator.
  return absl::StatusOr<NcclCommunicator>(absl::in_place_t(), comm);
}

TEST(NcclCommunicator, AbortSucceeds) {
  absl::StatusOr<NcclCommunicator> comm = CreateNonBlockingCommunicator();
  if (comm.status().message() == kCudaError) {
    GTEST_SKIP() << "unhandled cuda error";
  }
  TF_ASSERT_OK(comm.status());
  TF_ASSERT_OK(comm->Abort());
}

TEST(NcclCommunicator, DoubleAbortFails) {
  absl::StatusOr<NcclCommunicator> comm = CreateNonBlockingCommunicator();
  if (comm.status().message() == kCudaError) {
    GTEST_SKIP() << "unhandled cuda error";
  }
  TF_ASSERT_OK(comm.status());
  TF_ASSERT_OK(comm->Abort());
  ASSERT_THAT(comm->Abort(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("aborted")));
}

TEST(NcclCommunicator, OperationsFailAfterAbort) {
  auto assert_aborted = [](absl::Status s) {
    ASSERT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                            HasSubstr("aborted")));
  };

  auto assert_event_aborted =
      [](tsl::AsyncValueRef<Communicator::Event> event) {
        tsl::BlockUntilReady(event);
        ASSERT_TRUE(event.IsError());
        ASSERT_THAT(event.GetError(),
                    StatusIs(absl::StatusCode::kFailedPrecondition,
                             HasSubstr("aborted")));
      };

  // Declare placeholder variables to make the operations below compile.
  se::DeviceMemoryBase buf;
  PrimitiveType dtype = PrimitiveType::U64;
  size_t count = 0;
  ReductionKind rk = ReductionKind::SUM;
  GpuCollectives::Executor executor(nullptr);

  // Execute NcclCommunicator operations. They should all immediately fail
  // because the communicator has been aborted.
  absl::StatusOr<NcclCommunicator> comm = CreateNonBlockingCommunicator();
  if (comm.status().message() == kCudaError) {
    GTEST_SKIP() << "unhandled cuda error";
  }
  TF_ASSERT_OK(comm.status());
  TF_ASSERT_OK(comm->Abort());
  assert_aborted(comm->HealthCheck());
  assert_aborted(comm->NumRanks().status());
  assert_aborted(comm->RegisterBuffer(buf).status());
  assert_event_aborted(comm->AllReduce(buf, buf, dtype, count, rk, executor));
  assert_event_aborted(
      comm->Broadcast(buf, buf, dtype, count, RankId(0), executor));
  assert_event_aborted(
      comm->ReduceScatter(buf, buf, dtype, count, rk, executor));
  assert_event_aborted(comm->AllGather(buf, buf, dtype, count, executor));
  assert_event_aborted(comm->AllToAll({}, {}, dtype, count, executor));
  assert_event_aborted(
      comm->CollectivePermute(buf, buf, dtype, count, {}, {}, executor));
  assert_event_aborted(comm->Send(buf, dtype, count, RankId(0), executor));
  assert_event_aborted(comm->Recv(buf, dtype, count, RankId(0), executor));
}

}  // namespace
}  // namespace xla::gpu
