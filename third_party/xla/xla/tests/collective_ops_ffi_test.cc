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
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/communicator.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/collective_ops_e2e_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {

class CollectiveOpsTestFFI : public CollectiveOpsE2ETestBase {
 public:
  CollectiveOpsTestFFI()
      : CollectiveOpsE2ETestBase(/*memory_size=*/1 * kMB,
                                 /*collectives_memory_size=*/0) {}
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

  // Ask XLA:GPU runtime to acquire a clique for this key. Later we will be able
  // to get access to it from the execute handler.
  TF_RETURN_IF_ERROR(clique_requests->RequestClique(clique_key));

  return absl::OkStatus();
}

// FFI handler that uses XLA:GPU collectives API to perform an all reduce. This
// is just a test that demonstrates how to use XLA:GPU collectives API in an FFI
// handler, builtin all-reduce is a much better option.
static absl::Status AllReduce(se::Stream* stream, ffi::BufferR0<U32> src,
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

  Future<> future = comm->AllReduce(
      src.device_memory(), dst->device_memory(), src.element_type(),
      src.element_count(), ReductionKind::SUM, GpuCollectives::On(*stream));
  return future.Await();
}

XLA_FFI_DEFINE_HANDLER(kPrepareAllReduce, PrepareAllReduce,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliqueRequests>());

XLA_FFI_DEFINE_HANDLER(kAllReduce, AllReduce,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::BufferR0<U32>>()  // src
                           .Ret<ffi::BufferR0<U32>>()  // dst
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveCliques>());

// Register handler bundle for the custom all-reduce operation.
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$all_reduce", "gpu",
                         XLA_FFI_Handler_Bundle{
                             /*instantiate=*/nullptr,
                             /*prepare=*/kPrepareAllReduce,
                             /*initialize=*/nullptr,
                             /*execute=*/kAllReduce,
                         });

TEST_F(CollectiveOpsTestFFI, AllReduce) {
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  constexpr absl::string_view hlo_string = R"(
      HloModule m

      ENTRY test_computation {
        id = u32[] replica-id()
        ROOT all-reduce = u32[] custom-call(id),
          custom_call_target="__xla_test$$all_reduce",
          api_version=API_VERSION_TYPED_FFI
      }
    )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  absl::Span<const Literal> results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

}  // namespace xla::gpu
