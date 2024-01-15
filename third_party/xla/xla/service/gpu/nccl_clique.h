/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_NCCL_CLIQUE_H_
#define XLA_SERVICE_GPU_NCCL_CLIQUE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/gpu/nccl_types.h"
#include "xla/service/lockable.h"
#include "tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// NCCL clique (collective clique) is a set of devices that execute collective
// operations (e.g. all-reduce). It is notoriously easy to misuse NCCL
// communicators (see link below) and get a dead lock at run time, so in XLA we
// take extra care to order all collective operations in a way that would not
// lead to a deadlock.
//
// We rely on exclusive access to a NCCL clique (using Lockable<T> mechanism) to
// guarantee that only a set of threads executing a particular collective
// operation can schedule new work using communicators belonging to a clique.
//
// In XLA process we have multiple cliques for different combinations of
// participating devices and properties of collective operations launched on
// them, e.g. mixing NCCL operations launched from CUDA graphs with regularly
// launched operations is prone to dead locks, and we keep them separate. See
// NcclCliqueKey for details.
//
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently

//===----------------------------------------------------------------------===//
// NcclUniqueId
//===----------------------------------------------------------------------===//

// Returns true if the NCCL config is global (NCCL_COMM_ID env variable is set).
bool IsGlobalNcclConfig();

// Returns a clique id callback passed as an argument if it's not null or a
// default callback to get create a clique id if we are running in local mode.
absl::StatusOr<const NcclCliqueIdCallback*> GetNcclCliqueIdCallback(
    const NcclCliqueIdCallback* clique_id_callback,  // may be null
    bool is_local);

//===----------------------------------------------------------------------===//
// NcclComm
//===----------------------------------------------------------------------===//

TSL_LIB_GTL_DEFINE_INT_TYPE(OpId, int64_t);

struct NcclComm : public Lockable<NcclCommHandle> {
  explicit NcclComm(NcclCommHandle comm) : Lockable(comm) {}
};

// Acquires an exclusive access to NCCL communicator owned by a NCCL clique.
absl::StatusOr<NcclComm::Lock> AcquireNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclCliqueIdCallback& clique_id_callback, int32_t rank,
    int64_t stream_id, bool enable_clique_optimization);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_NCCL_CLIQUE_H_
