/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

struct ncclComm;
using ncclComm_t = ncclComm*;

namespace xla {
namespace gpu {

struct NcclClique;

struct NcclCollectiveConfig {
  NcclCollectiveConfig();
  NcclCollectiveConfig(NcclCollectiveConfig&&);
  ~NcclCollectiveConfig();

  NcclCollectiveConfig& operator=(NcclCollectiveConfig&&);

  int64 operand_count;
  std::vector<PrimitiveType> operand_element_type;
  int64 replica_count;
  std::vector<ReplicaGroup> replica_groups;
  RendezvousKey::CollectiveOpKind collective_op_kind;
  int64 op_id;
};

NcclCollectiveConfig GetNcclCollectiveConfig(const HloInstruction* hlo,
                                             int64 replica_count);

// Thunk base class for NCCL collective operations.
class NcclCollectiveThunk : public Thunk {
 public:
  using Thunk::Thunk;

  // Returns whether NCCL operations appear possible to perform; e.g. if we
  // haven't done a build with the CUDA compiler enabled, we can't compile the
  // NCCL header, and thus this will be false.
  //
  // When this is false, the ExecuteOnStream() call will simply return a status
  // error.
  static bool NcclIsEnabled();

  Status ExecuteOnStream(const ExecuteParams& params) override
      ABSL_LOCKS_EXCLUDED(mu_);

 protected:
  virtual Status RunNcclCollective(const ExecuteParams& params,
                                   ncclComm_t comm) = 0;
  virtual const NcclCollectiveConfig& config() const = 0;

 private:
  // Creating NCCL cliques is expensive, so we cache them.
  absl::Mutex mu_;
  absl::flat_hash_set<std::shared_ptr<NcclClique>> cliques_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
