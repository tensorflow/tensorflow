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
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

// Common place for all collective thunks to source nccl/rccl headers.
// Also, all the RunNcclCollective() functions for various thunks should
// use XLA_ENABLE_XCCL to guard use NCCL/RCCL usage (and not use GOOGLE_XCCL).
#if GOOGLE_XCCL
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define XLA_ENABLE_XCCL 1
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // GOOGLE_XCCL

#if XLA_ENABLE_XCCL
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#error "Neither CUDA nor ROCm enabled but NCCL/RCCL enabled"
#endif

// Also include this file required by all collective thunks.
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"

#endif  // XLA_ENABLE_XCCL

struct ncclComm;
using ncclComm_t = ncclComm*;

namespace xla {
namespace gpu {

class NcclClique;

struct NcclCollectiveConfig {
  NcclCollectiveConfig();
  NcclCollectiveConfig(NcclCollectiveConfig&&);
  ~NcclCollectiveConfig();

  NcclCollectiveConfig& operator=(NcclCollectiveConfig&&);

  int64_t operand_count;
  std::vector<PrimitiveType> operand_element_type;
  std::vector<ReplicaGroup> replica_groups;
  RendezvousKey::CollectiveOpKind collective_op_kind;
  int64_t op_id;
  CollectiveOpGroupMode group_mode;

  template <typename OpT>
  void SetCollectiveOpKindAndID(OpT op);
  bool IsDegenerate(int64_t replica_count, int64_t partition_count) const;
};

template <typename OpT>
void NcclCollectiveConfig::SetCollectiveOpKindAndID(OpT op) {
  if (op.channel_id()) {
    collective_op_kind = RendezvousKey::kCrossModule;
    op_id = static_cast<int64_t>(op.channel_id()->handle().getInt());
  } else {
    collective_op_kind = RendezvousKey::kCrossReplica;
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    mlir::IntegerAttr unique_id =
        parent->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");
    op_id = static_cast<int64_t>(unique_id.getInt());
  }
}

template <typename OpT>
NcclCollectiveConfig GetNcclCollectiveConfigForMlir(
    OpT op, absl::optional<bool> use_global_device_ids) {
  NcclCollectiveConfig config;
  config.operand_count = op.operands().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    const Shape shape = GetShape(op.operands()[i]);
    config.operand_element_type.push_back(shape.element_type());
  }
  config.replica_groups =
      ConvertReplicaGroups(op.replica_groups()).ValueOrDie();
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetCollectiveOpGroupMode(op.channel_id().hasValue(),
                                               use_global_device_ids)
                          .ValueOrDie();
  return config;
}

// Thunk base class for NCCL collective operations.
class NcclCollectiveThunk : public Thunk {
 public:
  using Thunk::Thunk;

  struct Buffer {
    int64_t element_count;
    BufferAllocation::Slice source_buffer;
    BufferAllocation::Slice destination_buffer;
  };

  // Returns whether NCCL operations appear possible to perform; e.g. if we
  // haven't done a build with the CUDA compiler enabled, we can't compile the
  // NCCL header, and thus this will be false.
  //
  // When this is false, the ExecuteOnStream() call will simply return a status
  // error.
  static bool NcclIsEnabled();

  Status ExecuteOnStream(const ExecuteParams& params) override;

 protected:
  virtual Status RunNcclCollective(const ExecuteParams& params,
                                   ncclComm_t comm) = 0;
  virtual const NcclCollectiveConfig& config() const = 0;

  // Logging support.
  std::string GetDeviceString(const ExecuteParams& params) const;
};

// Returns if the given data type is supported by NCCL.
// Note: Keep this in sync with ToNcclDataType().
bool IsTypeSupportedByNccl(PrimitiveType element_type);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
