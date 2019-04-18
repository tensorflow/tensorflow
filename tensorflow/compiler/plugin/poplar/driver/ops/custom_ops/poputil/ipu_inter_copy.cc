#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poputil/TileMapping.hpp>

namespace xla {
namespace poplarplugin {
namespace {

class IpuInterCopyOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override;
};

StatusOr<poplar::program::Program> IpuInterCopyOp::Creator(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));

  const auto src = inst->operand(0);

  if (!inst->has_sharding()) {
    return xla::FailedPrecondition("Missing shard information on %s",
                                   inst->name());
  }
  if (!src->has_sharding()) {
    return xla::FailedPrecondition("Missing shard information on %s",
                                   src->name());
  }

  const auto& src_sharding = GetShardingDeviceIdVector(src->sharding());
  const auto& dst_sharding = GetShardingDeviceIdVector(inst->sharding());
  if (src_sharding.size() != dst_sharding.size()) {
    return xla::FailedPrecondition("Mismatched sharding info on %s",
                                   inst->name());
  }

  // Should this be done by flattening, concatenating and copying a single
  // tensor?
  for (int index = 0; index < src_sharding.size(); index++) {
    if (src_sharding[index] != dst_sharding[index]) {
      out = poputil::copyToIpu(
          res.main_graph, out, seq, dst_sharding[index], GetDebugName(inst),
          poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, index, out));
    }
  }
  return seq;
}

REGISTER_POPLIBS_OP(Poputil, IpuInterCopy, IpuInterCopyOp);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
