#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poprand_ops.h"

#include <string>
#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {

namespace {
StatusOr<poplar::Tensor> NonAllocatingOp(
    poplar::Graph& graph, CompilerResources& res, const std::string& name,
    const TensorTarget& tensor_target,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map,
    const TensorMap& tensor_map) {
  return xla::FailedPrecondition("Non-allocating op should not be allocating.");
}

StatusOr<poplar::program::Program> CreateTruncatedNormalOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  poplar::program::Program prog;

  TF_ASSIGN_OR_RETURN(prog,
                      TruncatedNormal(res, inst, output_shape, tensor_map));

  poplar::program::Sequence seq;
  seq.add(prog);
  return seq;
}
}  // namespace

const absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo>&
GetPoprandOpInfoMap() {
  static absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo> info_map = {
      {PoplibsOp::TruncatedNormal, {NonAllocatingOp, CreateTruncatedNormalOp}}};
  return info_map;
}

}  // namespace poplarplugin
}  // namespace xla
