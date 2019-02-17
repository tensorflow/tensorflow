#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popops_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include <popops/ElementWise.hpp>
#include <string>

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<popops::expr::UnaryOpType> LookupUnaryFnForCustomOp(
    const HloInstruction* inst) {
  PoplibsLib poplibs_lib;
  PoplibsOp poplibs_op;
  auto ret = GetPoplibsCustomOp(inst);
  if (ret == absl::nullopt) {
    return tensorflow::errors::Unknown(absl::StrCat(
        "[Poplar] Invalid opcode lookup on instruction ", inst->name()));
  }
  std::tie(poplibs_lib, poplibs_op) = ret.value();

  switch (poplibs_op) {
    case PoplibsOp::Sqrt:
      return popops::expr::UnaryOpType::SQRT;
    case PoplibsOp::Rsqrt:
      return popops::expr::UnaryOpType::RSQRT;
    default:
      return tensorflow::errors::Unknown(absl::StrCat(
          "[Poplar] Invalid opcode lookup ", PoplibsOpToString(poplibs_op)));
  }
}
}  // namespace

const absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo>& GetPopopsOpInfoMap() {
  static absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo> info_map = {
      {PoplibsOp::Sqrt, {AllocateUnaryOp, CreateUnaryOp}},
      {PoplibsOp::Rsqrt, {AllocateUnaryOp, CreateUnaryOp}},
  };
  return info_map;
}

StatusOr<poplar::Tensor> AllocateUnaryOp(
    poplar::Graph& graph, CompilerResources& res, const std::string& name,
    const TensorTarget& tensor_target,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map,
    const TensorMap& tensor_map) {
  return xla::FailedPrecondition("UnaryOp should not be allocating.");
}

StatusOr<poplar::program::Program> CreateUnaryOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
  VLOG(1) << "Processing " << inst->name() << " as a unary op.";

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, res, inst, seq, false));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor in = inputs[0][0];

  TF_ASSIGN_OR_RETURN(popops::expr::UnaryOpType op,
                      LookupUnaryFnForCustomOp(inst));

  popops::mapInPlace(graph, op, in, seq, GetDebugName(inst));

  TF_ASSIGN_OR_RETURN(in, BroadcastTensor(in, output_shape));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, in));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
