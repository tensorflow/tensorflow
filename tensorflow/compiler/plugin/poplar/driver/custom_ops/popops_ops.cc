#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

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

absl::flat_hash_map<std::string, CustomPoplibOpInfo> info_map = {
    {"sqrt", {AllocateUnaryOp, CreateUnaryOp}},
    {"rsqrt", {AllocateUnaryOp, CreateUnaryOp}},
};

StatusOr<popops::expr::UnaryOpType> LookupUnaryFnForCustomOp(
    const HloInstruction* inst) {
  std::vector<std::string> op_info =
      absl::StrSplit(inst->custom_call_target(), "::");
  if (op_info.size() != 2) {
    return xla::FailedPrecondition("Invalid custom poplibs call info: %s",
                                   inst->custom_call_target().c_str());
  }
  std::string op_name = op_info[1];

  if (op_name == "sqrt") {
    return popops::expr::UnaryOpType::SQRT;

  } else if (op_name == "rsqrt") {
    return popops::expr::UnaryOpType::RSQRT;
  }
  return tensorflow::errors::Unknown(
      absl::StrCat("[Poplar] Invalid opcode lookup ", op_name));
}
}  // namespace

const absl::flat_hash_map<std::string, CustomPoplibOpInfo>&
GetPopopsOpInfoMap() {
  return info_map;
}

StatusOr<poplar::Tensor> AllocateUnaryOp(
    poplar::Graph& graph, CompilerResources& res, const std::string& name,
    const HloInstruction* inst, const int64 target_idx,
    const IPUCustomKernelsUtil::AttributeMap& attribute_map) {
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
