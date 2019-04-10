#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include <poprand/RandomGen.hpp>

using tensorflow::strings::StrCat;

namespace xla {
namespace poplarplugin {
namespace {
inline const HloInstruction* LookThroughBroadcast(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kBroadcast ? inst->operand(0) : inst;
}
}  // namespace

StatusOr<poplar::program::Program> TruncatedNormal(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor ref,
      AddTensor(graph, std::make_pair(inst, 0), output_shape, res, tensor_map));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  auto out = poprand::truncatedNormal(graph, nullptr, 0, ref, dtype, 0.0, 1.0,
                                      1.0, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> RandomNormalScale(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* root =
      inst->fused_instructions_computation()->root_instruction();
  const HloInstruction* mean1 = LookThroughBroadcast(root->operand(1));
  CHECK_EQ(mean1->opcode(), HloOpcode::kConstant);
  const HloInstruction* sd1 =
      LookThroughBroadcast(root->operand(0)->operand(1));
  CHECK_EQ(sd1->opcode(), HloOpcode::kConstant);
  const HloInstruction* mean2 = root->operand(0)->operand(0)->operand(0);
  CHECK_EQ(mean2->opcode(), HloOpcode::kConstant);
  const HloInstruction* sd2 = root->operand(0)->operand(0)->operand(1);
  CHECK_EQ(sd2->opcode(), HloOpcode::kConstant);

  TF_ASSIGN_OR_RETURN(double mean1_val,
                      LiteralScalarToNativeType<double>(mean1->literal()));
  TF_ASSIGN_OR_RETURN(double mean2_val,
                      LiteralScalarToNativeType<double>(mean2->literal()));
  TF_ASSIGN_OR_RETURN(double sd1_val,
                      LiteralScalarToNativeType<double>(sd1->literal()));
  TF_ASSIGN_OR_RETURN(double sd2_val,
                      LiteralScalarToNativeType<double>(sd2->literal()));

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor ref,
      AddTensor(graph, std::make_pair(inst, 0), output_shape, res, tensor_map));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  auto out =
      poprand::normal(graph, nullptr, 0, ref, dtype, mean1_val + mean2_val,
                      sd1_val * sd2_val, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> RandomUniformScale(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* root =
      inst->fused_instructions_computation()->root_instruction();
  const HloInstruction* shift = LookThroughBroadcast(root->operand(1));
  CHECK_EQ(shift->opcode(), HloOpcode::kConstant);
  const HloInstruction* scale =
      LookThroughBroadcast(root->operand(0)->operand(1));
  CHECK_EQ(scale->opcode(), HloOpcode::kConstant);
  const HloInstruction* lower = root->operand(0)->operand(0)->operand(0);
  CHECK_EQ(lower->opcode(), HloOpcode::kConstant);
  const HloInstruction* upper = root->operand(0)->operand(0)->operand(1);
  CHECK_EQ(upper->opcode(), HloOpcode::kConstant);

  TF_ASSIGN_OR_RETURN(double shift_val,
                      LiteralScalarToNativeType<double>(shift->literal()));
  TF_ASSIGN_OR_RETURN(double scale_val,
                      LiteralScalarToNativeType<double>(scale->literal()));
  TF_ASSIGN_OR_RETURN(double lower_val,
                      LiteralScalarToNativeType<double>(lower->literal()));
  TF_ASSIGN_OR_RETURN(double upper_val,
                      LiteralScalarToNativeType<double>(upper->literal()));

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor ref,
      AddTensor(graph, std::make_pair(inst, 0), output_shape, res, tensor_map));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  auto out = poprand::uniform(
      graph, nullptr, 0, ref, dtype, lower_val * scale_val + shift_val,
      upper_val * scale_val + shift_val, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> RandomNormal(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* mean = inst->operand(0);
  const HloInstruction* sd = inst->operand(1);

  TF_ASSIGN_OR_RETURN(double mean_val,
                      LiteralScalarToNativeType<double>(mean->literal()));
  TF_ASSIGN_OR_RETURN(double sd_val,
                      LiteralScalarToNativeType<double>(sd->literal()));

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor ref,
      AddTensor(graph, std::make_pair(inst, 0), output_shape, res, tensor_map));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  auto out = poprand::normal(graph, nullptr, 0, ref, dtype, mean_val, sd_val,
                             seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> RandomUniform(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output_shape,
                                                 TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* lower = inst->operand(0);
  const HloInstruction* upper = inst->operand(1);

  TF_ASSIGN_OR_RETURN(double lower_val,
                      LiteralScalarToNativeType<double>(lower->literal()));
  TF_ASSIGN_OR_RETURN(double upper_val,
                      LiteralScalarToNativeType<double>(upper->literal()));

  if (ShapeUtil::ElementIsIntegral(output_shape)) {
    upper_val -= 1.0;
  }

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor ref,
      AddTensor(graph, std::make_pair(inst, 0), output_shape, res, tensor_map));

  TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  auto out = poprand::uniform(graph, nullptr, 0, ref, dtype, lower_val,
                              upper_val, seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
