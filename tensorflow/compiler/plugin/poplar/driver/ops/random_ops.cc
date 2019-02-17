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

static StatusOr<double> DoubleValueOfScalarLiteral(const xla::Literal& lit) {
  if (ShapeUtil::ElementsIn(lit.shape()) != 1) {
    return xla::FailedPrecondition("Literal element count != 1");
  }

  Literal double_lit;
  TF_ASSIGN_OR_RETURN(double_lit, lit.Convert(F64));

  const double* val = static_cast<const double*>(double_lit.untyped_data());
  return *val;
}

StatusOr<poplar::program::Program> TruncatedNormal(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, std::make_pair(inst, 0),
                                     output_shape, res, tensor_map));

  poplar::program::Sequence seq;
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  res.random.truncatedNormal(graph, out, 0.0, 1.0, 1.0, seq,
                             GetDebugName(inst));

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

  double mean1_val;
  TF_ASSIGN_OR_RETURN(mean1_val, DoubleValueOfScalarLiteral(mean1->literal()));
  double mean2_val;
  TF_ASSIGN_OR_RETURN(mean2_val, DoubleValueOfScalarLiteral(mean2->literal()));
  double sd1_val;
  TF_ASSIGN_OR_RETURN(sd1_val, DoubleValueOfScalarLiteral(sd1->literal()));
  double sd2_val;
  TF_ASSIGN_OR_RETURN(sd2_val, DoubleValueOfScalarLiteral(sd2->literal()));

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, std::make_pair(inst, 0),
                                     output_shape, res, tensor_map));

  poplar::program::Sequence seq;
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  res.random.normal(graph, out, mean1_val + mean2_val, sd1_val * sd2_val, seq,
                    GetDebugName(inst));

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

  double shift_val;
  TF_ASSIGN_OR_RETURN(shift_val, DoubleValueOfScalarLiteral(shift->literal()));
  double scale_val;
  TF_ASSIGN_OR_RETURN(scale_val, DoubleValueOfScalarLiteral(scale->literal()));
  double lower_val;
  TF_ASSIGN_OR_RETURN(lower_val, DoubleValueOfScalarLiteral(lower->literal()));
  double upper_val;
  TF_ASSIGN_OR_RETURN(upper_val, DoubleValueOfScalarLiteral(upper->literal()));

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, std::make_pair(inst, 0),
                                     output_shape, res, tensor_map));

  poplar::program::Sequence seq;
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  res.random.uniform(graph, out, lower_val * scale_val + shift_val,
                     upper_val * scale_val + shift_val, seq,
                     GetDebugName(inst));

  return seq;
}

StatusOr<poplar::program::Program> RandomNormal(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* mean = inst->operand(0);
  const HloInstruction* sd = inst->operand(1);

  double mean_val;
  TF_ASSIGN_OR_RETURN(mean_val, DoubleValueOfScalarLiteral(mean->literal()));
  double sd_val;
  TF_ASSIGN_OR_RETURN(sd_val, DoubleValueOfScalarLiteral(sd->literal()));

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, std::make_pair(inst, 0),
                                     output_shape, res, tensor_map));

  poplar::program::Sequence seq;
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  res.random.normal(graph, out, mean_val, sd_val, seq, GetDebugName(inst));

  return seq;
}

StatusOr<poplar::program::Program> RandomUniform(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output_shape,
                                                 TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* lower = inst->operand(0);
  const HloInstruction* upper = inst->operand(1);

  double lower_val;
  TF_ASSIGN_OR_RETURN(lower_val, DoubleValueOfScalarLiteral(lower->literal()));
  double upper_val;
  TF_ASSIGN_OR_RETURN(upper_val, DoubleValueOfScalarLiteral(upper->literal()));

  if (ShapeUtil::ElementIsIntegral(output_shape)) {
    upper_val -= 1.0;
  }

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, std::make_pair(inst, 0),
                                     output_shape, res, tensor_map));

  poplar::program::Sequence seq;
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  res.random.uniform(graph, out, lower_val, upper_val, seq, GetDebugName(inst));

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
