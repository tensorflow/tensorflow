#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/strcat.h"

#include <poprand/RandomGen.hpp>

namespace xla {
namespace poplarplugin {

static StatusOr<double> DoubleValueOfScalarLiteral(const xla::Literal& lit) {
  if (ShapeUtil::ElementsIn(lit.shape()) != 1) {
    return xla::FailedPrecondition("Literal element count != 1");
  }

  std::unique_ptr<Literal> double_lit;
  TF_ASSIGN_OR_RETURN(double_lit, lit.Convert(F64));

  const double* val = static_cast<const double*>(double_lit->untyped_data());
  return *val;
}

StatusOr<poplar::program::Program> TruncatedNormalScale(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  const HloInstruction* root = inst->to_apply()->root_instruction();
  const HloInstruction* mean = root->operand(1);
  const HloInstruction* sd = root->operand(0)->operand(1);

  double mean_val;
  TF_ASSIGN_OR_RETURN(mean_val, DoubleValueOfScalarLiteral(mean->literal()));
  double sd_val;
  TF_ASSIGN_OR_RETURN(sd_val, DoubleValueOfScalarLiteral(sd->literal()));

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      out, AddTensor(graph, std::make_pair(inst, 0), output_shape, res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  poplar::program::Sequence seq;
  res.random.truncatedNormal(graph, out, mean_val, sd_val, 1.0, seq,
                             GetDebugName(inst));

  return seq;
}

StatusOr<poplar::program::Program> TruncatedNormal(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      out, AddTensor(graph, std::make_pair(inst, 0), output_shape, res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  poplar::program::Sequence seq;
  res.random.truncatedNormal(graph, out, 0.0, 1.0, 1.0, seq,
                             GetDebugName(inst));

  return seq;
}

StatusOr<poplar::program::Program> RandomNormalScale(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  const HloInstruction* root = inst->to_apply()->root_instruction();
  const HloInstruction* mean1 = root->operand(1);
  const HloInstruction* sd1 = root->operand(0)->operand(1);
  const HloInstruction* mean2 = root->operand(0)->operand(0)->operand(0);
  const HloInstruction* sd2 = root->operand(0)->operand(0)->operand(1);

  double mean1_val;
  TF_ASSIGN_OR_RETURN(mean1_val, DoubleValueOfScalarLiteral(mean1->literal()));
  double mean2_val;
  TF_ASSIGN_OR_RETURN(mean2_val, DoubleValueOfScalarLiteral(mean2->literal()));
  double sd1_val;
  TF_ASSIGN_OR_RETURN(sd1_val, DoubleValueOfScalarLiteral(sd1->literal()));
  double sd2_val;
  TF_ASSIGN_OR_RETURN(sd2_val, DoubleValueOfScalarLiteral(sd2->literal()));

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      out, AddTensor(graph, std::make_pair(inst, 0), output_shape, res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  poplar::program::Sequence seq;
  res.random.normal(graph, out, mean1_val + mean2_val, sd1_val * sd2_val, seq,
                    GetDebugName(inst));

  return seq;
}

StatusOr<poplar::program::Program> RandomUniformScale(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  const HloInstruction* root = inst->to_apply()->root_instruction();
  const HloInstruction* shift = root->operand(1);
  const HloInstruction* scale = root->operand(0)->operand(1);
  const HloInstruction* lower = root->operand(0)->operand(0)->operand(0);
  const HloInstruction* upper = root->operand(0)->operand(0)->operand(1);

  double shift_val;
  TF_ASSIGN_OR_RETURN(shift_val, DoubleValueOfScalarLiteral(shift->literal()));
  double scale_val;
  TF_ASSIGN_OR_RETURN(scale_val, DoubleValueOfScalarLiteral(scale->literal()));
  double lower_val;
  TF_ASSIGN_OR_RETURN(lower_val, DoubleValueOfScalarLiteral(lower->literal()));
  double upper_val;
  TF_ASSIGN_OR_RETURN(upper_val, DoubleValueOfScalarLiteral(upper->literal()));

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      out, AddTensor(graph, std::make_pair(inst, 0), output_shape, res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  poplar::program::Sequence seq;
  res.random.uniform(graph, out, lower_val * scale_val + shift_val,
                     upper_val * scale_val + shift_val, seq,
                     GetDebugName(inst));

  return seq;
}

StatusOr<poplar::program::Program> RandomNormal(poplar::Graph& graph,
                                                CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output_shape,
                                                TensorMap& tensor_map) {
  const HloInstruction* root = inst->to_apply()->root_instruction();
  const HloInstruction* mean = root->operand(0);
  const HloInstruction* sd = root->operand(1);

  double mean_val;
  TF_ASSIGN_OR_RETURN(mean_val, DoubleValueOfScalarLiteral(mean->literal()));
  double sd_val;
  TF_ASSIGN_OR_RETURN(sd_val, DoubleValueOfScalarLiteral(sd->literal()));

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      out, AddTensor(graph, std::make_pair(inst, 0), output_shape, res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  poplar::program::Sequence seq;
  res.random.normal(graph, out, mean_val, sd_val, seq, GetDebugName(inst));

  return seq;
}

StatusOr<poplar::program::Program> RandomUniform(poplar::Graph& graph,
                                                 CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output_shape,
                                                 TensorMap& tensor_map) {
  const HloInstruction* root = inst->to_apply()->root_instruction();
  const HloInstruction* lower = root->operand(0);
  const HloInstruction* upper = root->operand(1);

  double lower_val;
  TF_ASSIGN_OR_RETURN(lower_val, DoubleValueOfScalarLiteral(lower->literal()));
  double upper_val;
  TF_ASSIGN_OR_RETURN(upper_val, DoubleValueOfScalarLiteral(upper->literal()));

  if (ShapeUtil::ElementIsIntegral(output_shape)) {
    upper_val -= 1.0;
  }

  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      out, AddTensor(graph, std::make_pair(inst, 0), output_shape, res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  poplar::program::Sequence seq;
  res.random.uniform(graph, out, lower_val, upper_val, seq, GetDebugName(inst));

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
