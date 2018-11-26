#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <popnn/BatchNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {
namespace {
poplar::Tensor convertVarianceToInvStdDev(poplar::Graph& graph,
                                          const poplar::Tensor& variance,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  auto expression =
      pe::Divide(pe::Const(1), pe::Sqrt(pe::Add(pe::_1, pe::Const(epsilon))));

  return popops::map(graph, expression, {variance}, seq,
                     debug_name + "/VarToInvStdDev");
}

poplar::Tensor convertInvStdDevToVariance(poplar::Graph& graph,
                                          const poplar::Tensor& inv_sd,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  auto expression =
      pe::Sub(pe::Divide(pe::Const(1), pe::Square(pe::_1)), pe::Const(epsilon));

  return popops::map(graph, expression, {inv_sd}, seq,
                     debug_name + "/InvStdDevToVar");
}

poplar::Tensor batchNormalise(
    poplar::Graph& graph, const poplar::Tensor& operand,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const poplar::Tensor& mean, const poplar::Tensor& inv_sd,
    poplar::program::Sequence& seq, const std::string& debug_name) {
  auto multiplicand_expression = pe::Mul(pe::_1, pe::_2);
  poplar::Tensor multiplicand =
      popops::map(graph, multiplicand_expression, {scale, inv_sd}, seq,
                  debug_name + "/Multiplicand");
  auto addend_expression = pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3));
  poplar::Tensor addend =
      popops::map(graph, addend_expression, {offset, multiplicand, mean}, seq,
                  debug_name + "/Addend");
  return popnn::bn::batchNormalise(graph, operand, multiplicand, addend, seq,
                                   debug_name);
}

std::pair<poplar::Tensor, std::vector<std::size_t>>
ShuffleBatchNormInputToPoplar(const poplar::Tensor& input,
                              const unsigned feature_dimension) {
  std::vector<std::size_t> non_broadcast_dims;
  poplar::Tensor input_shuffled;
  if (input.rank() == 4) {
    input_shuffled = input.dimShufflePartial({feature_dimension}, {1});
  } else {
    const unsigned final_dim = input.rank() - 1;
    input_shuffled = input.dimShufflePartial({feature_dimension}, {final_dim});
    non_broadcast_dims = input_shuffled.shape();
    non_broadcast_dims.pop_back();

    std::size_t count = input.numElements() / input.dim(feature_dimension);
    input_shuffled = input_shuffled.reshapePartial(0, final_dim, {count});
  }
  return {input_shuffled, non_broadcast_dims};
}

poplar::Tensor ShuffleBatchNormOutputToTensorflow(
    const poplar::Tensor& output, const unsigned feature_dimension,
    const std::vector<std::size_t>& non_broadcast_dims) {
  poplar::Tensor output_shuffled;
  if (output.rank() == 4) {
    output_shuffled = output.dimShufflePartial({1}, {feature_dimension});
  } else {
    const unsigned final_dim = output.rank() - 1;
    output_shuffled = output.reshapePartial(0, 1, {non_broadcast_dims});
    output_shuffled =
        output_shuffled.dimShufflePartial({final_dim}, {feature_dimension});
  }
  return output_shuffled;
}
}

StatusOr<poplar::program::Program> CreateBatchNormInf(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormInstruction* batch_inf_inst =
      Cast<HloBatchNormInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor offset,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor mean,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor variance,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));

  const auto epsilon = batch_inf_inst->epsilon();
  const unsigned dimension = batch_inf_inst->feature_index();
  const unsigned final_dim = operand.rank() - 1;

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor out = graph.addConstant(operand.elementType(), {1}, 0);
    TF_ASSIGN_OR_RETURN(out,
                        BroadcastTensor(out, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }

  std::vector<std::size_t> non_broadcast_dims;
  poplar::Tensor operand_view;
  std::tie(operand_view, non_broadcast_dims) =
      ShuffleBatchNormInputToPoplar(operand, dimension);

  auto name = GetDebugName(inst);

  auto inv_sd = convertVarianceToInvStdDev(graph, variance, epsilon, seq, name);

  poplar::Tensor out = batchNormalise(graph, operand_view, scale, offset, mean,
                                      inv_sd, seq, name);

  out = ShuffleBatchNormOutputToTensorflow(out, dimension, non_broadcast_dims);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateBatchNormTraining(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormTrainingInstruction* batch_train_inst =
      Cast<HloBatchNormTrainingInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor offset,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  const auto epsilon = batch_train_inst->epsilon();
  const unsigned dimension = batch_train_inst->feature_index();
  const unsigned final_dim = operand.rank() - 1;

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor out = graph.addConstant(operand.elementType(), {1}, 0);
    TF_ASSIGN_OR_RETURN(out,
                        BroadcastTensor(out, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    poplar::Tensor mean = graph.addConstant(operand.elementType(), {1}, NAN);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
    poplar::Tensor variance =
        graph.addConstant(operand.elementType(), {1}, NAN);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance));
    return seq;
  }

  std::vector<std::size_t> non_broadcast_dims;

  poplar::Tensor operand_view;
  std::tie(operand_view, non_broadcast_dims) =
      ShuffleBatchNormInputToPoplar(operand, dimension);

  auto name = GetDebugName(inst);
  poplar::Tensor mean, inv_sd;

  std::tie(mean, inv_sd) = popnn::bn::batchNormEstimates(
      graph, operand_view, epsilon, seq, false, poplar::FLOAT, name);

  poplar::Tensor out = batchNormalise(graph, operand_view, scale, offset, mean,
                                      inv_sd, seq, name);

  auto variance = convertInvStdDevToVariance(graph, inv_sd, epsilon, seq, name);

  out = ShuffleBatchNormOutputToTensorflow(out, dimension, non_broadcast_dims);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance));

  return seq;
}

StatusOr<poplar::program::Program> CreateBatchNormGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormGradInstruction* batch_grad_inst =
      Cast<HloBatchNormGradInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor mean,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor variance,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor grad_output,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));
  const auto epsilon = batch_grad_inst->epsilon();
  const unsigned dimension = batch_grad_inst->feature_index();
  const unsigned final_dim = operand.rank() - 1;

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor operand_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    TF_ASSIGN_OR_RETURN(
        operand_grad,
        BroadcastTensor(operand_grad, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
    poplar::Tensor scale_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
    poplar::Tensor offset_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
    return seq;
  }

  auto name = GetDebugName(inst);

  // Reshape the input
  std::vector<std::size_t> non_broadcast_dims;

  poplar::Tensor operand_view, grad_output_view;
  std::tie(operand_view, non_broadcast_dims) =
      ShuffleBatchNormInputToPoplar(operand, dimension);
  std::tie(grad_output_view, non_broadcast_dims) =
      ShuffleBatchNormInputToPoplar(grad_output, dimension);

  auto inv_sd = convertVarianceToInvStdDev(graph, variance, epsilon, seq, name);

  // Compute the whitened activations.
  poplar::Tensor operand_whitened = popnn::bn::batchNormWhiten(
      graph, operand_view, mean, inv_sd, seq, name + "/WhitenedActs");

  // Compute the deltas for scaled and offset
  poplar::Tensor scale_grad, offset_grad;
  std::tie(scale_grad, offset_grad) =
      popnn::bn::batchNormDeltas(graph, operand_whitened, grad_output_view, seq,
                                 poplar::FLOAT, name + "/Deltas");
  // Compute the delta for the operand grad
  poplar::Tensor operand_grad = popnn::bn::batchNormGradients(
      graph, operand_whitened, grad_output_view, scale_grad, offset_grad,
      inv_sd, scale, seq, poplar::FLOAT, name + "/Grad");

  operand_grad = ShuffleBatchNormOutputToTensorflow(operand_grad, dimension,
                                                    non_broadcast_dims);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
