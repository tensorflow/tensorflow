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

  unsigned dimension = batch_inf_inst->feature_index();
  unsigned final_dim = operand.rank() - 1;

  std::vector<std::size_t> non_broadcast_dims;

  poplar::Tensor operand_view;

  if (operand.rank() == 4) {
    operand_view = operand.dimShufflePartial({dimension}, {1});
  } else {
    operand_view = operand.dimShufflePartial({dimension}, {final_dim});
    non_broadcast_dims = operand_view.shape();

    std::size_t count = operand.numElements() / operand.dim(dimension);
    operand_view = operand_view.reshapePartial(0, final_dim, {count});
  }

  auto name = GetDebugName(inst);

  auto var_expression = pe::Divide(
      pe::Const(1),
      pe::Sqrt(pe::Add(pe::_1, pe::Const(batch_inf_inst->epsilon()))));

  auto inv_sd = popops::map(graph, var_expression, {variance}, seq, name);

  auto bn = popnn::bn::batchNormalise(graph, operand_view, scale, offset, mean,
                                      inv_sd, seq, name);

  poplar::Tensor out;
  if (operand.rank() == 4) {
    out = bn.first.dimShufflePartial({1}, {dimension});
  } else {
    out = bn.first.reshapePartial(0, 1, {non_broadcast_dims});
    out = out.dimShufflePartial({final_dim}, {dimension});
  }

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

  unsigned dimension = batch_train_inst->feature_index();
  unsigned final_dim = operand.rank() - 1;

  std::vector<std::size_t> non_broadcast_dims;

  poplar::Tensor operand_view;

  if (operand.rank() == 4) {
    operand_view = operand.dimShufflePartial({dimension}, {1});
  } else {
    operand_view = operand.dimShufflePartial({dimension}, {final_dim});
    non_broadcast_dims = operand_view.shape();

    std::size_t count = operand.numElements() / operand.dim(dimension);
    operand_view = operand_view.reshapePartial(0, final_dim, {count});
  }

  auto name = GetDebugName(inst);

  auto est = popnn::bn::batchNormEstimates(graph, operand_view,
                                           batch_train_inst->epsilon(), seq,
                                           poplar::FLOAT, name);
  poplar::Tensor mean = est.first;
  poplar::Tensor inv_sd = est.second;

  auto bn = popnn::bn::batchNormalise(graph, operand_view, scale, offset, mean,
                                      inv_sd, seq, name);

  auto var_expression = pe::Sub(pe::Divide(pe::Const(1), pe::Square(pe::_1)),
                                pe::Const(batch_train_inst->epsilon()));

  auto variance = popops::map(graph, var_expression, {inv_sd}, seq, name);

  poplar::Tensor out;
  if (operand.rank() == 4) {
    out = bn.first.dimShufflePartial({1}, {dimension});
  } else {
    out = bn.first.reshapePartial(0, 1, {non_broadcast_dims});
    out = out.dimShufflePartial({final_dim}, {dimension});
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance));

  return seq;
}

StatusOr<poplar::program::Program> CreateBatchNormGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormGradInstruction* batch_grad_inst =
      Cast<HloBatchNormGradInstruction>(inst);

  poplar::program::Sequence seq;
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
