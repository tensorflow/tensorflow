#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/norm_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <popnn/BatchNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

std::pair<poplar::Tensor, std::vector<std::size_t>> ShuffleNormInputToPoplar(
    const poplar::Tensor& input, const unsigned feature_dimension) {
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

poplar::Tensor ShuffleNormOutputToTensorflow(
    const poplar::Tensor& output, const unsigned feature_dimension,
    const std::vector<std::size_t>& non_broadcast_dims) {
  poplar::Tensor output_shuffled;
  if (output.rank() == 4) {
    output_shuffled = output.dimShufflePartial({1}, {feature_dimension});
  } else {
    output_shuffled = output.reshapePartial(0, 1, {non_broadcast_dims});
    const unsigned final_dim = output_shuffled.rank() - 1;
    output_shuffled =
        output_shuffled.dimShufflePartial({final_dim}, {feature_dimension});
  }
  return output_shuffled;
}

StatusOr<poplar::program::Program> CreateBatchNormInf(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormInstruction* batch_inf_inst =
      Cast<HloBatchNormInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  const auto epsilon = batch_inf_inst->epsilon();
  const unsigned dimension = batch_inf_inst->feature_index();

  return CreateNormInference(NormType::BatchNorm, graph, res, inst, epsilon,
                             dimension, absl::nullopt, tensor_map);
}

StatusOr<poplar::program::Program> CreateNormInference(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor offset,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor mean,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor variance_or_inv_std_dev,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor out = graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(out, 0);
    TF_ASSIGN_OR_RETURN(out,
                        BroadcastTensor(out, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }

  std::vector<std::size_t> non_broadcast_dims;
  poplar::Tensor operand_view;
  std::tie(operand_view, non_broadcast_dims) =
      ShuffleNormInputToPoplar(operand, feature_dimension);

  auto out = norm_graph_caching::DoCachedNormInference(
      norm_type, graph, res, operand_view, scale, offset, mean,
      variance_or_inv_std_dev, epsilon, optional_num_groups,
      GetShardingDeviceId(inst), seq, GetDebugName(inst));

  out =
      ShuffleNormOutputToTensorflow(out, feature_dimension, non_broadcast_dims);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

StatusOr<poplar::program::Program> CreateBatchNormTraining(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormTrainingInstruction* batch_train_inst =
      Cast<HloBatchNormTrainingInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);
  const auto epsilon = batch_train_inst->epsilon();
  const unsigned dimension = batch_train_inst->feature_index();
  return CreateNormTraining(NormType::BatchNorm, graph, res, inst, epsilon,
                            dimension, absl::nullopt, tensor_map);
}

StatusOr<poplar::program::Program> CreateNormTraining(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor offset,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor out = graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(out, 0);
    TF_ASSIGN_OR_RETURN(out,
                        BroadcastTensor(out, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    poplar::Tensor mean = graph.addConstant(operand.elementType(), {1}, NAN);
    graph.setTileMapping(mean, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
    poplar::Tensor variance_or_inv_std_dev =
        graph.addConstant(operand.elementType(), {1}, NAN);
    graph.setTileMapping(variance_or_inv_std_dev, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance_or_inv_std_dev));
    return seq;
  }

  std::vector<std::size_t> non_broadcast_dims;
  poplar::Tensor operand_view;
  std::tie(operand_view, non_broadcast_dims) =
      ShuffleNormInputToPoplar(operand, feature_dimension);

  poplar::Tensor out, mean, variance_or_inv_std_dev;
  std::tie(out, mean, variance_or_inv_std_dev) =
      norm_graph_caching::DoCachedNormTraining(
          norm_type, graph, res, operand_view, scale, offset, epsilon,
          optional_num_groups, GetShardingDeviceId(inst), seq,
          GetDebugName(inst));

  out =
      ShuffleNormOutputToTensorflow(out, feature_dimension, non_broadcast_dims);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, mean));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, variance_or_inv_std_dev));

  return seq;
}

StatusOr<poplar::program::Program> CreateBatchNormGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map) {
  const HloBatchNormGradInstruction* batch_grad_inst =
      Cast<HloBatchNormGradInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  const auto epsilon = batch_grad_inst->epsilon();
  const unsigned dimension = batch_grad_inst->feature_index();
  return CreateNormGrad(NormType::BatchNorm, graph, res, inst, epsilon,
                        dimension, absl::nullopt, tensor_map);
}

StatusOr<poplar::program::Program> CreateNormGrad(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor mean,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor variance_or_inv_std_dev,
                      FindInstructionInput(tensor_map, res, inst, 3, seq));
  TF_ASSIGN_OR_RETURN(poplar::Tensor grad_output,
                      FindInstructionInput(tensor_map, res, inst, 4, seq));
  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor operand_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(operand_grad, 0);
    TF_ASSIGN_OR_RETURN(
        operand_grad,
        BroadcastTensor(operand_grad, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
    poplar::Tensor scale_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(scale_grad, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
    poplar::Tensor offset_grad =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(offset_grad, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
    return seq;
  }

  // Reshape the input
  std::vector<std::size_t> non_broadcast_dims;

  poplar::Tensor operand_view, grad_output_view;
  std::tie(operand_view, non_broadcast_dims) =
      ShuffleNormInputToPoplar(operand, feature_dimension);
  std::tie(grad_output_view, non_broadcast_dims) =
      ShuffleNormInputToPoplar(grad_output, feature_dimension);

  poplar::Tensor operand_grad, scale_grad, offset_grad;

  std::tie(operand_grad, scale_grad, offset_grad) =
      norm_graph_caching::DoCachedNormGrad(
          norm_type, graph, res, operand_view, scale, mean,
          variance_or_inv_std_dev, grad_output_view, epsilon,
          optional_num_groups, GetShardingDeviceId(inst), seq,
          GetDebugName(inst));

  operand_grad = ShuffleNormOutputToTensorflow(operand_grad, feature_dimension,
                                               non_broadcast_dims);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand_grad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, scale_grad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 2, offset_grad));
  return seq;
}

StatusOr<poplar::program::Program> CreateNormStatistics(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const HloInstruction* inst, const float epsilon,
    const uint32 feature_dimension, absl::optional<uint32> optional_num_groups,
    TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));

  // Special case - zero sized array
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::Tensor mean = graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(mean, 0);
    TF_ASSIGN_OR_RETURN(mean,
                        BroadcastTensor(mean, inst->operand(0)->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, mean));
    poplar::Tensor variance_or_inv_std_dev =
        graph.addConstant(operand.elementType(), {1}, 0);
    graph.setTileMapping(variance_or_inv_std_dev, 0);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, variance_or_inv_std_dev));
    return seq;
  }

  // Reshape the input
  std::vector<std::size_t> non_broadcast_dims;

  poplar::Tensor operand_view, grad_output_view;
  std::tie(operand_view, non_broadcast_dims) =
      ShuffleNormInputToPoplar(operand, feature_dimension);

  poplar::Tensor mean, variance_or_inv_std_dev;
  std::tie(mean, variance_or_inv_std_dev) =
      norm_graph_caching::DoCachedNormStatistics(
          norm_type, graph, res, operand_view, epsilon, optional_num_groups,
          GetShardingDeviceId(inst), seq, GetDebugName(inst));

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, mean));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, variance_or_inv_std_dev));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
