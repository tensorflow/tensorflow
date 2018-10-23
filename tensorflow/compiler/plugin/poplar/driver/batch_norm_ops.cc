#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

namespace xla {
namespace poplarplugin {

static std::string BatchNormVertex(poplar::Type a) {
  return "BatchNormVertex<" + a.toString() + ">";
}

poplar::program::Execute CreateBatchNormInf(
    poplar::Graph& graph, poplar::Tensor result, poplar::Tensor operand,
    poplar::Tensor scale, poplar::Tensor offset, poplar::Tensor mean,
    poplar::Tensor variance, float epsilon, int64 dimension,
    const std::string& debug_name = "") {
  std::vector<unsigned> permutation(operand.rank());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation.front(), permutation[dimension]);
  poplar::Tensor operand_view = operand.dimShuffle(permutation);
  poplar::Tensor result_view = result.dimShuffle(permutation);

  poplar::Tensor epsilon_const = graph.addConstant(poplar::FLOAT, {}, epsilon);

  auto bn_cs = graph.addComputeSet(debug_name);
  const std::string vertex_type = BatchNormVertex(operand.elementType());

  for (std::size_t i = 0; i < operand_view.dim(0); ++i) {
    poplar::Tensor scale_slice = scale[i];
    poplar::Tensor offset_slice = offset[i];
    poplar::Tensor mean_slice = mean[i];
    poplar::Tensor variance_slice = variance[i];
    poplar::Tensor operand_slice = operand_view[i].flatten();
    poplar::Tensor result_slice = result_view[i].flatten();

    const auto tile_intervals = graph.getTileMapping(operand_slice);
    for (std::size_t tile = 0; tile < tile_intervals.size(); ++tile) {
      for (const auto& interval : tile_intervals[tile]) {
        if (interval.size() > 0) {
          auto v = graph.addVertex(bn_cs, vertex_type);
          graph.setTileMapping(v, tile);
          graph.setCycleEstimate(v, 20 * interval.size());

          graph.connect(v["operand"], operand_slice.slice(interval));
          graph.connect(v["result"], result_slice.slice(interval));
          graph.connect(v["scale"], scale_slice);
          graph.connect(v["offset"], offset_slice);
          graph.connect(v["mean"], mean_slice);
          graph.connect(v["variance"], variance_slice);
          graph.connect(v["epsilon"], epsilon_const);
        }
      }
    }
  }

  return poplar::program::Execute(bn_cs);
}

StatusOr<poplar::program::Program> CreateBatchNormInf(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    TensorMap& tensor_map) {
  const HloBatchNormInstruction* batch_inf_inst =
      Cast<HloBatchNormInstruction>(inst);

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, inst, 0));
  TF_ASSIGN_OR_RETURN(poplar::Tensor scale,
                      FindInstructionInput(tensor_map, inst, 1));
  TF_ASSIGN_OR_RETURN(poplar::Tensor offset,
                      FindInstructionInput(tensor_map, inst, 2));
  TF_ASSIGN_OR_RETURN(poplar::Tensor mean,
                      FindInstructionInput(tensor_map, inst, 3));
  TF_ASSIGN_OR_RETURN(poplar::Tensor variance,
                      FindInstructionInput(tensor_map, inst, 4));

  poplar::Tensor output = graph.clone(operand);
  poplar::program::Sequence result;
  result.add(CreateBatchNormInf(graph, output, operand, scale, offset, mean,
                                variance, batch_inf_inst->epsilon(),
                                batch_inf_inst->feature_index(),
                                GetDebugName(inst)));

  TF_CHECK_OK(AddOutputTensor(graph, res, result, tensor_map, inst, 0, output)
                  .status());

  return result;
}

}  // namespace poplarplugin
}  // namespace xla
