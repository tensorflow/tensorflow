#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <popops/Gather.hpp>

namespace xla {
namespace poplarplugin {

StatusOr<poplar::program::Sequence> CreateGather(
    CompilerResources& res, const HloGatherInstruction* inst,
    TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  const auto slice_sizes = inst->gather_slice_sizes();
  const auto dim_numbers = inst->gather_dimension_numbers();

  const auto index_vector_dim = dim_numbers.index_vector_dim();
  const auto offset_dims = dim_numbers.offset_dims();
  const auto collapsed_slice_dims = dim_numbers.collapsed_slice_dims();
  const auto start_index_map = dim_numbers.start_index_map();

  poplar::program::Sequence prog;

  TF_ASSIGN_OR_RETURN(poplar::Tensor operand,
                      FindInstructionInput(tensor_map, res, inst, 0, prog));

  TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  auto result =
      popops::gather(graph, operand, indices.reinterpret(poplar::UNSIGNED_INT),
                     index_vector_dim, {offset_dims.begin(), offset_dims.end()},
                     {slice_sizes.begin(), slice_sizes.end()},
                     {collapsed_slice_dims.begin(), collapsed_slice_dims.end()},
                     {start_index_map.begin(), start_index_map.end()}, prog);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, result));
  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
