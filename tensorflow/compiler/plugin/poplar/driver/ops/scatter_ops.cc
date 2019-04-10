#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <popops/ElementWise.hpp>
#include <popops/Scatter.hpp>

namespace xla {
namespace poplarplugin {

StatusOr<poplar::program::Program> CreateScatter(
    CompilerResources& res, const HloScatterInstruction* inst,
    TensorMap& tensor_map) {
  const auto update_computation = inst->to_apply();
  const auto dim_numbers = inst->scatter_dimension_numbers();

  const auto update_window_dims = dim_numbers.update_window_dims();
  const auto inserted_window_dims = dim_numbers.inserted_window_dims();
  const auto scatter_dims_to_operand_dims =
      dim_numbers.scatter_dims_to_operand_dims();
  const auto index_vector_dim = dim_numbers.index_vector_dim();

  poplar::program::Sequence prog;
  poplar::Graph& graph = GetGraph(res, inst);

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, prog));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor operand = inputs[0][0];

  TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  TF_ASSIGN_OR_RETURN(poplar::Tensor updates,
                      FindInstructionInput(tensor_map, res, inst, 2, prog));

  popops::UpdateComputationFunc update_computation_func;
  auto root_inst = update_computation->root_instruction();

  auto tmp = graph.addVariable(operand.elementType(), {});
  graph.setTileMapping(tmp, 0);
  ArgVectors args = {{tmp}, {graph.clone(tmp)}};

  TF_ASSIGN_OR_RETURN(
      auto update_comp_visitor,
      GetOrCompileSubComputation(res, args, update_computation));

  // Fast path the gradient accumulation case
  if (root_inst->opcode() == HloOpcode::kAdd &&
      root_inst->operand_count() == 2 &&
      root_inst->operand(0)->opcode() == HloOpcode::kParameter &&
      root_inst->operand(1)->opcode() == HloOpcode::kParameter) {
    update_computation_func =
        [&](poplar::Graph& g, poplar::Tensor& a, poplar::Tensor& b,
            poplar::program::Sequence& p) -> poplar::Tensor {
      popops::addInPlace(g, b, a, p);

      return b;
    };
  } else {
    // Handle the general case
    update_computation_func =
        [&](poplar::Graph& g, poplar::Tensor& a, poplar::Tensor& b,
            poplar::program::Sequence& p) -> poplar::Tensor {
      auto result = g.clone(b);
      for (int i = 0; i < a.numElements(); ++i) {
        auto a_elem = a.flatten()[i];
        auto b_elem = b.flatten()[i];
        auto o_elem = result.flatten()[i];

        // Copy the inputs in
        p.add(
            poplar::program::Copy(a_elem, update_comp_visitor->inputs()[0][0]));
        p.add(
            poplar::program::Copy(b_elem, update_comp_visitor->inputs()[1][0]));

        // Add the sequence
        p.add(update_comp_visitor->sequence);

        // Copy the output out
        p.add(poplar::program::Copy(update_comp_visitor->outputs()[0], o_elem));
      }

      return result;
    };
  }

  popops::scatter(graph, operand, indices, updates, index_vector_dim,
                  {update_window_dims.begin(), update_window_dims.end()},
                  {inserted_window_dims.begin(), inserted_window_dims.end()},
                  {scatter_dims_to_operand_dims.begin(),
                   scatter_dims_to_operand_dims.end()},
                  update_computation_func, prog);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));

  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
