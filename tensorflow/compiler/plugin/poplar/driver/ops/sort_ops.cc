#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"

#include <popops/ElementWise.hpp>
#include <popops/Sort.hpp>

namespace xla {
namespace poplarplugin {

namespace {
bool IsSimpleComparison(const HloInstruction* inst) {
  HloInstruction* root(inst->to_apply()->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  if (root->opcode() != HloOpcode::kCompare) {
    return false;
  }

  switch (root->comparison_direction()) {
    case ComparisonDirection::kGe:
    case ComparisonDirection::kGt:
    case ComparisonDirection::kLe:
    case ComparisonDirection::kLt:
      return true;
    default:
      return false;
  }
}

bool ReverseSortOutput(const HloInstruction* inst) {
  HloInstruction* root(inst->to_apply()->root_instruction());
  switch (root->comparison_direction()) {
    case ComparisonDirection::kGe:
    case ComparisonDirection::kGt:
      return true;
    default:
      return false;
  }
}
}

StatusOr<poplar::program::Program> CreateSort(CompilerResources& res,
                                              const HloInstruction* inst,
                                              TensorMap& tensor_map) {
  const HloSortInstruction* sort = Cast<HloSortInstruction>(inst);

  if (!IsSimpleComparison(inst)) {
    return xla::Unimplemented(
        "Current Sort implementation only supports GT/LT/GE/LE comparisons");
  }

  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;
  // Get the inplace input/outputs.
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      FindInplaceOutputTensors(tensor_map, res, inst, prog));
  if (sort->operand_count() == 1) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor to_sort = inputs[0][0];

    popops::sortInPlace(graph, to_sort, sort->dimensions(0), prog);

    if (ReverseSortOutput(inst)) {
      TF_ASSIGN_OR_RETURN(to_sort, ReverseTensor(to_sort, sort->dimensions()));
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, to_sort));
  } else if (sort->operand_count() == 2) {
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(inputs[0].size(), 1);
    CHECK_EQ(inputs[1].size(), 1);
    poplar::Tensor key = inputs[0][0];
    poplar::Tensor value = inputs[1][0];

    popops::sortKeyValueInPlace(graph, key, value, sort->dimensions(0), prog);

    if (ReverseSortOutput(inst)) {
      TF_ASSIGN_OR_RETURN(key, ReverseTensor(key, sort->dimensions()));
      TF_ASSIGN_OR_RETURN(value, ReverseTensor(value, sort->dimensions()));
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, key));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, value));
  } else {
    return xla::Unimplemented(
        "Current Sort implementation only supports up to 2 operands, where as "
        "%s has %d",
        sort->name().c_str(), sort->operand_count());
  }

  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
