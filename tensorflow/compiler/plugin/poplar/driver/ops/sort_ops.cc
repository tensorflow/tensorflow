#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <popops/ElementWise.hpp>
#include <popops/Sort.hpp>

namespace xla {
namespace poplarplugin {

StatusOr<poplar::program::Program> CreateSort(CompilerResources& res,
                                              const HloInstruction* inst,
                                              TensorMap& tensor_map) {
  const HloSortInstruction* sort = Cast<HloSortInstruction>(inst);

  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence prog;
  // Get the inplace input/outputs.
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, prog));
  if (sort->operand_count() == 1) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor to_sort = inputs[0][0];

    popops::sortInPlace(graph, to_sort, sort->dimensions(0), prog);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, to_sort));
  } else if (sort->operand_count() == 2) {
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(inputs[0].size(), 1);
    CHECK_EQ(inputs[1].size(), 1);
    poplar::Tensor key = inputs[0][0];
    poplar::Tensor value = inputs[1][0];

    popops::sortKeyValueInPlace(graph, key, value, sort->dimensions(0), prog);

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
