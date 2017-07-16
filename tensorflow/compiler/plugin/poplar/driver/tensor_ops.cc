#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

port::StatusOr<poplar::program::Program>
CreateSliceUpdateOp(poplar::Graph &graph,
                    CompilerResources& res,
                    const HloInstruction *inst,
                    const xla::Shape& output_shape,
                    TensorMap& tensor_map) {

  poplar::Tensor input;
  TF_ASSIGN_OR_RETURN(input,
                      FindInstructionInput(tensor_map, inst, 0, 0));
  poplar::Tensor update;
  TF_ASSIGN_OR_RETURN(update,
                      FindInstructionInput(tensor_map, inst, 1, 0));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, input));

  const HloInstruction* root = inst->fused_expression_root();

  std::vector<int64> begin;
  TF_ASSIGN_OR_RETURN(begin,
                      LiteralVectorToInt64Vector(root->operand(2)->literal()));

  if (begin.size() != input.rank()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Invalid update slice start");
  }

  std::vector<std::size_t> s_begin =
          convert_array<std::vector<std::size_t>>(begin);
  std::vector<std::size_t> s_end = s_begin;
  for (unsigned int i = 0; i < s_end.size(); i++) {
    s_end[i] += update.dim(i);
  }
  poplar::Tensor slice = input.slice(s_begin, s_end);

  return poplar::program::Copy(update, slice);
}

port::StatusOr<poplar::program::Program>
CreateSliceOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map) {
  poplar::Tensor input;
  TF_ASSIGN_OR_RETURN(input,
                      FindInstructionInput(tensor_map, inst, 0, 0));

  const HloInstruction* root = inst->fused_expression_root();

  std::vector<int64> begin;
  TF_ASSIGN_OR_RETURN(begin,
                      LiteralVectorToInt64Vector(root->operand(1)->literal()));

  if (begin.size() != input.rank()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Invalid update slice start");
  }

  std::vector<std::size_t> s_begin =
          convert_array<std::vector<std::size_t>>(begin);
  std::vector<std::size_t> s_end = s_begin;
  for (unsigned int i = 0; i < s_end.size(); i++) {
    s_end[i] += output_shape.dimensions(i);
  }

  poplar::Tensor slice = input.slice(s_begin, s_end);
  poplar::Tensor out = graph.clone(slice);

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return poplar::program::Copy(slice, out);
}

}
}

