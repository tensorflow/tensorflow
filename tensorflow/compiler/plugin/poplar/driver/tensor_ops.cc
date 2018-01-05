#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popstd/DynamicSlice.hpp>

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
                      FindInstructionInput(tensor_map, inst, 0));
  poplar::Tensor update;
  TF_ASSIGN_OR_RETURN(update,
                      FindInstructionInput(tensor_map, inst, 1));

  const HloInstruction* root = inst->to_apply()->root_instruction();

  std::vector<int64> begin;
  TF_ASSIGN_OR_RETURN(begin,
                      LiteralVectorToInt64Vector(root->operand(2)->literal()));

  if (begin.size() != input.rank()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Invalid update slice start");
  }

  // We try to update in-place but it is possible that the input is a constant
  // tensor in which case we need to make a copy of it to update it.
  poplar::program::Sequence seq;
  if (!input.isParallelWriteable()) {
    poplar::Tensor copy;
    TF_ASSIGN_OR_RETURN(copy,
                        AddTensor(graph,
                                  std::make_pair(inst,0),
                                  XlaShapeFromPoplarShape(
                                          output_shape.element_type(),
                                          input.shape()),
                                  res));
    seq.add(poplar::program::Copy(input, copy));
    input = copy;
  }

  std::vector<std::size_t> s_begin =
          convert_array<std::vector<std::size_t>>(begin);
  std::vector<std::size_t> s_end = s_begin;
  for (unsigned int i = 0; i < s_end.size(); i++) {
    s_end[i] += update.dim(i);
  }
  poplar::Tensor slice = input.slice(s_begin, s_end);
  seq.add(poplar::program::Copy(update, slice));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, input));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateSliceOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map) {
  poplar::Tensor input;
  TF_ASSIGN_OR_RETURN(input,
                      FindInstructionInput(tensor_map, inst, 0));

  const HloInstruction* root = inst->to_apply()->root_instruction();

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

port::StatusOr<poplar::program::Program>
CreateDynamicSliceUpdateOp(poplar::Graph &graph,
                           CompilerResources& res,
                           const HloInstruction *inst,
                           const xla::Shape& output_shape,
                           TensorMap& tensor_map) {
  poplar::Tensor input;
  TF_ASSIGN_OR_RETURN(input,
                      FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor update;
  TF_ASSIGN_OR_RETURN(update,
                      FindInstructionInput(tensor_map, inst, 1));

  poplar::Tensor indices;
  TF_ASSIGN_OR_RETURN(indices,
                      FindInstructionInput(tensor_map, inst, 2));

  // We try to update in-place but it is possible that the input is a constant
  // tensor in which case we need to make a copy of it to update it.
  poplar::program::Sequence seq;
  if (!input.isParallelWriteable()) {
    poplar::Tensor copy;
    TF_ASSIGN_OR_RETURN(copy,
                        AddTensor(graph,
                                  std::make_pair(inst,0),
                                  XlaShapeFromPoplarShape(
                                          output_shape.element_type(),
                                          input.shape()),
                                  res));
    seq.add(poplar::program::Copy(input, copy));
    input = copy;
  }

  // popstd::dynamicUpdate() expects unsigned integer offsets, whereas
  // Tensorflow prefers signed int. Convert if necessary.
  auto type = indices.elementType();
  if (type == poplar::INT) {
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  // `slice_dims` is the list of dimensions to slice on. popstd::dynamicUpdate()
  // optimises the order. A possible future optimisation might be to omit
  // dimensions that aren't actually sliced.
  std::vector<std::size_t> slice_dims(inst->shape().dimensions_size());
  std::iota(slice_dims.begin(), slice_dims.end(), 0);

  // Add the dynamic update operations to `seq`. This automatically
  // creates the required compute set.
  popstd::dynamicUpdate(graph,
                        input,
                        update,
                        indices,
                        slice_dims,
                        update.shape(),
                        seq,
                        inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, input));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateDynamicSliceOp(poplar::Graph &graph,
                     CompilerResources& res,
                     const HloInstruction *inst,
                     const xla::Shape& output_shape,
                     TensorMap& tensor_map) {
  poplar::Tensor input;
  TF_ASSIGN_OR_RETURN(input,
                      FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor indices;
  TF_ASSIGN_OR_RETURN(indices,
                      FindInstructionInput(tensor_map, inst, 1));

  // popstd::dynamicUpdate() expects unsigned integer offsets, whereas
  // Tensorflow prefers signed int. Convert if necessary.
  auto type = indices.elementType();
  if (type == poplar::INT) {
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  // `slice_dims` is the list of dimensions to slice on. popstd::dynamicUpdate()
  // optimises the order. A possible future optimisation might be to omit
  // dimensions that aren't actually sliced.
  std::vector<std::size_t> slice_dims(inst->shape().dimensions_size());
  std::iota(slice_dims.begin(), slice_dims.end(), 0);

  // The program to execute the dynamic slice.
  poplar::program::Sequence seq;

  // Add the dynamic slice operations to `seq`. This automatically
  // creates the required compute set.
  poplar::Tensor out =
    popstd::dynamicSlice(graph,
                         input,
                         indices,
                         slice_dims,
                         PoplarShapeFromXlaShape(output_shape),
                         seq,
                         inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

}
}

