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
#include <popops/DynamicSlice.hpp>

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

  poplar::program::Sequence seq;
  poplar::Tensor copy;

  if (!input.isParallelWriteable()) {
    TF_ASSIGN_OR_RETURN(copy,
                        AddTensor(graph,
                                  std::make_pair(inst,0),
                                  XlaShapeFromPoplarShape(
                                          output_shape.element_type(),
                                          input.shape()),
                                  res));
    seq.add(poplar::program::Copy(input, copy));
    input = copy;
  } else {
    copy = graph.clone(input);
    seq.add(poplar::program::Copy(input, copy));
  }

  std::vector<std::size_t> s_begin =
          convert_array<std::vector<std::size_t>>(begin);
  std::vector<std::size_t> s_end = s_begin;
  for (unsigned int i = 0; i < s_end.size(); i++) {
    s_end[i] += update.dim(i);
  }
  poplar::Tensor slice = copy.slice(s_begin, s_end);
  seq.add(poplar::program::Copy(update, slice));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, copy));

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
  poplar::Tensor out = graph.clone(slice, inst->name());

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

  auto type = indices.elementType();
  if (type == poplar::INT) {
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  std::vector<std::size_t> slice_dims;
  std::vector<std::size_t> slice_sizes;
  poplar::Tensor slice_indices;
  for (unsigned d=0; d<inst->shape().dimensions_size(); d++) {
    auto t = indices.index({d}).reshape({1});
    bool same_shape = inst->shape().dimensions(d) == update.shape()[d];
    unsigned int index;
    bool zero_index = t.getConstantValue(&index) && (index == 0);

    if (!(same_shape && zero_index)) {
      if (slice_dims.size() == 0) {
        slice_indices = t;
      } else {
        slice_indices = poplar::concat(slice_indices, t);
      }
      slice_dims.push_back(d);
      slice_sizes.push_back(update.shape()[d]);
    }
  }

  if (slice_dims.size() > 0) {
    popops::dynamicUpdate(graph,
                          input,
                          update,
                          slice_indices,
                          slice_dims,
                          slice_sizes,
                          seq,
                          inst->name());
  } else {
    seq.add(poplar::program::Copy(update, input));
  }

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

  auto type = indices.elementType();
  if (type == poplar::INT) {
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  std::vector<std::size_t> slice_dims;
  std::vector<std::size_t> slice_sizes;
  poplar::Tensor slice_indices;
  for (unsigned d=0; d<inst->shape().dimensions_size(); d++) {
    auto t = indices.index({d}).reshape({1});
    bool same_shape = inst->shape().dimensions(d) == input.shape()[d];
    unsigned int index;
    bool zero_index = t.getConstantValue(&index) && (index == 0);

    if (!(same_shape && zero_index)) {
      if (slice_dims.size() == 0) {
        slice_indices = t;
      } else {
        slice_indices = poplar::concat(slice_indices, t, 0);
      }
      slice_dims.push_back(d);
      slice_sizes.push_back(inst->shape().dimensions(d));
    }
  }

  // The program to execute the dynamic slice.
  poplar::program::Sequence seq;

  // Add the dynamic slice operations to `seq`. This automatically
  // creates the required compute set.
  poplar::Tensor out;

  if (slice_dims.size() > 0) {
    out = popops::dynamicSlice(graph,
                               input,
                               slice_indices,
                               slice_dims,
                               slice_sizes,
                               seq,
                               inst->name());
  } else {
    out = input;
  }

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateWideConstant(poplar::Graph &graph,
                   CompilerResources& res,
                   const HloInstruction *inst,
                   const xla::Shape& output_shape,
                   TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  const HloInstruction* root = inst->to_apply()->root_instruction();
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddConstantTensor(graph,
                                             std::make_pair(inst, 0),
                                             inst->shape(),
                                             root->operand(0)->literal(),
                                             res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

}
}

