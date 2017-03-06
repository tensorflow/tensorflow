#include <algorithm>
#include <limits>

#include "tensorflow/compiler/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popnn/Reduce.hpp>

namespace xla {
namespace poplarplugin {



static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string out_conn("out");

static const std::string reduction_add("ReductionAdd");
static const std::string reduction_sub("ReductionSub");
static const std::string reduction_mul("ReductionMul");
static const std::string reduction_div("ReductionDiv");
static const std::string reduction_max("ReductionMax");
static const std::string reduction_min("ReductionMin");
static const std::string reduction_and("ReductionAnd");
static const std::string reduction_or("ReductionOr");

static const std::string reduction_unknown("UnknownReduction");

port::StatusOr<bool>
IsComputationReducableArtithmetic(HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kLogicalAnd:
    case HloOpcode::kLogicalOr:
      return true;
    default:
      return false;
  }
}

port::StatusOr<bool>
IsComputationSimpleSelection(HloComputation* computation)
{
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
      return true;
    default:
      return false;
  }
}

static const std::string&
ReductionVertexBaseName(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
      return reduction_add;
    case HloOpcode::kSubtract:
      return reduction_sub;
    case HloOpcode::kMultiply:
      return reduction_mul;
    case HloOpcode::kDivide:
      return reduction_div;
    case HloOpcode::kMaximum:
      return reduction_max;
    case HloOpcode::kMinimum:
      return reduction_min;
    case HloOpcode::kLogicalAnd:
      return reduction_and;
    case HloOpcode::kLogicalOr:
      return reduction_or;
    default:
      // Cannot reach here
      return reduction_unknown;
  }
}

port::StatusOr<poplar::program::Program>
CreateSimpleReduction(poplar::Graph &graph,
                      const HloInstruction *inst,
                      const xla::Shape& output_shape,
                      TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor to_reduce;
  TF_ASSIGN_OR_RETURN(to_reduce, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor init_val;
  TF_ASSIGN_OR_RETURN(init_val, FindInstructionInput(tensor_map, inst, 1, 0));

  // Find the type and vertex
  HloInstruction* root(inst->to_apply()->root_instruction());
  std::string vertex_name = templateVertex(ReductionVertexBaseName(root),
                                           graph.getTensorElementType(to_reduce));


  // Convert the tensor into a NxM 2D tensor with the dimensions
  // to reduce in the minor part
  int64 reduction_flatten_elements = 1;
  std::set<unsigned> reduction_dims;
  for (auto d : inst->dimensions()) {
    reduction_dims.insert(d);
    reduction_flatten_elements *= to_reduce.dim(d);
  }

  std::vector<unsigned int> dim_shuffle(to_reduce.rank());
  std::iota(dim_shuffle.begin(), dim_shuffle.end(), 0);

  std::sort(dim_shuffle.begin(), dim_shuffle.end(),
            [&reduction_dims](unsigned a, unsigned b) {
              bool a_is_reduction =
                      (reduction_dims.find(a) != reduction_dims.end());
              bool b_is_reduction =
                      (reduction_dims.find(b) != reduction_dims.end());
              if (a_is_reduction && !b_is_reduction) return false;
              if (b_is_reduction && !a_is_reduction) return true;
              return a < b;
            });

  poplar::Tensor shuffled = to_reduce.dimShuffle(dim_shuffle);

  // Reshape to a 2D tensor
  std::vector<size_t> reshaped(2);
  reshaped[0] = to_reduce.numElements() / reduction_flatten_elements;
  reshaped[1] = reduction_flatten_elements;
  shuffled = shuffled.reshape(reshaped);

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  // One vertex per non-reduced element
  auto cs = graph.addComputeSet(inst->name());

  //   reduceByDstMapping(graph, shuffled, out, graph.getTileMapping(out), cs);

  const unsigned long N = out.dim(0);
  const auto &device_info = graph.getDevice().getDeviceInfo();

  for (unsigned i = 0; i < N; ++i) {
    auto v = graph.addVertex(cs, vertex_name,
                             {{"a", shuffled[i]},
                              {"out", out.slice(i, i+1)}});
    graph.setTileMapping(v, (i / device_info.numWorkerContexts) % device_info.getNumTiles());
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateSimpleWindowReduction(poplar::Graph &graph,
                            const HloInstruction *inst,
                            const xla::Shape& output_shape,
                            TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor to_reduce;
  TF_ASSIGN_OR_RETURN(to_reduce, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor init_val;
  TF_ASSIGN_OR_RETURN(init_val, FindInstructionInput(tensor_map, inst, 1, 0));

  // Find the type and vertex
  HloInstruction* root(inst->to_apply()->root_instruction());
  std::string vertex_name = templateVertex(ReductionVertexBaseName(root),
                                           graph.getTensorElementType(to_reduce));

  const Window& window(inst->window());

  // Find the number of windows in each dimension
  std::vector<unsigned> window_count(ShapeUtil::Rank(output_shape));
  for (unsigned d=0; d<window.dimensions().size(); d++) {
    std::size_t input_dim(to_reduce.dim(d));
    input_dim += window.dimensions(d).padding_low();
    input_dim += window.dimensions(d).padding_high();

    window_count[d] = window_util::StridedBound(input_dim,
                                                window.dimensions(d).size(),
                                                window.dimensions(d).stride());
  }

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->name());
  const unsigned long N = out.dim(0);
  const auto &device_info = graph.getDevice().getDeviceInfo();

  unsigned dim_count(to_reduce.rank());

  // Vector for walking the window through the tensor
  std::vector<std::size_t> pos(dim_count, 0);

  // Slice boundaries
  std::vector<std::size_t> start(dim_count);
  std::vector<std::size_t> end(dim_count);

  for (unsigned i = 0; i < N; ++i) {
    // Find the window
    for (unsigned d=0; d<dim_count; d++) {
      const auto& wd(window.dimensions(d));

      int s(pos[d] * wd.stride() - wd.padding_low());
      int e(s + wd.size());
      start[d] = std::max(s, 0);
      end[d] = std::min(e, (int)to_reduce.dim(d));
    }

    poplar::Tensor w = to_reduce.slice(start, end).flatten();

    // Create the vertex
    auto v = graph.addVertex(cs, vertex_name, {{"a", w}, {"out", out.slice(i,i+1)}});
    graph.setTileMapping(v, (i / device_info.numWorkerContexts) % device_info.getNumTiles());

    // Advance the window
    for (int d=dim_count-1; d>=0; d--) {
      pos[d]++;
      if (pos[d] < window_count[d]) break;
      pos[d] = 0;
    }
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateSimpleSelectAndScatter(poplar::Graph &graph,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor to_reduce;
  TF_ASSIGN_OR_RETURN(to_reduce, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor init_val;
  TF_ASSIGN_OR_RETURN(init_val, FindInstructionInput(tensor_map, inst, 1, 0));

  // Find the type and vertex
  HloInstruction* root(inst->to_apply()->root_instruction());
  std::string vertex_name = templateVertex(ReductionVertexBaseName(root),
                                           graph.getTensorElementType(to_reduce));

  const Window& window(inst->window());

  // Find the number of windows in each dimension
  std::vector<unsigned> window_count(ShapeUtil::Rank(output_shape));
  for (unsigned d=0; d<window.dimensions().size(); d++) {
    std::size_t input_dim(to_reduce.dim(d));
    input_dim += window.dimensions(d).padding_low();
    input_dim += window.dimensions(d).padding_high();

    window_count[d] = window_util::StridedBound(input_dim,
                                                window.dimensions(d).size(),
                                                window.dimensions(d).stride());
  }

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->name());
  const unsigned long N = out.dim(0);
  const auto &device_info = graph.getDevice().getDeviceInfo();

  unsigned dim_count(to_reduce.rank());

  // Vector for walking the window through the tensor
  std::vector<std::size_t> pos(dim_count, 0);

  // Slice boundaries
  std::vector<std::size_t> start(dim_count);
  std::vector<std::size_t> end(dim_count);

  for (unsigned i = 0; i < N; ++i) {
    // Find the window
    for (unsigned d=0; d<dim_count; d++) {
      const auto& wd(window.dimensions(d));

      int s(pos[d] * wd.stride() - wd.padding_low());
      int e(s + wd.size());
      start[d] = std::max(s, 0);
      end[d] = std::min(e, (int)to_reduce.dim(d));
    }

    poplar::Tensor w = to_reduce.slice(start, end).flatten();

    // Create the vertex
    auto v = graph.addVertex(cs, vertex_name, {{"a", w}, {"out", out.slice(i,i+1)}});
    graph.setTileMapping(v, (i / device_info.numWorkerContexts) % device_info.getNumTiles());

    // Advance the window
    for (int d=dim_count-1; d>=0; d--) {
      pos[d]++;
      if (pos[d] < window_count[d]) break;
      pos[d] = 0;
    }
  }

  return poplar::program::Execute(cs);
}

}
}

