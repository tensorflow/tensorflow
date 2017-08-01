#include <algorithm>
#include <limits>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
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
#include <popnn/PoolingDef.hpp>
#include <popnn/Pooling.hpp>

namespace xla {
namespace poplarplugin {



static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string out_conn("out");

static const std::string reduction_add("ReductionAdd");
static const std::string reduction_mul("ReductionMul");
static const std::string reduction_max("ReductionMax");
static const std::string reduction_min("ReductionMin");
static const std::string reduction_and("ReductionAnd");
static const std::string reduction_or("ReductionOr");

static const std::string reduction_ge("SelectionGe");
static const std::string reduction_gt("SelectionGt");
static const std::string reduction_le("SelectionLe");
static const std::string reduction_lt("SelectionLt");

static const std::string unknown("Unknown");

bool
IsReducableArtithmetic(const HloInstruction* inst,
                       const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kLogicalAnd:
    case HloOpcode::kLogicalOr:
      return true;
    default:
      return false;
  }
}

bool
IsSimpleSelection(const HloInstruction* inst,
                  const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
      return true;
    default:
      return false;
  }
}

bool
IsPoplibsPool(const HloInstruction* inst,
              const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kMaximum:
    case HloOpcode::kAdd:
      break;
    default:
      return false;
  }

  if (ShapeUtil::Rank(inst->shape()) != 4) {
    return false;
  }

  const Window& window(inst->window());
  if (window.dimensions(0).size() != 1 ||
      window.dimensions(0).stride() != 1 ||
      window.dimensions(0).padding_low() != 0 ||
      window.dimensions(0).padding_high() != 0 ||
      window.dimensions(3).size() != 1 ||
      window.dimensions(3).stride() != 1 ||
      window.dimensions(3).padding_low() != 0 ||
      window.dimensions(3).padding_high() != 0) {
    return false;
  }

  return true;
}

static const std::string&
ReductionVertexBaseName(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
      return reduction_add;
    case HloOpcode::kMultiply:
      return reduction_mul;
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
      return unknown;
  }
}

static const std::string&
SelectionVertexBaseName(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kGe:
      return reduction_ge;
    case HloOpcode::kGt:
      return reduction_gt;
    case HloOpcode::kLe:
      return reduction_le;
    case HloOpcode::kLt:
      return reduction_lt;
    default:
      // Cannot reach here
      return unknown;
  }
}

static std::vector<int64>
MaxWindowOverlap(const Window& window) {
  std::vector<int64> overlap;
  for (auto& d : window.dimensions()) {
    int64 o = ((d.size() + d.stride() - 1) / d.stride());
    overlap.push_back(o);
  }
  return overlap;
}

template<typename Tpos, typename Tlimit>
static std::size_t
GetOverlapLayerNum(const Tpos& pos,
                   const Tlimit& limit) {
  std::size_t layer = 0;
  std::size_t mult = 1;
  for (size_t d=0; d<pos.size(); d++) {
    std::size_t v = (pos[d] % limit[d]) * mult;
    layer += v;
    mult *= limit[d];
  }
  return layer;
}

port::StatusOr<poplar::program::Program>
CreateSimpleReduction(poplar::Graph &graph,
                      CompilerResources& res,
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
                                           to_reduce.elementType());


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
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst, output_shape, res));
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
                            CompilerResources& res,
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
                                           to_reduce.elementType());

  const Window& window(inst->window());

  // Find the number of windows in each dimension
  std::vector<unsigned> window_count(ShapeUtil::Rank(output_shape));
  for (int64 d=0; d<window.dimensions().size(); d++) {
    std::size_t input_dim(to_reduce.dim(d));
    input_dim += window.dimensions(d).padding_low();
    input_dim += window.dimensions(d).padding_high();

    window_count[d] = window_util::StridedBound(input_dim,
                                                window.dimensions(d).size(),
                                                window.dimensions(d).stride());
  }

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst, output_shape, res));
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
CreatePoplibsWindowReduction(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map) {
  const HloInstruction* pooling_inst;

  PoolingType reduction_type;

  // Find the type of the reduction
  if (inst->opcode() == HloOpcode::kFusion) {
    reduction_type = PoolingType::AVG;
    pooling_inst = inst->fused_expression_root()->operand(0);
  } else if (inst->to_apply()->root_instruction()->opcode() ==
          HloOpcode::kMaximum) {
    reduction_type = PoolingType::MAX;
    pooling_inst = inst;
  } else {
    reduction_type = PoolingType::SUM;
    pooling_inst = inst;
  }

  // Find the input tensors
  poplar::Tensor to_reduce;
  TF_ASSIGN_OR_RETURN(to_reduce, FindInstructionInput(tensor_map, inst, 0, 0));

  const Window& window(pooling_inst->window());
  std::vector<std::size_t> kernel_shape = {
    (std::size_t)window.dimensions(1).size(),
    (std::size_t)window.dimensions(2).size()
  };
  std::vector<unsigned> stride = {
    (unsigned)window.dimensions(1).stride(),
    (unsigned)window.dimensions(2).stride()
  };
  std::vector<int> padding_lower = {
    (int)window.dimensions(1).padding_low(),
    (int)window.dimensions(2).padding_low()
  };
  std::vector<int> padding_upper = {
    (int)window.dimensions(1).padding_high(),
    (int)window.dimensions(2).padding_high()
  };

  poplar::program::Sequence prog;
  poplar::Tensor out = popnn::pooling::pool(graph, reduction_type,
                                            kernel_shape, stride,
                                            padding_lower, padding_upper,
                                            to_reduce, prog, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return prog;
}

port::StatusOr<poplar::program::Program>
CreateSimpleSelectAndScatter(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map) {

  poplar::program::Sequence program_seq;

  // Find the input tensors
  poplar::Tensor operand;
  TF_ASSIGN_OR_RETURN(operand, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor source;
  TF_ASSIGN_OR_RETURN(source, FindInstructionInput(tensor_map, inst, 1, 0));

  poplar::Tensor init_val;
  TF_ASSIGN_OR_RETURN(init_val, FindInstructionInput(tensor_map, inst, 2, 0));

  /*
   * Selection
   */

  HloInstruction* select_root(inst->select()->root_instruction());
  std::string select_vertex_name =
          templateVertex(SelectionVertexBaseName(select_root),
                         operand.elementType());

  const Window& window(inst->window());

  std::vector<int64> overlap(MaxWindowOverlap(window));
  int64 overlap_count(std::accumulate(overlap.begin(), overlap.end(), 1,
                                      [](int64 a, int64 b) { return a * b; }));

  xla::Shape partial_shape(output_shape);
  partial_shape.add_dimensions(overlap_count);

  poplar::Tensor partial;
  TF_ASSIGN_OR_RETURN(partial,
                      AddPlainTensor(graph, inst, partial_shape));

  poplar::Tensor init;
  TF_ASSIGN_OR_RETURN(init, BroadcastTensor(init_val, partial_shape));

  program_seq.add(poplar::program::Copy(init, partial));

  // Find the number of windows in each dimension
  std::vector<unsigned> window_count(ShapeUtil::Rank(output_shape));
  for (int64 d=0; d<window.dimensions().size(); d++) {
    std::size_t input_dim(operand.dim(d));
    input_dim += window.dimensions(d).padding_low();
    input_dim += window.dimensions(d).padding_high();

    window_count[d] = window_util::StridedBound(input_dim,
                                                window.dimensions(d).size(),
                                                window.dimensions(d).stride());
  }

  auto select_cs = graph.addComputeSet(inst->name());
  program_seq.add(poplar::program::Execute(select_cs));

  const unsigned long num_windows = source.numElements();
  const auto &device_info = graph.getDevice().getDeviceInfo();

  unsigned dim_count(operand.rank());

  // Vector for walking the window through the tensor
  std::vector<std::size_t> pos(dim_count, 0);

  // Slice boundaries
  std::vector<std::size_t> start_in(dim_count);
  std::vector<std::size_t> end_in(dim_count);

  std::vector<std::size_t> start_par(dim_count+1);
  std::vector<std::size_t> end_par(dim_count+1);

  for (unsigned i = 0; i < num_windows; ++i) {
    // Find the windows
    for (unsigned d=0; d<dim_count; d++) {
      const auto& wd(window.dimensions(d));

      int s(pos[d] * wd.stride() - wd.padding_low());
      int e(s + wd.size());
      start_in[d] = std::max(s, 0);
      end_in[d] = std::min(e, (int)operand.dim(d));

      start_par[d] = start_in[d];
      end_par[d] = end_in[d];
    }
    start_par[dim_count] = GetOverlapLayerNum(pos, overlap);
    end_par[dim_count] = start_par[dim_count] + 1;

    // TODO - move this into poplar
    std::vector<std::size_t> pos_plus_one(dim_count);
    for (unsigned d=0; d<dim_count; d++) {
      pos_plus_one[d] = pos[d] + 1;
    }

    poplar::Tensor w_in = operand.slice(start_in, end_in).flatten();
    poplar::Tensor w_par = partial.slice(start_par, end_par).flatten();
    poplar::Tensor s = source.slice(pos, pos_plus_one).flatten();

    // Create the vertex
    auto v = graph.addVertex(select_cs, select_vertex_name,
                             {{"a", w_in},
                              {"b", s},
                              {"out", w_par}});
    graph.setTileMapping(v,
                         (i / device_info.numWorkerContexts) %
                                 device_info.getNumTiles());

    // Advance the window
    for (int d=dim_count-1; d>=0; d--) {
      pos[d]++;
      if (pos[d] < window_count[d]) break;
      pos[d] = 0;
    }
  }

  /*
   * Reduction
   */

  HloInstruction* scatter_root(inst->scatter()->root_instruction());
  std::string scatter_vertex_name =
          templateVertex(ReductionVertexBaseName(scatter_root),
                         operand.elementType());

  std::vector<size_t> reshaped(2);
  reshaped[0] = partial.numElements() / overlap_count;
  reshaped[1] = overlap_count;
  partial = partial.reshape(reshaped);

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst, output_shape, res));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  // One vertex per non-reduced element
  auto scatter_cs = graph.addComputeSet(inst->name());
  program_seq.add(poplar::program::Execute(scatter_cs));

  const unsigned long num_reductions = out.dim(0);

  for (unsigned i = 0; i < num_reductions; ++i) {
    auto v = graph.addVertex(scatter_cs, scatter_vertex_name,
                             {{"a", partial[i]},
                              {"out", out.slice(i, i+1)}});
    graph.setTileMapping(v, (i / device_info.numWorkerContexts)
                            % device_info.getNumTiles());
  }

  return program_seq;
}

}
}

