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
#include <popreduce/Reduce.hpp>

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
IsReducableArtithmetic(const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());
  if (!hlo_query::AllOperandsAreParameters(*root)) {
    return false;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
      return true;
    default:
      return false;
  }
}

bool
IsSimpleSelection(const HloComputation* computation) {
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
      window.dimensions(1).size() != 1 ||
      window.dimensions(1).stride() != 1 ||
      window.dimensions(1).padding_low() != 0 ||
      window.dimensions(1).padding_high() != 0) {
    return false;
  }

  return true;
}

static Literal
GetIdentityConstantLiteral(const HloInstruction* root) {
  switch (root->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    default:
      return Literal::Zero(root->shape().element_type());
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
      return Literal::One(root->shape().element_type());
    case HloOpcode::kMaximum:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
      return Literal::MinValue(root->shape().element_type());
    case HloOpcode::kMinimum:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
      return Literal::MaxValue(root->shape().element_type());
  }
}

static popreduce::Operation
PoplibsReductionOperation(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
      return popreduce::Operation::ADD;
    case HloOpcode::kMultiply:
      return popreduce::Operation::MUL;
    case HloOpcode::kMaximum:
      return popreduce::Operation::MAX;
    case HloOpcode::kMinimum:
      return popreduce::Operation::MIN;
    case HloOpcode::kAnd:
      return popreduce::Operation::AND;
    case HloOpcode::kOr:
      return popreduce::Operation::OR;
    default:
      // Cannot reach here
      return popreduce::Operation::ADD;
  }
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
    case HloOpcode::kAnd:
      return reduction_and;
    case HloOpcode::kOr:
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
  poplar::program::Sequence seq;
  poplar::Tensor out;

  if (ShapeUtil::HasZeroElements(inst->operand(0)->shape())) {
    TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 1));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  } else {
    // Find the input tensors
    poplar::Tensor to_reduce;
    TF_ASSIGN_OR_RETURN(to_reduce, FindInstructionInput(tensor_map, inst, 0));

    HloInstruction* root(inst->to_apply()->root_instruction());
    popreduce::Operation op = PoplibsReductionOperation(root);

    std::vector<std::size_t> reduction_dims;
    for (auto d : inst->dimensions()) {
      reduction_dims.push_back(d);
    }

    poplar::Tensor out = popreduce::reduce(graph, to_reduce, reduction_dims,
                                           op, seq, inst->name());

    // Apply initial value
    Literal identity_literal = GetIdentityConstantLiteral(root);
    auto* init_inst = inst->operand(1);
    if (!(init_inst->IsConstant() &&
          init_inst->literal() == identity_literal)) {

      poplar::Tensor init_val;
      TF_ASSIGN_OR_RETURN(init_val, FindInstructionInput(tensor_map, inst, 1));

      // Create a binary op with the scatter_root opcode
      TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

      popstd_binary_fn fn;
      TF_ASSIGN_OR_RETURN(fn, LookupBinaryFn(root));

      out = fn(graph, out, init_val, seq, inst->name() + "_initval");
    }

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  }

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateSimpleWindowReduction(poplar::Graph &graph,
                            CompilerResources& res,
                            const HloInstruction *inst,
                            const xla::Shape& output_shape,
                            TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  poplar::Tensor out;

  if (ShapeUtil::HasZeroElements(inst->operand(0)->shape())) {
    TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 1));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  } else {
    // Find the input tensors
    poplar::Tensor to_reduce;
    TF_ASSIGN_OR_RETURN(to_reduce, FindInstructionInput(tensor_map, inst, 0));

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
    TF_ASSIGN_OR_RETURN(out, AddTensor(graph, std::make_pair(inst,0), output_shape, res));
    poplar::Tensor out_flat = out.flatten();

    auto cs = graph.addComputeSet(inst->name());
    const unsigned long N = out_flat.dim(0);

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
        start[d] = std::min(std::max(s, 0), (int)to_reduce.dim(d));
        end[d] = std::min(std::max(e, 0), (int)to_reduce.dim(d));
      }

      poplar::Tensor w = to_reduce.slice(start, end).flatten();

      // Create the vertex
      auto v = graph.addVertex(cs,
                               vertex_name,
                               {{"a", w},
                                {"out", out_flat.slice(i,i+1)}});
      graph.setTileMapping(v, (i / graph.getTarget().getNumWorkerContexts())
                              % graph.getTarget().getNumTiles());

      // Advance the window
      for (int d=dim_count-1; d>=0; d--) {
        pos[d]++;
        if (pos[d] < window_count[d]) break;
        pos[d] = 0;
      }
    }

    seq.add(poplar::program::Execute(cs));

    // Apply initial value
    Literal identity_literal = GetIdentityConstantLiteral(root);
    auto* init_inst = inst->operand(1);
    if (!(init_inst->IsConstant() &&
          init_inst->literal() == identity_literal)) {

      poplar::Tensor init_val;
      TF_ASSIGN_OR_RETURN(init_val, FindInstructionInput(tensor_map, inst, 1));

      // Create a binary op with the scatter_root opcode
      TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

      popstd_binary_fn fn;
      TF_ASSIGN_OR_RETURN(fn, LookupBinaryFn(root));

      out = fn(graph, out, init_val, seq, inst->name() + "_initval");
    }
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  }

  return seq;
}

port::StatusOr<poplar::program::Program>
CreatePoplibsWindowReduction(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map) {
  poplar::program::Sequence prog;
  poplar::Tensor out;

  if (ShapeUtil::HasZeroElements(inst->operand(0)->shape())) {
    TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 1));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  } else {
    const HloInstruction* pooling_inst;

    popnn::PoolingType reduction_type;

    // Find the type of the reduction
    if (inst->opcode() == HloOpcode::kCall) {
      reduction_type = popnn::PoolingType::AVG;
      pooling_inst = inst->to_apply()->root_instruction()->operand(0);
    } else if (inst->to_apply()->root_instruction()->opcode() ==
               HloOpcode::kMaximum) {
      reduction_type = popnn::PoolingType::MAX;
      pooling_inst = inst;
    } else {
      reduction_type = popnn::PoolingType::SUM;
      pooling_inst = inst;
    }

    // Find the input tensors
    poplar::Tensor to_reduce;
    TF_ASSIGN_OR_RETURN(to_reduce, FindInstructionInput(tensor_map, inst, 0));

    // Find which dimensions are being reduced
    const Window& window(pooling_inst->window());
    std::set<unsigned int> reduction_dims;
    for (int64 i=0; i<window.dimensions_size(); i++) {
      auto& d = window.dimensions(i);
      if (d.size() != 1 ||
          d.stride() != 1 ||
          d.padding_low() != 0 ||
          d.padding_high() != 0) {
        reduction_dims.insert(i);
      }
    }

    if (reduction_dims.size() == 0) {
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, to_reduce));
      return prog;
    }

    if (reduction_dims.size() == 1) {
      if (reduction_dims.count(window.dimensions_size()-1) == 0) {
        reduction_dims.insert(window.dimensions_size()-1);
      } else {
        reduction_dims.insert(window.dimensions_size()-2);
      }
    }

    if (reduction_dims.size() != 2) {
      return port::Status(port::error::FAILED_PRECONDITION,
                          "poplar pooling only supports 2D pooling");
    }

    std::vector<std::size_t> kernel_shape;
    std::vector<unsigned> stride;
    std::vector<int> padding_lower;
    std::vector<int> padding_upper;
    for (auto i=reduction_dims.begin(); i!=reduction_dims.end(); i++) {
      auto& d = window.dimensions(*i);
      kernel_shape.push_back((std::size_t)d.size());
      stride.push_back((unsigned)d.stride());
      padding_lower.push_back((int)d.padding_low());
      padding_upper.push_back((int)d.padding_high());
    }

    std::vector<unsigned int> shuffle_in;
    for (int i=0; i<window.dimensions_size(); i++) {
      if (reduction_dims.count(i) == 0) {
        shuffle_in.push_back(i);
      }
    }
    shuffle_in.insert(shuffle_in.end(),
                      reduction_dims.begin(),
                      reduction_dims.end());
    to_reduce = to_reduce.dimShuffle(shuffle_in);

    out = popnn::pooling::pool(graph, reduction_type,
                               kernel_shape, stride,
                               padding_lower, padding_upper,
                               to_reduce, prog, inst->name());

    std::vector<unsigned int> shuffle_out(shuffle_in.size());
    for (int i=0; i<window.dimensions_size(); i++) {
      shuffle_out[shuffle_in[i]] = i;
    }
    out = out.dimShuffle(shuffle_out);

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  }

  return prog;
}

port::StatusOr<poplar::program::Program>
CreateSimpleSelectAndScatter(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map) {

  poplar::Tensor out;
  poplar::program::Sequence program_seq;

  // Find the input tensors
  poplar::Tensor operand;
  TF_ASSIGN_OR_RETURN(operand, FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor source;
  TF_ASSIGN_OR_RETURN(source, FindInstructionInput(tensor_map, inst, 1));

  HloInstruction* select_root(inst->select()->root_instruction());
  HloInstruction* scatter_root(inst->scatter()->root_instruction());

  /*
   * Selection
   */

  std::string select_vertex_name =
          templateVertex(SelectionVertexBaseName(select_root),
                         operand.elementType());

  const Window& window(inst->window());

  std::vector<int64> overlap(MaxWindowOverlap(window));
  int64 overlap_count(std::accumulate(overlap.begin(), overlap.end(), 1,
                                      [](int64 a, int64 b) { return a * b; }));

  // Create a partials tensor for reduction
  std::vector<std::size_t> poplar_shape = operand.shape();
  poplar_shape.push_back(1);

  poplar::Tensor extended_operand = operand.reshape(poplar_shape);
  poplar::Tensor partial = graph.clone(extended_operand);

  for (int64 i=1; i<overlap_count; i++) {
    partial = poplar::concat(partial, graph.clone(extended_operand), partial.rank() - 1);
  }

  xla::Shape partial_shape(output_shape);
  partial_shape.add_dimensions(overlap_count);
  LayoutUtil::ClearLayout(&partial_shape);
  partial_shape.mutable_layout()->set_format(DENSE);

  Literal identity_literal = GetIdentityConstantLiteral(scatter_root);

  poplar::Tensor identity_val;
  TF_ASSIGN_OR_RETURN(identity_val,
                      AddConstantTensor(graph,
                                        std::make_pair(inst, 0),
                                        partial_shape,
                                        identity_literal,
                                        res));
  program_seq.add(poplar::program::Copy(identity_val, partial));

  // Find the number of windows in each dimension
  std::vector<unsigned> window_count(ShapeUtil::Rank(output_shape));
  for (int64 d=0; d<window.dimensions().size(); d++) {
    std::size_t input_dim(operand.dim(d));
    input_dim += window.dimensions(d).padding_low();
    input_dim += window.dimensions(d).padding_high();

    window_count[d] =
            window_util::StridedBound(input_dim,
                                      window.dimensions(d).size(),
                                      window.dimensions(d).stride());
  }

  auto select_cs = graph.addComputeSet(inst->name() + "_select");
  program_seq.add(poplar::program::Execute(select_cs));

  const unsigned long num_windows = source.numElements();

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
      start_in[d] = std::min(std::max(s, 0), (int)operand.dim(d));
      end_in[d] = std::min(std::max(e, 0), (int)operand.dim(d));

      start_par[d] = start_in[d];
      end_par[d] = end_in[d];
    }
    start_par[dim_count] = GetOverlapLayerNum(pos, overlap);
    end_par[dim_count] = start_par[dim_count] + 1;

    poplar::Tensor w_in = operand.slice(start_in, end_in).flatten();
    poplar::Tensor w_par = partial.slice(start_par, end_par).flatten();
    poplar::Tensor s = source.index(pos);

    auto m = graph.getTileMapping(w_in);
    unsigned int tile_with_max_elements = 0;
    std::size_t max_elements = 0;
    for (unsigned int t = 0; t<m.size(); t++) {
      std::size_t element_count = 0;
      for (auto interval : m[t]) {
        element_count += interval.size();
      }
      if (element_count > max_elements) {
        max_elements = element_count;
        tile_with_max_elements = t;
      }
    }

    // Create the vertex
    auto v = graph.addVertex(select_cs, select_vertex_name,
                             {{"a", w_in},
                              {"b", s},
                              {"out", w_par}});
    TF_RETURN_IF_ERROR(SetVertexField(graph, v["initval"], identity_literal));
    graph.setTileMapping(v, tile_with_max_elements);

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
  popreduce::Operation op = PoplibsReductionOperation(scatter_root);

  std::vector<std::size_t> reduction_dims;
  reduction_dims.push_back(partial.rank() - 1);

  out = popreduce::reduce(graph, partial, reduction_dims, op, program_seq,
                          inst->name() + "_reduce");

  /*
   * Initial value application
   */
  auto* init_inst = inst->operand(2);
  if (!(init_inst->IsConstant() &&
        init_inst->literal() == identity_literal)) {

    poplar::Tensor init_val;
    TF_ASSIGN_OR_RETURN(init_val, FindInstructionInput(tensor_map, inst, 2));

    // Create a binary op with the scatter_root opcode
    TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

    popstd_binary_fn fn;
    TF_ASSIGN_OR_RETURN(fn, LookupBinaryFn(scatter_root));

    out = fn(graph, out, init_val, program_seq, inst->name() + "_initval");
  }

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return program_seq;
}

}
}

