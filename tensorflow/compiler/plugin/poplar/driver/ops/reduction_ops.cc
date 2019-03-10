#include <algorithm>
#include <limits>

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/window_util.h"

#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>

using ::absl::StrCat;

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

bool IsReducableArtithmetic(const HloComputation* computation) {
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

bool IsSimpleSelection(const HloComputation* computation) {
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

bool IsPoplibsPool(const HloInstruction* inst,
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

  if (inst->shape().rank() != 4) {
    return false;
  }

  const Window& window(inst->window());
  unsigned reduction_count = 0;
  for (int64 i = 0; i < window.dimensions_size(); i++) {
    auto& d = window.dimensions(i);
    if (d.size() != 1 || d.stride() != 1 || d.padding_low() != 0 ||
        d.padding_high() != 0) {
      reduction_count++;
    }
  }

  return (reduction_count <= 2);
}

static Literal GetIdentityConstantLiteral(const HloOpcode opcode,
                                          const HloInstruction* reduce) {
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    default:
      return LiteralUtil::Zero(reduce->shape().element_type());
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
      return LiteralUtil::One(reduce->shape().element_type());
    case HloOpcode::kMaximum:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
      return LiteralUtil::MinValue(reduce->shape().element_type());
    case HloOpcode::kMinimum:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
      return LiteralUtil::MaxValue(reduce->shape().element_type());
  }
}

static Literal GetIdentityConstantLiteral(const HloInstruction* root,
                                          const HloInstruction* reduce) {
  return GetIdentityConstantLiteral(root->opcode(), reduce);
}

static popops::Operation PoplibsReductionOperation(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
      return popops::Operation::ADD;
    case HloOpcode::kMultiply:
      return popops::Operation::MUL;
    case HloOpcode::kMaximum:
      return popops::Operation::MAX;
    case HloOpcode::kMinimum:
      return popops::Operation::MIN;
    case HloOpcode::kAnd:
      return popops::Operation::LOGICAL_AND;
    case HloOpcode::kOr:
      return popops::Operation::LOGICAL_OR;
    default:
      // Cannot reach here
      return popops::Operation::ADD;
  }
}

static const std::string& ReductionVertexBaseName(const HloInstruction* inst) {
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

static const std::string& SelectionVertexBaseName(const HloInstruction* inst) {
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

static std::vector<int64> MaxWindowOverlap(const Window& window) {
  std::vector<int64> overlap;
  for (auto& d : window.dimensions()) {
    int64 o = ((d.size() + d.stride() - 1) / d.stride());
    overlap.push_back(o);
  }
  return overlap;
}

template <typename Tpos, typename Tlimit>
static std::size_t GetOverlapLayerNum(const Tpos& pos, const Tlimit& limit) {
  std::size_t layer = 0;
  std::size_t mult = 1;
  for (size_t d = 0; d < pos.size(); d++) {
    std::size_t v = (pos[d] % limit[d]) * mult;
    layer += v;
    mult *= limit[d];
  }
  return layer;
}

std::set<unsigned int> GetPoolingReductionDims(const Window& window) {
  std::set<unsigned int> reduction_dims;
  for (int64 i = 0; i < window.dimensions_size(); i++) {
    auto& d = window.dimensions(i);
    if (d.size() != 1 || d.stride() != 1 || d.padding_low() != 0 ||
        d.padding_high() != 0) {
      reduction_dims.insert(i);
    }
  }

  // TODO make sure this works with T5956.
  // When we have a single reduction dimension, add an adjacent dimension for
  // reduction as well.
  if (reduction_dims.size() == 1) {
    if (reduction_dims.count(window.dimensions_size() - 1) == 0) {
      reduction_dims.insert(window.dimensions_size() - 1);
    } else {
      reduction_dims.insert(window.dimensions_size() - 2);
    }
  }
  return reduction_dims;
}

static std::vector<unsigned int> GetShuffleInputDimensionsForPoplar(
    const Window& window, const std::set<unsigned int> reduction_dims) {
  std::vector<unsigned int> shuffle_in;
  for (int i = 0; i < window.dimensions_size(); i++) {
    if (reduction_dims.count(i) == 0) {
      shuffle_in.push_back(i);
    }
  }
  shuffle_in.insert(shuffle_in.end(), reduction_dims.begin(),
                    reduction_dims.end());
  return shuffle_in;
}

static std::vector<unsigned int> GetShuffleOutputDimensionsForPoplar(
    const std::vector<unsigned int> shuffle_in) {
  std::vector<unsigned int> shuffle_out(shuffle_in.size());
  for (int i = 0; i < shuffle_in.size(); i++) {
    shuffle_out[shuffle_in[i]] = i;
  }
  return shuffle_out;
}

static poplar::Type GetReductionType(const popnn::PoolingType& pooling_type,
                                     const poplar::Type& input_type) {
  switch (pooling_type) {
    case popnn::PoolingType::AVG:
    case popnn::PoolingType::SUM:
      return (input_type == poplar::HALF) ? poplar::FLOAT : input_type;
    case popnn::PoolingType::MAX:
    default:
      return input_type;
  }
}

static popnn::pooling::PoolParams GetPoplibsPoolParams(
    const popnn::PoolingType& pooling_type, const Window& window,
    const std::vector<std::size_t>& input_shape,
    const std::set<unsigned int>& reduction_dims,
    const poplar::Type& input_data_type) {
  // TODO assume here that batch dimension and the channel dimension order
  // doesn't actually matter - it's just the non field dimensions.
  const auto batch_size = input_shape.front();
  const auto num_channels = input_shape[1];
  std::vector<std::size_t> input_field_shape(std::next(input_shape.begin(), 2),
                                             input_shape.end());

  std::vector<std::size_t> kernel_shape;
  std::vector<unsigned> stride;
  std::vector<int> padding_lower;
  std::vector<int> padding_upper;

  for (auto reduction_dim : reduction_dims) {
    auto& d = window.dimensions(reduction_dim);
    kernel_shape.push_back((std::size_t)d.size());
    stride.push_back((unsigned)d.stride());
    padding_lower.push_back((int)d.padding_low());
    padding_upper.push_back((int)d.padding_high());
  }

  return {pooling_type, input_field_shape, kernel_shape,
          stride,       padding_lower,     padding_upper,
          num_channels, batch_size,        input_data_type};
}

StatusOr<poplar::program::Program> CreateSimpleReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  poplar::Tensor out;

  poplar::Graph& graph = GetGraph(res, inst);

  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    TF_ASSIGN_OR_RETURN(out,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  } else {
    // Find the input tensors
    poplar::Tensor to_reduce;
    TF_ASSIGN_OR_RETURN(to_reduce,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));

    HloInstruction* root(inst->to_apply()->root_instruction());
    popops::Operation op = PoplibsReductionOperation(root);

    std::vector<std::size_t> reduction_dims;
    for (auto d : inst->dimensions()) {
      reduction_dims.push_back(d);
    }

    poplar::Tensor out = popops::reduce(graph, to_reduce, reduction_dims, op,
                                        seq, GetDebugName(inst));

    // Apply initial value
    Literal identity_literal = GetIdentityConstantLiteral(root, inst);
    auto* init_inst = inst->operand(1);
    if (!(init_inst->IsConstant() &&
          init_inst->literal() == identity_literal)) {
      poplar::Tensor init_val;
      TF_ASSIGN_OR_RETURN(init_val,
                          FindInstructionInput(tensor_map, res, inst, 1, seq));

      // Create a binary op with the scatter_root opcode
      TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

      popops::expr::BinaryOpType op;
      TF_ASSIGN_OR_RETURN(op, LookupBinaryFn(root));

      popops::mapInPlace(graph, op, out, init_val, seq,
                         GetDebugName(inst) + "_initval");
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateSimpleWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::program::Sequence seq;
  poplar::Tensor out;

  poplar::Graph& graph = GetGraph(res, inst);

  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    TF_ASSIGN_OR_RETURN(out,
                        FindInstructionInput(tensor_map, res, inst, 1, seq));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  } else {
    // Find the input tensors
    poplar::Tensor to_reduce;
    TF_ASSIGN_OR_RETURN(to_reduce,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));

    // Find the type and vertex
    HloInstruction* root(inst->to_apply()->root_instruction());
    std::string vertex_name =
        templateVertex(ReductionVertexBaseName(root), to_reduce.elementType());

    const Window& window(inst->window());

    // Find the number of windows in each dimension
    std::vector<unsigned> window_count(output_shape.rank());
    for (int64 d = 0; d < window.dimensions().size(); d++) {
      std::size_t input_dim(to_reduce.dim(d));
      input_dim += window.dimensions(d).padding_low();
      input_dim += window.dimensions(d).padding_high();

      window_count[d] =
          window_util::StridedBound(input_dim, window.dimensions(d).size(),
                                    window.dimensions(d).stride());
    }

    // Allocate the output tensor
    TF_ASSIGN_OR_RETURN(out, AddTensor(graph, std::make_pair(inst, 0),
                                       output_shape, res, tensor_map));
    poplar::Tensor out_flat = out.flatten();

    auto cs = graph.addComputeSet(GetDebugName(inst));
    const unsigned long N = out_flat.dim(0);

    unsigned dim_count(to_reduce.rank());

    // Vector for walking the window through the tensor
    std::vector<std::size_t> pos(dim_count, 0);

    // Slice boundaries
    std::vector<std::size_t> start(dim_count);
    std::vector<std::size_t> end(dim_count);

    for (unsigned i = 0; i < N; ++i) {
      // Find the window
      for (unsigned d = 0; d < dim_count; d++) {
        const auto& wd(window.dimensions(d));

        int s(pos[d] * wd.stride() - wd.padding_low());
        int e(s + wd.size());
        start[d] = std::min(std::max(s, 0), (int)to_reduce.dim(d));
        end[d] = std::min(std::max(e, 0), (int)to_reduce.dim(d));
      }

      poplar::Tensor w = to_reduce.slice(start, end).flatten();

      // Create the vertex
      auto v = graph.addVertex(cs, vertex_name,
                               {{"a", w}, {"out", out_flat.slice(i, i + 1)}});
      graph.setTileMapping(v, (i / graph.getTarget().getNumWorkerContexts()) %
                                  graph.getTarget().getNumTiles());
      graph.setCycleEstimate(v, 1);

      // Advance the window
      for (int d = dim_count - 1; d >= 0; d--) {
        pos[d]++;
        if (pos[d] < window_count[d]) break;
        pos[d] = 0;
      }
    }

    seq.add(poplar::program::Execute(cs));

    // Apply initial value
    Literal identity_literal = GetIdentityConstantLiteral(root, inst);
    auto* init_inst = inst->operand(1);
    if (!(init_inst->IsConstant() &&
          init_inst->literal() == identity_literal)) {
      poplar::Tensor init_val;
      TF_ASSIGN_OR_RETURN(init_val,
                          FindInstructionInput(tensor_map, res, inst, 1, seq));

      // Create a binary op with the scatter_root opcode
      TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

      popops::expr::BinaryOpType op;
      TF_ASSIGN_OR_RETURN(op, LookupBinaryFn(root));

      popops::mapInPlace(graph, op, out, init_val, seq,
                         GetDebugName(inst) + "_initval");
    }
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreatePoplibsWindowReduction(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  if (ShapeUtil::IsZeroElementArray(inst->operand(0)->shape())) {
    poplar::program::Sequence prog;
    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        FindInstructionInput(tensor_map, res, inst, 1, prog));
    TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, inst->shape(), {}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return prog;
  } else {
    popnn::PoolingType reduction_type;
    CHECK_EQ(inst->opcode(), HloOpcode::kReduceWindow);
    switch (inst->to_apply()->root_instruction()->opcode()) {
      case HloOpcode::kMaximum: {
        reduction_type = popnn::PoolingType::MAX;
        break;
      }
      case HloOpcode::kAdd: {
        reduction_type = popnn::PoolingType::SUM;
        break;
      }
      default: {
        return xla::FailedPrecondition("Unsupported window reduction %s.",
                                       inst->name());
      }
    }
    return CreatePoplibsPooling(res, inst, tensor_map, reduction_type,
                                inst->window(), inst);
  }
}

StatusOr<poplar::program::Program> CreatePoplibsPooling(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window,
    absl::optional<const HloInstruction*> optional_reduction_op) {
  poplar::Graph& graph = GetGraph(res, inst);
  poplar::program::Sequence prog;

  TF_ASSIGN_OR_RETURN(poplar::Tensor to_reduce,
                      FindInstructionInput(tensor_map, res, inst, 0, prog));
  auto reduction_dims = GetPoolingReductionDims(window);

  if (reduction_dims.size() == 0) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, to_reduce));
    return prog;
  }

  if (reduction_dims.size() != 2) {
    return xla::FailedPrecondition("Popnn pooling only supports 2D pooling.");
  }

  const auto shuffle_in =
      GetShuffleInputDimensionsForPoplar(window, reduction_dims);
  to_reduce = to_reduce.dimShuffle(shuffle_in);

  // TODO(T7321) The default expected behaviour is to do any partial
  // calculations in FP32, however popnn:pooling currently does not support
  // having an FP16 input and FP32 partials. We therefore upcast the input if
  // required and then downcast the result.
  auto input_type = to_reduce.elementType();
  auto reduction_type = GetReductionType(pooling_type, input_type);
  const bool cast_required = input_type != reduction_type;
  // Do the cast to make sure partials are at higher precision.
  if (cast_required) {
    to_reduce = popops::cast(graph, to_reduce, reduction_type, prog,
                             GetDebugName(inst) + "/PreCast");
  }
  auto pool_params = GetPoplibsPoolParams(
      pooling_type, window, to_reduce.shape(), reduction_dims, reduction_type);

  poplar::Tensor out =
      popnn::pooling::pool(graph, pool_params, to_reduce, prog,
                           GetDebugName(inst), res.default_pooling_options);

  // Do the cast to to the original input type.
  if (cast_required) {
    out = popops::cast(graph, out, input_type, prog,
                       GetDebugName(inst) + "/PostCast");
  }

  if (optional_reduction_op) {
    // We apply the initial_value of the pooling in the non-default base
    // case. This needs to be after poplibs pool, as it does not support the
    // non-default base case.
    const HloInstruction* reduction_op = *optional_reduction_op;
    const HloInstruction* reducing_op =
        reduction_op->to_apply()->root_instruction();
    const HloInstruction* initial_value = reduction_op->operand(1);

    // What is the default base case for the reduction_op, MAX: -largest, SUM:
    // 0, etc.
    Literal identity_literal = GetIdentityConstantLiteral(reducing_op, inst);

    // Apply the base case if necessary
    if (!(initial_value->IsConstant() &&
          initial_value->literal() == identity_literal)) {
      TF_ASSIGN_OR_RETURN(poplar::Tensor init_val,
                          FindInstructionInput(tensor_map, res, inst, 1, prog));
      init_val = init_val.reshape({1})
                     .broadcast(out.numElements(), 0)
                     .reshape(out.shape());
      popops::expr::BinaryOpType op;
      TF_ASSIGN_OR_RETURN(op, LookupBinaryFn(reducing_op));
      popops::mapInPlace(graph, op, out, init_val, prog,
                         GetDebugName(reducing_op) + "_initval");
    }
  }

  const auto shuffle_out = GetShuffleOutputDimensionsForPoplar(shuffle_in);
  out = out.dimShuffle(shuffle_out);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return prog;
}

StatusOr<poplar::program::Program> CreatePoplibsMaxPoolGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    const Window& window) {
  poplar::program::Sequence seq;
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::Tensor input;
  TF_ASSIGN_OR_RETURN(input,
                      FindInstructionInput(tensor_map, res, inst, 0, seq));
  poplar::Tensor output;
  TF_ASSIGN_OR_RETURN(output,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));
  poplar::Tensor output_grad;
  TF_ASSIGN_OR_RETURN(output_grad,
                      FindInstructionInput(tensor_map, res, inst, 2, seq));

  const auto reduction_dims = GetPoolingReductionDims(window);
  if (reduction_dims.size() != 2) {
    return xla::FailedPrecondition("Popnn pooling only supports 2D pooling.");
  }

  const auto shuffle_in =
      GetShuffleInputDimensionsForPoplar(window, reduction_dims);

  input = input.dimShuffle(shuffle_in);
  output_grad = output_grad.dimShuffle(shuffle_in);
  output = output.dimShuffle(shuffle_in);

  auto pool_params =
      GetPoplibsPoolParams(popnn::PoolingType::MAX, window, input.shape(),
                           reduction_dims, input.elementType());

  poplar::Tensor out = popnn::pooling::poolInputGradient(
      graph, pool_params, input, output, output_grad, false, seq,
      GetDebugName(inst), res.default_pooling_options);
  // Shuffle back
  const auto shuffle_out = GetShuffleOutputDimensionsForPoplar(shuffle_in);
  out = out.dimShuffle(shuffle_out);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

StatusOr<poplar::program::Program> CreatePoplibsPoolingGrad(
    CompilerResources& res, const HloInstruction* inst, TensorMap& tensor_map,
    popnn::PoolingType pooling_type, const Window& window) {
  if (pooling_type == popnn::PoolingType::MAX) {
    return xla::FailedPrecondition("Calling invalid function for MaxPoolGrad.");
  }

  poplar::program::Sequence prog;
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::Tensor output_grad;
  TF_ASSIGN_OR_RETURN(output_grad,
                      FindInstructionInput(tensor_map, res, inst, 0, prog));
  // Get the input shape (same shape as the input grad i.e. the output shape).
  auto optional_input_shape =
      convert_array<std::vector<std::size_t>>(inst->shape().dimensions());
  if (!optional_input_shape) {
    return xla::FailedPrecondition(
        "CreatePoplibsPoolingGrad - cannot cast input_shape.");
  }
  std::vector<std::size_t> input_shape = *optional_input_shape;

  const auto reduction_dims = GetPoolingReductionDims(window);
  if (reduction_dims.size() != 2) {
    return xla::FailedPrecondition("Popnn pooling only supports 2D pooling.");
  }

  const auto shuffle_in =
      GetShuffleInputDimensionsForPoplar(window, reduction_dims);
  // Shuffle the input shape and the output grad tensor into a NC... shape.
  std::vector<std::size_t> input_shape_shuffled(input_shape.size());
  for (uint64 i = 0; i < input_shape.size(); i++) {
    input_shape_shuffled[i] = input_shape[shuffle_in[i]];
  }
  output_grad = output_grad.dimShuffle(shuffle_in);

  // TODO(T7321) The default expected behaviour is to do any partial
  // calculations in FP32, however popnn:pooling currently does not support
  // having an FP16 input and FP32 partials. We therefore upcast the input if
  // required and then downcast the result.
  auto output_grad_type = output_grad.elementType();
  auto reduction_type = GetReductionType(pooling_type, output_grad_type);
  const bool cast_required = output_grad_type != reduction_type;
  if (cast_required) {
    output_grad = popops::cast(graph, output_grad, reduction_type, prog,
                               GetDebugName(inst) + "/PreCast");
  }

  auto pool_params =
      GetPoplibsPoolParams(pooling_type, window, input_shape_shuffled,
                           reduction_dims, reduction_type);
  poplar::Tensor out = popnn::pooling::poolInputGradient(
      graph, pool_params, 1, output_grad, prog, GetDebugName(inst),
      res.default_pooling_options);

  if (cast_required) {
    out = popops::cast(graph, out, output_grad_type, prog,
                       GetDebugName(inst) + "/PreCast");
  }

  // Shuffle back
  const auto shuffle_out = GetShuffleOutputDimensionsForPoplar(shuffle_in);
  out = out.dimShuffle(shuffle_out);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return prog;
}

StatusOr<poplar::program::Program> CreateSimpleSelectAndScatter(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  poplar::Tensor out;
  poplar::program::Sequence prog;

  poplar::Graph& graph = GetGraph(res, inst);

  // Find the input tensors
  poplar::Tensor operand;
  TF_ASSIGN_OR_RETURN(operand,
                      FindInstructionInput(tensor_map, res, inst, 0, prog));

  poplar::Tensor source;
  TF_ASSIGN_OR_RETURN(source,
                      FindInstructionInput(tensor_map, res, inst, 1, prog));

  HloInstruction* select_root(inst->select()->root_instruction());
  HloInstruction* scatter_root(inst->scatter()->root_instruction());

  /*
   * Selection
   */

  std::string select_vertex_name = templateVertex(
      SelectionVertexBaseName(select_root), operand.elementType());

  const Window& window(inst->window());

  std::vector<int64> overlap(MaxWindowOverlap(window));
  int64 overlap_count(std::accumulate(overlap.begin(), overlap.end(), 1,
                                      [](int64 a, int64 b) { return a * b; }));

  // Create a partials tensor for reduction
  std::vector<std::size_t> poplar_shape = operand.shape();
  poplar_shape.push_back(1);

  auto name = StrCat(GetDebugName(inst), "_partial");
  poplar::Tensor extended_operand = operand.reshape(poplar_shape);
  poplar::Tensor partial = graph.clone(extended_operand, name);

  for (int64 i = 1; i < overlap_count; i++) {
    partial = poplar::concat(partial, graph.clone(extended_operand, name),
                             partial.rank() - 1);
  }

  xla::Shape partial_shape(output_shape);
  partial_shape.add_dimensions(overlap_count);
  LayoutUtil::ClearLayout(&partial_shape);
  partial_shape.mutable_layout()->set_format(DENSE);

  Literal identity_literal = GetIdentityConstantLiteral(scatter_root, inst);

  poplar::Tensor identity_val;
  TF_ASSIGN_OR_RETURN(
      identity_val,
      AddConstantTensor(graph, std::make_pair(inst, 0), partial_shape,
                        identity_literal, res, tensor_map));
  prog.add(poplar::program::Copy(identity_val, partial));

  // Find the number of windows in each dimension
  std::vector<unsigned> window_count(output_shape.rank());
  for (int64 d = 0; d < window.dimensions().size(); d++) {
    std::size_t input_dim(operand.dim(d));
    input_dim += window.dimensions(d).padding_low();
    input_dim += window.dimensions(d).padding_high();

    window_count[d] = window_util::StridedBound(
        input_dim, window.dimensions(d).size(), window.dimensions(d).stride());
  }

  auto select_cs = graph.addComputeSet(GetDebugName(inst) + "_select");
  prog.add(poplar::program::Execute(select_cs));

  const unsigned long num_windows = source.numElements();

  unsigned dim_count(operand.rank());

  // Vector for walking the window through the tensor
  std::vector<std::size_t> pos(dim_count, 0);

  // Slice boundaries
  std::vector<std::size_t> start_in(dim_count);
  std::vector<std::size_t> end_in(dim_count);

  std::vector<std::size_t> start_par(dim_count + 1);
  std::vector<std::size_t> end_par(dim_count + 1);

  for (unsigned i = 0; i < num_windows; ++i) {
    // Find the windows
    for (unsigned d = 0; d < dim_count; d++) {
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
    for (unsigned int t = 0; t < m.size(); t++) {
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
                             {{"a", w_in}, {"b", s}, {"out", w_par}});
    TF_RETURN_IF_ERROR(SetVertexField(graph, v["initval"], identity_literal));
    graph.setTileMapping(v, tile_with_max_elements);
    graph.setCycleEstimate(v, 1);

    // Advance the window
    for (int d = dim_count - 1; d >= 0; d--) {
      pos[d]++;
      if (pos[d] < window_count[d]) break;
      pos[d] = 0;
    }
  }

  /*
   * Reduction
   */
  popops::Operation op = PoplibsReductionOperation(scatter_root);

  std::vector<std::size_t> reduction_dims;
  reduction_dims.push_back(partial.rank() - 1);

  out = popops::reduce(graph, partial, reduction_dims, op, prog,
                       GetDebugName(inst) + "_reduce");

  /*
   * Initial value application
   */
  auto* init_inst = inst->operand(2);
  if (!(init_inst->IsConstant() && init_inst->literal() == identity_literal)) {
    poplar::Tensor init_val;
    TF_ASSIGN_OR_RETURN(init_val,
                        FindInstructionInput(tensor_map, res, inst, 2, prog));

    // Create a binary op with the scatter_root opcode
    TF_ASSIGN_OR_RETURN(init_val, BroadcastTensor(init_val, output_shape));

    popops::expr::BinaryOpType op;
    TF_ASSIGN_OR_RETURN(op, LookupBinaryFn(scatter_root));

    popops::mapInPlace(graph, op, out, init_val, prog,
                       GetDebugName(inst) + "_initval");
  }

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

StatusOr<poplar::program::Program> CreatePaddingReduceWindow(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map) {
  poplar::program::Sequence seq;

  poplar::Graph& graph = GetGraph(res, inst);

  const HloInstruction* root =
      inst->fused_instructions_computation()->root_instruction();
  const Window& window(root->window());
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, res, inst, 0, seq));

  std::vector<std::ptrdiff_t> paddingLower;
  std::vector<std::ptrdiff_t> paddingUpper;
  for (auto& d : window.dimensions()) {
    paddingLower.push_back(d.padding_low());
    paddingUpper.push_back(d.padding_high());
  }

  poplar::Tensor init_val;
  TF_ASSIGN_OR_RETURN(init_val,
                      FindInstructionInput(tensor_map, res, inst, 1, seq));

  out = popops::pad(graph, out, paddingLower, paddingUpper, init_val);

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
