#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Add.hpp>
#include <popops/SubtractFrom.hpp>
#include <popops/HadamardProduct.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/NonLinearity.hpp>

namespace xla {
namespace poplarplugin {

static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string c_conn("c");
static const std::string out_conn("out");

#define POPLAR_OPCODE(O, N) case HloOpcode::O: return std::string(N)
#define UNUSED_OPCODE(O) case HloOpcode::O: break;


port::StatusOr<popops_unary_fn>
LookupUnaryFn(const HloInstruction* inst) {
  HloOpcode opcode = inst->opcode();
  switch (opcode) {
    case HloOpcode::kAbs: return popops::abs;
    case HloOpcode::kCeil: return popops::ceil;
    case HloOpcode::kCos: return popops::cos;
    case HloOpcode::kExp: return popops::exp;
    case HloOpcode::kFloor: return popops::floor;
    case HloOpcode::kLog: return popops::log;
    case HloOpcode::kNegate: return popops::neg;
    case HloOpcode::kRoundNearestAfz: return popops::round;
    case HloOpcode::kSign: return popops::signum;
    case HloOpcode::kSin: return popops::sin;
    case HloOpcode::kTanh: return popops::tanh;
    case HloOpcode::kIsFinite: return popops::isFinite;
    default:
      break;
  }

  if (opcode == HloOpcode::kNot) {
    if (inst->shape().element_type() == PRED) {
      return popops::logicalNot;
    } else {
      return popops::bitwiseNot;
    }
  }

  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
}

port::StatusOr<popops_binary_fn>
LookupBinaryFn(const HloInstruction* inst) {
  HloOpcode opcode = inst->opcode();
  switch (opcode) {
    case HloOpcode::kAdd: return popops::add;
    case HloOpcode::kAtan2: return popops::atan2;
    case HloOpcode::kDivide: return popops::div;
    case HloOpcode::kEq: return popops::eq;
    case HloOpcode::kGt: return popops::gt;
    case HloOpcode::kGe: return popops::gteq;
    case HloOpcode::kLt: return popops::lt;
    case HloOpcode::kLe: return popops::lteq;
    case HloOpcode::kMaximum: return popops::max;
    case HloOpcode::kMinimum: return popops::min;
    case HloOpcode::kMultiply: return popops::mul;
    case HloOpcode::kNe: return popops::neq;
    case HloOpcode::kPower: return popops::pow;
    case HloOpcode::kRemainder: return popops::rem;
    case HloOpcode::kShiftLeft: return popops::shiftLeft;
    case HloOpcode::kShiftRightArithmetic: return popops::shiftRightSignExtend;
    case HloOpcode::kShiftRightLogical: return popops::shiftRight;
    case HloOpcode::kSubtract: return popops::sub;
    default:
      break;
  }

  if (opcode == HloOpcode::kAnd) {
    if (inst->shape().element_type() == PRED) {
      return popops::logicalAnd;
    } else {
      return popops::bitwiseAnd;
    }
  }

  if (opcode == HloOpcode::kOr) {
    if (inst->shape().element_type() == PRED) {
      return popops::logicalOr;
    } else {
      return popops::bitwiseOr;
    }
  }

  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
}

port::StatusOr<popops_inplace_fn>
LookupBinaryInPlaceFn(const HloInstruction* inst) {
  HloOpcode opcode = inst->opcode();
  switch (opcode) {
    case HloOpcode::kAdd: return popops::addTo;
    case HloOpcode::kMultiply: return popops::hadamardProduct;
    case HloOpcode::kSubtract: return popops::subtractFrom;
    default:
      break;
  }
  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
}

static poplin::FullyConnectedPass GetMatMulPass(const HloInstruction* inst) {
  if (IsForwardMatMul(inst)) {
    return poplin::FullyConnectedPass::TRAINING_FWD;
  }
  if (IsGradientMatMul(inst)) {
    return poplin::FullyConnectedPass::TRAINING_BWD;
  }
  if (IsWeightUpdateMatMul(inst)) {
    return poplin::FullyConnectedPass::TRAINING_WU;
  }
  return poplin::FullyConnectedPass::INFERENCE_FWD;
}

port::StatusOr<poplar::program::Program>
CreateUnaryElementwiseOp(poplar::Graph &graph,
                         CompilerResources& res,
                         const HloInstruction *inst,
                         const xla::Shape& output_shape,
                         TensorMap& tensor_map){

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  popops_unary_fn fn;
  TF_ASSIGN_OR_RETURN(fn, LookupUnaryFn(inst));

  poplar::program::Sequence seq;
  poplar::Tensor out = fn(graph, in, seq, inst->name());

  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateBinaryElementwiseOp(poplar::Graph &graph,
                          CompilerResources& res,
                          const HloInstruction *inst,
                          const xla::Shape& output_shape,
                          TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 1));

  if (res.inplace_instructions.count(inst) == 1 &&
      (in0.shape() == in1.shape()) &&
      in0.isParallelWriteable()) {

    popops_inplace_fn fn;
    TF_ASSIGN_OR_RETURN(fn, LookupBinaryInPlaceFn(inst));

    poplar::program::Sequence seq;
    fn(graph, in0, in1, seq, inst->name());

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, in0));

    return seq;

  } else {

    if (in0.shape() != in1.shape()) {

      tensorflow::BCast::Vec shape1 =
              convert_array<tensorflow::BCast::Vec>(in0.shape());
      tensorflow::BCast::Vec shape2 =
              convert_array<tensorflow::BCast::Vec>(in1.shape());

      tensorflow::BCast bcast(shape1, shape2);
      if (!bcast.IsValid()) {
        return port::Status(port::error::FAILED_PRECONDITION,
                            port::StrCat("Incompatible broadcast on ",
                                         inst->name()));
      }

      in0 = in0.reshape(convert_array<std::vector<size_t>>(bcast.x_reshape()));
      in1 = in1.reshape(convert_array<std::vector<size_t>>(bcast.y_reshape()));

      in0 = TileTensor(bcast.x_bcast(), in0);
      in1 = TileTensor(bcast.y_bcast(), in1);
    }

    popops_binary_fn fn;
    TF_ASSIGN_OR_RETURN(fn, LookupBinaryFn(inst));

    poplar::program::Sequence seq;
    poplar::Tensor out = fn(graph, in0, in1, seq, inst->name());

    // Occasionally, due to an interplay of implicit broadcasting and
    // arithmetic re-arrangement, the output of an op is larger than the inputs
    // generate
    if (ShapeUtil::ElementsIn(output_shape) != out.numElements()) {
      TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));
    }

    out = out.reshape(PoplarShapeFromXlaShape(output_shape));

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

    return seq;
  }
}

port::StatusOr<poplar::program::Program>
CreateMatMulOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output_shape,
               TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 1));

  poplar::Tensor out;
  poplar::program::Sequence seq;

  if (in0.rank() > 2 || in1.rank() > 2) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        port::StrCat("Unsupported Dot operation on ",
                                     inst->name()));
  }

  if (in0.rank() == 1) {
    in0 = in0.reshape({1, in0.dim(0)});
  }

  if (in1.rank() == 1) {
    in1 = in1.reshape({in1.dim(0), 1});
  }

  poplin::MatMulOptions opts;
  opts.cache = &res.dot_cache;
  opts.fullyConnectedPass = GetMatMulPass(inst);

  out = poplin::matMul(graph, in0, in1, seq, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateSelectOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output_shape,
               TensorMap& tensor_map) {

  poplar::Tensor pred;
  TF_ASSIGN_OR_RETURN(pred, FindInstructionInput(tensor_map, inst, 0));

  ArgVector in0 = FindInstructionInputs(tensor_map, inst, 1);
  ArgVector in1 = FindInstructionInputs(tensor_map, inst, 2);

  if (in0.size() != in1.size()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        port::StrCat("Mismatching tuple sizes on ",
                                     inst->name()));
  }

  poplar::program::Sequence seq;

  for (unsigned int i=0; i<in0.size(); i++) {
    poplar::Tensor p = pred;
    poplar::Tensor i0 = in0[i];
    poplar::Tensor i1 = in1[i];

    if (p.numElements() == 1) {
      p = p.reshape({1});
      p = p.broadcast(i0.numElements(), 0);
      p = p.reshape(i0.shape());
    }

    poplar::Tensor out = popops::select(graph, i0, i1, p, seq, inst->name());

    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, out));
  }

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateCastOp(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output_shape,
             TensorMap& tensor_map){

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  poplar::Type poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  poplar::Tensor out = popops::cast(graph, in, poplar_type, seq, inst->name());

  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateClampOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map) {

  poplar::Tensor min;
  TF_ASSIGN_OR_RETURN(min, FindInstructionInput(tensor_map, inst, 0));
  if (!PoplarShapeMatchesXLAShape(min, output_shape)) {
    TF_ASSIGN_OR_RETURN(min, BroadcastTensor(min, output_shape));
  }

  poplar::Tensor arg;
  TF_ASSIGN_OR_RETURN(arg, FindInstructionInput(tensor_map, inst, 1));
  if (!PoplarShapeMatchesXLAShape(arg, output_shape)) {
    TF_ASSIGN_OR_RETURN(arg, BroadcastTensor(arg, output_shape));
  }

  poplar::Tensor max;
  TF_ASSIGN_OR_RETURN(max, FindInstructionInput(tensor_map, inst, 2));
  if (!PoplarShapeMatchesXLAShape(max, output_shape)) {
    TF_ASSIGN_OR_RETURN(max, BroadcastTensor(max, output_shape));
  }

  poplar::program::Sequence seq;
  poplar::Tensor out = popops::clamp(graph, arg, min, max, seq, inst->name());

  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateReluOp(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output_shape,
             TensorMap& tensor_map) {
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, 0));

  poplar::program::Sequence seq;
  poplar::Tensor out = graph.clone(t, inst->name());

  seq.add(poplar::program::Copy(t, out));
  popnn::relu(graph, out, seq, inst->name());

  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateReluGradOp(poplar::Graph &graph,
                 CompilerResources& res,
                 const HloInstruction *inst,
                 const xla::Shape& output_shape,
                 TensorMap& tensor_map) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor outgrad;
  TF_ASSIGN_OR_RETURN(outgrad, FindInstructionInput(tensor_map, inst, 1));

  poplar::program::Sequence seq;
  poplar::Tensor t = popnn::nonLinearityInputGradient(graph,
                                                      popnn::NON_LINEARITY_RELU,
                                                      out, outgrad, seq,
                                                      inst->name());

  TF_ASSIGN_OR_RETURN(t, BroadcastTensor(t, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateSigmoidOp(poplar::Graph &graph,
                CompilerResources& res,
                const HloInstruction *inst,
                const xla::Shape& output_shape,
                TensorMap& tensor_map) {
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, 0));

  poplar::program::Sequence seq;
  poplar::Tensor out = graph.clone(t, inst->name());

  seq.add(poplar::program::Copy(t, out));
  popnn::sigmoid(graph, out, seq, inst->name());

  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

}
}

