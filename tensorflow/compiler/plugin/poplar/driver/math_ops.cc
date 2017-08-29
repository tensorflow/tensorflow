#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

#include <popstd/Cast.hpp>
#include <popstd/Operations.hpp>
#include <popstd/Add.hpp>
#include <popstd/SubtractFrom.hpp>
#include <popstd/HadamardProduct.hpp>
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


port::StatusOr<popstd_unary_fn>
LookupUnaryFn(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAbs: return popstd::abs;
    case HloOpcode::kCeil: return popstd::ceil;
    case HloOpcode::kCos: return popstd::cos;
    case HloOpcode::kExp: return popstd::exp;
    case HloOpcode::kFloor: return popstd::floor;
    case HloOpcode::kLog: return popstd::log;
    case HloOpcode::kLogicalNot: return popstd::logicalNot;
    case HloOpcode::kNegate: return popstd::neg;
    case HloOpcode::kSign: return popstd::signum;
    case HloOpcode::kSin: return popstd::sin;
    case HloOpcode::kTanh: return popstd::tanh;
    case HloOpcode::kIsFinite: return popstd::isFinite;
    
    default:
      break;
  }
  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
}

port::StatusOr<popstd_binary_fn>
LookupBinaryFn(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAdd: return popstd::add;
    case HloOpcode::kDivide: return popstd::div;
    case HloOpcode::kEq: return popstd::eq;
    case HloOpcode::kGt: return popstd::gt;
    case HloOpcode::kGe: return popstd::gteq;
    case HloOpcode::kLt: return popstd::lt;
    case HloOpcode::kLe: return popstd::lteq;
    case HloOpcode::kLogicalAnd: return popstd::logicalAnd;
    case HloOpcode::kLogicalOr: return popstd::logicalOr;
    case HloOpcode::kMaximum: return popstd::max;
    case HloOpcode::kMinimum: return popstd::min;
    case HloOpcode::kMultiply: return popstd::mul;
    case HloOpcode::kNe: return popstd::neq;
    case HloOpcode::kPower: return popstd::pow;
    case HloOpcode::kRemainder: return popstd::rem;
    case HloOpcode::kSubtract: return popstd::sub;
    default:
      break;
  }
  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
}

port::StatusOr<popstd_inplace_fn>
LookupBinaryInPlaceFn(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAdd: return popstd::addTo;
    case HloOpcode::kMultiply: return popstd::hadamardProduct;
    case HloOpcode::kSubtract: return popstd::subtractFrom;
    default:
      break;
  }
  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
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

  popstd_unary_fn fn;
  TF_ASSIGN_OR_RETURN(fn, LookupUnaryFn(inst->opcode()));

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

    popstd_inplace_fn fn;
    TF_ASSIGN_OR_RETURN(fn, LookupBinaryInPlaceFn(inst->opcode()));

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

    popstd_binary_fn fn;
    TF_ASSIGN_OR_RETURN(fn, LookupBinaryFn(inst->opcode()));

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

  out = poplin::matMul(graph, in0, in1, seq);

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

  std::vector<poplar::Tensor> in0 = FindInstructionInputs(tensor_map, inst, 1);
  std::vector<poplar::Tensor> in1 = FindInstructionInputs(tensor_map, inst, 2);

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

    poplar::Tensor out = popstd::select(graph, i0, i1, p, seq, inst->name());

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

  std::string poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(output_shape));

  poplar::program::Sequence seq;
  poplar::Tensor out = popstd::cast(graph, in, poplar_type, seq, inst->name());

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
  poplar::Tensor out = popstd::clamp(graph, arg, min, max, seq, inst->name());

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
  poplar::Tensor out = graph.clone(t);

  seq.add(poplar::program::Copy(t, out));
  popnn::relu(graph, out, seq, inst->name());

  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

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
  poplar::Tensor out = graph.clone(t);

  seq.add(poplar::program::Copy(t, out));
  popnn::sigmoid(graph, out, seq, inst->name());

  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out, output_shape));

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

}
}

