#include <algorithm>

#include "tensorflow/compiler/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

#include <poplin/MatMul.hpp>

namespace xla {
namespace poplarplugin {

static const std::string a_conn("a");
static const std::string b_conn("b");
static const std::string c_conn("c");
static const std::string out_conn("out");

#define POPLAR_OPCODE(O, N) case HloOpcode::O: return std::string(N)
#define UNUSED_OPCODE(O) case HloOpcode::O: break;

// NOTE 'Unused' Opcodes are implemented as a fixed function in the visitor
static port::StatusOr<std::string>
LookupPoplarVertexName(HloOpcode opcode) {
  switch (opcode) {
    POPLAR_OPCODE(kAbs, "Abs");
    POPLAR_OPCODE(kAdd, "Add");
    UNUSED_OPCODE(kBitcast);
    UNUSED_OPCODE(kBroadcast);
    UNUSED_OPCODE(kCall);
    POPLAR_OPCODE(kCeil, "Ceil");
    POPLAR_OPCODE(kClamp, "");
    UNUSED_OPCODE(kConcatenate);
    UNUSED_OPCODE(kConstant);
    UNUSED_OPCODE(kConvert);
    UNUSED_OPCODE(kConvolution);
    UNUSED_OPCODE(kCopy);
    POPLAR_OPCODE(kCrossReplicaSum, "");
    UNUSED_OPCODE(kCustomCall);
    POPLAR_OPCODE(kDivide, "Div");
    POPLAR_OPCODE(kDot, "Dot");
    UNUSED_OPCODE(kDynamicSlice);
    UNUSED_OPCODE(kDynamicUpdateSlice);
    POPLAR_OPCODE(kEq, "EqualTo");
    POPLAR_OPCODE(kExp, "Exp");
    POPLAR_OPCODE(kFloor, "Floor");
    POPLAR_OPCODE(kFusion, "");
    POPLAR_OPCODE(kGe, "GreaterEqual");
    UNUSED_OPCODE(kGetTupleElement);
    POPLAR_OPCODE(kGt, "GreaterThan");
    POPLAR_OPCODE(kIndex, "");
    UNUSED_OPCODE(kInfeed);
    POPLAR_OPCODE(kIsFinite, "");
    POPLAR_OPCODE(kLe, "LessEqual");
    POPLAR_OPCODE(kLog, "Log");
    POPLAR_OPCODE(kLogicalAnd, "LogicalAnd");
    POPLAR_OPCODE(kLogicalNot, "LogicalNot");
    POPLAR_OPCODE(kLogicalOr, "LogicalOr");
    POPLAR_OPCODE(kLt, "LessThan");
    UNUSED_OPCODE(kMap);
    POPLAR_OPCODE(kMaximum, "Maximum");
    POPLAR_OPCODE(kMinimum, "Minimum");
    POPLAR_OPCODE(kMultiply, "Mul");
    POPLAR_OPCODE(kNe, "NotEqual");
    POPLAR_OPCODE(kNegate, "Neg");
    UNUSED_OPCODE(kOutfeed);
    UNUSED_OPCODE(kPad);
    UNUSED_OPCODE(kParameter);
    POPLAR_OPCODE(kPower, "Pow");
    UNUSED_OPCODE(kRecv);
    UNUSED_OPCODE(kReduce);
    UNUSED_OPCODE(kReduceWindow);
    POPLAR_OPCODE(kRemainder, "Remainder");
    UNUSED_OPCODE(kReshape);
    POPLAR_OPCODE(kReverse, "");
    POPLAR_OPCODE(kRng, "");
    POPLAR_OPCODE(kSelect, "Select");
    POPLAR_OPCODE(kSelectAndScatter, "");
    UNUSED_OPCODE(kSend);
    POPLAR_OPCODE(kSign, "Sign");
    UNUSED_OPCODE(kSlice);
    POPLAR_OPCODE(kSort, "");
    POPLAR_OPCODE(kSubtract, "Sub");
    POPLAR_OPCODE(kTanh, "Tanh");
    POPLAR_OPCODE(kTrace, "");
    UNUSED_OPCODE(kTranspose);
    UNUSED_OPCODE(kTuple);
    POPLAR_OPCODE(kUpdate, "");
    POPLAR_OPCODE(kWhile, "");
  }
  return port::Status(port::error::UNKNOWN,
                      port::StrCat("[Poplar] Invalid opcode lookup ",
                                   HloOpcodeString(opcode)));
}

static bool
IsInPlaceUpdate(const HloInstruction *inst) {
  const HloOpcode opcode(inst->opcode());

  if (!(opcode == HloOpcode::kAdd ||
        opcode == HloOpcode::kDivide ||
        opcode == HloOpcode::kMaximum ||
        opcode == HloOpcode::kMinimum ||
        opcode == HloOpcode::kMultiply ||
        opcode == HloOpcode::kPower ||
        opcode == HloOpcode::kRemainder ||
        opcode == HloOpcode::kSubtract)) {
    return false;
  }

  // Operation must be part of an TF core update
  const OpMetadata& md(inst->metadata());
  const std::string& tf_op(md.op_type());
  if (!(tf_op == "AssignAddVariableOp" ||
        tf_op == "AssignSubVariableOp" ||
        tf_op == "ResourceApplyGradientDescent" ||
        tf_op == "ResourceApplyMomentum" ||
        tf_op == "ResourceApplyAdagrad" ||
        tf_op == "ResourceApplyRMSProp")) {
    return false;
  }

  // Operation must have a Parameter as an input
  const HloInstruction* op0(inst->operand(0));
  if (op0->opcode() != HloOpcode::kParameter) return false;

  // Operation must be the root or have the root as an output
  const HloInstruction* root(inst->parent()->root_instruction());
  if (inst == root) return true;

  const std::vector<HloInstruction*>& users(inst->users());
  if (users.size() != 1) return false;
  if (users[0] == root) return true;

  return false;
}

port::StatusOr<poplar::program::Program>
CreateUnaryElementwiseOp(poplar::Graph &graph,
                         const HloInstruction *inst,
                         const xla::Shape& output_shape,
                         TensorMap& tensor_map){

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0, 0));
  in = in.flatten();

  const std::string& poplar_data_type(graph.getTensorElementType(in));

  std::string vrtxTemplate;
  TF_ASSIGN_OR_RETURN(vrtxTemplate, LookupPoplarVertexName(inst->opcode()));
  std::string vertex_name = templateVertex(vrtxTemplate, poplar_data_type);

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{a_conn, in.slice(begin, end)},
                              {out_conn, out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateBinaryElementwiseOp(poplar::Graph &graph,
                          const HloInstruction *inst,
                          const xla::Shape& output_shape,
                          TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 1, 0));

  const std::string& poplar_data_type(graph.getTensorElementType(in0));

  bool in_place_update(IsInPlaceUpdate(inst) && (in0.shape() == in1.shape()));

  std::string vrtxTemplate;
  TF_ASSIGN_OR_RETURN(vrtxTemplate, LookupPoplarVertexName(inst->opcode()));
  if (in_place_update) {
    vrtxTemplate = vrtxTemplate + "InPlace";
  }
  std::string vertex_name = templateVertex(vrtxTemplate, poplar_data_type);

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

    poplar::Tensor r0 =
            in0.reshape(convert_array<std::vector<size_t>>(bcast.x_reshape()));
    poplar::Tensor r1 =
            in1.reshape(convert_array<std::vector<size_t>>(bcast.y_reshape()));

    in0 = TileTensor(bcast.x_bcast(), r0);
    in1 = TileTensor(bcast.y_bcast(), r1);
  }

  // Allocate the output tensor
  poplar::Tensor out;
  if (!in_place_update) {
    TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  } else {
    out = in0;
  }
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  // And now flatten
  in0 = in0.flatten();
  in1 = in1.flatten();
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->name());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    if (in_place_update) {
      auto v = graph.addVertex(cs, vertex_name,
                               {{a_conn, in0.slice(begin, end)},
                                {b_conn, in1.slice(begin, end)}});
      graph.setTileMapping(v, i / device_info.numWorkerContexts);
    } else {
      auto v = graph.addVertex(cs, vertex_name,
                               {{a_conn, in0.slice(begin, end)},
                                {b_conn, in1.slice(begin, end)},
                                {out_conn, out.slice(begin, end)}});
      graph.setTileMapping(v, i / device_info.numWorkerContexts);
    }
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateMatMulOp(poplar::Graph &graph,
               const HloInstruction *inst,
               const xla::Shape& output_shape,
               TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 1, 0));

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

  out = poplin::matMul(graph, in0, in1, seq);

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return seq;
}

// TODO - select needs to support scalars
port::StatusOr<poplar::program::Program>
CreateSelectOp(poplar::Graph &graph,
               const HloInstruction *inst,
               const xla::Shape& output_shape,
               TensorMap& tensor_map) {

  // Find the input tensors
  poplar::Tensor pred;
  TF_ASSIGN_OR_RETURN(pred, FindInstructionInput(tensor_map, inst, 0, 0));
  pred = pred.flatten();

  poplar::Tensor in0;
  TF_ASSIGN_OR_RETURN(in0, FindInstructionInput(tensor_map, inst, 1, 0));
  in0 = in0.flatten();

  poplar::Tensor in1;
  TF_ASSIGN_OR_RETURN(in1, FindInstructionInput(tensor_map, inst, 2, 0));
  in1 = in1.flatten();

  const std::string& poplar_data_type(graph.getTensorElementType(in0));

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  if (pred.dim(0) == 1) {
    std::string vertex_name = templateVertex("ScalarSelect", poplar_data_type);

    for (unsigned i = 0; i < num_workers; ++i) {
      const auto begin = i * N / num_workers;
      const auto end = (i + 1) * N / num_workers;
      auto v = graph.addVertex(cs, vertex_name,
                               {{"pred", pred[0]},
                                {a_conn, in0.slice(begin, end)},
                                {b_conn, in1.slice(begin, end)},
                                {out_conn, out.slice(begin, end)}});
      graph.setTileMapping(v, i / device_info.numWorkerContexts);
    }
  } else {
    std::string vertex_name = templateVertex("Select", poplar_data_type);

    for (unsigned i = 0; i < num_workers; ++i) {
      const auto begin = i * N / num_workers;
      const auto end = (i + 1) * N / num_workers;
      auto v = graph.addVertex(cs, vertex_name,
                               {{"pred", pred.slice(begin, end)},
                                {a_conn, in0.slice(begin, end)},
                                {b_conn, in1.slice(begin, end)},
                                {out_conn, out.slice(begin, end)}});
      graph.setTileMapping(v, i / device_info.numWorkerContexts);
    }

  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateCastOp(poplar::Graph &graph,
             const HloInstruction *inst,
             const xla::Shape& output_shape,
             TensorMap& tensor_map){

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0, 0));
  in = in.flatten();

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  const std::string& poplar_in_data_type(graph.getTensorElementType(in));
  const std::string& poplar_out_data_type(graph.getTensorElementType(out));

  std::string vertex_name = templateVertex("Cast",
                                           poplar_in_data_type,
                                           poplar_out_data_type);

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{a_conn, in.slice(begin, end)},
                              {out_conn, out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

port::StatusOr<poplar::program::Program>
CreateClampOp(poplar::Graph &graph,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map) {

  poplar::Tensor min;
  TF_ASSIGN_OR_RETURN(min, FindInstructionInput(tensor_map, inst, 0, 0));
  if (!PoplarShapeMatchesXLAShape(min, output_shape)) {
    TF_ASSIGN_OR_RETURN(min, BroadcastTensor(min, output_shape));
  }
  min = min.flatten();

  poplar::Tensor arg;
  TF_ASSIGN_OR_RETURN(arg, FindInstructionInput(tensor_map, inst, 1, 0));
  if (!PoplarShapeMatchesXLAShape(arg, output_shape)) {
    TF_ASSIGN_OR_RETURN(arg, BroadcastTensor(arg, output_shape));
  }
  arg = arg.flatten();

  poplar::Tensor max;
  TF_ASSIGN_OR_RETURN(max, FindInstructionInput(tensor_map, inst, 2, 0));
  if (!PoplarShapeMatchesXLAShape(max, output_shape)) {
    TF_ASSIGN_OR_RETURN(max, BroadcastTensor(max, output_shape));
  }
  max = max.flatten();

  // Allocate the output tensor
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, AddTensor(graph, inst->name(), output_shape));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  out = out.flatten();

  const std::string& poplar_data_type(graph.getTensorElementType(arg));

  std::string vertex_name = templateVertex("Clamp", poplar_data_type);

  auto cs = graph.addComputeSet(inst->ToString());
  const auto &device_info = graph.getDevice().getDeviceInfo();

  const unsigned long N = ShapeUtil::ElementsIn(output_shape);

  unsigned long num_workers = device_info.getNumTiles() * device_info.numWorkerContexts;
  num_workers = std::min(num_workers, N);

  for (unsigned i = 0; i < num_workers; ++i) {
    const auto begin = i * N / num_workers;
    const auto end = (i + 1) * N / num_workers;
    auto v = graph.addVertex(cs, vertex_name,
                             {{a_conn, min.slice(begin, end)},
                              {b_conn, arg.slice(begin, end)},
                              {c_conn, max.slice(begin, end)},
                              {out_conn, out.slice(begin, end)}});
    graph.setTileMapping(v, i / device_info.numWorkerContexts);
  }

  return poplar::program::Execute(cs);
}

}
}

