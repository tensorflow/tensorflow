#include <algorithm>

#include "tensorflow/compiler/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/poplar/driver/call_visitor.h"

#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

class ParallelMapTester : public DfsHloVisitorWithDefault {
public:
  ParallelMapTester() : _is_ok(true) {}

  Status DefaultAction(HloInstruction* inst) override {
    switch (inst->opcode()) {
      case HloOpcode::kAbs:
      case HloOpcode::kAdd:
      case HloOpcode::kCeil:
      case HloOpcode::kClamp:
      case HloOpcode::kConstant:
      case HloOpcode::kConvert:
      case HloOpcode::kDivide:
      case HloOpcode::kEq:
      case HloOpcode::kExp:
      case HloOpcode::kFloor:
      case HloOpcode::kGe:
      case HloOpcode::kGt:
      case HloOpcode::kLe:
      case HloOpcode::kLog:
      case HloOpcode::kLogicalAnd:
      case HloOpcode::kLogicalNot:
      case HloOpcode::kLogicalOr:
      case HloOpcode::kLt:
      case HloOpcode::kMultiply:
      case HloOpcode::kNe:
      case HloOpcode::kNegate:
      case HloOpcode::kParameter:
      case HloOpcode::kPower:
      case HloOpcode::kRemainder:
      case HloOpcode::kSign:
      case HloOpcode::kTanh:
      case HloOpcode::kSelect:
      case HloOpcode::kSubtract:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
        return Status::OK();
      default:
        _is_ok = false;
        return Status::OK();
    }
  }

  bool _is_ok;
};

port::StatusOr<bool>
IsComputationParallelMap(HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());

  ParallelMapTester tester;
  TF_RETURN_IF_ERROR(root->Accept(&tester, false));

  return tester._is_ok;
}

port::StatusOr<poplar::program::Program>
CreateParallelMap(poplar::Graph &graph,
                  const HloInstruction *inst,
                  const xla::Shape& output,
                  TensorMap& tensor_map) {

  int64 op_count(inst->operand_count());
  std::vector<poplar::Tensor> inputs;

  for (int64 i = 0; i < op_count; i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
    inputs.push_back(t);
  }

  PoplarMapVisitor visitor(&graph, inputs, output);
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  for (size_t i=0; i<visitor.output.size(); i++) {
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, visitor.output[i]));
  }

  return visitor.sequence;
}

port::StatusOr<poplar::program::Program>
CreateCallOp(poplar::Graph &graph,
             const HloInstruction *inst,
             const xla::Shape& output,
             TensorMap& tensor_map) {

  int64 op_count(inst->operand_count());
  std::vector<poplar::Tensor> inputs;

  for (int64 i = 0; i < op_count; i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
    inputs.push_back(t);
  }

  PoplarCallVisitor visitor(&graph, inputs);
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  for (size_t i=0; i<visitor.output.size(); i++) {
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, visitor.output[i]));
  }

  return visitor.sequence;
}

port::StatusOr<poplar::program::Program>
CreateWhileOp(poplar::Graph &graph,
              const HloInstruction *inst,
              const xla::Shape& output,
              TensorMap& tensor_map) {

  if (ShapeUtil::IsTuple(inst->operand(0)->shape())) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Poplar doesn't support tuple arguments to 'while' "
                        "operations");
  }

  poplar::Tensor init;
  TF_ASSIGN_OR_RETURN(init, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::Tensor body_input;
  TF_ASSIGN_OR_RETURN(body_input,
                      AddTensor(graph,
                                port::StrCat(inst->name(), "_input"),
                                output));

  poplar::program::Sequence main_seq;
  main_seq.add(poplar::program::Copy(init, body_input));

  // Body
  LOG(INFO) << "Generating body";
  std::vector<poplar::Tensor> body_inputs(1, body_input);
  PoplarCallVisitor body_visitor(&graph, body_inputs);
  TF_RETURN_IF_ERROR(inst->while_body()->Accept(&body_visitor));

  body_visitor.sequence.add(poplar::program::Copy(body_visitor.output[0],
                                                  body_input));
  //body_visitor.sequence.add(poplar::program::PrintTensor(body_visitor.output[0]));
  //body_visitor.sequence.add(poplar::program::PrintTensor(body_input));
  poplar::Tensor body_output = body_visitor.output[0];
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, body_output));

  // Condition
  LOG(INFO) << "Generating condition";
  std::vector<poplar::Tensor> condition_inputs(1, body_output);
  PoplarCallVisitor condition_visitor(&graph, condition_inputs);
  TF_RETURN_IF_ERROR(inst->while_condition()->Accept(&condition_visitor));

  poplar::program::RepeatWhileTrue repeat_while_true(condition_visitor.sequence,
                                                     body_visitor.sequence);

  main_seq.add(repeat_while_true);

  return main_seq;
}

}
}

