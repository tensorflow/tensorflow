#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_inline_call.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_map.h"

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
#include <popstd/AllTrue.hpp>

namespace xla {
namespace poplarplugin {

class ParallelMapTester : public DfsHloVisitorWithDefault {
public:
  ParallelMapTester() : _is_ok(true) {}

  Status DefaultAction(HloInstruction* inst) override {
    if (inst->IsElementwise()) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kParameter) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kTuple) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kSelect) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kCall) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kMap) {
      return Status::OK();
    } else {
      LOG(INFO) << "Map didn't have a parallel computation " << inst->name();
      _is_ok = false;
      return Status::OK();
    }
  }

  bool _is_ok;
};

port::StatusOr<bool>
IsParallelMap(const HloInstruction* inst,
              const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());

  ParallelMapTester tester;
  TF_RETURN_IF_ERROR(root->Accept(&tester, false));

  return tester._is_ok;
}

port::StatusOr<poplar::program::Program>
CreateParallelMap(poplar::Graph &graph,
                  CompilerResources& res,
                  const HloInstruction *inst,
                  const xla::Shape& output,
                  TensorMap& tensor_map) {

  int64 op_count(inst->operand_count());
  std::vector<poplar::Tensor> inputs;

  for (int64 i = 0; i < op_count; i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i));
    inputs.push_back(t);
  }

  MapVisitor visitor(&graph, res, inputs, output);
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  auto outputs = visitor.outputs();
  for (size_t i=0; i<outputs.size(); i++) {
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i,
                                       outputs[i]));
  }

  return visitor.sequence;
}

port::StatusOr<poplar::program::Program>
CreateCallOp(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output,
             TensorMap& tensor_map) {

  int64 op_count(inst->operand_count());
  HloComputation* comp = inst->to_apply();
  poplar::program::Sequence seq;

  auto subcomp_visitor(res.computation_map.find(comp));
  if (subcomp_visitor == res.computation_map.end()) {
    // Inline the sub-computation

    std::vector <poplar::Tensor> inputs;

    for (int64 i = 0; i < op_count; i++) {
      poplar::Tensor t;
      TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i));
      inputs.push_back(t);
    }

    InlineCallVisitor inline_visitor(&graph, res, inputs);
    TF_RETURN_IF_ERROR(comp->Accept(&inline_visitor));

    seq.add(inline_visitor.sequence);

    for (size_t i = 0; i < inline_visitor.outputs().size(); i++) {
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i,
                                         inline_visitor.outputs()[i]));
    }

  } else {
    // Pre-compiled callable sub-computation exists

    for (int64 i = 0; i < op_count; i++) {
      poplar::Tensor t;
      TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i));
      seq.add(poplar::program::Copy(t, subcomp_visitor->second.inputs()[i]));
    }

    seq.add(subcomp_visitor->second.sequence);

    for (size_t i=0; i<subcomp_visitor->second.outputs().size(); i++) {
      poplar::Tensor o = graph.clone(subcomp_visitor->second.outputs()[i]);
      seq.add(poplar::program::Copy(subcomp_visitor->second.outputs()[i], o));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, o));
    }
  }

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateFusionOp(poplar::Graph &graph,
               CompilerResources& res,
               const HloInstruction *inst,
               const xla::Shape& output,
               TensorMap& tensor_map) {

  int64 op_count(inst->operand_count());
  HloComputation* comp = inst->fused_instructions_computation();
  poplar::program::Sequence seq;

  std::vector <poplar::Tensor> inputs;

  for (int64 i = 0; i < op_count; i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i));
    inputs.push_back(t);
  }

  InlineCallVisitor inline_visitor(&graph, res, inputs);
  TF_RETURN_IF_ERROR(comp->Accept(&inline_visitor));

  seq.add(inline_visitor.sequence);

  for (size_t i = 0; i < inline_visitor.outputs().size(); i++) {
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i,
                                       inline_visitor.outputs()[i]));
  }

  return seq;
}



port::StatusOr<poplar::program::Program>
CreateWhileOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output,
              TensorMap& tensor_map) {

  auto body(res.computation_map.find(inst->while_body()));
  if (body == res.computation_map.end()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Couldn't find body sub-computation for while op");
  }

  auto condition(res.computation_map.find(inst->while_condition()));
  if (condition == res.computation_map.end()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Couldn't find condition sub-computation for while op");
  }

  const std::vector<poplar::Tensor>& inits =
          FindInstructionInputs(tensor_map, inst, 0);

  unsigned int param_count = inits.size();

  const std::vector<poplar::Tensor>& body_inputs = body->second.inputs();
  const std::vector<poplar::Tensor>& body_outputs = body->second.outputs();
  const std::vector<poplar::Tensor>& cond_inputs = condition->second.inputs();
  const std::vector<poplar::Tensor>& cond_outputs = condition->second.outputs();

  if (body_inputs.size() != param_count) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Invalid number of body inputs");
  }
  if (body_outputs.size() != param_count) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Invalid number of body outputs");
  }
  if (cond_inputs.size() != param_count) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Invalid number of condition inputs");
  }
  if (cond_outputs.size() != 1) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Invalid number of condition outputs");
  }


  poplar::program::Sequence main_seq;
  for (unsigned int i=0; i<param_count; i++) {
    main_seq.add(poplar::program::Copy(inits[i], body_outputs[i]));
  }

  // Body
  poplar::program::Sequence body_seq;
  for (unsigned int i=0; i<param_count; i++) {
    if (body_outputs[i] != body_inputs[i]) {
      if (body_outputs[i].intersectsWith(body_inputs[i])) {
        poplar::Tensor temp = graph.clone(body_outputs[i]);
        body_seq.add(poplar::program::Copy(body_outputs[i], temp));
        body_seq.add(poplar::program::Copy(temp, body_inputs[i]));
      } else {
        body_seq.add(poplar::program::Copy(body_outputs[i], body_inputs[i]));
      }
    }
  }
  body_seq.add(body->second.sequence);

  // Condition
  poplar::program::Sequence cond_seq;
  for (unsigned int i=0; i<param_count; i++) {
    cond_seq.add(poplar::program::Copy(body_outputs[i], cond_inputs[i]));
  }
  cond_seq.add(condition->second.sequence);
  popstd::allTrue(graph, cond_outputs[0], cond_seq);

  // Main
  main_seq.add(poplar::program::RepeatWhileTrue(cond_seq, body_seq));

  for (unsigned int i=0; i<param_count; i++) {
    poplar::Tensor o = graph.clone(body_outputs[i]);
    main_seq.add(poplar::program::Copy(body_outputs[i], o));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, o));
  }

  return main_seq;
}

}
}

