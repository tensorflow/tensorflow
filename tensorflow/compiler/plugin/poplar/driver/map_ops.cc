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

static port::StatusOr<ComputationMap::iterator>
GetOrCompileSubComputation(poplar::Graph &graph,
                           CompilerResources& res,
                           const ArgVectors& inputs,
                           const HloComputation* comp) {

  auto body(res.computation_map.find(comp));
  if (body != res.computation_map.end()) {
    return body;
  }

  VLOG(1) << "Compiling sub-computation " << comp->name();
  XLA_VLOG_LINES(1, comp->ToString());

  auto compiled = res.computation_map.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(comp),
          std::forward_as_tuple(&graph, res, inputs));
  TF_RETURN_IF_ERROR(comp->Accept(&(res.computation_map.at(comp))));

  return compiled.first;
}

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
      VLOG(1) << "Map didn't have a parallel computation " << inst->name();
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
  ArgVector inputs;

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

  ArgVectors args;
  for (int64 i = 0; i < op_count; i++) {
    ArgVector t = FindInstructionInputs(tensor_map, inst, i);
    args.push_back(t);
  }

  ComputationMap::iterator subcomp_visitor;
  TF_ASSIGN_OR_RETURN(subcomp_visitor,
                      GetOrCompileSubComputation(graph, res, args, comp));

  for (int64 o = 0; o < op_count; o++) {
    auto& inputs = subcomp_visitor->second.inputs()[o];
    if (inputs.size() != args[o].size()) {
      return port::Status(port::error::FAILED_PRECONDITION,
                          "Mismatched number of inputs");
    }
    for (int64 i = 0; i < inputs.size(); i++) {
      if (subcomp_visitor->second.input_valid(o, i)) {
        seq.add(poplar::program::Copy(args[o][i], inputs[i]));
      }
    }
  }

  seq.add(subcomp_visitor->second.sequence);

  for (size_t i=0; i<subcomp_visitor->second.outputs().size(); i++) {
    poplar::Tensor o = graph.clone(subcomp_visitor->second.outputs()[i]);
    seq.add(poplar::program::Copy(subcomp_visitor->second.outputs()[i], o));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, o));
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

  ArgVectors inputs;

  for (int64 i = 0; i < op_count; i++) {
    ArgVector t = FindInstructionInputs(tensor_map, inst, i);
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

  ArgVectors inputs;
  inputs.push_back(FindInstructionInputs(tensor_map, inst, 0));

  ComputationMap::iterator body;
  TF_ASSIGN_OR_RETURN(body,
                      GetOrCompileSubComputation(graph, res, inputs,
                                                 inst->while_body()));

  ComputationMap::iterator cond;
  TF_ASSIGN_OR_RETURN(cond,
                      GetOrCompileSubComputation(graph, res, inputs,
                                                 inst->while_condition()));

  unsigned int param_count = inputs[0].size();

  const ArgVector& body_inputs = body->second.inputs()[0];
  const ArgVector& body_outputs = body->second.outputs();
  const ArgVector& cond_inputs = cond->second.inputs()[0];
  const ArgVector& cond_outputs = cond->second.outputs();

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
    if (body_outputs[i].isParallelWriteable()) {
      main_seq.add(poplar::program::Copy(inputs[0][i], body_outputs[i]));
    }
  }

  // Body

  poplar::program::Sequence body_seq;

  // A body output can be:
  // - an independent new tensor (0)
  // - containing an alias for one of the inputs (1)
  // - a simple passthrough of its own input (2)
  // - not required because the input is unused (3)

  // Find outputs which are aliases of inputs
  std::vector<int> alias_type(param_count, 0);
  for (unsigned int o = 0; o < param_count; o++) {
    if (body->second.input_valid(0, o)) {
      for (unsigned int i=0; i<param_count; i++) {
        if (body->second.input_valid(0, i)) {
          if (body_outputs[o].intersectsWith(body_inputs[i])) {
            alias_type[o] = 1;
          }
        }
      }

      if (body_outputs[o] == body_inputs[o]) {
        alias_type[o] = 2;
      }
    } else {
      alias_type[o] = 3;
    }
  }

  // Create a temporary copy location for outputs which need preserving
  std::vector<poplar::Tensor> copies(param_count);
  for (unsigned int o=0; o<param_count; o++) {
    if (alias_type[o] == 1) {
      copies[o] = graph.clone(body_outputs[o]);
      body_seq.add(poplar::program::Copy(body_outputs[o], copies[o]));
    }
  }

  for (unsigned int o=0; o<param_count; o++) {
    switch (alias_type[o]) {
      case 0:
        body_seq.add(poplar::program::Copy(body_outputs[o], body_inputs[o]));
        break;
      case 1:
        body_seq.add(poplar::program::Copy(copies[o], body_inputs[o]));
        break;
      case 2:
      case 3:
        // nothing required
        break;
    }
  }
  body_seq.add(body->second.sequence);

  // Condition
  poplar::program::Sequence cond_seq;
  for (unsigned int i=0; i<param_count; i++) {
    if (cond->second.input_valid(0, i)) {
      cond_seq.add(poplar::program::Copy(body_outputs[i], cond_inputs[i]));
    }
  }
  cond_seq.add(cond->second.sequence);
  popstd::allTrue(graph, cond_outputs[0], cond_seq, inst->name());

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

