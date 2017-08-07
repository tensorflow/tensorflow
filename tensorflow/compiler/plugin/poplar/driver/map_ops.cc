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
    } else if (inst->opcode() == HloOpcode::kFusion) {
      switch (static_cast<int>(inst->fusion_kind())) {
        case FUSED_RELU:
        case FUSED_SIGMOID:
        case FUSED_TRUNCATED_NORMAL_WITH_SCALE:
        case FUSED_TRUNCATED_NORMAL:
        case FUSED_RANDOM_NORMAL_WITH_SCALE:
        case FUSED_RANDOM_UNIFORM_WITH_SCALE:
        case FUSED_RANDOM_NORMAL:
        case FUSED_RANDOM_UNIFORM:
        case FUSED_BERNOULLI:
        case FUSED_WIDE_CONSTANT:
          return Status::OK();
        default:
          _is_ok = false;
          return Status::OK();
      }
    } else {
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
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
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

    int64 op_count(inst->operand_count());
    std::vector <poplar::Tensor> inputs;

    for (int64 i = 0; i < op_count; i++) {
      poplar::Tensor t;
      TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
      inputs.push_back(t);
    }

    InlineCallVisitor inline_visitor(&graph, res, inputs);
    TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&inline_visitor));

    seq.add(inline_visitor.sequence);

    for (size_t i = 0; i < inline_visitor.outputs().size(); i++) {
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i,
                                         inline_visitor.outputs()[i]));
    }

  } else {
    // Pre-compiled callable sub-computation exists

    for (int64 i = 0; i < op_count; i++) {
      poplar::Tensor t;
      TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
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
CreateWhileOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output,
              TensorMap& tensor_map) {

  if (ShapeUtil::IsTuple(inst->operand(0)->shape())) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Poplar doesn't support tuple arguments to 'while' "
                        "operations");
  }
  if (ShapeUtil::IsTuple(inst->shape())) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Poplar doesn't support tuple return from 'while' "
                                "operations");
  }

  auto body_visitor(res.computation_map.find(inst->while_body()));
  if (body_visitor == res.computation_map.end()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Couldn't find body sub-computation for while op");
  }

  auto condition_visitor(res.computation_map.find(inst->while_condition()));
  if (condition_visitor == res.computation_map.end()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Couldn't find condition sub-computation for while op");
  }

  poplar::Tensor body_input = body_visitor->second.inputs()[0];
  poplar::Tensor body_output = body_visitor->second.outputs()[0];
  poplar::Tensor cond_input = condition_visitor->second.inputs()[0];
  poplar::Tensor cond_output = condition_visitor->second.outputs()[0];

  poplar::Tensor init;
  TF_ASSIGN_OR_RETURN(init, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::program::Sequence main_seq;
  main_seq.add(poplar::program::Copy(init, body_input));

  // Body
  poplar::program::Sequence body_seq;
  body_seq.add(body_visitor->second.sequence);
  body_seq.add(poplar::program::Copy(body_output, body_input));
  body_seq.add(poplar::program::Copy(body_output, cond_input));

  // Condition
  poplar::program::Sequence cond_seq;
  cond_seq.add(condition_visitor->second.sequence);

  popstd::allTrue(graph, cond_output, cond_seq);

  poplar::program::RepeatWhileTrue repeat_while_true(cond_seq, body_seq);

  main_seq.add(repeat_while_true);

  poplar::Tensor o = graph.clone(body_output);
  main_seq.add(poplar::program::Copy(body_output, o));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, o));

  return main_seq;
}

}
}

