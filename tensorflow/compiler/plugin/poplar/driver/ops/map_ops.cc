#include <algorithm>

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/custom_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_arithmetic_expr.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_inline_call.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Graph.hpp>
#include <popops/AllTrue.hpp>

using ::absl::StrCat;
using tensorflow::str_util::StartsWith;

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<std::pair<poplar::program::Sequence, ArgVector>>
GetWhileAndRepeatAliasingCopies(poplar::Graph& graph,
                                SubComputationVisitor& visitor,
                                const ArgVector& body_inputs,
                                const ArgVector& body_outputs,
                                unsigned int param_count,
                                const std::string& debug_name) {
  poplar::program::Sequence body_seq;
  // A body input can be:
  // - an independent new tensor (0)
  // - containing an alias for one of the outputs (1)
  // - a simple passthrough to output (2)
  // - not required because it is unused (3)

  // Find outputs which are aliases of inputs
  std::vector<int> alias_type(param_count, 0);
  for (unsigned int i = 0; i < param_count; i++) {
    if (visitor.InputIsAllocated(0, i)) {
      for (unsigned int o = 0; o < param_count; o++) {
        if (visitor.InputIsAllocated(0, o)) {
          if (body_outputs[o].intersectsWith(body_inputs[i])) {
            alias_type[i] = 1;
          }
        }
      }
      if (body_inputs[i] == body_outputs[i]) {
        alias_type[i] = 2;
      }
    } else {
      alias_type[i] = 3;
    }
  }

  ArgVector unaliased_body_outputs(body_outputs);
  ArgVector while_loop_state(body_inputs);
  for (unsigned int i = 0; i < param_count; i++) {
    if (alias_type[i] == 1) {
      auto name = StrCat(debug_name, "_bodyout_temp_", i);
      unaliased_body_outputs[i] = graph.clone(body_outputs[i], name);
      body_seq.add(
          poplar::program::Copy(body_outputs[i], unaliased_body_outputs[i]));
    } else if (alias_type[i] == 3) {
      while_loop_state[i] = body_outputs[i];
    }
  }

  for (unsigned int i = 0; i < param_count; i++) {
    switch (alias_type[i]) {
      case 0:
      case 1:
        body_seq.add(
            poplar::program::Copy(unaliased_body_outputs[i], body_inputs[i]));
        break;
      case 2:
      case 3:
        // nothing required
        break;
    }
  }
  return std::make_pair(body_seq, while_loop_state);
}

ArgVectors GetCallInputs(CompilerResources& res, const HloInstruction* inst,
                         TensorMap& tensor_map, poplar::program::Sequence& seq,
                         const bool expand_constants = true) {
  ArgVectors args;
  for (int64 i = 0; i < inst->operand_count(); i++) {
    ArgVector t =
        FindInstructionInputs(tensor_map, res, inst, i, seq, expand_constants);
    args.push_back(t);
  }
  return args;
}

}  // namespace

static StatusOr<std::shared_ptr<SubComputationVisitor>>
GetOrCompileSubComputation(CompilerResources& res, const ArgVectors& inputs,
                           const HloComputation* comp,
                           bool inplace_inputs = false,
                           const std::vector<const SubComputationVisitor*>&
                               dependent_subcomputations = {}) {
  // We can reuse sub computation if it's not inplace.
  if (!inplace_inputs) {
    auto itr = res.computation_map.find(comp);
    if (itr != res.computation_map.end()) {
      return itr->second;
    }
  }

  VLOG(1) << "Compiling sub-computation " << comp->name();
  XLA_VLOG_LINES(1, comp->ToString());

  auto visitor = std::make_shared<SubComputationVisitor>(
      res, inputs, inplace_inputs, dependent_subcomputations);
  auto order = comp->parent()->schedule().sequence(comp).instructions();
  TF_RETURN_IF_ERROR(comp->AcceptOrdered(visitor.get(), order));

  // We can reuse sub computation if it's not inplace.
  if (!inplace_inputs) {
    res.computation_map[comp] = visitor;
  }

  return visitor;
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

StatusOr<bool> IsParallelMap(const HloInstruction* inst,
                             const HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());

  ParallelMapTester tester;
  TF_RETURN_IF_ERROR(root->Accept(&tester, false));

  return tester._is_ok;
}

StatusOr<poplar::program::Program> CreateParallelMap(CompilerResources& res,
                                                     const HloInstruction* inst,
                                                     const xla::Shape& output,
                                                     TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }
  MapVisitor visitor(res, inputs, output);
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  seq.add(visitor.sequence);

  auto outputs = visitor.outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, outputs[i]));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateCallOp(CompilerResources& res,
                                                const HloInstruction* inst,
                                                const xla::Shape& output,
                                                TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);

  int64 op_count(inst->operand_count());
  HloComputation* comp = inst->to_apply();
  poplar::program::Sequence seq;

  if (StartsWith(comp->name(), "__inline")) {
    ArgVectors args = GetCallInputs(res, inst, tensor_map, seq);
    InlineCallVisitor inline_visitor(res, args);
    TF_RETURN_IF_ERROR(comp->Accept(&inline_visitor));

    seq.add(inline_visitor.sequence);

    for (size_t i = 0; i < inline_visitor.outputs().size(); i++) {
      poplar::Tensor out;
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, i, inline_visitor.outputs()[i]));
    }
  } else if (StartsWith(comp->name(), "__arithmetic")) {
    ArgVectors args = GetCallInputs(res, inst, tensor_map, seq, false);
    ArithmeticExprVisitor arithmetic_visitor(res, args);
    TF_RETURN_IF_ERROR(comp->Accept(&arithmetic_visitor));

    seq.add(arithmetic_visitor.sequence);

    for (size_t i = 0; i < arithmetic_visitor.outputs().size(); i++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i,
                                  arithmetic_visitor.outputs()[i]));
    }
  } else if (IsRepeatLoop(inst)) {
    TF_ASSIGN_OR_RETURN(seq, CreateRepeatOp(res, inst, output, tensor_map));
  } else {
    ArgVectors args = GetCallInputs(res, inst, tensor_map, seq);
    TF_ASSIGN_OR_RETURN(auto subcomp_visitor,
                        GetOrCompileSubComputation(res, args, comp));

    for (int64 o = 0; o < op_count; o++) {
      auto& inputs = subcomp_visitor->inputs()[o];
      if (inputs.size() != args[o].size()) {
        return xla::FailedPrecondition("Mismatched number of inputs");
      }
      for (int64 i = 0; i < inputs.size(); i++) {
        if (subcomp_visitor->InputIsUsed(o, i)) {
          seq.add(poplar::program::Copy(args[o][i], inputs[i]));
        }
      }
    }

    seq.add(subcomp_visitor->sequence);

    for (size_t i = 0; i < subcomp_visitor->outputs().size(); i++) {
      auto name = StrCat(GetDebugName(inst), "_out_", i);
      poplar::Tensor o = graph.clone(subcomp_visitor->outputs()[i], name);
      seq.add(poplar::program::Copy(subcomp_visitor->outputs()[i], o));
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, o));
    }
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateCustomCallOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);
  if (IsPoplibsCustomOp(inst)) {
    VLOG(1) << "Processing " << inst->name() << " as Poplibs call";
    return CreatePoplibsOp(graph, res, inst, output, tensor_map);
  } else if (IsInterIpuCopy(inst)) {
    return CreateInterIpuCopy(res, inst, output, tensor_map);
  } else {
    return xla::FailedPrecondition("Unrecognised kCustomCall %s.",
                                   inst->ToString().c_str());
  }
}

StatusOr<poplar::program::Program> CreateFusionOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  HloComputation* comp = inst->fused_instructions_computation();
  poplar::program::Sequence seq;
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }
  InlineCallVisitor inline_visitor(res, inputs);
  TF_RETURN_IF_ERROR(comp->Accept(&inline_visitor));

  seq.add(inline_visitor.sequence);

  for (size_t i = 0; i < inline_visitor.outputs().size(); i++) {
    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, i, inline_visitor.outputs()[i]));
  }

  return seq;
}

StatusOr<poplar::program::Program> CreateWhileOp(CompilerResources& res,
                                                 const HloInstruction* inst,
                                                 const xla::Shape& output,
                                                 TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence main_seq;
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, main_seq));
  CHECK_EQ(inputs.size(), 1);

  // Conditional should not change the inputs - therefore it's not inplace.
  TF_ASSIGN_OR_RETURN(
      auto cond,
      GetOrCompileSubComputation(res, inputs, inst->while_condition(), false));

  // Body of the while loop is inplace.
  TF_ASSIGN_OR_RETURN(
      auto body, GetOrCompileSubComputation(res, inputs, inst->while_body(),
                                            true, {cond.get()}));

  unsigned int param_count = inputs[0].size();
  const ArgVector& inplace_inputs = inputs[0];
  const ArgVector& body_inputs = body->inputs()[0];
  const ArgVector& body_outputs = body->outputs();
  const ArgVector& cond_inputs = cond->inputs()[0];
  const ArgVector& cond_outputs = cond->outputs();

  if (body_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body inputs.");
  }
  if (body_outputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body outputs.");
  }
  if (cond_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of condition inputs.");
  }
  if (cond_outputs.size() != 1) {
    return xla::FailedPrecondition("Invalid number of condition outputs.");
  }

  // Even though while loop is inplace, some of the while loop inputs might
  // allocate their inputs as they have allocation targets. In these cases make
  // sure to copy the values of the tensors.
  for (unsigned int i = 0; i < param_count; i++) {
    if (body->InputHasAllocationTarget(0, i)) {
      main_seq.add(poplar::program::Copy(inplace_inputs[i], body_inputs[i]));
    }
  }

  // Before executing the condition, copy inputs which are required by
  // the condition to cond_inputs.
  poplar::program::Sequence cond_seq;
  for (unsigned int i = 0; i < param_count; i++) {
    if (cond->InputIsUsed(0, i)) {
      cond_seq.add(poplar::program::Copy(body_inputs[i], cond_inputs[i]));
    }
  }
  cond_seq.add(cond->sequence);
  poplar::Tensor pred =
      popops::allTrue(graph, cond_outputs[0], cond_seq, GetDebugName(inst));

  // Body
  poplar::program::Sequence body_seq(body->sequence);
  TF_ASSIGN_OR_RETURN(auto seq_argvector_pair,
                      GetWhileAndRepeatAliasingCopies(
                          graph, *body.get(), body_inputs, body_outputs,
                          param_count, GetDebugName(inst)));
  body_seq.add(seq_argvector_pair.first);
  const ArgVector while_loop_state(seq_argvector_pair.second);

  main_seq.add(poplar::program::RepeatWhileTrue(cond_seq, pred, body_seq));

  for (unsigned int i = 0; i < param_count; i++) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, while_loop_state[i]));
  }

  return main_seq;
}

StatusOr<poplar::program::Program> CreateRepeatOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence main_seq;

  TF_ASSIGN_OR_RETURN(PoplarBackendConfig cfg,
                      inst->backend_config<PoplarBackendConfig>());
  int64 repeat_count = cfg.repeat_config().repeat_count();
  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, main_seq));
  CHECK_EQ(inputs.size(), 1);

  TF_ASSIGN_OR_RETURN(auto body, GetOrCompileSubComputation(
                                     res, inputs, inst->to_apply(), true));

  unsigned int param_count = inputs[0].size();

  const ArgVector& inplace_inputs = inputs[0];
  const ArgVector& body_inputs = body->inputs()[0];
  const ArgVector& body_outputs = body->outputs();

  if (body_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body inputs");
  }
  if (body_outputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body outputs");
  }

  // Even though repeat loop is inplace, some of the repeat loop inputs might
  // allocate their inputs as they have allocation targets. In these cases make
  // sure to copy the values of the tensors.
  for (unsigned int i = 0; i < param_count; i++) {
    if (body->InputHasAllocationTarget(0, i)) {
      main_seq.add(poplar::program::Copy(inplace_inputs[i], body_inputs[i]));
    }
  }

  // Body
  poplar::program::Sequence body_seq(body->sequence);
  TF_ASSIGN_OR_RETURN(auto seq_argvector_pair,
                      GetWhileAndRepeatAliasingCopies(
                          graph, *body.get(), body_inputs, body_outputs,
                          param_count, GetDebugName(inst)));
  body_seq.add(seq_argvector_pair.first);
  const ArgVector while_loop_state(seq_argvector_pair.second);

  main_seq.add(poplar::program::Repeat(repeat_count, body_seq));

  for (unsigned int i = 0; i < param_count; i++) {
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, while_loop_state[i]));
  }

  return main_seq;
}

StatusOr<poplar::program::Program> CreateIfOp(CompilerResources& res,
                                              const HloInstruction* inst,
                                              const xla::Shape& output,
                                              TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      GetInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor pred = inputs[0][0];

  CHECK_EQ(inputs[1].size(), CountShapes(inst->operand(1)->shape()));
  ArgVectors true_inputs({inputs[1]});

  CHECK_EQ(inputs[2].size(), CountShapes(inst->operand(2)->shape()));
  ArgVectors false_inputs({inputs[2]});

  TF_ASSIGN_OR_RETURN(
      auto true_body,
      GetOrCompileSubComputation(res, true_inputs, inst->true_computation()));

  TF_ASSIGN_OR_RETURN(
      auto false_body,
      GetOrCompileSubComputation(res, false_inputs, inst->false_computation()));

  poplar::Tensor scalar_pred =
      popops::allTrue(graph, pred, seq, GetDebugName(inst));

  if (true_body->inputs().size() != 1 || false_body->inputs().size() != 1) {
    return xla::FailedPrecondition("Invalid input count");
  }

  poplar::program::Sequence true_seq;
  for (unsigned int i = 0; i < true_body->inputs()[0].size(); i++) {
    if (true_body->InputIsUsed(0, i)) {
      true_seq.add(
          poplar::program::Copy(true_inputs[0][i], true_body->inputs()[0][i]));
    }
  }
  true_seq.add(true_body->sequence);

  poplar::program::Sequence false_seq;
  for (unsigned int i = 0; i < false_body->inputs()[0].size(); i++) {
    if (false_body->InputIsUsed(0, i)) {
      false_seq.add(poplar::program::Copy(false_inputs[0][i],
                                          false_body->inputs()[0][i]));
    }
  }
  false_seq.add(false_body->sequence);

  unsigned int output_count = true_body->outputs().size();
  if (output_count != false_body->outputs().size()) {
    return xla::FailedPrecondition("Mismatched output size");
  }

  for (unsigned int i = 0; i < output_count; i++) {
    poplar::Tensor out = graph.clone(true_body->outputs()[i]);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));

    true_seq.add(poplar::program::Copy(true_body->outputs()[i], out));
    false_seq.add(poplar::program::Copy(false_body->outputs()[i], out));
  }

  seq.add(poplar::program::If(scalar_pred, true_seq, false_seq));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
