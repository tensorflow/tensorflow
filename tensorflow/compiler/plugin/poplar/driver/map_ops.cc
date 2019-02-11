#include <algorithm>

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/custom_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_arithmetic_expr.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_inline_call.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_map.h"
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
                                ComputationMap::iterator body,
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
    if (body->second.InputIsAllocated(0, i)) {
      for (unsigned int o = 0; o < param_count; o++) {
        if (body->second.InputIsAllocated(0, o)) {
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

static StatusOr<ComputationMap::iterator> GetOrCompileSubComputation(
    CompilerResources& res, const ArgVectors& inputs,
    const HloComputation* comp,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations =
        {}) {
  auto body(res.computation_map.find(comp));
  if (body != res.computation_map.end()) {
    return body;
  }

  VLOG(1) << "Compiling sub-computation " << comp->name();
  XLA_VLOG_LINES(1, comp->ToString());

  auto compiled = res.computation_map.emplace(
      std::piecewise_construct, std::forward_as_tuple(comp),
      std::forward_as_tuple(res, inputs, dependent_subcomputations));
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
  } else {
    ArgVectors args = GetCallInputs(res, inst, tensor_map, seq);
    ComputationMap::iterator subcomp_visitor;
    TF_ASSIGN_OR_RETURN(subcomp_visitor,
                        GetOrCompileSubComputation(res, args, comp));

    for (int64 o = 0; o < op_count; o++) {
      auto& inputs = subcomp_visitor->second.inputs()[o];
      if (inputs.size() != args[o].size()) {
        return xla::FailedPrecondition("Mismatched number of inputs");
      }
      for (int64 i = 0; i < inputs.size(); i++) {
        if (subcomp_visitor->second.InputIsUsed(o, i)) {
          seq.add(poplar::program::Copy(args[o][i], inputs[i]));
        }
      }
    }

    seq.add(subcomp_visitor->second.sequence);

    for (size_t i = 0; i < subcomp_visitor->second.outputs().size(); i++) {
      auto name = StrCat(GetDebugName(inst), "_out_", i);
      poplar::Tensor o =
          graph.clone(subcomp_visitor->second.outputs()[i], name);
      seq.add(poplar::program::Copy(subcomp_visitor->second.outputs()[i], o));
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
    LOG(FATAL) << "Unrecognised kCustomCall " << inst->ToString();
  }
}

StatusOr<poplar::program::Program> CreateFusionOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map) {
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
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence main_seq;

  ArgVectors inputs;
  inputs.push_back(FindInstructionInputs(tensor_map, res, inst, 0, main_seq));

  ComputationMap::iterator cond;
  TF_ASSIGN_OR_RETURN(
      cond, GetOrCompileSubComputation(res, inputs, inst->while_condition()));

  ComputationMap::iterator body;
  TF_ASSIGN_OR_RETURN(
      body, GetOrCompileSubComputation(res, inputs, inst->while_body(),
                                       {&cond->second}));

  unsigned int param_count = inputs[0].size();

  const ArgVector& body_inputs = body->second.inputs()[0];
  const ArgVector& body_outputs = body->second.outputs();
  const ArgVector& cond_inputs = cond->second.inputs()[0];
  const ArgVector& cond_outputs = cond->second.outputs();

  if (body_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body inputs");
  }
  if (body_outputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body outputs");
  }
  if (cond_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of condition inputs");
  }
  if (cond_outputs.size() != 1) {
    return xla::FailedPrecondition("Invalid number of condition outputs");
  }

  // The flow of tensors in while loops goes as follows:
  // 1. Copy the input tensors which are allocated (used by the condition or and
  // the body) into body_inputs.
  // 2. Before executing the condition, copy body_inputs which are required by
  // the condition to cond_inputs.
  // 3. After executing the body, copy the outputs into inputs.
  for (unsigned int i = 0; i < param_count; i++) {
    if (body->second.InputIsAllocated(0, i)) {
      main_seq.add(poplar::program::Copy(inputs[0][i], body_inputs[i]));
    }
  }

  // Condition
  poplar::program::Sequence cond_seq;
  for (unsigned int i = 0; i < param_count; i++) {
    if (cond->second.InputIsUsed(0, i)) {
      cond_seq.add(poplar::program::Copy(body_inputs[i], cond_inputs[i]));
    }
  }
  cond_seq.add(cond->second.sequence);
  poplar::Tensor pred =
      popops::allTrue(graph, cond_outputs[0], cond_seq, GetDebugName(inst));

  // Body
  poplar::program::Sequence body_seq(body->second.sequence);
  TF_ASSIGN_OR_RETURN(
      auto seq_argvector_pair,
      GetWhileAndRepeatAliasingCopies(graph, body, body_inputs, body_outputs,
                                      param_count, GetDebugName(inst)));
  body_seq.add(seq_argvector_pair.first);
  const ArgVector while_loop_state(seq_argvector_pair.second);

  main_seq.add(poplar::program::RepeatWhileTrue(cond_seq, pred, body_seq));

  for (unsigned int i = 0; i < param_count; i++) {
    auto name = StrCat(GetDebugName(inst), "_out_", i);
    poplar::Tensor o = graph.clone(while_loop_state[i], name);
    main_seq.add(poplar::program::Copy(while_loop_state[i], o));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, o));
  }

  return main_seq;
}

StatusOr<poplar::program::Program> CreateRepeatOp(CompilerResources& res,
                                                  const HloInstruction* inst,
                                                  const xla::Shape& output,
                                                  TensorMap& tensor_map) {
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence main_seq;

  CHECK_EQ(inst->operand(0)->opcode(), HloOpcode::kConstant);
  TF_ASSIGN_OR_RETURN(int64 repeat_count, LiteralScalarToNativeType<int64>(
                                              inst->operand(0)->literal()));

  ArgVectors inputs;
  inputs.push_back(FindInstructionInputs(tensor_map, res, inst, 1, main_seq));

  ComputationMap::iterator body;
  TF_ASSIGN_OR_RETURN(
      body, GetOrCompileSubComputation(res, inputs, GetRepeatBody(inst)));

  unsigned int param_count = inputs[0].size();

  const ArgVector& body_inputs = body->second.inputs()[0];
  const ArgVector& body_outputs = body->second.outputs();

  if (body_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body inputs");
  }
  if (body_outputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body outputs");
  }

  // The flow of tensors in while loops goes as follows:
  // 1. Copy the input tensors which are used by the body into body_inputs.
  // 2. After executing the body, copy the outputs into inputs.
  for (unsigned int i = 0; i < param_count; i++) {
    if (body->second.InputIsUsed(0, i)) {
      main_seq.add(poplar::program::Copy(inputs[0][i], body_inputs[i]));
    }
  }

  // Body
  poplar::program::Sequence body_seq(body->second.sequence);
  TF_ASSIGN_OR_RETURN(
      auto seq_argvector_pair,
      GetWhileAndRepeatAliasingCopies(graph, body, body_inputs, body_outputs,
                                      param_count, GetDebugName(inst)));
  body_seq.add(seq_argvector_pair.first);
  const ArgVector while_loop_state(seq_argvector_pair.second);

  main_seq.add(poplar::program::Repeat(repeat_count, body_seq));

  for (unsigned int i = 0; i < param_count; i++) {
    auto name = StrCat(GetDebugName(inst), "_out_", i);
    poplar::Tensor o = graph.clone(while_loop_state[i], name);
    main_seq.add(poplar::program::Copy(while_loop_state[i], o));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, o));
  }

  return main_seq;
}

StatusOr<poplar::program::Program> CreateIfOp(CompilerResources& res,
                                              const HloInstruction* inst,
                                              const xla::Shape& output,
                                              TensorMap& tensor_map) {
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

  ComputationMap::iterator true_body;
  TF_ASSIGN_OR_RETURN(
      true_body,
      GetOrCompileSubComputation(res, true_inputs, inst->true_computation()));

  ComputationMap::iterator false_body;
  TF_ASSIGN_OR_RETURN(
      false_body,
      GetOrCompileSubComputation(res, false_inputs, inst->false_computation()));

  poplar::Tensor scalar_pred =
      popops::allTrue(graph, pred, seq, GetDebugName(inst));

  if (true_body->second.inputs().size() != 1 ||
      false_body->second.inputs().size() != 1) {
    return xla::FailedPrecondition("Invalid input count");
  }

  poplar::program::Sequence true_seq;
  for (unsigned int i = 0; i < true_body->second.inputs()[0].size(); i++) {
    if (true_body->second.InputIsUsed(0, i)) {
      true_seq.add(poplar::program::Copy(true_inputs[0][i],
                                         true_body->second.inputs()[0][i]));
    }
  }
  true_seq.add(true_body->second.sequence);

  poplar::program::Sequence false_seq;
  for (unsigned int i = 0; i < false_body->second.inputs()[0].size(); i++) {
    if (false_body->second.InputIsUsed(0, i)) {
      false_seq.add(poplar::program::Copy(false_inputs[0][i],
                                          false_body->second.inputs()[0][i]));
    }
  }
  false_seq.add(false_body->second.sequence);

  unsigned int output_count = true_body->second.outputs().size();
  if (output_count != false_body->second.outputs().size()) {
    return xla::FailedPrecondition("Mismatched output size");
  }

  for (unsigned int i = 0; i < output_count; i++) {
    poplar::Tensor out = graph.clone(true_body->second.outputs()[i]);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));

    true_seq.add(poplar::program::Copy(true_body->second.outputs()[i], out));
    false_seq.add(poplar::program::Copy(false_body->second.outputs()[i], out));
  }

  seq.add(poplar::program::If(scalar_pred, true_seq, false_seq));
  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
