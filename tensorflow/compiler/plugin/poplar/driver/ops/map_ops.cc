#include <algorithm>

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/custom_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
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
  enum class AliasType {
    NO_ALIAS_NOT_USED,
    NO_ALIAS_USED,
    PARTIAL_ALIAS_OUTPUT_ONLY,
    PARTIAL_ALIAS,
    IDENTICAL_ALIAS,
  };

  poplar::program::Sequence body_seq;
  // A body output at index `i` can:
  // 1. contain no aliases to any of the inputs and the input `i` is not used in
  // the computation (NO_ALIAS_NOT_USED).
  // 2. contain no aliases to any of the inputs and the input `i` is used in the
  // computation (NO_ALIAS_USED).
  // 3. contain an alias to one of the inputs and the input `i` is not used in
  // the computation (PARTIAL_ALIAS_OUTPUT_ONLY).
  // 4. contain an alias to one of the inputs and the input `i` is used in the
  // computation (PARTIAL_ALIAS).
  // 5. be the exact same tensor as input `i` (IDENTICAL_ALIAS).

  // Find all the alias informations.
  std::vector<AliasType> alias_type(param_count, AliasType::NO_ALIAS_USED);
  for (unsigned int i = 0; i < param_count; i++) {
    const bool input_i_used = visitor.InputIsAllocated(0, i);
    if (input_i_used) {
      if (body_inputs[i] == body_outputs[i]) {
        alias_type[i] = AliasType::IDENTICAL_ALIAS;
      }
      // Check if we need to add a temporary copy.
      for (unsigned int o = 0; o < param_count; o++) {
        if (i != o && visitor.InputIsAllocated(0, o)) {
          if (body_outputs[o].intersectsWith(body_inputs[i])) {
            alias_type[i] = AliasType::PARTIAL_ALIAS;
          }
        }
      }
    } else {
      // If the input is not used, check that the output at that index does not
      // alias any of the inputs which might have changed during computation.
      alias_type[i] = AliasType::NO_ALIAS_NOT_USED;
      for (unsigned int j = 0; j < param_count; j++) {
        if (visitor.InputIsAllocated(0, j)) {
          if (body_outputs[i].intersectsWith(body_inputs[j])) {
            alias_type[i] = AliasType::PARTIAL_ALIAS_OUTPUT_ONLY;
          }
        }
      }
    }
  }

  // For partial aliasing types, we create temporary tensors from inputs in
  // order to remove any aliasing.
  ArgVector unaliased_body_outputs(body_outputs);
  ArgVector while_loop_state(body_inputs);
  for (unsigned int i = 0; i < param_count; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS_OUTPUT_ONLY:
      case AliasType::PARTIAL_ALIAS: {
        auto name = StrCat(debug_name, "_bodyout_temp_", i);
        unaliased_body_outputs[i] = graph.clone(body_outputs[i], name);
        body_seq.add(
            poplar::program::Copy(body_outputs[i], unaliased_body_outputs[i]));
        break;
      }
      default:
        break;
    }
  }

  for (unsigned int i = 0; i < param_count; i++) {
    switch (alias_type[i]) {
      case AliasType::PARTIAL_ALIAS:
      case AliasType::NO_ALIAS_USED: {
        // Get the input ready for the next iteration.
        body_seq.add(
            poplar::program::Copy(unaliased_body_outputs[i], body_inputs[i]));
        break;
      }
      case AliasType::PARTIAL_ALIAS_OUTPUT_ONLY:
      case AliasType::NO_ALIAS_NOT_USED: {
        // The input is never used so we don't need a copy - just change the
        // while loop state as by default it contains the input tensors.
        while_loop_state[i] = unaliased_body_outputs[i];
        break;
      }
      case AliasType::IDENTICAL_ALIAS:
      default:
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

StatusOr<std::shared_ptr<SubComputationVisitor>> GetOrCompileSubComputation(
    CompilerResources& res, const ArgVectors& inputs,
    const HloComputation* comp, bool inplace_inputs,
    const std::vector<const SubComputationVisitor*>&
        dependent_subcomputations) {
  // We can reuse sub computation if it's not inplace.
  if (!inplace_inputs) {
    auto itr = res.computation_map.find(comp);
    if (itr != res.computation_map.end()) {
      return itr->second;
    }
  }

  VLOG(2) << "Compiling sub-computation " << comp->name();
  XLA_VLOG_LINES(2, comp->ToString());

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
                      FindInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }
  MapVisitor visitor(res, inputs, output);
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  seq.add(visitor.GetSequence());

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

    seq.add(inline_visitor.GetSequence());

    for (size_t i = 0; i < inline_visitor.outputs().size(); i++) {
      poplar::Tensor out;
      TF_CHECK_OK(
          AddOutputTensor(tensor_map, inst, i, inline_visitor.outputs()[i]));
    }
  } else if (StartsWith(comp->name(), "__arithmetic")) {
    ArgVectors args = GetCallInputs(res, inst, tensor_map, seq, false);
    ArithmeticExprVisitor arithmetic_visitor(res, args);
    TF_RETURN_IF_ERROR(comp->Accept(&arithmetic_visitor));

    seq.add(arithmetic_visitor.GetSequence());

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
        return xla::FailedPrecondition("Mismatched number of inputs.");
      }
      for (int64 i = 0; i < inputs.size(); i++) {
        if (subcomp_visitor->InputIsUsed(o, i)) {
          seq.add(poplar::program::Copy(args[o][i], inputs[i]));
        }
      }
    }

    seq.add(subcomp_visitor->GetSequence());

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
  if (IsPoplibsHloCustomOp(inst)) {
    VLOG(1) << "Processing " << inst->custom_call_target()
            << " as Poplibs call";
    return CreatePoplibsOp(graph, res, inst, output, tensor_map);
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
                      FindInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  for (int64 op = 0; op < inst->operand_count(); op++) {
    CHECK_EQ(inputs[op].size(), CountShapes(inst->operand(op)->shape()));
  }
  InlineCallVisitor inline_visitor(res, inputs);
  TF_RETURN_IF_ERROR(comp->Accept(&inline_visitor));

  seq.add(inline_visitor.GetSequence());

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
  TF_ASSIGN_OR_RETURN(ArgVectors inputs, FindInplaceOutputTensors(
                                             tensor_map, res, inst, main_seq));
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
  cond_seq.add(cond->GetSequence());
  poplar::Tensor pred =
      popops::allTrue(graph, cond_outputs[0], cond_seq, GetDebugName(inst));

  // Body
  poplar::program::Sequence body_seq(body->GetSequence());
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
  TF_ASSIGN_OR_RETURN(ArgVectors inputs, FindInplaceOutputTensors(
                                             tensor_map, res, inst, main_seq));
  CHECK_EQ(inputs.size(), 1);

  TF_ASSIGN_OR_RETURN(auto body, GetOrCompileSubComputation(
                                     res, inputs, inst->to_apply(), true));

  unsigned int param_count = inputs[0].size();

  const ArgVector& inplace_inputs = inputs[0];
  const ArgVector& body_inputs = body->inputs()[0];
  const ArgVector& body_outputs = body->outputs();

  if (body_inputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body inputs.");
  }
  if (body_outputs.size() != param_count) {
    return xla::FailedPrecondition("Invalid number of body outputs.");
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
  poplar::program::Sequence body_seq(body->GetSequence());
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

StatusOr<poplar::program::Program> CreateConditionalOp(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output, TensorMap& tensor_map) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(res, inst);

  poplar::program::Sequence seq;

  bool is_switch;
  if (inst->operand(0)->shape().element_type() == PRED) {
    is_switch = false;
  } else if (inst->operand(0)->shape().element_type() == S32) {
    is_switch = true;
  } else {
    return xla::FailedPrecondition(
        "Conditional %s has unsupported condition input type %s.",
        PrimitiveType_Name(inst->operand(0)->shape().element_type()));
  }

  int n_branches = inst->operand_count() - 1;
  if (n_branches == 0) {
    return xla::FailedPrecondition("Conditional %s has no branches.",
                                   inst->name().c_str());
  }

  TF_ASSIGN_OR_RETURN(ArgVectors inputs,
                      FindInplaceOutputTensors(tensor_map, res, inst, seq));
  CHECK_EQ(inputs.size(), inst->operand_count());
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor pred = inputs[0][0];

  std::vector<std::shared_ptr<SubComputationVisitor>> bodies;
  const auto& comps = inst->called_computations();

  // Compile each branch into a sequence
  for (auto b = 0; b < n_branches; b++) {
    CHECK_EQ(inputs[b + 1].size(), CountShapes(inst->operand(b + 1)->shape()));
    TF_ASSIGN_OR_RETURN(
        auto body, GetOrCompileSubComputation(res, {inputs[b + 1]}, comps[b]));
    bodies.push_back(body);
  }

  unsigned int output_count = bodies[0]->outputs().size();

  // Add final output tensors for the conditional op
  std::vector<poplar::Tensor> outputs;
  for (unsigned int i = 0; i < output_count; i++) {
    poplar::Tensor out = graph.clone(bodies[0]->outputs()[i]);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
    outputs.push_back(out);
  }

  // Create the full program sequences for each branch
  std::vector<poplar::program::Sequence> seqs(bodies.size());
  for (auto b = 0; b < n_branches; b++) {
    if (bodies[b]->inputs().size() != 1) {
      return xla::FailedPrecondition("Invalid input count on branch %d.", b);
    }
    if (bodies[b]->outputs().size() != output_count) {
      return xla::FailedPrecondition("Mismatched output size on branch %d.", b);
    }

    // Add input copies
    for (unsigned int i = 0; i < bodies[b]->inputs()[0].size(); i++) {
      if (bodies[b]->InputIsUsed(0, i)) {
        seqs[b].add(
            poplar::program::Copy(inputs[b + 1][i], bodies[b]->inputs()[0][i]));
      }
    }

    // Add the actual body
    seqs[b].add(bodies[b]->GetSequence());

    // Add output copies
    for (unsigned int i = 0; i < output_count; i++) {
      seqs[b].add(poplar::program::Copy(bodies[b]->outputs()[i], outputs[i]));
    }
  }

  if (!is_switch) {
    poplar::Tensor scalar_pred =
        popops::allTrue(graph, pred, seq, GetDebugName(inst));

    seq.add(poplar::program::If(scalar_pred, seqs[0], seqs[1]));
  } else {
    std::vector<std::pair<int32, poplar::program::Program>> cases;
    for (auto c = 0; c < seqs.size() - 1; c++) {
      cases.push_back(std::make_pair(c, seqs[c]));
    }
    seq.add(poplar::program::Switch(pred, cases, seqs.back()));
  }

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
