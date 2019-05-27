/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/poplar/driver/passes/recompute_instructions.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/relu.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/classification_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

#include <set>
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

/*
  Runs a series of Matcher classes over each instruction to see if a particular
  pattern can be idendified. If it can be identified a Replacer class is
  returned and stored in a list of replacers. Once we've iterated over all
  instructions we then iterate over all the Replacers (all matches found) and
  execute them to replace each found match.
*/
namespace {

using TuplePair = std::pair<HloInstruction*, HloInstruction*>;

// Look through a tuple to find the "real" use. The first use matching
// |predicate_function| will be returned.
template <typename Predicate>
static HloInstruction* LookThroughTuple(HloInstruction* inst,
                                        Predicate predicate_function) {
  if (DynCast<HloGetTupleElementInstruction>(inst) == nullptr) {
    return nullptr;
  }

  for (HloInstruction* user : inst->users()) {
    if (predicate_function(user)) return user;
  }

  return nullptr;
}

// Helper wrapper so we can keep the API of the "Is" check functions taking a
// single instruction as input while still providing access to the annotation
// structure.
struct CheckWrapper {
  static const CompilerAnnotations* annotations;

  static void InitAnnotations(const CompilerAnnotations* a) { annotations = a; }

  static bool IsBackwardsConv(HloInstruction* inst) {
    return poplarplugin::IsBackpropInput(inst, *annotations) ||
           poplarplugin::IsBackpropFilter(inst, *annotations);
  }
};

const CompilerAnnotations* CheckWrapper::annotations = nullptr;

struct Replacer {
  virtual Status Replace() = 0;
};

struct Matcher {
  virtual std::unique_ptr<Replacer> PatternMatch(
      HloInstruction* inst) const = 0;
};

struct ConvNormNLReplacer : public Replacer {
  // The convolution at the start of the chain.
  HloInstruction* convolution;

  // The norm forward operation which takes input from the convolution.
  HloInstruction* norm_forward;

  // The norm backwards operation which takes input from all the forward ops
  // plus the NonLinearityGrad.
  HloInstruction* norm_backward_gradient;

  // The Get Tuple Element links between the norm forward operation and the norm
  // backwards operation.
  std::vector<HloInstruction*> norm_forward_to_back_GTEs;

  // The pair of forward non-linearity operation to the tuple which is its input
  // from the forward norm op.
  TuplePair forward_non_linearity_tuple;

  // The non linearity gradient operation in the backwards pass.
  HloInstruction* backward_non_linearity;

  // The convolution gradient operation in the backwards pass. This uses the
  // non-linearity as well so we will want to replace the uses.
  HloInstruction* backwards_convolution;

  ConvNormNLReplacer(HloInstruction* conv)
      : convolution(conv),
        norm_forward(nullptr),
        norm_backward_gradient(nullptr),
        norm_forward_to_back_GTEs({}),
        forward_non_linearity_tuple({nullptr, nullptr}),
        backward_non_linearity(nullptr),
        backwards_convolution(nullptr) {}

  Status Replace() final {
    HloComputation* parent_computation = convolution->parent();

    // Clone the convolution.
    HloInstruction* new_convolution =
        parent_computation->AddInstruction(convolution->Clone());

    // Replace the use of the convolution in the norm backward gradient.
    TF_RETURN_IF_ERROR(
        convolution->ReplaceUseWith(norm_backward_gradient, new_convolution));

    // Add a new norm forward operation.
    HloInstruction* new_fwd_op =
        parent_computation->AddInstruction(norm_forward->Clone());

    // For each of the tuples connecting the norm forward node to the backward
    // node, replace each tuple use with the new one.
    for (HloInstruction* tuple : norm_forward_to_back_GTEs) {
      // Create a new tuple.
      HloInstruction* new_tuple =
          parent_computation->AddInstruction(tuple->Clone());

      TF_RETURN_IF_ERROR(norm_forward->ReplaceUseWith(new_tuple, new_fwd_op));

      TF_RETURN_IF_ERROR(
          tuple->ReplaceUseWith(norm_backward_gradient, new_tuple));
    }

    // Replace the old forward conv use in the norm op with the new convolution.
    TF_RETURN_IF_ERROR(
        convolution->ReplaceUseWith(new_fwd_op, new_convolution));

    // Add a new forward non-linearity.
    HloInstruction* new_non_linearity = parent_computation->AddInstruction(
        forward_non_linearity_tuple.first->Clone());

    // Replace the use of the non-linearity in the non-linearity gradient
    // operation with the new non-linearity.
    TF_RETURN_IF_ERROR(forward_non_linearity_tuple.first->ReplaceUseWith(
        backward_non_linearity, new_non_linearity));

    // Replace the use of the non-linearity in the convolution grad op with the
    // new non-linearity.
    TF_RETURN_IF_ERROR(forward_non_linearity_tuple.first->ReplaceUseWith(
        backwards_convolution, new_non_linearity));

    // Make a new tuple connecting the forward norm pass to the new
    // non-linearity.
    HloInstruction* new_norm_to_non_linearity_tuple =
        parent_computation->AddInstruction(
            forward_non_linearity_tuple.second->Clone());

    // Make the connection between the forward pass and the new non-linearity.
    TF_RETURN_IF_ERROR(forward_non_linearity_tuple.second->ReplaceUseWith(
        new_non_linearity, new_norm_to_non_linearity_tuple));

    // Make the tuple be a tuple of the new norm operation.
    TF_RETURN_IF_ERROR(norm_forward->ReplaceUseWith(
        new_norm_to_non_linearity_tuple, new_fwd_op));

    // Unfortunately since we've added new instructions to the graph we need to
    // recompute the reachability map.
    std::unique_ptr<HloReachabilityMap> reachability_map =
        HloReachabilityMap::Build(parent_computation);

    // If we can't reach the instruction we can add it as a control dept as we
    // won't be adding a cycle.
    auto is_unreachable = [&](HloInstruction* inst) {
      return !reachability_map->IsReachable(inst, new_convolution) &&
             !reachability_map->IsReachable(new_convolution, inst);
    };

    // Add a control dependency from all of the operands to the new
    // instruction to hopefuly make sure it is executed right before its
    // user.
    for (HloInstruction* operand : backward_non_linearity->unique_operands()) {
      if (is_unreachable(operand)) {
        TF_RETURN_IF_ERROR(operand->AddControlDependencyTo(new_convolution));
        break;
      }
    }

    return Status::OK();
  }
};

struct ConvNormReplacer : public Replacer {
  // The convolution at the start of the chain.
  HloInstruction* convolution;

  // The norm forward operation which takes input from the convolution.
  HloInstruction* norm_forward;

  // The norm backwards operation which takes input from all the forward ops
  // plus the NonLinearityGrad.
  HloInstruction* norm_backward_gradient;

  // The Get Tuple Element links between the norm forward operation and the norm
  // backwards operation.
  std::vector<HloInstruction*> norm_forward_to_back_GTEs;

  ConvNormReplacer(HloInstruction* conv)
      : convolution(conv),
        norm_forward(nullptr),
        norm_backward_gradient(nullptr),
        norm_forward_to_back_GTEs({}) {}

  Status Replace() final {
    HloComputation* parent_computation = convolution->parent();

    // Clone and add to graph.
    HloInstruction* new_convolution =
        parent_computation->AddInstruction(convolution->Clone());

    // Replace the use.
    TF_RETURN_IF_ERROR(
        convolution->ReplaceUseWith(norm_backward_gradient, new_convolution));

    // Create the forward norm operation.
    HloInstruction* new_fwd_op =
        parent_computation->AddInstruction(norm_forward->Clone());

    // Replace the old forward conv use in the norm op with the new convolution.
    TF_RETURN_IF_ERROR(
        convolution->ReplaceUseWith(new_fwd_op, new_convolution));

    // Create new tuples for each of the Norm->Tuple->NormGrad links.
    for (HloInstruction* tuple : norm_forward_to_back_GTEs) {
      // Create a new tuple.
      HloInstruction* new_tuple =
          parent_computation->AddInstruction(tuple->Clone());

      TF_RETURN_IF_ERROR(norm_forward->ReplaceUseWith(new_tuple, new_fwd_op));
      TF_RETURN_IF_ERROR(
          tuple->ReplaceUseWith(norm_backward_gradient, new_tuple));
    }

    // Unfortunately since we've added new instructions to the graph we need to
    // recompute the reachability map.
    std::unique_ptr<HloReachabilityMap> reachability_map =
        HloReachabilityMap::Build(parent_computation);

    // If we can't reach the instruction we can add it as a control dept as we
    // won't be adding a cycle.
    auto is_unreachable = [&](HloInstruction* inst) {
      return !reachability_map->IsReachable(inst, new_convolution);
    };

    // Add a control dependency from the first backward operation of the
    // backward gradient to the new instruction to hopefuly make sure it is
    // executed right before its use.
    for (HloInstruction* operand : norm_backward_gradient->unique_operands()) {
      HloInstruction* tuple_user = LookThroughTuple(operand, is_unreachable);
      // Tuple will be a user of norm_backward_gradient which will also meet
      // the condition so we manually skip that.
      if (tuple_user != nullptr && tuple_user != norm_backward_gradient) {
        TF_RETURN_IF_ERROR(operand->AddControlDependencyTo(new_convolution));
        break;
      }
    }

    return Status::OK();
  }
};

// Assigns |inst| to |ret| if |predicate| is true.
template <typename Value, typename Predicate>
void AssignValueIfPredicate(HloInstruction* inst, Value ret,
                            Predicate predicate) {
  if (predicate(inst)) {
    ret.get() = inst;
  }
}

// Assigns |inst| to |ret| if |predicate| is true.
template <typename Value, typename Predicate, typename... Pack>
void AssignValueIfPredicate(HloInstruction* inst, Value ret,
                            Predicate predicate, Pack&&... pack) {
  // Do the first predicate pair.
  AssignValueIfPredicate(inst, ret, predicate);
  // Then the rest of the pack.
  AssignValueIfPredicate(inst, std::forward<Pack>(pack)...);
}

// Iterates through the container from |begin| to |end| and executes each
// instruction/predicate in the parameter |pack|. Each parameter in the pack is
// expected to be an std::ref(HloInstruction*) followed by the predicate that
// any instruction should meet in order to be assigned to the std::ref.
template <typename Container, typename... Pack>
void FindInContainer(Container& container, Pack&&... pack) {
  absl::c_for_each(container, [&](HloInstruction* inst) {
    AssignValueIfPredicate(inst, std::forward<Pack>(pack)...);
  });
}

// We are looking for the pattern InputOp->Norm->NonLinearity and will
// remove the links from the forward pass to the backwards pass for the
// convolution, norm, and one of the links for the non-linearity and will
// recompute them in the backwards pass using the inputs convolution.
struct ConvNormNLMatcher : public Matcher {
  std::unique_ptr<Replacer> PatternMatch(
      HloInstruction* convolution) const final {
    // We expect the input op to have two outputs, one pointing at the
    // backwards pass which we will replace and another pointing to the norm
    // function.
    if (convolution->opcode() != HloOpcode::kConvolution ||
        convolution->user_count() != 2) {
      return nullptr;
    }

    // Create and populate the structure which will do the actual replacement
    // after all matches have been made.
    std::unique_ptr<ConvNormNLReplacer> replacer =
        absl::make_unique<ConvNormNLReplacer>(convolution);

    // Look for the norm forward and backward grad ops in the convolution users.
    FindInContainer(convolution->users(), std::ref(replacer->norm_forward),
                    IsNormTraining, std::ref(replacer->norm_backward_gradient),
                    IsNormGradient);

    // If we couldn't find them leave cause we can't match the pattern.
    if (!replacer->norm_backward_gradient || !replacer->norm_forward) {
      return nullptr;
    }

    // Functor to check that the Norm->NormGrad is the same NormGrad as the
    // Conv->NormGrad.
    auto is_same_back_func = [&](const HloInstruction* i) -> bool {
      return i == replacer->norm_backward_gradient;
    };

    // Expecting Norm -> (NonLinearity, 1-multiple tuple inputs to
    // NormBackwards, others allowed but ignored).
    absl::c_for_each(
        replacer->norm_forward->users(), [&](HloInstruction* inst) {
          // Find the NonLinearity.
          if (HloInstruction* result = LookThroughTuple(inst, IsNonLinearity)) {
            replacer->forward_non_linearity_tuple = {result, inst};
          }

          // Find all the tuple connections to the backwards gradient function.
          if (nullptr != LookThroughTuple(inst, is_same_back_func)) {
            replacer->norm_forward_to_back_GTEs.push_back(inst);
          }
        });

    if (!replacer->forward_non_linearity_tuple.first ||
        replacer->norm_forward_to_back_GTEs.empty()) {
      return nullptr;
    }

    // Finally look through the forward non-linearity operation to find the
    // backwards non-linearity gradient operation and the backwards convolution
    // gradient operation.
    FindInContainer(replacer->forward_non_linearity_tuple.first->users(),
                    std::ref(replacer->backward_non_linearity),
                    IsNonLinearityGradient,
                    std::ref(replacer->backwards_convolution),
                    CheckWrapper::IsBackwardsConv);

    if (!replacer->backward_non_linearity || !replacer->backwards_convolution) {
      return nullptr;
    }

    VLOG(1) << "Found pattern of Conv->Norm->NonLinearity, cloning forward "
               "Conv/Norm/NonLinearity in backward operations to save memory";
    return std::move(replacer);
  }
};

struct ConvNormMatcher : public Matcher {
  std::unique_ptr<Replacer> PatternMatch(
      HloInstruction* convolution) const final {
    // Start with the convolution.

    // We expect the convolution to have two outputs, Conv->Norm and
    // Conv->NormGrad.
    if (convolution->user_count() != 2) {
      return nullptr;
    }

    // Create and populate the structure which will do the actual replacement
    // after all matches have been made.
    std::unique_ptr<ConvNormReplacer> replacer =
        absl::make_unique<ConvNormReplacer>(convolution);

    // Find the forward and backward Norm operations in the convolution users.
    FindInContainer(convolution->users(), std::ref(replacer->norm_forward),
                    IsNormTraining, std::ref(replacer->norm_backward_gradient),
                    IsNormGradient);

    // If we couldn't find them leave cause we can't match the pattern.
    if (!replacer->norm_backward_gradient || !replacer->norm_forward) {
      return nullptr;
    }

    // Functor to check that the Norm->NormGrad is the same NormGrad as the
    // Conv->NormGrad.
    auto is_same_back_func = [&](const HloInstruction* i) {
      return i == replacer->norm_backward_gradient;
    };

    absl::c_for_each(replacer->norm_forward->users(),
                     [&](HloInstruction* inst) {
                       // Find the backwards pass.
                       if (LookThroughTuple(inst, is_same_back_func)) {
                         replacer->norm_forward_to_back_GTEs.push_back(inst);
                       }
                     });

    if (replacer->norm_forward_to_back_GTEs.size() == 0) {
      return nullptr;
    }

    VLOG(1) << "Found pattern of Conv->Norm, cloning forward Conv/Norm in "
               "backward operations to save memory";
    return std::move(replacer);
  }
};

}  // namespace

RecomputeInstructions::RecomputeInstructions(bool allow_recompute,
                                             CompilerAnnotations& annotations)
    : allow_recompute_(allow_recompute), annotations_(annotations) {
  CheckWrapper::InitAnnotations(&annotations_);
}

StatusOr<bool> RecomputeInstructions::Run(HloModule* module) {
  if (!allow_recompute_) {
    return false;
  }
  bool changed = false;

  std::list<std::unique_ptr<Replacer>> replacers;
  std::list<HloComputation*> changed_comps;

  const std::array<std::unique_ptr<Matcher>, 2> matchers{
      // Try catch the cases of Conv->Norm->Non-Linearity that we can.
      absl::make_unique<ConvNormNLMatcher>(),

      // If not try and get cases of just Conv/Norm.
      absl::make_unique<ConvNormMatcher>()};

  for (HloComputation* comp : module->computations()) {
    if (IsPopOpsFusion(comp) || comp->instruction_count() < 4) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      for (const std::unique_ptr<Matcher>& matcher : matchers) {
        std::unique_ptr<Replacer> replacer = matcher->PatternMatch(inst);
        if (replacer) {
          replacers.push_back(std::move(replacer));
          changed_comps.push_back(comp);
          // Move on to the next instruction if we get a match. We don't want
          // multiple patterns hitting the same instruction.
          break;
        }
      }
    }
  }

  if (!replacers.empty()) {
    for (HloComputation* comp : changed_comps) {
      VLOG(2) << "-----ORIGINAL-----\n\n\n"
              << comp->ToString() << "\n\n\n-----ORIGINAL-----";
    }
  }

  for (std::unique_ptr<Replacer>& replacer : replacers) {
    TF_RETURN_IF_ERROR(replacer->Replace());
    changed = true;
  }

  if (changed) {
    for (HloComputation* comp : changed_comps) {
      VLOG(2) << "-----REPLACED-----\n\n\n"
              << comp->ToString() << "\n\n\n-----REPLACED-----";
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
