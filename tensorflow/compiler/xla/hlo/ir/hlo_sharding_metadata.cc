/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding_metadata.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

// AssignmentKind and kUnassignedDevice are used during tuple domain sharding
// propagation in order to distinguish among three cases:
// kUnassigned: no assignment has occurred
// kAssigned: at least an assignment has occurred
// kConflict: no assignment has occurred because of conflicting propagations,
// which occurs when multiple users of an instruction have different
// shardings.
enum class AssignmentKind { kUnassigned, kAssigned, kConflict };

// kUnassignedDevice can only be assigned to tuple leaf shardings to indicate
// absence of sharding information for that particular sub-sharding during
// sharding propagation. It is used to be able to express tuple shardings with
// partial information. At the end of the propagation the sharding of
// tuple-shaped instructions using kUnassignedDevice's is cleared.
// TODO(b/112883246): Centralized enum of reserved devices.
constexpr int64_t kUnassignedDevice = -2;

struct PassThrough {
  PassThrough(HloInstruction* user, HloInstruction* operand)
      : user(user), operand(operand) {}

  HloInstruction* user = nullptr;
  HloInstruction* operand = nullptr;
};

void SetSingleSharding(HloInstruction* instruction,
                       const HloSharding& sharding) {
  VLOG(4) << "  " << instruction->name() << " to " << sharding;
  instruction->set_single_sharding(sharding);
}

bool ShardingMatches(const HloSharding& sharding1,
                     const HloSharding& sharding2) {
  auto single_sharding1 = sharding1.ExtractSingleSharding();
  if (single_sharding1) {
    auto single_sharding2 = sharding2.ExtractSingleSharding();
    if (single_sharding2) {
      return *single_sharding1 == single_sharding2;
    }
  }
  // Anything which is not unique across all elements, gets a full sharding
  // compare.
  return sharding1 == sharding2;
}

// When we create domains, they are never "empty", where with empty we mean
// that a kDomain instruction has as operand another kDomain instruction of the
// same kind.
// But when the HLO optimizations are run, empty domains can be created.
// For example:
//
//  Domain(device=None, device=0) ->
//    Tuple(device=0) ->
//      GTE(device=0) ->
//        Domain(device=0, device=None)
//
// In that case the tuple simplifier could create something like:
//
//  Domain(device=None, device=0) -> Domain(device=0, device=None)
//
// Which is a so called empty domain.
// In the case above, crossing an empty domain which was transiting through
// device 0, requires the normalization phase to fixup the empty domain by
// adding back a Tuple+GTE pair with the proper device.
// One particular case where this can create problems is the result of the
// entry computation, where the GTE assignments are used by TF to tell the
// XLA where the results should be sent.
std::vector<PassThrough> LocatePassThroughDomainLinks(
    const DomainMetadata::Domain& domain) {
  std::vector<PassThrough> pass_through;
  for (HloInstruction* instruction : domain.enter_domains) {
    CHECK(instruction->opcode() == HloOpcode::kDomain)
        << "Instruction is not a kDomain: " << instruction->ToString();
    for (HloInstruction* user : instruction->users()) {
      if (user->opcode() == HloOpcode::kDomain &&
          domain.exit_domains.contains(user)) {
        pass_through.emplace_back(user, instruction);
        VLOG(2) << "Found passthrough domain link:";
        VLOG(2) << "  " << user->ToString();
        VLOG(2) << "  " << instruction->ToString();
      }
    }
    if (instruction == instruction->parent()->root_instruction()) {
      pass_through.emplace_back(nullptr, instruction);
      VLOG(2) << "Found passthrough domain link:";
      VLOG(2) << "  <root>";
      VLOG(2) << "  " << instruction->ToString();
    }
  }
  return pass_through;
}

Status FixupPassThroughDomainLinks(const DomainMetadata::Domain& domain,
                                   const HloSharding& sharding) {
  for (auto& pass_through : LocatePassThroughDomainLinks(domain)) {
    HloInstruction* tuple = pass_through.operand->parent()->AddInstruction(
        HloInstruction::CreateTuple({pass_through.operand}));
    HloInstruction* gte = pass_through.operand->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(pass_through.operand->shape(),
                                              tuple, 0));
    gte->set_sharding(sharding.NormalizeTupleSharding(gte->shape()));
    if (pass_through.user != nullptr) {
      TF_RETURN_IF_ERROR(
          pass_through.operand->ReplaceUseWith(pass_through.user, gte));
    } else {
      pass_through.operand->parent()->set_root_instruction(gte);
    }
  }
  return OkStatus();
}

// For tuple shardings if every element have the same sharding then we want to
// treat them as single element shardings to insert less domain separation as a
// domain can prevent some optimizations and we want to minimize that from
// happening.
std::shared_ptr<const HloSharding> CloneShardingForDomain(
    std::shared_ptr<const HloSharding> sharding) {
  auto single_sharding = sharding->ExtractSingleSharding();
  if (!single_sharding) {
    return sharding;
  }
  return std::make_shared<const HloSharding>(*single_sharding);
}

Status ApplyDomainSingleSharding(const DomainMetadata::Domain& domain,
                                 const HloSharding& sharding) {
  VLOG(4) << "Applying " << sharding << " sharding";
  for (HloInstruction* instruction : domain.instructions) {
    // We only change instructions without sharding, since otherwise we might
    // mess up with eventual HLO passes which has knowledge of it.
    if (!instruction->has_sharding()) {
      SetSingleSharding(instruction, sharding);
    } else {
      VLOG(4) << "  " << instruction->name() << " already has sharding "
              << instruction->sharding();
    }
  }
  return OkStatus();
}

// Return the ShapeTree<HloSharding> of the user argument. The user argument
// is assumed to be a user of the instruction argument.
// If user is a tuple instruction, return the tuple subsharding corresponding to
// the operand matching the instruction argument, because that is the
// subsharding corresponding to instruction.
StatusOr<ShapeTree<HloSharding>> GetShardingTreeFromUser(
    const HloInstruction& instruction, const HloInstruction& user) {
  if (user.opcode() == HloOpcode::kTuple) {
    return user.sharding()
        .GetSubSharding(user.shape(), {user.operand_index(&instruction)})
        .AsShapeTree(instruction.shape());
  }
  return user.sharding().AsShapeTree(user.shape());
}

// Assign rhs to lhs. If rhs is unassigned (assigned to kUnassignedDevice)
// then no assignment is made. Therefore kUnassignedDevice is never propagated.
// kConflict is returned if lhs is already assigned and rhs is assigned to a
// different device.
StatusOr<AssignmentKind> AssignLeafSharding(HloSharding* lhs,
                                            const HloSharding& rhs) {
  TF_RET_CHECK(!lhs->IsTuple() && !rhs.IsTuple());
  if (rhs.UsesDevice(kUnassignedDevice)) {
    return AssignmentKind::kUnassigned;
  }
  if (lhs->UsesDevice(kUnassignedDevice)) {
    *lhs = rhs;
    return AssignmentKind::kAssigned;
  }
  return lhs->UniqueDevice() != rhs.UniqueDevice()
             ? AssignmentKind::kConflict
             : AssignmentKind::kUnassigned;
}

// Assigns the whole rhs tree to lhs_tree, starting at lhs_it.
// In case of conflicting assignment AssignmentKind::kConflict is returned. In
// this case lhs_tree is partially assigned, up to the conflicting leaf. It is
// up to the caller to discard the partial assignment in case of conflict.
StatusOr<AssignmentKind> AssignTreeSharding(
    ShapeTree<HloSharding>* lhs_tree, ShapeTree<HloSharding>::iterator lhs_it,
    const ShapeTree<HloSharding>& rhs_tree) {
  AssignmentKind assigned = AssignmentKind::kUnassigned;
  auto rhs_it = rhs_tree.begin();
  for (; lhs_it != lhs_tree->end() && rhs_it != rhs_tree.end();
       ++lhs_it, ++rhs_it) {
    // TODO(b/112885211): Add ShapeTree::IsLeaf(const ShapeTreeIterator &it)
    if (rhs_tree.IsLeaf(rhs_it->first)) {
      TF_RET_CHECK(lhs_tree->IsLeaf(lhs_it->first));
      TF_ASSIGN_OR_RETURN(AssignmentKind sub_assigned,
                          AssignLeafSharding(&lhs_it->second, rhs_it->second));
      if (sub_assigned == AssignmentKind::kConflict) {
        // In case of conflict we return conflict to the caller. At this point
        // partial assignments to lhs_tree may have been made already. It is up
        // to the caller to discard the partial assignment in case of conflict.
        return AssignmentKind::kConflict;
      } else if (sub_assigned == AssignmentKind::kAssigned) {
        assigned = sub_assigned;
      }
    }
  }
  TF_RET_CHECK(rhs_it == rhs_tree.end());
  return assigned;
}

StatusOr<bool> ApplyShardingFromUsers(HloInstruction* instruction,
                                      const DomainMetadata::Domain& domain,
                                      const HloSharding& domain_sharding) {
  if (instruction->users().empty()) {
    // No sharding from users, use domain_sharding, after checking
    // compatibility.
    TF_RET_CHECK(instruction->shape().IsTuple() &&
                 ShapeUtil::GetLeafCount(instruction->shape()) ==
                     domain_sharding.tuple_elements().size());
    instruction->set_sharding(domain_sharding);
    return true;
  }
  AssignmentKind assigned = AssignmentKind::kUnassigned;
  // The sharding_tree leaves are initialized to kUnassignedDevice. Only Tuple
  // subshardings can result in a final sharding assignment containing
  // kUnassignedDevice leaves, in case some tuple indexes are not used, or are
  // used by users that don't have a sharding.
  // Non-tuple shardings are either assigned to a real sharding, or are not
  // assigned at all. As such they will never get assigned to kUnassignedDevice.
  // In any case, kUnassignedDevice is never propagated, from the implementation
  // of AssignLeafSharding.
  ShapeTree<HloSharding> sharding_tree(
      instruction->shape(), HloSharding::AssignDevice(kUnassignedDevice));
  for (HloInstruction* user : instruction->users()) {
    if (user->opcode() == HloOpcode::kDomain &&
        domain.exit_domains.contains(user)) {
      // If a user is a domain and it is registered in the domain exits, then
      // the instruction sharding is taken directly from the domain, and no
      // further users need to be visited.
      instruction->set_sharding(domain_sharding);
      return true;
    }
    if (!user->has_sharding()) {
      continue;
    }
    AssignmentKind sub_assigned = AssignmentKind::kUnassigned;
    TF_ASSIGN_OR_RETURN(ShapeTree<HloSharding> user_sharding_tree,
                        GetShardingTreeFromUser(*instruction, *user));
    if (instruction->shape().IsTuple()) {
      // For tuple-shaped instructions collect individual tuple subshardings
      // from the uses, and then combine them into the tuple sharding.
      // If the user is a GTE its sharding concerns only the subtree of
      // sharding_tree at index user->tuple_index, otherwise the whole
      // sharding_tree is affected.
      ShapeTree<HloSharding>::iterator sharding_tree_begin =
          user->opcode() == HloOpcode::kGetTupleElement
              ? sharding_tree.find({user->tuple_index()})
              : sharding_tree.begin();
      TF_ASSIGN_OR_RETURN(
          sub_assigned, AssignTreeSharding(&sharding_tree, sharding_tree_begin,
                                           user_sharding_tree));
    } else {
      // Non-tuple shape: assign common users sharding.
      TF_RET_CHECK(user_sharding_tree.leaf_count() == 1)
          << "Expected non-tuple user sharding";
      TF_ASSIGN_OR_RETURN(
          sub_assigned,
          AssignTreeSharding(&sharding_tree, sharding_tree.begin(),
                             user_sharding_tree));
    }

    if (sub_assigned == AssignmentKind::kConflict) {
      // In case of conflict we don't assign any sharding.
      return false;
    } else if (sub_assigned == AssignmentKind::kAssigned) {
      assigned = sub_assigned;
    }
  }

  if (assigned == AssignmentKind::kAssigned) {
    if (instruction->shape().IsTuple()) {
      instruction->set_sharding(HloSharding::Tuple(sharding_tree));
    } else {
      TF_RET_CHECK(sharding_tree.leaf_count() == 1);
      instruction->set_sharding(sharding_tree.leaf_begin()->second);
    }
    return true;
  }
  return false;
}

// Tries to propagate the sharding information into the instructions that are
// part of the domain, in a reverse post order manner (users propagate to
// instruction).
StatusOr<int64_t> ApplyDomainShardingPass(const DomainMetadata::Domain& domain,
                                          const HloSharding& domain_sharding) {
  int64_t assigned = 0;
  // domain.instructions are ordered in a post-order manner. As we do
  // user->operand propagation we process instructions in reverse order. In so
  // doing we are guaranteed to process all users before their operands.
  for (auto it = domain.instructions.rbegin(); it != domain.instructions.rend();
       ++it) {
    HloInstruction* instruction = *it;
    if (instruction->has_sharding()) {
      continue;
    }
    // Take the sharding from the users.
    TF_ASSIGN_OR_RETURN(
        bool instruction_assigned,
        ApplyShardingFromUsers(instruction, domain, domain_sharding));
    if (instruction_assigned) {
      ++assigned;
      VLOG(4) << "  " << instruction->name() << " to sharding "
              << instruction->sharding();
    }
  }
  return assigned;
}

Status ApplyDomainSharding(const DomainMetadata::Domain& domain,
                           const HloSharding& sharding) {
  // None of the external normalizers handled the domain sharding, try to see
  // whether this is a single sharding first.
  auto single_sharding = sharding.ExtractSingleSharding();
  if (single_sharding) {
    // Shortcut the simple case. We have a unique sharding, so we call
    // the ApplyDomainSingleSharding() API which will apply array or tuple
    // shaped sharding to the domain instructions.
    return ApplyDomainSingleSharding(domain, *single_sharding);
  }
  VLOG(1) << "Assigning non-trivial sharding " << sharding;
  TF_RETURN_IF_ERROR(ApplyDomainShardingPass(domain, sharding).status());

  int64_t unassigned = 0;
  for (HloInstruction* instruction : domain.instructions) {
    if (!instruction->has_sharding()) {
      LOG(WARNING) << "Unassigned instruction: " << instruction->ToString();
      ++unassigned;
    } else {
      // Un-set sharding of tuples whose sub-shardings are assigned to
      // kUnassignedDevice. Indeed in case of doubt it is better to leave the
      // entire tuple unassigned, and let the device placer decide for it.
      // Do not clear the tuple sharding when the instruction is kParameter. The
      // sharding of the tuple might not be able to reconstructed if its users
      // are removed during DCE.
      if (instruction->sharding().UsesDevice(kUnassignedDevice) &&
          instruction->opcode() != HloOpcode::kParameter) {
        TF_RET_CHECK(instruction->shape().IsTuple())
            << "Only tuples can have kUnassignedDevice sub shardings";
        instruction->clear_sharding();
      }
    }
  }
  // Should we error out if unassigned > 0?
  return OkStatus();
}

StatusOr<std::shared_ptr<const HloSharding>> ExtractOriginalCommonSharding(
    absl::Span<HloInstruction* const> instructions) {
  // If we are here, all the instructions being passed had the same sharding
  // (or no sharding), by the means of the ShardingMatches() API.
  // As such, no kDomain was inserted, and here we are asked to extract the
  // original common sharding.
  // All the instructions passed to this API are part of the same computation.
  std::shared_ptr<const HloSharding> sharding;
  for (HloInstruction* instruction : instructions) {
    if (instruction->has_sharding()) {
      if (sharding == nullptr) {
        sharding = instruction->sharding_ptr();
      } else {
        TF_RET_CHECK(ShardingMatches(*sharding, instruction->sharding()))
            << "Sharding " << *sharding << " does not match the one in "
            << instruction->ToString();
      }
    }
  }
  if (sharding == nullptr) {
    return std::shared_ptr<const HloSharding>();
  }
  VLOG(4) << "Extracted sharding is " << *sharding;
  return CloneShardingForDomain(sharding);
}

}  // namespace

std::unique_ptr<DomainMetadata> ShardingMetadata::Clone() const {
  std::unique_ptr<HloSharding> sharding;
  if (sharding_ != nullptr) {
    sharding = std::make_unique<HloSharding>(*sharding_);
  }
  return std::make_unique<ShardingMetadata>(std::move(sharding));
}

bool ShardingMetadata::Matches(const DomainMetadata& other) const {
  const ShardingMetadata* other_ptr =
      dynamic_cast<const ShardingMetadata*>(&other);
  if (other_ptr == nullptr) {
    // If other is not a ShardingMetadata, then it is clearly a no match.
    return false;
  }
  if (sharding_ == nullptr) {
    return other_ptr->sharding_ == nullptr;
  }
  return other_ptr->sharding_ != nullptr
             ? ShardingMatches(*sharding_, *other_ptr->sharding_)
             : false;
}

std::string ShardingMetadata::ToString() const {
  return sharding_ != nullptr ? sharding_->ToString() : "{}";
}

/*static*/ StatusOr<const ShardingMetadata*>
ShardingMetadata::ToShardingMetadata(const DomainMetadata* metadata) {
  if (metadata->Kind() != ShardingMetadata::KindName()) {
    return Status(
        absl::StatusCode::kInvalidArgument,
        "ShardingMetadata normalizer called with incorrect domain metadata");
  }
  return static_cast<const ShardingMetadata*>(metadata);
}

Status ShardingMetadata::NormalizeShardingDomain(
    const DomainMetadata::Domain& domain, const DomainMetadata* metadata) {
  if (metadata != nullptr) {
    TF_ASSIGN_OR_RETURN(const auto& sharding_metadata,
                        ToShardingMetadata(metadata));
    const HloSharding* sharding = sharding_metadata->sharding();
    if (sharding != nullptr) {
      VLOG(4) << "Normalizing sharding to " << sharding->ToString() << ":";
      TF_RETURN_IF_ERROR(ApplyDomainSharding(domain, *sharding));
      TF_RETURN_IF_ERROR(FixupPassThroughDomainLinks(domain, *sharding));
    }
  } else {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<const HloSharding> sharding,
                        ExtractOriginalCommonSharding(domain.instructions));
    if (sharding != nullptr) {
      VLOG(4) << "Normalizing sharding-less domain to " << sharding->ToString();
      TF_RETURN_IF_ERROR(ApplyDomainSharding(domain, *sharding));
    } else {
      VLOG(1) << "Unable to find common sharding";
    }
  }
  return OkStatus();
}

// Creates a kDomain instruction to be placed between instruction and operand.
// The kDomain instruction will be created only if the sharding differ between
// the instruction and the operand.
HloInstruction* ShardingDomainCreator::operator()(HloInstruction* instruction,
                                                  HloInstruction* root,
                                                  HloInstruction* operand) {
  auto instruction_sharding = instruction->sharding_ptr();
  auto root_sharding = root->sharding_ptr();
  // No need for domain if they both have no sharding.
  if (instruction_sharding == nullptr && root_sharding == nullptr) {
    return nullptr;
  }
  // No need for domain if they match.
  if (instruction_sharding != nullptr && root_sharding != nullptr &&
      ShardingMatches(*instruction_sharding, *root_sharding)) {
    return nullptr;
  }

  if (instruction_sharding != nullptr) {
    instruction_sharding = CloneShardingForDomain(instruction_sharding);
  }
  if (root_sharding != nullptr) {
    root_sharding = CloneShardingForDomain(root_sharding);
  }

  auto it = domain_cse_map_.find({operand, instruction_sharding});
  if (it != domain_cse_map_.end()) {
    return it->second;
  }

  VLOG(3) << "Creating domain:";
  VLOG(3) << "  Instruction: " << instruction->name();
  VLOG(3) << "  Operand: " << operand->name();
  VLOG(3) << "    User side sharding: "
          << (instruction_sharding != nullptr ? instruction_sharding->ToString()
                                              : "None");
  VLOG(3) << "    Operand side sharding: "
          << (root_sharding != nullptr ? root_sharding->ToString() : "None");

  HloInstruction* domain =
      operand->parent()->AddInstruction(HloInstruction::CreateDomain(
          operand->shape(), operand,
          std::make_unique<ShardingMetadata>(root_sharding),
          std::make_unique<ShardingMetadata>(instruction_sharding)));
  domain_cse_map_.emplace(DomainCseMapKey{operand, instruction_sharding},
                          domain);
  return domain;
}

bool ShardingDomainCreator::DomainCseMapKey::operator==(
    const ShardingDomainCreator::DomainCseMapKey& other) const {
  if (instruction != other.instruction) {
    return false;
  }
  if (sharding == nullptr && other.sharding == nullptr) {
    return true;
  }
  if (sharding == nullptr || other.sharding == nullptr) {
    return false;
  }
  return *sharding == *other.sharding;
}

}  // namespace xla
