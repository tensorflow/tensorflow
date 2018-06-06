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

#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

struct PassThrough {
  PassThrough(HloInstruction* user, HloInstruction* operand)
      : user(user), operand(operand) {}

  HloInstruction* user = nullptr;
  HloInstruction* operand = nullptr;
};

void SetDeviceSharding(HloInstruction* instruction, int64 device) {
  VLOG(4) << "  " << instruction->name() << " to device " << device;
  instruction->set_device_sharding(device);
}

tensorflow::gtl::optional<int64> ShardingUniqueDevice(
    const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
    auto device = sharding.UniqueDevice();
    if (device.ok()) {
      return device.ValueOrDie();
    }
  }
  return tensorflow::gtl::optional<int64>();
}

bool ShardingMatches(const HloSharding& sharding1,
                     const HloSharding& sharding2) {
  auto device1 = ShardingUniqueDevice(sharding1);
  if (device1) {
    auto device2 = ShardingUniqueDevice(sharding2);
    if (device2) {
      return *device1 == *device2;
    }
  }
  // Anything which is not tile maximal with unique device, gets a full sharding
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
          domain.exit_domains.count(user) != 0) {
        pass_through.emplace_back(user, instruction);
        VLOG(2) << "Found passthrough domain link:";
        VLOG(2) << "  " << user->ToString();
        VLOG(2) << "  " << instruction->ToString();
      }
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
    gte->set_sharding(sharding);
    TF_RETURN_IF_ERROR(
        pass_through.operand->ReplaceUseWith(pass_through.user, gte));
  }
  return Status::OK();
}

std::unique_ptr<HloSharding> CloneShardingForDomain(
    const HloSharding& sharding) {
  auto device = ShardingUniqueDevice(sharding);
  if (!device) {
    return MakeUnique<HloSharding>(sharding);
  }
  return MakeUnique<HloSharding>(HloSharding::AssignDevice(*device));
}

Status ApplyDomainDeviceSharding(const DomainMetadata::Domain& domain,
                                 int64 device) {
  VLOG(4) << "Applying device " << device << " sharding";
  for (HloInstruction* instruction : domain.instructions) {
    // We only change instructions without sharding, since otherwise we might
    // mess up with eventual HLO passes which has knowledge of it.
    if (!instruction->has_sharding()) {
      SetDeviceSharding(instruction, device);
    } else {
      VLOG(4) << "  " << instruction->name() << " already has sharding "
              << instruction->sharding();
    }
  }
  return Status::OK();
}

// Retrieves the sharding of a tuple shaped instruction in form of a ShapeTree.
// If the instruction has no sharding, a ShapeTree with HloSharding::Replicate()
// sharding will be returned.
ShapeTree<HloSharding> GetTupleSharding(HloInstruction* tuple) {
  if (tuple->has_sharding()) {
    return tuple->sharding().GetAsShapeTree(tuple->shape());
  }
  return ShapeTree<HloSharding>(tuple->shape(), HloSharding::Replicate());
}

// Retrieves the sharding of operand, asked from a user instruction which is
// within domain. If operand is a kDomain, it means that sharding argument is
// the operand sharding, otherwise the operand's own sharding will be returned.
const HloSharding* GetOperandSharding(const HloInstruction* operand,
                                      const DomainMetadata::Domain& domain,
                                      const HloSharding& sharding) {
  DCHECK_EQ(domain.reach_set.count(const_cast<HloInstruction*>(operand)), 1);
  // Here the user of operand is within the domain instruction set, and since it
  // is user of operand, we need to look into the enter_domains set. If this is
  // not a kDomain within the user domains set, then return the operand
  // sharding, if any.
  if (operand->opcode() != HloOpcode::kDomain ||
      domain.enter_domains.count(const_cast<HloInstruction*>(operand)) == 0) {
    return operand->has_sharding() ? &operand->sharding() : nullptr;
  }
  // At this point operand is a kDomain of the currently processed domain, so we
  // can refer to sharding as the domain sharding.
  return &sharding;
}

// Tries to propagate the sharding information into the instructions that are
// part of the domain, in a post order manner (operand propagate to user).
StatusOr<int64> ApplyDomainShardingPass(const DomainMetadata::Domain& domain,
                                        const HloSharding& sharding) {
  int64 assigned = 0;
  for (HloInstruction* instruction : domain.instructions) {
    if (instruction->has_sharding()) {
      continue;
    }
    if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      HloInstruction* tuple = instruction->mutable_operand(0);
      const HloSharding* tuple_sharding =
          GetOperandSharding(tuple, domain, sharding);
      if (tuple_sharding != nullptr) {
        TF_RET_CHECK(tuple_sharding->IsTuple()) << tuple->ToString();
        HloSharding sub_sharding = tuple_sharding->GetSubSharding(
            tuple->shape(), {instruction->tuple_index()});
        VLOG(4) << "  " << instruction->name() << " to sharding "
                << sub_sharding;
        instruction->set_sharding(sub_sharding);
        ++assigned;
      }
    } else if (instruction->opcode() == HloOpcode::kTuple) {
      int64 tuple_assigned = 0;
      ShapeTree<HloSharding> shape_tree = GetTupleSharding(instruction);
      for (int64 i = 0; i < instruction->operand_count(); ++i) {
        const HloSharding* operand_sharding =
            GetOperandSharding(instruction->operand(i), domain, sharding);
        if (operand_sharding != nullptr &&
            shape_tree.element({i}) != *operand_sharding) {
          *shape_tree.mutable_element({i}) = *operand_sharding;
          ++tuple_assigned;
        }
      }
      if (tuple_assigned > 0) {
        HloSharding tuple_sharding = HloSharding::Tuple(shape_tree);
        VLOG(4) << "  " << instruction->name() << " to sharding "
                << tuple_sharding;
        instruction->set_sharding(tuple_sharding);
        ++assigned;
      }
    } else {
      // If all the operand of the given instruction has the same single device
      // assignment, assign that device to this instruction as well.
      const HloSharding* common_sharding = nullptr;
      for (const HloInstruction* operand : instruction->operands()) {
        const HloSharding* operand_sharding =
            GetOperandSharding(operand, domain, sharding);
        if (operand_sharding != nullptr) {
          if (common_sharding != nullptr &&
              *common_sharding != *operand_sharding) {
            common_sharding = nullptr;
            break;
          }
          common_sharding = operand_sharding;
        }
      }
      if (common_sharding != nullptr) {
        VLOG(4) << "  " << instruction->name() << " to sharding "
                << *common_sharding;
        instruction->set_sharding(*common_sharding);
        ++assigned;
      }
    }
  }
  return assigned;
}

Status ApplyDomainSharding(const DomainMetadata::Domain& domain,
                           const HloSharding& sharding) {
  auto device = ShardingUniqueDevice(sharding);
  if (device) {
    // Shortcut the simple case. We have a unique device sharding, so we call
    // the ApplyDomainDeviceSharding() API which will apply array or tuple
    // shaped device sharding to the domain instructions.
    return ApplyDomainDeviceSharding(domain, *device);
  }
  VLOG(1) << "Assigning non-trivial sharding " << sharding;
  for (;;) {
    TF_ASSIGN_OR_RETURN(int64 assigned,
                        ApplyDomainShardingPass(domain, sharding));
    if (assigned == 0) {
      break;
    }
  }
  int64 unassigned = 0;
  for (HloInstruction* instruction : domain.instructions) {
    if (!instruction->has_sharding()) {
      LOG(WARNING) << "Unassigned instruction: " << instruction->ToString();
      ++unassigned;
    }
  }
  // Should we error out if unassigned > 0?
  return Status::OK();
}

// Creates a kDomain instruction to be placed between instruction and operand.
// The kDomain instruction will be created only if the sharding differ between
// the instruction and the operand.
std::unique_ptr<HloInstruction> CreateDomain(HloInstruction* instruction,
                                             HloInstruction* operand) {
  const HloSharding* instruction_sharding =
      instruction->has_sharding() ? &instruction->sharding() : nullptr;
  const HloSharding* operand_sharding =
      operand->has_sharding() ? &operand->sharding() : nullptr;
  // No need for domain if they both have no sharding.
  if (instruction_sharding == nullptr && operand_sharding == nullptr) {
    return nullptr;
  }
  // No need for domain if they match.
  if (instruction_sharding != nullptr && operand_sharding != nullptr &&
      ShardingMatches(*instruction_sharding, *operand_sharding)) {
    return nullptr;
  }
  std::unique_ptr<HloSharding> real_instruction_sharding;
  std::unique_ptr<HloSharding> real_operand_sharding;
  if (instruction_sharding != nullptr) {
    real_instruction_sharding = CloneShardingForDomain(*instruction_sharding);
  }
  if (operand_sharding != nullptr) {
    real_operand_sharding = CloneShardingForDomain(*operand_sharding);
  }
  VLOG(3) << "Creating domain:";
  VLOG(3) << "  Instruction: " << instruction->name();
  VLOG(3) << "  Operand: " << operand->name();
  VLOG(3) << "    User side sharding: "
          << (real_instruction_sharding != nullptr
                  ? real_instruction_sharding->ToString()
                  : "None");
  VLOG(3) << "    Operand side sharding: "
          << (real_operand_sharding != nullptr
                  ? real_operand_sharding->ToString()
                  : "None");

  std::unique_ptr<DomainMetadata> operand_side_metadata =
      MakeUnique<ShardingMetadata>(std::move(real_operand_sharding));
  std::unique_ptr<DomainMetadata> user_side_metadata =
      MakeUnique<ShardingMetadata>(std::move(real_instruction_sharding));
  return HloInstruction::CreateDomain(operand->shape(), operand,
                                      std::move(operand_side_metadata),
                                      std::move(user_side_metadata));
}

StatusOr<std::unique_ptr<HloSharding>> ExtractOriginalCommonSharding(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions) {
  // If we are here, all the instructions being passed had the same sharding
  // (or no sharding), by the means of the ShardingMatches() API.
  // As such, no kDomain was inserted, and here we are asked to extract the
  // original common sharding.
  // All the instructions passed to this API are part of the same computation.
  const HloSharding* sharding = nullptr;
  for (HloInstruction* instruction : instructions) {
    if (instruction->has_sharding()) {
      if (sharding == nullptr) {
        sharding = &instruction->sharding();
      } else {
        TF_RET_CHECK(ShardingMatches(*sharding, instruction->sharding()))
            << "Sharding " << *sharding << " does not match the one in "
            << instruction->ToString();
      }
    }
  }
  if (sharding == nullptr) {
    return std::unique_ptr<HloSharding>();
  }
  VLOG(4) << "Extracted sharding is " << *sharding;
  return CloneShardingForDomain(*sharding);
}

}  // namespace

std::unique_ptr<DomainMetadata> ShardingMetadata::Clone() const {
  std::unique_ptr<HloSharding> sharding;
  if (sharding_ != nullptr) {
    sharding = MakeUnique<HloSharding>(*sharding_);
  }
  return MakeUnique<ShardingMetadata>(std::move(sharding));
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

string ShardingMetadata::ToString() const {
  return sharding_ != nullptr ? sharding_->ToString() : "None";
}

Status ShardingMetadata::NormalizeInstructions(
    const DomainMetadata::Domain& domain) const {
  if (sharding_ != nullptr) {
    VLOG(4) << "Normalizing sharding to " << sharding_->ToString() << ":";
    TF_RETURN_IF_ERROR(ApplyDomainSharding(domain, *sharding_));
    TF_RETURN_IF_ERROR(FixupPassThroughDomainLinks(domain, *sharding_));
  }
  return Status::OK();
}

Status NormalizeShardingDomain(const DomainMetadata::Domain& domain) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloSharding> sharding,
                      ExtractOriginalCommonSharding(domain.instructions));
  if (sharding != nullptr) {
    VLOG(4) << "Normalizing sharding-less domain to " << sharding->ToString()
            << ":";
    TF_RETURN_IF_ERROR(ApplyDomainSharding(domain, *sharding));
  } else {
    VLOG(1) << "Unable to find common sharding";
  }
  return Status::OK();
}

std::unique_ptr<HloInstruction> CreateShardingDomain(
    HloInstruction* instruction, HloInstruction* operand) {
  return CreateDomain(instruction, operand);
}

}  // namespace xla
