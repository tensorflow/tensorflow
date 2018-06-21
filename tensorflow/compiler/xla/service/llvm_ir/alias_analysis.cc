/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/alias_analysis.h"

#include <unordered_set>

#include "llvm/IR/MDBuilder.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace llvm_ir {

// Sentry allocation used to represent parameters of the entry computation in
// alias_scope_metadata_ and noalias_metadata_.
static const BufferAllocation* kParameterAllocation = new BufferAllocation(
    /*index=*/-1, /*size=*/0, /*is_thread_local=*/false, /*is_reusable=*/false,
    LogicalBuffer::Color(0));

void AliasAnalysis::AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                                    llvm_ir::IrArray* array,
                                                    const ShapeIndex& index) {
  BufferAllocation::Slice buffer_slice;
  if (hlo.opcode() == HloOpcode::kParameter) {
    // Parameters may alias with each other but may not alias with our temporary
    // buffers.
    buffer_slice = BufferAllocation::Slice(kParameterAllocation, 0, 0);
  } else {
    const std::set<BufferAllocation::Slice> slices =
        assignment_.GetAllSlices(&hlo, index);
    if (slices.empty() || slices.size() > 1) {
      // Skip HLOs which don't have a buffer assigned or for which the
      // buffer can't be determined statically. We cannot determine their
      // aliasing properties in these cases.
      return;
    }
    buffer_slice = *slices.begin();
  }

  if (module_.config().debug_options().xla_llvm_enable_alias_scope_metadata()) {
    llvm::MDNode*& alias_scope_md = alias_scope_metadata_[buffer_slice];
    if (alias_scope_md == nullptr) {
      alias_scope_md =
          GetAliasScopeMetadataForBuffer(buffer_slice, GetAliasDomain());
    }
    if (alias_scope_md != nullptr) {
      array->AddAliasScopeMetadata(alias_scope_md);
    }
  }

  if (module_.config().debug_options().xla_llvm_enable_noalias_metadata()) {
    llvm::MDNode*& noalias_md = noalias_metadata_[buffer_slice];
    if (noalias_md == nullptr) {
      noalias_md = GetNoaliasMetadataForBuffer(buffer_slice, GetAliasDomain(),
                                               assignment_, hlo);
    }
    if (noalias_md != nullptr) {
      array->AddNoaliasMetadata(noalias_md);
    }
  }

  if (module_.config()
          .debug_options()
          .xla_llvm_enable_invariant_load_metadata()) {
    // Parameters of the entry computation are never stored to, loading from a
    // parameter pointer should always return the same result within a loop.
    if (hlo.opcode() == HloOpcode::kParameter) {
      const std::vector<HloInstruction*>& parameter_instructions =
          module_.entry_computation()->parameter_instructions();
      if (std::find(parameter_instructions.begin(),
                    parameter_instructions.end(),
                    &hlo) != parameter_instructions.end()) {
        array->MarkInvariantOverWholeProgram(context_);
      }
    }
  }
}

llvm::MDNode* AliasAnalysis::GetAliasDomain() {
  llvm::MDBuilder metadata_builder(*context_);
  if (alias_domain_ == nullptr) {
    // We use createAliasScopeDomain rather than createAnonymousAliasScopeDomain
    // so that when functions get inlined, we continue using the one domain,
    // rather than duplicating it (and thus having two AA domains in one
    // function).
    //
    // A side-effect of this is that if you ever compile two HLO modules in the
    // same LLVM module, they'll have the same alias scope domain.  This isn't a
    // problem because the two HLO modules will never interact with one another.
    alias_domain_ =
        metadata_builder.createAliasScopeDomain("XLA global AA domain");
  }
  return alias_domain_;
}

llvm::MDNode* AliasAnalysis::GetAliasScopeMetadataForBuffer(
    const BufferAllocation::Slice& buffer_slice, llvm::MDNode* domain) {
  // While we could synthesize an alias.scope, doing so is not more profitable
  // than LLVM's default behavior.
  if (buffer_slice.allocation() == kParameterAllocation) {
    return nullptr;
  }

  llvm::MDBuilder metadata_builder(domain->getContext());
  llvm::MDNode* scope = metadata_builder.createAliasScope(
      AsStringRef("buffer: " + buffer_slice.ToString()), domain);
  llvm::MDNode* scope_list = llvm::MDNode::get(domain->getContext(), scope);
  return scope_list;
}

llvm::MDNode* AliasAnalysis::GetNoaliasMetadataForBuffer(
    const BufferAllocation::Slice& buffer_slice, llvm::MDNode* domain,
    const BufferAssignment& assignment, const HloInstruction& hlo) {
  // We want to construct a list of buffers which:
  //
  // 1. Do not alias the given buffer.
  // 2. Will plausibly be used in the vicinity of the given buffer.
  //
  // Making the noalias set overly large will result in either a massive
  // slowdown in LLVM or LLVM will just ignore the noalias set.
  //
  // A plausible list of instructions are:
  // 1. Users of the given hlo.
  // 2. Operands of users of the given hlo.
  // 3. Operands of the given hlo.
  //
  // This set can be increased as we need.
  std::vector<const LogicalBuffer*> worklist;
  auto add_buffers_to_worklist =
      [&worklist, &assignment](const HloInstruction* instruction) {
        ShapeUtil::ForEachSubshape(
            instruction->shape(),
            [&](const Shape& /*shape*/, const ShapeIndex& index) {
              for (const LogicalBuffer* buffer :
                   assignment.GetSourceBuffers(instruction, index)) {
                worklist.push_back(buffer);
              }
            });
      };

  for (HloInstruction* user : hlo.users()) {
    add_buffers_to_worklist(user);
    for (HloInstruction* operand : user->operands()) {
      add_buffers_to_worklist(operand);
    }
  }

  add_buffers_to_worklist(&hlo);
  for (HloInstruction* operand : hlo.operands()) {
    add_buffers_to_worklist(operand);
  }

  tensorflow::gtl::FlatSet<BufferAllocation::Slice,
                           BufferAllocation::Slice::Hasher>
      buffers;
  for (const LogicalBuffer* buffer : worklist) {
    // Skip buffers which cannot be added to the noalias set.
    if (!assignment.HasAllocation(*buffer) ||
        buffer->instruction()->opcode() == HloOpcode::kParameter) {
      continue;
    }
    const BufferAllocation::Slice noalias_slice =
        assignment.GetAssignedAllocation(*buffer).GetSlice(*buffer);
    // Our buffer must not overlap with the noalias slice.
    if (!buffer_slice.OverlapsWith(noalias_slice)) {
      buffers.insert(noalias_slice);
      // Some instructions have too many operands, causing the noalias set to be
      // too large. To reduce compilation time (b/31901575), truncate noalias
      // sets to at most 500 elements.
      //
      // Future work: improvements to LLVM's scoped AA that avoid creating a
      // MDNode set for every alias query can help to reduce the compilation
      // time as well.
      constexpr int kMaxNoAliasSetSize = 500;
      if (buffers.size() >= kMaxNoAliasSetSize) {
        break;
      }
    }
  }

  // Don't bother constructing a noalias metadata node if it would be empty.
  if (buffers.empty()) {
    return nullptr;
  }

  llvm::MDBuilder metadata_builder(domain->getContext());
  std::vector<llvm::Metadata*> scopes;
  for (const BufferAllocation::Slice noalias_slice : buffers) {
    llvm::MDNode* scope = metadata_builder.createAliasScope(
        AsStringRef("buffer: " + noalias_slice.ToString()), domain);
    scopes.push_back(scope);
  }
  llvm::MDNode* noalias_list =
      llvm::MDNode::get(domain->getContext(), AsArrayRef(scopes));
  return noalias_list;
}

}  // namespace llvm_ir
}  // namespace xla
