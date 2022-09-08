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

#include "tensorflow/compiler/xla/service/logical_buffer_analysis.h"

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

namespace {

// Gather fusion instructions from 'instruction' into 'fusion_instructions'.
void GatherFusionInstructions(
    HloInstruction* instruction,
    std::vector<HloInstruction*>* fusion_instructions) {
  CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
  for (auto* fused : instruction->fused_instructions()) {
    if (fused->opcode() == HloOpcode::kFusion) {
      GatherFusionInstructions(fused, fusion_instructions);
    }
  }
  fusion_instructions->push_back(instruction);
}

}  // namespace

/* static */ StatusOr<std::unique_ptr<LogicalBufferAnalysis>>
LogicalBufferAnalysis::Run(const HloModule* module) {
  std::unique_ptr<LogicalBufferAnalysis> analysis(
      new LogicalBufferAnalysis(module));
  TF_RETURN_IF_ERROR(analysis->Analyze());
  return std::move(analysis);
}

Status LogicalBufferAnalysis::Analyze() {
  // Empirically we usually have a few more logical buffers than instructions,
  // so reserve 10% more than the number of instructions to avoid frequent
  // resizes.
  logical_buffers_.clear();
  logical_buffers_.reserve((module_->instruction_count() * 11) / 10);

  // We filter out fusion computations, and get to them through fusion
  // instructions. This is because it's possible to have orphaned (unreachable)
  // fusion computations, and we don't want to try to assign buffers to those.
  std::vector<HloInstruction*> fusion_instructions;
  for (auto* computation : module_->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(this));
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }
      GatherFusionInstructions(instruction, &fusion_instructions);
    }
  }
  for (auto* instruction : fusion_instructions) {
    TF_RETURN_IF_ERROR(instruction->fused_expression_root()->Accept(this));
  }
  return OkStatus();
}

LogicalBuffer& LogicalBufferAnalysis::GetBuffer(LogicalBuffer::Id id) const {
  return *logical_buffers_.at(id);
}

LogicalBuffer& LogicalBufferAnalysis::GetBuffer(HloInstruction* instruction,
                                                const ShapeIndex& index) const {
  return *output_buffers_.at(std::make_pair(instruction, index));
}

void LogicalBufferAnalysis::NewLogicalBuffer(HloInstruction* instruction,
                                             const ShapeIndex& index) {
  LogicalBuffer::Id id = logical_buffers_.size();
  auto buffer = std::make_unique<LogicalBuffer>(instruction, index, id);
  auto position = std::make_pair(instruction, index);
  CHECK(output_buffers_.insert({position, buffer.get()}).second);
  logical_buffers_.push_back(std::move(buffer));
}

Status LogicalBufferAnalysis::DefaultAction(HloInstruction* hlo_instruction) {
  // Create a logical buffer for each output of the instruction.
  ShapeUtil::ForEachSubshape(
      hlo_instruction->shape(),
      [this, hlo_instruction](const Shape& shape, const ShapeIndex& index) {
        NewLogicalBuffer(hlo_instruction, index);
      });

  return OkStatus();
}

Status LogicalBufferAnalysis::HandleGetTupleElement(HloInstruction*) {
  // GetTupleElement does not create buffers.
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleAddDependency(
    HloInstruction* add_dependency) {
  // AddDependency just forwards the value of its zero-th operand and does not
  // create buffers.
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleCopy(HloInstruction* copy) {
  // The top-level buffer (index={}) for kCopy is newly created, but all other
  // buffers (in the case of a tuple shape) come from the operand
  NewLogicalBuffer(copy, /*index=*/{});
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleBitcast(HloInstruction*) {
  // A kBitcast instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand.
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleDomain(HloInstruction*) {
  // A kDomain instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand.
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleRecvDone(HloInstruction* recv_done) {
  // RecvDone produces a two-element tuple containing the data value (which
  // aliases part of its operand) and a token. Only the tuple index table and
  // the token are defined by the RecvDone.
  NewLogicalBuffer(recv_done, /*index=*/{});
  NewLogicalBuffer(recv_done, /*index=*/{1});
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleSend(HloInstruction* send) {
  // Send creates new buffers for the top-level tuple, the context (tuple
  // element at {1}), and the token (tuple element at {2}). Tuple element at {0}
  // is an alias of the Send operand, so we don't need to create a new Logical
  // Buffer for that.
  NewLogicalBuffer(send, /*index=*/{});
  NewLogicalBuffer(send, /*index=*/{1});
  NewLogicalBuffer(send, /*index=*/{2});
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleCopyStart(HloInstruction* copy_start) {
  // CopyStart defines the tuple, target buffer at index {0}, and context at
  // index {2}.
  NewLogicalBuffer(copy_start, /*index=*/{});
  NewLogicalBuffer(copy_start, /*index=*/{0});
  NewLogicalBuffer(copy_start, /*index=*/{2});
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleCopyDone(HloInstruction* copy_done) {
  // The output of CopyDone aliases with operand {0}. CopyDone doesn't create
  // any buffers.
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleTuple(HloInstruction* tuple) {
  // A Tuple instruction only creates the top-level buffer.
  NewLogicalBuffer(tuple, /*index=*/{});
  return OkStatus();
}

Status LogicalBufferAnalysis::HandleCustomCall(HloInstruction* custom_call) {
  auto ccall = Cast<HloCustomCallInstruction>(custom_call);
  absl::flat_hash_set<ShapeIndex> aliased_outputs;
  for (const auto& pair : ccall->output_to_operand_aliasing()) {
    aliased_outputs.insert(pair.first);
  }
  ShapeUtil::ForEachSubshape(ccall->shape(),
                             [&](const Shape& shape, const ShapeIndex& index) {
                               if (!aliased_outputs.contains(index)) {
                                 NewLogicalBuffer(custom_call, index);
                               }
                             });
  return OkStatus();
}

}  // namespace xla
