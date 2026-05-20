/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"

#include <optional>
#include <string>
#include <variant>

#include "absl/functional/overload.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace {

using ::mlir::triton::MemSemantic;
using ::mlir::triton::MemSyncScope;

static constexpr llvm::StringRef kMask = "mask";
static constexpr llvm::StringRef kNoMask = "nomask";
static constexpr llvm::StringRef kXlaPrefix = "xla";

// Helper to parse MemSemantic from string
absl::StatusOr<MemSemantic> ParseMemSemantic(llvm::StringRef semantic_str) {
  if (semantic_str == "relaxed") {
    return MemSemantic::RELAXED;
  }
  if (semantic_str == "acquire") {
    return MemSemantic::ACQUIRE;
  }
  if (semantic_str == "release") {
    return MemSemantic::RELEASE;
  }
  if (semantic_str == "acqrel") {
    return MemSemantic::ACQUIRE_RELEASE;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown memory semantic: %s",
      absl::string_view(semantic_str.data(), semantic_str.size())));
}

// Helper to parse MemSyncScope from string
absl::StatusOr<MemSyncScope> ParseMemSyncScope(llvm::StringRef scope_str) {
  if (scope_str == "system") {
    return MemSyncScope::SYSTEM;
  }
  if (scope_str == "gpu") {
    return MemSyncScope::GPU;
  }
  if (scope_str == "cta") {
    return MemSyncScope::CTA;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown memory sync scope: %s",
                      absl::string_view(scope_str.data(), scope_str.size())));
}

// Helper to parse Comparator from string
absl::StatusOr<Comparator> ParseComparator(llvm::StringRef comparator_str) {
  if (comparator_str == "eq") {
    return Comparator::EQ;
  }
  if (comparator_str == "lt") {
    return Comparator::LT;
  }
  if (comparator_str == "le") {
    return Comparator::LE;
  }
  if (comparator_str == "gt") {
    return Comparator::GT;
  }
  if (comparator_str == "ge") {
    return Comparator::GE;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown comparator: %s",
      absl::string_view(comparator_str.data(), comparator_str.size())));
}

// Helper to parse mask from string
absl::StatusOr<bool> ParseMask(llvm::StringRef mask_str) {
  if (mask_str == kMask) {
    return true;
  }
  if (mask_str == kNoMask) {
    return false;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown mask: %s", absl::string_view(mask_str.data(), mask_str.size())));
}

// Helper to convert MemSemantic to string
absl::string_view MemSemanticToString(MemSemantic semantic) {
  switch (semantic) {
    case MemSemantic::RELAXED:
      return "relaxed";
    case MemSemantic::ACQUIRE:
      return "acquire";
    case MemSemantic::RELEASE:
      return "release";
    case MemSemantic::ACQUIRE_RELEASE:
      return "acqrel";
  }
  LOG(FATAL) << "Unknown MemSemantic value";
}

// Helper to convert MemSyncScope to string
absl::string_view MemSyncScopeToString(MemSyncScope scope) {
  switch (scope) {
    case MemSyncScope::SYSTEM:
      return "system";
    case MemSyncScope::GPU:
      return "gpu";
    case MemSyncScope::CTA:
      return "cta";
  }
  LOG(FATAL) << "Unknown MemSyncScope value";
}

// Helper to convert Comparator to string
absl::string_view ComparatorToString(Comparator comparator) {
  switch (comparator) {
    case Comparator::EQ:
      return "eq";
    case Comparator::LT:
      return "lt";
    case Comparator::LE:
      return "le";
    case Comparator::GT:
      return "gt";
    case Comparator::GE:
      return "ge";
  }
  LOG(FATAL) << "Unknown Comparator value";
}

// Helper to convert mask boolean to string
absl::string_view MaskToString(bool has_mask) {
  return has_mask ? kMask : kNoMask;
}

// Helper to convert MemSemantic to LLVM AtomicOrdering
LLVM::AtomicOrdering MemSemanticToAtomicOrdering(triton::MemSemantic semantic) {
  switch (semantic) {
    case triton::MemSemantic::RELAXED:
      return LLVM::AtomicOrdering::monotonic;
    case triton::MemSemantic::ACQUIRE:
      return LLVM::AtomicOrdering::acquire;
    case triton::MemSemantic::RELEASE:
      return LLVM::AtomicOrdering::release;
    case triton::MemSemantic::ACQUIRE_RELEASE:
      return LLVM::AtomicOrdering::acq_rel;
  }
  LOG(FATAL) << "Unknown MemSemantic value";
}

LLVM::ICmpPredicate ComparatorToICmpPredicate(Comparator comparator) {
  switch (comparator) {
    case Comparator::EQ:
      return LLVM::ICmpPredicate::eq;
    case Comparator::LT:
      return LLVM::ICmpPredicate::ult;
    case Comparator::LE:
      return LLVM::ICmpPredicate::ule;
    case Comparator::GT:
      return LLVM::ICmpPredicate::ugt;
    case Comparator::GE:
      return LLVM::ICmpPredicate::uge;
  }
  LOG(FATAL) << "Unknown Comparator value";
}

// Helper to convert MemSyncScope to LLVM syncscope string for target backend
llvm::StringRef MemSyncScopeToSyncScope(triton::MemSyncScope scope,
                                        TargetBackend target) {
  if (target == TargetBackend::CUDA) {
    // NVPTX memory model (LLVM standard syncscope names)
    switch (scope) {
      case triton::MemSyncScope::SYSTEM:
        return "";  // System scope for cross-GPU visibility
      case triton::MemSyncScope::GPU:
        return "device";
      case triton::MemSyncScope::CTA:
        return "block";
    }
  } else {  // ROCM
    // AMDGPU memory model
    switch (scope) {
      case triton::MemSyncScope::SYSTEM:
        return "";  // System scope for cross-GPU visibility
      case triton::MemSyncScope::GPU:
        return "agent";
      case triton::MemSyncScope::CTA:
        return "workgroup";
    }
  }
  LOG(FATAL) << "Unknown MemSyncScope value";
}

// Create LLVM ops for GetThreadIdInstruction
mlir::Value CreateGetThreadIdOps(const LLVMOpCreationParams& params) {
  mlir::OpBuilder& builder = params.builder;
  mlir::Type i32_type = builder.getI32Type();

  // Create intrinsic call (backend-specific)
  mlir::StringAttr intrinsic_name = builder.getStringAttr(
      params.target == TargetBackend::CUDA ? "llvm.nvvm.read.ptx.sreg.tid.x"
                                           : "llvm.amdgcn.workitem.id.x");

  LLVM::CallIntrinsicOp intrinsic_call = LLVM::CallIntrinsicOp::create(
      builder, params.loc, i32_type, intrinsic_name, mlir::ValueRange{});

  return intrinsic_call->getResult(0);
}

// Create LLVM ops for AtomicWriteInstruction
mlir::Value CreateAtomicWriteOps(const AtomicWriteInstruction& instruction,
                                 const LLVMOpCreationParams& params) {
  mlir::OpBuilder& builder = params.builder;
  mlir::ValueRange operands = params.operands;
  mlir::Type i32_type = builder.getI32Type();

  // Expected operand layout: [ptr, value, mask?]
  mlir::Value addr = operands[0];
  mlir::Value value = operands[1];
  mlir::Value mask = instruction.has_mask ? operands[2] : mlir::Value{};

  llvm::StringRef syncscope =
      MemSyncScopeToSyncScope(instruction.scope, params.target);
  LLVM::AtomicOrdering ordering =
      MemSemanticToAtomicOrdering(instruction.semantic);

  // Prepare atomic store location
  mlir::Block* exit_block = nullptr;
  if (mask) {
    // Masked atomic: if (mask != 0) { atomic_store } else { nop }
    mlir::Block* current_block = builder.getBlock();
    mlir::Block* atomic_block =
        current_block->splitBlock(builder.getInsertionPoint());
    exit_block = atomic_block->splitBlock(builder.getInsertionPoint());

    // Check mask and branch
    builder.setInsertionPointToEnd(current_block);
    LLVM::ConstantOp zero = LLVM::ConstantOp::create(
        builder, params.loc, i32_type, builder.getI32IntegerAttr(0));
    LLVM::ICmpOp mask_nonzero = LLVM::ICmpOp::create(
        builder, params.loc, LLVM::ICmpPredicate::ne, mask, zero);
    LLVM::CondBrOp::create(builder, params.loc, mask_nonzero, atomic_block,
                           exit_block);

    // Set insertion point for atomic store
    builder.setInsertionPointToStart(atomic_block);
  }

  // Perform atomic store
  LLVM::StoreOp::create(builder, params.loc, value, addr, /*alignment=*/4,
                        /*isVolatile=*/false,
                        /*isNonTemporal=*/false, /*isInvariantGroup=*/false,
                        ordering, builder.getStringAttr(syncscope));

  if (mask) {
    // Complete masked path: branch to exit
    LLVM::BrOp::create(builder, params.loc, exit_block);
    builder.setInsertionPointToStart(exit_block);
  }

  // Return poison value (result not expected to be used)
  return LLVM::PoisonOp::create(builder, params.loc, i32_type);
}

// Create LLVM ops for AtomicSpinWaitInstruction
mlir::Value CreateAtomicSpinWaitOps(
    const AtomicSpinWaitInstruction& instruction,
    const LLVMOpCreationParams& params) {
  mlir::OpBuilder& builder = params.builder;
  mlir::ValueRange operands = params.operands;
  mlir::Type i32_type = builder.getI32Type();

  // Expected operand layout: [ptr, expected, mask?]
  mlir::Value addr = operands[0];
  mlir::Value expected = operands[1];
  mlir::Value mask = instruction.has_mask ? operands[2] : mlir::Value{};

  llvm::StringRef syncscope =
      MemSyncScopeToSyncScope(instruction.scope, params.target);
  LLVM::AtomicOrdering ordering =
      MemSemanticToAtomicOrdering(instruction.semantic);

  // acq_rel is not valid for loads (only for RMW operations)
  // This is guaranteed through parsing and validation
  CHECK_NE(ordering, LLVM::AtomicOrdering::acq_rel)
      << "acq_rel ordering is not supported for atomic loads";

  // Create block structure (common for both masked and unmasked)
  mlir::Block* current_block = builder.getBlock();
  mlir::Block* loop_block =
      current_block->splitBlock(builder.getInsertionPoint());
  // Need to set insertion point to loop_block before splitting it
  builder.setInsertionPointToStart(loop_block);
  mlir::Block* exit_block = loop_block->splitBlock(builder.getInsertionPoint());
  exit_block->addArgument(i32_type, params.loc);

  builder.setInsertionPointToEnd(current_block);

  if (mask) {
    // Masked: conditional branch based on mask (if mask==0, skip loop)
    LLVM::ConstantOp zero = LLVM::ConstantOp::create(
        builder, params.loc, i32_type, builder.getI32IntegerAttr(0));
    LLVM::ICmpOp mask_nonzero = LLVM::ICmpOp::create(
        builder, params.loc, LLVM::ICmpPredicate::ne, mask, zero);
    LLVM::CondBrOp::create(builder, params.loc, mask_nonzero, loop_block,
                           mlir::ValueRange{}, exit_block,
                           mlir::ValueRange{zero}, std::nullopt);
  } else {
    // Unmasked: unconditional branch to loop (required terminator)
    LLVM::BrOp::create(builder, params.loc, mlir::ValueRange{}, loop_block);
  }

  // Loop: atomic load + compare + conditional branch
  builder.setInsertionPointToStart(loop_block);
  LLVM::LoadOp loaded = LLVM::LoadOp::create(
      builder, params.loc, i32_type, addr, /*alignment=*/4,
      /*isVolatile=*/false,
      /*isNonTemporal=*/false, /*isInvariant=*/false,
      /*isInvariantGroup=*/false, ordering, builder.getStringAttr(syncscope));
  LLVM::ICmpPredicate predicate =
      ComparatorToICmpPredicate(instruction.comparator);
  LLVM::ICmpOp condition =
      LLVM::ICmpOp::create(builder, params.loc, predicate, loaded, expected);
  LLVM::CondBrOp::create(
      /*builder=*/builder,
      /*location=*/params.loc,
      /*condition=*/condition,
      /*trueDest=*/exit_block,  // When condition becomes true, exit loop
      /*trueDestOperands=*/mlir::ValueRange{loaded},
      /*falseDest=*/loop_block,  // When condition is false, loop again
      /*falseDestOperands=*/mlir::ValueRange{},
      /*branchWeights=*/std::nullopt);
  // Return exit block argument
  builder.setInsertionPointToStart(exit_block);
  return exit_block->getArgument(0);
}

}  // namespace

absl::StatusOr<ExternFunctionInstruction> ParseExternFunctionName(
    llvm::StringRef func_name) {
  // Function name format: xla_<functionname>_<arg1>_<arg2>_...
  // Split by underscore to get tokens
  llvm::SmallVector<llvm::StringRef, 6> tokens;
  func_name.split(tokens, '_');

  // Must have at least 2 tokens: kXlaPrefix and function name
  if (tokens.size() < 2 || tokens[0] != kXlaPrefix) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid extern function name: %s",
                        absl::string_view(func_name.data(), func_name.size())));
  }

  llvm::StringRef fn_name = tokens[1];
  const int num_args = tokens.size() - 2;

  // xla_getthreadid (2 tokens total)
  if (fn_name == GetThreadIdInstruction::kName) {
    using Instruction = GetThreadIdInstruction;
    if (num_args != Instruction::kNumArgs) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s expects %d arguments, got %d", Instruction::kName,
                          Instruction::kNumArgs, num_args));
    }
    return Instruction{};
  }

  // xla_atomicwrite_<semantic>_<scope>_<mask|nomask> (5 tokens total)
  if (fn_name == AtomicWriteInstruction::kName) {
    using Instruction = AtomicWriteInstruction;
    if (num_args != Instruction::kNumArgs) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s expects %d arguments, got %d", Instruction::kName,
                          Instruction::kNumArgs, num_args));
    }
    TF_ASSIGN_OR_RETURN(MemSemantic semantic, ParseMemSemantic(tokens[2]));
    TF_ASSIGN_OR_RETURN(MemSyncScope scope, ParseMemSyncScope(tokens[3]));
    TF_ASSIGN_OR_RETURN(bool has_mask, ParseMask(tokens[4]));
    return Instruction{semantic, scope, has_mask};
  }

  // xla_atomicspinwait_<semantic>_<scope>_<comparator> (6 tokens total)
  if (fn_name == AtomicSpinWaitInstruction::kName) {
    using Instruction = AtomicSpinWaitInstruction;
    if (num_args != Instruction::kNumArgs) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s expects %d arguments, got %d", Instruction::kName,
                          Instruction::kNumArgs, num_args));
    }

    TF_ASSIGN_OR_RETURN(MemSemantic semantic, ParseMemSemantic(tokens[2]));
    TF_ASSIGN_OR_RETURN(MemSyncScope scope, ParseMemSyncScope(tokens[3]));
    TF_ASSIGN_OR_RETURN(Comparator comparator, ParseComparator(tokens[4]));
    TF_ASSIGN_OR_RETURN(bool has_mask, ParseMask(tokens[5]));
    return Instruction{semantic, scope, comparator, has_mask};
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown extern function name: %s",
                      absl::string_view(func_name.data(), func_name.size())));
}

std::string SerializeExternFunctionName(
    const ExternFunctionInstruction& instruction) {
  return std::visit(
      absl::Overload{
          [](const GetThreadIdInstruction&) -> std::string {
            return absl::StrJoin({kXlaPrefix, GetThreadIdInstruction::kName},
                                 "_");
          },
          [](const AtomicWriteInstruction& arg) -> std::string {
            return absl::StrJoin(
                {
                    kXlaPrefix,
                    AtomicWriteInstruction::kName,
                    MemSemanticToString(arg.semantic),
                    MemSyncScopeToString(arg.scope),
                    MaskToString(arg.has_mask),
                },
                "_");
          },
          [](const AtomicSpinWaitInstruction& arg) -> std::string {
            return absl::StrJoin({kXlaPrefix, AtomicSpinWaitInstruction::kName,
                                  MemSemanticToString(arg.semantic),
                                  MemSyncScopeToString(arg.scope),
                                  ComparatorToString(arg.comparator),
                                  MaskToString(arg.has_mask)},
                                 "_");
          },
      },
      instruction);
}

absl::Status ValidateMemorySemantic(
    const ExternFunctionInstruction& instruction) {
  return std::visit(
      absl::Overload{
          [](const GetThreadIdInstruction&) -> absl::Status {
            // No memory semantic validation needed for GetThreadId
            return absl::OkStatus();
          },
          [](const AtomicWriteInstruction& arg) -> absl::Status {
            // AtomicWrite only supports RELAXED or RELEASE semantics
            if (arg.semantic != MemSemantic::RELAXED &&
                arg.semantic != MemSemantic::RELEASE) {
              return absl::InvalidArgumentError(
                  "AtomicWriteOp only supports RELAXED or RELEASE semantics");
            }
            return absl::OkStatus();
          },
          [](const AtomicSpinWaitInstruction& arg) -> absl::Status {
            // AtomicSpinWait supports RELAXED, ACQUIRE, or ACQUIRE_RELEASE
            // semantics
            if (arg.semantic != MemSemantic::RELAXED &&
                arg.semantic != MemSemantic::ACQUIRE) {
              return absl::InvalidArgumentError(
                  "AtomicSpinWaitOp only supports RELAXED or ACQUIRE "
                  "semantics");
            }
            return absl::OkStatus();
          },
      },
      instruction);
}

mlir::Value CreateLLVMOpsForInstruction(
    const ExternFunctionInstruction& instruction,
    const LLVMOpCreationParams& params) {
  return std::visit(
      absl::Overload{
          [&params](const GetThreadIdInstruction&) -> mlir::Value {
            return CreateGetThreadIdOps(params);
          },
          [&params](const AtomicWriteInstruction& arg) -> mlir::Value {
            return CreateAtomicWriteOps(arg, params);
          },
          [&params](const AtomicSpinWaitInstruction& arg) -> mlir::Value {
            return CreateAtomicSpinWaitOps(arg, params);
          },
      },
      instruction);
}

}  // namespace mlir::triton::xla
