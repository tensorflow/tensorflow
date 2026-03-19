/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"

#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {

SmallVector<InterpreterValue> Interpret(InterpreterState& state,
                                        Operation& op) {
  auto fn = detail::GetFunction(op.getName().getStringRef());
  if (!fn) {
    llvm::errs() << "Unsupported op: " << op.getName().getStringRef() << "\n";
    op.dump();
    state.AddFailure("unsupported op");
    return {};
  }
  SmallVector<InterpreterValue> operands;
  for (auto operand : op.getOperands()) {
    operands.push_back(state.GetTopScope()->Get(operand));
  }
  state.GetOptions().listener->BeforeOp(operands, &op);
  auto results = fn(operands, &op, state);
  for (auto* scope = state.GetTopScope(); scope != nullptr;
       scope = scope->GetParentScope()) {
    scope->Verify();
  }
  if (state.HasFailure()) {
    llvm::errs() << "Encountered failure while executing " << op << "\n";
  }
  state.GetOptions().listener->AfterOp(results);
  state.Step();
  return results;
}

SmallVector<InterpreterValue> Interpret(InterpreterState& state, Region& region,
                                        ArrayRef<InterpreterValue> bbargs) {
  if (state.HasFailure()) return {};
  assert(region.hasOneBlock() && "expected region to have one block");
  state.GetOptions().listener->EnterRegion(bbargs, region);
  InterpreterScope scope(state);

  auto& block = region.getBlocks().front();
  for (auto [value, interpreter_value] :
       llvm::zip(block.getArguments(), bbargs)) {
    scope.Set(value, interpreter_value);
  }

  std::optional<SmallVector<InterpreterValue>> block_results;
  for (mlir::Operation& op : block) {
    auto results = Interpret(state, op);
    if (state.HasFailure()) return {};
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      assert(!block_results.has_value() && "Expected at most one terminator");
      block_results = results;
    } else {
      if (results.size() != op.getNumResults()) {
        llvm::errs() << "Unexpected number of results while interpreting "
                     << op.getName().getStringRef() << ". Interpreter bug?\n";
        llvm_unreachable("unexpected number of results");
      }
      for (auto [v, iv] : llvm::zip(op.getResults(), results)) {
        scope.Set(v, iv);
      }
    }
  }
  if (!block_results) {
    block_results = SmallVector<InterpreterValue>{};
  }
  state.GetOptions().listener->LeaveRegion(*block_results);
  return *std::move(block_results);
}

InterpreterState::InterpreterState(const mlir::SymbolTable& symbols,
                                   InterpreterOptions options)
    : symbols_(symbols), options_(options) {
  if (!options_.listener) {
    static auto& no_op_listener = *new InterpreterListener();
    this->options_.listener = &no_op_listener;
  }
  if (options_.max_steps) {
    remaining_steps_ = *options_.max_steps;
  }
}

void InterpreterState::AddFailure(llvm::StringRef failure) {
  failed_ = true;
  options_.error_handler(failure);
}

void InterpreterScope::Verify() const {
  for (auto& [_, value] : values_) {
    if (value.IsTensor() && value.GetBuffer() &&
        !value.GetBuffer()->GetFailure().empty()) {
      state_.AddFailure(value.GetBuffer()->GetFailure());
      break;
    }
  }
}

InterpreterScope::~InterpreterScope() {
  Verify();
  state_.top_scope_ = parent_scope_;
}

absl::StatusOr<SmallVector<InterpreterValue>> RunInterpreter(
    const mlir::SymbolTable& symbols, mlir::func::FuncOp function,
    ArrayRef<InterpreterValue> args, InterpreterOptions options) {
  InterpreterState state{symbols, std::move(options)};
  auto results = Interpret(state, function.getBody(), args);
  if (state.HasFailure()) {
    return absl::InvalidArgumentError("Interpreter failed, check error logs");
  }
  return results;
}

}  // namespace interpreter
}  // namespace mlir
