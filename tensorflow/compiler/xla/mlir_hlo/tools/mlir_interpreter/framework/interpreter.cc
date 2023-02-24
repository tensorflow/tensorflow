/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tools/mlir_interpreter/framework/interpreter.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {

SmallVector<InterpreterValue> interpret(InterpreterState& state,
                                        Operation& op) {
  auto fn = detail::getFunction(op.getName().getStringRef());
  if (!fn) {
    llvm::errs() << "Unsupported op: " << op.getName().getStringRef() << "\n";
    op.dump();
    state.addFailure("unsupported op");
    return {};
  }
  SmallVector<InterpreterValue> operands;
  for (auto operand : op.getOperands()) {
    operands.push_back(state.getTopScope()->Get(operand));
  }
  state.getOptions().listener->beforeOp(operands, &op);
  auto results = fn(operands, &op, state);
  for (auto* scope = state.getTopScope(); scope != nullptr;
       scope = scope->getParentScope()) {
    scope->verify();
  }
  if (state.hasFailure()) {
    llvm::errs() << "Encountered failure while executing " << op << "\n";
  }
  state.getOptions().listener->afterOp(results);
  state.step();
  return results;
}

SmallVector<InterpreterValue> interpret(InterpreterState& state, Region& region,
                                        ArrayRef<InterpreterValue> bbargs) {
  if (state.hasFailure()) return {};
  assert(region.hasOneBlock() && "expected region to have one block");
  state.getOptions().listener->enterRegion(bbargs, region);
  InterpreterScope scope(state);

  auto& block = region.getBlocks().front();
  for (auto [value, interpreter_value] :
       llvm::zip(block.getArguments(), bbargs)) {
    scope.Set(value, interpreter_value);
  }

  for (mlir::Operation& op : block.without_terminator()) {
    auto results = interpret(state, op);
    if (state.hasFailure()) return {};
    if (results.size() != op.getNumResults()) {
      llvm::errs() << "Unexpected number of results while interpreting "
                   << op.getName().getStringRef() << ". Interpreter bug?\n";
      llvm_unreachable("unexpected number of results");
    }
    for (auto [v, iv] : llvm::zip(op.getResults(), results)) {
      scope.Set(v, iv);
    }
  }
  auto result = interpret(state, *block.getTerminator());
  if (state.hasFailure()) return {};

  state.getOptions().listener->leaveRegion(result);
  return result;
}

InterpreterState::InterpreterState(const mlir::SymbolTable& symbols,
                                   InterpreterOptions options)
    : symbols(symbols), options(options) {
  if (!options.listener) {
    static auto& noOpListener = *new InterpreterListener();
    this->options.listener = &noOpListener;
  }
  if (options.maxSteps) {
    remainingSteps = *options.maxSteps;
  }
}

void InterpreterState::addFailure(llvm::StringRef failure) {
  failed = true;
  options.errorHandler(failure);
}

void InterpreterScope::verify() const {
  for (auto& [_, value] : values) {
    if (value.isTensor() && value.buffer() &&
        !value.buffer()->getFailure().empty()) {
      state.addFailure(value.buffer()->getFailure());
      break;
    }
  }
}

InterpreterScope::~InterpreterScope() {
  verify();
  state.topScope = parentScope;
}

mlir::FailureOr<SmallVector<InterpreterValue>> runInterpreter(
    const mlir::SymbolTable& symbols, mlir::func::FuncOp function,
    ArrayRef<InterpreterValue> args, InterpreterOptions options) {
  InterpreterState state{symbols, options};
  auto results = interpret(state, function.getBody(), args);
  if (state.hasFailure()) {
    return failure();
  }
  return results;
}

}  // namespace interpreter
}  // namespace mlir
