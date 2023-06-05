/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tools/mlir_interpreter/framework/registration.h"

#include <functional>
#include <utility>

namespace mlir {
namespace interpreter {
namespace detail {
namespace {

// Aliases and function names are wrapped in functions because function
// registrations are called from static initializers, whose execution order is
// undefined.
DenseMap<llvm::StringRef, llvm::StringRef>& getOpAliases() {
  static DenseMap<llvm::StringRef, llvm::StringRef>* aliases = nullptr;
  if (!aliases) {
    aliases = new DenseMap<llvm::StringRef, llvm::StringRef>();
  }
  return *aliases;
}

DenseMap<llvm::StringRef, InterpreterFunction>& getFunctions() {
  static DenseMap<llvm::StringRef, InterpreterFunction>* functions = nullptr;
  if (!functions) {
    functions = new DenseMap<llvm::StringRef, InterpreterFunction>();
  }
  return *functions;
}

}  // namespace

InterpreterFunction getFunction(llvm::StringRef name) {
  const auto& fns = getFunctions();
  auto fn = fns.find(name);
  if (fn != fns.end()) {
    return fn->second;
  }
  const auto& aliases = getOpAliases();
  auto alias = aliases.find(name);
  if (alias != aliases.end()) {
    return fns.find(alias->second)->second;
  }
  return nullptr;
}

void registerInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&)) {
  registerInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 1 && "unexpected number of operands");
        return {fn(operands[0])};
      });
}

void registerInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&,
                                                  const InterpreterValue&)) {
  registerInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 2 && "unexpected number of operands");
        return {fn(operands[0], operands[1])};
      });
}

void registerInterpreterOp(
    llvm::StringRef name,
    InterpreterValue (*fn)(MutableArrayRef<InterpreterValue>)) {
  registerInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        return {fn(operands)};
      });
}

void registerInterpreterOp(llvm::StringRef name,
                           void (*fn)(MutableArrayRef<InterpreterValue>)) {
  registerInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        fn(operands);
        return {};
      });
}

void registerInterpreterOp(
    llvm::StringRef name,
    std::function<llvm::SmallVector<InterpreterValue>(
        MutableArrayRef<InterpreterValue>, mlir::Operation*, InterpreterState&)>
        fn) {
  getFunctions()[name] = std::move(fn);
}

void registerInterpreterOp(llvm::StringRef name, llvm::StringRef original) {
  getOpAliases()[name] = original;
}

}  // namespace detail
}  // namespace interpreter
}  // namespace mlir
