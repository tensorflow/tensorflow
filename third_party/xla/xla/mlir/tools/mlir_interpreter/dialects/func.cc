/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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

#include <tuple>
#include <type_traits>

#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/symbol_finder.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

template <typename T>
bool TypeMatches(mlir::Type type) {
  if constexpr (std::is_same_v<T, float>) {
    return type.isF32();
  } else if constexpr (std::is_same_v<T, double>) {
    return type.isF64();
  } else {
    return false;
  }
}

template <typename Dummy>
bool TypesMatch(ArrayRef<mlir::Type> types) {
  return types.empty();
}

template <typename Dummy, typename T, typename... R>
bool TypesMatch(ArrayRef<mlir::Type> types) {
  if (types.empty() || !TypeMatches<T>(types.front())) return false;
  return TypesMatch<Dummy, R...>(types.drop_front());
}

template <int n, typename... Args>
using Arg = std::tuple_element_t<n, std::tuple<Args...>>;

template <typename Ret, typename... Args>
bool TryCall(void* sym, func::FuncOp callee,
             MutableArrayRef<InterpreterValue> args, InterpreterValue& ret) {
  if (args.size() != callee.getNumArguments() || callee.getNumResults() != 1) {
    return false;
  }

  if (!TypeMatches<Ret>(callee.getResultTypes()[0])) {
    return false;
  }
  if (!TypesMatch<void, Args...>(callee.getArgumentTypes())) {
    return false;
  }

  static_assert(sizeof...(Args) <= 2);
  using FnType = Ret (*)(Args...);
  auto fn = reinterpret_cast<FnType>(sym);
  constexpr int n = sizeof...(Args);

  if constexpr (n == 1) {
    ret = {fn(std::get<Arg<0, Args...>>(args[0].storage))};
  } else {
    static_assert(n == 2);
    ret = {fn(std::get<Arg<0, Args...>>(args[0].storage),
              std::get<Arg<1, Args...>>(args[1].storage))};
  }
  return true;
}

llvm::SmallVector<InterpreterValue> Call(MutableArrayRef<InterpreterValue> args,
                                         mlir::Operation* op,
                                         InterpreterState& state) {
  auto call = llvm::cast<func::CallOp>(op);
  auto callee =
      llvm::cast<func::FuncOp>(state.GetSymbols().lookup(call.getCallee()));
  if (callee->getRegion(0).hasOneBlock()) {
    return Interpret(state, callee.getRegion(), args);
  }

  // Use the new FindSymbolInProcess function
  absl::StatusOr<void*> symbol_status =
      FindSymbolInProcess(callee.getSymName().str());
  if (!symbol_status.ok()) {
    state.AddFailure(symbol_status.status().message());
    return {};
  }
  void* sym = *symbol_status;  // Retrieve the symbol pointer

  InterpreterValue result;
  if (TryCall<float, float>(sym, callee, args, result) ||
      TryCall<float, float, float>(sym, callee, args, result) ||
      TryCall<double, double>(sym, callee, args, result) ||
      TryCall<double, double, double>(sym, callee, args, result)) {
    return {result};
  }
  state.AddFailure("unsupported call target");
  return {};
}

REGISTER_MLIR_INTERPRETER_OP("func.call", Call);
REGISTER_MLIR_INTERPRETER_OP("func.return", NoOpTerminator);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
