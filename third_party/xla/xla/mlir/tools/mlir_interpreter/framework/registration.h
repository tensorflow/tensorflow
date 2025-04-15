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
#ifndef XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_REGISTRATION_H_
#define XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_REGISTRATION_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TypeName.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"

#define MLIR_INTERPRETER_CONCAT_IMPL(x, y) x##y
#define MLIR_INTERPRETER_CONCAT(x, y) MLIR_INTERPRETER_CONCAT_IMPL(x, y)
#define REGISTER_MLIR_INTERPRETER_OP(args...)                     \
  static int MLIR_INTERPRETER_CONCAT(init_, __COUNTER__) = []() { \
    ::mlir::interpreter::detail::RegisterInterpreterOp(args);     \
    return 1;                                                     \
  }();

namespace mlir {
namespace interpreter {
namespace detail {

// The generic signature for interpreter functions. Typically the type-checked
// form should be used instead.
using InterpreterFunction = std::function<SmallVector<InterpreterValue>(
    MutableArrayRef<InterpreterValue>, mlir::Operation*, InterpreterState&)>;

// Returns the given registered function, or nullptr if not found.
InterpreterFunction GetFunction(llvm::StringRef name);

// Simple unary ops.
void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&));

// Simple binary ops.
void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&,
                                                  const InterpreterValue&));

template <typename T>
struct is_optional : std::false_type {};  // NOLINT

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};  // NOLINT

// Converts the given arguments to the requested type. Supported target types:
// - InterpreterValue storage types (e.g. uint8_t, TensorOrMemref<float>).
//   Scalars will be cast if necessary.
// - std::optional of InterpreterValue storage types
// - ArrayRef of InterpreterValue storage types. Will cast if necessary (e.g.
//   int32_t -> int64_t).
// - no-op conversions: InterpreterValue, ArrayRef<InterpreterValue>
template <typename ArgT>
auto TypedInterpreterOpConvertArg(MutableArrayRef<InterpreterValue> args,
                                  InterpreterState& state) {
  constexpr bool optional = is_optional<ArgT>::value;
  constexpr bool single = std::is_same_v<ArgT, InterpreterValue> ||
                          is_valid_interpreter_value_v<ArgT>;
  if constexpr (optional) {
    if (args.empty()) {
      return ArgT{};
    }
  }

  if constexpr (single || optional) {
    if (args.size() != 1) {
      state.AddFailure("Expected a single argument for the operand");
      return ArgT{};
    }
  }

  auto fail = [&]() {
    state.AddFailure(absl::StrCat("Unable to convert argument (variant index ",
                                  args[0].storage.index(), ") to ",
                                  llvm::getTypeName<ArgT>().str()));
    return ArgT{};
  };

  if constexpr (single) {
    if (auto arg = InterpreterValueDynCast<ArgT>(args[0])) {
      return ArgT{*arg};
    }
    return fail();
  } else if constexpr (optional) {
    using T = std::decay_t<decltype(*std::declval<ArgT>())>;
    if (auto arg = InterpreterValueDynCast<T>(args[0])) {
      return arg;
    }
    return fail();
  } else {
    using E = std::decay_t<decltype(*std::declval<ArgT>().begin())>;
    // Container argument types (e.g. MutableArrayRef<InterpreterValue>,
    // ArrayRef<int64_t>).
    if constexpr (std::is_same_v<E, InterpreterValue>) {
      return ArgT{args};
    } else {
      // Note: we don't cast to ArgT here, because that's typically an ArrayRef,
      // which would lead to returning a reference to a temporary.
      return UnpackInterpreterValues<E>(args);
    }
  }
}

// Converts the given return value. Supported target types:
// - InterpreterValue (no-op conversion)
// - InterpreterValue storage types (the value will be boxed)
// - SmallVector<InterpreterValue>
template <typename RetT>
SmallVector<InterpreterValue> TypedInterpreterOpConvertRet(RetT ret) {
  if constexpr (std::is_same_v<RetT, InterpreterValue>) {
    return {ret};
  } else if constexpr (is_valid_interpreter_value_v<RetT>) {
    return {InterpreterValue{ret}};
  } else if constexpr (std::is_same_v<RetT, SmallVector<InterpreterValue>>) {
    return ret;
  } else {
    using E = std::decay_t<decltype(*std::declval<RetT>().begin())>;
    return PackInterpreterValues(ArrayRef<E>(ret));
  }
}

// Adapts the given function to the generic handler signature
// (SmallVector<InterpreterValue>(MutableArrayRef<InterpreterValue>, Operation*,
// InterpreterState&)).
// See the function below for usage.
template <typename Op, typename Ret, typename... T, size_t... Indices>
void RegisterTypedInterpreterOpImpl(Ret (*fn)(InterpreterState&, Op, T... args),
                                    std::index_sequence<Indices...>) {
  RegisterInterpreterOp(
      Op::getOperationName(),
      [fn](MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
           InterpreterState& state) -> SmallVector<InterpreterValue> {
        auto cast = llvm::dyn_cast<Op>(op);
        if (!cast) {
          state.AddFailure(absl::StrCat(
              "failed to cast op '", op->getName().getStringRef().str(),
              "' to expected type (", llvm::getTypeName<Op>().str(), ")"));
          return {};
        }
        int64_t used_args = 0;
        for (auto i : llvm::seq<uint64_t>(0, sizeof...(T))) {
          used_args += cast.getODSOperandIndexAndLength(i).second;
        }
        if (args.size() != used_args) {
          state.AddFailure("Op handler did not use all arguments");
          return {};
        }

        auto extract_arg = [&](auto index, auto* dummy) {
          auto [pos, length] = cast.getODSOperandIndexAndLength(index);
          using ArgT = std::decay_t<decltype(*dummy)>;
          return TypedInterpreterOpConvertArg<ArgT>(args.slice(pos, length),
                                                    state);
        };

        if constexpr (std::is_same_v<Ret, void>) {
          fn(state, cast, extract_arg(Indices, (std::decay_t<T>*){nullptr})...);
          return {};
        } else {
          Ret ret = fn(state, cast,
                       extract_arg(Indices, (std::decay_t<T>*){nullptr})...);
          return TypedInterpreterOpConvertRet(ret);
        }
      });
}

// registers the given function. The function should take one argument per
// Op operand, in the same order as the Op.
// The argument types should match the operation's operand types:
// - Variadic<...> becomes ArrayRef<...>
// - Optional<...> becomes std::optional<...>
// - Unboxing is optionally supported, e.g. an Optional<Index> operand can be
//   passed to either a std::optional<int64_t> or a
//   std::optional<InterpreterValue>.
// Valid return types are InterpreterValue, SmallVector<InterpreterValue>, void,
// and any type boxable in an InterpreterValue.
template <typename Op, typename Ret, typename... T>
void RegisterInterpreterOp(Ret (*fn)(InterpreterState&, Op, T...)) {
  RegisterTypedInterpreterOpImpl(fn, std::index_sequence_for<T...>{});
}

// Simple variadic ops (single output).
void RegisterInterpreterOp(
    llvm::StringRef name,
    InterpreterValue (*fn)(MutableArrayRef<InterpreterValue>));

// Generic ops.
void RegisterInterpreterOp(
    llvm::StringRef name,
    std::function<llvm::SmallVector<InterpreterValue>(
        MutableArrayRef<InterpreterValue>, mlir::Operation*, InterpreterState&)>
        fn);

void RegisterInterpreterOp(llvm::StringRef name, llvm::StringRef original);

}  // namespace detail
}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_REGISTRATION_H_
