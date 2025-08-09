/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_INTRINSIC_INTRINSIC_H_
#define XLA_CODEGEN_INTRINSIC_INTRINSIC_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

// A scalar argument or result.
struct Scalar {
  PrimitiveType type;
};

// A vector argument or result.
struct Vec {
  PrimitiveType type;
  size_t width;
};

class Type : public std::variant<Scalar, Vec> {
 public:
  using std::variant<Scalar, Vec>::variant;
  Type(PrimitiveType type, std::optional<size_t> vector_width);

  std::string name() const;
  bool is_scalar() const;
  bool is_vector() const;
  PrimitiveType element_type() const;
  std::optional<size_t> vector_width() const;
  llvm::Type* to_ir_type(llvm::LLVMContext& context) const;
  mlir::Type to_ir_type(mlir::MLIRContext& context) const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Type& type) {
    absl::Format(&sink, "%s", type.name());
  }

  // Shortened builders for the scalar and vector types defined above.
  static constexpr Type S(PrimitiveType type) { return Scalar{type}; }
  static constexpr Type V(PrimitiveType type, size_t width) {
    return Vec{type, width};
  }

  // Verifies that the two types have the same width.
  static absl::Status VerifySameWidth(const Type& a, const Type& b);

  // Verifies that the two types have the same width and element type.
  static absl::Status VerifySameWidthAndElementType(const Type& a,
                                                    const Type& b);

  // Returns the LLVM IR type for the given intrinsic type.
  static llvm::Type* TypeToIrType(Type type, llvm::LLVMContext& context);

  // Returns the MLIR type for the given intrinsic type.
  static mlir::Type TypeToIrType(Type type, mlir::MLIRContext& context);

  // Returns the intrinsic type for the given MLIR type.
  static Type TypeFromIrType(mlir::Type type);

  // Returns the intrinsic type for the given LLVM type.
  static Type TypeFromIrType(llvm::Type* type);

  // Returns the intrinsic type for the given type name, e.g. v4f32.
  static Type FromName(absl::string_view name);
};

// Intrinsics are provided by XLA to expose special features (functions) that
// may only be implemented with code generator support.
//
// XLA intrinsics are conceptually similar to intrinsics in LLVM IR, but
// implemented inside XLA, and during the compilation process they are lowered
// into the LLVM IR.
//
// Some of the builting LLVM intrinsics are lowered to external function calls
// (i.e. llvm.tanh.* is lowered to libm call), or to the 'compiler-rt' function
// calls (i.e. f32 to bf16 truncation). Having external function calls on a hot
// path is very expensive, furthermore they are ofter can't be vectorized by the
// LLVM, which also adds a large performance penalty. We choose to implement
// intrinsics critical for performance inside XLA iteself:
//
// 1. Math functions (exp, tanh, etc.) using polynomial approximations, to avoid
//    external functions calls to libm and to get the same numerics as TPU and
//    HLO Interpreter.
//
// 2. Conversion between different floating point types, to avoid expensive
//    external function calls to compiler-rt, and again to get consistent
//    numerics with TPU and HLO Interpreter (handling Nans and infinities).
//
// Similar to LLVM intrinsics, XLA intrinsics are overloaded on the data type(s)
// and vector width(s) of the argument and result.
template <typename Derived>
class Intrinsic {
 public:
  // Whether the simd function is masked.
  static constexpr bool kIsMasked = false;
  // If false, the first argument is the return type.
  static constexpr bool kLastArgIsReturnType = false;
  // How many arguments this function takes.
  static constexpr int8_t kNumArgs = 1;

  template <typename... Types>
  static std::string Name(Types... args) {
    std::vector<std::string> arg_names = {args.name()...};
    if (Derived::kLastArgIsReturnType) {
      arg_names.insert(--arg_names.end(), "to");
    }
    return absl::StrCat("xla.", Derived::kName, ".",
                        absl::StrJoin(arg_names, "."));
  }

  template <typename... Args>
  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          const Args... types) {
    static_assert(sizeof...(Args) > 0, "At least one argument is required.");
    static_assert((std::is_convertible_v<Args, Type> && ...),
                  "All arguments must be intrinsic::Type.");
    return Derived::CreateDefinition(module, types...);
  }

  template <typename... Args>
  static mlir::func::FuncOp GetOrInsertDeclaration(mlir::OpBuilder& b,
                                                   mlir::ModuleOp& module,
                                                   Args... args) {
    static_assert(sizeof...(Args) > 0, "At least one argument is required.");
    static_assert((std::is_convertible_v<Args, Type> && ...),
                  "All arguments must be intrinsic::Type.");

    std::vector<mlir::Type> types{
        Type::TypeToIrType(args, *module.getContext())...};
    mlir::Type return_type = types.front();
    if (Derived::kLastArgIsReturnType) {
      return_type = types.back();
      types.pop_back();
    }
    mlir::FunctionType type =
        mlir::FunctionType::get(module.getContext(), types, {return_type});

    // Check if the function already exists, and has the correct type.
    std::string name = Name(args...);
    if (auto func = module.lookupSymbol<mlir::func::FuncOp>(name);
        func && func.getFunctionType() == type) {
      return func;
    }

    // If not found or type mismatch, create the declaration.
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());

    auto decl = b.create<mlir::func::FuncOp>(module.getLoc(), name, type);
    decl.setPrivate();
    return decl;
  }

  template <typename... Args>
  static llvm::Function* GetOrInsertDeclaration(llvm::Module* module,
                                                Args... args) {
    static_assert(sizeof...(Args) > 0, "At least one argument is required.");
    static_assert((std::is_convertible_v<Args, Type> && ...),
                  "All arguments must be intrinsic::Type.");
    std::vector<llvm::Type*> types{
        Type::TypeToIrType(args, module->getContext())...};
    llvm::Type* return_type = types.front();
    if (Derived::kLastArgIsReturnType) {
      return_type = types.back();
      types.pop_back();
    }
    auto* function_type = llvm::FunctionType::get(return_type, types, false);
    return llvm::cast<llvm::Function>(
        module->getOrInsertFunction(Name(args...), function_type).getCallee());
  }

  static bool IsSupported(absl::string_view features, Type type) {
    std::vector<std::vector<Type>> supported_types;
    if constexpr (std::is_invocable_v<decltype(Derived::SupportedVectorTypes),
                                      absl::string_view>) {
      supported_types = Derived::SupportedVectorTypes(features);
    } else {
      supported_types = Derived::SupportedVectorTypes();
    }
    for (const auto& supported_args : supported_types) {
      const Type& first_arg = supported_args.front();
      if (first_arg.element_type() == type.element_type() &&
          first_arg.vector_width() == type.vector_width()) {
        return true;
      }
    }
    return false;
  }
};
}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_INTRINSIC_H_
