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

#ifndef XLA_CODEGEN_MATH_INTRINSIC_H_
#define XLA_CODEGEN_MATH_INTRINSIC_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen {

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
class Intrinsic {
 public:
  // Forward declare supported XLA intrinsics. Individual intrinsics are
  // implemented in separate headers, and this class simply defines templates
  // that get forwarded to the concrete implementation.
  //
  // go/keep-sorted start
  class Erf;
  class Exp;
  class FpTrunc;
  class Ldexp;
  class Log1p;
  class Rsqrt;
  // go/keep-sorted end

  // A scalar argument or result.
  struct Scalar {
    PrimitiveType type;
  };

  // A vector argument or result.
  struct Vec {
    PrimitiveType type;
    size_t width;
  };

  // Intrinsics overloaded on the arguments and result types.
  class Type : public std::variant<Scalar, Vec> {
   public:
    using std::variant<Scalar, Vec>::variant;

    Type(PrimitiveType type, std::optional<size_t> vector_width);

    std::string name() const;
    bool is_scalar() const;
    bool is_vector() const;
    PrimitiveType element_type() const;
    std::optional<size_t> vector_width() const;

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Type& type) {
      absl::Format(&sink, "%s", type.name());
    }
  };

  // Shortened builders for the scalar and vector types defined above.
  static Type S(PrimitiveType type) { return Scalar{type}; }
  static Type V(PrimitiveType type, size_t width) { return Vec{type, width}; }

  // Verifies that the two types have the same width.
  static absl::Status VerifySameWidth(const Type& a, const Type& b);

  // Verifies that the two types have the same width and element type.
  static absl::Status VerifySameWidthAndElementType(const Type& a,
                                                    const Type& b);

  // Returns the LLVM IR type for the given intrinsic type.
  static llvm::Type* TypeToIrType(Type type, llvm::LLVMContext& context);

  // Returns the MLIR type for the given intrinsic type.
  static mlir::Type TypeToIrType(Type type, mlir::MLIRContext& context);

  // Returns the name of the scalar intrinsic for the given data type.
  template <typename Intrinsic>
  static std::string Name(PrimitiveType t0) {
    return Intrinsic::Name(t0);
  }

  // Returns the name of the vector intrinsic for the given data type.
  template <typename Intrinsic>
  static std::string Name(PrimitiveType t0, int64_t vector_width) {
    return Intrinsic::Name(t0, vector_width);
  }

  // Returns the name of the scalar intrinsic for the given data types.
  template <typename Intrinsic>
  static std::string Name(PrimitiveType t0, PrimitiveType t1) {
    return Intrinsic::Name(t0, t1);
  }

  // Returns the name of the vector intrinsic for the given data types (argument
  // and result types have the same vector width).
  template <typename Intrinsic>
  static std::string Name(PrimitiveType t0, PrimitiveType t1,
                          int64_t vector_width) {
    return Intrinsic::Name(t0, t1, vector_width);
  }

  // Returns the declaration of the scalar intrinsic for the given data type.
  template <typename Intrinsic>
  static llvm::Function* GetOrInsertDeclaration(llvm::Module* module,
                                                PrimitiveType t0) {
    return Intrinsic::GetOrInsertDeclaration(module, t0);
  }

  // Returns the declaration of the scalar intrinsic for the given data types.
  template <typename Intrinsic>
  static llvm::Function* GetOrInsertDeclaration(llvm::Module* module,
                                                PrimitiveType t0,
                                                PrimitiveType t1) {
    return Intrinsic::GetOrInsertDeclaration(module, t0, t1);
  }

  // Creates the definition of the vector intrinsic for the given data types.
  template <typename Intrinsic>
  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          PrimitiveType from,
                                                          PrimitiveType to,
                                                          size_t vector_width) {
    return Intrinsic::CreateDefinition(module, from, to, vector_width);
  }

  // Creates the definition of the vector intrinsic for the given data types.
  template <typename Intrinsic>
  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          PrimitiveType type,
                                                          size_t vector_width) {
    return Intrinsic::CreateDefinition(module, type, vector_width);
  }

  static std::string ScalarName(PrimitiveType type) {
    return primitive_util::LowercasePrimitiveTypeName(type);
  }

  static std::string VectorName(PrimitiveType type, int64_t vector_width) {
    return absl::StrCat("v", vector_width, ScalarName(type));
  }

 private:
  static mlir::func::FuncOp GetOrInsertDeclaration(mlir::OpBuilder& b,
                                                   mlir::ModuleOp& module,
                                                   absl::string_view name,
                                                   mlir::FunctionType type);
};

namespace intrinsics {
template <typename Derived>
class UnaryIntrinsic {
 public:
  static std::string Name(PrimitiveType type) {
    return absl::StrCat("xla.", Derived::kName, ".",
                        Intrinsic::ScalarName(type));
  }

  static std::string Name(PrimitiveType type, int64_t vector_width) {
    return absl::StrCat("xla.", Derived::kName, ".",
                        Intrinsic::VectorName(type, vector_width));
  }

  static llvm::Function* GetOrInsertDeclaration(llvm::Module* module,
                                                PrimitiveType prim_type) {
    auto* type =
        llvm_ir::PrimitiveTypeToIrType(prim_type, module->getContext());
    auto* function_type = llvm::FunctionType::get(type, {type}, false);
    return llvm::cast<llvm::Function>(
        module->getOrInsertFunction(Name(prim_type), function_type)
            .getCallee());
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(
      llvm::Module* module, PrimitiveType prim_type, size_t vector_width) {
    return Derived::CreateDefinition(module, prim_type, vector_width);
  }
};
}  // namespace intrinsics

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_MATH_INTRINSIC_H_
