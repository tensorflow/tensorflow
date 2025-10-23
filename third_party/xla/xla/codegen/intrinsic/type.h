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

#ifndef XLA_CODEGEN_INTRINSIC_TYPE_H_
#define XLA_CODEGEN_INTRINSIC_TYPE_H_

#include <cstddef>
#include <optional>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Type.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "xla/xla_data.pb.h"
namespace xla::codegen::intrinsics {

// A scalar argument or result.
struct Scalar {
  PrimitiveType type;

  bool operator==(const Scalar& other) const { return type == other.type; }
};

// A vector argument or result.
struct Vec {
  PrimitiveType type;
  size_t width;

  bool operator==(const Vec& other) const {
    return type == other.type && width == other.width;
  }
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

}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_TYPE_H_
