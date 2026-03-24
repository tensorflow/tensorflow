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

#include "xla/codegen/intrinsic/type.h"

#include <cstddef>
#include <optional>
#include <string>
#include <variant>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

namespace {
std::string LowercaseLLVMPrimitiveTypeName(PrimitiveType type) {
  std::string name = primitive_util::LowercasePrimitiveTypeName(type);
  switch (type) {
    case S1:
    case S2:
    case S4:
    case S8:
    case S16:
    case S32:
    case S64:
      name[0] = 'i';
      return name;
    default:
      return name;
  }
}

PrimitiveType FromLowercaseLLVMTypeName(absl::string_view in) {
  auto name = std::string(in);
  if (name[0] == 'i') {
    name[0] = 's';
  }
  return primitive_util::StringToPrimitiveType(name).value();
}
}  // namespace

Type::Type(PrimitiveType type, std::optional<size_t> vector_width) {
  if (vector_width) {
    emplace<1>(Vec{type, *vector_width});
  } else {
    emplace<0>(Scalar{type});
  }
}

namespace {
template <typename ScalarFn, typename VectorFn>
static absl::Status VerifyTypes(ScalarFn scalar, VectorFn vector, const Type& a,
                                const Type& b) {
  // A pair of scalar types.
  auto* sa = std::get_if<Scalar>(&a);
  auto* sb = std::get_if<Scalar>(&b);
  if (sa && sb) {
    return scalar(*sa, *sb);
  }

  // A pair of vector types.
  auto* va = std::get_if<Vec>(&a);
  auto* vb = std::get_if<Vec>(&b);
  if (va && vb) {
    return vector(*va, *vb);
  }

  return InvalidArgument("Expected types of the same kind, but got %s and %s",
                         a.name(), b.name());
}

template <typename R, typename ScalarFn, typename VectorFn>
static R Visit(ScalarFn scalar, VectorFn vector, const Type* type) {
  if (auto* s = std::get_if<Scalar>(type)) {
    return scalar(*s);
  }
  return vector(std::get<Vec>(*type));
}
}  // namespace

std::string Type::name() const {
  return Visit<std::string>(
      [](const Scalar& scalar) {
        return LowercaseLLVMPrimitiveTypeName(scalar.type);
      },
      [](const Vec& vec) {
        return absl::StrCat("v", vec.width,
                            LowercaseLLVMPrimitiveTypeName(vec.type));
      },
      this);
}

Type Type::FromName(absl::string_view name) {
  if (name[0] == 'v') {
    size_t len = absl::ascii_isdigit(name[2]) ? 2 : 1;
    size_t width;
    CHECK(absl::SimpleAtoi(name.substr(1, len), &width)) << name;
    return Type(FromLowercaseLLVMTypeName(name.substr(len + 1)), width);
  }
  return Type(FromLowercaseLLVMTypeName(name), std::nullopt);
}

bool Type::is_scalar() const { return std::holds_alternative<Scalar>(*this); }

bool Type::is_vector() const { return std::holds_alternative<Vec>(*this); }

PrimitiveType Type::element_type() const {
  return Visit<PrimitiveType>([](const Scalar& scalar) { return scalar.type; },
                              [](const Vec& vec) { return vec.type; }, this);
}

std::optional<size_t> Type::vector_width() const {
  return Visit<std::optional<size_t>>(
      [](const Scalar& scalar) { return std::nullopt; },
      [](const Vec& vec) { return vec.width; }, this);
}

absl::Status Type::VerifySameWidth(const Type& a, const Type& b) {
  return VerifyTypes(
      [&](const Scalar&, const Scalar&) { return absl::OkStatus(); },
      [&](const Vec& va, const Vec& vb) -> absl::Status {
        if (va.width != vb.width) {
          return InvalidArgument(
              "Expected vector types with the same width, but got %s and %s",
              a.name(), b.name());
        }
        return absl::OkStatus();
      },
      a, b);
}

absl::Status Type::VerifySameWidthAndElementType(const Type& a, const Type& b) {
  return VerifyTypes(
      [&](const Scalar&, const Scalar&) { return absl::OkStatus(); },
      [&](const Vec& va, const Vec& vb) -> absl::Status {
        if (va.width != vb.width || va.type != vb.type) {
          return InvalidArgument(
              "Expected vector types with the same width and element type, but "
              "got %s and %s",
              a.name(), b.name());
        }
        return absl::OkStatus();
      },
      a, b);
}

llvm::Type* Type::TypeToIrType(Type type, llvm::LLVMContext& context) {
  auto* elt_type = llvm_ir::PrimitiveTypeToIrType(type.element_type(), context);
  if (auto width = type.vector_width()) {
    return llvm::VectorType::get(elt_type, *width, false);
  }
  return elt_type;
}

mlir::Type Type::TypeToIrType(Type type, mlir::MLIRContext& context) {
  auto elt_type = ConvertPrimitiveTypeToMlirType(type.element_type(),
                                                 mlir::Builder(&context));
  if (auto width = type.vector_width()) {
    return mlir::VectorType::get(*width, elt_type.value());
  }
  return elt_type.value();
}

llvm::Type* Type::to_ir_type(llvm::LLVMContext& context) const {
  return Type::TypeToIrType(*this, context);
}

mlir::Type Type::to_ir_type(mlir::MLIRContext& context) const {
  return Type::TypeToIrType(*this, context);
}

Type Type::TypeFromIrType(mlir::Type type) {
  if (auto vec_type = mlir::dyn_cast<mlir::VectorType>(type)) {
    CHECK_EQ(vec_type.getRank(), 1) << "Expected rank 1 for vector type.";
    return Type(ConvertMlirTypeToPrimitiveType(vec_type.getElementType()),
                vec_type.getShape().front());
  }
  return Type(ConvertMlirTypeToPrimitiveType(type), std::nullopt);
}

Type Type::TypeFromIrType(llvm::Type* type) {
  if (llvm::isa<llvm::VectorType>(type)) {
    return Type(llvm_ir::PrimitiveTypeFromIrType(type),
                llvm::cast<llvm::VectorType>(type)
                    ->getElementCount()
                    .getKnownMinValue());
  }
  return Type(llvm_ir::PrimitiveTypeFromIrType(type), std::nullopt);
}

}  // namespace xla::codegen::intrinsics
