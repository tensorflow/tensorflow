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

#include "xla/codegen/math/intrinsic.h"

#include <cstddef>
#include <optional>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/util.h"

namespace xla::codegen {

template <typename R, typename Scalar, typename Vector>
static R Visit(Scalar scalar, Vector vector, const Intrinsic::Type* type) {
  if (auto* s = std::get_if<Intrinsic::Scalar>(type)) {
    return scalar(*s);
  }
  return vector(std::get<Intrinsic::Vec>(*type));
}

std::string Intrinsic::Type::name() const {
  return Visit<std::string>(
      [](const Scalar& scalar) {
        return primitive_util::LowercasePrimitiveTypeName(scalar.type);
      },
      [](const Vec& vec) {
        return absl::StrCat(
            "v", vec.width,
            primitive_util::LowercasePrimitiveTypeName(vec.type));
      },
      this);
}

bool Intrinsic::Type::is_scalar() const {
  return std::holds_alternative<Scalar>(*this);
}

bool Intrinsic::Type::is_vector() const {
  return std::holds_alternative<Vec>(*this);
}

PrimitiveType Intrinsic::Type::element_type() const {
  return Visit<PrimitiveType>([](const Scalar& scalar) { return scalar.type; },
                              [](const Vec& vec) { return vec.type; }, this);
}

std::optional<size_t> Intrinsic::Type::vector_width() const {
  return Visit<std::optional<size_t>>(
      [](const Scalar& scalar) { return std::nullopt; },
      [](const Vec& vec) { return vec.width; }, this);
}

template <typename Scalar, typename Vector>
static absl::Status VerifyTypes(Scalar scalar, Vector vector,
                                const Intrinsic::Type& a,
                                const Intrinsic::Type& b) {
  // A pair of scalar types.
  auto* sa = std::get_if<Intrinsic::Scalar>(&a);
  auto* sb = std::get_if<Intrinsic::Scalar>(&b);
  if (sa && sb) {
    return scalar(*sa, *sb);
  }

  // A pair of vector types.
  auto* va = std::get_if<Intrinsic::Vec>(&a);
  auto* vb = std::get_if<Intrinsic::Vec>(&b);
  if (va && vb) {
    return vector(*va, *vb);
  }

  return InvalidArgument("Expected types of the same kind, but got %s and %s",
                         a.name(), b.name());
}

absl::Status Intrinsic::VerifySameWidth(const Type& a, const Type& b) {
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

absl::Status Intrinsic::VerifySameWidthAndElementType(const Type& a,
                                                      const Type& b) {
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

llvm::Type* Intrinsic::TypeToIrType(Type type, llvm::LLVMContext& context) {
  auto* elt_type = llvm_ir::PrimitiveTypeToIrType(type.element_type(), context);
  if (auto width = type.vector_width()) {
    return llvm::VectorType::get(elt_type, *width, false);
  }
  return elt_type;
}

}  // namespace xla::codegen
