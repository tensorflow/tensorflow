/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/ir/tfl_types.h"

#include <cstddef>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

constexpr StringLiteral kTfTypeDialectNamespace = "tf_type";

// Returns true if `a` and `b` are element types (or non-shaped types) that a
// single run-time value can have.
bool AreElementTypesCastCompatible(Type a, Type b) {
  if (a == b) return true;
  // Resource and variant types of the same kind are compatible regardless of
  // their subtypes.
  if (isa<ResourceType>(a) && isa<ResourceType>(b)) return true;
  if (isa<VariantType>(a) && isa<VariantType>(b)) return true;
  for (StringRef mnemonic : {"resource", "variant"}) {
    if (IsTfTypeWithMnemonic(a, mnemonic) &&
        IsTfTypeWithMnemonic(b, mnemonic)) {
      return true;
    }
  }
  return false;
}

bool AreTypesCastCompatible(Type a, Type b) {
  if (a == b) return true;
  auto a_tensor = dyn_cast<TensorType>(a);
  auto b_tensor = dyn_cast<TensorType>(b);
  if (!a_tensor || !b_tensor) {
    return !a_tensor && !b_tensor && AreElementTypesCastCompatible(a, b);
  }
  return AreElementTypesCastCompatible(a_tensor.getElementType(),
                                       b_tensor.getElementType()) &&
         succeeded(verifyCompatibleShape(a_tensor, b_tensor));
}

template <typename T>
std::optional<Type> ParseTypeWithSubtypes(MLIRContext* context,
                                          StringRef mnemonic,
                                          DialectAsmParser& parser) {
  SmallVector<TensorType, 1> subtypes;
  if (succeeded(parser.parseOptionalLess())) {
    do {
      const llvm::SMLoc loc = parser.getCurrentLocation();
      Type subtype;
      if (parser.parseType(subtype)) return Type();
      auto tensor_type = dyn_cast<TensorType>(subtype);
      if (!tensor_type) {
        parser.emitError(loc) << "expected a tensor subtype for !tfl."
                              << mnemonic << ", got " << subtype;
        return Type();
      }
      subtypes.push_back(tensor_type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater()) return Type();
  }
  return T::get(context, subtypes);
}

void PrintSubtypes(ArrayRef<TensorType> subtypes, DialectAsmPrinter& printer) {
  if (subtypes.empty()) return;
  printer << "<";
  llvm::interleaveComma(subtypes, printer);
  printer << ">";
}

}  // namespace

bool IsTfTypeWithMnemonic(Type type, StringRef mnemonic) {
  if (!type || type.getDialect().getNamespace() != kTfTypeDialectNamespace) {
    return false;
  }
  // The registered names of the `tf_type` types come from
  // tensorflow/core/ir/types/dialect.h: HANDLE_TF_TYPE stringizes the quoted
  // mnemonic (e.g. "\"quint8\""), while ResourceType and VariantType use
  // "tf_type.resource" and "tf_type.variant". Accept both forms. Keep in sync
  // with that file.
  StringRef name = type.getAbstractType().getName();
  if (name.consume_front("\"")) {
    if (!name.consume_back("\"")) return false;
  } else if (!name.consume_front("tf_type.")) {
    return false;
  }
  return name == mnemonic;
}

bool IsStringType(Type type) {
  return isa_and_present<StringType>(type) ||
         IsTfTypeWithMnemonic(type, "string");
}

bool IsQuint8Type(Type type) {
  return isa_and_present<Quint8Type>(type) ||
         IsTfTypeWithMnemonic(type, "quint8");
}

bool IsQuintType(Type type, unsigned width) {
  switch (width) {
    case 8:
      return IsQuint8Type(type);
    case 16:
      return IsTfTypeWithMnemonic(type, "quint16");
    default:
      return false;
  }
}

bool IsResourceType(Type type) {
  return isa_and_present<ResourceType>(type) ||
         IsTfTypeWithMnemonic(type, "resource");
}

bool IsVariantType(Type type) {
  return isa_and_present<VariantType>(type) ||
         IsTfTypeWithMnemonic(type, "variant");
}

bool AreCastCompatible(TypeRange types) {
  // Shape compatibility and element type compatibility are both pairwise
  // conditions that hold for all types iff a common refinement of the types
  // exists.
  for (size_t i = 0; i < types.size(); ++i) {
    for (size_t j = i + 1; j < types.size(); ++j) {
      if (!AreTypesCastCompatible(types[i], types[j])) return false;
    }
  }
  return true;
}

bool ArraysAreCastCompatible(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto [lhs_type, rhs_type] : llvm::zip_equal(lhs, rhs)) {
    if (!AreTypesCastCompatible(lhs_type, rhs_type)) return false;
  }
  return true;
}

std::optional<Type> ParseTflType(MLIRContext* context, StringRef mnemonic,
                                 DialectAsmParser& parser) {
  if (mnemonic == "string") return StringType::get(context);
  if (mnemonic == "quint8") return Quint8Type::get(context);
  if (mnemonic == "resource") {
    return ParseTypeWithSubtypes<ResourceType>(context, mnemonic, parser);
  }
  if (mnemonic == "variant") {
    return ParseTypeWithSubtypes<VariantType>(context, mnemonic, parser);
  }
  return std::nullopt;
}

bool PrintTflType(Type type, DialectAsmPrinter& printer) {
  if (isa<StringType>(type)) {
    printer << "string";
    return true;
  }
  if (isa<Quint8Type>(type)) {
    printer << "quint8";
    return true;
  }
  if (auto resource_type = dyn_cast<ResourceType>(type)) {
    printer << "resource";
    PrintSubtypes(resource_type.getSubtypes(), printer);
    return true;
  }
  if (auto variant_type = dyn_cast<VariantType>(type)) {
    printer << "variant";
    PrintSubtypes(variant_type.getSubtypes(), printer);
    return true;
  }
  return false;
}

}  // namespace TFL
}  // namespace mlir
