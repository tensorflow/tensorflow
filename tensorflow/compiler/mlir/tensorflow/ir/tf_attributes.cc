/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

#define GET_ATTRDEF_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.cc.inc"

namespace mlir {
namespace TF {

namespace detail {

// The storage class for ShapeAttr.
struct ShapeAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<ArrayRef<int64_t>, bool>;

  explicit ShapeAttrStorage(ArrayRef<int64_t> shape, bool unranked = false)
      : shape(shape), unranked(unranked) {}

  bool operator==(const KeyTy& key) const {
    return key == KeyTy(shape, unranked);
  }
  static unsigned hashKey(const KeyTy& key) {
    return llvm::hash_combine(key.first, static_cast<char>(key.second));
  }

  // NOLINTNEXTLINE
  static ShapeAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                     const KeyTy& key) {
    return new (allocator.allocate<ShapeAttrStorage>())
        ShapeAttrStorage(allocator.copyInto(key.first), key.second);
  }

  ArrayRef<int64_t> shape;
  bool unranked = false;
};

}  // namespace detail

// Get or create a shape attribute.
ShapeAttr ShapeAttr::get(mlir::MLIRContext* context,
                         llvm::Optional<ArrayRef<int64_t>> shape) {
  if (shape) return Base::get(context, *shape, /*unranked=*/false);

  return Base::get(context, ArrayRef<int64_t>(), /*unranked=*/true);
}

// Get or create a shape attribute.
ShapeAttr ShapeAttr::get(mlir::MLIRContext* context, ShapedType shaped_type) {
  if (shaped_type.hasRank())
    return Base::get(context, shaped_type.getShape(), /*unranked=*/false);

  return Base::get(context, ArrayRef<int64_t>(), /*unranked=*/true);
}

llvm::Optional<ArrayRef<int64_t>> ShapeAttr::getValue() const {
  if (hasRank()) return getShape();
  return llvm::None;
}

bool ShapeAttr::hasRank() const { return !getImpl()->unranked; }

int64_t ShapeAttr::getRank() const {
  assert(hasRank());
  return getImpl()->shape.size();
}

ArrayRef<int64_t> ShapeAttr::getShape() const {
  assert(hasRank());
  return getImpl()->shape;
}

bool ShapeAttr::hasStaticShape() const {
  if (!hasRank()) return false;

  for (auto dim : getShape()) {
    if (dim < 0) return false;
  }

  return true;
}

void TensorFlowDialect::registerAttributes() {
  addAttributes<ShapeAttr, FuncAttr, PlaceholderAttr>();
}

// Print a #tf.func attribute of the following format:
//
//   #tf.func<@symbol, {attr = "value"}>
// or
//   #tf.func<"", {attr = "value"}>
// in case of null symbol ref.
void FuncAttr::print(mlir::DialectAsmPrinter& os) const {
  if (getName().getRootReference().empty())
    os << "func<\"\", " << getAttrs() << ">";
  else
    os << "func<" << getName() << ", " << getAttrs() << ">";
}

// Parses a #tf.func attribute of the following format:
//
//   #tf.func<@symbol, {attr = "value"}>
//
// where the first element is a SymbolRefAttr and the second element is a
// DictionaryAttr.
Attribute FuncAttr::parse(MLIRContext* context, DialectAsmParser& parser,
                          Type type) {
  if (failed(parser.parseLess())) return {};
  llvm::SMLoc loc = parser.getCurrentLocation();
  Attribute name, dict;
  if (failed(parser.parseAttribute(name))) {
    parser.emitError(loc) << "expected symbol while parsing tf.func attribute";
    return {};
  }
  if (auto func_name_str = name.dyn_cast<StringAttr>()) {
    if (!func_name_str.getValue().empty()) {
      parser.emitError(loc)
          << "expected empty string or symbol while parsing tf.func "
             "attribute";
      return {};
    }
    name = SymbolRefAttr::get(context, "");
  }
  if (!name.isa<SymbolRefAttr>()) {
    parser.emitError(loc) << "expected symbol while parsing tf.func attribute";
    return {};
  }
  if (failed(parser.parseComma())) return {};
  loc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(dict)) || !dict.isa<DictionaryAttr>()) {
    parser.emitError(loc)
        << "expected Dictionary attribute while parsing tf.func attribute";
    return {};
  }
  if (failed(parser.parseGreater())) return {};
  return FuncAttr::get(context, name.cast<SymbolRefAttr>(),
                       dict.cast<DictionaryAttr>());
}

void PlaceholderAttr::print(DialectAsmPrinter& os) const {
  os << "placeholder<" << StringAttr::get(getContext(), getValue()) << ">";
}

Attribute PlaceholderAttr::parse(MLIRContext* context, DialectAsmParser& parser,
                                 Type type) {
  if (failed(parser.parseLess())) return {};
  StringRef content;
  if (failed(parser.parseOptionalString(&content))) {
    parser.emitError(parser.getCurrentLocation())
        << "expected string while parsing tf.placeholder attribute";
    return {};
  }
  if (failed(parser.parseGreater())) return {};
  return PlaceholderAttr::get(context, content);
}

OptionalParseResult ParseTensorFlowAttribute(MLIRContext* context,
                                             DialectAsmParser& parser,
                                             StringRef mnemonic, Type type,
                                             Attribute& value) {
  return generatedAttributeParser(context, parser, mnemonic, type, value);
}

void printTensorFlowAttribute(Attribute attr, DialectAsmPrinter& os) {
  (void)generatedAttributePrinter(attr, os);
}

}  // namespace TF
}  // namespace mlir
