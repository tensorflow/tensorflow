//===- Attribute.h - Attribute wrapper class --------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Attribute wrapper to simplify using TableGen Record defining a MLIR
// Attribute.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ATTRIBUTE_H_
#define MLIR_TABLEGEN_ATTRIBUTE_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Constraint.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class with helper methods for accessing attribute constraints defined
// in TableGen.
class AttrConstraint : public Constraint {
public:
  explicit AttrConstraint(const llvm::Record *record);

  static bool classof(const Constraint *c) { return c->getKind() == CK_Attr; }

  // Returns true if this constraint is a subclass of the given `className`
  // class defined in TableGen.
  bool isSubClassOf(StringRef className) const;
};

// Wrapper class providing helper methods for accessing MLIR Attribute defined
// in TableGen. This class should closely reflect what is defined as class
// `Attr` in TableGen.
class Attribute : public AttrConstraint {
public:
  explicit Attribute(const llvm::Record *record);
  explicit Attribute(const llvm::DefInit *init);

  // Returns the storage type if set. Returns the default storage type
  // ("Attribute") otherwise.
  StringRef getStorageType() const;

  // Returns the return type for this attribute.
  StringRef getReturnType() const;

  // Returns the template getter method call which reads this attribute's
  // storage and returns the value as of the desired return type.
  // The call will contain a `{0}` which will be expanded to this attribute.
  StringRef getConvertFromStorageCall() const;

  // Returns true if this attribute can be built from a constant value.
  bool isConstBuildable() const;

  // Returns the template that can be used to produce an instance of the
  // attribute.
  // Syntax: {0} should be replaced with a builder, {1} should be replaced with
  // the constant value.
  StringRef getConstBuilderTemplate() const;

  // Returns the base-level attribute that this attribute constraint is
  // built upon.
  Attribute getBaseAttr() const;

  // Returns whether this attribute has a default value's initializer.
  bool hasDefaultValueInitializer() const;
  // Returns the default value's initializer for this attribute.
  StringRef getDefaultValueInitializer() const;

  // Returns whether this attribute is optional.
  bool isOptional() const;

  // Returns true if this attribute is a derived attribute (i.e., a subclass
  // of `DerivedAttr`).
  bool isDerivedAttr() const;

  // Returns true if this attribute is a type attribute (i.e., a subclass
  // of `TypeAttrBase`).
  bool isTypeAttr() const;

  // Returns true if this attribute is an enum attribute (i.e., a subclass of
  // `EnumAttrInfo`)
  bool isEnumAttr() const;

  // Returns this attribute's TableGen def name. If this is an `OptionalAttr`
  // or `DefaultValuedAttr` without explicit name, returns the base attribute's
  // name.
  StringRef getAttrDefName() const;

  // Returns the code body for derived attribute. Aborts if this is not a
  // derived attribute.
  StringRef getDerivedCodeBody() const;
};

// Wrapper class providing helper methods for accessing MLIR constant attribute
// defined in TableGen. This class should closely reflect what is defined as
// class `ConstantAttr` in TableGen.
class ConstantAttr {
public:
  explicit ConstantAttr(const llvm::DefInit *init);

  // Returns the attribute kind.
  Attribute getAttribute() const;

  // Returns the constant value.
  StringRef getConstantValue() const;

private:
  // The TableGen definition of this constant attribute.
  const llvm::Record *def;
};

// Wrapper class providing helper methods for accessing enum attribute cases
// defined in TableGen. This is used for enum attribute case backed by both
// StringAttr and IntegerAttr.
class EnumAttrCase : public Attribute {
public:
  explicit EnumAttrCase(const llvm::DefInit *init);

  // Returns true if this EnumAttrCase is backed by a StringAttr.
  bool isStrCase() const;

  // Returns the symbol of this enum attribute case.
  StringRef getSymbol() const;

  // Returns the value of this enum attribute case.
  int64_t getValue() const;
};

// Wrapper class providing helper methods for accessing enum attributes defined
// in TableGen.This is used for enum attribute case backed by both StringAttr
// and IntegerAttr.
class EnumAttr : public Attribute {
public:
  explicit EnumAttr(const llvm::Record *record);
  explicit EnumAttr(const llvm::Record &record);
  explicit EnumAttr(const llvm::DefInit *init);

  // Returns true if this is a bit enum attribute.
  bool isBitEnum() const;

  // Returns the enum class name.
  StringRef getEnumClassName() const;

  // Returns the C++ namespaces this enum class should be placed in.
  StringRef getCppNamespace() const;

  // Returns the underlying type.
  StringRef getUnderlyingType() const;

  // Returns the name of the utility function that converts a value of the
  // underlying type to the corresponding symbol.
  StringRef getUnderlyingToSymbolFnName() const;

  // Returns the name of the utility function that converts a string to the
  // corresponding symbol.
  StringRef getStringToSymbolFnName() const;

  // Returns the name of the utility function that converts a symbol to the
  // corresponding string.
  StringRef getSymbolToStringFnName() const;

  // Returns the return type of the utility function that converts a symbol to
  // the corresponding string.
  StringRef getSymbolToStringFnRetType() const;

  // Returns the name of the utilit function that returns the max enum value
  // used within the enum class.
  StringRef getMaxEnumValFnName() const;

  // Returns all allowed cases for this enum attribute.
  std::vector<EnumAttrCase> getAllCases() const;
};

class StructFieldAttr {
public:
  explicit StructFieldAttr(const llvm::Record *record);
  explicit StructFieldAttr(const llvm::Record &record);
  explicit StructFieldAttr(const llvm::DefInit *init);

  StringRef getName() const;
  Attribute getType() const;

private:
  const llvm::Record *def;
};

// Wrapper class providing helper methods for accessing struct attributes
// defined in TableGen.
class StructAttr : public Attribute {
public:
  explicit StructAttr(const llvm::Record *record);
  explicit StructAttr(const llvm::Record &record) : StructAttr(&record){};
  explicit StructAttr(const llvm::DefInit *init);

  // Returns the struct class name.
  StringRef getStructClassName() const;

  // Returns the C++ namespaces this struct class should be placed in.
  StringRef getCppNamespace() const;

  std::vector<StructFieldAttr> getAllFields() const;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_ATTRIBUTE_H_
