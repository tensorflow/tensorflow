//===- Attributes.h - MLIR Attribute Classes --------------------*- C++ -*-===//
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

#ifndef MLIR_IR_ATTRIBUTES_H
#define MLIR_IR_ATTRIBUTES_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class MLIRContext;
class AffineMap;

/// Instances of the Attribute class are immutable, uniqued, immortal, and owned
/// by MLIRContext.  As such, they are passed around by raw non-const pointer.
class Attribute {
public:
  enum class Kind {
    Bool,
    Integer,
    Float,
    String,
    Array,
    AffineMap,
    // TODO: Function references.
  };

  /// Return the classification for this attribute.
  Kind getKind() const {
    return kind;
  }

  /// Print the attribute.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  explicit Attribute(Kind kind) : kind(kind) {}
  ~Attribute() {}

private:
  /// Classification of the subclass, used for type checking.
  Kind kind : 8;

  Attribute(const Attribute&) = delete;
  void operator=(const Attribute&) = delete;
};

inline raw_ostream &operator<<(raw_ostream &os, const Attribute &attr) {
  attr.print(os);
  return os;
}

class BoolAttr : public Attribute {
public:
  static BoolAttr *get(bool value, MLIRContext *context);

  bool getValue() const {
    return value;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Bool;
  }
private:
  BoolAttr(bool value) : Attribute(Kind::Bool), value(value) {}
  ~BoolAttr() = delete;
  bool value;
};

class IntegerAttr : public Attribute {
public:
  static IntegerAttr *get(int64_t value, MLIRContext *context);

  int64_t getValue() const {
    return value;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Integer;
  }
private:
  IntegerAttr(int64_t value) : Attribute(Kind::Integer), value(value) {}
  ~IntegerAttr() = delete;
  int64_t value;
};

class FloatAttr : public Attribute {
public:
  static FloatAttr *get(double value, MLIRContext *context);

  double getValue() const {
    return value;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Float;
  }
private:
  FloatAttr(double value) : Attribute(Kind::Float), value(value) {}
  ~FloatAttr() = delete;
  double value;
};

class StringAttr : public Attribute {
public:
  static StringAttr *get(StringRef bytes, MLIRContext *context);

  StringRef getValue() const {
    return value;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::String;
  }
private:
  StringAttr(StringRef value) : Attribute(Kind::String), value(value) {}
  ~StringAttr() = delete;
  StringRef value;
};

class ArrayAttr : public Attribute {
public:
  static ArrayAttr *get(ArrayRef<Attribute*> value, MLIRContext *context);

  ArrayRef<Attribute*> getValue() const {
    return value;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::Array;
  }
private:
  ArrayAttr(ArrayRef<Attribute*> value) : Attribute(Kind::Array), value(value){}
  ~ArrayAttr() = delete;
  ArrayRef<Attribute*> value;
};

class AffineMapAttr : public Attribute {
public:
  static AffineMapAttr *get(AffineMap *value, MLIRContext *context);

  AffineMap *getValue() const {
    return value;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Attribute *attr) {
    return attr->getKind() == Kind::AffineMap;
  }
private:
  AffineMapAttr(AffineMap *value) : Attribute(Kind::AffineMap), value(value) {}
  ~AffineMapAttr() = delete;
  AffineMap *value;
};

} // end namespace mlir.

#endif
