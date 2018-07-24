//===- OpImplementation.h - Classes for implementing Op types ---*- C++ -*-===//
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
// This classes used by the implementation details of Op types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPIMPLEMENTATION_H
#define MLIR_IR_OPIMPLEMENTATION_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class AffineMap;
class AffineExpr;

/// This is a pure-virtual base class that exposes the asmprinter hooks
/// necessary to implement a custom print() method.
class OpAsmPrinter {
public:
  OpAsmPrinter() {}
  virtual ~OpAsmPrinter();
  virtual raw_ostream &getStream() const = 0;

  /// Print implementations for various things an operation contains.
  virtual void printOperand(const SSAValue *value) = 0;
  virtual void printType(const Type *type) = 0;
  virtual void printAttribute(const Attribute *attr) = 0;
  virtual void printAffineMap(const AffineMap *map) = 0;
  virtual void printAffineExpr(const AffineExpr *expr) = 0;

  /// Print the entire operation with the default verbose formatting.
  virtual void printDefaultOp(const Operation *op) = 0;

private:
  OpAsmPrinter(const OpAsmPrinter &) = delete;
  void operator=(const OpAsmPrinter &) = delete;
};

// Make the implementations convenient to use.
inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const SSAValue &value) {
  p.printOperand(&value);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const Type &type) {
  p.printType(&type);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const Attribute &attr) {
  p.printAttribute(&attr);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const AffineMap &map) {
  p.printAffineMap(&map);
  return p;
}

template <typename T>
inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const T &other) {
  p.getStream() << other;
  return p;
}

} // end namespace mlir

#endif
