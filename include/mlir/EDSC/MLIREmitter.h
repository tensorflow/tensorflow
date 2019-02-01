//===- MLIREmitter.h - MLIR EDSC Emitter Class ------------------*- C++ -*-===//
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
// Provides a simple interface to bind leaf edsc::Expr to Value* and emit the
// corresponding MLIR.
//
// In a first approximation this EDSC can be viewed as simple helper classes
// around MLIR builders. This bears resemblance with Halide but it is more
// generally designed to be automatically generated from various IR dialects in
// the future.
// The implementation is supported by a lightweight by-value abstraction on a
// scoped BumpAllocator with similarities to AffineExpr and NestedPattern.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_EDSC_MLIREMITTER_H_
#define MLIR_LIB_EDSC_MLIREMITTER_H_

#include <tuple>
#include <utility>

#include "mlir/EDSC/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
class FuncBuilder;
class Value;

namespace edsc {

/// The MLIREmitter class is the supporting abstraction to make arbitrary MLIR
/// dialects programmable in a declarative style. As such it is a "generator"
/// counterpart of pattern-matchers.
/// The purpose of the MLIREmitter is to:
///  1. maintain a map of `Bindable` leaf expressions to concrete Value*;
///  2. provide helper functionality to specify bindings of `Bindable` classes
///     to Value* while verifying comformable types;
///  3. traverse the `Expr` and emit the MLIR at the point of insertion of the
///     FuncBuilder.
struct MLIREmitter {
  using BindingMap = llvm::DenseMap<Expr, Value *>;

  MLIREmitter(FuncBuilder *builder, Location location);

  FuncBuilder *getBuilder() { return builder; }
  Location getLocation() { return location; }

  /// Registers a new binding and type-checks. If a certain Expr type is
  /// registered, makes sure the Value is of the proper type.
  MLIREmitter &bind(Bindable e, Value *v);
  /// Constant values can be created on the spot and bound.
  template <typename SSAConstantType, typename... Args>
  MLIREmitter &bindConstant(Bindable e, Args... args) {
    return bind(e, builder->create<SSAConstantType>(location, args...));
  }

  /// Registers new bindings and type-checks. If a certain Expr type is
  /// registered, makes sure the Value is of the proper type.
  ///
  /// Binds elements one at a time. This may seem inefficient at first glance,
  /// but each binding is actually type checked.
  template <typename ZipRangeType>
  MLIREmitter &bindZipRange(const ZipRangeType &range) {
    static_assert(std::tuple_size<decltype(range.begin().iterators)>::value ==
                      2,
                  "Need a zip between 2 collections");
    for (auto it : range) {
      bind(std::get<0>(it), std::get<1>(it));
    }
    return *this;
  }

  template <typename SSAConstantType, typename ZipRangeType>
  MLIREmitter &bindZipRangeConstants(const ZipRangeType &range) {
    static_assert(std::tuple_size<decltype(range.begin().iterators)>::value ==
                      2,
                  "Need a zip between 2 collections");
    for (auto it : range) {
      bindConstant<SSAConstantType>(std::get<0>(it), std::get<1>(it));
    }
    return *this;
  }

  /// Emits the MLIR for `stmt` and inserts at the `builder`'s insertion point.
  /// Prerequisites: all the Bindables have been bound.
  void emitStmt(const Stmt &stmt);
  void emitStmts(llvm::ArrayRef<Stmt> stmts);

  /// Returns the Value* bound to expr.
  /// Prerequisite: it must exist.
  Value *getValue(Expr expr) { return ssaBindings.lookup(expr); }

  /// Returns unique zero and one Expr that are bound to the corresponding
  /// constant index value.
  Expr zero() { return zeroIndex; }
  Expr one() { return oneIndex; }

  /// Returns a list of Expr that are bound to the function arguments.
  SmallVector<edsc::Expr, 8>
  makeBoundFunctionArguments(mlir::Function *function);

  /// Returns a list of `Bindable` that are bound to the dimensions of the
  /// memRef. The proper DimOp and ConstantOp are constructed at the current
  /// insertion point in `builder`. They can be later hoisted and simplified in
  /// a separate pass.
  ///
  /// Prerequisite:
  /// `memRef` is a Value of type MemRefType.
  SmallVector<edsc::Expr, 8> makeBoundMemRefShape(Value *memRef);

  /// A BoundMemRefView represents the information required to step through a
  /// memref. It has placeholders for non-contiguous tensors that fit within the
  /// Fortran subarray model.
  /// It is extracted from a memref
  struct BoundMemRefView {
    SmallVector<edsc::Expr, 8> lbs;
    SmallVector<edsc::Expr, 8> ubs;
    SmallVector<edsc::Expr, 8> steps;
    edsc::Expr dim(unsigned idx) const { return ubs[idx]; }
    unsigned rank() const { return lbs.size(); }
  };
  BoundMemRefView makeBoundMemRefView(Value *memRef);
  BoundMemRefView makeBoundMemRefView(Expr memRef);
  template <typename IterType>
  SmallVector<BoundMemRefView, 8> makeBoundMemRefViews(IterType begin,
                                                       IterType end) {
    static_assert(!std::is_same<decltype(*begin), Indexed>(),
                  "Must use Bindable or Expr");
    SmallVector<mlir::edsc::MLIREmitter::BoundMemRefView, 8> res;
    for (auto it = begin; it != end; ++it) {
      res.push_back(makeBoundMemRefView(*it));
    }
    return res;
  }
  template <typename T>
  SmallVector<BoundMemRefView, 8>
  makeBoundMemRefViews(std::initializer_list<T> args) {
    static_assert(!std::is_same<T, Indexed>(), "Must use Bindable or Expr");
    SmallVector<mlir::edsc::MLIREmitter::BoundMemRefView, 8> res;
    for (auto m : args) {
      res.push_back(makeBoundMemRefView(m));
    }
    return res;
  }

private:
  /// Emits the MLIR for `expr` and inserts at the `builder`'s insertion point.
  /// This function must only be called once on a given emitter.
  /// Prerequisites: all the Bindables have been bound.
  Value *emitExpr(Expr expr);
  llvm::SmallVector<Value *, 8> emitExprs(llvm::ArrayRef<Expr> exprs);

  FuncBuilder *builder;
  Location location;
  BindingMap ssaBindings;
  // These are so ubiquitous that we make them bound and available to all.
  Expr zeroIndex, oneIndex;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_LIB_EDSC_MLIREMITTER_H_
