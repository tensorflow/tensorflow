//===- StorageUniquerSupport.h - MLIR Storage Uniquer Utilities -*- C++ -*-===//
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
// This file defines utility classes for interfacing with StorageUniquer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STORAGEUNIQUERSUPPORT_H
#define MLIR_IR_STORAGEUNIQUERSUPPORT_H

#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {
class Location;
class MLIRContext;

namespace detail {
/// Utility class for implementing users of storage classes uniqued by a
/// StorageUniquer. Clients are not expected to interact with this class
/// directly.
template <typename ConcreteT, typename BaseT, typename StorageT,
          typename UniquerT>
class StorageUserBase : public BaseT {
public:
  using BaseT::BaseT;

  /// Utility declarations for the concrete attribute class.
  using Base = StorageUserBase<ConcreteT, BaseT, StorageT, UniquerT>;
  using ImplType = StorageT;

  /// Return a unique identifier for the concrete type.
  static ClassID *getClassID() { return ClassID::getID<ConcreteT>(); }

  /// Provide a default implementation of 'classof' that invokes a 'kindof'
  /// method on the concrete type.
  template <typename T> static bool classof(T val) {
    static_assert(std::is_convertible<ConcreteT, T>::value,
                  "casting from a non-convertible type");
    return ConcreteT::kindof(val.getKind());
  }

protected:
  /// Get or create a new ConcreteT instance within the ctx. This
  /// function is guaranteed to return a non null object and will assert if
  /// the arguments provided are invalid.
  template <typename... Args>
  static ConcreteT get(MLIRContext *ctx, unsigned kind, Args... args) {
    // Ensure that the invariants are correct for construction.
    assert(succeeded(
        ConcreteT::verifyConstructionInvariants(llvm::None, ctx, args...)));
    return UniquerT::template get<ConcreteT>(ctx, kind, args...);
  }

  /// Get or create a new ConcreteT instance within the ctx, defined at
  /// the given, potentially unknown, location. If the arguments provided are
  /// invalid then emit errors and return a null object.
  template <typename... Args>
  static ConcreteT getChecked(const Location &loc, MLIRContext *ctx,
                              unsigned kind, Args... args) {
    // If the construction invariants fail then we return a null attribute.
    if (failed(ConcreteT::verifyConstructionInvariants(loc, ctx, args...)))
      return ConcreteT();
    return UniquerT::template get<ConcreteT>(ctx, kind, args...);
  }

  /// Default implementation that just returns success.
  template <typename... Args>
  static LogicalResult verifyConstructionInvariants(Args... args) {
    return success();
  }

  /// Utility for easy access to the storage instance.
  ImplType *getImpl() const { return static_cast<ImplType *>(this->impl); }
};
} // namespace detail
} // namespace mlir

#endif
