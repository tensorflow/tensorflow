//===- MLIRContext.h - MLIR Global Context Class ----------------*- C++ -*-===//
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

#ifndef MLIR_IR_MLIRCONTEXT_H
#define MLIR_IR_MLIRCONTEXT_H

#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>
#include <vector>

namespace mlir {
class AbstractOperation;
class DiagnosticEngine;
class Dialect;
class InFlightDiagnostic;
class Location;
class MLIRContextImpl;
class StorageUniquer;

/// MLIRContext is the top-level object for a collection of MLIR modules.  It
/// holds immortal uniqued objects like types, and the tables used to unique
/// them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with
/// a very generic name ("Context") and because it is uncommon for clients to
/// interact with it.
///
class MLIRContext {
public:
  explicit MLIRContext();
  ~MLIRContext();

  /// Return information about all registered IR dialects.
  std::vector<Dialect *> getRegisteredDialects();

  /// Get a registered IR dialect with the given namespace. If an exact match is
  /// not found, then return nullptr.
  Dialect *getRegisteredDialect(StringRef name);

  /// Get a registered IR dialect for the given derived dialect type. The
  /// derived type must provide a static 'getDialectNamespace' method.
  template <typename T> T *getRegisteredDialect() {
    return static_cast<T *>(getRegisteredDialect(T::getDialectNamespace()));
  }

  /// Return information about all registered operations.  This isn't very
  /// efficient: typically you should ask the operations about their properties
  /// directly.
  std::vector<AbstractOperation *> getRegisteredOperations();

  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() { return *impl; }

  /// Returns the diagnostic engine for this context.
  DiagnosticEngine &getDiagEngine();

  /// Returns the storage uniquer used for creating affine constructs.
  StorageUniquer &getAffineUniquer();

  /// Returns the storage uniquer used for constructing type storage instances.
  /// This should not be used directly.
  StorageUniquer &getTypeUniquer();

  /// Returns the storage uniquer used for constructing attribute storage
  /// instances. This should not be used directly.
  StorageUniquer &getAttributeUniquer();

private:
  const std::unique_ptr<MLIRContextImpl> impl;

  MLIRContext(const MLIRContext &) = delete;
  void operator=(const MLIRContext &) = delete;
};
} // end namespace mlir

#endif // MLIR_IR_MLIRCONTEXT_H
