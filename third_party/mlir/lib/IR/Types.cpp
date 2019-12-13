//===- Types.cpp - MLIR Type Classes --------------------------------------===//
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

#include "mlir/IR/Types.h"
#include "TypeDetail.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

unsigned Type::getKind() const { return impl->getKind(); }

/// Get the dialect this type is registered to.
Dialect &Type::getDialect() const { return impl->getDialect(); }

MLIRContext *Type::getContext() const { return getDialect().getContext(); }

unsigned Type::getSubclassData() const { return impl->getSubclassData(); }
void Type::setSubclassData(unsigned val) { impl->setSubclassData(val); }

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

FunctionType FunctionType::get(ArrayRef<Type> inputs, ArrayRef<Type> results,
                               MLIRContext *context) {
  return Base::get(context, Type::Kind::Function, inputs, results);
}

ArrayRef<Type> FunctionType::getInputs() const {
  return getImpl()->getInputs();
}

unsigned FunctionType::getNumResults() const { return getImpl()->numResults; }

ArrayRef<Type> FunctionType::getResults() const {
  return getImpl()->getResults();
}

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

OpaqueType OpaqueType::get(Identifier dialect, StringRef typeData,
                           MLIRContext *context) {
  return Base::get(context, Type::Kind::Opaque, dialect, typeData);
}

OpaqueType OpaqueType::getChecked(Identifier dialect, StringRef typeData,
                                  MLIRContext *context, Location location) {
  return Base::getChecked(location, context, Kind::Opaque, dialect, typeData);
}

/// Returns the dialect namespace of the opaque type.
Identifier OpaqueType::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

/// Returns the raw type data of the opaque type.
StringRef OpaqueType::getTypeData() const { return getImpl()->typeData; }

/// Verify the construction of an opaque type.
LogicalResult OpaqueType::verifyConstructionInvariants(Optional<Location> loc,
                                                       MLIRContext *context,
                                                       Identifier dialect,
                                                       StringRef typeData) {
  if (!Dialect::isValidNamespace(dialect.strref()))
    return emitOptionalError(loc, "invalid dialect namespace '", dialect, "'");
  return success();
}
