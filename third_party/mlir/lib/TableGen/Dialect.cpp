//===- Dialect.cpp - Dialect wrapper class --------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Dialect.h"
#include "llvm/TableGen/Record.h"

namespace mlir {
namespace tblgen {

StringRef tblgen::Dialect::getName() const {
  return def->getValueAsString("name");
}

StringRef tblgen::Dialect::getCppNamespace() const {
  return def->getValueAsString("cppNamespace");
}

static StringRef getAsStringOrEmpty(const llvm::Record &record,
                                    StringRef fieldName) {
  if (auto valueInit = record.getValueInit(fieldName)) {
    if (llvm::isa<llvm::CodeInit>(valueInit) ||
        llvm::isa<llvm::StringInit>(valueInit))
      return record.getValueAsString(fieldName);
  }
  return "";
}

StringRef tblgen::Dialect::getSummary() const {
  return getAsStringOrEmpty(*def, "summary");
}

StringRef tblgen::Dialect::getDescription() const {
  return getAsStringOrEmpty(*def, "description");
}

bool Dialect::operator==(const Dialect &other) const {
  return def == other.def;
}

bool Dialect::operator<(const Dialect &other) const {
  return getName() < other.getName();
}

} // end namespace tblgen
} // end namespace mlir
