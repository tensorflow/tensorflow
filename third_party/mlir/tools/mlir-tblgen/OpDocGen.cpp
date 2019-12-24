//===- OpDocGen.cpp - MLIR operation documentation generator --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDocGen uses the description of operations to generate documentation for the
// operations.
//
//===----------------------------------------------------------------------===//

#include "DocGenUtilities.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

using mlir::tblgen::Operator;

// Emit the description by aligning the text to the left per line (e.g.,
// removing the minimum indentation across the block).
//
// This expects that the description in the tablegen file is already formatted
// in a way the user wanted but has some additional indenting due to being
// nested in the op definition.
void mlir::tblgen::emitDescription(StringRef description, raw_ostream &os) {
  // Determine the minimum number of spaces in a line.
  size_t min_indent = -1;
  StringRef remaining = description;
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    size_t indent = split.first.find_first_not_of(" \t");
    if (indent != StringRef::npos)
      min_indent = std::min(indent, min_indent);
    remaining = split.second;
  }

  // Print out the description indented.
  os << "\n";
  remaining = description;
  bool printed = false;
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    if (split.second.empty()) {
      // Skip last line with just spaces.
      if (split.first.ltrim().empty())
        break;
    }
    // Print empty new line without spaces if line only has spaces, unless no
    // text has been emitted before.
    if (split.first.ltrim().empty()) {
      if (printed)
        os << "\n";
    } else {
      os << split.first.substr(min_indent) << "\n";
      printed = true;
    }
    remaining = split.second;
  }
}

// Emits `str` with trailing newline if not empty.
static void emitIfNotEmpty(StringRef str, raw_ostream &os) {
  if (!str.empty()) {
    emitDescription(str, os);
    os << "\n";
  }
}

static void emitOpDocForDialect(const Dialect &dialect,
                                const std::vector<Operator> &ops,
                                const std::vector<Type> &types,
                                raw_ostream &os) {
  os << "# Dialect '" << dialect.getName() << "' definition\n\n";
  emitIfNotEmpty(dialect.getSummary(), os);
  emitIfNotEmpty(dialect.getDescription(), os);

  // TODO(b/143543720) Generate TOC where extension is not supported.
  os << "[TOC]\n\n";

  // TODO(antiagainst): Add link between use and def for types
  if (!types.empty())
    os << "## Type definition\n\n";
  for (auto type : types) {
    os << "### " << type.getDescription() << "\n";
    emitDescription(type.getTypeDescription(), os);
    os << "\n";
  }

  if (!ops.empty())
    os << "## Operation definition\n\n";
  for (auto op : ops) {
    os << "### " << op.getOperationName() << " (" << op.getQualCppClassName()
       << ")";

    // Emit summary & description of operator.
    if (op.hasSummary())
      os << "\n" << op.getSummary() << "\n";
    os << "\n#### Description:\n\n";
    if (op.hasDescription())
      mlir::tblgen::emitDescription(op.getDescription(), os);

    // Emit operands & type of operand. All operands are numbered, some may be
    // named too.
    os << "\n#### Operands:\n\n";
    for (const auto &operand : op.getOperands()) {
      os << "1. ";
      if (!operand.name.empty())
        os << "`" << operand.name << "`: ";
      else
        os << "&laquo;unnamed&raquo;: ";
      os << operand.constraint.getDescription() << "\n";
    }

    // Emit attributes.
    // TODO: Attributes are only documented by TableGen name, with no further
    // info. This should be improved.
    os << "\n#### Attributes:\n\n";
    if (op.getNumAttributes() > 0) {
      os << "| Attribute | MLIR Type | Description |\n"
         << "| :-------: | :-------: | ----------- |\n";
    }
    for (auto namedAttr : op.getAttributes()) {
      os << "| `" << namedAttr.name << "` | `"
         << namedAttr.attr.getStorageType() << "` | "
         << namedAttr.attr.getDescription() << " attribute |\n";
    }

    // Emit results.
    os << "\n#### Results:\n\n";
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      os << "1. ";
      auto name = op.getResultName(i);
      if (name.empty())
        os << "&laquo;unnamed&raquo;: ";
      else
        os << "`" << name << "`: ";
      os << op.getResultTypeConstraint(i).getDescription() << "\n";
    }

    os << "\n";
  }
}

static void emitOpDoc(const RecordKeeper &recordKeeper, raw_ostream &os) {
  const auto &opDefs = recordKeeper.getAllDerivedDefinitions("Op");
  const auto &typeDefs = recordKeeper.getAllDerivedDefinitions("DialectType");

  std::map<Dialect, std::vector<Operator>> dialectOps;
  std::map<Dialect, std::vector<Type>> dialectTypes;
  for (auto *opDef : opDefs) {
    Operator op(opDef);
    dialectOps[op.getDialect()].push_back(op);
  }
  for (auto *typeDef : typeDefs) {
    Type type(typeDef);
    if (auto dialect = type.getDialect())
      dialectTypes[dialect].push_back(type);
  }

  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  for (auto dialectWithOps : dialectOps)
    emitOpDocForDialect(dialectWithOps.first, dialectWithOps.second,
                        dialectTypes[dialectWithOps.first], os);
}

static mlir::GenRegistration
    genRegister("gen-op-doc", "Generate operation documentation",
                [](const RecordKeeper &records, raw_ostream &os) {
                  emitOpDoc(records, os);
                  return false;
                });
