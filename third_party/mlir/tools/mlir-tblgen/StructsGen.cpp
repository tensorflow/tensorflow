//===- StructsGen.cpp - MLIR struct utility generator ---------------------===//
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
// StructsGen generates common utility functions for grouping attributes into a
// set of structured data.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::StructAttr;

static void
emitStructClass(const Record &structDef, StringRef structName,
                llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                StringRef description, raw_ostream &os) {
  const char *structInfo = R"(
// {0}
class {1} : public mlir::DictionaryAttr)";
  const char *structInfoEnd = R"( {
public:
  using DictionaryAttr::DictionaryAttr;
  static bool classof(mlir::Attribute attr);
)";
  os << formatv(structInfo, description, structName) << structInfoEnd;

  // Declares a constructor function for the tablegen structure.
  //   TblgenStruct::get(MLIRContext context, Type1 Field1, Type2 Field2, ...);
  const char *getInfoDecl = "  static {0} get(\n";
  const char *getInfoDeclArg = "      {0} {1},\n";
  const char *getInfoDeclEnd = "      mlir::MLIRContext* context);\n\n";

  os << llvm::formatv(getInfoDecl, structName);

  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << llvm::formatv(getInfoDeclArg, storage, name);
  }
  os << getInfoDeclEnd;

  // Declares an accessor for the fields owned by the tablegen structure.
  //   namespace::storage TblgenStruct::field1() const;
  const char *fieldInfo = R"(  {0} {1}() const;
)";
  for (const auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << formatv(fieldInfo, storage, name);
  }

  os << "};\n\n";
}

static void emitStructDecl(const Record &structDef, raw_ostream &os) {
  StructAttr structAttr(&structDef);
  StringRef structName = structAttr.getStructClassName();
  StringRef cppNamespace = structAttr.getCppNamespace();
  StringRef description = structAttr.getDescription();
  auto fields = structAttr.getAllFields();

  // Wrap in the appropriate namespace.
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  // Emit the struct class definition
  emitStructClass(structDef, structName, fields, description, os);

  // Close the declared namespace.
  for (auto ns : namespaces)
    os << "} // namespace " << ns << "\n";
}

static bool emitStructDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Struct Utility Declarations", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("StructAttr");
  for (const auto *def : defs) {
    emitStructDecl(*def, os);
  }

  return false;
}

static void emitFactoryDef(llvm::StringRef structName,
                           llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                           raw_ostream &os) {
  const char *getInfoDecl = "{0} {0}::get(\n";
  const char *getInfoDeclArg = "    {0} {1},\n";
  const char *getInfoDeclEnd = "    mlir::MLIRContext* context) {";

  os << llvm::formatv(getInfoDecl, structName);

  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << llvm::formatv(getInfoDeclArg, storage, name);
  }
  os << getInfoDeclEnd;

  const char *fieldStart = R"(
  llvm::SmallVector<mlir::NamedAttribute, {0}> fields;
)";
  os << llvm::formatv(fieldStart, fields.size());

  const char *getFieldInfo = R"(
  assert({0});
  auto {0}_id = mlir::Identifier::get("{0}", context);
  fields.emplace_back({0}_id, {0});
)";

  for (auto field : fields) {
    os << llvm::formatv(getFieldInfo, field.getName());
  }

  const char *getEndInfo = R"(
  Attribute dict = mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<{0}>();
}
)";
  os << llvm::formatv(getEndInfo, structName);
}

static void emitClassofDef(llvm::StringRef structName,
                           llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                           raw_ostream &os) {
  const char *classofInfo = R"(
bool {0}::classof(mlir::Attribute attr))";

  const char *classofInfoHeader = R"(
   auto derived = attr.dyn_cast<mlir::DictionaryAttr>();
   if (!derived)
     return false;
   if (derived.size() != {0})
     return false;
)";

  os << llvm::formatv(classofInfo, structName) << " {";
  os << llvm::formatv(classofInfoHeader, fields.size());

  const char *classofArgInfo = R"(
  auto {0} = derived.get("{0}");
  if (!{0} || !{0}.isa<{1}>())
    return false;
)";
  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << llvm::formatv(classofArgInfo, name, storage);
  }

  const char *classofEndInfo = R"(
  return true;
}
)";
  os << classofEndInfo;
}

static void
emitAccessorDef(llvm::StringRef structName,
                llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                raw_ostream &os) {
  const char *fieldInfo = R"(
{0} {2}::{1}() const {
  auto derived = this->cast<mlir::DictionaryAttr>();
  auto {1} = derived.get("{1}");
  assert({1} && "attribute not found.");
  assert({1}.isa<{0}>() && "incorrect Attribute type found.");
  return {1}.cast<{0}>();
}
)";
  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << llvm::formatv(fieldInfo, storage, name, structName);
  }
}

static void emitStructDef(const Record &structDef, raw_ostream &os) {
  StructAttr structAttr(&structDef);
  StringRef cppNamespace = structAttr.getCppNamespace();
  StringRef structName = structAttr.getStructClassName();
  mlir::tblgen::FmtContext ctx;
  auto fields = structAttr.getAllFields();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  emitFactoryDef(structName, fields, os);
  emitClassofDef(structName, fields, os);
  emitAccessorDef(structName, fields, os);

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
}

static bool emitStructDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Struct Utility Definitions", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("StructAttr");
  for (const auto *def : defs)
    emitStructDef(*def, os);

  return false;
}

// Registers the struct utility generator to mlir-tblgen.
static mlir::GenRegistration
    genStructDecls("gen-struct-attr-decls",
                   "Generate struct utility declarations",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitStructDecls(records, os);
                   });

// Registers the struct utility generator to mlir-tblgen.
static mlir::GenRegistration
    genStructDefs("gen-struct-attr-defs", "Generate struct utility definitions",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    return emitStructDefs(records, os);
                  });
