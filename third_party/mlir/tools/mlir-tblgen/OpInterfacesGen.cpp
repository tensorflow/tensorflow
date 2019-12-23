//===- OpInterfacesGen.cpp - MLIR op interface utility generator ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpInterfacesGen generates definitions for operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "DocGenUtilities.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using mlir::tblgen::OpInterface;
using mlir::tblgen::OpInterfaceMethod;

// Emit the method name and argument list for the given method. If
// 'addOperationArg' is true, then an Operation* argument is added to the
// beginning of the argument list.
static void emitMethodNameAndArgs(const OpInterfaceMethod &method,
                                  raw_ostream &os, bool addOperationArg) {
  os << method.getName() << '(';
  if (addOperationArg)
    os << "Operation *tablegen_opaque_op" << (method.arg_empty() ? "" : ", ");
  interleaveComma(method.getArguments(), os,
                  [&](const OpInterfaceMethod::Argument &arg) {
                    os << arg.type << " " << arg.name;
                  });
  os << ')';
}

// Get an array of all OpInterface definitions but exclude those subclassing
// "DeclareOpInterfaceMethods".
static std::vector<Record *>
getAllOpInterfaceDefinitions(const RecordKeeper &recordKeeper) {
  std::vector<Record *> defs =
      recordKeeper.getAllDerivedDefinitions("OpInterface");

  llvm::erase_if(defs, [](const Record *def) {
    return def->isSubClassOf("DeclareOpInterfaceMethods");
  });
  return defs;
}

//===----------------------------------------------------------------------===//
// GEN: Interface definitions
//===----------------------------------------------------------------------===//

static void emitInterfaceDef(OpInterface &interface, raw_ostream &os) {
  StringRef interfaceName = interface.getName();

  // Insert the method definitions.
  for (auto &method : interface.getMethods()) {
    os << method.getReturnType() << " " << interfaceName << "::";
    emitMethodNameAndArgs(method, os, /*addOperationArg=*/false);

    // Forward to the method on the concrete operation type.
    os << " {\n      return getImpl()->" << method.getName() << '(';
    if (!method.isStatic())
      os << "getOperation()" << (method.arg_empty() ? "" : ", ");
    interleaveComma(
        method.getArguments(), os,
        [&](const OpInterfaceMethod::Argument &arg) { os << arg.name; });
    os << ");\n  }\n";
  }
}

static bool emitInterfaceDefs(const RecordKeeper &recordKeeper,
                              raw_ostream &os) {
  llvm::emitSourceFileHeader("Operation Interface Definitions", os);

  for (const auto *def : getAllOpInterfaceDefinitions(recordKeeper)) {
    OpInterface interface(def);
    emitInterfaceDef(interface, os);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface declarations
//===----------------------------------------------------------------------===//

static void emitConceptDecl(OpInterface &interface, raw_ostream &os) {
  os << "  class Concept {\n"
     << "  public:\n"
     << "    virtual ~Concept() = default;\n";

  // Insert each of the pure virtual concept methods.
  for (auto &method : interface.getMethods()) {
    os << "    virtual " << method.getReturnType() << " ";
    emitMethodNameAndArgs(method, os, /*addOperationArg=*/!method.isStatic());
    os << " = 0;\n";
  }
  os << "  };\n";
}

static void emitModelDecl(OpInterface &interface, raw_ostream &os) {
  os << "  template<typename ConcreteOp>\n";
  os << "  class Model : public Concept {\npublic:\n";

  // Insert each of the virtual method overrides.
  for (auto &method : interface.getMethods()) {
    os << "    " << method.getReturnType() << " ";
    emitMethodNameAndArgs(method, os, /*addOperationArg=*/!method.isStatic());
    os << " final {\n";

    // Provide a definition of the concrete op if this is non static.
    if (!method.isStatic()) {
      os << "      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);\n"
         << "      (void)op;\n";
    }

    // Check for a provided body to the function.
    if (auto body = method.getBody()) {
      os << body << "\n    }\n";
      continue;
    }

    // Forward to the method on the concrete operation type.
    os << "      return " << (method.isStatic() ? "ConcreteOp::" : "op.");

    // Add the arguments to the call.
    os << method.getName() << '(';
    interleaveComma(
        method.getArguments(), os,
        [&](const OpInterfaceMethod::Argument &arg) { os << arg.name; });
    os << ");\n    }\n";
  }
  os << "  };\n";
}

static void emitTraitDecl(OpInterface &interface, raw_ostream &os,
                          StringRef interfaceName,
                          StringRef interfaceTraitsName) {
  os << "  template <typename ConcreteOp>\n  "
     << llvm::formatv("struct Trait : public OpInterface<{0},"
                      " detail::{1}>::Trait<ConcreteOp> {{\n",
                      interfaceName, interfaceTraitsName);

  // Insert the default implementation for any methods.
  for (auto &method : interface.getMethods()) {
    auto defaultImpl = method.getDefaultImplementation();
    if (!defaultImpl)
      continue;

    os << "  " << (method.isStatic() ? "static " : "") << method.getReturnType()
       << " ";
    emitMethodNameAndArgs(method, os, /*addOperationArg=*/false);
    os << " {\n" << defaultImpl.getValue() << "  }\n";
  }

  os << "  };\n";
}

static void emitInterfaceDecl(OpInterface &interface, raw_ostream &os) {
  StringRef interfaceName = interface.getName();
  auto interfaceTraitsName = (interfaceName + "InterfaceTraits").str();

  // Emit the traits struct containing the concept and model declarations.
  os << "namespace detail {\n"
     << "struct " << interfaceTraitsName << " {\n";
  emitConceptDecl(interface, os);
  emitModelDecl(interface, os);
  os << "};\n} // end namespace detail\n";

  // Emit the main interface class declaration.
  os << llvm::formatv("class {0} : public OpInterface<{1}, detail::{2}> {\n"
                      "public:\n"
                      "  using OpInterface<{1}, detail::{2}>::OpInterface;\n",
                      interfaceName, interfaceName, interfaceTraitsName);

  // Emit the derived trait for the interface.
  emitTraitDecl(interface, os, interfaceName, interfaceTraitsName);

  // Insert the method declarations.
  for (auto &method : interface.getMethods()) {
    os << "  " << method.getReturnType() << " ";
    emitMethodNameAndArgs(method, os, /*addOperationArg=*/false);
    os << ";\n";
  }
  os << "};\n";
}

static bool emitInterfaceDecls(const RecordKeeper &recordKeeper,
                               raw_ostream &os) {
  llvm::emitSourceFileHeader("Operation Interface Declarations", os);

  for (const auto *def : getAllOpInterfaceDefinitions(recordKeeper)) {
    OpInterface interface(def);
    emitInterfaceDecl(interface, os);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface documentation
//===----------------------------------------------------------------------===//

/// Emit a string corresponding to a C++ type, followed by a space if necessary.
static raw_ostream &emitCPPType(StringRef type, raw_ostream &os) {
  type = type.trim();
  os << type;
  if (type.back() != '&' && type.back() != '*')
    os << " ";
  return os;
}

static void emitInterfaceDoc(const Record &interfaceDef, raw_ostream &os) {
  OpInterface interface(&interfaceDef);

  // Emit the interface name followed by the description.
  os << "## " << interface.getName() << " (" << interfaceDef.getName() << ")";
  if (auto description = interface.getDescription())
    mlir::tblgen::emitDescription(*description, os);

  // Emit the methods required by the interface.
  os << "\n### Methods:\n";
  for (const auto &method : interface.getMethods()) {
    // Emit the method name.
    os << "#### `" << method.getName() << "`\n\n```c++\n";

    // Emit the method signature.
    if (method.isStatic())
      os << "static ";
    emitCPPType(method.getReturnType(), os) << method.getName() << '(';
    interleaveComma(method.getArguments(), os,
                    [&](const OpInterfaceMethod::Argument &arg) {
                      emitCPPType(arg.type, os) << arg.name;
                    });
    os << ");\n```\n";

    // Emit the description.
    if (auto description = method.getDescription())
      mlir::tblgen::emitDescription(*description, os);

    // If the body is not provided, this method must be provided by the
    // operation.
    if (!method.getBody())
      os << "\nNOTE: This method *must* be implemented by the operation.\n\n";
  }
}

static bool emitInterfaceDocs(const RecordKeeper &recordKeeper,
                              raw_ostream &os) {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  os << "# Operation Interface definition\n";

  for (const auto *def : getAllOpInterfaceDefinitions(recordKeeper))
    emitInterfaceDoc(*def, os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface registration hooks
//===----------------------------------------------------------------------===//

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genInterfaceDecls("gen-op-interface-decls",
                      "Generate op interface declarations",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitInterfaceDecls(records, os);
                      });

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genInterfaceDefs("gen-op-interface-defs",
                     "Generate op interface definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       return emitInterfaceDefs(records, os);
                     });

// Registers the operation interface document generator to mlir-tblgen.
static mlir::GenRegistration
    genInterfaceDocs("gen-op-interface-doc",
                     "Generate op interface documentation",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       return emitInterfaceDocs(records, os);
                     });
