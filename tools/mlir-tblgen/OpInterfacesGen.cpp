//===- OpInterfacesGen.cpp - MLIR op interface utility generator ----------===//
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
// OpInterfacesGen generates definitions for operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

namespace {
// This struct represents a single method argument.
struct MethodArgument {
  StringRef type, name;
};

// Wrapper class around a single interface method.
class OpInterfaceMethod {
public:
  explicit OpInterfaceMethod(const llvm::Record *def) : def(def) {
    llvm::DagInit *args = def->getValueAsDag("arguments");
    for (unsigned i = 0, e = args->getNumArgs(); i != e; ++i) {
      arguments.push_back(
          {llvm::cast<llvm::StringInit>(args->getArg(i))->getValue(),
           args->getArgNameStr(i)});
    }
  }

  // Return the return type of this method.
  StringRef getReturnType() const {
    return def->getValueAsString("returnType");
  }

  // Return the name of this method.
  StringRef getName() const { return def->getValueAsString("name"); }

  // Return if this method is static.
  bool isStatic() const { return def->isSubClassOf("StaticInterfaceMethod"); }

  // Return the body for this method if it has one.
  llvm::Optional<StringRef> getBody() const {
    auto value = def->getValueAsString("body");
    return value.empty() ? llvm::Optional<StringRef>() : value;
  }

  // Arguments.
  ArrayRef<MethodArgument> getArguments() const { return arguments; }
  bool arg_empty() const { return arguments.empty(); }

protected:
  // The TableGen definition of this method.
  const llvm::Record *def;

  // The arguments of this method.
  SmallVector<MethodArgument, 2> arguments;
};

// Wrapper class with helper methods for accessing OpInterfaces defined in
// TableGen.
class OpInterface {
public:
  explicit OpInterface(const llvm::Record *def) : def(def) {
    auto *listInit = dyn_cast<llvm::ListInit>(def->getValueInit("methods"));
    for (llvm::Init *init : listInit->getValues())
      methods.emplace_back(cast<llvm::DefInit>(init)->getDef());
  }

  // Return the name of this interface.
  StringRef getName() const { return def->getValueAsString("cppClassName"); }

  // Return the methods of this interface.
  ArrayRef<OpInterfaceMethod> getMethods() const { return methods; }

protected:
  // The TableGen definition of this interface.
  const llvm::Record *def;

  // The methods of this interface.
  SmallVector<OpInterfaceMethod, 8> methods;
};
} // end anonymous namespace

// Emit the method name and argument list for the given method. If
// 'addOperationArg' is true, then an Operation* argument is added to the
// beginning of the argument list.
static void emitMethodNameAndArgs(const OpInterfaceMethod &method,
                                  raw_ostream &os, bool addOperationArg) {
  os << method.getName() << '(';
  if (addOperationArg)
    os << "Operation *tablegen_opaque_op" << (method.arg_empty() ? "" : ", ");
  interleaveComma(method.getArguments(), os, [&](const MethodArgument &arg) {
    os << arg.type << " " << arg.name;
  });
  os << ')';
}

static void emitInterfaceDef(const Record &interfaceDef, raw_ostream &os) {
  OpInterface interface(&interfaceDef);
  StringRef interfaceName = interface.getName();

  // Insert the method definitions.
  for (auto &method : interface.getMethods()) {
    os << method.getReturnType() << " " << interfaceName << "::";
    emitMethodNameAndArgs(method, os, /*addOperationArg=*/false);

    // Forward to the method on the concrete operation type.
    os << " {\n      return getImpl()->" << method.getName() << '(';
    if (!method.isStatic())
      os << "getOperation()" << (method.arg_empty() ? "" : ", ");
    interleaveComma(method.getArguments(), os,
                    [&](const MethodArgument &arg) { os << arg.name; });
    os << ");\n  }\n";
  }
}

static bool emitInterfaceDefs(const RecordKeeper &recordKeeper,
                              raw_ostream &os) {
  llvm::emitSourceFileHeader("Operation Interface Definitions", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("OpInterface");
  for (const auto *def : defs)
    emitInterfaceDef(*def, os);
  return false;
}

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
    interleaveComma(method.getArguments(), os,
                    [&](const MethodArgument &arg) { os << arg.name; });
    os << ");\n    }\n";
  }
  os << "  };\n";
}

static void emitInterfaceDecl(const Record &interfaceDef, raw_ostream &os) {
  OpInterface interface(&interfaceDef);
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

  auto defs = recordKeeper.getAllDerivedDefinitions("OpInterface");
  for (const auto *def : defs)
    emitInterfaceDecl(*def, os);
  return false;
}

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
