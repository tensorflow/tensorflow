//===- ReferenceImplGen.cpp - MLIR reference implementation generator -----===//
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
// ReferenceImplGen uses the description of operations to generate reference
// implementations for the ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

using mlir::tblgen::Operator;

static void emitReferenceImplementations(const RecordKeeper &recordKeeper,
                                         raw_ostream &os) {
  emitSourceFileHeader("Reference implementation file", os);
  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");

  os << "void printRefImplementation(StringRef opName, mlir::Function *f) {\n"
     << "  using namespace ::mlir::edsc;\n"
     << "  edsc::ScopedEDSCContext raiiContext;\n"
     << "  Stmt block;\n"
     << "  FuncBuilder builder(f);\n"
     << "  if (false) {}";
  for (auto *def : defs) {
    Operator op(def);
    auto ref = def->getValueInit("referenceImplementation");
    if (!ref)
      continue;
    os << "else if (opName == \"" << op.getOperationName() << "\") {\n"
       << "  edsc::MLIREmitter emitter(&builder, f->getLoc());\n"
       << ref->getAsUnquotedString() << "\n"
       << "}\n";
  }
  os << " else {"
     << "  f->emitError(\"no reference implementation for \" + opName);\n"
     << "  return;\n}\n";
  os << "  block.print(llvm::outs());\n llvm::outs() << \"\\n\";\n"
     << "}\n";
}

mlir::GenRegistration genRegister("gen-reference-implementations",
                                  "Generate reference implemenations",
                                  [](const RecordKeeper &records,
                                     raw_ostream &os) {
                                    emitReferenceImplementations(records, os);
                                    return false;
                                  });
