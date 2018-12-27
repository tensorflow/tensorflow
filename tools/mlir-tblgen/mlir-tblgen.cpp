//===- mlir-tblgen.cpp - Top-Level TableGen implementation for MLIR -------===//
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
// This file contains the main function for MLIR's TableGen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

static llvm::ManagedStatic<std::vector<GenInfo>> generatorRegistry;

mlir::GenRegistration::GenRegistration(StringRef arg, StringRef description,
                                       GenFunction function) {
  generatorRegistry->emplace_back(arg, description, function);
}

GenNameParser::GenNameParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const GenInfo *>(opt) {
  for (const auto &kv : *generatorRegistry) {
    addLiteralOption(kv.getGenArgument(), &kv, kv.getGenDescription());
  }
}

void GenNameParser::printOptionInfo(const llvm::cl::Option &O,
                                    size_t GlobalWidth) const {
  GenNameParser *TP = const_cast<GenNameParser *>(this);
  llvm::array_pod_sort(TP->Values.begin(), TP->Values.end(),
                       [](const GenNameParser::OptionInfo *VT1,
                          const GenNameParser::OptionInfo *VT2) {
                         return VT1->Name.compare(VT2->Name);
                       });
  using llvm::cl::parser;
  parser<const GenInfo *>::printOptionInfo(O, GlobalWidth);
}

// Generator that prints records.
GenRegistration printRecords("print-records", "Print all records to stdout",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               os << records;
                               return false;
                             });

// Generator to invoke.
const mlir::GenInfo *generator;

// TableGenMain requires a function pointer so this function is passed in which
// simply wraps the call to the generator.
static bool MlirTableGenMain(raw_ostream &os, RecordKeeper &records) {
  assert(generator && "no generator specified");
  return generator->invoke(records, os);
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm::cl::opt<const mlir::GenInfo *, false, mlir::GenNameParser> generator(
      "", llvm::cl::desc("Generator to run"));
  cl::ParseCommandLineOptions(argc, argv);
  ::generator = generator.getValue();

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &MlirTableGenMain);
}
