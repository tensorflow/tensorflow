//===- mlir-rewriter-gen.cpp - MLIR pattern rewriter generator ------------===//
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
// This is a command line utility that generates rewrite patterns from
// declaritive description.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

enum ActionType { GenRewriters };

static cl::opt<ActionType>
    action(cl::desc("Action to perform:"),
           cl::values(clEnumValN(GenRewriters, "gen-rewriters",
                                 "Generate rewriter definitions")));

static void emitRewriters(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = recordKeeper.getAllDerivedDefinitions("Pattern");

  // Ensure unique patterns simply by appending unique suffix.
  unsigned rewritePatternCount = 0;
  std::string baseRewriteName = "GeneratedConvert";
  for (Record *pattern : patterns) {
    DagInit *tree = pattern->getValueAsDag("PatternToMatch");

    StringMap<int> nameToOrdinal;
    for (int i = 0, e = tree->getNumArgs(); i != e; ++i)
      nameToOrdinal[tree->getArgNameStr(i)] = i;

    // TODO(jpienaar): Expand to multiple matches.
    for (auto arg : tree->getArgs()) {
      if (isa<DagInit>(arg))
        PrintFatalError(pattern->getLoc(),
                        "Only single pattern inputs supported");
    }

    // Emit RewritePattern for Pattern.
    DefInit *root = cast<DefInit>(tree->getOperator());
    std::string rewriteName =
        baseRewriteName + llvm::utostr(rewritePatternCount++);
    auto *rootName = cast<StringInit>(root->getDef()->getValueInit("name"));
    os << "struct " << rewriteName << " : public RewritePattern {\n"
       << "  " << rewriteName << "(MLIRContext *context) : RewritePattern("
       << rootName->getAsString() << ", 1, context) {}\n"
       << "  PatternMatchResult match(Operation *op) const override {\n"
       << "    // TODO: This just handle 1 result\n"
       << "    if (op->getNumResults() != 1) return matchFailure();\n"
       << "    return matchSuccess();\n  }\n";

    ListInit *resultOps = pattern->getValueAsListInit("ResultOps");
    if (resultOps->size() != 1)
      PrintFatalError("Can only handle single result rules");
    DagInit *resultTree = cast<DagInit>(resultOps->getElement(0));

    // TODO(jpienaar): Expand to multiple results.
    for (auto result : resultTree->getArgs()) {
      if (isa<DagInit>(result))
        PrintFatalError(pattern->getLoc(), "Only single op result supported");
    }
    DefInit *resultRoot = cast<DefInit>(resultTree->getOperator());
    std::string opName = resultRoot->getAsUnquotedString();

    SmallVector<StringRef, 2> split;
    SplitString(opName, split, "_");
    auto className = join(split, "::");
    os << "  void rewrite(Operation *op, PatternRewriter &rewriter) const "
       << "override {\n   rewriter.replaceOpWithNewOp<" << className
       << ">(op, op->getResult(0)->getType()";
    for (auto arg : resultTree->getArgNames()) {
      if (!arg)
        continue;
      // TODO(jpienaar): Change to /*x=*/ form once operands are named.
      os << ", /* " << arg->getAsUnquotedString() << " */op->getOperand("
         << nameToOrdinal[arg->getAsUnquotedString()] << ")";
    }
    os << ");\n  }\n};\n";
  }

  // Emit function to add the generated matchers to the pattern list.
  os << "void populateWithGenerated(MLIRContext *context, "
     << "OwningRewritePatternList *patterns) {\n";
  for (unsigned i = 0; i != rewritePatternCount; ++i) {
    os << " patterns->push_back(std::make_unique<" << baseRewriteName << i
       << ">(context));\n";
  }
  os << "}\n";
}

static bool MlirOpTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case GenRewriters:
    emitRewriters(records, os);
    return false;
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &MlirOpTableGenMain);
}
