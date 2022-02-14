/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/Operator.h"  // from @llvm-project

using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::Operator;

void EmitDynamicRangeOp(const RecordKeeper &record_keeper,
                        std::vector<Record *> &defs, raw_ostream *ostream) {
  std::string dynamic_quant_kernel_support_regex =
      "bool GetDynamicRangeQuantKernelSupport() { return true; }";
  raw_ostream &os = *ostream;
  std::vector<std::string> weight_only;
  llvm::sort(defs, LessRecord());

  os.indent(0) << "const std::set<std::string> &ExportDynamicRangeSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  // Retrieve all the ops that have DynamicRangeQuantizedOpInterface trait.
  for (const auto *def : defs) {
    Operator op(def);
    if (!op.getTrait("DynamicRangeQuantizedOpInterface::Trait")) continue;

    auto op_name = op.getCppClassName();
    auto op_extra_declaration = op.getExtraClassDeclaration().str();

    bool kernel_support = absl::StrContains(
        absl::StrReplaceAll(op_extra_declaration, {{"\n", " "}}),
        dynamic_quant_kernel_support_regex);

    // Classify dynamic range and weight-only fallback
    if (kernel_support) {
      os.indent(6) << "\"" << op_name << "\",\n";
    } else {
      weight_only.push_back(op_name.str());
    }
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";

  os.indent(0)
      << "const std::set<std::string> &ExportDynamicRangeWeightOnlySpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  // Retrieve weight-only fallback.
  for (const auto &op_name : weight_only) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitSparseOp(const RecordKeeper &record_keeper,
                  std::vector<Record *> &defs, raw_ostream *ostream) {
  raw_ostream &os = *ostream;
  llvm::sort(defs, LessRecord());

  os.indent(0) << "const std::set<std::string> &ExportSparsitySpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  // Retrieve all the ops that have SparseOp trait.
  for (const auto *def : defs) {
    Operator op(def);
    if (!op.getTrait("SparseOpInterface::Trait")) {
      continue;
    }
    os.indent(6) << "\"" << op.getCppClassName() << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

static bool TFLiteOpCoverageSpecWritersMain(raw_ostream &os,
                                            RecordKeeper &records) {
  std::vector<Record *> op_defs = records.getAllDerivedDefinitions("TFL_Op");
  EmitDynamicRangeOp(records, op_defs, &os);
  EmitSparseOp(records, op_defs, &os);
  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &TFLiteOpCoverageSpecWritersMain);
}
