/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Operator.h"  // TF:local_config_mlir

using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::Operator;

// Helper macro that returns indented os.
#define OUT(X) os.indent((X))

// Emits TensorFlow derived attribute populator functions for each of the ops.
static void EmitOpAttrPopulators(const std::vector<Operator> &ops,
                                 raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  for (const auto &op : ops) {
    // TODO(hinsu): Introduce a derived attribute property for ops with no
    // type attributes. That way an error can be generated if no derived type
    // attribute or the property is set. This will make sure derived type
    // attributes are not omitted by mistake.

    // Emit function signature.
    auto op_name = op.getCppClassName();
    OUT(0) << "static Status Populate" << op_name
           << "DerivedAttrs(mlir::TF::" << op_name
           << "& op, AttrValueMap *values) {\n";

    for (const auto &named_attr : op.getAttributes()) {
      auto attr_name = named_attr.name;
      const auto &attr = named_attr.attr;
      if (!attr.isDerivedAttr()) continue;
      auto retType = attr.getReturnType();
      if (retType == "Type" || retType == "mlir::OperandElementTypeRange" ||
          retType == "mlir::ResultElementTypeRange") {
        OUT(2) << "TF_RETURN_IF_ERROR(SetAttribute(\"" << attr_name << "\", op."
               << attr_name << "(), values));\n";
      } else {
        PrintFatalError(op.getLoc(),
                        "NYI: Derived attributes other than DerivedTypeAttr");
      }
    }

    OUT(2) << "return Status::OK();\n";
    OUT(0) << "}\n\n";
  }
}

// Emits TensorFlow derived attribute populator function taking an Operation
// as argument.
static void EmitInstAttrPopulator(const std::vector<Operator> &ops,
                                  raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  // Emit function signature.
  OUT(0) << "static Status PopulateDerivedAttrs(mlir::Operation* op, "
            "AttrValueMap* values) {\n";

  for (const auto &op : ops) {
    auto op_name = op.getCppClassName();

    // Emit conditional for the op and then call populator for the op on match.
    OUT(2) << "if (auto tfOp = llvm::dyn_cast<mlir::TF::" << op_name
           << ">(op)) {\n";
    OUT(4) << "TF_RETURN_IF_ERROR(Populate" << op_name
           << "DerivedAttrs(tfOp, values));\n";
    OUT(2) << "}\n";
  }
  OUT(2) << "return Status::OK();\n";
  OUT(0) << "}\n\n";
}

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool DerivedAttrWritersMain(raw_ostream &os, RecordKeeper &records) {
  emitSourceFileHeader("MLIR Derived TensorFlow Attributes Populators", os);

  // Retrieve all the definitions derived from TF_Op and sort by record name.
  std::vector<Record *> defs = records.getAllDerivedDefinitions("TF_Op");
  llvm::sort(defs, LessRecord());

  std::vector<Operator> ops;
  ops.reserve(defs.size());

  // Wrap TensorFlow op definitions into tblgen Operator wrapper and verify
  // them.
  for (const auto *def : defs) {
    ops.emplace_back(Operator(def));

    const Operator &op = ops.back();
    if (op.getDialectName() != "tf")
      PrintFatalError(op.getLoc(),
                      "unexpected op name format: 'TF_' prefix missing");
    if (!op.getCppClassName().endswith("Op"))
      PrintFatalError(op.getLoc(),
                      "unexpected op name format: 'Op' suffix missing");
  }

  EmitOpAttrPopulators(ops, &os);
  EmitInstAttrPopulator(ops, &os);

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &DerivedAttrWritersMain);
}
