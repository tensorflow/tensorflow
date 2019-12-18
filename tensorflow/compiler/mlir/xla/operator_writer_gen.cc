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

#include <sstream>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/Support/STLExtras.h"  // TF:local_config_mlir
#include "mlir/TableGen/Operator.h"  // TF:local_config_mlir

using llvm::raw_ostream;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::interleaveComma;
using mlir::tblgen::Attribute;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::NamedTypeConstraint;
using mlir::tblgen::Operator;

static std::string GetDefaultAttrExport(
    const mlir::tblgen::NamedAttribute& named_attr) {
  Attribute attr = named_attr.attr;
  StringRef storage_type = attr.getStorageType();
  // For some attribute types we have a general conversion, so use that.
  if (!attr.isEnumAttr() && (storage_type.endswith("IntegerAttr") ||
                             storage_type.endswith("FloatAttr") ||
                             storage_type.endswith("StringAttr"))) {
    return "Convert" + attr.getReturnType().str();
  }
  return "Convert_" + named_attr.name.str();
}

static void BuildOperator(const Operator& op, raw_ostream* output) {
  auto& os = *output;
  os << "    auto& value_map = *lowering_context.values;\n"
     << "    auto result = xla_op.getResult();\n";

  // Build a conversion for each of the arguments.
  int operand_number = 0;
  for (int index : llvm::seq<int>(0, op.getNumArgs())) {
    auto arg = op.getArg(index);

    // Emit an argument for an operand.
    if (auto* operand_cst = arg.dyn_cast<NamedTypeConstraint*>()) {
      // Handle a non-variadic operand.
      if (!operand_cst->isVariadic()) {
        os << "    auto xla_arg_" << index
           << " = value_map[*xla_op.getODSOperands(" << operand_number++
           << ").begin()];\n";
        continue;
      }

      // Otherwise, this is a varidiac operand list.
      os << "    std::vector<xla::XlaOp> xla_arg_" << index << ";"
         << "    for (auto operand : xla_op.getODSOperands(" << operand_number++
         << "))\n      xla_arg_" << index
         << ".push_back(value_map[operand]);\n";
      continue;
    }

    // Otherwise, this is an attribute.
    auto named_attr = arg.get<NamedAttribute*>();
    os << "    auto xla_arg_" << index << " = "
       << GetDefaultAttrExport(*named_attr) << "(xla_op."
       << op.getArgName(index) << "());\n";
  }

  // Assumes that the client builder method names closely follow the op names
  // in the dialect. For e.g., AddOp -> xla::Add method.
  StringRef op_name = op.getCppClassName();
  os << "    auto xla_result = xla::" << op_name.drop_back(2) << "(";

  // Emit each of the arguments.
  interleaveComma(llvm::seq<int>(0, op.getNumArgs()), os,
                  [&](int i) { os << "Unwrap(xla_arg_" << i << ')'; });
  os << ");\n";

  os << "    value_map[result] = xla_result;\n";
  os << "    return mlir::success();\n";
}

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OperatorWritersMain(raw_ostream& os, RecordKeeper& records) {
  emitSourceFileHeader("MLIR XLA Builders", os);

  // Emit a function to generate an XLA operation for the operations with
  // auto-generated builders.
  os << "mlir::LogicalResult ExportXlaOperator(\n"
        "mlir::Operation* op, OpLoweringContext lowering_context) {\n";

  // Retrieve all the definitions derived from HLO_Op and sort by record name.
  for (const auto* def : records.getAllDerivedDefinitions("HLO_Op")) {
    // Skip operations that have a custom exporter.
    Operator op(def);

    // Cast to the current operation and build the exporter.
    os << "  if (auto xla_op = llvm::dyn_cast<mlir::xla_hlo::"
       << op.getCppClassName() << ">(op)) {\n";
    if (def->getValueAsBit("hasCustomHLOConverter")) {
      os << "    return mlir::xla_hlo::ExportXlaOp(xla_op, "
            "lowering_context);\n";
    } else {
      BuildOperator(op, &os);
    }
    os << "  }\n";
  }

  os << "  return mlir::failure();\n"
        "}\n";
  return false;
}

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OperatorWritersMain);
}
