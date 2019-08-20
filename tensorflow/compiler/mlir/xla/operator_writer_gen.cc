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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Operator.h"  // TF:local_config_mlir

using llvm::dyn_cast;
using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::Operator;

// Returns the builder function name for the given op definition.
// E.g., AddOp -> CreateAddOp
static inline std::string GetOperatorBuilderName(StringRef op_name) {
  return "Create" + op_name.str();
}

static std::string GetConversionFunction(
    mlir::tblgen::NamedAttribute named_attr) {
  auto storage_type = named_attr.attr.getStorageType();
  // For some attribute types we have a general conversion, so use that.
  if (storage_type.endswith("IntegerAttr") ||
      storage_type.endswith("FloatAttr")) {
    return "Convert" + named_attr.attr.getReturnType().str();
  }
  return "Convert_" + named_attr.name.str();
}

using ArgumentName = std::string;
using ArgumentDeclaration = std::string;
using Argument = std::pair<ArgumentName, ArgumentDeclaration>;
using ArgumentList = std::vector<Argument>;

static std::string BuildOperator(const Operator& op) {
  std::stringstream os;
  StringRef op_name = op.getCppClassName();
  std::string xla_op_name = op_name.drop_back(2).str();

  // Signature.
  os << "static xla::XlaOp " << GetOperatorBuilderName(op_name)
     << "(mlir::XLA::" << op_name.str() << " xla_op, "
     << "llvm::DenseMap<mlir::Value*, xla::XlaOp>* "
        "value_lowering) {\n";

  os << "  auto& value_map = *value_lowering;\n"
     << "  auto result = xla_op.getResult();\n";

  // Invoke the conversion function for each attribute.
  for (const auto& named_attr : op.getAttributes()) {
    os << "  auto " << named_attr.name.str() << " = "
       << GetConversionFunction(named_attr) << "("
       << "xla_op." << named_attr.name.str() << "());\n";
  }

  // Assumes that the client builder method names closely follow the op names
  // in the dialect. For e.g., AddOp -> xla::Add method.
  os << "  auto xla_result = xla::" << xla_op_name << "(";

  int num_operands = op.getNumOperands();
  if (num_operands == 1) {
    os << "value_map[xla_op.getOperand()]";
  } else {
    for (auto i = 0; i < num_operands; i++) {
      os << "value_map[xla_op.getOperand(" << i << ")]";
      if (i != num_operands - 1) {
        os << ", ";
      }
    }
  }

  for (const auto& named_attr : op.getAttributes()) {
    os << ", Unwrap(" << named_attr.name.str() << ")";
  }

  os << ");\n";

  os << "  value_map[result] = xla_result;\n";
  os << "  return xla_result;\n";
  os << "}\n\n";
  return os.str();
}

// For each XLA op, emits a builder function that constructs the XLA op using
// the HLO client builder.
static void EmitOperatorBuilders(const RecordKeeper& record_keeper,
                                 const std::vector<Record*>& defs,
                                 raw_ostream* ostream) {
  raw_ostream& os = *ostream;

  for (const auto* def : defs) {
    // Skip operations that have a custom converter.
    if (def->getValueAsBit("hasCustomHLOConverter")) continue;

    Operator op(def);
    os << BuildOperator(op);
  }
}

// Emits a builder function that returns the XlaOp object given a
// mlir::Operation.
//
// The signature of the function is:
//
//   llvm::Optional<xla::XlaOp>
//   mlir::CreateXlaOperator(
//       mlir::Operation* op,
//       llvm::DenseMap<mlir::Value*, xla::XlaOp>
//       *value_lowering);
static void EmitBuilder(const std::vector<Record*>& defs,
                        raw_ostream* ostream) {
  raw_ostream& os = *ostream;

  // Signature
  os << "llvm::Optional<xla::XlaOp>\n"
        "mlir::CreateXlaOperator(mlir::Operation* op, "
        "llvm::DenseMap<mlir::Value*, xla::XlaOp> "
        "*value_lowering) {\n";

  for (const auto* def : defs) {
    // Skip operations that have a custom converter.
    if (def->getValueAsBit("hasCustomHLOConverter")) continue;

    StringRef op_name = def->getName().drop_front(4);

    // Try to cast to each op and call the corresponding op builder.
    os << "  if (auto xla_op = llvm::dyn_cast<mlir::XLA::" << op_name
       << ">(op))\n     return " << GetOperatorBuilderName(op_name)
       << "(xla_op, value_lowering);\n";
  }

  os << "  return llvm::None;\n"
        "}\n";
}

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OperatorWritersMain(raw_ostream& os, RecordKeeper& records) {
  emitSourceFileHeader("MLIR XLA Builders", os);

  // Retrieve all the definitions derived from XLA_Op and sort by record name.
  std::vector<Record*> defs = records.getAllDerivedDefinitions("XLA_Op");
  llvm::sort(defs, LessRecord());

  for (const auto* def : defs) {
    // XLA ops in the .td file are expected to follow the naming convention:
    // XLA_<OpName>Op.
    // The generated XLA op C++ class should be XLA::<OpName>Op.
    if (!def->getName().startswith("XLA_"))
      PrintFatalError(def->getLoc(),
                      "unexpected op name format: 'XLA_' prefix missing");
    if (!def->getName().endswith("Op"))
      PrintFatalError(def->getLoc(),
                      "unexpected op name format: 'Op' suffix missing");
  }

  EmitOperatorBuilders(records, defs, &os);
  os << "\n\n";
  EmitBuilder(defs, &os);

  return false;
}

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OperatorWritersMain);
}
