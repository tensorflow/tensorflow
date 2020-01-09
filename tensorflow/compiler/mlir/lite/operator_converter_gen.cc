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

#include <assert.h>

#include <sstream>
#include <string>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Attribute.h"  // TF:llvm-project

using llvm::DefInit;
using llvm::dyn_cast;
using llvm::formatv;
using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::RecordRecTy;
using llvm::SmallVector;
using llvm::StringInit;
using llvm::StringRef;

// Returns the associated option name for the given op definition.
static inline std::string GetOperatorOptionName(const Record &def) {
  assert(def.getName().startswith("TFL_") && "unexpected op prefix");
  assert(def.getName().endswith("Op") && "unexpected op suffix");

  auto *custom_option = dyn_cast<StringInit>(def.getValueInit("customOption"));
  std::ostringstream oss;
  if (custom_option)
    oss << custom_option->getValue().str();
  else
    oss << def.getName().drop_front(4).drop_back(2).str() << "Options";
  return oss.str();
}

// Returns the builder function name for the given op definition.
static inline std::string GetOperatorBuilderName(StringRef op_name) {
  assert(op_name.startswith("TFL_") && "unexpected op prefix");
  assert(op_name.endswith("Op") && "unexpected op suffix");

  // E.g., AddOp -> CreateAddOperator
  std::ostringstream oss;
  oss << "Create" << op_name.drop_front(4).str() << "erator";
  return oss.str();
}

static void EmitOptionBuilders(const RecordKeeper &record_keeper,
                               const std::vector<Record *> &defs,
                               raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  const auto attr_type = record_keeper.getClass("Attr");
  for (const auto *def : defs) {
    // TFLite ops without options are skipped over.
    if (!def->getValueAsBit("hasOptions")) continue;

    StringRef op_name = def->getName().drop_front(4);  // Strip 'TFL_' prefix
    std::string option_name = GetOperatorOptionName(*def);
    std::string tflite_option_name =
        option_name == "BasicLSTMOptions" ? "LSTMOptions" : option_name;

    os << "flatbuffers::Offset<tflite::" << tflite_option_name << "> Create"
       << option_name << "(mlir::TFL::" << op_name
       << " op, flatbuffers::FlatBufferBuilder *fbb) {\n";

    // Construct all the builder option needed.
    SmallVector<std::string, 8> options;
    // Add options due to attributes (not-derived).
    auto *arg_values = def->getValueAsDag("arguments");
    for (unsigned i = 0, e = arg_values->getNumArgs(); i != e; ++i) {
      auto arg = arg_values->getArg(i);
      DefInit *arg_def = dyn_cast<DefInit>(arg);
      if (!arg_def) continue;
      if (arg_def->getDef()->isSubClassOf(attr_type)) {
        // This binds the name of the attribute in the TD file with the name
        // of the add function of the builder and also with the conversion
        // function to convert from the internal representation to the format
        // expected by the flatbuffer builder. While this constrains the
        // naming of the ops/attributes in the TD file, this also removes the
        // need for specifying indirection. This tool is specific to TFLite
        // conversion generation and so the simplicity was chosen over the
        // flexibility.
        StringRef arg_name = arg_values->getArgNameStr(i);
        os << formatv(
            "  auto {0} = Convert{1}ForOptionWriter(op.{0}(), fbb);\n",
            arg_name, mlir::tblgen::Attribute(arg_def).getAttrDefName());
        options.push_back(arg_name.str());
      }
    }

    // Add options due to derived attributes.
    for (const auto &val : def->getValues()) {
      if (auto *record = dyn_cast<RecordRecTy>(val.getType())) {
        if (record->isSubClassOf(attr_type)) {
          if (record->getClasses().size() != 1) {
            PrintFatalError(
                def->getLoc(),
                "unsupported attribute modelling, only single class expected");
          }
          os << formatv(
              "  auto {0} = Convert{1}ForOptionWriter(op.{0}(), fbb);\n",
              val.getName(), record->getClasses()[0]->getName());
          options.push_back(val.getName());
        }
      }
    }

    os << "  tflite::" << tflite_option_name << "Builder b(*fbb);\n";
    for (const auto &option : options)
      os << formatv("  b.add_{0}(std::move({0}));\n", option);
    os << "  return b.Finish();\n}\n";
  }
}

// For each TFLite op, emits a builder function that packs the TFLite op into
// the corresponding FlatBuffer object.
//
// TODO(hinsu): Revisit if only builtin_options and mutating_variable_inputs
// arguments that depend on op definitions should be auto-generated and then
// operator should be built by the caller because it does not require
// auto-generation.
static void EmitOperatorBuilders(const std::vector<Record *> &defs,
                                 raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  for (const auto *def : defs) {
    StringRef op_name = def->getName().drop_front(4);

    // Signature
    os << "static flatbuffers::Offset<tflite::Operator> "
       << GetOperatorBuilderName(def->getName()) << "(mlir::TFL::" << op_name
       << " tflOp, uint32_t opcode_index, "
       << "const std::vector<int32_t>& operands,"
       << "const std::vector<int32_t>& results,"
       << "flatbuffers::FlatBufferBuilder *fbb) {\n";

    // Inputs & outputs
    os << "  auto inputs = fbb->CreateVector(operands);\n"
          "  auto outputs = fbb->CreateVector(results);\n\n";

    // Build the FlatBuffer operator
    os << "  return tflite::CreateOperator(\n"
          "      *fbb, opcode_index, inputs, outputs,\n";
    if (def->getValueAsBit("hasOptions")) {
      auto option_name = GetOperatorOptionName(*def);
      std::string tflite_option_name =
          option_name == "BasicLSTMOptions" ? "LSTMOptions" : option_name;
      os << "      tflite::BuiltinOptions_" << tflite_option_name << ", "
         << "Create" << option_name << "(tflOp, fbb).Union(),\n";
    } else {
      os << "      tflite::BuiltinOptions_NONE, /*builtin_options=*/0,\n";
    }
    // Only builtin ops' builders are auto-generated. custom_options are only
    // used by custom or flex ops and those ops are handled manually.
    os << "      /*custom_options=*/0, "
          "tflite::CustomOptionsFormat_FLEXBUFFERS,\n"
          "      /*mutating_variable_inputs=*/0);\n"
          "}\n\n";
  }
}

static inline std::string GetOperatorName(const Record &def) {
  auto name = def.getValueAsString("opName");
  // Special case for basic_lstm.
  if (name == "basic_lstm") {
    return "LSTM";
  }
  return name.upper();
}

// Emits a function that returns builtin operator code for each TFLite op.
//
// The signature of the function is:
//
//   llvm::Optional<tflite::BuiltinOperator>
//   mlir::GetBuiltinOpCode(mlir::Operation* op);
//
// TODO(hinsu): Consider converting this to a static constant associative
// container instead of a series of if conditions, if required.
static void EmitGetBuiltinOpCode(const std::vector<Record *> &defs,
                                 raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  os << "llvm::Optional<tflite::BuiltinOperator> "
        "mlir::GetBuiltinOpCode(mlir::Operation* op) {\n";

  for (const auto *def : defs) {
    StringRef op_name = def->getName().drop_front(4);
    os << "  if (isa<mlir::TFL::" << op_name << ">(op))\n"
       << "    return tflite::BuiltinOperator_" << GetOperatorName(*def)
       << ";\n";
  }

  os << "  return llvm::None;\n"
        "}\n";
}

// Emits a builder function that returns the packed FlatBuffer object given
// a general mlir::Operation.
//
// The signature of the function is:
//
//   llvm::Optional<Flatbuffers::Offset<tflite::Operator>>
//   mlir::CreateFlatBufferOperator(
//       mlir::Operation* op,
//       uint32_t opcode_index,
//       const std::vector<int32_t>& operands,
//       const std::vector<int32_t>& results,
//       flatbuffers::FlatBufferBuilder *fbb);
static void EmitBuildOperator(const std::vector<Record *> &defs,
                              raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  // Signature
  os << "llvm::Optional<flatbuffers::Offset<tflite::Operator>>\n"
        "mlir::CreateFlatBufferOperator(mlir::Operation* op, "
        "uint32_t opcode_index, "
        "const std::vector<int32_t>& operands,"
        "const std::vector<int32_t>& results,"
        "flatbuffers::FlatBufferBuilder *fbb) {\n";

  for (const auto *def : defs) {
    StringRef op_name = def->getName().drop_front(4);

    // Try to cast to each op case and call the corresponding op builder
    os << "  if (auto tflOp = llvm::dyn_cast<mlir::TFL::" << op_name
       << ">(op))\n"
       << "    return " << GetOperatorBuilderName(def->getName())
       << "(tflOp, opcode_index, operands, results, fbb);\n";
  }

  os << "  return llvm::None;\n"
        "}\n";
}

// Emit a function that converts a BuiltinOptionsUnion to a vector of attributes
// Signature:
// void mlir::BuiltinOptionsToAttributes(
//     tflite::BuiltinOptionsUnion op_union,
//     mlir::Builder builder,
//     llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes);
static void EmitBuiltinOptionsToAttributes(const RecordKeeper &record_keeper,
                                           const std::vector<Record *> &defs,
                                           raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  // Signature
  os << "void mlir::BuiltinOptionsToAttributes("
        "tflite::BuiltinOptionsUnion op_union, "
        "mlir::Builder builder, "
        "llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes) {\n";

  const auto attr_type = record_keeper.getClass("Attr");
  for (const auto *def : defs) {
    if (!def->getValueAsBit("hasOptions")) continue;
    auto option_name = GetOperatorOptionName(*def);
    // Basic LSTM and LSTM ops share the same option to attribute converter.
    if (option_name == "BasicLSTMOptions") {
      continue;
    }

    os << formatv("  if(const auto *op = op_union.As{0}()) {{\n", option_name);

    // We only care about options that are in arguments
    auto *arg_values = def->getValueAsDag("arguments");
    for (unsigned i = 0, e = arg_values->getNumArgs(); i != e; ++i) {
      auto arg = arg_values->getArg(i);
      DefInit *arg_def = dyn_cast<DefInit>(arg);
      if (!arg_def) continue;
      if (arg_def->getDef()->isSubClassOf(attr_type)) {
        StringRef arg_name = arg_values->getArgNameStr(i);
        StringRef attr_type = mlir::tblgen::Attribute(arg_def).getAttrDefName();
        os << formatv(
            "    attributes.emplace_back(builder.getNamedAttr(\"{0}\","
            " Build{1}(op->{0}, builder)));\n",
            arg_name, attr_type);
      }
    }

    os << "    return;\n";
    os << "  }\n";
  }
  // Fallthrough case is no attributes
  os << "}";
}
// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OperatorWritersMain(raw_ostream &os, RecordKeeper &records) {
  emitSourceFileHeader("MLIR TFLite FlatBuffer Builders", os);

  // Retrieve all the definitions derived from TFL_Op and sort by record name.
  std::vector<Record *> defs = records.getAllDerivedDefinitions("TFL_Op");
  llvm::sort(defs, LessRecord());

  for (const auto *def : defs) {
    // TFLite ops in the .td file are expected to follow the naming convention:
    // TFL_<OpName>Op.
    // The generated TFLite op C++ class should be TFL::<OpName>Op.
    // The generated operator's options should be tflite::<OpName>Options.
    // The option builder should be Create<OpName>Options.
    if (!def->getName().startswith("TFL_"))
      PrintFatalError(def->getLoc(),
                      "unexpected op name format: 'TFL_' prefix missing");
    if (!def->getName().endswith("Op"))
      PrintFatalError(def->getLoc(),
                      "unexpected op name format: 'Op' suffix missing");
  }

  EmitOptionBuilders(records, defs, &os);
  os << "\n\n";
  EmitOperatorBuilders(defs, &os);
  os << "\n\n";
  EmitGetBuiltinOpCode(defs, &os);
  os << "\n\n";
  EmitBuildOperator(defs, &os);
  os << "\n\n";
  EmitBuiltinOptionsToAttributes(records, defs, &os);

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OperatorWritersMain);
}
