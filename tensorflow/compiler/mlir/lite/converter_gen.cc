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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Attribute.h"  // from @llvm-project
#include "mlir/TableGen/Format.h"  // from @llvm-project
#include "mlir/TableGen/Operator.h"  // from @llvm-project
#include "mlir/TableGen/Predicate.h"  // from @llvm-project

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

enum ActionType {
  OpConv,
  RuntimeVerify,
};

// NOLINTNEXTLINE
llvm::cl::opt<ActionType> action(
    llvm::cl::desc("Action to perform:"),
    llvm::cl::values(clEnumValN(OpConv, "gen-operator-converters",
                                "Generate operator converters"),
                     clEnumValN(RuntimeVerify, "gen-runtime-verifiers",
                                "Generate TFLite runtime verifiers")));

// Returns the associated option name for the given op definition.
static inline std::string GetOperatorOptionName(const Record &def) {
  assert(def.getName().starts_with("TFL_") && "unexpected op prefix");
  assert(def.getName().ends_with("Op") && "unexpected op suffix");

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
  assert(op_name.starts_with("TFL_") && "unexpected op prefix");
  assert(op_name.ends_with("Op") && "unexpected op suffix");

  // E.g., AddOp -> CreateAddOperator
  std::ostringstream oss;
  oss << "Create" << op_name.drop_front(4).str() << "erator";
  return oss.str();
}

static inline bool IsLstmOp(const StringRef op_name) {
  return op_name.take_back(6) == "LSTMOp";
}

static int HasOptions(const Record &def) {
  if (def.getValueAsBit("hasOptions")) {
    return 1;
  }
  if (def.getValueAsBit("hasOptions2")) {
    return 2;
  }
  return 0;
}

static void EmitOptionBuilders(const RecordKeeper &record_keeper,
                               const std::vector<const Record *> &defs,
                               raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  const auto attr_type = record_keeper.getClass("Attr");
  for (const auto *def : defs) {
    const int has_options = HasOptions(*def);
    // TFLite ops without options are skipped over.
    if (!has_options) {
      continue;
    }

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
    mlir::tblgen::Operator op(*def);
    for (unsigned i = 0, e = arg_values->getNumArgs(); i != e; ++i) {
      auto arg = arg_values->getArg(i);
      const auto *arg_def = dyn_cast<DefInit>(arg);
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
        // Skip any "intermiadiateXXX" attribute as they are specially handled
        // in the exporter. They are special because though they are attributes
        // in the MLIR they are expressed as tensors in the flatbuffer instead
        // of option.
        if (IsLstmOp(op_name) && arg_name.take_back(12) == "intermediate")
          continue;
        os << formatv(
            "  auto {0} = Convert{1}ForOptionWriter(op.{2}(), fbb);\n",
            arg_name, mlir::tblgen::Attribute(arg_def).getAttrDefName(),
            op.getGetterName(arg_name));
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
              "  auto {0} = Convert{1}ForOptionWriter(op.{2}(), fbb);\n",
              val.getName(), record->getClasses()[0]->getName(),
              op.getGetterName(val.getName()));
          options.push_back(std::string(val.getName()));
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
static void EmitOperatorBuilders(const std::vector<const Record *> &defs,
                                 raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  for (const auto *def : defs) {
    StringRef op_name = def->getName().drop_front(4);

    const bool has_intermediates = op_name.take_back(6) == "LSTMOp";
    // Signature
    os << "static flatbuffers::Offset<tflite::Operator> "
       << GetOperatorBuilderName(def->getName()) << "(mlir::TFL::" << op_name
       << " tflOp, uint32_t opcode_index, "
       << "const std::vector<int32_t>& operands,"
       << "const std::vector<int32_t>& results,"
       << (has_intermediates ? "const std::vector<int32_t>& intermediate_index,"
                             : "")
       << "flatbuffers::FlatBufferBuilder *fbb,"
       << "int debug_metadata_index) {\n";

    // Inputs & outputs
    os << "  auto inputs = fbb->CreateVector(operands);\n"
          "  auto outputs = fbb->CreateVector(results);\n\n";
    // Intermediates for LSTM.
    if (has_intermediates) {
      os << "  auto intermediates = fbb->CreateVector(intermediate_index);\n";
    }

    // Build the FlatBuffer operator
    os << "  return tflite::CreateOperator(\n"
          "      *fbb, opcode_index, inputs, outputs,\n";
    const int has_options = HasOptions(*def);
    if (has_options == 1) {
      auto option_name = GetOperatorOptionName(*def);
      std::string tflite_option_name =
          option_name == "BasicLSTMOptions" ? "LSTMOptions" : option_name;
      os << "      tflite::BuiltinOptions_" << tflite_option_name << ", "
         << "Create" << option_name << "(tflOp, fbb).Union(),\n";
    } else {
      os << "      tflite::BuiltinOptions_NONE, /*builtin_options=*/0,\n";
    }
    // Only built-in ops' builders are auto-generated. custom_options are only
    // used by custom or flex ops and those ops are handled manually.
    os << "      /*custom_options=*/0, "
       << "tflite::CustomOptionsFormat_FLEXBUFFERS,\n"
       << "      /*mutating_variable_inputs=*/0,"
       << (has_intermediates ? "intermediates" : "/*intermediates=*/0");

    if (has_options == 2) {
      os << ",\n"
         << "      /*large_custom_options_offset=*/0,\n"
         << "      /*large_custom_options_size=*/0";
      os << ",\n";
      const std::string option_name = GetOperatorOptionName(*def);
      os << "      tflite::BuiltinOptions2_" << option_name << ", "
         << "Create" << option_name << "(tflOp, fbb).Union()";
    } else {
      os << ",\n"
         << "      /*large_custom_options_offset=*/0,\n"
         << "      /*large_custom_options_size=*/0";
      os << ",\n";
      os << "      tflite::BuiltinOptions2_NONE, /*builtin_options2=*/0";
    }

    os << ",\n"
       << "      /*debug_metadata_index=*/debug_metadata_index";

    os << ");\n}\n\n";
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

// Emits a function that returns built-in operator code for each TFLite op.
//
// The signature of the function is:
//
//   std::optional<tflite::BuiltinOperator>
//   mlir::GetBuiltinOpCode(mlir::Operation* op);
//
// TODO(hinsu): Consider converting this to a static constant associative
// container instead of a series of if conditions, if required.
static void EmitGetBuiltinOpCode(const std::vector<const Record *> &defs,
                                 raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  os << "std::optional<tflite::BuiltinOperator> "
        "mlir::GetBuiltinOpCode(mlir::Operation* op) {\n";

  for (const auto *def : defs) {
    StringRef op_name = def->getName().drop_front(4);
    auto operator_name = GetOperatorName(*def);
    os << "  if (isa<mlir::TFL::" << op_name << ">(op))\n"
       << "    return tflite::BuiltinOperator_" << operator_name << ";\n";
  }

  os << "  return std::nullopt;\n"
        "}\n";
}

// Emits functions that return the min/max operand numbers for a given tflite op
// name.
//
// Signature:
// llvm::MinMax mlir::OperandNumbersMinMax(llvm::StringRef op_name) {
//   if(const auto *op = op_union.AsOptions()) {
//     return {min, max};
//   }
//   ...
//   return {0, 0};
// }
static void EmitOperandNumbers(const RecordKeeper &record_keeper,
                               const std::vector<const Record *> &defs,
                               raw_ostream *ostream) {
  raw_ostream &os = *ostream;
  const auto attr_type = record_keeper.getClass("Attr");
  const auto optional_tensor = record_keeper.getClass("TFL_TensorOfOrNone");
  os << "llvm::MinMax mlir::OperandNumbersMinMax(llvm::StringRef op_name) {\n";
  for (const auto *def : defs) {
    auto op_name = def->getValueAsString("opName");
    int tail_optional_tensor = 0, tensor_number_max = 0;
    auto *arg_values = def->getValueAsDag("arguments");
    for (int i = 0, e = arg_values->getNumArgs(); i < e; ++i) {
      auto arg = arg_values->getArg(i);
      auto *arg_def = dyn_cast<DefInit>(arg);
      if (!arg_def) continue;
      if (!arg_def->getDef()->isSubClassOf(attr_type)) {
        tensor_number_max++;
        if (arg_def->getDef()->isSubClassOf(optional_tensor)) {
          tail_optional_tensor++;
        } else {
          tail_optional_tensor = 0;
        }
      }
    }
    const int tensor_number_min = tensor_number_max - tail_optional_tensor;

    os << formatv("  if (op_name == \"tfl.{0}\") {{\n", op_name)
       << "    return {" << tensor_number_min << ", " << tensor_number_max
       << "};\n  }\n";
  }
  os << "  return {0, 0};\n}\n";
}

// Emits a builder function that returns the packed FlatBuffer object given
// a general mlir::Operation.
//
// The signature of the function is:
//
//   std::optional<Flatbuffers::Offset<tflite::Operator>>
//   mlir::CreateFlatBufferOperator(
//       mlir::Operation* op,
//       uint32_t opcode_index,
//       const std::vector<int32_t>& operands,
//       const std::vector<int32_t>& results,
//       const std::vector<int32_t>& intermediates,
//       flatbuffers::FlatBufferBuilder *fbb,
//       std::optional<int> debug_metadata_index);
static void EmitBuildOperator(const std::vector<const Record *> &defs,
                              raw_ostream *ostream) {
  raw_ostream &os = *ostream;

  // Signature
  os << "std::optional<flatbuffers::Offset<tflite::Operator>>\n"
        "mlir::CreateFlatBufferOperator(mlir::Operation* op, "
        "uint32_t opcode_index, "
        "const std::vector<int32_t>& operands,"
        "const std::vector<int32_t>& results,"
        "const std::vector<int32_t>& intermediates,"
        "flatbuffers::FlatBufferBuilder *fbb,"
        "std::optional<int> debug_metadata_index) {\n";

  for (const auto *def : defs) {
    StringRef op_name = def->getName().drop_front(4);

    // Try to cast to each op case and call the corresponding op builder
    os << "  if (auto tflOp = llvm::dyn_cast<mlir::TFL::" << op_name
       << ">(op))\n"
       << "    return " << GetOperatorBuilderName(def->getName())
       << "(tflOp, opcode_index, operands, results, "
       << (op_name.take_back(6) == "LSTMOp" ? "intermediates, " : "")
       << "fbb, debug_metadata_index.value_or(-1));\n";
  }

  os << "  return std::nullopt;\n"
        "}\n";
}

// Emit a function that converts a BuiltinOptionsUnion to a vector of attributes
// Signature:
// void mlir::BuiltinOptions{id}ToAttributes(
//     tflite::BuiltinOptions{id}Union op_union,
//     mlir::Builder builder,
//     llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes);
//
// where id is an empty string if builtin_options_id is 1, or builtin_options_id
// otherwise.
static void EmitBuiltinOptionsToAttributes(
    const RecordKeeper &record_keeper, const std::vector<const Record *> &defs,
    raw_ostream *ostream, const int builtin_options_id) {
  raw_ostream &os = *ostream;

  const std::string builtin_options_suffix = [&] {
    switch (builtin_options_id) {
      case 1:
        return "";
      case 2:
        return "2";
    }
    return "UnknownId";
  }();

  // Signature
  os << "void mlir::BuiltinOptions" << builtin_options_suffix
     << "ToAttributes("
        "tflite::BuiltinOptions"
     << builtin_options_suffix
     << "Union op_union, "
        "mlir::Builder builder, "
        "llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes) {\n";

  const auto attr_type = record_keeper.getClass("Attr");
  for (const auto *def : defs) {
    const int has_options = HasOptions(*def);
    if (has_options != builtin_options_id) {
      continue;
    }
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
      const auto *arg_def = dyn_cast<DefInit>(arg);
      if (!arg_def) continue;
      if (arg_def->getDef()->isSubClassOf(attr_type)) {
        StringRef arg_name = arg_values->getArgNameStr(i);
        // Already handle this case in flatbuffer_import.cc.
        if ((option_name == "LSTMOptions" ||
             option_name == "UnidirectionalSequenceLSTMOptions") &&
            arg_name.take_back(12) == "intermediate")
          continue;
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
  if (builtin_options_id == 2) {
    os << "  BuiltinOptions2ToAttributesManual(op_union, builder, "
          "attributes);\n";
  }
  // Fallthrough case is no attributes
  os << "}";
}

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OperatorWritersMain(raw_ostream &os, const RecordKeeper &records) {
  emitSourceFileHeader("MLIR TFLite FlatBuffer Builders", os);

  // Retrieve all the definitions derived from TFL_Op and sort by record name.
  std::vector<const Record *> defs = records.getAllDerivedDefinitions("TFL_Op");
  llvm::sort(defs, LessRecord());

  for (const auto *def : defs) {
    // TFLite ops in the .td file are expected to follow the naming convention:
    // TFL_<OpName>Op.
    // The generated TFLite op C++ class should be TFL::<OpName>Op.
    // The generated operator's options should be tflite::<OpName>Options.
    // The option builder should be Create<OpName>Options.
    if (!def->getName().starts_with("TFL_"))
      PrintFatalError(def->getLoc(),
                      "unexpected op name format: 'TFL_' prefix missing");
    if (!def->getName().ends_with("Op"))
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
  EmitBuiltinOptionsToAttributes(records, defs, &os, /*builtin_options_id=*/1);
  os << "\n\n";
  EmitBuiltinOptionsToAttributes(records, defs, &os, /*builtin_options_id=*/2);
  os << "\n\n";
  EmitOperandNumbers(records, defs, &os);

  return false;
}

static void GenOperandResultVerifier(raw_ostream &os,
                                     llvm::ArrayRef<const llvm::Init *> values,
                                     StringRef valueKind) {
  mlir::tblgen::FmtContext fctx;

  bool first = true;
  for (const auto &static_value : llvm::enumerate(values)) {
    auto *definit = llvm::cast<llvm::DefInit>(static_value.value());
    auto *val = definit->getDef()->getValue("tflRuntimeTypePredicate");
    if (!val) continue;

    // Create code block on first type to verify.
    if (first) {
      os << "  {\n";
      os << "    unsigned index = " << static_value.index() << ";\n";
      first = false;
    }

    mlir::tblgen::Pred pred(dyn_cast<llvm::DefInit>(val->getValue()));
    auto desc =
        definit->getDef()->getValueAsString("tflRuntimeTypeDescription");

    // Emit a loop to check all operands.
    os << formatv("    for (Value v : top.getODS{0}{1}s({2})) {{\n",
                  // Capitalize the first letter to match the function name
                  valueKind.substr(0, 1).upper(), valueKind.substr(1),
                  static_value.index());

    os << "      (void)v;\n"
       << "      if (!("
       << tgfmt(pred.getCondition(), &fctx.withSelf("v.getType()")) << ")) {\n"
       << "        if (emit_error_on_verify_fail) {\n"
       << formatv(
              "        return op->emitOpError(\"{0} #\") << index "
              "<< \" must be {1}, but got \" << v.getType();\n",
              valueKind, desc)
       << "        } else {\n"
       << "          return failure();\n"
       << "        }\n"
       << "      }\n"  // if
       << "      ++index;\n"
       << "    }\n";  // for
  }

  // Emit closing brace if needed.
  if (!first) os << "  }\n";
}

// NOLINTNEXTLINE
static bool RuntimeVerifierWriterMain(raw_ostream &os,
                                      const RecordKeeper &records) {
  emitSourceFileHeader("MLIR TFLite Runtime Verifiers", os);

  // Retrieve all the definitions derived from TFL_Op and sort by record name.
  std::vector<const Record *> defs = records.getAllDerivedDefinitions("Op");
  llvm::sort(defs, LessRecord());

  // Iterate through all the ops defined.
  for (const auto *def : defs) {
    mlir::tblgen::Operator op(*def);
    if (!op.getTrait("TflRuntimeVerifyOpInterface::Trait")) continue;

    mlir::tblgen::FmtContext verify_ctx;
    os << "::mlir::LogicalResult " << op.getCppClassName()
       << "::VerifyTflRuntimeConstraints(::mlir::Operation *op, bool "
          "emit_error_on_verify_fail) {\n";
    os << "  auto top = cast<" << op.getCppClassName() << ">(op); (void)top;\n";
    verify_ctx.addSubst("_op", "(*op)");

    for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
      auto &value = op.getOperand(i);
      // Skip from first variadic operands for now. Else getOperand index used
      // below doesn't match.
      if (value.isVariableLength()) break;
      if (!value.name.empty())
        verify_ctx.addSubst(value.name, formatv("op->getOperand({0})", i));
    }
    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
      auto &value = op.getResult(i);
      // Skip from first variadic results for now. Else getResult index used
      // below doesn't match.
      if (value.isVariableLength()) break;
      if (!value.name.empty())
        verify_ctx.addSubst(value.name, formatv("op->getResult({0})", i));
    }
    GenOperandResultVerifier(os, def->getValueAsDag("arguments")->getArgs(),
                             "operand");
    GenOperandResultVerifier(os, def->getValueAsDag("results")->getArgs(),
                             "result");

    for (auto &trait : op.getTraits()) {
      if (!trait.getDef().isSubClassOf("GenInternalOpTrait")) {
        continue;
      }
      if (trait.getDef().getValueAsString("trait") !=
          "::mlir::OpTrait::TFLRuntimeOpTrait") {
        continue;
      }

      auto *val = trait.getDef().getValue("tflRuntimePredicate");
      if (!val) continue;

      auto desc = trait.getDef().getValueAsString("tflRuntimeDescription");

      mlir::tblgen::Pred pred(dyn_cast<llvm::DefInit>(val->getValue()));
      os << tgfmt(
          "  if (!($0)) {\n    "
          "    if (emit_error_on_verify_fail) {\n"
          "      return top.emitOpError(\"failed to verify that $1\");\n"
          "    } else {\n"
          "      return failure();\n  }\n  }\n",
          &verify_ctx, tgfmt(pred.getCondition(), &verify_ctx), desc);
    }
    os << "    if (!emit_error_on_verify_fail) {\n";
    os << "// Ignore transient errors by registering an no-op handler.\n"
          "// Applying legalization patterns will emit unwanted, transient \n"
          "// errors when the replaced TFLite ops do not meet the sanity \n"
          "// checks. \n"
          "// In order to ignore the transient errors, the following lines \n"
          "// override a diagnostic handler with an no-op handler only\n"
          "// while this pass runs.\n"
          "uint64_t current_thread_id = llvm::get_threadid();\n"
          "ScopedDiagnosticHandler scoped_diag_handler(\n"
          "      top.getContext(), [&current_thread_id](Diagnostic&) -> "
          "LogicalResult "
          "{\n"
          "        // Consume only errors that are coming from the same thread "
          "in order not\n"
          "  // to ignore errors from other passes that are running. Things\n"
          "         // running\n"
          "     // in the pass manager can be multi-threaded.\n"
          "        return success(current_thread_id == llvm::get_threadid());\n"
          "   });\n";
    os << "  return top.verifyInvariants();\n";
    os << "    } else {\n";
    os << "  return top.verifyInvariants();\n}\n";
    os << "}\n";
  }

  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (action == ActionType::OpConv)
    return TableGenMain(argv[0], &OperatorWritersMain);
  return TableGenMain(argv[0], &RuntimeVerifierWriterMain);
}
