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

#include <list>
#include <map>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/Operator.h"  // from @llvm-project
#include "tsl/platform/regexp.h"

using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::Operator;

enum class InputDataType { INT8, UINT8, INT16 };

// One InputDataType will be likely mapped to multiple types in near future so
// the two structures are separated.
const std::map<std::string, std::string> &GetTypeToStringRepresentation() {
  static auto *entries = new std::map<std::string, std::string>({
      {"F32", "32-bit float"},
      {"I32", "32-bit signless integer"},
      {"I64", "64-bit signless integer"},
      {"QI16", "QI16 type"},
      {"I8", "8-bit signless integer"},
      {"UI8", "8-bit unsigned integer"},
      {"QI8", "QI8 type"},
      {"QUI8", "QUI8 type"},
      {"TFL_Quint8", "TFLite quint8 type"},
  });

  return *entries;
}

void EmitDynamicRangeOp(std::vector<const Record *> &defs,
                        raw_ostream *ostream) {
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

void EmitSparseOp(std::vector<const Record *> &defs, raw_ostream *ostream) {
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

bool CheckTypeConstraints(const llvm::Init *input_value,
                          std::list<std::string> required_types,
                          bool per_axis) {
  auto *def_init = llvm::cast<llvm::DefInit>(input_value);
  auto *val = def_init->getDef()->getValue("tflRuntimeTypePredicate");

  // For non-per-axis op, no predicate means accepting AnyTensor.
  if (!val) return !per_axis;

  llvm::StringRef supported_types =
      def_init->getDef()->getValueAsString("tflRuntimeTypeDescription");

  for (const std::string &type : required_types) {
    if (!absl::StrContains(supported_types.str(), type)) return false;
  }
  return true;
}

void GenerateStaticQuantOp(std::vector<const Record *> &defs,
                           std::vector<std::string> &result,
                           InputDataType act_type, const bool per_axis,
                           const bool is_toco) {
  std::list<std::string> required_types = {
      GetTypeToStringRepresentation().at("F32")};

  switch (act_type) {
    case InputDataType::INT8: {
      required_types.push_back(GetTypeToStringRepresentation().at("QI8"));
      break;
    }
    case InputDataType::UINT8: {
      required_types.push_back(GetTypeToStringRepresentation().at("QUI8"));
      break;
    }
    case InputDataType::INT16: {
      required_types.push_back(GetTypeToStringRepresentation().at("QI16"));
      break;
    }
    default: {
      // Quantization not applied.
      return;
    }
  }

  // Dimension equals to -1 means per-channel quantization is not supported for
  // the op. Therefore check whether the return value is positive integer as
  // well.
  static const LazyRE2 per_channel_support_regex = {
      "int GetQuantizationDimIndex\\(\\) \\{ return (\\d*); \\}"};

  for (const auto *def : defs) {
    Operator op(def);
    if (!op.getTrait("::mlir::OpTrait::TFL::QuantizableResult")) continue;

    const llvm::DagInit *args_in_dag = def->getValueAsDag("arguments");
    // Assumes argument name is "input" for input activations. Otherwise, assume
    // the first argument is the input activation.
    int input_idx = 0;
    for (int i = 0; i < args_in_dag->getNumArgs(); i++) {
      if (args_in_dag->getArgName(i)->getAsString() == "\"input\"")
        input_idx = i;
    }
    if (CheckTypeConstraints(args_in_dag->getArg(input_idx), required_types,
                             per_axis)) {
      std::string op_name = op.getCppClassName().str();

      // TODO(b/197195711): Please add the additional operations for 16x8 MLIR
      // quantizer. This code is temporary until 16x8 is fully supported in MLIR
      // quantizer.
      if (act_type == InputDataType::INT16) {
        if (is_toco) {
          // Conditions when using TOCO.
          if (absl::StrContains(op_name, "LSTMOp")) continue;
        } else {
          // Conditions when using MLIR.
          if (!(absl::StrContains(op_name, "LSTMOp") ||
                absl::StrContains(op_name, "SoftmaxOp") ||
                absl::StrContains(op_name, "LogisticOp") ||
                absl::StrContains(op_name, "L2NormalizationOp") ||
                absl::StrContains(op_name, "TanhOp"))) {
            continue;
          }
        }
      }

      if (per_axis) {
        std::string op_extra_declaration = op.getExtraClassDeclaration().str();
        bool per_axis_support = RE2::PartialMatch(
            absl::StrReplaceAll(op_extra_declaration, {{"\n", " "}}),
            *per_channel_support_regex);
        if (per_axis_support) result.emplace_back(op_name);
      } else {
        result.emplace_back(op_name);
      }
    }
  }
}

void EmitStaticInt8PerAxisQuantOp(std::vector<const Record *> &defs,
                                  raw_ostream &os) {
  os.indent(0)
      << "const std::set<std::string> &ExportStaticInt8PerAxisSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::INT8, /*per_axis=*/true,
                        /*is_toco=*/false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticInt8PerTensorQuantOp(std::vector<const Record *> &defs,
                                    raw_ostream &os) {
  os.indent(0)
      << "const std::set<std::string> &ExportStaticInt8PerTensorSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::INT8, /*per_axis=*/false,
                        /*is_toco=*/false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticUInt8PerAxisQuantOp(std::vector<const Record *> &defs,
                                   raw_ostream &os) {
  os.indent(0)
      << "const std::set<std::string> &ExportStaticUInt8PerAxisSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::UINT8, /*per_axis=*/true,
                        /*is_toco=*/false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticUInt8PerTensorQuantOp(std::vector<const Record *> &defs,
                                     raw_ostream &os) {
  os.indent(0)
      << "const std::set<std::string> &ExportStaticUInt8PerTensorSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::UINT8, /*per_axis=*/false,
                        /*is_toco=*/false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticQuantOp(std::vector<const Record *> &defs,
                       raw_ostream *ostream) {
  raw_ostream &os = *ostream;
  llvm::sort(defs, LessRecord());

  EmitStaticInt8PerAxisQuantOp(defs, os);
  EmitStaticInt8PerTensorQuantOp(defs, os);
  EmitStaticUInt8PerAxisQuantOp(defs, os);
  EmitStaticUInt8PerTensorQuantOp(defs, os);
}

void EmitStaticQuantWithInt16ActOp(std::vector<const Record *> &defs,
                                   raw_ostream *ostream) {
  raw_ostream &os = *ostream;
  llvm::sort(defs, LessRecord());

  os.indent(0)
      << "const std::set<std::string> &ExportStaticInt8WithInt16ActSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::INT16, /*per_axis=*/false,
                        /*is_toco=*/false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticQuantWithInt16ActTocoOp(std::vector<const Record *> &defs,
                                       raw_ostream *ostream) {
  raw_ostream &os = *ostream;
  llvm::sort(defs, LessRecord());

  os.indent(0) << "const std::set<std::string> "
                  "&ExportStaticInt8WithInt16ActTocoSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::INT16, /*per_axis=*/false,
                        /*is_toco=*/true);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

static bool TFLiteOpCoverageSpecWritersMain(raw_ostream &os,
                                            const RecordKeeper &records) {
  std::vector<const Record *> op_defs =
      records.getAllDerivedDefinitions("TFL_Op");
  EmitStaticQuantOp(op_defs, &os);
  EmitDynamicRangeOp(op_defs, &os);
  EmitStaticQuantWithInt16ActOp(op_defs, &os);
  EmitStaticQuantWithInt16ActTocoOp(op_defs, &os);
  EmitSparseOp(op_defs, &os);
  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &TFLiteOpCoverageSpecWritersMain);
}
