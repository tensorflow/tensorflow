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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Operator.h"  // from @llvm-project

using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::Operator;

// Helper macro that returns indented os.
#define OUT(X) os.indent((X))

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OpQuantSpecWriter(raw_ostream &os, RecordKeeper &records) {
  llvm::Regex acc_uniform_trait_regex{"AccumulatorUniformScale<([0-9]*),"};
  llvm::Regex coeff_index_trait_regex{"AffineOpCoefficient<(-?[0-9]*),"};
  llvm::Regex fixed_uniform_trait_regex{
      "FixedResultUniformScale<([0-9]+).*(true|false)>"};
  emitSourceFileHeader("Generated Ops Quant Spec Getters", os);

  // Retrieve all the definitions derived from Op definition and sort by record
  // name.
  std::vector<Record *> defs = records.getAllDerivedDefinitions("Op");
  llvm::sort(defs, LessRecord());

  OUT(0) << "static std::unique_ptr<quant::OpQuantSpec> "
            "GetOpQuantSpec(mlir::Operation *op) {\n";
  // TODO(b/176258587): Move to OpTrait if this should be generalized.
  // Add special handling for LSTM.
  OUT(2) << "if (auto lstm_op = llvm::dyn_cast<TFL::LSTMOp>(op)) {\n";
  OUT(4) << "return GetLstmOpQuantSpec<TFL::LSTMOp>(lstm_op);\n";
  OUT(2) << "} else if (auto lstm_op = "
            "llvm::dyn_cast<TFL::UnidirectionalSequenceLSTMOp>(op)) {\n";
  OUT(4) << "return "
            "GetLstmOpQuantSpec<TFL::UnidirectionalSequenceLSTMOp>(lstm_op);\n";
  OUT(2) << "}\n";

  OUT(2) << "auto spec = absl::make_unique<quant::OpQuantSpec>();\n";
  llvm::SmallVector<llvm::StringRef, 3> matches;
  for (auto *def : defs) {
    Operator op(def);
    for (const auto t : op.getTraits()) {
      if (auto opTrait = llvm::dyn_cast<mlir::tblgen::NativeTrait>(&t)) {
        auto trait_str = opTrait->getFullyQualifiedTraitName();
        if (!llvm::StringRef{trait_str}.consume_front(
                "::mlir::OpTrait::quant::"))
          continue;

        OUT(2) << "if (auto tfl = llvm::dyn_cast<" << op.getQualCppClassName()
               << ">(op)) {\n";
        // There is a "FixedResultUniformScale" trait, set the type for result.
        if (fixed_uniform_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "for (int i = 0, e = op->getNumResults(); i != e; ++i)\n";
          OUT(6) << "spec->restricted_output_params[std::make_pair("
                 << matches[1] << ", " << matches[2]
                 << ")].push_back(tfl.::mlir::OpTrait::quant::" << trait_str
                 << "<" << op.getQualCppClassName()
                 << ">::GetResultQuantizedType(i));\n";
          matches.clear();
        }
        // There is a "AccumulatorUniformScale" trait, set the type for bias.
        if (acc_uniform_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "spec->biases_params.emplace(std::make_pair(" << matches[1]
                 << ", std::make_pair(tfl.GetAllNonBiasOperands(),"
                 << "quant::GetUniformQuantizedTypeForBias)));\n";
          matches.clear();
        }
        // There is a "QuantChannelDim" trait, set the quantization dimension.
        if (coeff_index_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "spec->coeff_op_quant_dim[tfl.GetCoefficientOperandIndex()"
                 << "] = tfl.GetQuantizationDim();\n";
          matches.clear();
        }

        OUT(2) << "}\n";
      }
    }
  }
  OUT(2) << "return spec;\n";
  OUT(0) << "}\n";
  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OpQuantSpecWriter);
}
