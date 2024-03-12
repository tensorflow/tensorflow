/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/op_stat_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_util.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_traits.h"

namespace mlir {
namespace odml {

// Defines a pass class for print the summary of non-converted ops similar to
// mlir::PrintOpStatsPass, but this pass will show the simpler information.
namespace {
class PrintOpStatsPass : public PassWrapper<PrintOpStatsPass, OperationPass<>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpStatsPass)

  explicit PrintOpStatsPass(raw_ostream *os = &llvm::errs(),
                            std::vector<std::string> accepted_dialects = {})
      : accepted_dialects_(accepted_dialects), os_(os), total_ops_(0) {}

  // Prints the resultant operation statistics pos_t iterating over the module.
  void runOnOperation() override;

  // Prints summary of op stats.
  void PrintSummary();

  // Keeps track of dtype counts per op.
  void CountOp(DenseMap<StringRef, llvm::StringMap<int64_t>> &op_count_map,
               StringRef op_name, StringRef dtype);

 private:
  llvm::StringMap<int64_t> op_with_dialect_count_;
  DenseMap<StringRef, llvm::StringMap<int64_t>> op_dtype_count_;
  llvm::StringMap<int64_t> dialect_count_;
  llvm::StringMap<StringRef> dialect_name_of_;
  llvm::StringMap<StringRef> op_name_of_;
  std::vector<std::string> accepted_dialects_;
  std::vector<std::string> optional_accepted_dialects_;
  raw_ostream *os_;
  int total_ops_;
};
}  // namespace

void PrintOpStatsPass::runOnOperation() {
  op_with_dialect_count_.clear();
  dialect_count_.clear();
  dialect_name_of_.clear();

  // Compute the operation statistics for the currently visited operation.
  total_ops_ = 0;

  getOperation()->walk([&](Operation *op) {
    auto op_with_dialect_name = op->getName().getStringRef();
    auto op_name = op->getName().stripDialect();
    auto dialect_name = op->getDialect()->getNamespace();

    if (op->getNumResults() > 0 &&
        isa<ShapedType>(op->getResult(0).getType())) {
      // Use rhs operand to detect types for dynamic range quantizable ops.
      Value value_for_deducing_op_type =
          (dyn_cast_or_null<DynamicRangeQuantizedOpInterface>(op))
              ? op->getOperand(1)
              : op->getResult(0);
      ShapedType value_shaped_type =
          value_for_deducing_op_type.getType().dyn_cast_or_null<ShapedType>();
      if (value_shaped_type != nullptr) {
        auto operand_or_result = value_shaped_type.getElementType();
        std::string dtype;

        TypeSwitch<Type>(operand_or_result)
            .Case<IntegerType>([&](Type) {
              dtype =
                  absl::StrCat("i", operand_or_result.getIntOrFloatBitWidth());
            })
            .Case<FloatType>([&](Type) {
              dtype =
                  absl::StrCat("f", operand_or_result.getIntOrFloatBitWidth());
            })
            .Case<UniformQuantizedType>([&](Type) {
              auto uniform_quantized_dtype =
                  operand_or_result.dyn_cast_or_null<UniformQuantizedType>()
                      .getStorageType();
              dtype = absl::StrCat(
                  "uq_", uniform_quantized_dtype.getIntOrFloatBitWidth());
            })
            .Case<quant::UniformQuantizedPerAxisType>([&](Type) {
              auto uniform_quantized_dtype =
                  operand_or_result
                      .dyn_cast_or_null<quant::UniformQuantizedPerAxisType>()
                      .getStorageType();
              dtype = absl::StrCat(
                  "uq_", uniform_quantized_dtype.getIntOrFloatBitWidth());
            })
            .Default([&](Type) { LOG(ERROR) << "Unsupported data type."; });
        CountOp(op_dtype_count_, op_with_dialect_name, dtype);
      }
    }
    ++op_with_dialect_count_[op_with_dialect_name];
    ++dialect_count_[dialect_name];
    dialect_name_of_[op_with_dialect_name] = dialect_name;
    op_name_of_[op_with_dialect_name] = op_name;
    ++total_ops_;
  });
  PrintSummary();
}

void PrintOpStatsPass::CountOp(
    DenseMap<StringRef, llvm::StringMap<int64_t>> &op_count_map,
    StringRef op_name, StringRef dtype) {
  auto &op_counts = op_count_map[op_name];
  if (auto it = op_counts.find(dtype); it != op_counts.end()) {
    it->second++;
  } else {
    op_counts[dtype] = 1;
  }
}

void PrintOpStatsPass::PrintSummary() {
  *os_ << "Summary on the non-converted ops:\n";
  *os_ << "---------------------------------\n";
  SmallVector<StringRef, 64> sorted_op(op_with_dialect_count_.keys());
  SmallVector<StringRef, 64> sorted_dialect(dialect_count_.keys());
  llvm::sort(sorted_op);
  llvm::sort(sorted_dialect);

  *os_ << " * Accepted dialects: ";
  int num_dialect = 0;
  // Print the accepted dialect list.
  for (const auto &dialect_name : accepted_dialects_) {
    *os_ << dialect_name;
    if (++num_dialect < accepted_dialects_.size()) {
      *os_ << ", ";
    }
  }

  int converted_ops = 0;
  for (const auto &dialect_name : accepted_dialects_) {
    converted_ops += dialect_count_[dialect_name];
  }
  int non_converted_ops = total_ops_ - converted_ops;
  float percentage =
      (static_cast<float>(non_converted_ops) / static_cast<float>(total_ops_)) *
      100.0;
  // Non-Converted Ops: 25, Total Ops 100, % non-converted = 25%
  *os_ << absl::StrFormat(
      "\n * Non-Converted Ops: %d, Total Ops %d, %% non-converted = %.2f %%",
      non_converted_ops, total_ops_, percentage);

  *os_ << "\n * ";
  int num_unaccepted = sorted_dialect.size() - accepted_dialects_.size();
  num_dialect = 0;
  // Print the number of unconverted ops in the non-accepted dialects.
  for (const auto &dialect_name : sorted_dialect) {
    if (!IsAcceptedDialect(dialect_name, accepted_dialects_)) {
      *os_ << absl::StrFormat("%d %s ops", dialect_count_[dialect_name],
                              absl::AsciiStrToUpper(dialect_name));
      if (++num_dialect < num_unaccepted) {
        *os_ << ", ";
      }
    }
  }

  *os_ << "\n\n";

  for (const auto &op_with_dialect_name : sorted_op) {
    if (!IsAcceptedOp(dialect_name_of_[op_with_dialect_name],
                      op_name_of_[op_with_dialect_name], accepted_dialects_)) {
      *os_ << absl::StrFormat("- %s: %4d occurrences", op_with_dialect_name,
                              op_with_dialect_count_[op_with_dialect_name]);
    }

    auto op_count_map = op_dtype_count_[op_with_dialect_name];
    if (!op_count_map.empty()) {
      std::string delim;
      *os_ << "  (";
      for (const auto &[dtype, cnt] : op_count_map) {
        *os_ << delim << absl::StrFormat("%s: %d", dtype, cnt);
        delim = ", ";
      }
      *os_ << ")";
    }
    *os_ << "\n";
  }
}

}  // namespace odml
}  // namespace mlir

std::unique_ptr<mlir::Pass> mlir::odml::createPrintOpStatsPass(
    std::vector<std::string> accepted_dialects) {
  return std::make_unique<mlir::odml::PrintOpStatsPass>(&llvm::errs(),
                                                        accepted_dialects);
}
