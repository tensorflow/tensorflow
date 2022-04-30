/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <numeric>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/core/lib/io/path.h"

namespace mlir {
namespace TF {
namespace {

static constexpr int kTextFileIndex_WholeLine = -2;
static constexpr int kTextFileIndex_LineNumber = -1;

// InitTextFileToImportPass converts InitializeTableFromTextFileV2Op to the
// corresponding LookupTableImportV2Op if possible.
class InitTextFileToImportPass
    : public InitTextFileToImportPassBase<InitTextFileToImportPass> {
 public:
  InitTextFileToImportPass() {}
  InitTextFileToImportPass(const InitTextFileToImportPass&) {}
  explicit InitTextFileToImportPass(std::string saved_model_dir) {
    saved_model_dir_ = saved_model_dir;
  }

 private:
  void runOnOperation() override;
};

class ConvertInitializeTableFromTextFileV2
    : public OpRewritePattern<InitializeTableFromTextFileV2Op> {
 public:
  explicit ConvertInitializeTableFromTextFileV2(mlir::MLIRContext* context,
                                                StringRef saved_model_dir)
      : OpRewritePattern<InitializeTableFromTextFileV2Op>(context),
        saved_model_dir_(saved_model_dir) {}

  LogicalResult matchAndRewrite(InitializeTableFromTextFileV2Op op,
                                PatternRewriter& rewriter) const override {
    // Now, this pattern matching only supports the following case, which is
    // commonly used among inference use cases:
    //
    // tf.lookup.TextFileInitializer(
    //   "test.txt", tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
    //   tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" ")
    //
    // In the above case, the delimiter will be not used since the key is just a
    // whole line and value is a line number.
    if (op.key_index() != kTextFileIndex_WholeLine ||
        op.value_index() != kTextFileIndex_LineNumber) {
      return failure();
    }

    // Try to find filename from constant op.
    DenseStringElementsAttr filename_attr;
    if (!matchPattern(op.filename().getDefiningOp(),
                      m_Constant(&filename_attr))) {
      return failure();
    }

    if (filename_attr.getRawStringData().size() != 1) {
      return failure();
    }
    std::string filename = filename_attr.getRawStringData()[0].str();

    if (!saved_model_dir_.empty()) {
      filename = tensorflow::io::JoinPath(
          saved_model_dir_.str(),
          tensorflow::io::JoinPath("assets",
                                   tensorflow::io::Basename(filename)));
    }

    // Read the content of the file.
    std::string error_message;
    auto file = openInputFile(filename, &error_message);
    if (!file) {
      return op.emitOpError("failed to open vocabulary file")
             << " (" << filename << "): " << error_message;
    }

    // Splits into lines.
    SmallVector<StringRef, 8> lines;
    file->getBuffer().split(lines, "\n", -1, false);
    // The resize method is used since split operator puts tail value in the end
    // without splitting the leftovers.
    if (op.vocab_size() != -1) lines.resize(op.vocab_size());

    // Map each line to line number, starting from zero.
    SmallVector<int64_t, 8> line_nums;
    line_nums.resize(lines.size());
    std::iota(line_nums.begin(), line_nums.end(), 0);

    // Create constant ops for keys an values.
    Value key_constant_tensor = rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        DenseStringElementsAttr::get(
            RankedTensorType::get(static_cast<int64_t>(lines.size()),
                                  StringType::get(rewriter.getContext())),
            lines));

    Value value_constant_tensor = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI64TensorAttr(line_nums));

    // Replace the given op with LookupTableImportV2Op.
    rewriter.create<LookupTableImportV2Op>(op.getLoc(), op.table_handle(),
                                           key_constant_tensor,
                                           value_constant_tensor);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  StringRef saved_model_dir_;
};

void InitTextFileToImportPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  MLIRContext* context = &getContext();
  func::FuncOp func = getOperation();

  patterns.add<ConvertInitializeTableFromTextFileV2>(
      context, StringRef(saved_model_dir_));
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Replace InitializeTableFromTextFileV2Ops with LookupTableImportV2Ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateInitTextFileToImportPass(
    std::string saved_model_dir) {
  return std::make_unique<InitTextFileToImportPass>(saved_model_dir);
}

}  // namespace TF
}  // namespace mlir
