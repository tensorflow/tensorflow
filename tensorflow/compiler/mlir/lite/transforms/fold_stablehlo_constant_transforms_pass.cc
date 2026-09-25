/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_FOLDSTABLEHLOCONSTANTTRANSFORMSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

static void ComputePermutation(
    llvm::ArrayRef<int64_t> perms, llvm::ArrayRef<int64_t> output_shape,
    const char* raw_input, const int element_byte_size,
    const int64_t current_axis, char*& output_element_addr,
    llvm::MutableArrayRef<uint64_t> current_input_index,
    mlir::ShapedType input_shape_type) {
  const int64_t input_axis = perms[current_axis];
  const bool is_last_axis = current_axis == output_shape.size() - 1;

  for (int i = 0; i < output_shape[current_axis]; ++i) {
    current_input_index[input_axis] = i;
    if (is_last_axis) {
      int64_t input_flat_index = ElementsAttr::getFlattenedIndex(
          input_shape_type, current_input_index);
      const char* input_element_addr =
          raw_input + input_flat_index * element_byte_size;
      std::memcpy(output_element_addr, input_element_addr, element_byte_size);
      output_element_addr += element_byte_size;
    } else {
      ComputePermutation(perms, output_shape, raw_input, element_byte_size,
                         current_axis + 1, output_element_addr,
                         current_input_index, input_shape_type);
    }
  }
}

static ElementsAttr GetElementsAttrFromValue(mlir::Value val) {
  if (!val) return nullptr;
  ElementsAttr attr;
  if (matchPattern(val, m_Constant(&attr))) {
    return attr;
  }
  if (auto cst = val.getDefiningOp<stablehlo::ConstantOp>()) {
    return mlir::dyn_cast_or_null<ElementsAttr>(cst.getValue());
  }
  if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
    return mlir::dyn_cast_or_null<ElementsAttr>(cst.getValue());
  }
  return nullptr;
}

// Pattern to fold stablehlo.reshape on constant inputs.
struct FoldStablehloReshapeOpPattern
    : public OpRewritePattern<stablehlo::ReshapeOp> {
  using OpRewritePattern<stablehlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    mlir::Value input = op.getOperand();
    ElementsAttr input_attr = GetElementsAttrFromValue(input);
    if (!input_attr) return failure();

    auto result_type = mlir::cast<mlir::ShapedType>(op.getType());
    if (!result_type.hasStaticShape()) return failure();

    if (auto dense_elements = mlir::dyn_cast<DenseElementsAttr>(input_attr)) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, dense_elements.reshape(result_type));
      return success();
    } else if (auto dense_resource =
                   mlir::dyn_cast<DenseResourceElementsAttr>(input_attr)) {
      AsmResourceBlob* blob = dense_resource.getRawHandle().getBlob();
      if (!blob && dense_resource.getRawHandle().getResource()) {
        blob = dense_resource.getRawHandle().getResource()->getBlob();
      }
      if (!blob) return failure();

      static std::atomic<int64_t> reshape_counter{0};
      auto key = dense_resource.getRawHandle().getKey();
      std::string new_key =
          llvm::formatv("{0}_reshaped_{1}", key, reshape_counter++);

      auto new_blob = mlir::HeapAsmResourceBlob::allocate(
          blob->getData().size(), /*align=*/64, true);
      std::memcpy(const_cast<char*>(new_blob.getData().data()),
                  blob->getData().data(), blob->getData().size());
      auto new_attr = DenseResourceElementsAttr::get(result_type, new_key,
                                                     std::move(new_blob));
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, new_attr);
      return success();
    }
    return failure();
  }
};

// Pattern to fold stablehlo.transpose on constant inputs.
struct FoldStablehloTransposeOpPattern
    : public OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern<stablehlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter& rewriter) const override {
    mlir::Value input = op.getOperand();
    ElementsAttr input_attr = GetElementsAttrFromValue(input);
    if (!input_attr) return failure();

    auto input_type = mlir::cast<mlir::ShapedType>(input_attr.getType());
    auto result_type = mlir::cast<mlir::ShapedType>(op.getType());
    if (!result_type.hasStaticShape() || !input_type.hasStaticShape()) {
      return failure();
    }

    if (!input_type.getElementType().isIntOrIndexOrFloat()) {
      return failure();
    }

    const int num_dimensions = input_type.getRank();
    if (num_dimensions <= 1) {
      rewriter.replaceOp(op, input);
      return success();
    }

    llvm::ArrayRef<int64_t> perms = op.getPermutation();
    if (static_cast<int>(perms.size()) != num_dimensions) {
      return failure();
    }

    llvm::SmallVector<int64_t> output_shape;
    for (int i = 0; i < num_dimensions; ++i) {
      output_shape.push_back(input_type.getDimSize(perms[i]));
    }

    if (auto dense_elements = mlir::dyn_cast<DenseElementsAttr>(input_attr)) {
      if (dense_elements.isSplat()) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, dense_elements.reshape(result_type));
        return success();
      }

      const int element_byte_size =
          dense_elements.getElementType().getIntOrFloatBitWidth() / 8;
      if (element_byte_size <= 0) return failure();

      const char* raw_input = dense_elements.getRawData().data();
      llvm::SmallVector<uint64_t> current_input_index(num_dimensions, 0);
      llvm::SmallVector<char> raw_output_arr(
          dense_elements.getRawData().begin(),
          dense_elements.getRawData().end());
      char* raw_output = raw_output_arr.data();

      ComputePermutation(perms, output_shape, raw_input, element_byte_size,
                         /*current_axis=*/0, raw_output, current_input_index,
                         input_type);

      if (!DenseElementsAttr::isValidRawBuffer(result_type, raw_output_arr)) {
        return failure();
      }

      auto new_attr =
          DenseElementsAttr::getFromRawBuffer(result_type, raw_output_arr);
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, new_attr);
      return success();
    } else if (auto dense_resource =
                   mlir::dyn_cast<DenseResourceElementsAttr>(input_attr)) {
      AsmResourceBlob* blob = dense_resource.getRawHandle().getBlob();
      if (!blob && dense_resource.getRawHandle().getResource()) {
        blob = dense_resource.getRawHandle().getResource()->getBlob();
      }
      if (!blob) return failure();

      const int element_byte_size =
          input_type.getElementType().getIntOrFloatBitWidth() / 8;
      if (element_byte_size <= 0) return failure();

      int64_t total_elements = result_type.getNumElements();
      size_t total_bytes = total_elements * element_byte_size;

      auto raw_output_blob = mlir::HeapAsmResourceBlob::allocate(
          total_bytes, /*align=*/64, /*dataIsMutable=*/true);
      llvm::ArrayRef<char> data = raw_output_blob.getDataAs<char>();
      llvm::MutableArrayRef<char> raw_output_arr(const_cast<char*>(data.data()),
                                                 data.size());
      char* raw_output = raw_output_arr.data();
      const char* raw_input = blob->getData().data();
      if (!raw_input) return failure();

      llvm::SmallVector<uint64_t> current_input_index(num_dimensions, 0);
      ComputePermutation(perms, output_shape, raw_input, element_byte_size,
                         /*current_axis=*/0, raw_output, current_input_index,
                         input_type);

      static std::atomic<int64_t> transpose_counter{0};
      std::string new_key = llvm::formatv("stablehlo_transpose_fold_result_{0}",
                                          transpose_counter++);
      auto new_attr = DenseResourceElementsAttr::get(
          result_type, new_key, std::move(raw_output_blob));
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, new_attr);
      return success();
    }

    return failure();
  }
};

struct FoldStablehloConstantTransformsPass
    : public impl::FoldStablehloConstantTransformsPassBase<
          FoldStablehloConstantTransformsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FoldStablehloConstantTransformsPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* context = func.getContext();

    RewritePatternSet patterns(context);
    patterns
        .add<FoldStablehloTransposeOpPattern, FoldStablehloReshapeOpPattern>(
            context);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateFoldStablehloConstantTransformsPass() {
  return std::make_unique<FoldStablehloConstantTransformsPass>();
}

}  // namespace TFL
}  // namespace mlir
