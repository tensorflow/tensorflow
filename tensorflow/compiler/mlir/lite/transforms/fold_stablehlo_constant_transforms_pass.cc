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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_in_place_transpose.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/resource_constant_fold_utils.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_FOLDSTABLEHLOCONSTANTTRANSFORMSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Upper bound on the rounds of reshape and transpose folding, so chains like
// transpose -> reshape -> transpose fold completely.
constexpr int kMaxFoldRounds = 8;

static ElementsAttr GetElementsAttrFromValue(mlir::Value val) {
  if (!val) return nullptr;
  ElementsAttr attr;
  if (matchPattern(val, m_Constant(&attr))) {
    return attr;
  }
  return nullptr;
}

// Returns the number of bits each element of a resource of `type` takes in a
// blob of `blob_size` bytes, or 0 if the blob does not hold such a resource.
// Sub-byte types are either packed, with the first element in the least
// significant bits of a byte, or stored one element per byte. i1 is not
// supported.
static unsigned GetStorageBitWidth(ShapedType type, size_t blob_size) {
  const unsigned bit_width = type.getElementType().getIntOrFloatBitWidth();
  const int64_t num_elements = type.getNumElements();
  if (bit_width == 1) return 0;
  if (bit_width < 8) {
    if (blob_size == num_elements) return 8;
    if (blob_size == llvm::divideCeil(num_elements * bit_width, 8)) {
      return bit_width;
    }
    return 0;
  }
  const unsigned element_bytes = llvm::divideCeil(bit_width, 8);
  return blob_size == num_elements * element_bytes ? 8 * element_bytes : 0;
}

// Folds stablehlo.reshape on constant inputs. A resource is not copied: the
// result reuses the blob of the input under the new type.
struct FoldStablehloReshapeOpPattern
    : public OpRewritePattern<stablehlo::ReshapeOp> {
  using OpRewritePattern<stablehlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    ElementsAttr input_attr = GetElementsAttrFromValue(op.getOperand());
    if (!input_attr) return failure();

    auto result_type = mlir::cast<mlir::ShapedType>(op.getType());
    if (!result_type.hasStaticShape()) return failure();

    if (auto dense_elements = mlir::dyn_cast<DenseElementsAttr>(input_attr)) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, dense_elements.reshape(result_type));
      return success();
    }
    if (auto dense_resource =
            mlir::dyn_cast<DenseResourceElementsAttr>(input_attr)) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, DenseResourceElementsAttr::get(result_type,
                                             dense_resource.getRawHandle()));
      return success();
    }
    return failure();
  }
};

// Folds stablehlo.transpose on DenseElementsAttr constants, and trivial
// transposes of any constant. Transposes of resources are folded by
// `StablehloTransposeResourceFolder`.
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

    if (input_type.getRank() <= 1) {
      rewriter.replaceOp(op, input);
      return success();
    }

    auto dense_elements = mlir::dyn_cast<DenseElementsAttr>(input_attr);
    if (!dense_elements) return failure();
    if (dense_elements.isSplat()) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, dense_elements.reshape(result_type));
      return success();
    }

    // DenseElementsAttr stores sub-byte types (i2, i4, ...) in one byte per
    // element, and i1 bit packed.
    const unsigned bit_width =
        input_type.getElementType().getIntOrFloatBitWidth();
    if (bit_width == 1) return failure();
    const unsigned storage_bits = 8 * llvm::divideCeil(bit_width, 8);

    llvm::ArrayRef<char> raw_input = dense_elements.getRawData();
    llvm::SmallVector<char> raw_output(raw_input.size());
    if (!TransposeBuffer(raw_input.data(), raw_output.data(),
                         input_type.getShape(), op.getPermutation(),
                         storage_bits)) {
      return failure();
    }
    if (!DenseElementsAttr::isValidRawBuffer(result_type, raw_output)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, DenseElementsAttr::getFromRawBuffer(result_type, raw_output));
    return success();
  }
};

// Describes stablehlo.transpose of a resource constant to
// `FoldResourceOpPattern`, which accounts for blobs that are shared by several
// constants.
struct StablehloTransposeResourceFolder {
  static bool Match(stablehlo::TransposeOp op, Operation*& const_op,
                    DenseResourceElementsAttr& resource_attr) {
    mlir::Value input = op.getOperand();
    resource_attr = mlir::dyn_cast_or_null<DenseResourceElementsAttr>(
        GetElementsAttrFromValue(input));
    if (!resource_attr) return false;
    const_op = input.getDefiningOp();
    if (!const_op) return false;

    auto input_type = mlir::cast<ShapedType>(resource_attr.getType());
    auto result_type = mlir::cast<ShapedType>(op.getType());
    if (!input_type.hasStaticShape() || !result_type.hasStaticShape()) {
      return false;
    }
    if (input_type.getRank() <= 1) return false;
    if (!input_type.getElementType().isIntOrIndexOrFloat()) return false;

    AsmResourceBlob* blob = GetResourceBlob(resource_attr);
    if (!blob || !blob->getData().data()) return false;
    return GetStorageBitWidth(input_type, blob->getData().size()) != 0;
  }

  static std::string GetSignature(stablehlo::TransposeOp op) {
    std::string sig = "transpose";
    for (int64_t dim : op.getPermutation()) absl::StrAppend(&sig, "_", dim);
    return sig;
  }

  static LogicalResult Fold(stablehlo::TransposeOp op,
                            DenseResourceElementsAttr resource_attr,
                            AsmResourceBlob* blob,
                            DenseResourceElementsAttr& new_attr,
                            bool in_place) {
    auto input_type = mlir::cast<ShapedType>(resource_attr.getType());
    ShapedType result_type = GetResultType(op);
    llvm::ArrayRef<char> src = blob->getData();
    const unsigned storage_bits = GetStorageBitWidth(input_type, src.size());
    if (storage_bits == 0) return failure();

    if (in_place) {
      if (!InPlaceTranspose(blob->getMutableData().data(),
                            input_type.getShape(), op.getPermutation(),
                            storage_bits)) {
        return failure();
      }
      new_attr = DenseResourceElementsAttr::get(result_type,
                                                resource_attr.getRawHandle());
      return success();
    }

    auto new_blob = HeapAsmResourceBlob::allocate(src.size(), /*align=*/64,
                                                  /*dataIsMutable=*/true);
    if (!TransposeBuffer(src.data(), new_blob.getMutableData().data(),
                         input_type.getShape(), op.getPermutation(),
                         storage_bits)) {
      return failure();
    }
    std::string new_key = absl::StrCat(
        resource_attr.getRawHandle().getKey().str(), "_", GetSignature(op));
    new_attr = DenseResourceElementsAttr::get(result_type, new_key,
                                              std::move(new_blob));
    return success();
  }

  static ShapedType GetResultType(stablehlo::TransposeOp op) {
    return mlir::cast<ShapedType>(op.getType());
  }

  static Operation* CreateConstant(OpBuilder& builder, Location loc,
                                   ShapedType type, ElementsAttr value) {
    return builder.create<stablehlo::ConstantOp>(loc, value);
  }
};

struct FoldStablehloConstantTransformsPass
    : public impl::FoldStablehloConstantTransformsPassBase<
          FoldStablehloConstantTransformsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FoldStablehloConstantTransformsPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* context = module.getContext();

    RewritePatternSet pattern_set(context);
    pattern_set
        .add<FoldStablehloTransposeOpPattern, FoldStablehloReshapeOpPattern>(
            context);
    FrozenRewritePatternSet patterns(std::move(pattern_set));

    // The patterns fold dense constants and reshapes, which never copy a
    // resource, and erase constants left without users. Transposes of
    // resources are folded across the whole module so a blob shared by
    // several constants is transposed once, in place when possible.
    bool changed = true;
    for (int round = 0; changed && round < kMaxFoldRounds; ++round) {
      bool patterns_changed = false;
      if (failed(applyPatternsGreedily(module, patterns, GreedyRewriteConfig(),
                                       &patterns_changed))) {
        signalPassFailure();
        return;
      }
      bool resources_changed = false;
      if (failed(FoldResourceOpPattern<stablehlo::TransposeOp,
                                       StablehloTransposeResourceFolder>(
              module, &resources_changed))) {
        signalPassFailure();
        return;
      }
      changed = patterns_changed || resources_changed;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateFoldStablehloConstantTransformsPass() {
  return std::make_unique<FoldStablehloConstantTransformsPass>();
}

}  // namespace TFL
}  // namespace mlir
