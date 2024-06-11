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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tpu_embedding_ops_registry.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalize_tf_with_tf2xla_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/tf2xla_rewriter.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {

namespace {

// Returns true if the given type is a ranked tensor type with static or bounded
// dimensions.
bool IsBounded(Type ty) {
  auto ranked_ty = mlir::dyn_cast<RankedTensorType>(ty);
  if (!ranked_ty) return false;

  if (ranked_ty.hasStaticShape()) return true;

  auto encoding =
      mlir::dyn_cast_or_null<TypeExtensionsAttr>(ranked_ty.getEncoding());
  if (!encoding) return false;

  for (int i = 0; i < ranked_ty.getRank(); ++i) {
    if (ranked_ty.isDynamicDim(i) &&
        encoding.getBounds()[i] == ShapedType::kDynamic) {
      return false;
    }
  }
  return true;
}

bool HasSymbolRefAttr(Operation* op) {
  for (const auto& attr : op->getAttrs()) {
    Attribute attr_value = attr.getValue();
    if (mlir::isa<SymbolRefAttr>(attr_value)) {
      return true;
    } else if (auto array_attr = mlir::dyn_cast<ArrayAttr>(attr_value)) {
      if (!array_attr.empty() &&
          mlir::isa<SymbolRefAttr>(*array_attr.begin())) {
        return true;
      }
    }
  }
  return false;
}

class Tf2XlaRewritePattern : public ConversionPattern {
 public:
  explicit Tf2XlaRewritePattern(MLIRContext* ctx, TypeConverter& converter,
                                const std::string& device_type,
                                bool prefer_tf2xla)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        device_type_(device_type),
        prefer_tf2xla_(prefer_tf2xla) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // This pattern is a conversion pattern because we want to specify a type
    // converter. However, this pattern still uses the original op's operands
    // while creating the ops so make sure there aren't any type changes between
    // the original op operands and the operands during the conversion.
    for (auto&& [old_val, new_val] : llvm::zip(op->getOperands(), operands)) {
      if (old_val.getType() != new_val.getType()) return failure();
    }

    auto abstractOp = op->getRegisteredInfo();
    if (!abstractOp) return failure();

    if (!(IsOpAllowedTf2xlaFallback(abstractOp->getTypeID()) ||
          (prefer_tf2xla_ &&
           IsOpAllowedTf2xlaPreferred(abstractOp->getTypeID())))) {
      return failure();
    }

    return Tf2XlaRewriter::RewriteOp(op, rewriter, device_type_);
  }

 private:
  std::string device_type_;
  bool prefer_tf2xla_;
  bool use_tf2xla_hlo_importer_;
};

bool ShouldRefineTypeTo(Type original_ty, Type updated_ty) {
  auto updated = mlir::dyn_cast<ShapedType>(updated_ty);
  auto original = mlir::dyn_cast<ShapedType>(original_ty);

  // Both types must be shaped types.
  if (!original || !updated) return false;

  // Element types must match.
  if (original.getElementType() != updated.getElementType()) return false;

  // If the updated type doesn't have a rank, then it can't be a more refined
  // type.
  if (!updated.hasRank()) return false;

  // If the original type doesn't have a rank, then refine as the updated type
  // has a rank.
  if (!original.hasRank()) return true;

  // Both types must have the same rank.
  if (original.getRank() != updated.getRank()) return false;

  // Refine if the updated type is bounded.
  return IsBounded(updated);
}

// Propagates more refined type by cloning op using the new operands. This
// allows all rewrite patterns that requires refined types to work without
// requiring a rewrite to the conversion pattern. Declarative rewrite pattern
// (DRR) doesn't even support conversion patterns with TableGen.
class TypePropagator : public ConversionPattern {
 public:
  explicit TypePropagator(MLIRContext* ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // This could be generalized to other ops as needs arise. We could even
    // remove this restriction altogether except for the terminators that
    // require function signature change and shouldn't be
    if (op->getName().getDialectNamespace() !=
        TF::TensorFlowDialect::getDialectNamespace())
      return failure();

    // Refining types may have implications to the attached regions or symbol
    // references so do not update such ops.
    if (!op->getRegions().empty() || HasSymbolRefAttr(op)) return failure();

    IRMapping mapper;
    bool has_type_change = false;
    for (auto [original, updated] : llvm::zip(op->getOperands(), operands)) {
      Type original_ty = original.getType();
      Type updated_ty = updated.getType();
      if (original_ty != updated_ty) has_type_change = true;

      if (!ShouldRefineTypeTo(original_ty, updated_ty)) return failure();
      mapper.map(original, updated);
    }
    if (!has_type_change) return failure();

    Operation* cloned_op = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, cloned_op->getResults());
    return success();
  }
};

}  // end namespace

Tf2XlaTypeConverter::Tf2XlaTypeConverter() {
  // Currently, we don't do any type conversions. Any TensorFlow op with a type
  // that is not supported in MHLO will fail conversion. Quantized types are
  // going to handled separately so we don't need to handle those.
  addConversion([](Type ty) { return ty; });

  // This materialization is helpful in cases where we have more refined types
  // after conversion to mhlo compared to the original type in TF. For example,
  // a TF op with result type tensor<*xf32> will have a bounded type after
  // fallback legalization.
  auto cast_value = [&](OpBuilder& builder, Type result_type, ValueRange inputs,
                        Location loc) -> Value {
    return builder.create<mlir::tensor::CastOp>(loc, result_type,
                                                inputs.front());
  };
  addSourceMaterialization(cast_value);
}

void PopulateLegalizeTfWithTf2XlaPatterns(llvm::StringRef device_type,
                                          RewritePatternSet& patterns,
                                          MLIRContext* ctx,
                                          Tf2XlaTypeConverter& converter,
                                          bool prefer_tf2xla) {
  patterns.add<TypePropagator>(ctx);
  patterns.add<Tf2XlaRewritePattern>(ctx, converter, device_type.str(),
                                     prefer_tf2xla);
}

}  // end namespace mhlo
}  // end namespace mlir
