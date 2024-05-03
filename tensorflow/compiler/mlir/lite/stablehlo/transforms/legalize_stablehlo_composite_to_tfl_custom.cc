/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"

#define DEBUG_TYPE "composite-to-custom"

namespace mlir {
namespace odml {

#define GEN_PASS_DEF_LEGALIZECOMPOSITETOCUSTOMOPPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

namespace {
bool IsSupportedComposite(::mlir::stablehlo::CompositeOp op) {
  // List of supported composites to represent using CustomOp.
  return llvm::is_contained(
      {"odml.update_kv_cache", "odml.scaled_dot_product_attention"},
      op.getName());
}

bool IsKVCacheCompositeOp(::mlir::stablehlo::CompositeOp op) {
  return op.getName() == "odml.update_kv_cache";
}

TFL::ConstBytesAttr CustomOption(OpBuilder* builder,
                                 const std::string& content) {
  return TFL::ConstBytesAttr::get(builder->getContext(),
                                  StringRef(content.data(), content.size()));
}

LogicalResult BuildOption(flexbuffers::Builder* fbb, Operation* op,
                          NamedAttribute pair) {
  const char* key = pair.getName().data();
  const auto attr = pair.getValue();

  if (attr.isa<::mlir::IntegerAttr>()) {
    fbb->Int(key, attr.dyn_cast<mlir::IntegerAttr>().getInt());
    return success();
  }

  if (attr.isa<::mlir::FloatAttr>()) {
    fbb->Double(key, attr.dyn_cast<mlir::FloatAttr>().getValueAsDouble());
    return success();
  }

  return op->emitWarning("serialization not supported for : ") << key;
}

TFL::CustomOp BuildCustomOp(stablehlo::CompositeOp composite,
                            const std::string& custom_option_buffer) {
  OpBuilder builder(composite->getContext());
  builder.setInsertionPoint(composite);
  if (IsKVCacheCompositeOp(composite)) {
    return builder.create<TFL::CustomOp>(
        composite->getLoc(), composite->getResultTypes(),
        composite->getOperands().slice(2, 3), composite.getName(),
        CustomOption(&builder, custom_option_buffer));
  }
  return builder.create<TFL::CustomOp>(
      composite->getLoc(), composite->getResultTypes(),
      composite->getOperands(), composite.getName(),
      CustomOption(&builder, custom_option_buffer));
}

}  // namespace

// Legalize stablehlo::CompositeOp to TFL::CustomOp for runtime-supported
// composites. See `IsSupportedComposite` for list of supported ops.
//
// Example:
//   %0 = stablehlo.composite "odml.some_op" <args> {
//      composite_attrs = {<attrs>},
//      version = 0 : i32
//   }
//   ==>
//   %0 = tfl.custom(<args>) {
//     custom_code = "odml.some_op",
//     custom_option = #tfl<const_bytes : "flexbuffer_serialized_attrs">
//   }
struct LegalizeCompositeToCustomOpPass
    : public impl::LegalizeCompositeToCustomOpPassBase<
          LegalizeCompositeToCustomOpPass> {
  using LegalizeCompositeToCustomOpPassBase::
      LegalizeCompositeToCustomOpPassBase;

  void runOnOperation() override {
    func::FuncOp fn = getOperation();

    int num_layers = 0, current_layer_index = 0;
    // First walk the function to count number of KV Caches.
    fn.walk([&](Operation* op) {
      auto composite = llvm::dyn_cast<stablehlo::CompositeOp>(op);
      if (!composite || !IsKVCacheCompositeOp(composite)) return;
      num_layers++;
    });

    fn.walk([&](Operation* op) {
      // Process only StableHLO composite ops.
      auto composite = llvm::dyn_cast<stablehlo::CompositeOp>(op);
      if (!composite || !IsSupportedComposite(composite)) return;

      if (IsKVCacheCompositeOp(composite)) {
        auto comp_attr = composite.getCompositeAttributes();
        mlir::Builder builder(composite->getContext());

        // num_layers Composite Attribute.
        mlir::StringAttr num_layers_str = builder.getStringAttr("num_layers");
        NamedAttribute num_layers_attr(
            num_layers_str,
            IntegerAttr::get(IntegerType::get(fn.getContext(), /*width=*/32),
                             num_layers));

        // current_layer_index Composite Attribute.
        mlir::StringAttr current_layer_str =
            builder.getStringAttr("layer_index");
        NamedAttribute current_layer_attr(
            current_layer_str,
            IntegerAttr::get(IntegerType::get(fn.getContext(), /*width=*/32),
                             current_layer_index++));

        // Build a new CompositeAttributes attr, add in the above,
        // and set for the op.
        mlir::NamedAttrList attributes(comp_attr);
        attributes.append(num_layers_attr);
        attributes.append(current_layer_attr);
        comp_attr = attributes.getDictionary(builder.getContext());
        composite.setCompositeAttributesAttr(comp_attr);
      }

      // Build flexbuffer options.
      std::string custom_option_buffer;
      auto fbb = std::make_unique<flexbuffers::Builder>();
      size_t map_start = fbb->StartMap();
      for (const NamedAttribute& pair : composite.getCompositeAttributes()) {
        // Allows skipping unsupported attributes, will warn.
        (void)BuildOption(fbb.get(), op, pair);
      }
      fbb->EndMap(map_start);
      fbb->Finish();
      custom_option_buffer.assign(fbb->GetBuffer().begin(),
                                  fbb->GetBuffer().end());

      // Build TFL custom op, replace composite with custom op.
      TFL::CustomOp tfl_custom_op =
          BuildCustomOp(composite, custom_option_buffer);
      composite->replaceAllUsesWith(tfl_custom_op);
      composite->erase();
    });
  }
};

static PassRegistration<LegalizeCompositeToCustomOpPass> pass;

}  // namespace odml
}  // namespace mlir
