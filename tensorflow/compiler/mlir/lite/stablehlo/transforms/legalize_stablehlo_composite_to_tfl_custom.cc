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
    fn.walk([&](Operation* op) {
      // Process only StableHLO composite ops.
      auto composite = llvm::dyn_cast<stablehlo::CompositeOp>(op);
      if (!composite || !IsSupportedComposite(composite)) return;

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
