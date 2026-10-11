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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // build_cleaner: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // build_cleaner: keep
#include "stablehlo/dialect/VhloTypes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace mlir::odml {

#define GEN_PASS_DEF_LEGALIZEVHLOQUANTCUSTOMCALLSPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

namespace {

class VhloToStablehloTypeConverter : public vhlo::VhloTypeConverter {
 public:
  VhloToStablehloTypeConverter()
      : vhlo::VhloTypeConverter(/*allowOtherDialects=*/true) {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](vhlo::TokenV1Type token) -> Type {
      return stablehlo::TokenType::get(token.getContext());
    });
    addVhloToBuiltinConversions();
  }

  Attribute convertEncoding(Attribute attr) const final {
    if (auto vhloAttr =
            mlir::dyn_cast_or_null<vhlo::TypeExtensionsV1Attr>(attr)) {
      return stablehlo::TypeExtensionsAttr::get(vhloAttr.getContext(),
                                                vhloAttr.getBounds());
    }
    return attr;
  }
};

Attribute ConvertVhloAttrToBuiltin(Attribute attr) {
  if (!attr) return {};
  if (auto vhlo_str = mlir::dyn_cast<vhlo::StringV1Attr>(attr)) {
    return StringAttr::get(attr.getContext(), vhlo_str.getValue());
  }
  if (auto vhlo_bool = mlir::dyn_cast<vhlo::BooleanV1Attr>(attr)) {
    return BoolAttr::get(attr.getContext(), vhlo_bool.getValue());
  }
  if (auto vhlo_int = mlir::dyn_cast<vhlo::IntegerV1Attr>(attr)) {
    VhloToStablehloTypeConverter type_converter;
    Type type = type_converter.convertType(vhlo_int.getType());
    if (!type) type = vhlo_int.getType();
    return IntegerAttr::get(type, vhlo_int.getValue());
  }
  if (auto vhlo_float = mlir::dyn_cast<vhlo::FloatV1Attr>(attr)) {
    VhloToStablehloTypeConverter type_converter;
    Type type = type_converter.convertType(vhlo_float.getType());
    if (!type) type = vhlo_float.getType();
    return FloatAttr::get(type, vhlo_float.getValue());
  }
  if (auto vhlo_tensor = mlir::dyn_cast<vhlo::TensorV1Attr>(attr)) {
    return vhlo_tensor.getData();
  }
  return attr;
}

struct LegalizeVhloQuantCustomCallPattern : public RewritePattern {
  explicit LegalizeVhloQuantCustomCallPattern(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    StringRef op_name = op->getName().getStringRef();
    if (op_name != "vhlo.custom_call_v1" && op_name != "vhlo.custom_call") {
      return failure();
    }

    Attribute raw_target = op->getAttr("call_target_name");
    if (!raw_target) return failure();

    StringRef target_name;
    if (auto attr = mlir::dyn_cast<vhlo::StringV1Attr>(raw_target)) {
      target_name = attr.getValue();
    } else if (auto attr = mlir::dyn_cast<StringAttr>(raw_target)) {
      target_name = attr.getValue();
    }

    if (target_name != "quant.dequantize" && target_name != "quant.quantize" &&
        target_name != "quant.fake_quant") {
      return failure();
    }

    VhloToStablehloTypeConverter type_converter;

    SmallVector<Type> result_types;
    for (Type t : op->getResultTypes()) {
      Type conv = type_converter.convertType(t);
      result_types.push_back(conv ? conv : t);
    }

    SmallVector<Value> operands;
    for (Value val : op->getOperands()) {
      Type conv = type_converter.convertType(val.getType());
      if (conv && conv != val.getType()) {
        val =
            rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), conv, val)
                .getResult(0);
      }
      operands.push_back(val);
    }

    SmallVector<NamedAttribute> new_attrs;
    new_attrs.push_back(rewriter.getNamedAttr(
        "call_target_name", rewriter.getStringAttr(target_name)));

    static const char* const kIntrinsicAttrs[] = {"api_version",
                                                  "backend_config",
                                                  "call_target_name",
                                                  "called_computations",
                                                  "has_side_effect",
                                                  "operand_layouts",
                                                  "output_operand_aliases",
                                                  "result_layouts",
                                                  "result_tilings"};

    for (NamedAttribute attr : op->getAttrs()) {
      StringRef name = attr.getName().strref();
      bool is_intrinsic = false;
      for (const char* kAttr : kIntrinsicAttrs) {
        if (name == kAttr) {
          is_intrinsic = true;
          break;
        }
      }
      if (is_intrinsic) continue;

      Attribute builtin_val = ConvertVhloAttrToBuiltin(attr.getValue());
      if (builtin_val) {
        new_attrs.push_back(rewriter.getNamedAttr(name, builtin_val));
      }
    }

    auto new_op = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), result_types, operands, new_attrs);

    if (new_op->getNumResults() != op->getNumResults()) {
      return failure();
    }

    SmallVector<Value> replacement_vals;
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      Value new_res = new_op.getResult(i);
      Type orig_type = op->getResult(i).getType();
      if (new_res.getType() != orig_type) {
        new_res = rewriter
                      .create<UnrealizedConversionCastOp>(op->getLoc(),
                                                          orig_type, new_res)
                      .getResult(0);
      }
      replacement_vals.push_back(new_res);
    }

    rewriter.replaceOp(op, replacement_vals);
    return success();
  }
};

class LegalizeVhloQuantCustomCallsPass
    : public impl::LegalizeVhloQuantCustomCallsPassBase<
          LegalizeVhloQuantCustomCallsPass> {
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LegalizeVhloQuantCustomCallPattern>(context);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateLegalizeVhloQuantCustomCallsPass() {
  return std::make_unique<LegalizeVhloQuantCustomCallsPass>();
}

static PassRegistration<LegalizeVhloQuantCustomCallsPass> pass;

}  // namespace mlir::odml
