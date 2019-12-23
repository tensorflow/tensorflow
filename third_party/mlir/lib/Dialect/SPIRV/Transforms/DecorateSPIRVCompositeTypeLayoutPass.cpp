//===- DecorateSPIRVCompositeTypeLayoutPass.cpp - Decorate composite type -===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a pass to decorate the composite types used by
// composite objects in the StorageBuffer, PhysicalStorageBuffer, Uniform, and
// PushConstant storage classes with layout information. See SPIR-V spec
// "2.16.2. Validation Rules for Shader Capabilities" for more details.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class SPIRVGlobalVariableOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::GlobalVariableOp> {
public:
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::GlobalVariableOp op,
                                     PatternRewriter &rewriter) const override {
    spirv::StructType::LayoutInfo structSize = 0;
    VulkanLayoutUtils::Size structAlignment = 1;
    SmallVector<NamedAttribute, 4> globalVarAttrs;

    auto ptrType = op.type().cast<spirv::PointerType>();
    auto structType = VulkanLayoutUtils::decorateType(
        ptrType.getPointeeType().cast<spirv::StructType>(), structSize,
        structAlignment);
    auto decoratedType =
        spirv::PointerType::get(structType, ptrType.getStorageClass());

    // Save all named attributes except "type" attribute.
    for (const auto &attr : op.getAttrs()) {
      if (attr.first == "type") {
        continue;
      }
      globalVarAttrs.push_back(attr);
    }

    rewriter.replaceOpWithNewOp<spirv::GlobalVariableOp>(
        op, TypeAttr::get(decoratedType), globalVarAttrs);
    return matchSuccess();
  }
};

class SPIRVAddressOfOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::AddressOfOp> {
public:
  using OpRewritePattern<spirv::AddressOfOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::AddressOfOp op,
                                     PatternRewriter &rewriter) const override {
    auto spirvModule = op.getParentOfType<spirv::ModuleOp>();
    auto varName = op.variable();
    auto varOp = spirvModule.lookupSymbol<spirv::GlobalVariableOp>(varName);

    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(
        op, varOp.type(), rewriter.getSymbolRefAttr(varName));
    return matchSuccess();
  }
};
} // namespace

static void populateSPIRVLayoutInfoPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *ctx) {
  patterns.insert<SPIRVGlobalVariableOpLayoutInfoDecoration,
                  SPIRVAddressOfOpLayoutInfoDecoration>(ctx);
}

namespace {
class DecorateSPIRVCompositeTypeLayoutPass
    : public ModulePass<DecorateSPIRVCompositeTypeLayoutPass> {
private:
  void runOnModule() override;
};
} // namespace

void DecorateSPIRVCompositeTypeLayoutPass::runOnModule() {
  auto module = getModule();
  OwningRewritePatternList patterns;
  populateSPIRVLayoutInfoPatterns(patterns, module.getContext());
  ConversionTarget target(*(module.getContext()));
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addLegalOp<FuncOp>();
  target.addDynamicallyLegalOp<spirv::GlobalVariableOp>(
      [](spirv::GlobalVariableOp op) {
        return VulkanLayoutUtils::isLegalType(op.type());
      });

  // Change the type for the direct users.
  target.addDynamicallyLegalOp<spirv::AddressOfOp>([](spirv::AddressOfOp op) {
    return VulkanLayoutUtils::isLegalType(op.pointer()->getType());
  });

  // TODO: Change the type for the indirect users such as spv.Load, spv.Store,
  // spv.FunctionCall and so on.

  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (failed(applyFullConversion(spirvModule, target, patterns))) {
      signalPassFailure();
    }
  }
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::spirv::createDecorateSPIRVCompositeTypeLayoutPass() {
  return std::make_unique<DecorateSPIRVCompositeTypeLayoutPass>();
}

static PassRegistration<DecorateSPIRVCompositeTypeLayoutPass>
    pass("decorate-spirv-composite-type-layout",
         "Decorate SPIR-V composite type with layout info");
