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

#include "tensorflow/compiler/mlir/lite/python/preserve_regions.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::TFL {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// `tfl.reshape(tfl.reshape(x))` is folded into one reshape by the TFL
// canonicalizer, which makes it a simple probe for "was this optimized".
constexpr llvm::StringLiteral kSingleRegion = R"mlir(
module {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
    %s0 = "tfl.pseudo_const"() <{value = dense<[6, 4]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %s1 = "tfl.pseudo_const"() <{value = dense<24> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = stablehlo.composite "odml.preserve_region" %arg0, %s0, %s1 {
      decomposition = @region
    } : (tensor<2x3x4xf32>, tensor<2xi32>, tensor<1xi32>) -> tensor<24xf32>
    return %0 : tensor<24xf32>
  }
  func.func private @region(%x: tensor<2x3x4xf32>, %a: tensor<2xi32>, %b: tensor<1xi32>) -> tensor<24xf32> {
    %0 = "tfl.reshape"(%x, %a) : (tensor<2x3x4xf32>, tensor<2xi32>) -> tensor<6x4xf32>
    %1 = "tfl.reshape"(%0, %b) : (tensor<6x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    return %1 : tensor<24xf32>
  }
}
)mlir";

class PreserveRegionsTest : public ::testing::Test {
 protected:
  PreserveRegionsTest() {
    context_
        .loadDialect<arith::ArithDialect, func::FuncDialect,
                     stablehlo::StablehloDialect, TFL::TensorFlowLiteDialect>();
  }

  OwningOpRef<ModuleOp> Parse(llvm::StringRef ir) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context_);
    EXPECT_TRUE(module);
    return module;
  }

  // Runs the canonicalizer on `module`, standing in for the main pipeline.
  void Canonicalize(ModuleOp module) {
    PassManager pm(&context_);
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(pm.run(module)));
  }

  static int Count(mlir::Operation* root, llvm::StringRef op_name) {
    int count = 0;
    root->walk([&](mlir::Operation* op) {
      if (op->getName().getStringRef() == op_name) ++count;
    });
    return count;
  }

  static SmallVector<std::string> FuncNames(ModuleOp module) {
    SmallVector<std::string> names;
    for (auto func : module.getOps<func::FuncOp>()) {
      names.push_back(func.getSymName().str());
    }
    return names;
  }

  // Runs `fn` and returns the diagnostics it emitted.
  std::string CaptureErrors(llvm::function_ref<void()> fn) {
    std::string diagnostics;
    ScopedDiagnosticHandler handler(&context_, [&](Diagnostic& diag) {
      diagnostics += diag.str();
      return success();
    });
    fn();
    return diagnostics;
  }

  MLIRContext context_;
};

TEST_F(PreserveRegionsTest, DetectsOnlyPreserveRegionComposites) {
  OwningOpRef<ModuleOp> with_region = Parse(kSingleRegion);
  EXPECT_TRUE(HasPreserveRegions(*with_region));

  OwningOpRef<ModuleOp> other = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.something_else" %arg0 {
          decomposition = @body
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @body(%x: tensor<4xf32>) -> tensor<4xf32> {
        return %x : tensor<4xf32>
      }
    })mlir");
  EXPECT_FALSE(HasPreserveRegions(*other));
}

TEST_F(PreserveRegionsTest, SplitMovesBodyAndSinksConstants) {
  OwningOpRef<ModuleOp> module = Parse(kSingleRegion);
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));

  // The region module holds one body, with the constant operands cloned in
  // and the corresponding arguments no longer used.
  SmallVector<std::string> region_funcs = FuncNames(**regions);
  ASSERT_THAT(region_funcs, SizeIs(1));
  EXPECT_EQ(Count(**regions, "tfl.reshape"), 2);
  EXPECT_EQ(Count(**regions, "tfl.pseudo_const"), 2);
  auto body = *(*regions)->getOps<func::FuncOp>().begin();
  EXPECT_FALSE(body.getArgument(0).use_empty());
  EXPECT_TRUE(body.getArgument(1).use_empty());
  EXPECT_TRUE(body.getArgument(2).use_empty());

  // The composite now refers to a declaration of the same name in the main
  // module; the original decomposition is gone.
  stablehlo::CompositeOp composite;
  module->walk([&](stablehlo::CompositeOp op) { composite = op; });
  ASSERT_TRUE(composite);
  EXPECT_EQ(composite.getDecomposition(), region_funcs.front());
  auto placeholder = module->lookupSymbol<func::FuncOp>(region_funcs.front());
  ASSERT_TRUE(placeholder);
  EXPECT_TRUE(placeholder.isExternal());
  EXPECT_TRUE(placeholder.isPrivate());
  EXPECT_FALSE(module->lookupSymbol("region"));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(succeeded(verify(**regions)));
}

TEST_F(PreserveRegionsTest, MainPipelineDoesNotReachRegionBody) {
  OwningOpRef<ModuleOp> module = Parse(kSingleRegion);
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));

  // The main module only has a declaration left, so the main pipeline has
  // nothing of the region to rewrite ...
  Canonicalize(*module);
  EXPECT_EQ(Count(*module, "tfl.reshape"), 0);

  // ... and the stitched result is the untouched body.
  ASSERT_TRUE(succeeded(StitchPreserveRegions(*module, **regions)));
  EXPECT_EQ(Count(*module, "stablehlo.composite"), 0);
  EXPECT_EQ(Count(*module, "tfl.reshape"), 2);
  EXPECT_THAT(FuncNames(*module), ElementsAre("main"));
  // The constants feeding the composite were replaced by the body's copies.
  EXPECT_EQ(Count(*module, "tfl.pseudo_const"), 2);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(PreserveRegionsTest, SharedDecompositionGetsOneBodyPerCallSite) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
        %c1 = "tfl.pseudo_const"() <{value = dense<1.0> : tensor<4xf32>}> : () -> tensor<4xf32>
        %c2 = "tfl.pseudo_const"() <{value = dense<2.0> : tensor<4xf32>}> : () -> tensor<4xf32>
        %0 = stablehlo.composite "odml.preserve_region" %arg0, %c1 {
          decomposition = @add
        } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1 = stablehlo.composite "odml.preserve_region" %arg0, %c2 {
          decomposition = @add
        } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        return %0, %1 : tensor<4xf32>, tensor<4xf32>
      }
      func.func private @add(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
        %0 = tfl.add %x, %y {fused_activation_function = "NONE"} : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));
  EXPECT_THAT(FuncNames(**regions), SizeIs(2));

  // Each body sank its own call site's constant.
  SmallVector<float> sunk;
  (*regions)->walk([&](TFL::ConstOp op) {
    sunk.push_back(
        cast<DenseElementsAttr>(op.getValue()).getSplatValue<float>());
  });
  EXPECT_THAT(sunk, UnorderedElementsAre(1.0f, 2.0f));

  ASSERT_TRUE(succeeded(StitchPreserveRegions(*module, **regions)));
  EXPECT_EQ(Count(*module, "tfl.add"), 2);
  EXPECT_THAT(FuncNames(*module), ElementsAre("main"));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(PreserveRegionsTest, NestedRegionsAreFlattenedIntoTheOuterBody) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %arg0 {
          decomposition = @outer
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @outer(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %x {
          decomposition = @inner
        } : (tensor<4xf32>) -> tensor<4xf32>
        %1 = "tfl.relu"(%0) : (tensor<4xf32>) -> tensor<4xf32>
        return %1 : tensor<4xf32>
      }
      func.func private @inner(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = "tfl.neg"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));
  EXPECT_THAT(FuncNames(**regions), SizeIs(1));
  EXPECT_EQ(Count(**regions, "stablehlo.composite"), 0);
  EXPECT_EQ(Count(**regions, "tfl.neg"), 1);
  EXPECT_EQ(Count(**regions, "tfl.relu"), 1);
  // Both original decompositions are gone; only the placeholder is left, and
  // it is flat too.
  EXPECT_THAT(FuncNames(*module), SizeIs(2));
  EXPECT_EQ(Count(*module, "stablehlo.composite"), 1);

  ASSERT_TRUE(succeeded(StitchPreserveRegions(*module, **regions)));
  EXPECT_EQ(Count(*module, "stablehlo.composite"), 0);
  EXPECT_EQ(Count(*module, "tfl.neg"), 1);
  EXPECT_EQ(Count(*module, "tfl.relu"), 1);
  EXPECT_THAT(FuncNames(*module), ElementsAre("main"));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(PreserveRegionsTest, SplitFailsWithoutDecomposition) {
  // The parser verifies symbol uses, so drop the decomposition after parsing.
  OwningOpRef<ModuleOp> module = Parse(kSingleRegion);
  cast<func::FuncOp>(module->lookupSymbol("region")).erase();

  std::string diagnostics = CaptureErrors(
      [&] { EXPECT_TRUE(failed(SplitPreserveRegions(*module))); });
  EXPECT_THAT(diagnostics, HasSubstr("decomposition @region not found"));
}

TEST_F(PreserveRegionsTest, StitchFailsOnBoundaryTypeMismatch) {
  OwningOpRef<ModuleOp> module = Parse(kSingleRegion);
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));

  // Simulate the region pipeline legalizing the result type differently.
  auto body = *(*regions)->getOps<func::FuncOp>().begin();
  body.setType(FunctionType::get(
      &context_, body.getArgumentTypes(),
      {RankedTensorType::get({24}, Float16Type::get(&context_))}));

  std::string diagnostics = CaptureErrors(
      [&] { EXPECT_TRUE(failed(StitchPreserveRegions(*module, **regions))); });
  EXPECT_THAT(diagnostics, HasSubstr("signature does not match"));
}

TEST_F(PreserveRegionsTest, StitchFailsWithoutRegionBody) {
  OwningOpRef<ModuleOp> module = Parse(kSingleRegion);
  OwningOpRef<ModuleOp> empty = ModuleOp::create(UnknownLoc::get(&context_));

  std::string diagnostics = CaptureErrors(
      [&] { EXPECT_TRUE(failed(StitchPreserveRegions(*module, *empty))); });
  EXPECT_THAT(diagnostics, HasSubstr("no region body named @region"));
}

TEST_F(PreserveRegionsTest, ConstantSharedWithMainOpIsKept) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
        %c = "tfl.pseudo_const"() <{value = dense<1.0> : tensor<4xf32>}> : () -> tensor<4xf32>
        %0 = stablehlo.composite "odml.preserve_region" %arg0, %c {
          decomposition = @add
        } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1 = tfl.mul %arg0, %c {fused_activation_function = "NONE"} : tensor<4xf32>
        return %0, %1 : tensor<4xf32>, tensor<4xf32>
      }
      func.func private @add(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
        %0 = tfl.add %x, %y {fused_activation_function = "NONE"} : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));
  ASSERT_TRUE(succeeded(StitchPreserveRegions(*module, **regions)));
  // One constant for the main op, one copy inside the stitched body.
  EXPECT_EQ(Count(*module, "tfl.pseudo_const"), 2);
  EXPECT_EQ(Count(*module, "tfl.mul"), 1);
  EXPECT_EQ(Count(*module, "tfl.add"), 1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(PreserveRegionsTest, RegionInNonMainFunction) {
  // Tests the API directly: in the converter, the lowering inliner inlines
  // `@helper` into `@main` before the split.
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = func.call @helper(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @helper(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %arg0 {
          decomposition = @region
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @region(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = "tfl.neg"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));
  ASSERT_TRUE(succeeded(StitchPreserveRegions(*module, **regions)));
  EXPECT_THAT(FuncNames(*module), ElementsAre("main", "helper"));
  EXPECT_EQ(Count(*module, "tfl.neg"), 1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(PreserveRegionsTest, PublicDecompositionIsKept) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %arg0 {
          decomposition = @region
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func @region(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = "tfl.neg"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));
  EXPECT_TRUE(module->lookupSymbol("region"));
  ASSERT_TRUE(succeeded(StitchPreserveRegions(*module, **regions)));
  EXPECT_THAT(FuncNames(*module), ElementsAre("main", "region"));
}

TEST_F(PreserveRegionsTest, SplitFailsOnOtherSymbolInRegionBody) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %arg0 {
          decomposition = @region
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @region(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "foo.bar" %x {
          decomposition = @foo
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @foo(%x: tensor<4xf32>) -> tensor<4xf32> {
        return %x : tensor<4xf32>
      }
    })mlir");
  std::string diagnostics = CaptureErrors(
      [&] { EXPECT_TRUE(failed(SplitPreserveRegions(*module))); });
  EXPECT_THAT(diagnostics, HasSubstr("refers to symbol @foo"));
  // Nothing was modified.
  EXPECT_TRUE(module->lookupSymbol("region"));
  EXPECT_THAT(FuncNames(*module), SizeIs(3));
}

TEST_F(PreserveRegionsTest, SplitFailsOnRecursiveDecomposition) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %arg0 {
          decomposition = @region
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @region(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %x {
          decomposition = @region
        } : (tensor<4xf32>) -> tensor<4xf32>
        %1 = stablehlo.composite "odml.preserve_region" %0 {
          decomposition = @region
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %1 : tensor<4xf32>
      }
    })mlir");
  std::string diagnostics = CaptureErrors(
      [&] { EXPECT_TRUE(failed(SplitPreserveRegions(*module))); });
  EXPECT_THAT(diagnostics, HasSubstr("decomposition @region is recursive"));
}

TEST_F(PreserveRegionsTest, SplitFailsOnNestedRegionsSharedWithOtherUses) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
        %0 = stablehlo.composite "odml.preserve_region" %arg0 {
          decomposition = @outer
        } : (tensor<4xf32>) -> tensor<4xf32>
        %1 = stablehlo.composite "foo.bar" %arg0 {
          decomposition = @outer
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0, %1 : tensor<4xf32>, tensor<4xf32>
      }
      func.func private @outer(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %x {
          decomposition = @inner
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @inner(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = "tfl.neg"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  std::string diagnostics = CaptureErrors(
      [&] { EXPECT_TRUE(failed(SplitPreserveRegions(*module))); });
  EXPECT_THAT(diagnostics,
              HasSubstr("also referenced outside of a preserve region"));
}

TEST_F(PreserveRegionsTest, SplitFailsOnPublicDecompositionWithNestedRegions) {
  // A public decomposition survives the split, and with it the nested
  // composite in it, which would have no body to stitch back.
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %arg0 {
          decomposition = @outer
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func @outer(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.composite "odml.preserve_region" %x {
          decomposition = @inner
        } : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
      func.func private @inner(%x: tensor<4xf32>) -> tensor<4xf32> {
        %0 = "tfl.neg"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  std::string diagnostics = CaptureErrors(
      [&] { EXPECT_TRUE(failed(SplitPreserveRegions(*module))); });
  EXPECT_THAT(diagnostics, HasSubstr("contains nested preserve regions and is "
                                     "public"));
  // Nothing was modified.
  EXPECT_THAT(FuncNames(*module), ElementsAre("main", "outer", "inner"));
  EXPECT_EQ(Count(*module, "stablehlo.composite"), 2);
}

TEST_F(PreserveRegionsTest, ModuleWithoutRegionsIsUnchanged) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = "tfl.neg"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    })mlir");
  auto regions = SplitPreserveRegions(*module);
  ASSERT_TRUE(succeeded(regions));
  EXPECT_THAT(FuncNames(**regions), IsEmpty());
  ASSERT_TRUE(succeeded(StitchPreserveRegions(*module, **regions)));
  EXPECT_EQ(Count(*module, "tfl.neg"), 1);
}

}  // namespace
}  // namespace mlir::TFL
