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

#include "tensorflow/compiler/mlir/lite/python/location_passes.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/utils/name_utils.h"

namespace mlir::TFL {
namespace {

using ::testing::Contains;
using ::testing::Each;
using ::testing::HasSubstr;
using ::testing::Not;

void InitMlirContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                  mlir::stablehlo::StablehloDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

std::vector<std::string> CollectOpLocationNames(mlir::ModuleOp module) {
  std::vector<std::string> names;
  module.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::ReturnOp>(
            op)) {
      return;
    }
    names.push_back(mlir::GetNameFromLoc(op->getLoc()));
  });
  return names;
}

TEST(LocationPassesTest, NormalizesInlinedCallSitesAndComposites) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func private @_where(
          %p: tensor<2xi1>, %x: tensor<2xf32>, %y: tensor<2xf32>)
          -> tensor<2xf32> {
        %0 = stablehlo.select %p, %x, %y : tensor<2xi1>, tensor<2xf32>
            loc("select_n"("_where"("lax_numpy.py":10:3)))
        return %0 : tensor<2xf32>
      }

      func.func private @bmm_decomp(
          %lhs: tensor<2xf32>, %rhs: tensor<2xf32>) -> tensor<2xf32> {
        %0 = stablehlo.multiply %lhs, %rhs : tensor<2xf32>
        return %0 : tensor<2xf32>
      }

      func.func @main(
          %p: tensor<2xi1>, %x: tensor<2xf32>, %y: tensor<2xf32>)
          -> tensor<2xf32> {
        %0 = func.call @_where(%p, %x, %y)
            : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            loc("pjit:"("jit(decode)/blocks/0/attn/jit(_where)"(
                "attend"("attn.py":20:5))))
        %1 = stablehlo.composite "odml.runtime_bmm" %0, %y
            {decomposition = @bmm_decomp}
            : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            loc("composite:"("jit(decode)/blocks/0/attn/composite"(
                "runtime_bmm"("bmm.py":42:3))))
        return %1 : tensor<2xf32>
      }
    }
  )mlir";

  mlir::MLIRContext context;
  InitMlirContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlir, &context);
  ASSERT_TRUE(static_cast<bool>(module));

  mlir::PassManager pm(&context);
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(CreateNormalizeLocationsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  std::vector<std::string> names = CollectOpLocationNames(*module);
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/attn/select_n"));
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/attn/odml.runtime_bmm"));
  EXPECT_THAT(names, Each(Not(HasSubstr("jit(_where)"))));
}

TEST(LocationPassesTest, SanitizesCrossScopeFusedLocAndPreservesSemicolons) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
        %0 = stablehlo.add %x, %y : tensor<2xf32>
            loc(fused[
                "jit(decode)/blocks/4/attn/div"("a.py":5:1),
                "jit(decode)/blocks/0/attn/max"("a.py":6:1)])
        %1 = stablehlo.multiply %0, %y : tensor<2xf32>
            loc(fused[
                "model.layers.4;aten.mul"("t.py":1:1),
                "model.layers.0;aten.add"("t.py":2:1)])
        return %1 : tensor<2xf32>
      }
    }
  )mlir";

  mlir::MLIRContext context;
  InitMlirContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlir, &context);
  ASSERT_TRUE(static_cast<bool>(module));

  mlir::PassManager pm(&context);
  pm.addPass(CreateFinalizeExportLocationsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  std::vector<std::string> names = CollectOpLocationNames(*module);
  EXPECT_THAT(names, Contains("jit(decode)/blocks/4/attn/div"));
  EXPECT_THAT(names,
              Contains("model.layers.4;aten.mul;model.layers.0;aten.add"));
}

TEST(LocationPassesTest, UpdatesEntryFunctionOutputsWithUniqueHierarchy) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func @decode(%x: tensor<2xf32>, %y: tensor<2xf32>)
          -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
          attributes {
            tf.entry_function = {
              inputs = "decode_x,decode_y",
              outputs = "decode_o0_output,decode_o1_output,decode_o2_output"
            }
          } {
        %0 = stablehlo.add %x, %y : tensor<2xf32>
            loc("jit(decode)/blocks/0/attn/update/odml.cache_update"(
                "cache.py":10:3))
        %1 = stablehlo.subtract %x, %y : tensor<2xf32>
            loc("jit(decode)/blocks/0/attn/update/odml.cache_update"(
                "cache.py":11:3))
        %2 = stablehlo.multiply %0, %1 : tensor<2xf32>
            loc("jit(decode)/lm_head/bt,vh->bv/mul"("head.py":20:3))
        return %0, %1, %2 : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
      }
    }
  )mlir";

  mlir::MLIRContext context;
  InitMlirContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlir, &context);
  ASSERT_TRUE(static_cast<bool>(module));

  mlir::PassManager pm(&context);
  pm.addPass(CreateFinalizeExportLocationsPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  mlir::func::FuncOp func = *module->getOps<mlir::func::FuncOp>().begin();
  auto dict_attr =
      func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  ASSERT_TRUE(static_cast<bool>(dict_attr));
  auto outputs_attr =
      mlir::dyn_cast_or_null<mlir::StringAttr>(dict_attr.get("outputs"));
  ASSERT_TRUE(static_cast<bool>(outputs_attr));
  EXPECT_EQ(outputs_attr.getValue().str(),
            "jit(decode)/blocks/0/attn/update/odml.cache_update,"
            "jit(decode)/blocks/0/attn/update/odml.cache_update1,"
            "jit(decode)/lm_head/bt_vh->bv/mul");
}

}  // namespace
}  // namespace mlir::TFL
