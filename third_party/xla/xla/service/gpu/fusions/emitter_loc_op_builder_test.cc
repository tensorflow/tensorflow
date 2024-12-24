/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/fusions/emitter_loc_op_builder.h"

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/fusions/triton/triton_fusion_emitter.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

using mlir::NameLoc;
using mlir::StringAttr;
using ::tsl::testing::IsOkAndHolds;

class EmitterLocOpBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override { LoadMlirDialectsForTriton(context_); }

  mlir::MLIRContext context_;
};

NameLoc NameLoc(mlir::MLIRContext& context, absl::string_view name) {
  return NameLoc::get(StringAttr::get(&context, name));
}

mlir::OwningOpRef<mlir::ModuleOp> MakeModuleWithOneOp(
    mlir::MLIRContext& context, EmitterLocOpBuilder& b) {
  auto loc = NameLoc(context, "module");
  auto triton_module = llvm_ir::CreateMlirModuleOp(loc);
  b.setInsertionPointToEnd(triton_module->getBody());
  auto i32_type = b.getI32Type();
  auto attr = b.getIntegerAttr(i32_type, 42);
  b.create<mlir::arith::ConstantOp>(attr);
  return triton_module;
}

TEST_F(EmitterLocOpBuilderTest, IRWithAnnotations) {
  auto loc = NameLoc(context_, "IRWithAnnotations");
  EmitterLocOpBuilder b(loc, &context_, /*annotate_loc=*/true);
  auto triton_module = MakeModuleWithOneOp(context_, b);
  std::string ir = DumpTritonIR(triton_module.get(), /*dump_annotations=*/true);
  if constexpr (EmitterLocOpBuilder::kSourceLocationSupported) {
    EXPECT_THAT(RunFileCheck(ir, R"(
      CHECK: "IRWithAnnotations -> [[FILE:.*_test.cc]]:[[LINE:[0-9]+]]"
    )"),
                IsOkAndHolds(true));
  } else {
    EXPECT_THAT(RunFileCheck(ir, R"(
      CHECK: "IRWithAnnotations"
    )"),
                IsOkAndHolds(true));
  }
}

TEST_F(EmitterLocOpBuilderTest, IRWithoutAnnotations) {
  auto loc = NameLoc(context_, "IRWithoutAnnotations");
  EmitterLocOpBuilder b(loc, &context_, /*annotate_loc=*/false);
  auto triton_module = MakeModuleWithOneOp(context_, b);
  std::string ir =
      DumpTritonIR(triton_module.get(), /*dump_annotations=*/false);
  EXPECT_THAT(RunFileCheck(ir, R"(
    CHECK-NOT: IRWithoutAnnotations
  )"),
              IsOkAndHolds(true));
}

}  // namespace

}  // namespace xla::gpu
