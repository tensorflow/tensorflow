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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
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
  void SetUp() override { LoadMlirDialectsForTriton(mlir_context_); }

  NameLoc NameLoc(absl::string_view name) {
    return NameLoc::get(StringAttr::get(&mlir_context_, name));
  }

  auto BuildModule(EmitterLocOpBuilder& b) {
    auto triton_module = llvm_ir::CreateMlirModuleOp(NameLoc("module"));
    b.setInsertionPointToEnd(triton_module->getBody());
    auto i32_type = b.getI32Type();
    auto attr = b.getIntegerAttr(i32_type, 42);
    b.create<mlir::arith::ConstantOp>(attr);
    return triton_module;
  }
  mlir::MLIRContext mlir_context_;
};

TEST_F(EmitterLocOpBuilderTest, IRWithAnnotations) {
  EmitterLocOpBuilder b(NameLoc("IRWithAnnotations"), &mlir_context_, true);
  auto triton_module = BuildModule(b);
  std::string ir = DumpTritonIR(triton_module.get(), true);
  EXPECT_THAT(RunFileCheck(ir, R"(
    CHECK: "IRWithAnnotations -> [[FILE:.*_test.cc]]:[[LINE:[0-9]+]]"
  )"),
              IsOkAndHolds(true));
}

TEST_F(EmitterLocOpBuilderTest, IRWithoutAnnotations) {
  EmitterLocOpBuilder b(NameLoc("IRWithoutAnnotations"), &mlir_context_, false);

  auto triton_module = BuildModule(b);
  std::string ir = DumpTritonIR(triton_module.get(), false);
  EXPECT_THAT(RunFileCheck(ir, R"(
    CHECK-NOT: IRWithoutAnnotations
  )"),
              IsOkAndHolds(true));
}

}  // namespace

}  // namespace xla::gpu
