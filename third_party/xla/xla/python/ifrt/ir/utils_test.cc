/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/utils.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(UtilsTest, GetPrettyLocationUnknownLoc) {
  mlir::MLIRContext context;
  mlir::Location loc = mlir::UnknownLoc::get(&context);
  EXPECT_EQ(GetPrettyLocation(loc), "\t<unknown location>\n");
}

TEST(UtilsTest, GetPrettyLocationNameLocWithUnknownChild) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::Location loc = mlir::NameLoc::get(builder.getStringAttr("my_arg"),
                                          mlir::UnknownLoc::get(&context));
  EXPECT_EQ(GetPrettyLocation(loc), "\t\"my_arg\"\n");
}

TEST(UtilsTest, GetPrettyLocationFileLineColLoc) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::Location loc = mlir::FileLineColLoc::get(
      &context, builder.getStringAttr("my_file.py"), 42, 10);
  EXPECT_EQ(GetPrettyLocation(loc), "\t\"my_file.py\":42:10 to 10\n");
}

TEST(UtilsTest, GetPrettyLocationNameLocWithFileLineColLoc) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::Location file_loc = mlir::FileLineColLoc::get(
      &context, builder.getStringAttr("my_file.py"), 42, 10);
  mlir::Location loc =
      mlir::NameLoc::get(builder.getStringAttr("my_arg"), file_loc);
  EXPECT_EQ(GetPrettyLocation(loc),
            "\t\"my_file.py\":42:10 to 10\n\t ^ \"my_arg\"\n");
}

TEST(UtilsTest, GetPrettyLocationCallSiteLoc) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::Location callee = mlir::FileLineColLoc::get(
      &context, builder.getStringAttr("callee.py"), 10, 2);
  mlir::Location caller = mlir::FileLineColLoc::get(
      &context, builder.getStringAttr("caller.py"), 25, 4);
  mlir::Location call_site = mlir::CallSiteLoc::get(callee, caller);
  EXPECT_EQ(GetPrettyLocation(call_site),
            "\t\"callee.py\":10:2 to 2\n\t\"caller.py\":25:4 to 4\n");
}

TEST(UtilsTest, GetArgPrettyLocation) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  const char* kModuleStr = R"mlir(
module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    return %arg0 : tensor<2xf32>
  }
}
)mlir";
  auto module_or = support::ParseMlirModuleString(kModuleStr, context);
  ASSERT_TRUE(module_or.ok());
  mlir::OwningOpRef<mlir::ModuleOp> module = *std::move(module_or);

  // By default, parsed arguments have a FileLineColLoc from the parsed text.
  EXPECT_THAT(GetArgPrettyLocation(0, *module), HasSubstr("mlir:3:"));

  // Override argument locations with a NameLoc with UnknownLoc child.
  mlir::OpBuilder builder(&context);
  mlir::func::FuncOp func = GetMainFunction(*module);
  ASSERT_TRUE(func != nullptr);
  func.getArgument(0).setLoc(mlir::NameLoc::get(
      builder.getStringAttr("x"), mlir::UnknownLoc::get(&context)));
  func.getArgument(1).setLoc(mlir::NameLoc::get(
      builder.getStringAttr("y"), mlir::UnknownLoc::get(&context)));

  EXPECT_EQ(GetArgPrettyLocation(0, *module), "\t\"x\"\n");
  EXPECT_EQ(GetArgPrettyLocation(1, *module), "\t\"y\"\n");
}

TEST(UtilsTest, IfrtDevicesAttrRejectsEmptyDeviceList) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
  IfrtDevicesAttr empty_devices = IfrtDevicesAttr::getChecked(
      [&]() { return mlir::emitError(mlir::UnknownLoc::get(&context)); },
      &context, /*ids=*/llvm::ArrayRef<int>());
  EXPECT_EQ(empty_devices.getAsOpaquePointer(), nullptr);
  EXPECT_THAT(diagnostic_handler.ConsumeStatus(),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("Device list cannot be empty")));
}

TEST(UtilsTest, IfrtArrayTypeRejectsNullParameters) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  mlir::OpBuilder builder(&context);
  auto emit_error = [&]() {
    return mlir::emitError(mlir::UnknownLoc::get(&context));
  };
  mlir::RankedTensorType valid_shape =
      mlir::RankedTensorType::get({2}, builder.getI32Type());
  IfrtShardingAttrInterface valid_sharding =
      IfrtUnspecifiedShardingAttr::get(&context);
  IfrtDevicesAttr valid_devices =
      IfrtDevicesAttr::get(&context, /*ids=*/llvm::ArrayRef<int>({0, 1}));

  {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
    IfrtArrayType null_shape = IfrtArrayType::getChecked(
        emit_error, &context, /*shape=*/mlir::RankedTensorType(),
        valid_sharding, valid_devices, /*memory_kind_attr=*/mlir::StringAttr(),
        /*layout_attr=*/mlir::StringAttr());
    EXPECT_EQ(null_shape, nullptr);
    EXPECT_THAT(
        diagnostic_handler.ConsumeStatus(),
        StatusIs(absl::StatusCode::kUnknown,
                 HasSubstr("expected non-null IfrtArrayType parameters")));
  }

  {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
    IfrtArrayType null_sharding = IfrtArrayType::getChecked(
        emit_error, &context, valid_shape,
        /*sharding_attr=*/IfrtShardingAttrInterface(), valid_devices,
        /*memory_kind_attr=*/mlir::StringAttr(),
        /*layout_attr=*/mlir::StringAttr());
    EXPECT_EQ(null_sharding, nullptr);
    EXPECT_THAT(
        diagnostic_handler.ConsumeStatus(),
        StatusIs(absl::StatusCode::kUnknown,
                 HasSubstr("expected non-null IfrtArrayType parameters")));
  }

  {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
    IfrtArrayType null_devices = IfrtArrayType::getChecked(
        emit_error, &context, valid_shape, valid_sharding,
        /*devices_attr=*/IfrtDevicesAttr(),
        /*memory_kind_attr=*/mlir::StringAttr(),
        /*layout_attr=*/mlir::StringAttr());
    EXPECT_EQ(null_devices, nullptr);
    EXPECT_THAT(
        diagnostic_handler.ConsumeStatus(),
        StatusIs(absl::StatusCode::kUnknown,
                 HasSubstr("expected non-null IfrtArrayType parameters")));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
