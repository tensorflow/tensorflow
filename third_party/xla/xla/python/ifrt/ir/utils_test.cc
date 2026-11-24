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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout_util.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::SizeIs;

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

TEST(UtilsTest, GetIfrtFunctionsSnapshotsOnlyIfrtFunctions) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  const char* kModuleStr = R"mlir(
!arr = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func private @atom_helper(%x: tensor<2xi32>) -> tensor<2xi32> {
    return %x : tensor<2xi32>
  }
  func.func @main(%x: !arr) -> !arr
      attributes {ifrt.function} {
    return %x : !arr
  }
  func.func private @reshard_fn(%x: !arr) -> !arr
      attributes {ifrt.reshard_function} {
    return %x : !arr
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       support::ParseMlirModuleString(kModuleStr, context));
  llvm::SmallVector<mlir::func::FuncOp> ifrt_funcs = GetIfrtFunctions(*module);
  ASSERT_EQ(ifrt_funcs.size(), 2u);
  EXPECT_EQ(ifrt_funcs[0].getSymName(), "main");
  EXPECT_EQ(ifrt_funcs[1].getSymName(), "reshard_fn");
}

TEST(UtilsTest, ArrayTypeLocalShapeAndHasSameLayout) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  const char* kModuleStr = R"mlir(
!arr_default = !ifrt.array<tensor<4x6xf32>,
                           #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!arr_explicit_default = !ifrt.array<tensor<4x6xf32>,
                                    #ifrt.sharding_param<2x1 to [0] on 2>,
                                    [0, 1], layout = "default">
!arr_user_layout = !ifrt.array<tensor<4x6xf32>,
                               #ifrt.sharding_param<2x1 to [0] on 2>,
                               [0, 1], layout = "{0,1}">
!arr_other_user_layout = !ifrt.array<tensor<4x6xf32>,
                                     #ifrt.sharding_param<2x1 to [0] on 2>,
                                     [0, 1], layout = "{1,0}">
!arr_unspec = !ifrt.array<tensor<4x6xf32>, #ifrt.sharding_unspecified, [0, 1]>
module {
  func.func @main(%a: !arr_default, %b: !arr_explicit_default,
                  %c: !arr_user_layout, %d: !arr_other_user_layout,
                  %e: !arr_unspec)
      attributes {ifrt.function} {
    return
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       support::ParseMlirModuleString(kModuleStr, context));
  mlir::func::FuncOp main_func = GetMainFunction(*module);
  IfrtArrayType arr_default = GetArrayType(main_func.getArgument(0));
  IfrtArrayType arr_explicit_default = GetArrayType(main_func.getArgument(1));
  IfrtArrayType arr_user_layout = GetArrayType(main_func.getArgument(2));
  IfrtArrayType arr_other_user_layout = GetArrayType(main_func.getArgument(3));
  IfrtArrayType arr_unspec = GetArrayType(main_func.getArgument(4));

  EXPECT_THAT(arr_default.LocalShape(), IsOkAndHolds(ElementsAre(2, 6)));
  EXPECT_THAT(arr_unspec.LocalShape(), IsOkAndHolds(ElementsAre(4, 6)));

  EXPECT_TRUE(arr_default.HasSameLayout(arr_explicit_default));
  EXPECT_FALSE(arr_default.HasSameLayout(arr_user_layout));
  EXPECT_TRUE(arr_user_layout.HasSameLayout(arr_user_layout));
  EXPECT_FALSE(arr_user_layout.HasSameLayout(arr_other_user_layout));
}

TEST(UtilsTest, OutlinedAtomProgramModuleHelpers) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  const char* kModuleStr = R"mlir(
!arr = !ifrt.array<tensor<2x2xi32>,
                   #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
module @root {
  func.func private @flat_callee(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %a : tensor<2x2xi32>
  }
  module @outlined_mod <sym_visibility = "private"> {
    func.func @main(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
      return %a : tensor<2x2xi32>
    }
    func.func @other(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
      return %a : tensor<2x2xi32>
    }
  }
  module @outer_mod <sym_visibility = "private"> {
    module @inner_mod {
      func.func @main(%a: tensor<2x2xi32>) -> tensor<2x2xi32> {
        return %a : tensor<2x2xi32>
      }
    }
  }
  func.func @main(%x: !arr) attributes {ifrt.function} {
    %0, %c0 = ifrt.Call @flat_callee(%x) on devices [0, 1] : (!arr) -> !arr
    %1, %c1 = ifrt.Call @outlined_mod::@main(%x) on devices [0, 1]
        : (!arr) -> !arr
    %2, %c2 = ifrt.Call @outlined_mod::@other(%x) on devices [0, 1]
        : (!arr) -> !arr
    return
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       support::ParseMlirModuleString(kModuleStr, context));
  mlir::func::FuncOp main_func = GetMainFunction(*module);
  llvm::SmallVector<CallOp> calls(main_func.getOps<CallOp>());
  ASSERT_EQ(calls.size(), 3u);

  mlir::func::FuncOp flat_callee =
      module->lookupSymbol<mlir::func::FuncOp>("flat_callee");
  mlir::ModuleOp outlined_mod =
      module->lookupSymbol<mlir::ModuleOp>("outlined_mod");
  mlir::ModuleOp outer_mod = module->lookupSymbol<mlir::ModuleOp>("outer_mod");
  ASSERT_NE(flat_callee, nullptr);
  ASSERT_NE(outlined_mod, nullptr);
  ASSERT_NE(outer_mod, nullptr);
  mlir::ModuleOp inner_mod =
      outer_mod.lookupSymbol<mlir::ModuleOp>("inner_mod");
  ASSERT_NE(inner_mod, nullptr);
  mlir::func::FuncOp outlined_main = GetMainFunction(outlined_mod);
  mlir::func::FuncOp inner_main = GetMainFunction(inner_mod);

  EXPECT_EQ(MakeOutlinedCalleeSymbolRef(outlined_mod, outlined_main),
            calls[1].getCalleeAttr());
  EXPECT_EQ(GetCalleeRootSymbolOp(flat_callee, *module),
            flat_callee.getOperation());
  EXPECT_EQ(GetCalleeRootSymbolOp(outlined_main, *module),
            outlined_mod.getOperation());
  EXPECT_EQ(GetCalleeRootSymbolOp(inner_main, *module), nullptr);

  mlir::SymbolTableCollection symbol_table;
  {
    mlir::BaseScopedDiagnosticHandler diag(&context);
    EXPECT_TRUE(
        mlir::failed(GetOutlinedAtomProgramModule(calls[0], symbol_table)));
    EXPECT_THAT(diag.ConsumeStatus(),
                StatusIs(absl::StatusCode::kUnknown,
                         AllOf(HasSubstr("requires callee outlined"),
                               HasSubstr("in enclosing ModuleOp"))));
  }
  {
    mlir::BaseScopedDiagnosticHandler diag(&context);
    EXPECT_TRUE(
        mlir::failed(GetOutlinedAtomProgramModule(calls[2], symbol_table)));
    EXPECT_THAT(diag.ConsumeStatus(),
                StatusIs(absl::StatusCode::kUnknown,
                         AllOf(HasSubstr("requires callee outlined"),
                               HasSubstr("in child ModuleOp"))));
  }

  mlir::FailureOr<mlir::ModuleOp> resolved_mod =
      GetOutlinedAtomProgramModule(calls[1], symbol_table);
  ASSERT_TRUE(mlir::succeeded(resolved_mod));
  EXPECT_EQ(*resolved_mod, outlined_mod);

  ClonedAtomProgramModule cloned =
      CloneOutlinedAtomProgramModule(outlined_mod, symbol_table);
  ASSERT_NE(cloned.module, nullptr);
  ASSERT_NE(cloned.main_func, nullptr);
  EXPECT_NE(cloned.module, outlined_mod);
  EXPECT_EQ(cloned.module->getPrevNode(), outlined_mod.getOperation());
  EXPECT_EQ(mlir::SymbolTable::getSymbolVisibility(cloned.module),
            mlir::SymbolTable::Visibility::Private);
  EXPECT_EQ(cloned.callee_ref,
            MakeOutlinedCalleeSymbolRef(cloned.module, cloned.main_func));
}

TEST(UtilsTest, IsTrueOrUnitAttr) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  EXPECT_FALSE(IsTrueOrUnitAttr(mlir::Attribute()));
  EXPECT_TRUE(IsTrueOrUnitAttr(builder.getUnitAttr()));
  EXPECT_TRUE(IsTrueOrUnitAttr(builder.getBoolAttr(true)));
  EXPECT_FALSE(IsTrueOrUnitAttr(builder.getBoolAttr(false)));
  EXPECT_FALSE(IsTrueOrUnitAttr(builder.getI32IntegerAttr(1)));
  EXPECT_FALSE(IsTrueOrUnitAttr(builder.getStringAttr("true")));
}

TEST(UtilsTest, HasDonatedOrAliasedDictAttr) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  auto make_dict = [&](llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    return builder.getDictionaryAttr(attrs);
  };

  EXPECT_FALSE(HasDonatedOrAliasedDictAttr(mlir::DictionaryAttr()));
  EXPECT_FALSE(HasDonatedOrAliasedDictAttr(make_dict({})));
  EXPECT_FALSE(HasDonatedOrAliasedDictAttr(
      make_dict({builder.getNamedAttr("other", builder.getUnitAttr())})));
  EXPECT_FALSE(HasDonatedOrAliasedDictAttr(make_dict({builder.getNamedAttr(
      kBufferDonationAttrName, builder.getBoolAttr(false))})));
  EXPECT_TRUE(HasDonatedOrAliasedDictAttr(make_dict({builder.getNamedAttr(
      kBufferDonationAttrName, builder.getBoolAttr(true))})));
  EXPECT_TRUE(HasDonatedOrAliasedDictAttr(make_dict(
      {builder.getNamedAttr(kBufferDonationAttrName, builder.getUnitAttr())})));
  EXPECT_TRUE(HasDonatedOrAliasedDictAttr(make_dict({builder.getNamedAttr(
      kAliasingOutputAttrName, builder.getI32IntegerAttr(0))})));
  EXPECT_TRUE(HasDonatedOrAliasedDictAttr(make_dict(
      {builder.getNamedAttr(kIfrtDonatedArgAttrName, builder.getUnitAttr())})));
  EXPECT_FALSE(HasDonatedOrAliasedDictAttr(make_dict({builder.getNamedAttr(
      kIfrtDonatedArgAttrName, builder.getBoolAttr(false))})));
  EXPECT_TRUE(HasDonatedOrAliasedDictAttr(make_dict({builder.getNamedAttr(
      kIfrtDonatedArgAttrName, builder.getBoolAttr(true))})));
}

TEST(UtilsTest, IsTokenArrayType) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  const char* kModuleStr = R"mlir(
!arr_f32 = !ifrt.array<tensor<2xf32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
!arr_tok = !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0] on 1>, [0]>
module {
  func.func @main(%a: !arr_f32, %t: !arr_tok) attributes {ifrt.function} {
    return
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       support::ParseMlirModuleString(kModuleStr, context));
  mlir::func::FuncOp main_func = GetMainFunction(*module);

  EXPECT_FALSE(IsTokenArrayType(mlir::Type()));
  EXPECT_FALSE(IsTokenArrayType(mlir::IntegerType::get(&context, 32)));
  EXPECT_FALSE(IsTokenArrayType(IfrtTokenType::get(&context)));
  EXPECT_FALSE(IsTokenArrayType(IfrtControlType::get(&context)));
  EXPECT_FALSE(IsTokenArrayType(
      mlir::RankedTensorType::get({}, IfrtTokenType::get(&context))));
  EXPECT_FALSE(IsTokenArrayType(main_func.getArgument(0).getType()));
  EXPECT_TRUE(IsTokenArrayType(main_func.getArgument(1).getType()));
}

TEST(UtilsTest, GetArrayInputIndex) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  const char* kModuleStr = R"mlir(
!arr = !ifrt.array<tensor<2xf32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func private @zero_in() {
    return
  }
  func.func private @two_in(%x: tensor<2xf32>, %y: tensor<2xf32>) {
    return
  }
  func.func @main(%a: !arr, %b: !arr) attributes {ifrt.function} {
    %ctrl0 = ifrt.Call @zero_in() on devices [0] : () -> ()
    %ctrl1 = ifrt.Call @zero_in() after %ctrl0 on devices [0] : () -> ()
    %ctrl2 = ifrt.Call @two_in(%a, %b) after %ctrl1 on devices [0] : (!arr, !arr) -> ()
    return
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       support::ParseMlirModuleString(kModuleStr, context));
  mlir::func::FuncOp main_func = GetMainFunction(*module);
  llvm::SmallVector<CallOp> calls;
  for (CallOp call_op : main_func.getOps<CallOp>()) {
    calls.push_back(call_op);
  }
  ASSERT_EQ(calls.size(), 3u);

  // Empty inputs range with a control input at operand index 0.
  EXPECT_EQ(GetArrayInputIndex(calls[1].getInputs(), 0), std::nullopt);

  // Two array inputs at operand indices 0 and 1, control input at index 2.
  mlir::OperandRange two_inputs = calls[2].getInputs();
  EXPECT_EQ(GetArrayInputIndex(two_inputs, 0), 0u);
  EXPECT_EQ(GetArrayInputIndex(two_inputs, 1), 1u);
  EXPECT_EQ(GetArrayInputIndex(two_inputs, 2), std::nullopt);

  // Sub-range starting at operand index 1 rejects operand index 0.
  mlir::OperandRange shifted_inputs = two_inputs.drop_front(1);
  EXPECT_EQ(GetArrayInputIndex(shifted_inputs, 0), std::nullopt);
  EXPECT_EQ(GetArrayInputIndex(shifted_inputs, 1), 0u);
  EXPECT_EQ(GetArrayInputIndex(shifted_inputs, 2), std::nullopt);
}

TEST(UtilsTest, IfrtArrayTypeEnforcesShapeMaxNumElements) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  mlir::OpBuilder builder(&context);
  auto emit_error = [&]() {
    return mlir::emitError(mlir::UnknownLoc::get(&context));
  };
  IfrtShardingAttrInterface sharding =
      IfrtUnspecifiedShardingAttr::get(&context);
  IfrtDevicesAttr devices =
      IfrtDevicesAttr::get(&context, /*ids=*/llvm::ArrayRef<int>({0}));

  IfrtArrayType max_allowed = IfrtArrayType::getChecked(
      emit_error, &context,
      mlir::RankedTensorType::get({Shape::kMaxNumElements},
                                  builder.getI8Type()),
      sharding, devices, /*memory_kind_attr=*/mlir::StringAttr(),
      /*layout_attr=*/mlir::StringAttr());
  EXPECT_NE(max_allowed, nullptr);

  IfrtArrayType zero_with_max_allowed = IfrtArrayType::getChecked(
      emit_error, &context,
      mlir::RankedTensorType::get({0, Shape::kMaxNumElements},
                                  builder.getI8Type()),
      sharding, devices, /*memory_kind_attr=*/mlir::StringAttr(),
      /*layout_attr=*/mlir::StringAttr());
  EXPECT_NE(zero_with_max_allowed, nullptr);

  for (mlir::RankedTensorType bad_tensor_type :
       {mlir::RankedTensorType::get({Shape::kMaxNumElements + 1},
                                    builder.getI8Type()),
        mlir::RankedTensorType::get({0, Shape::kMaxNumElements + 1},
                                    builder.getI8Type())}) {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
    IfrtArrayType oversized = IfrtArrayType::getChecked(
        emit_error, &context, bad_tensor_type, sharding, devices,
        /*memory_kind_attr=*/mlir::StringAttr(),
        /*layout_attr=*/mlir::StringAttr());
    EXPECT_EQ(oversized, nullptr);
    EXPECT_THAT(diagnostic_handler.ConsumeStatus(),
                StatusIs(absl::StatusCode::kUnknown,
                         HasSubstr("exceeds maximum supported elements")));
  }
}

TEST(UtilsTest, ShardingFromIfrtArrayTypeValidatesShardingAttrAndDeviceIds) {
  mlir::MLIRContext context;
  support::RegisterMlirDialects(context);
  const char* kModuleStr = R"mlir(
!arr_valid = !ifrt.array<tensor<4x2xf32>,
                         #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1],
                         memory_kind = "pinned_host">
!arr_unspec = !ifrt.array<tensor<4x2xf32>, #ifrt.sharding_unspecified, [0, 1]>
!arr_oob = !ifrt.array<tensor<4x2xf32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
module {
  func.func @main(%a: !arr_valid, %b: !arr_unspec, %c: !arr_oob)
      attributes {ifrt.function} {
    return
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       support::ParseMlirModuleString(kModuleStr, context));
  mlir::func::FuncOp main_func = GetMainFunction(*module);

  NiceMock<MockDevice> device0;
  NiceMock<MockDevice> device1;
  DeviceListRef device_list = BasicDeviceList::Create({&device0, &device1});
  NiceMock<MockClient> client;
  ON_CALL(client, MakeDeviceList)
      .WillByDefault([](absl::Span<Device* const> devs) {
        return BasicDeviceList::Create(devs);
      });

  ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding,
      ShardingFromIfrtArrayType(GetArrayType(main_func.getArgument(0)), &client,
                                device_list));
  const auto* hlo_sharding = xla::ifrt::dyn_cast<HloSharding>(sharding.get());
  ASSERT_NE(hlo_sharding, nullptr);
  EXPECT_EQ(hlo_sharding->xla_hlo_sharding(),
            xla::HloSharding::IotaTile({2, 1}));
  EXPECT_EQ(sharding->memory_kind(), MemoryKind("pinned_host"));
  EXPECT_THAT(sharding->devices()->devices(), ElementsAre(&device0, &device1));

  EXPECT_THAT(ShardingFromIfrtArrayType(GetArrayType(main_func.getArgument(1)),
                                        &client, device_list),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("IfrtShardingParamAttr")));
  EXPECT_THAT(ShardingFromIfrtArrayType(GetArrayType(main_func.getArgument(2)),
                                        &client, device_list),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Logical device id 2 is out of range")));
}

TEST(UtilsTest, GetOpShardingsReturnsOpShardingsOrNulloptForNonHloSharding) {
  NiceMock<MockDevice> device0;
  NiceMock<MockDevice> device1;
  DeviceListRef device_list = BasicDeviceList::Create({&device0, &device1});

  const std::vector<ArraySpec> hlo_specs = {
      ArraySpec{
          /*dtype=*/DType(DType::kF32),
          /*shape=*/Shape({4, 2}),
          /*sharding=*/
          HloSharding::Create(device_list, MemoryKind(),
                              xla::HloSharding::IotaTile({2, 1})),
      },
  };
  std::optional<std::vector<xla::OpSharding>> op_shardings =
      GetOpShardings(hlo_specs, "parameter");
  ASSERT_TRUE(op_shardings.has_value());
  ASSERT_THAT(*op_shardings, SizeIs(1));
  ASSERT_OK_AND_ASSIGN(xla::HloSharding roundtripped,
                       xla::HloSharding::FromProto((*op_shardings)[0]));
  EXPECT_EQ(roundtripped, xla::HloSharding::IotaTile({2, 1}));

  const std::vector<ArraySpec> opaque_specs = {
      ArraySpec{
          /*dtype=*/DType(DType::kF32),
          /*shape=*/Shape({4, 2}),
          /*sharding=*/OpaqueSharding::Create(device_list, MemoryKind()),
      },
  };
  EXPECT_EQ(GetOpShardings(opaque_specs, "output"), std::nullopt);
}

TEST(UtilsTest, GetLayoutsPropagatesStatusAndExtractsLayouts) {
  NiceMock<MockDevice> device0;
  DeviceListRef device_list = BasicDeviceList::Create({&device0});
  auto layout0 = std::make_shared<const xla::PjRtLayout>(
      xla::LayoutUtil::MakeLayout({1, 0}));
  const std::vector<ArraySpec> specs = {
      ArraySpec{
          /*dtype=*/DType(DType::kF32),
          /*shape=*/Shape({2, 2}),
          /*sharding=*/OpaqueSharding::Create(device_list, MemoryKind()),
          /*layout=*/layout0,
      },
      ArraySpec{
          /*dtype=*/DType(DType::kF32),
          /*shape=*/Shape({2, 2}),
          /*sharding=*/OpaqueSharding::Create(device_list, MemoryKind()),
          /*layout=*/nullptr,
      },
  };

  EXPECT_THAT(
      GetLayouts(absl::UnimplementedError("layouts unavailable"), specs),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("layouts unavailable")));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts,
      GetLayouts(absl::OkStatus(), specs));
  EXPECT_THAT(layouts, ElementsAre(layout0, nullptr));
}

TEST(UtilsTest, GetOutputMemoryKindsReturnsInternedViews) {
  std::vector<std::vector<absl::string_view>> memory_kinds;
  {
    NiceMock<MockDevice> device0;
    DeviceListRef device_list = BasicDeviceList::Create({&device0});
    std::string dynamic_kind = absl::StrCat("custom_mem_", "kind");
    std::vector<ArraySpec> out_specs = {
        ArraySpec{
            /*dtype=*/DType(DType::kF32),
            /*shape=*/Shape({2, 2}),
            /*sharding=*/OpaqueSharding::Create(device_list, MemoryKind()),
        },
        ArraySpec{
            /*dtype=*/DType(DType::kF32),
            /*shape=*/Shape({2, 2}),
            /*sharding=*/
            OpaqueSharding::Create(device_list, MemoryKind(dynamic_kind)),
        },
    };
    ASSERT_OK_AND_ASSIGN(memory_kinds, GetOutputMemoryKinds(out_specs));
  }
  ASSERT_THAT(memory_kinds, SizeIs(1));
  EXPECT_THAT(memory_kinds[0], ElementsAre("device", "custom_mem_kind"));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
