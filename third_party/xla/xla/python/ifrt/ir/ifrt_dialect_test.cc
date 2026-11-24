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

#include "xla/python/ifrt/ir/ifrt_dialect.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class IfrtDialectTest : public test_util::IfrtIrTestBase {
 protected:
  IfrtDialectTest()
      : builder_(&mlir_context_),
        loc_(builder_.getUnknownLoc()),
        shape_(mlir::RankedTensorType::get({4, 6}, builder_.getF32Type())),
        sharding_param_(IfrtShardingParamAttr::get(
            &mlir_context_,
            ShardingParam(/*dim_shards=*/{2, 1},
                          /*minor_to_major=*/{/*permutation=*/{0},
                                              /*axis_sizes=*/{2}}))),
        sharding_iface_(sharding_param_),
        devices_(IfrtDevicesAttr::get(&mlir_context_,
                                      /*ids=*/llvm::ArrayRef<int>({0, 1}))) {}

  // Emits a diagnostic at `loc_` for `getChecked` and `verify` calls.
  mlir::InFlightDiagnostic EmitError() { return mlir::emitError(loc_); }

  // Expects `IfrtArrayType::getChecked` to fail with a diagnostic matching
  // `expected_substr`.
  template <typename ShapeT, typename ShardingT, typename DevicesT,
            typename MemoryKindT, typename LayoutT>
  void ExpectGetCheckedError(ShapeT shape, ShardingT sharding_attr,
                             DevicesT devices_attr,
                             MemoryKindT memory_kind_attr, LayoutT layout_attr,
                             absl::string_view expected_substr) {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context_);
    EXPECT_EQ(IfrtArrayType::getChecked(
                  [this]() { return EmitError(); }, &mlir_context_, shape,
                  sharding_attr, devices_attr, memory_kind_attr, layout_attr),
              nullptr);
    EXPECT_THAT(
        diagnostic_handler.ConsumeStatus(),
        StatusIs(absl::StatusCode::kUnknown, HasSubstr(expected_substr)));
  }

  // Expects direct `IfrtArrayType::verify` to fail with a diagnostic matching
  // `expected_substr`.
  void ExpectVerifyError(mlir::StringAttr memory_kind_attr,
                         mlir::StringAttr layout_attr,
                         absl::string_view expected_substr) {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context_);
    EXPECT_TRUE(mlir::failed(IfrtArrayType::verify(
        [this]() { return EmitError(); }, shape_, sharding_iface_, devices_,
        memory_kind_attr, layout_attr)));
    EXPECT_THAT(
        diagnostic_handler.ConsumeStatus(),
        StatusIs(absl::StatusCode::kUnknown, HasSubstr(expected_substr)));
  }

  mlir::OpBuilder builder_;
  mlir::Location loc_;
  mlir::RankedTensorType shape_;
  IfrtShardingParamAttr sharding_param_;
  IfrtShardingAttrInterface sharding_iface_;
  IfrtDevicesAttr devices_;
};

TEST_F(IfrtDialectTest, ArrayTypeCanonicalizesParsedMemoryKindAndLayout) {
  constexpr const char* kModuleStr = R"mlir(
!arr_omitted = !ifrt.array<tensor<4x6xf32>,
                           #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!arr_device = !ifrt.array<tensor<4x6xf32>,
                          #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1],
                          memory_kind = "device", layout = "default">
!arr_pinned = !ifrt.array<tensor<4x6xf32>,
                          #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1],
                          memory_kind = "pinned_host", layout = "{0,1}">
!arr_mk_default = !ifrt.array<tensor<4x6xf32>,
                              #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1],
                              memory_kind = "default">
!arr_auto = !ifrt.array<tensor<4x6xf32>,
                        #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1],
                        layout = "auto">
module {
  func.func @main(%a: !arr_omitted, %b: !arr_device, %c: !arr_pinned,
                  %d: !arr_mk_default, %e: !arr_auto)
      attributes {ifrt.function} {
    return
  }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kModuleStr));
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);
  IfrtArrayType arr_omitted = GetArrayType(main_func.getArgument(0));
  IfrtArrayType arr_device = GetArrayType(main_func.getArgument(1));
  IfrtArrayType arr_pinned = GetArrayType(main_func.getArgument(2));
  IfrtArrayType arr_mk_default = GetArrayType(main_func.getArgument(3));
  IfrtArrayType arr_auto = GetArrayType(main_func.getArgument(4));

  // Omitted and explicit default memory_kind / layout collapse to one type.
  EXPECT_EQ(arr_omitted, arr_device);
  EXPECT_EQ(arr_omitted.getMemoryKindAttr().getValue(), "device");
  EXPECT_EQ(arr_omitted.getLayoutAttr().getValue(), "default");
  EXPECT_TRUE(arr_omitted.HasDefaultMemoryKind());
  EXPECT_TRUE(arr_omitted.HasDefaultLayout());
  EXPECT_TRUE(arr_omitted.HasSameMemoryKind(arr_device));
  EXPECT_TRUE(arr_omitted.HasSameLayout(arr_device));
  EXPECT_TRUE(arr_omitted.MemoryKind().is_default());
  EXPECT_EQ(arr_omitted.LayoutMode().mode, xla::LayoutMode::Mode::kDefault);

  // Non-default memory_kind and user layout are preserved.
  EXPECT_FALSE(arr_pinned.HasDefaultMemoryKind());
  EXPECT_FALSE(arr_pinned.HasDefaultLayout());
  EXPECT_FALSE(arr_pinned.HasSameMemoryKind(arr_omitted));
  EXPECT_FALSE(arr_pinned.HasSameLayout(arr_omitted));
  EXPECT_EQ(arr_pinned.MemoryKind(), MemoryKind("pinned_host"));
  EXPECT_EQ(arr_pinned.LayoutMode().mode,
            xla::LayoutMode::Mode::kUserSpecified);
  EXPECT_EQ(arr_pinned.LayoutMode().ToString(), "{0,1}");

  // `memory_kind = "default"` stays a distinct non-default IFRT MemoryKind,
  // and `layout = "auto"` preserves auto mode.
  EXPECT_FALSE(arr_mk_default.HasDefaultMemoryKind());
  EXPECT_EQ(arr_mk_default.MemoryKind(), MemoryKind("default"));
  EXPECT_NE(arr_mk_default.MemoryKind(), MemoryKind());
  EXPECT_FALSE(arr_auto.HasDefaultLayout());
  EXPECT_FALSE(arr_auto.HasSameLayout(arr_omitted));
  EXPECT_EQ(arr_auto.LayoutMode().mode, xla::LayoutMode::Mode::kAuto);

  // Printing omits canonical defaults and prints non-defaults.
  EXPECT_EQ(mlir::debugString(arr_omitted),
            "!ifrt.array<tensor<4x6xf32>, #ifrt.sharding_param<2x1 to [0] on "
            "2>, [0, 1]>");
  EXPECT_EQ(mlir::debugString(arr_pinned),
            "!ifrt.array<tensor<4x6xf32>, #ifrt.sharding_param<2x1 to [0] on "
            "2>, [0, 1], memory_kind = \"pinned_host\", layout = \"{0,1}\">");
}

TEST_F(IfrtDialectTest, ArrayTypeGetAndGetCheckedCanonicalizeArguments) {
  auto emit_error = [this]() { return EmitError(); };
  mlir::StringAttr device_mk = builder_.getStringAttr("device");
  mlir::StringAttr default_layout = builder_.getStringAttr("default");
  mlir::StringAttr pinned_mk = builder_.getStringAttr("pinned_host");
  mlir::StringAttr user_layout = builder_.getStringAttr("{0,1}");

  IfrtArrayType arr_default =
      IfrtArrayType::get(&mlir_context_, shape_, sharding_iface_, devices_,
                         device_mk, default_layout);
  IfrtArrayType arr_pinned =
      IfrtArrayType::get(&mlir_context_, shape_, sharding_iface_, devices_,
                         pinned_mk, user_layout);

  EXPECT_EQ(IfrtArrayType::get(&mlir_context_, shape_, sharding_iface_,
                               devices_, /*memory_kind_attr=*/nullptr,
                               /*layout_attr=*/nullptr),
            arr_default);
  EXPECT_EQ(IfrtArrayType::get(&mlir_context_, shape_, sharding_param_,
                               devices_, /*memory_kind_attr=*/nullptr,
                               /*layout_attr=*/nullptr),
            arr_default);
  EXPECT_EQ(IfrtArrayType::get(&mlir_context_, shape_, sharding_param_,
                               devices_, pinned_mk, user_layout),
            arr_pinned);
  EXPECT_EQ(IfrtArrayType::getChecked(loc_, &mlir_context_, shape_,
                                      sharding_iface_, devices_,
                                      /*memory_kind_attr=*/nullptr,
                                      /*layout_attr=*/nullptr),
            arr_default);
  EXPECT_EQ(IfrtArrayType::getChecked(emit_error, &mlir_context_, shape_,
                                      sharding_param_, devices_,
                                      /*memory_kind_attr=*/nullptr,
                                      /*layout_attr=*/nullptr),
            arr_default);
  EXPECT_EQ(IfrtArrayType::getChecked(emit_error, &mlir_context_, shape_,
                                      sharding_param_, devices_, pinned_mk,
                                      user_layout),
            arr_pinned);

  IfrtUnspecifiedShardingAttr unspec =
      IfrtUnspecifiedShardingAttr::get(&mlir_context_);
  IfrtArrayType unspec_default = IfrtArrayType::get(
      &mlir_context_, shape_, unspec, devices_, device_mk, default_layout);
  EXPECT_EQ(IfrtArrayType::getChecked(loc_, &mlir_context_, shape_, unspec,
                                      devices_, /*memory_kind_attr=*/nullptr,
                                      /*layout_attr=*/nullptr),
            unspec_default);
}

TEST_F(IfrtDialectTest, ArrayTypeRejectsInvalidAndNonCanonicalAttributes) {
  ExpectGetCheckedError(shape_, sharding_iface_, devices_,
                        builder_.getStringAttr(""), /*layout_attr=*/nullptr,
                        "Memory kind cannot be empty");
  ExpectGetCheckedError(shape_, sharding_iface_, devices_,
                        /*memory_kind_attr=*/nullptr,
                        builder_.getStringAttr(""), "Invalid layout mode");
  ExpectGetCheckedError(shape_, sharding_iface_, devices_,
                        /*memory_kind_attr=*/nullptr,
                        builder_.getStringAttr("{0, 1}"),
                        "Non-canonical layout mode");
  ExpectGetCheckedError(shape_, sharding_iface_, devices_,
                        /*memory_kind_attr=*/nullptr,
                        builder_.getStringAttr("{0,1:S(0)}"),
                        "Non-canonical layout mode");
  ExpectGetCheckedError(shape_, sharding_iface_, devices_,
                        /*memory_kind_attr=*/nullptr,
                        builder_.getStringAttr("{0,1:M(8)}"),
                        "dynamic_shape_metadata_prefix_bytes must be 0");
  ExpectGetCheckedError(/*shape=*/nullptr, sharding_iface_, devices_,
                        /*memory_kind_attr=*/nullptr, /*layout_attr=*/nullptr,
                        "expected non-null IfrtArrayType parameters");
  ExpectGetCheckedError(shape_, /*sharding_attr=*/nullptr, devices_,
                        /*memory_kind_attr=*/nullptr, /*layout_attr=*/nullptr,
                        "expected non-null IfrtArrayType parameters");
  ExpectGetCheckedError(shape_, sharding_iface_, /*devices_attr=*/nullptr,
                        /*memory_kind_attr=*/nullptr, /*layout_attr=*/nullptr,
                        "expected non-null IfrtArrayType parameters");

  ExpectVerifyError(/*memory_kind_attr=*/nullptr,
                    builder_.getStringAttr("default"),
                    "expected non-null IfrtArrayType parameters");
  ExpectVerifyError(builder_.getStringAttr("device"), /*layout_attr=*/nullptr,
                    "expected non-null IfrtArrayType parameters");
}

TEST_F(IfrtDialectTest, Rank0MemorySpaceLayoutPreservesHasSameLayoutSoundness) {
  constexpr const char* kValidModuleStr = R"mlir(
!a = !ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 1>, [0],
                 layout = "{:S(1)}">
module {
  func.func @main(%a: !a) attributes {ifrt.function} { return }
}
)mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> valid_module,
                       LoadFromSource(kValidModuleStr));
  mlir::func::FuncOp main_func =
      valid_module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main_func, nullptr);
  EXPECT_EQ(GetArrayType(main_func.getArgument(0)).getLayoutAttr().getValue(),
            "{:S(1)}");

  constexpr const char* kInvalidModuleStr = R"mlir(
!a = !ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 1>, [0],
                 layout = "{:S(1)P(f32[])}">
!b = !ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 1>, [0],
                 layout = "{:S(1)P(f32[]{:M(8)})}">
module {
  func.func @main(%a: !a, %b: !b) attributes {ifrt.function} { return }
}
)mlir";
  EXPECT_THAT(LoadFromSource(kInvalidModuleStr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("physical_shape")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
