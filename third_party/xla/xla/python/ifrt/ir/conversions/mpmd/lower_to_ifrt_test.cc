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

#include "xla/python/ifrt/ir/conversions/mpmd/lower_to_ifrt.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/utils.h"

namespace xla::ifrt::mpmd {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

using LowerToIfrtTest = ::xla::ifrt::test_util::IfrtIrTestBase;

TEST_F(LowerToIfrtTest, GetCompileOptionsPopulatesPerMeshOptions) {
  constexpr absl::string_view kProgram = R"mlir(
    !array0 = !ifrt.array<tensor<2xi32>,
                          #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
    !array1 = !ifrt.array<tensor<2xi32>,
                          #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
    module {
      func.func @main(%arg0: !array0, %arg1: !array1)
          -> (!array0, !array1) attributes {ifrt.function} {
        %0, %ctrl_0 = ifrt.Call @stage1::@main(%arg0) on devices [0, 1]
            {ifrt.mesh_name = "mesh1"} : (!array0) -> !array0
        %1, %ctrl_1 = ifrt.Call @stage2::@main(%arg1) on devices [2, 3]
            {ifrt.mesh_name = "mesh2"} : (!array1) -> !array1
        return %0, %1 : !array0, !array1
      }

      module @stage1 <sym_visibility = "private"> {
        func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32>
            attributes {reserved_hbm_bytes = 1024 : i64} {
          return %arg0 : tensor<2xi32>
        }
      }

      module @stage2 <sym_visibility = "private"> {
        func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
          return %arg0 : tensor<2xi32>
        }
      }
    }
  )mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kProgram));

  absl::flat_hash_map<std::string, const EnvOptionsOverride> overrides = {
      {"mesh2", {{"xla_flag", std::string("val")}}},
  };
  std::vector<int64_t> reserved_bytes_seen;
  ASSERT_OK_AND_ASSIGN(
      CompileOptionsMap options_map,
      GetCompileOptions(
          *module, overrides, /*threshold_for_argument_tupling=*/2000,
          [&](xla::ExecutableBuildOptions&, int64_t reserved_bytes) {
            reserved_bytes_seen.push_back(reserved_bytes);
          }));

  EXPECT_THAT(options_map, UnorderedElementsAre(Key("stage1_mesh_mesh1"),
                                                Key("stage2_mesh_mesh2")));
  EXPECT_THAT(reserved_bytes_seen, ElementsAre(1024));
  EXPECT_THAT(options_map.at("stage2_mesh_mesh2").env_option_overrides,
              ElementsAre(Pair("xla_flag", xla::CompileOptions::OptionOverride(
                                               std::string("val")))));

  mlir::func::FuncOp main_func = GetMainFunction(*module);
  std::vector<std::string> compile_option_keys;
  for (CallOp call_op : main_func.getOps<CallOp>()) {
    mlir::StringAttr key_attr =
        call_op->getAttrOfType<mlir::StringAttr>(kIfrtCompileOptionsKey);
    ASSERT_NE(key_attr, nullptr);
    compile_option_keys.push_back(key_attr.str());
  }
  EXPECT_THAT(compile_option_keys,
              ElementsAre("stage1_mesh_mesh1", "stage2_mesh_mesh2"));
}

TEST_F(LowerToIfrtTest, GetCompileOptionsFailsWhenMeshNameMissing) {
  constexpr absl::string_view kProgram = R"mlir(
    !array0 = !ifrt.array<tensor<2xi32>,
                          #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
    module {
      func.func @main(%arg0: !array0) -> !array0 attributes {ifrt.function} {
        %0, %ctrl_0 = ifrt.Call @stage1::@main(%arg0) on devices [0, 1]
            : (!array0) -> !array0
        return %0 : !array0
      }

      module @stage1 <sym_visibility = "private"> {
        func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
          return %arg0 : tensor<2xi32>
        }
      }
    }
  )mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kProgram));

  absl::flat_hash_map<std::string, const EnvOptionsOverride> overrides;
  EXPECT_THAT(GetCompileOptions(*module, overrides,
                                /*threshold_for_argument_tupling=*/2000,
                                [](xla::ExecutableBuildOptions&, int64_t) {}),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("is missing ifrt.mesh_name attribute")));
}

}  // namespace
}  // namespace xla::ifrt::mpmd
