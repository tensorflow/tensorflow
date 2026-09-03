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

#include "xla/python/ifrt/ir/ifrt_ir_program.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_compile_options.pb.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"
#include "xla/python/ifrt/mlir/fingerprint_utils.h"
#include "xla/service/device_assignment.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

TEST(IfrtIRCompileOptionsTest, ToFromProto) {
  IfrtIrCompileOptionsProto proto;
  int num_devices = 8;
  for (int i = 0; i < num_devices; ++i) {
    proto.add_device_ids(i);
  }
  for (int i = 0; i < 4; ++i) {
    xla::CompileOptions src;
    xla::ExecutableBuildOptions build_option;
    build_option.set_device_assignment(xla::DeviceAssignment(2, 4));
    src.executable_build_options = build_option;
    ASSERT_OK_AND_ASSIGN(CompileOptionsProto compile_options_proto,
                         src.ToProto());
    proto.mutable_compile_option_overrides()->insert(
        {absl::StrCat("key", i), compile_options_proto});
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<IfrtIRCompileOptions> options,
                       IfrtIRCompileOptions::FromProto(proto));

  EXPECT_EQ(options->compile_options_overrides->size(), 4);
  EXPECT_EQ(options->device_assignments.size(), num_devices);
  ASSERT_OK_AND_ASSIGN(IfrtIrCompileOptionsProto from_to_proto,
                       options->ToProto());

  for (int i = 0; i < 4; ++i) {
    std::string key = absl::StrCat("key", i);
    EXPECT_EQ(
        from_to_proto.compile_option_overrides().at(key).SerializeAsString(),
        proto.compile_option_overrides().at(key).SerializeAsString());
  }
  EXPECT_EQ(from_to_proto.compile_option_overrides_size(),
            proto.compile_option_overrides_size());
  EXPECT_EQ(std::vector(from_to_proto.device_ids().begin(),
                        from_to_proto.device_ids().end()),
            std::vector(proto.device_ids().begin(), proto.device_ids().end()));
}

TEST(IfrtIRProgramTest, IfrtIrModuleSameFingerprint) {
  static constexpr absl::string_view kIfrtModule = R"(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = stablehlo.constant dense<1> : tensor<2xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       support::ParseMlirModuleString(kIfrtModule, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module));
  EXPECT_EQ(fp1, fp2);
}

TEST(IfrtIRProgramTest, IfrtIrModuleDifferentDevicesDifferentFingerprints) {
  static constexpr absl::string_view kIfrtModuleDevice0 = R"(
!array0 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array0) -> !array0 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array0) -> !array0
    return %0 : !array0
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      return %arg0 : tensor<2xi32>
    }
  }
}
)";
  static constexpr absl::string_view kIfrtModuleDevice1 = R"(
!array1 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array1) -> !array1 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [1]
        : (!array1) -> !array1
    return %0 : !array1
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      return %arg0 : tensor<2xi32>
    }
  }
}
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module0,
      support::ParseMlirModuleString(kIfrtModuleDevice0, context));
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module1,
      support::ParseMlirModuleString(kIfrtModuleDevice1, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp0, FingerprintModuleOp(*module0));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  EXPECT_NE(fp0, fp1);
}

TEST(IfrtIRProgramTest, IfrtIrModuleDifferentShardingDifferentFingerprints) {
  static constexpr absl::string_view kIfrtModuleSharding1 = R"(
!array = !ifrt.array<tensor<4xi32>, #ifrt.sharding_param<1 to [0] on 2>, [0, 1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity::@main(%arg0) on devices [0, 1]
        : (!array) -> !array
    return %0 : !array
  }

  module @identity {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      return %arg0 : tensor<4xi32>
    }
  }
}
)";
  static constexpr absl::string_view kIfrtModuleSharding2 = R"(
!array = !ifrt.array<tensor<4xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity::@main(%arg0) on devices [0, 1]
        : (!array) -> !array
    return %0 : !array
  }

  module @identity {
    func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      return %arg0 : tensor<4xi32>
    }
  }
}
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module1,
      support::ParseMlirModuleString(kIfrtModuleSharding1, context));
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module2,
      support::ParseMlirModuleString(kIfrtModuleSharding2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_NE(fp1, fp2);
}

TEST(IfrtIRProgramTest, IfrtIrModuleIgnoresDebugInfo) {
  static constexpr absl::string_view kIfrtModule1 = R"(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module @ifrt_mod {
  func.func @main(%arg0: !array loc("arg_loc1")) -> !array
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0]
        : (!array) -> !array loc("call_loc1")
    return %0 : !array loc("return_loc1")
  } loc("func_loc1")

  func.func @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      return %arg0 : tensor<2xi32>
  }
} loc("module_loc1")
)";
  static constexpr absl::string_view kIfrtModule2 = R"(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module @ifrt_mod {
  func.func @main(%arg0: !array loc("arg_loc2")) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0]
        : (!array) -> !array loc("call_loc2")
    return %0 : !array loc("return_loc2")
  } loc("func_loc2")

  func.func @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
} loc("module_loc2")
)";
  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module1,
                       support::ParseMlirModuleString(kIfrtModule1, context));
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module2,
                       support::ParseMlirModuleString(kIfrtModule2, context));
  ASSERT_OK_AND_ASSIGN(uint64_t fp1, FingerprintModuleOp(*module1));
  ASSERT_OK_AND_ASSIGN(uint64_t fp2, FingerprintModuleOp(*module2));
  EXPECT_EQ(fp1, fp2);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
