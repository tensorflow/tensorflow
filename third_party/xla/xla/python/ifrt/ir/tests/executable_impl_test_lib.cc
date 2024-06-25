/* Copyright 2023 The OpenXLA Authors.

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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/compiler.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/ir/tests/executable_impl_test_base.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::tsl::testing::IsOk;
using ::xla::ifrt::test_util::AssertPerShardData;

class IfrtIrExecutableImplTest
    : public test_util::IfrtIrExecutableImplTestBase {};

TEST_F(IfrtIrExecutableImplTest, CallXla) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceList devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LoadedExecutable> loaded_exec,
      client_->GetDefaultCompiler()->Compile(
          std::make_unique<IfrtIRProgram>(*mlir_module),
          std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices))));

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<Array> input,
      CreateArray({data0.data(), data1.data()}, xla::ifrt::Shape({2, 2}),
                  xla::ifrt::DType(xla::ifrt::DType::kS32),
                  xla::ifrt::ShardingParam({2, 1}, {{0}, {2}}), devices));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), /*options=*/{},
                           /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], xla::ifrt::DType(xla::ifrt::DType::kS32),
      xla::ifrt::Shape({1, 2}), {{1, 2}, {3, 4}}, devices));
}

TEST_F(IfrtIrExecutableImplTest, Reshard) {
  std::string source = R"(
module {
  func.func @main(%arg0: !ifrt.array<tensor<2xi32>,
                                     #ifrt.sharding_param<1 to [0] on 1>, [0]>)
      -> !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [1]>
      attributes {ifrt.function} {
    %0 = "ifrt.Reshard"(%arg0)
        : (!ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>)
        -> !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [1]>
    return %0 : !ifrt.array<tensor<2xi32>,
                            #ifrt.sharding_param<1 to [0] on 1>, [1]>
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceList devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LoadedExecutable> loaded_exec,
      client_->GetDefaultCompiler()->Compile(
          std::make_unique<IfrtIRProgram>(*mlir_module),
          std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices))));

  std::vector<int> data = {1, 2};
  TF_ASSERT_OK_AND_ASSIGN(tsl::RCReference<Array> input,
                          CreateArray({data.data()}, xla::ifrt::Shape({2}),
                                      xla::ifrt::DType(xla::ifrt::DType::kS32),
                                      xla::ifrt::ShardingParam({1}, {{0}, {1}}),
                                      DeviceList({devices[0]})));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), /*options=*/{},
                           /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], xla::ifrt::DType(xla::ifrt::DType::kS32),
      xla::ifrt::Shape({2}), {{1, 2}}, DeviceList({devices[1]})));
}

TEST_F(IfrtIrExecutableImplTest, ZeroInput) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module {
  func.func @main() -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @one() on devices [0,1] : () -> !array
    return %0 : !array
  }

  func.func private @one() -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceList devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LoadedExecutable> loaded_exec,
      client_->GetDefaultCompiler()->Compile(
          std::make_unique<IfrtIRProgram>(*mlir_module),
          std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices))));

  TF_ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                          loaded_exec->Execute(/*args=*/{}, /*options=*/{},
                                               /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], xla::ifrt::DType(xla::ifrt::DType::kS32),
      xla::ifrt::Shape({1, 2}), {{1, 1}, {1, 1}}, devices));
}

TEST_F(IfrtIrExecutableImplTest, ZeroOutput) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array) attributes {ifrt.function} {
    %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1] : (!array) -> ()
    return
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceList devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LoadedExecutable> loaded_exec,
      client_->GetDefaultCompiler()->Compile(
          std::make_unique<IfrtIRProgram>(*mlir_module),
          std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices))));

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<Array> input,
      CreateArray({data0.data(), data1.data()}, xla::ifrt::Shape({2, 2}),
                  xla::ifrt::DType(xla::ifrt::DType::kS32),
                  xla::ifrt::ShardingParam({2, 1}, {{0}, {2}}), devices));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), /*options=*/{},
                           /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 0);
}

TEST_F(IfrtIrExecutableImplTest, BufferDonation) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array {ifrt.donated}) -> !array
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceList devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LoadedExecutable> loaded_exec,
      client_->GetDefaultCompiler()->Compile(
          std::make_unique<IfrtIRProgram>(*mlir_module),
          std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices))));

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<Array> input,
      CreateArray({data0.data(), data1.data()}, xla::ifrt::Shape({2, 2}),
                  xla::ifrt::DType(xla::ifrt::DType::kS32),
                  xla::ifrt::ShardingParam({2, 1}, {{0}, {2}}), devices));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), /*options=*/{},
                           /*devices=*/std::nullopt));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], xla::ifrt::DType(xla::ifrt::DType::kS32),
      xla::ifrt::Shape({1, 2}), {{1, 2}, {3, 4}}, devices));

  std::vector<int> data(input->shape().num_elements());
  EXPECT_THAT(input
                  ->CopyToHostBuffer(data.data(), std::nullopt,
                                     ArrayCopySemantics::kAlwaysCopy)
                  .Await(),
              testing::Not(IsOk()));
}

TEST_F(IfrtIrExecutableImplTest, LoadedExecBinding) {
  TF_ASSERT_OK_AND_ASSIGN(DeviceList devices, PickDevices(2));
  std::string mhlo_source = R"(
module {
  func.func @main(
      %arg0: tensor<2x2xi32> {mhlo.sharding = "\08\03\1A\02\02\01\22\02\00\01"})
      -> (tensor<2x2xi32> {mhlo.sharding = "\08\03\1A\02\02\01\22\02\00\01"}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mhlo_module,
                          LoadFromSource(mhlo_source));
  xla::CompileOptions xla_options;
  {
    auto& exec_build_options = xla_options.executable_build_options;
    exec_build_options.set_num_replicas(1);
    exec_build_options.set_num_partitions(2);
    exec_build_options.set_use_spmd_partitioning(true);
    xla::DeviceAssignment device_assignment(1, 2);
    for (auto [logical, device_id] : llvm::enumerate(GetDeviceIds(devices))) {
      device_assignment(0, logical) = device_id.value();
    }
    exec_build_options.set_device_assignment(device_assignment);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LoadedExecutable> child_exec,
      client_->GetDefaultCompiler()->Compile(
          std::make_unique<xla::ifrt::HloProgram>(*mhlo_module),
          std::make_unique<XlaCompileOptions>(std::move(xla_options))));

  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @add_one(%arg0)
        : (!array) -> !array
    return %0 : !array
  }

  ifrt.LoadedExecutable @add_one on devices [0,1] : (!array) -> !array
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  auto options = std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices));
  options->loaded_exec_binding["add_one"] = std::move(child_exec);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LoadedExecutable> loaded_exec,
      client_->GetDefaultCompiler()->Compile(
          std::make_unique<IfrtIRProgram>(*mlir_module), std::move(options)));

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::RCReference<Array> input,
      CreateArray({data0.data(), data1.data()}, xla::ifrt::Shape({2, 2}),
                  xla::ifrt::DType(xla::ifrt::DType::kS32),
                  xla::ifrt::ShardingParam({2, 1}, {{0}, {2}}), devices));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), /*options=*/{},
                           /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], xla::ifrt::DType(xla::ifrt::DType::kS32),
      xla::ifrt::Shape({1, 2}), {{1, 2}, {3, 4}}, devices));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
