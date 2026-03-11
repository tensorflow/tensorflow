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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_executable_version.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable_test_base.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::Not;
using ::xla::ifrt::test_util::AssertPerShardData;

ExecuteOptions ExecuteOptionsWithFillStatus() {
  ExecuteOptions opts;
  opts.fill_status = true;
  return opts;
}

class IfrtIrLoadedExecutableTest
    : public xla::ifrt::test_util::IfrtIrLoadedExecutableTestBase {
 public:
  // Returns true if the test runs on TPU v4. This is a helper method called
  // from tests that require TPU v4.
  absl::StatusOr<bool> IsUsingTpuV4() {
    TF_ASSIGN_OR_RETURN(DeviceListRef devices, PickDevices(1));
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Topology> topology,
                        client_->GetTopologyForDevices(devices));
    return topology->DeviceDescriptions().front()->device_kind() == "TPU v4";
  }

 protected:
  absl::StatusOr<std::unique_ptr<DeserializeIfrtIRProgramOptions>>
  GetDeserializeOptions(DeviceListRef device_list) {
    std::vector<Device*> device_assignments;
    for (auto device : device_list->devices()) {
      device_assignments.push_back(device);
    }
    return std::make_unique<DeserializeIfrtIRProgramOptions>(
        nullptr, std::nullopt, device_assignments);
  }

  absl::StatusOr<std::unique_ptr<IfrtIrExecutableVersion>>
  DeserializeExecutableVersion(std::string serialized_executable_version_str,
                               DeviceListRef devices) {
    std::vector<DeviceId> device_assignments;
    for (auto device : devices->devices()) {
      device_assignments.push_back(device->Id());
    }
    auto deserialize_options =
        std::make_unique<IfrtIrExecutableVersionDeserializeOptions>(
            client_.get(), device_assignments);

    Serialized serialized_executable_version;
    serialized_executable_version.ParseFromString(
        serialized_executable_version_str);
    absl::StatusOr<std::unique_ptr<ExecutableVersion>>
        deserialized_executable_version = Deserialize<ExecutableVersion>(
            serialized_executable_version, std::move(deserialize_options));
    if (!deserialized_executable_version.ok()) {
      return deserialized_executable_version.status();
    }

    return ToIfrtIrExecutableVersion(
        std::move(*deserialized_executable_version));
  }
};

TEST_F(IfrtIrLoadedExecutableTest, CallXla) {
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  Shape shard_shape({1, 2});
  DType dtype(DType::kS32);
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data0.data(), data1.data()}, Shape({2, 2}),
                                  shard_shape, dtype, devices));

  TF_ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                          loaded_exec->Execute(absl::MakeSpan(&input, 1),
                                               ExecuteOptionsWithFillStatus(),
                                               /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{1, 2}, {3, 4}}, devices));
}

TEST_F(IfrtIrLoadedExecutableTest, ControlDepXla) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @add_one(%arg0) after %ctrl_0 on devices [0,1]
        : (!array) -> !array
    return %1 : !array
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data_shard0 = {0, 1};
  std::vector<int> data_shard1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data_shard0.data(), data_shard1.data()},
                                  Shape({2, 2}), shard_shape, dtype, devices));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1),
                           ExecuteOptionsWithFillStatus(), devices));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{1, 2}, {3, 4}}, devices));
}

TEST_F(IfrtIrLoadedExecutableTest, CopyArrays) {
  std::string source = R"(
!array0 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data = {1, 2};
  DType dtype(DType::kS32);
  Shape shape({2});
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list0,
                          client_->MakeDeviceList({devices->devices()[0]}));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data.data()}, shape, /*shard_shape=*/shape,
                                  dtype, std::move(device_list0)));

  TF_ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                          loaded_exec->Execute(absl::MakeSpan(&input, 1),
                                               ExecuteOptionsWithFillStatus(),
                                               /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                          client_->MakeDeviceList({devices->devices()[1]}));
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shape, {{1, 2}}, std::move(device_list1)));
}

TEST_F(IfrtIrLoadedExecutableTest, Reshard) {
  std::string source = R"(
!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array0) -> (!array1, !array2)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Reshard(%arg0, %arg0)
      : (!array0, !array0) -> (!array1, !array2)
    return %0, %1 : !array1, !array2
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data = {0, 1, 2, 3};
  DType dtype(DType::kS32);
  Shape shape({2, 2});
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list0,
                          client_->MakeDeviceList({devices->devices()[0]}));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data.data()}, shape, /*shard_shape=*/shape,
                                  dtype, std::move(device_list0)));

  TF_ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                          loaded_exec->Execute(absl::MakeSpan(&input, 1),
                                               ExecuteOptionsWithFillStatus(),
                                               /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 2);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, Shape({1, 2}), {{0, 1}, {2, 3}}, devices));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                          client_->MakeDeviceList({devices->devices()[1]}));
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(result.outputs[1], dtype,
                                                  shape, {{0, 1, 2, 3}},
                                                  std::move(device_list1)));
}

TEST_F(IfrtIrLoadedExecutableTest, ZeroInput) {
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(/*args=*/{}, ExecuteOptionsWithFillStatus(),
                           /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(
      AssertPerShardData<int>(result.outputs[0], DType(DType::kS32),
                              Shape({1, 2}), {{1, 1}, {1, 1}}, devices));
}

TEST_F(IfrtIrLoadedExecutableTest, ZeroOutput) {
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data0.data(), data1.data()}, Shape({2, 2}),
                                  shard_shape, DType(DType::kS32), devices));

  TF_ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                          loaded_exec->Execute(absl::MakeSpan(&input, 1),
                                               ExecuteOptionsWithFillStatus(),
                                               /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 0);
}

TEST_F(IfrtIrLoadedExecutableTest, BufferDonation) {
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data0.data(), data1.data()}, Shape({2, 2}),
                                  shard_shape, dtype, devices));

  TF_ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                          loaded_exec->Execute(absl::MakeSpan(&input, 1),
                                               ExecuteOptionsWithFillStatus(),
                                               /*devices=*/std::nullopt));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{1, 2}, {3, 4}}, devices));

  std::vector<int> data(input->shape().num_elements());
  EXPECT_THAT(input
                  ->CopyToHostBuffer(data.data(), std::nullopt,
                                     ArrayCopySemantics::kAlwaysCopy)
                  .Await(),
              Not(IsOk()));
}

TEST_F(IfrtIrLoadedExecutableTest, CallLoadedExecutableOpDonationOverride) {
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data0.data(), data1.data()}, Shape({2, 2}),
                                  shard_shape, dtype, devices));

  ExecuteOptions options;
  options.fill_status = true;
  options.non_donatable_input_indices.insert(0);
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), options,
                           /*devices=*/std::nullopt));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{1, 2}, {3, 4}}, devices));
  // Not using `CopyToHostBuffer` because some implementations don't support it.
  ASSERT_FALSE(input->IsDeleted());
  EXPECT_THAT(input->DisassembleIntoSingleDeviceArrays(
                  ArrayCopySemantics::kAlwaysCopy,
                  SingleDeviceShardSemantics::kAddressableShards),
              IsOk());
}

TEST_F(IfrtIrLoadedExecutableTest, CopyArraysOpDonationOverride) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array {ifrt.donated}) -> !array
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array) -> !array
    return %0 : !array
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data0.data(), data1.data()}, Shape({2, 2}),
                                  shard_shape, dtype, devices));

  ExecuteOptions options;
  options.fill_status = true;
  options.non_donatable_input_indices.insert(0);
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), options,
                           /*devices=*/std::nullopt));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{0, 1}, {2, 3}}, devices));
  // Not using `CopyToHostBuffer` because some implementations don't support it.
  ASSERT_FALSE(input->IsDeleted());
  EXPECT_THAT(input->DisassembleIntoSingleDeviceArrays(
                  ArrayCopySemantics::kAlwaysCopy,
                  SingleDeviceShardSemantics::kAddressableShards),
              IsOk());
}

TEST_F(IfrtIrLoadedExecutableTest, RemapArraysOpDonationOverride) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array0 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %1 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<0, 1, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>]
      : (!array) -> (!array0, !array1)
    return %0, %1 : !array0, !array1
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data_shard0 = {0, 1};
  std::vector<int> data_shard1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data_shard0.data(), data_shard1.data()},
                                  Shape({2, 2}), shard_shape, dtype, devices));

  ExecuteOptions options;
  options.fill_status = true;
  options.non_donatable_input_indices.insert(0);
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), options, devices));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list0,
                          client_->MakeDeviceList({devices->devices()[0]}));
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(result.outputs[0], dtype,
                                                  shard_shape, {{0, 1}},
                                                  std::move(device_list0)));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                          client_->MakeDeviceList({devices->devices()[1]}));
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(result.outputs[1], dtype,
                                                  shard_shape, {{2, 3}},
                                                  std::move(device_list1)));
  // Not using `CopyToHostBuffer` because some implementations don't support it.
  ASSERT_FALSE(input->IsDeleted());
  EXPECT_THAT(input->DisassembleIntoSingleDeviceArrays(
                  ArrayCopySemantics::kAlwaysCopy,
                  SingleDeviceShardSemantics::kAddressableShards),
              IsOk());
}

TEST_F(IfrtIrLoadedExecutableTest,
       RemapArraysOpMultipleInputsDonationOverrideIntroducesCopyArrayOp) {
  // RemapArrays with multiple inputs requires arguments to be donated.
  // If inputs to the program are not donated at runtime, then a CopyArrays op
  // will be inserted for non-donated inputs.
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array0 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array0 {ifrt.donated}, %arg1: !array1 {ifrt.donated})
      -> !array attributes {ifrt.function} {
    %0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      {donated=true}
      : (!array0, !array1) -> (!array)
    return %0 : !array
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list0,
                          client_->MakeDeviceList({devices->devices()[0]}));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input0,
      CreateArray({data0.data()},
                  /*shape=*/shard_shape, shard_shape, dtype, device_list0));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                          client_->MakeDeviceList({devices->devices()[1]}));
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input1,
      CreateArray({data1.data()},
                  /*shape=*/shard_shape, shard_shape, dtype, device_list1));

  ExecuteOptions options;
  options.fill_status = true;
  options.non_donatable_input_indices.insert(1);
  std::vector<ArrayRef> inputs = {input0, input1};
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(inputs), options, devices));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{0, 1}, {2, 3}}, devices));
  ASSERT_TRUE(input0->IsDeleted());
  ASSERT_FALSE(input1->IsDeleted());
}

TEST_F(IfrtIrLoadedExecutableTest, DonateOutputOfCall) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @add_one(%0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array) -> !array
    return %1 : !array
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data_shard0 = {0, 1};
  std::vector<int> data_shard1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data_shard0.data(), data_shard1.data()},
                                  Shape({2, 2}), shard_shape, dtype, devices));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1),
                           ExecuteOptionsWithFillStatus(), devices));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{2, 3}, {4, 5}}, devices));
}

TEST_F(IfrtIrLoadedExecutableTest, RemapFromOneToTwoArrays) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array0 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %1 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<0, 1, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>]
      : (!array) -> (!array0, !array1)
    return %0, %1 : !array0, !array1
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                          LoadFromSource(source));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data_shard0 = {0, 1};
  std::vector<int> data_shard1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data_shard0.data(), data_shard1.data()},
                                  Shape({2, 2}), shard_shape, dtype, devices));

  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1),
                           ExecuteOptionsWithFillStatus(), devices));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list0,
                          client_->MakeDeviceList({devices->devices()[0]}));
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(result.outputs[0], dtype,
                                                  shard_shape, {{0, 1}},
                                                  std::move(device_list0)));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                          client_->MakeDeviceList({devices->devices()[1]}));
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(result.outputs[1], dtype,
                                                  shard_shape, {{2, 3}},
                                                  std::move(device_list1)));
}

TEST_F(IfrtIrLoadedExecutableTest, LoadedExecBinding) {
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  std::string mhlo_source = R"(
module {
  func.func @main(
      %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
      -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
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
      LoadedExecutableRef child_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(std::make_unique<HloProgram>(*mhlo_module),
                           std::make_unique<XlaCompileOptions>(
                               std::move(xla_options), devices))
          .Await());

  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @add_one(%arg0) : (!array) -> !array
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
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(std::make_unique<IfrtIRProgram>(*mlir_module),
                           std::move(options))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input, CreateArray({data0.data(), data1.data()}, Shape({2, 2}),
                                  shard_shape, dtype, devices));

  TF_ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                          loaded_exec->Execute(absl::MakeSpan(&input, 1),
                                               ExecuteOptionsWithFillStatus(),
                                               /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{1, 2}, {3, 4}}, devices));
}

TEST_F(IfrtIrLoadedExecutableTest, ConcurrentCompilation) {
  constexpr absl::string_view source = R"(
!input = !ifrt.array<tensor<1x1xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array0 = !ifrt.array<tensor<1024x1024x512xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x512xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !input) -> (!array0) attributes {ifrt.function} {
    %out, %ctrl = ifrt.Call @generate_data(%arg0) on devices [0]
      : (!input) -> !array0
    %out_0, %ctrl_1 = ifrt.CopyArrays(%out) {donated=true}
      : (!array0) -> !array1
    %out_2, %ctrl_3 = ifrt.Call @identity(%out_0) on devices [1]
      {io_aliases = [array<i32: 0, 0>]} : (!array1) -> !array1
    %out_4, %ctrl_5 = ifrt.CopyArrays(%out_2) {donated=true}
      : (!array1) -> !array0
    %out_6, %ctrl_7 = ifrt.Call @identity(%out_4) on devices [0]
      {io_aliases = [array<i32: 0, 0>]} : (!array0) -> !array0
    return %out_6 : !array0
  }

  func.func private @generate_data(%arg0: tensor<1x1xi32>)
      -> (tensor<1024x1024x512xi32>) {
    %1 = stablehlo.reshape %arg0 : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %2 = "stablehlo.broadcast_in_dim"(%1)
      { broadcast_dimensions = array<i64: 0, 1, 2> }
      : (tensor<1x1x1xi32>) -> tensor<1024x1024x512xi32>
    return %2 : tensor<1024x1024x512xi32>
  }

  func.func private @identity(%arg0: tensor<1024x1024x512xi32>)
      -> tensor<1024x1024x512xi32> {
    return %arg0 : tensor<1024x1024x512xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));

  std::vector<std::unique_ptr<tsl::Thread>> threads;
  for (int i = 0; i < 100; ++i) {
    std::unique_ptr<tsl::Thread> thread =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "compile", [&]() {
              absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> mlir_module =
                  LoadFromSource(source);
              CHECK_OK(mlir_module);
              CHECK_OK(client_->GetDefaultCompiler()
                           ->CompileAndLoad(
                               std::make_unique<IfrtIRProgram>(**mlir_module),
                               std::make_unique<IfrtIRCompileOptions>(
                                   GetDeviceIds(devices)))
                           .Await());
            }));
    threads.push_back(std::move(thread));
  }
}

TEST_F(IfrtIrLoadedExecutableTest, RoundTripExecutableSerialization) {
  if (GetNumDevices() < 4) {
    GTEST_SKIP() << "Insufficient devices to run this test.";
  }
  std::string source = R"(
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [2,3]>
module {
  func.func @main(%arg0: !array1, %arg1: !array2) -> (!array1, !array2)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array1) -> !array1
    %1, %ctrl_1 = ifrt.Call @add_one(%arg1) on devices [2,3]
        : (!array2) -> !array2
    %2, %ctrl_2 = ifrt.Call @add_one(%1) on devices [2,3]
        : (!array2) -> !array2
    return %0, %2 : !array1, !array2
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(4));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  ASSERT_OK_AND_ASSIGN(std::string serialized_executable,
                       ifrt_ir_executable->Serialize());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeserializeIfrtIRProgramOptions> options,
                       GetDeserializeOptions(devices));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<LoadedExecutable> deserialized_executable,
      client_->GetDefaultCompiler()
          ->DeserializeLoadedExecutable(serialized_executable,
                                        std::move(options))
          .Await());
}

TEST_F(IfrtIrLoadedExecutableTest, RoundTripExecutableVersionSerialization) {
  if (GetNumDevices() < 4) {
    GTEST_SKIP() << "Insufficient devices to run this test.";
  }
  std::string source = R"(
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [2,3]>
module {
  func.func @main(%arg0: !array1, %arg1: !array2) -> (!array1, !array2)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array1) -> !array1
    %1, %ctrl_1 = ifrt.Call @add_one(%arg1) on devices [2,3]
        : (!array2) -> !array2
    %2, %ctrl_2 = ifrt.Call @add_one(%1) on devices [2,3]
        : (!array2) -> !array2
    return %0, %2 : !array1, !array2
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(4));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const ExecutableVersion> executable_version,
      ifrt_ir_executable->executable_version());
  ASSERT_OK_AND_ASSIGN(
      Serialized serialized_executable_version,
      Serialize(*executable_version, std::make_unique<SerializeOptions>()));
  std::string serialized_executable_version_str =
      serialized_executable_version.SerializeAsString();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIrExecutableVersion> deserialized_executable_version,
      DeserializeExecutableVersion(serialized_executable_version_str, devices));

  ASSERT_OK(deserialized_executable_version->IsCompatibleWith(
      *client_, *executable_version));
}

TEST_F(IfrtIrLoadedExecutableTest, CallXlaWithDifferentDevices) {
  if (GetNumDevices() < 4) {
    GTEST_SKIP() << "Insufficient devices to run this test.";
  }
  std::string source = R"(
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [2,3]>
module {
  func.func @main(%arg0: !array1, %arg1: !array2) -> (!array1, !array2)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array1) -> !array1
    %1, %ctrl_1 = ifrt.Call @add_one(%arg1) on devices [2,3]
        : (!array2) -> !array2
    %2, %ctrl_2 = ifrt.Call @add_one(%1) on devices [2,3]
        : (!array2) -> !array2
    return %0, %2 : !array1, !array2
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(4));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data1_shard0 = {0, 1};
  std::vector<int> data1_shard1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shape({2, 2});
  Shape shard_shape({1, 2});
  ASSERT_OK_AND_ASSIGN(
      DeviceListRef first_two_devices,
      client_->MakeDeviceList({devices->devices()[0], devices->devices()[1]}));
  ASSERT_OK_AND_ASSIGN(
      DeviceListRef last_two_devices,
      client_->MakeDeviceList({devices->devices()[2], devices->devices()[3]}));
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input1,
      CreateArray({data1_shard0.data(), data1_shard1.data()}, shape,
                  shard_shape, dtype, first_two_devices));
  std::vector<int> data2_shard0 = {10, 11};
  std::vector<int> data2_shard1 = {12, 13};
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input2,
      CreateArray({data2_shard0.data(), data2_shard1.data()}, shape,
                  shard_shape, dtype, last_two_devices));

  std::vector<ArrayRef> inputs = {input1, input2};
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(inputs),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 2);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(result.outputs[0], dtype,
                                                  shard_shape, {{1, 2}, {3, 4}},
                                                  first_two_devices));
  ASSERT_NO_FATAL_FAILURE(
      AssertPerShardData<int>(result.outputs[1], dtype, shard_shape,
                              {{12, 13}, {14, 15}}, last_two_devices));
}

TEST_F(IfrtIrLoadedExecutableTest, CallStableHlo) {
  // Test that verifies that StableHLO atom programs are compiled ok.
  std::string source = R"(
!array = !ifrt.array<tensor<4xf32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<4xf32>, #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @mjit_f {
  func.func public @main(%arg0: !array) -> !array1 attributes {ifrt.function} {
    %out_0, %ctrl_0 = ifrt.Call @stage1(%arg0) on devices [0] : (!array) -> !array
    %0, %ctrl = ifrt.CopyArrays(%out_0) : (!array) -> !array1
    %out_1, %ctrl_1 = ifrt.Call @stage2(%0) on devices [1] : (!array1) -> !array1
    return %out_1 : !array1
  }

  func.func private @stage1(%arg0: tensor<4xf32> {mhlo.layout_mode = "default"}) -> (tensor<4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func private @stage2(%arg0: tensor<4xf32> {mhlo.layout_mode = "default"}) -> (tensor<4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK(client_->GetDefaultCompiler()
                ->CompileAndLoad(std::move(program),
                                 std::make_unique<IfrtIRCompileOptions>(
                                     GetDeviceIds(devices)))
                .Await()
                .status());
}

TEST_F(IfrtIrLoadedExecutableTest, CopyArraysTpuToTpu) {
  // Test that verifies that we can copy arrays between different devices.
  // Note that this test passes the same argument twice to CopyArrays in order
  // to verify that the array is not removed incorrectly from the IFRT IR
  // program interpreter environment.
  std::string source = R"(
#tpu0 = #ifrt<devices[0]>
#tpu1 = #ifrt<devices[1]>
!array0 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, #tpu0>
!array1 = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, #tpu1>
module {
  func.func @main(%arg0: !array0) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.CopyArrays(%arg0, %arg0)
      : (!array0, !array0) -> (!array1, !array1)
    return %0, %1 : !array1, !array1
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data = {1, 2};
  DType dtype(DType::kS32);
  Shape shape({2});
  ASSERT_OK_AND_ASSIGN(DeviceListRef first_device,
                       client_->MakeDeviceList({devices->devices()[0]}));
  ASSERT_OK_AND_ASSIGN(DeviceListRef second_device,
                       client_->MakeDeviceList({devices->devices()[1]}));
  ASSERT_OK_AND_ASSIGN(ArrayRef input,
                       CreateArray({data.data()}, shape, /*shard_shape=*/shape,
                                   dtype, first_device));

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 2);
  for (int i = 0; i < result.outputs.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
        result.outputs[i], dtype, shape, {{1, 2}}, second_device));
  }
}

TEST_F(IfrtIrLoadedExecutableTest, CopyArraysCpuToTpu) {
  if (!PickDevices(1, "cpu").ok()) {
    GTEST_SKIP() << "Test requires at least one cpu device";
  }

  std::string source = R"(
#sharding = #ifrt.sharding_param<1x1 to [0] on 1>
#tpu = #ifrt<devices[0]>
#cpu = #ifrt<devices[1]>
module {
  func.func @main(%arg0: !ifrt.array<tensor<2x1xi32>, #sharding, #cpu>)
      -> !ifrt.array<tensor<2x1xi32>, #sharding, #tpu>
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0)
        : (!ifrt.array<tensor<2x1xi32>, #sharding, #cpu>)
        -> !ifrt.array<tensor<2x1xi32>, #sharding, #tpu>
    return %0 : !ifrt.array<tensor<2x1xi32>, #sharding, #tpu>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef device, PickDevices(1));
  ASSERT_OK_AND_ASSIGN(DeviceListRef cpu_device, PickDevices(1, "cpu"));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices,
                       client_->MakeDeviceList(
                           {device->devices()[0], cpu_device->devices()[0]}));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data = {0, 1};
  DType dtype(DType::kS32);
  Shape shape({2, 1});
  ASSERT_OK_AND_ASSIGN(ArrayRef input,
                       CreateArray({data.data()}, shape, /*shard_shape=*/shape,
                                   dtype, cpu_device));

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(result.outputs[0], dtype,
                                                  shape, {{0, 1}}, device));
}

TEST_F(IfrtIrLoadedExecutableTest, LoadedExecBindingWithDiffNumInputsErrors) {
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  std::string stablehlo_source = R"(
module {
  func.func @main(
      %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
      -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> stablehlo_module,
                       LoadFromSource(stablehlo_source));
  auto xla_options =
      std::make_unique<XlaCompileOptions>(xla::CompileOptions(), devices);
  {
    auto& exec_build_options =
        xla_options->compile_options.executable_build_options;
    exec_build_options.set_num_replicas(1);
    exec_build_options.set_num_partitions(2);
    exec_build_options.set_use_spmd_partitioning(true);
    xla::DeviceAssignment device_assignment(1, 2);
    for (auto [logical_id, device] : llvm::enumerate(devices->devices())) {
      auto device_id = device->Id();
      device_assignment(0, logical_id) = device_id.value();
    }
    exec_build_options.set_device_assignment(device_assignment);
  }
  ASSERT_OK_AND_ASSIGN(
      auto executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(std::make_unique<HloProgram>(*stablehlo_module),
                           std::move(xla_options))
          .Await());

  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @add_args(%arg0, %arg0)
        : (!array, !array) -> !array
    return %0 : !array
  }

  ifrt.LoadedExecutable @add_args on devices [0,1] : (!array, !array) -> !array
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  auto compile_options =
      std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices));
  ASSERT_TRUE(compile_options->loaded_exec_binding
                  .try_emplace("add_args", std::move(executable))
                  .second);
  EXPECT_THAT(
      client_->GetDefaultCompiler()
          ->CompileAndLoad(std::move(program), std::move(compile_options))
          .Await(),
      StatusIs(absl::StatusCode::kUnknown,
               HasSubstr("expects an executable with 2 inputs, but was "
                         "bound to an executable with 1 input")));
}

TEST_F(IfrtIrLoadedExecutableTest, LoadedExecBindingWithDiffShardingErrors) {
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  std::string stablehlo_source = R"(
module {
  func.func @main(
      %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
      -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> stablehlo_module,
                       LoadFromSource(stablehlo_source));
  auto xla_options =
      std::make_unique<XlaCompileOptions>(xla::CompileOptions(), devices);
  {
    auto& exec_build_options =
        xla_options->compile_options.executable_build_options;
    exec_build_options.set_num_replicas(1);
    exec_build_options.set_num_partitions(2);
    exec_build_options.set_use_spmd_partitioning(true);
    xla::DeviceAssignment device_assignment(1, 2);
    for (auto [logical_id, device] : llvm::enumerate(devices->devices())) {
      auto device_id = device->Id();
      device_assignment(0, logical_id) = device_id.value();
    }
    exec_build_options.set_device_assignment(device_assignment);
  }
  ASSERT_OK_AND_ASSIGN(
      auto executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(std::make_unique<HloProgram>(*stablehlo_module),
                           std::move(xla_options))
          .Await());

  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array_replicated = !ifrt.array<tensor<2x2xi32>,
                                #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array) -> !array_replicated attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @add_one(%arg0)
        : (!array) -> !array_replicated
    return %0 : !array_replicated
  }

  ifrt.LoadedExecutable @add_one on devices [0,1]
      : (!array) -> !array_replicated
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto compile_options =
      std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices));
  ASSERT_TRUE(compile_options->loaded_exec_binding
                  .try_emplace("add_one", std::move(executable))
                  .second);
  EXPECT_THAT(
      client_->GetDefaultCompiler()
          ->CompileAndLoad(std::make_unique<IfrtIRProgram>(*mlir_module),
                           std::move(compile_options))
          .Await(),
      StatusIs(absl::StatusCode::kUnknown,
               HasSubstr("expects an executable with output #0 sharding "
                         "{devices=[2,1]<=[2]}, but was bound to an "
                         "executable with sharding {replicated}")));
}

TEST_F(IfrtIrLoadedExecutableTest, UsingPartiallyDonatedArgThrowsError) {
  if (client_->platform_name() != xla::TpuName()) {
    GTEST_SKIP() << "Test CHECK fails in pjrt_stream_executor_client.h.";
  }
  // Verifies that an error is thrown if a shard of an array is first aliased
  // and then donated, followed by usage of the initial array is used.
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array0 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array {ifrt.donated}) -> !array
      attributes {ifrt.function} {
    %0 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array) -> !array0
    %2, %ctrl_2 = ifrt.Call @add_two(%0) on devices [0]
      {io_aliases = [array<i32: 0, 0>]} : (!array0) -> !array0
    %3, %ctrl_3 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %3 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }

  func.func private @add_two(%arg0: tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = stablehlo.constant dense<2> : tensor<1x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data_shard0 = {0, 1};
  std::vector<int> data_shard1 = {2, 3};
  Shape shard_shape({1, 2});
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input,
      CreateArray({data_shard0.data(), data_shard1.data()}, Shape({2, 2}),
                  shard_shape, DType(DType::kS32), devices));

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  EXPECT_THAT(
      result.status.Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid buffer passed to Execute() as argument 0")));
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_THAT(
      result.outputs[0]->GetReadyFuture().Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid buffer passed to Execute() as argument 0")));
}

TEST_F(IfrtIrLoadedExecutableTest, DonatingTwiceAliasedBufferThrowsError) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array0 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array0, !array0)
      attributes {ifrt.function} {
    %0 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array) -> !array0
    %1 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
      : (!array) -> !array0
    %2, %3, %ctrl_2 = ifrt.Call @callee(%0, %1) on devices [0]
      {io_aliases = [array<i32: 0, 0>, array<i32: 1, 1>]}
      : (!array0, !array0) -> (!array0, !array0)
    return %2, %3 : !array0, !array0
  }

  func.func private @callee(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2xi32>)
      -> (tensor<1x2xi32>, tensor<1x2xi32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1x2xi32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<1x2xi32>
    return %0, %1 : tensor<1x2xi32>, tensor<1x2xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data_shard0 = {0, 1};
  std::vector<int> data_shard1 = {2, 3};
  Shape shard_shape({1, 2});
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input,
      CreateArray({data_shard0.data(), data_shard1.data()}, Shape({2, 2}),
                  shard_shape, DType(DType::kS32), devices));

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  EXPECT_THAT(
      result.status.Await(),
      AnyOf(
          StatusIs(absl::StatusCode::kInvalidArgument,
                   HasSubstr(
                       "Attempt to donate the same buffer twice in Execute()")),
          StatusIs(
              absl::StatusCode::kNotFound,
              HasSubstr(
                  "The buffer was likely previously donated or deleted"))));
  ASSERT_EQ(result.outputs.size(), 2);
  EXPECT_THAT(
      result.outputs[0]->GetReadyFuture().Await(),
      AnyOf(
          StatusIs(absl::StatusCode::kInvalidArgument,
                   HasSubstr(
                       "Attempt to donate the same buffer twice in Execute()")),
          StatusIs(
              absl::StatusCode::kNotFound,
              HasSubstr(
                  "The buffer was likely previously donated or deleted"))));
}

TEST_F(IfrtIrLoadedExecutableTest, RemapFromTwoToOneArray) {
  std::string source = R"(
!array0 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!array2 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module {
  func.func @main(%arg0: !array0 {ifrt.donated}, %arg1: !array1 {ifrt.donated})
      -> !array2 attributes {ifrt.function} {
    %0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      {donated=true} : (!array0, !array1) -> (!array2)
    return %0 : !array2
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> array_0_data = {0, 1};
  DType dtype(DType::kS32);
  Shape shape({1, 2});
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list0,
                       client_->MakeDeviceList({devices->devices()[0]}));
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input0,
      CreateArray({array_0_data.data()}, shape, /*shard_shape=*/shape, dtype,
                  std::move(device_list0)));
  std::vector<int> array_1_data = {2, 3};
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                       client_->MakeDeviceList({devices->devices()[1]}));
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input1,
      CreateArray({array_1_data.data()}, shape, /*shard_shape=*/shape, dtype,
                  std::move(device_list1)));

  std::vector<ArrayRef> inputs = {input0, input1};
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(inputs),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shape, {{0, 1}, {2, 3}}, devices));

  // Check that the inputs got donated.
  EXPECT_TRUE(input0->IsDeleted());
  EXPECT_TRUE(input1->IsDeleted());
}

TEST_F(IfrtIrLoadedExecutableTest, GetParameterAndOutputLayouts) {
  if (GetNumDevices() < 4) {
    GTEST_SKIP() << "Insufficient devices to run this test.";
  }
  // Verifies that the layouts of the parameters and outputs are correctly
  // populated. The test verifies the following scenarios:
  // 1. Atom program with auto layout for both the input and output.
  // 2. Atom program with a custom layout.
  // 3. IFRT IR program argument that is just returned.
  // 4. Unused IFRT IR program argument.
  // 5. Atom program output that is returned multiple times.

  // TODO(b/382761415): Update test when layouts are populated in IFRT IR
  // array type.
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x2 to [0] on 4>, [0, 1, 2, 3]>
module @auto_layout {
  func.func public @main(%arg0: !array, %arg1: !array, %arg2: !array)
      -> (!array, !array, !array, !array) attributes {ifrt.function} {
    %out_0, %ctrl_0 = ifrt.Call @transpose::@main(%arg0) on devices [0, 1, 2, 3]
      : (!array) -> !array
    %out_1, %ctrl_1 = ifrt.Call @transpose_w_custom_layout::@main(%arg0)
      on devices [0, 1, 2, 3] : (!array) -> !array
    return %out_0, %out_0, %out_1, %arg1 : !array, !array, !array, !array
  }
  module @transpose attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {mhlo.layout_mode = "auto"})
        -> (tensor<2x2xi32> {mhlo.layout_mode = "auto"}) {
      %0 = stablehlo.transpose %arg0, dims = [1, 0]
        : (tensor<2x2xi32>) -> tensor<2x2xi32>
      return %0 : tensor<2x2xi32>
    }
  }
  module @transpose_w_custom_layout attributes {sym_visibility = "private"} {
    func.func public @main(%arg0: tensor<2x2xi32>
        {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"})
        -> (tensor<2x2xi32> {mhlo.layout_mode = "auto"}) {
      %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xi32>)
        -> tensor<2x2xi32>
      %1 = stablehlo.custom_call @Sharding(%0) {
        backend_config = "",
        mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"}
        : (tensor<2x2xi32>) -> tensor<2x2xi32>
      %2 = stablehlo.custom_call @LayoutConstraint(%1) {
        backend_config = "",
        operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
        result_layouts = [dense<[0, 1]> : tensor<2xindex>]}
        : (tensor<2x2xi32>) -> tensor<2x2xi32>
      return %2 : tensor<2x2xi32>
    }
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(4));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());
  ASSERT_OK_AND_ASSIGN(auto parameter_layouts,
                       ifrt_ir_executable->GetParameterLayouts());
  // Note `GetDefaultPjRtLayout` takes a sharded shape.
  ASSERT_OK_AND_ASSIGN(
      auto default_layout,
      client_->GetDefaultPjRtLayout(DType(DType::kS32), {1, 1},
                                    devices->devices()[0], MemoryKind()));
  ASSERT_EQ(parameter_layouts.size(), 3);
  ASSERT_EQ(*default_layout, *parameter_layouts[0]);
  ASSERT_EQ(*default_layout, *parameter_layouts[1]);
  ASSERT_EQ(*default_layout, *parameter_layouts[2]);
  ASSERT_OK_AND_ASSIGN(auto output_layouts,
                       ifrt_ir_executable->GetOutputLayouts());
  ASSERT_EQ(output_layouts.size(), 4);
  ASSERT_EQ(*default_layout, *output_layouts[0]);
  ASSERT_EQ(*default_layout, *output_layouts[1]);
  ASSERT_EQ("{0,1:T(1,128)}", output_layouts[2]->ToString());
  ASSERT_EQ(*default_layout, *output_layouts[3]);
}

TEST_F(IfrtIrLoadedExecutableTest, NonDonatablePinnedHostInput) {
  std::string source = R"(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                     memory_kind = "pinned_host">
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_exec,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::make_unique<IfrtIRProgram>(*mlir_module),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data0 = {0, 1};
  std::vector<int> data1 = {2, 3};
  DType dtype(DType::kS32);
  Shape shard_shape({1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef input,
      CreateArray({data0.data(), data1.data()}, Shape({2, 2}), shard_shape,
                  dtype, devices, MemoryKind("pinned_host")));
  ExecuteOptions options;
  options.fill_status = true;
  options.non_donatable_input_indices.insert(0);
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_exec->Execute(absl::MakeSpan(&input, 1), options,
                           /*devices=*/std::nullopt));
  TF_ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);

  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(
      result.outputs[0], dtype, shard_shape, {{1, 2}, {3, 4}}, devices));
  // Verify that the input was not donated, and that it has the same data as
  // initially set.
  ASSERT_NO_FATAL_FAILURE(AssertPerShardData<int>(input, dtype, shard_shape,
                                                  {{0, 1}, {2, 3}}, devices));
}

TEST_F(IfrtIrLoadedExecutableTest, VerifyDeletesDispatchedInOrder) {
  // The test creates the following behavior:
  // Device 0: generate_data() -> 20GB                    identity() -> 20GB
  //                                  \                  /
  // Device 1:                         identity() -> 20GB
  // The test verifies that the output of the first Execute() on device 0 is
  // deleted before the second Execute() on device 0. If this does not hold,
  // then the test will OOM.
  if (!IsUsingTpuV4().value_or(false)) {
    GTEST_SKIP() << "Test requires TPU v4.";
  }
  std::string source = R"(
!input = !ifrt.array<tensor<1x1xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array0 = !ifrt.array<tensor<1024x1024x5120xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x5120xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !input) -> (!array0) attributes {ifrt.function} {
    %out, %ctrl = ifrt.Call @generate_data(%arg0) on devices [0]
      : (!input) -> !array0
    %out_0, %ctrl_1 = ifrt.CopyArrays(%out) {donated=true}
      : (!array0) -> !array1
    %out_2, %ctrl_3 = ifrt.Call @identity(%out_0) on devices [1]
      {io_aliases = [array<i32: 0, 0>]} : (!array1) -> !array1
    %out_4, %ctrl_5 = ifrt.CopyArrays(%out_2) {donated=true}
      : (!array1) -> !array0
    %out_6, %ctrl_7 = ifrt.Call @identity(%out_4) on devices [0]
      {io_aliases = [array<i32: 0, 0>]} : (!array0) -> !array0
    return %out_6 : !array0
  }

  func.func private @generate_data(%arg0: tensor<1x1xi32>)
      -> (tensor<1024x1024x5120xi32>) {
    %1 = stablehlo.reshape %arg0 : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %2 = "stablehlo.broadcast_in_dim"(%1)
      { broadcast_dimensions = array<i64: 0, 1, 2> }
      : (tensor<1x1x1xi32>) -> tensor<1024x1024x5120xi32>
    return %2 : tensor<1024x1024x5120xi32>
  }

  func.func private @identity(%arg0: tensor<1024x1024x5120xi32>)
      -> tensor<1024x1024x5120xi32> {
    return %arg0 : tensor<1024x1024x5120xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());
  std::vector<int> input_data = {1};
  Shape shape({1, 1});
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                       client_->MakeDeviceList({devices->devices().front()}));
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input,
      CreateArray({input_data.data()}, shape, /*shard_shape=*/shape,
                  DType(DType::kS32), std::move(device_list)));

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  ASSERT_EQ(result.outputs.size(), 1);
  ASSERT_OK(result.outputs[0]->GetReadyFuture().Await());
}

TEST_F(IfrtIrLoadedExecutableTest, VerifyUnusedArraysDeleteInOrder) {
  // This test verifies that unused output arrays are deleted in order.
  // The test dispatches several time a computation that creates a 20GB array,
  // which is unused by other computations. Each computation has an artificial
  // runtime of at least 4 seconds implemented using
  // `xla_tpu_alloc_buffer_vdelay`. This is done to maximize the chances that
  // all executions are enqueued before the first one finishes.
  // Note: `wait_4sec_and_add` invokes the AllocateBuffer custom_call several
  // times because each call introduces accurate delays of up to 1-2 seconds,
  // but not longer.
  // Note: This test expects the test target to set the flag
  // `--xla_tpu_alloc_buffer_vdelay` to the frequency of the hardware used s.t.
  // each AllocateBuffer call introduces a delay of 1 second.
  if (!IsUsingTpuV4().value_or(false)) {
    GTEST_SKIP() << "Test requires TPU v4.";
  }

  std::string source = R"(
!array0 = !ifrt.array<tensor<1x1xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x5120xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array0) -> (!array0, !array1)
        attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Call @wait_4sec_and_add(%arg0) on devices [0]
        : (!array0) -> (!array0, !array1)
    %2, %3, %ctrl_1 = ifrt.Call @wait_4sec_and_add(%0) on devices [0]
        : (!array0) -> (!array0, !array1)
    %4, %5, %ctrl_2 = ifrt.Call @wait_4sec_and_add(%2) on devices [0]
        : (!array0) -> (!array0, !array1)
    %6, %7, %ctrl_3 = ifrt.Call @wait_4sec_and_add(%4) on devices [0]
        : (!array0) -> (!array0, !array1)
    return %6, %7 : !array0, !array1
  }

  func.func private @wait_4sec_and_add(%arg0: tensor<1x1xi32>)
        -> (tensor<1x1xi32>, tensor<1024x1024x5120xi32>) {
    %0 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x1xi32>
    %2 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %3 = stablehlo.add %1, %2 : tensor<1x1xi32>
    %4 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %5 = stablehlo.add %3, %4 : tensor<1x1xi32>
    %6 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %7 = stablehlo.add %5, %6 : tensor<1x1xi32>
    %8 = stablehlo.reshape %7 : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %9 = "stablehlo.broadcast_in_dim"(%8)
      { broadcast_dimensions = array<i64: 0, 1, 2> }
      : (tensor<1x1x1xi32>) -> tensor<1024x1024x5120xi32>
    return %7, %9 : tensor<1x1xi32>, tensor<1024x1024x5120xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(1));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data = {42};
  Shape shape({1, 1});
  ASSERT_OK_AND_ASSIGN(ArrayRef input,
                       CreateArray({data.data()}, shape, /*shard_shape=*/shape,
                                   DType(DType::kS32), devices));
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  // Check that no OOM happens.
  ASSERT_OK(result.outputs[0]->GetReadyFuture().Await());
  ASSERT_OK(result.outputs[1]->GetReadyFuture().Await());
}

TEST_F(IfrtIrLoadedExecutableTest, VerifyRunaheadWithAliasingDoesntOOM) {
  // This test verifies that we can enqueue many executions, each with a peak
  // memory usage of 20GB. It creates a chain of computations where each
  // computation aliases a 20GB input array.
  // See `VerifyUnusedArraysDeleteInOrder` for an explanation on how we maximize
  // the changes that the dispatches complete before the first computation.
  // Note: This test expects the test target to set the flag
  // `--xla_tpu_alloc_buffer_vdelay` to the frequency of the hardware used s.t.
  // each AllocateBuffer call introduces a delay of 1 second.
  if (!IsUsingTpuV4().value_or(false)) {
    GTEST_SKIP() << "Test requires TPU v4.";
  }

  std::string source = R"(
!array0 = !ifrt.array<tensor<1x1xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x5120xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array0) -> (!array0, !array1)
        attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @generate(%arg0) on devices [0]
        : (!array0) -> (!array1)
    %2, %3, %ctrl_1 = ifrt.Call @wait_4sec_and_add(%arg0, %0) on devices [0]
        {io_aliases = [array<i32: 1, 1>]}
        : (!array0, !array1) -> (!array0, !array1)
    %4, %5, %ctrl_2 = ifrt.Call @wait_4sec_and_add(%2, %3) on devices [0]
        {io_aliases = [array<i32: 1, 1>]}
        : (!array0, !array1) -> (!array0, !array1)
    %6, %7, %ctrl_3 = ifrt.Call @wait_4sec_and_add(%4, %5) on devices [0]
        {io_aliases = [array<i32: 1, 1>]}
        : (!array0, !array1) -> (!array0, !array1)
    return %6, %7 : !array0, !array1
  }

  func.func private @generate(%arg0: tensor<1x1xi32>)
        -> tensor<1024x1024x5120xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %1 = "stablehlo.broadcast_in_dim"(%0)
      { broadcast_dimensions = array<i64: 0, 1, 2> }
      : (tensor<1x1x1xi32>) -> tensor<1024x1024x5120xi32>
    return %1 : tensor<1024x1024x5120xi32>
  }

  func.func private @wait_4sec_and_add(
        %arg0: tensor<1x1xi32>, %arg1: tensor<1024x1024x5120xi32>)
        -> (tensor<1x1xi32>, tensor<1024x1024x5120xi32>) {
    %0 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x1xi32>
    %2 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %3 = stablehlo.add %1, %2 : tensor<1x1xi32>
    %4 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %5 = stablehlo.add %3, %4 : tensor<1x1xi32>
    %6 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %7 = stablehlo.add %5, %6 : tensor<1x1xi32>
    %8 = stablehlo.constant dense<1> : tensor<1024x1024x5120xi32>
    %9 = stablehlo.add %arg1, %8 : tensor<1024x1024x5120xi32>
    return %7, %9 : tensor<1x1xi32>, tensor<1024x1024x5120xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(1));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data = {42};
  Shape shape({1, 1});
  ASSERT_OK_AND_ASSIGN(ArrayRef input,
                       CreateArray({data.data()}, shape, /*shard_shape=*/shape,
                                   DType(DType::kS32), devices));
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  // Check that no OOM happens.
  ASSERT_OK(result.outputs[0]->GetReadyFuture().Await());
  ASSERT_OK(result.outputs[1]->GetReadyFuture().Await());
}

TEST_F(IfrtIrLoadedExecutableTest, VerifyRunaheadWithoutAliasingDoesntOOM) {
  // This test creates a chain of computations, each with a peak memory usage
  // of 26GB. Each computation takes at least 4 seconds to run and does not
  // alias any inputs. The test dispatches the executions as fast as possible
  // and verifies that no OOM happens.
  // See `VerifyUnusedArraysDeleteInOrder` for an explanation on how we maximize
  // the chances of the executions being enqueued before the first computation.
  // Note: This test expects the test target to set the flag
  // `--xla_tpu_alloc_buffer_vdelay` to the frequency of the hardware used s.t.
  // each AllocateBuffer call introduces a delay of 1 second.
  if (!IsUsingTpuV4().value_or(false)) {
    GTEST_SKIP() << "Test requires TPU v4.";
  }

  std::string source = R"(
!array0 = !ifrt.array<tensor<1x1xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x3328xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array0) -> (!array0, !array1)
        attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @generate(%arg0) on devices [0]
        : (!array0) -> (!array1)
    %2, %3, %ctrl_1 = ifrt.Call @wait_4sec_and_add(%arg0, %0) on devices [0]
        : (!array0, !array1) -> (!array0, !array1)
    %4, %5, %ctrl_2 = ifrt.Call @wait_4sec_and_add(%2, %3) on devices [0]
        : (!array0, !array1) -> (!array0, !array1)
    %6, %7, %ctrl_3 = ifrt.Call @wait_4sec_and_add(%4, %5) on devices [0]
        : (!array0, !array1) -> (!array0, !array1)
    %8, %9, %ctrl_4 = ifrt.Call @wait_4sec_and_add(%6, %7) on devices [0]
        : (!array0, !array1) -> (!array0, !array1)
    return %8, %9 : !array0, !array1
  }

  func.func private @generate(%arg0: tensor<1x1xi32>)
        -> tensor<1024x1024x3328xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %1 = "stablehlo.broadcast_in_dim"(%0)
      { broadcast_dimensions = array<i64: 0, 1, 2> }
      : (tensor<1x1x1xi32>) -> tensor<1024x1024x3328xi32>
    return %1 : tensor<1024x1024x3328xi32>
  }

  func.func private @wait_4sec_and_add(
        %arg0: tensor<1x1xi32>, %arg1: tensor<1024x1024x3328xi32>)
        -> (tensor<1x1xi32>, tensor<1024x1024x3328xi32>) {
    %0 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x1xi32>
    %2 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %3 = stablehlo.add %1, %2 : tensor<1x1xi32>
    %4 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %5 = stablehlo.add %3, %4 : tensor<1x1xi32>
    %6 = "stablehlo.custom_call"() {
        call_target_name = "AllocateBuffer", has_side_effect = false}
        : () -> tensor<1x1xi32>
    %7 = stablehlo.add %5, %6 : tensor<1x1xi32>
    %8 = stablehlo.constant dense<1> : tensor<1024x1024x3328xi32>
    %9 = stablehlo.add %arg1, %8 : tensor<1024x1024x3328xi32>
    return %7, %9 : tensor<1x1xi32>, tensor<1024x1024x3328xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  auto program = std::make_unique<IfrtIRProgram>(*mlir_module);
  ASSERT_OK_AND_ASSIGN(
      program,
      SerDeRoundTrip(std::move(program),
                     xla::ifrt::Version::CompatibilityRequirement::WEEK_4));
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(1));
  ASSERT_OK_AND_ASSIGN(
      auto ifrt_ir_executable,
      client_->GetDefaultCompiler()
          ->CompileAndLoad(
              std::move(program),
              std::make_unique<IfrtIRCompileOptions>(GetDeviceIds(devices)))
          .Await());

  std::vector<int> data = {42};
  Shape shape({1, 1});
  ASSERT_OK_AND_ASSIGN(ArrayRef input,
                       CreateArray({data.data()}, shape, /*shard_shape=*/shape,
                                   DType(DType::kS32), devices));
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      ifrt_ir_executable->Execute(absl::MakeSpan(&input, 1),
                                  ExecuteOptionsWithFillStatus(), devices));
  ASSERT_OK(result.status.Await());
  ASSERT_OK(result.outputs[0]->GetReadyFuture().Await());
  ASSERT_OK(result.outputs[1]->GetReadyFuture().Await());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
