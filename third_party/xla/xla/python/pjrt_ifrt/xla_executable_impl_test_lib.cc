/* Copyright 2022 The OpenXLA Authors.

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
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::IsOkAndHolds;

// Serialized `ModuleOp` that does add 1.
static const char* const module_add_one =
    R"(module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<2x3xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
})";

// Compiles an MLIR module on specified devices. If devices is empty, compiles
// it as a portable executable.
absl::StatusOr<std::unique_ptr<LoadedExecutable>> CompileOnDevices(
    Client* client, Compiler* compiler, absl::string_view mlir_module_str,
    absl::Span<Device* const> devices, bool replicated) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, context));

  xla::CompileOptions compile_options;
  ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  DeviceListRef device_list;
  if (devices.empty()) {
    compile_options.compile_portable_executable = true;
    device_list =
        client->MakeDeviceList({client->addressable_devices().front()});
  } else {
    build_options.set_device_ordinal(devices.front()->Id().value());
    if (replicated) {
      build_options.set_num_replicas(devices.size());
      build_options.set_num_partitions(1);
      build_options.set_use_spmd_partitioning(false);
      DeviceAssignment device_assignment(/*replica_count=*/devices.size(),
                                         /*computation_count=*/1);
      for (int i = 0; i < devices.size(); ++i) {
        device_assignment(i, 0) = devices[i]->Id().value();
      }
      build_options.set_device_assignment(device_assignment);
    } else {
      build_options.set_num_replicas(1);
      build_options.set_num_partitions(devices.size());
      build_options.set_use_spmd_partitioning(true);
      DeviceAssignment device_assignment(
          /*replica_count=*/1,
          /*computation_count=*/devices.size());
      for (int i = 0; i < devices.size(); ++i) {
        device_assignment(0, i) = devices[i]->Id().value();
      }
      build_options.set_device_assignment(device_assignment);
    }
    device_list = client->MakeDeviceList(devices);
  }
  auto xla_compile_options = std::make_unique<XlaCompileOptions>(
      compile_options, std::move(device_list));
  return compiler->Compile(std::make_unique<HloProgram>(*module),
                           std::move(xla_compile_options));
}

TEST(LoadedExecutableImplTest, GetDonatableInputIndices) {
  static const char* const multi_arg_add_all = R"(module {
    func.func @main(
        %arg0: tensor<2x3xf32> {jax.buffer_donor = true},
        %arg1: tensor<2x3xf32>,
        %arg2: tensor<2x3xf32> {jax.buffer_donor = true},
        %arg3: tensor<2x3xf32>
      ) -> tensor<2x3xf32> {
      %4 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
      %5 = stablehlo.add %arg2, %arg3 : tensor<2x3xf32>
      %6 = stablehlo.add %4, %5 : tensor<2x3xf32>
      return %6 : tensor<2x3xf32>
    }})";

  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, multi_arg_add_all, devices,
                       /*replicated=*/false));

  absl::StatusOr<absl::Span<const int>> donatable_input_indices =
      loaded_executable->GetDonatableInputIndices();

  if (absl::IsUnimplemented(donatable_input_indices.status())) {
    GTEST_SKIP() << "GetDonatableInputIndices() returned unimplemented error: "
                 << donatable_input_indices.status();
  }

  EXPECT_THAT(donatable_input_indices,
              IsOkAndHolds(UnorderedElementsAre(0, 2)));
}

TEST(LoadedExecutableImplTest, CompileAndExecute) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, module_add_one, devices,
                       /*replicated=*/false));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  ExecuteOptions execute_options;
  execute_options.fill_status = true;
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_executable->Execute(absl::MakeSpan(&array, 1), execute_options,
                                 /*devices=*/std::nullopt));
  TF_ASSERT_OK(result.status.Await());
  EXPECT_THAT(result.outputs, SizeIs(1));

  std::vector<float> out_data(6);
  auto future = result.outputs[0]->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());

  std::vector<float> expected_out_data(6);
  std::iota(expected_out_data.begin(), expected_out_data.end(), 1);
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST(LoadedExecutableImplTest, CompileAndExecutePortable) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, module_add_one, devices,
                       /*replicated=*/false));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, std::move(shape),
                      /*byte_strides=*/std::nullopt, std::move(sharding),
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  ExecuteOptions execute_options;
  execute_options.fill_status = true;
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_executable->Execute(absl::MakeSpan(&array, 1), execute_options,
                                 /*devices=*/client->MakeDeviceList({device})));
  TF_ASSERT_OK(result.status.Await());
  EXPECT_THAT(result.outputs, SizeIs(1));

  std::vector<float> out_data(6);
  auto future = result.outputs[0]->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());

  std::vector<float> expected_out_data(6);
  std::iota(expected_out_data.begin(), expected_out_data.end(), 1);
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST(LoadedExecutableImplTest, DoNotFillStatus) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, module_add_one, devices,
                       /*replicated=*/false));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  ExecuteOptions execute_options;
  execute_options.fill_status = false;
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_executable->Execute(absl::MakeSpan(&array, 1), execute_options,
                                 /*devices=*/std::nullopt));
  EXPECT_FALSE(result.status.IsValid());
  EXPECT_THAT(result.outputs, SizeIs(1));

  std::vector<float> out_data(6);
  auto future = result.outputs[0]->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());

  std::vector<float> expected_out_data(6);
  std::iota(expected_out_data.begin(), expected_out_data.end(), 1);
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST(LoadedExecutableImplTest, Delete) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, module_add_one, devices,
                       /*replicated=*/false));
  TF_EXPECT_OK(loaded_executable->Delete().Await());
}

TEST(LoadedExecutableImplTest, IsDeleted) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, module_add_one, devices,
                       /*replicated=*/false));
  EXPECT_FALSE(loaded_executable->IsDeleted());
  auto future = loaded_executable->Delete();
  EXPECT_TRUE(loaded_executable->IsDeleted());
  TF_EXPECT_OK(future.Await());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
