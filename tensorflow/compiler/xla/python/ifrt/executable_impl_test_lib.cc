/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::SizeIs;

// Serialized `ModuleOp` that does add 1.
static const char* const module_add_one =
    R"(module {
func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "mhlo.copy"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = "mhlo.broadcast"(%1) {broadcast_sizes = dense<[2, 3]> : tensor<2xi64>} : (tensor<f32>) -> tensor<2x3xf32>
  %3 = mhlo.add %0, %2 : tensor<2x3xf32>
  return %3 : tensor<2x3xf32>
}})";

// Compiles an MLIR module on specified devices.
StatusOr<std::unique_ptr<LoadedExecutable>> CompileOnDevices(
    Client* client, Compiler* compiler, absl::string_view mlir_module_str,
    absl::Span<Device* const> devices, bool replicated) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, context));

  auto compile_options = std::make_unique<CompileOptions>();
  ExecutableBuildOptions& build_options =
      compile_options->xla_options.executable_build_options;
  for (Device* device : devices) {
    build_options.set_device_ordinal(device->id());
    if (replicated) {
      DeviceAssignment device_assignment(/*replica_count=*/devices.size(),
                                         /*computation_count=*/1);
      for (int i = 0; i < devices.size(); ++i) {
        device_assignment(i, 0) = i;
      }
      build_options.set_device_assignment(device_assignment);
    } else {
      DeviceAssignment device_assignment(/*replica_count=*/1,
                                         /*computation_count=*/devices.size());
      for (int i = 0; i < devices.size(); ++i) {
        device_assignment(i, 0) = i;
      }
      build_options.set_device_assignment(device_assignment);
    }
  }
  return compiler->Compile(*module, std::move(compile_options));
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
  auto sharding = SingleDeviceSharding::Create(device);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  ExecuteOptions execute_options;
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
