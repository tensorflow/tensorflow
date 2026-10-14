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

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/client/executable_build_options.h"
#include "xla/literal.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/bundle.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/xla_client.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/device_assignment.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::SizeIs;

constexpr absl::string_view kModuleAddOne = R"mlir(
module @add_one {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<2x3xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
})mlir";

constexpr absl::string_view kModuleTwoOutputs = R"mlir(
module @two_outputs {
  func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<2x3xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<2x3xf32>
    return %arg0, %2 : tensor<2x3xf32>, tensor<2x3xf32>
  }
})mlir";

TEST(PjRtExecutableImplTest, HloOutputCallbackWrapper) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  bool invoked = false;
  auto hlo_cb = std::make_unique<xla::HloOutputCallback>();
  hlo_cb->callback_id = 42;
  hlo_cb->num_operands = 1;
  hlo_cb->callback =
      [&invoked](
          int64_t replica_id, int64_t partition_id,
          absl::Span<std::shared_ptr<const xla::Literal> const> literals) {
        invoked = true;
        EXPECT_EQ(replica_id, 0);
        EXPECT_EQ(partition_id, 0);
        ASSERT_EQ(literals.size(), 1);
        ASSERT_NE(literals[0], nullptr);
        EXPECT_EQ(literals[0]->Get<int32_t>({}), 123);
      };

  auto loaded_host_callback = tsl::MakeRef<PjRtHloOutputLoadedHostCallback>(
      client.get(), std::move(hlo_cb));

  EXPECT_EQ(loaded_host_callback->hlo_output_callback().callback_id, 42);
  EXPECT_EQ(loaded_host_callback->hlo_output_callback().num_operands, 1);

  ASSERT_OK_AND_ASSIGN(
      auto lit, xla::Literal::Make(xla::ShapeUtil::MakeShape(xla::S32, {})));
  lit.Set<int32_t>({}, 123);
  auto shared_lit = std::make_shared<const xla::Literal>(std::move(lit));
  loaded_host_callback->hlo_output_callback().callback(
      0, 0, absl::MakeSpan(&shared_lit, 1));
  EXPECT_TRUE(invoked);
}

TEST(PjRtExecutableImplTest, CompileAndLoadSeparately) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());
  ASSERT_GE(client->addressable_devices().size(), 2);
  Compiler* compiler = client->GetDefaultCompiler();

  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       xla::ParseMlirModuleString(kModuleAddOne, context));

  Device* device0 = client->addressable_devices().at(0);
  Device* device1 = client->addressable_devices().at(1);
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                       client->MakeDeviceList({device0, device1}));
  ASSERT_OK_AND_ASSIGN(DeviceListRef single_device_list,
                       client->MakeDeviceList({device0}));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Topology> topology,
                       client->GetTopologyForDevices(device_list));

  xla::CompileOptions compile_options;
  ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  build_options.set_num_replicas(2);
  build_options.set_num_partitions(1);
  build_options.set_use_spmd_partitioning(true);
  DeviceAssignment device_assignment(2, 1);
  device_assignment(0, 0) = device0->Id().value();
  device_assignment(1, 0) = device1->Id().value();
  build_options.set_device_assignment(device_assignment);

  auto xla_compile_options =
      std::make_unique<XlaCompileOptions>(compile_options, device_list);
  ASSERT_OK_AND_ASSIGN(ExecutableRef executable,
                       compiler
                           ->Compile(std::make_unique<HloProgram>(*module),
                                     *topology, std::move(xla_compile_options))
                           .Await());
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(executable->name(), "add_one");
  EXPECT_EQ(executable->num_devices(), 2);

  std::vector<ExecutableRef> executables = {executable, executable};
  std::vector<std::unique_ptr<LoadOptions>> load_options;
  load_options.push_back(std::make_unique<XlaLoadOptions>(
      device_list, /*loaded_host_callbacks=*/
      std::vector<tsl::RCReference<LoadedHostCallback>>(),
      /*outputs_bundle_slice_sizes=*/std::nullopt));
  load_options.push_back(std::make_unique<XlaLoadOptions>(
      /*devices=*/DeviceListRef(), /*loaded_host_callbacks=*/
      std::vector<tsl::RCReference<LoadedHostCallback>>(),
      /*outputs_bundle_slice_sizes=*/std::nullopt));
  ASSERT_OK_AND_ASSIGN(
      std::vector<tsl::Future<LoadedExecutableRef>> loaded_executable_futures,
      client->Load(executables, absl::MakeSpan(load_options)));
  ASSERT_THAT(loaded_executable_futures, SizeIs(2));
  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded_executable,
                       loaded_executable_futures[0].Await());
  ASSERT_NE(loaded_executable, nullptr);
  EXPECT_EQ(loaded_executable->name(), "add_one");
  ASSERT_TRUE(loaded_executable->devices().has_value());
  EXPECT_THAT((*loaded_executable->devices())->devices(),
              ElementsAre(device0, device1));

  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded_executable_default_devices,
                       loaded_executable_futures[1].Await());
  ASSERT_NE(loaded_executable_default_devices, nullptr);
  ASSERT_TRUE(loaded_executable_default_devices->devices().has_value());
  EXPECT_THAT((*loaded_executable_default_devices->devices())->devices(),
              ElementsAre(device0, device1));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  std::vector<float> expected_out_data(6);
  absl::c_iota(expected_out_data, 1);
  ShardingRef sharding =
      ConcreteEvenSharding::Create(device_list, MemoryKind(), shape, shape,
                                   /*is_fully_replicated=*/true);
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape, /*byte_strides=*/std::nullopt, sharding,
          /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));

  LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = true;
  for (const std::optional<DeviceListRef>& execute_devices :
       {std::optional<DeviceListRef>(),
        std::optional<DeviceListRef>(device_list)}) {
    ASSERT_OK_AND_ASSIGN(
        LoadedExecutable::ExecuteResult result,
        loaded_executable->Execute(absl::MakeSpan(&input, 1), execute_options,
                                   execute_devices));
    EXPECT_THAT(result.status.Await(), IsOk());
    ASSERT_THAT(result.outputs, SizeIs(1));

    ASSERT_OK_AND_ASSIGN(
        std::vector<ArrayRef> output_shards,
        result.outputs.front()->DisassembleIntoSingleDeviceArrays(
            ArrayCopySemantics::kDonateInput,
            SingleDeviceShardSemantics::kAddressableShards));
    ASSERT_THAT(output_shards, SizeIs(2));
    for (const ArrayRef& output_shard : output_shards) {
      std::vector<float> out_data(6);
      EXPECT_THAT(output_shard
                      ->CopyToHostBuffer(out_data.data(),
                                         /*byte_strides=*/std::nullopt,
                                         ArrayCopySemantics::kAlwaysCopy)
                      .Await(),
                  IsOk());
      EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
    }
  }

  EXPECT_THAT(loaded_executable->Execute(absl::MakeSpan(&input, 1),
                                         execute_options, single_device_list),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PjRtExecutableImplTest, CompileAndLoadWithOutputsBundleSliceSizes) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       xla::ParseMlirModuleString(kModuleTwoOutputs, context));

  Device* device = client->addressable_devices().at(0);
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                       client->MakeDeviceList({device}));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Topology> topology,
                       client->GetTopologyForDevices(device_list));

  xla::CompileOptions compile_options;
  ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  build_options.set_device_ordinal(device->Id().value());
  build_options.set_num_replicas(1);
  build_options.set_num_partitions(1);
  build_options.set_use_spmd_partitioning(true);
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = device->Id().value();
  build_options.set_device_assignment(device_assignment);

  auto xla_compile_options =
      std::make_unique<XlaCompileOptions>(compile_options, device_list);
  xla_compile_options->outputs_bundle_slice_sizes = std::vector<int>{1, 1};
  ASSERT_OK_AND_ASSIGN(ExecutableRef executable,
                       compiler
                           ->Compile(std::make_unique<HloProgram>(*module),
                                     *topology, std::move(xla_compile_options))
                           .Await());

  std::vector<ExecutableRef> executables = {executable, executable};
  std::vector<std::unique_ptr<LoadOptions>> load_options;
  load_options.push_back(std::make_unique<XlaLoadOptions>(
      device_list, /*loaded_host_callbacks=*/
      std::vector<tsl::RCReference<LoadedHostCallback>>(),
      /*outputs_bundle_slice_sizes=*/std::nullopt));
  load_options.push_back(std::make_unique<XlaLoadOptions>(
      device_list, /*loaded_host_callbacks=*/
      std::vector<tsl::RCReference<LoadedHostCallback>>(),
      /*outputs_bundle_slice_sizes=*/std::vector<int>{2}));
  ASSERT_OK_AND_ASSIGN(
      std::vector<tsl::Future<LoadedExecutableRef>> loaded_executable_futures,
      client->Load(executables, absl::MakeSpan(load_options)));
  ASSERT_THAT(loaded_executable_futures, SizeIs(2));

  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded_preserved,
                       loaded_executable_futures[0].Await());
  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded_overridden,
                       loaded_executable_futures[1].Await());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  std::vector<float> expected_out_data(6);
  absl::c_iota(expected_out_data, 1);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  ASSERT_OK_AND_ASSIGN(
      ArrayRef input,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape, /*byte_strides=*/std::nullopt, sharding,
          /*layout=*/nullptr,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));
  std::vector<ValueRef> input_values = {input};
  ASSERT_OK_AND_ASSIGN(BundleRef input_bundle,
                       client->Bundle(absl::MakeSpan(input_values),
                                      ArrayCopySemantics::kReuseInput));

  LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = true;

  // Verify compile-time outputs_bundle_slice_sizes ({1, 1}) is preserved when
  // LoadOptions::outputs_bundle_slice_sizes is std::nullopt.
  ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteBundleResult result_preserved,
                       loaded_preserved->ExecuteBundle(
                           absl::MakeSpan(&input_bundle, 1), execute_options));
  EXPECT_THAT(result_preserved.status.Await(), IsOk());
  ASSERT_THAT(result_preserved.outputs, SizeIs(2));
  EXPECT_EQ(result_preserved.outputs[0]->num_values(), 1);
  EXPECT_EQ(result_preserved.outputs[1]->num_values(), 1);

  for (int i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(std::vector<ValueRef> values,
                         result_preserved.outputs[i]->GetValues(
                             ArrayCopySemantics::kReuseInput));
    ASSERT_THAT(values, SizeIs(1));
    auto* out_array = xla::ifrt::dyn_cast<Array>(values.front().get());
    ASSERT_NE(out_array, nullptr);
    std::vector<float> out_data(6);
    EXPECT_THAT(out_array
                    ->CopyToHostBuffer(out_data.data(),
                                       /*byte_strides=*/std::nullopt,
                                       ArrayCopySemantics::kAlwaysCopy)
                    .Await(),
                IsOk());
    EXPECT_THAT(out_data, ElementsAreArray(i == 0 ? data : expected_out_data));
  }

  // Verify LoadOptions::outputs_bundle_slice_sizes ({2}) overrides compile-time
  // outputs_bundle_slice_sizes.
  ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteBundleResult result_overridden,
                       loaded_overridden->ExecuteBundle(
                           absl::MakeSpan(&input_bundle, 1), execute_options));
  EXPECT_THAT(result_overridden.status.Await(), IsOk());
  ASSERT_THAT(result_overridden.outputs, SizeIs(1));
  EXPECT_EQ(result_overridden.outputs.front()->num_values(), 2);

  ASSERT_OK_AND_ASSIGN(std::vector<ValueRef> overridden_values,
                       result_overridden.outputs.front()->GetValues(
                           ArrayCopySemantics::kReuseInput));
  ASSERT_THAT(overridden_values, SizeIs(2));
  for (int i = 0; i < 2; ++i) {
    auto* out_array = xla::ifrt::dyn_cast<Array>(overridden_values[i].get());
    ASSERT_NE(out_array, nullptr);
    std::vector<float> out_data(6);
    EXPECT_THAT(out_array
                    ->CopyToHostBuffer(out_data.data(),
                                       /*byte_strides=*/std::nullopt,
                                       ArrayCopySemantics::kAlwaysCopy)
                    .Await(),
                IsOk());
    EXPECT_THAT(out_data, ElementsAreArray(i == 0 ? data : expected_out_data));
  }
}

TEST(PjRtExecutableImplTest, LoadBatchWithDuplicateExecutables) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());
  ASSERT_GE(client->addressable_devices().size(), 2);
  Compiler* compiler = client->GetDefaultCompiler();

  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       xla::ParseMlirModuleString(kModuleAddOne, context));

  Device* device0 = client->addressable_devices().at(0);
  Device* device1 = client->addressable_devices().at(1);
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list0,
                       client->MakeDeviceList({device0}));
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                       client->MakeDeviceList({device1}));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Topology> topology,
                       client->GetTopologyForDevices(device_list0));

  xla::CompileOptions compile_options;
  compile_options.compile_portable_executable = true;
  auto xla_compile_options =
      std::make_unique<XlaCompileOptions>(compile_options, device_list0);
  ASSERT_OK_AND_ASSIGN(ExecutableRef executable,
                       compiler
                           ->Compile(std::make_unique<HloProgram>(*module),
                                     *topology, std::move(xla_compile_options))
                           .Await());

  std::vector<ExecutableRef> executables = {executable, executable, executable};
  std::vector<std::unique_ptr<LoadOptions>> load_options;
  load_options.push_back(std::make_unique<XlaLoadOptions>(
      device_list0, /*loaded_host_callbacks=*/
      std::vector<tsl::RCReference<LoadedHostCallback>>(),
      /*outputs_bundle_slice_sizes=*/std::nullopt));
  load_options.push_back(std::make_unique<XlaLoadOptions>(
      device_list1, /*loaded_host_callbacks=*/
      std::vector<tsl::RCReference<LoadedHostCallback>>(),
      /*outputs_bundle_slice_sizes=*/std::nullopt));
  load_options.push_back(std::make_unique<XlaLoadOptions>(
      /*devices=*/DeviceListRef(), /*loaded_host_callbacks=*/
      std::vector<tsl::RCReference<LoadedHostCallback>>(),
      /*outputs_bundle_slice_sizes=*/std::nullopt));

  ASSERT_OK_AND_ASSIGN(
      std::vector<tsl::Future<LoadedExecutableRef>> loaded_executable_futures,
      client->Load(executables, absl::MakeSpan(load_options)));
  ASSERT_THAT(loaded_executable_futures, SizeIs(3));

  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded0,
                       loaded_executable_futures[0].Await());
  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded1,
                       loaded_executable_futures[1].Await());
  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded2,
                       loaded_executable_futures[2].Await());
  ASSERT_TRUE(loaded0->devices().has_value());
  EXPECT_THAT((*loaded0->devices())->devices(), ElementsAre(device0));
  ASSERT_TRUE(loaded1->devices().has_value());
  EXPECT_THAT((*loaded1->devices())->devices(), ElementsAre(device1));
  EXPECT_EQ(loaded2->devices(), std::nullopt);

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  std::vector<float> expected_out_data(6);
  absl::c_iota(expected_out_data, 1);

  for (const auto& [loaded, device, execute_devices] :
       {std::tuple{loaded0.get(), device0, std::optional<DeviceListRef>()},
        std::tuple{loaded0.get(), device0,
                   std::optional<DeviceListRef>(device_list0)},
        std::tuple{loaded1.get(), device1, std::optional<DeviceListRef>()},
        std::tuple{loaded2.get(), device1,
                   std::optional<DeviceListRef>(device_list1)}}) {
    ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
    ASSERT_OK_AND_ASSIGN(
        ArrayRef input,
        client->MakeArrayFromHostBuffer(
            data.data(), dtype, shape, /*byte_strides=*/std::nullopt, sharding,
            /*layout=*/nullptr,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/{}));

    LoadedExecutable::ExecuteOptions execute_options;
    execute_options.fill_status = true;
    ASSERT_OK_AND_ASSIGN(LoadedExecutable::ExecuteResult result,
                         loaded->Execute(absl::MakeSpan(&input, 1),
                                         execute_options, execute_devices));
    EXPECT_THAT(result.status.Await(), IsOk());
    ASSERT_THAT(result.outputs, SizeIs(1));
    EXPECT_TRUE(xla::ifrt::isa<SingleDeviceSharding>(
        result.outputs.front()->sharding()));
    EXPECT_THAT(result.outputs.front()->sharding().devices()->devices(),
                ElementsAre(device));

    std::vector<float> out_data(6);
    EXPECT_THAT(result.outputs.front()
                    ->CopyToHostBuffer(out_data.data(),
                                       /*byte_strides=*/std::nullopt,
                                       ArrayCopySemantics::kAlwaysCopy)
                    .Await(),
                IsOk());
    EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
  }

  // Executing a load-time bound portable executable on a different device list,
  // or executing an unbound portable executable without specifying devices,
  // fails.
  {
    ShardingRef sharding = SingleDeviceSharding::Create(device0, MemoryKind());
    ASSERT_OK_AND_ASSIGN(
        ArrayRef input,
        client->MakeArrayFromHostBuffer(
            data.data(), dtype, shape, /*byte_strides=*/std::nullopt, sharding,
            /*layout=*/nullptr,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/{}));
    LoadedExecutable::ExecuteOptions execute_options;
    EXPECT_THAT(loaded0->Execute(absl::MakeSpan(&input, 1), execute_options,
                                 device_list1),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(loaded2->Execute(absl::MakeSpan(&input, 1), execute_options,
                                 /*devices=*/std::nullopt),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(PjRtExecutableImplTest, LoadValidationErrors) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());
  ASSERT_GE(client->addressable_devices().size(), 2);
  Compiler* compiler = client->GetDefaultCompiler();

  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       xla::ParseMlirModuleString(kModuleAddOne, context));

  Device* device = client->addressable_devices().at(0);
  Device* device1 = client->addressable_devices().at(1);
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                       client->MakeDeviceList({device}));
  ASSERT_OK_AND_ASSIGN(DeviceListRef device_list1,
                       client->MakeDeviceList({device1}));
  ASSERT_OK_AND_ASSIGN(DeviceListRef two_devices,
                       client->MakeDeviceList({device, device1}));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Topology> topology,
                       client->GetTopologyForDevices(device_list));

  // Passing `loaded_host_callbacks` to `Compiler::Compile` is rejected.
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> callback_module,
                         xla::ParseMlirModuleString(kModuleAddOne, context));
    xla::CompileOptions callback_compile_options;
    std::vector<tsl::RCReference<LoadedHostCallback>> callbacks = {
        tsl::RCReference<LoadedHostCallback>()};
    auto callback_xla_compile_options = std::make_unique<XlaCompileOptions>(
        callback_compile_options, device_list, std::move(callbacks));
    EXPECT_THAT(
        compiler
            ->Compile(std::make_unique<HloProgram>(*callback_module), *topology,
                      std::move(callback_xla_compile_options))
            .Await(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  xla::CompileOptions compile_options;
  compile_options.compile_portable_executable = true;
  auto xla_compile_options =
      std::make_unique<XlaCompileOptions>(compile_options, device_list);
  ASSERT_OK_AND_ASSIGN(ExecutableRef executable,
                       compiler
                           ->Compile(std::make_unique<HloProgram>(*module),
                                     *topology, std::move(xla_compile_options))
                           .Await());

  // Mismatched sizes.
  {
    std::vector<ExecutableRef> executables = {executable};
    std::vector<std::unique_ptr<LoadOptions>> options;
    EXPECT_THAT(client->Load(executables, absl::MakeSpan(options)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Null executable.
  {
    ExecutableRef null_executable;
    std::unique_ptr<LoadOptions> options = std::make_unique<XlaLoadOptions>(
        device_list, /*loaded_host_callbacks=*/
        std::vector<tsl::RCReference<LoadedHostCallback>>(),
        /*outputs_bundle_slice_sizes=*/std::nullopt);
    EXPECT_THAT(client->Load(absl::MakeConstSpan(&null_executable, 1),
                             absl::MakeSpan(&options, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Null options.
  {
    std::unique_ptr<LoadOptions> null_options;
    EXPECT_THAT(client->Load(absl::MakeConstSpan(&executable, 1),
                             absl::MakeSpan(&null_options, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Empty devices in options.
  {
    DeviceListRef empty_devices = BasicDeviceList::Create({});
    std::unique_ptr<LoadOptions> options = std::make_unique<XlaLoadOptions>(
        empty_devices, /*loaded_host_callbacks=*/
        std::vector<tsl::RCReference<LoadedHostCallback>>(),
        /*outputs_bundle_slice_sizes=*/std::nullopt);
    EXPECT_THAT(client->Load(absl::MakeConstSpan(&executable, 1),
                             absl::MakeSpan(&options, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Multiple devices in options for a portable executable.
  {
    std::unique_ptr<LoadOptions> options = std::make_unique<XlaLoadOptions>(
        two_devices, /*loaded_host_callbacks=*/
        std::vector<tsl::RCReference<LoadedHostCallback>>(),
        /*outputs_bundle_slice_sizes=*/std::nullopt);
    EXPECT_THAT(client->Load(absl::MakeConstSpan(&executable, 1),
                             absl::MakeSpan(&options, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Mismatched devices in options for a non-portable executable.
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> non_portable_module,
                         xla::ParseMlirModuleString(kModuleAddOne, context));
    xla::CompileOptions non_portable_compile_options;
    ExecutableBuildOptions& build_options =
        non_portable_compile_options.executable_build_options;
    build_options.set_device_ordinal(device->Id().value());
    build_options.set_num_replicas(1);
    build_options.set_num_partitions(1);
    build_options.set_use_spmd_partitioning(true);
    DeviceAssignment device_assignment(1, 1);
    device_assignment(0, 0) = device->Id().value();
    build_options.set_device_assignment(device_assignment);
    auto non_portable_xla_compile_options = std::make_unique<XlaCompileOptions>(
        non_portable_compile_options, device_list);
    ASSERT_OK_AND_ASSIGN(
        ExecutableRef non_portable_executable,
        compiler
            ->Compile(std::make_unique<HloProgram>(*non_portable_module),
                      *topology, std::move(non_portable_xla_compile_options))
            .Await());

    std::unique_ptr<LoadOptions> options = std::make_unique<XlaLoadOptions>(
        device_list1, /*loaded_host_callbacks=*/
        std::vector<tsl::RCReference<LoadedHostCallback>>(),
        /*outputs_bundle_slice_sizes=*/std::nullopt);
    EXPECT_THAT(client->Load(absl::MakeConstSpan(&non_portable_executable, 1),
                             absl::MakeSpan(&options, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
