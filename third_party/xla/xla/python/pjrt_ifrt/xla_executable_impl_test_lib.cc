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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pjrt_ifrt/xla_executable_version.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Optional;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::EquivToProto;

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

static const char* const module_add_sub = R"(
module @add_sub attributes {
  mhlo.num_replicas = 1 : i32,
  mhlo.num_partitions = 2 : i32
} {
  func.func @main(
    %arg0: tensor<2x3xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"},
    %arg1: tensor<2x3xi32> {mhlo.sharding = "{replicated}"}
  ) -> (
    tensor<2x3xi32> {mhlo.sharding = "{replicated}"},
    tensor<2x3xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}
  ) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xi32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<2x3xi32>
    return %0, %1 : tensor<2x3xi32>, tensor<2x3xi32>
  }
})";

// Compiles an MLIR module on specified devices. If devices is empty, compiles
// it as a portable executable.
// If `serialize` is true, serializes the compiled executable, deserializes it,
// and returns the deserialized executable. This is to test correctness of
// serialization round-trip of the executable.
absl::StatusOr<LoadedExecutableRef> CompileOnDevices(
    Client* client, Compiler* compiler, absl::string_view mlir_module_str,
    absl::Span<Device* const> devices, bool replicated, bool serialize) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, context));

  xla::CompileOptions compile_options;
  ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  DeviceListRef device_list;
  if (devices.empty()) {
    compile_options.compile_portable_executable = true;
    TF_ASSIGN_OR_RETURN(
        device_list,
        client->MakeDeviceList({client->addressable_devices().front()}));
  } else {
    if (devices.size() == 1) {
      build_options.set_device_ordinal(devices.front()->Id().value());
    }
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
    TF_ASSIGN_OR_RETURN(device_list, client->MakeDeviceList(devices));
  }
  auto xla_compile_options =
      std::make_unique<XlaCompileOptions>(compile_options, device_list);
  TF_ASSIGN_OR_RETURN(
      auto loaded_executable,
      compiler
          ->CompileAndLoad(std::make_unique<HloProgram>(*module),
                           std::move(xla_compile_options))
          .Await());
  if (!serialize) {
    return loaded_executable;
  }
  TF_ASSIGN_OR_RETURN(auto serialized_executable,
                      loaded_executable->Serialize());
  auto options = std::make_unique<XlaDeserializeExecutableOptions>();
  options->devices = std::move(device_list);
  return compiler
      ->DeserializeLoadedExecutable(std::move(serialized_executable),
                                    std::move(options))
      .Await();
}

class LoadedExecutableImplTest
    : public testing::TestWithParam</*serialize=*/bool> {};

TEST_P(LoadedExecutableImplTest, Properties) {
  bool serialize = GetParam();

  static constexpr absl::string_view kModule = R"(
module @add_sub attributes {
  mhlo.num_replicas = 1 : i32,
  mhlo.num_partitions = 2 : i32
} {
  func.func @main(
    %arg0: tensor<2x3xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"},
    %arg1: tensor<2x3xi32> {mhlo.sharding = "{replicated}"}
  ) -> (
    tensor<2x3xi32> {mhlo.sharding = "{replicated}"},
    tensor<2x3xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}
  ) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xi32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<2x3xi32>
    return %0, %1 : tensor<2x3xi32>, tensor<2x3xi32>
  }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  absl::Span<Device* const> devices =
      client->addressable_devices().subspan(0, 2);
  TF_ASSERT_OK_AND_ASSIGN(
      const LoadedExecutableRef executable,
      CompileOnDevices(client.get(), compiler, kModule, devices,
                       /*replicated=*/false, serialize));

  EXPECT_EQ(executable->name(), "add_sub");
  EXPECT_EQ(executable->num_devices(), devices.size());

  EXPECT_THAT(executable->GetParameterShardings(),
              Optional(ElementsAre(EquivToProto(R"pb(
                                     type: OTHER
                                     tile_assignment_dimensions: 2
                                     tile_assignment_dimensions: 1
                                     iota_reshape_dims: 2
                                     iota_transpose_perm: 0
                                   )pb"),
                                   EquivToProto(R"pb(type: REPLICATED)pb"))));

  EXPECT_THAT(executable->GetOutputShardings(),
              Optional(ElementsAre(EquivToProto(R"pb(type: REPLICATED)pb"),
                                   EquivToProto(R"pb(
                                     type: OTHER
                                     tile_assignment_dimensions: 2
                                     tile_assignment_dimensions: 1
                                     iota_reshape_dims: 2
                                     iota_transpose_perm: 0
                                   )pb"))));

  EXPECT_THAT(executable->GetOutputMemoryKinds(),
              AnyOf(IsOkAndHolds(ElementsAre(ElementsAre("device", "device"))),
                    StatusIs(absl::StatusCode::kUnimplemented)));
}

absl::StatusOr<const LoadedExecutableRef> SimpleAddExecutable(Client* client,
                                                              bool serialize) {
  static constexpr absl::string_view kModule = R"(
module @add attributes {
  mhlo.num_replicas = 1 : i32,
  mhlo.num_partitions = 2 : i32
} {
  func.func @main(
    %arg0: tensor<2x3xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}
  ) -> (tensor<2x3xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
})";
  Compiler* compiler = client->GetDefaultCompiler();
  return CompileOnDevices(client, compiler, kModule,
                          {client->addressable_devices().front()},
                          /*replicated=*/false, serialize);
}

TEST_P(LoadedExecutableImplTest, GetHloModules) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      const LoadedExecutableRef executable,
      SimpleAddExecutable(client.get(), /*serialize=*/GetParam()));

  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<std::shared_ptr<xla::HloModule>> hlo_modules,
      executable->GetHloModules());
  ASSERT_EQ(hlo_modules.size(), 1);
  EXPECT_EQ(hlo_modules.front()->name(), "add");
}

TEST_P(LoadedExecutableImplTest, ProgramText) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      const LoadedExecutableRef executable,
      SimpleAddExecutable(client.get(), /*serialize=*/GetParam()));

  TF_ASSERT_OK_AND_ASSIGN(const auto program_text,
                          executable->GetHumanReadableProgramText());
  EXPECT_THAT(program_text, HasSubstr("add."));
}

TEST_P(LoadedExecutableImplTest, Analysis) {
  if (const bool serialize = GetParam(); serialize) {
    GTEST_SKIP() << "Analysis is not supported for serialized executables.";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      const LoadedExecutableRef executable,
      SimpleAddExecutable(client.get(), /*serialize=*/false));

  TF_ASSERT_OK_AND_ASSIGN(const xla::CompiledMemoryStats compiled_memory_stats,
                          executable->GetCompiledMemoryStats());
  EXPECT_GT(compiled_memory_stats.argument_size_in_bytes, 0);

  TF_ASSERT_OK_AND_ASSIGN(const auto cost_analysis,
                          executable->GetCostAnalysis());
  EXPECT_FALSE(cost_analysis.IsEmpty());
}

TEST_P(LoadedExecutableImplTest, GetDonatableInputIndices) {
  bool serialize = GetParam();

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
                       /*replicated=*/false, serialize));

  absl::StatusOr<absl::Span<const int>> donatable_input_indices =
      loaded_executable->GetDonatableInputIndices();

  EXPECT_THAT(donatable_input_indices,
              IsOkAndHolds(UnorderedElementsAre(0, 2)));
}

TEST_P(LoadedExecutableImplTest, CompileAndExecute) {
  bool serialize = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  LoadedExecutableRef loaded_executable;
  {
    UserContextScope user_context_scope(test_util::MakeUserContext(20));
    TF_ASSERT_OK_AND_ASSIGN(
        loaded_executable,
        CompileOnDevices(client.get(), compiler, module_add_one, devices,
                         /*replicated=*/false, serialize));
  }
  EXPECT_EQ(loaded_executable->user_context()->Id(), UserContextId(20));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
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
  LoadedExecutable::ExecuteResult result;
  {
    UserContextScope user_context_scope(test_util::MakeUserContext(100));
    TF_ASSERT_OK_AND_ASSIGN(
        result,
        loaded_executable->Execute(absl::MakeSpan(&array, 1), execute_options,
                                   /*devices=*/std::nullopt));
  }
  TF_ASSERT_OK(result.status.Await());
  EXPECT_THAT(result.outputs, SizeIs(1));
  EXPECT_EQ(result.outputs[0]->user_context()->Id(), UserContextId(100));

  std::vector<float> out_data(6);
  auto future = result.outputs[0]->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());

  std::vector<float> expected_out_data(6);
  absl::c_iota(expected_out_data, 1);
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST_P(LoadedExecutableImplTest, CompileAndExecutePortable) {
  bool serialize = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {};
  LoadedExecutableRef loaded_executable;
  {
    UserContextScope user_context_scope(test_util::MakeUserContext(20));
    TF_ASSERT_OK_AND_ASSIGN(
        loaded_executable,
        CompileOnDevices(client.get(), compiler, module_add_one, devices,
                         /*replicated=*/false, serialize));
  }
  EXPECT_EQ(loaded_executable->user_context()->Id(), UserContextId(20));
  EXPECT_EQ(loaded_executable->devices(), std::nullopt);

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, std::move(shape),
                      /*byte_strides=*/std::nullopt, std::move(sharding),
                      /*layout=*/nullptr,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList({device}));
  ExecuteOptions execute_options;
  execute_options.fill_status = true;
  LoadedExecutable::ExecuteResult result;
  {
    UserContextScope user_context_scope(test_util::MakeUserContext(100));
    TF_ASSERT_OK_AND_ASSIGN(
        result,
        loaded_executable->Execute(absl::MakeSpan(&array, 1), execute_options,
                                   /*devices=*/std::move(device_list)));
  }
  TF_ASSERT_OK(result.status.Await());
  EXPECT_THAT(result.outputs, SizeIs(1));
  EXPECT_EQ(result.outputs[0]->user_context()->Id(), UserContextId(100));

  std::vector<float> out_data(6);
  auto future = result.outputs[0]->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());

  std::vector<float> expected_out_data(6);
  absl::c_iota(expected_out_data, 1);
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST_P(LoadedExecutableImplTest, CancelExecution) {
  bool serialize = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  LoadedExecutableRef loaded_executable;
  {
    UserContextScope user_context_scope(test_util::MakeUserContext(20));
    TF_ASSERT_OK_AND_ASSIGN(
        loaded_executable,
        CompileOnDevices(client.get(), compiler, module_add_one, devices,
                         /*replicated=*/false, serialize));
  }
  EXPECT_EQ(loaded_executable->user_context()->Id(), UserContextId(20));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
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
  LoadedExecutable::ExecuteResult result;
  {
    UserContextScope user_context_scope(test_util::MakeUserContext(100));
    TF_ASSERT_OK_AND_ASSIGN(
        result,
        loaded_executable->Execute(absl::MakeSpan(&array, 1), execute_options,
                                   /*devices=*/std::nullopt));

    // Smoke test for execution cancellation. Whether cancellation
    // succeeds/fails or is supported/unsupported, the API call should finish
    // without an error.
    client->CancelExecution(result.cancellation_handle,
                            absl::CancelledError("test"));
    // Execution cancellation is idempotent.
    client->CancelExecution(result.cancellation_handle,
                            absl::CancelledError("test"));
  }

  // After cancellation, the user code typically blocks on the execution result
  // future to ensure that the execution has completed or fully cancelled.
  std::vector<float> out_data(6);
  auto future = result.outputs[0]->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  future.Await().IgnoreError();
}

TEST_P(LoadedExecutableImplTest, DoNotFillStatus) {
  bool serialize = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  std::vector<Device*> devices = {client->addressable_devices().at(0)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, module_add_one, devices,
                       /*replicated=*/false, serialize));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
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
  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      loaded_executable->Execute(absl::MakeSpan(&array, 1), execute_options,
                                 /*devices=*/std::nullopt));
  EXPECT_FALSE(result.status.IsValid());
  EXPECT_THAT(result.outputs, SizeIs(1));
  EXPECT_EQ(result.outputs[0]->user_context()->Id(), UserContextId(100));

  std::vector<float> out_data(6);
  auto future = result.outputs[0]->CopyToHostBuffer(
      out_data.data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());

  std::vector<float> expected_out_data(6);
  absl::c_iota(expected_out_data, 1);
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST_P(LoadedExecutableImplTest, NoInputOutput) {
  bool serialize = GetParam();

  static constexpr absl::string_view kModule = R"(
module @nop attributes {
  mhlo.num_replicas = 1 : i32,
  mhlo.num_partitions = 2 : i32
} {
  func.func @main() {
    return
  }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  Device* const device = client->addressable_devices().front();
  TF_ASSERT_OK_AND_ASSIGN(
      const LoadedExecutableRef executable,
      CompileOnDevices(client.get(), compiler, kModule, {device},
                       /*replicated=*/false, serialize));

  ExecuteOptions options;
  options.fill_status = true;
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      executable->Execute({}, options, /*devices=*/std::nullopt));

  TF_ASSERT_OK(result.status.Await());
}

TEST_P(LoadedExecutableImplTest, Donation) {
  bool serialize = GetParam();

  static constexpr absl::string_view kModule = R"(
module @add_sub {
  func.func @main(
    %arg0: tensor<2x3xi32> {jax.buffer_donor = true},
    %arg1: tensor<2x3xi32> {jax.buffer_donor = true}
  ) -> (tensor<2x3xi32>, tensor<2x3xi32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xi32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<2x3xi32>
    return %0, %1 : tensor<2x3xi32>, tensor<2x3xi32>
  }
})";
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Compiler* compiler = client->GetDefaultCompiler();

  Device* const device = client->addressable_devices().front();
  TF_ASSERT_OK_AND_ASSIGN(
      const LoadedExecutableRef executable,
      CompileOnDevices(client.get(), compiler, kModule, {device},
                       /*replicated=*/false, serialize));

  // Create an input array.
  std::vector<ArrayRef> arrays;
  {
    ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
    for (int i = 0; i < 2; ++i) {
      std::vector<int32_t> data(6);
      absl::c_iota(data, 0);
      TF_ASSERT_OK_AND_ASSIGN(
          arrays.emplace_back(),
          client->MakeArrayFromHostBuffer(
              data.data(), DType(DType::kS32), Shape({2, 3}),
              /*byte_strides=*/std::nullopt, sharding,
              Client::HostBufferSemantics::kImmutableOnlyDuringCall,
              /*on_done_with_host_buffer=*/{}));
    }
  }

  // Enqueue a read operation just before donation. The scheduler must not
  // reorder read and donation.
  std::vector<int32_t> data(6);
  tsl::Future<> copy_future =
      arrays[0]->CopyToHostBuffer(data.data(), /*byte_strides=*/std::nullopt,
                                  ArrayCopySemantics::kAlwaysCopy);

  LoadedExecutable::ExecuteOptions execute_options;
  execute_options.non_donatable_input_indices.insert(1);
  execute_options.fill_status = true;
  TF_ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult result,
      executable->Execute(absl::MakeSpan(arrays), execute_options,
                          /*devices=*/std::nullopt));

  // The first input array is donated to the computation, so it should have been
  // marked as deleted after the execution is dispatched.
  EXPECT_TRUE(arrays[0]->IsDeleted());
  EXPECT_THAT(arrays[0]->GetReadyFuture().Await(), Not(IsOk()));

  // The second input array is marked as non-donatable in the execute option,
  // which should be respected by the execution.
  EXPECT_FALSE(arrays[1]->IsDeleted());
  TF_EXPECT_OK(arrays[1]->GetReadyFuture().Await());

  // Copy will succeed as long as the ordering is preserved.
  TF_ASSERT_OK(copy_future.Await());
  EXPECT_THAT(data, ElementsAre(0, 1, 2, 3, 4, 5));

  TF_ASSERT_OK(result.status.Await());
  EXPECT_THAT(result.outputs, SizeIs(2));

  {
    std::vector<int32_t> output(6);
    TF_ASSERT_OK(result.outputs[0]
                     ->CopyToHostBuffer(output.data(),
                                        /*byte_strides=*/std::nullopt,
                                        ArrayCopySemantics::kAlwaysCopy)
                     .Await());
    EXPECT_THAT(output, ElementsAre(0, 2, 4, 6, 8, 10));
  }
  {
    std::vector<int32_t> output(6);
    TF_ASSERT_OK(result.outputs[1]
                     ->CopyToHostBuffer(output.data(),
                                        /*byte_strides=*/std::nullopt,
                                        ArrayCopySemantics::kAlwaysCopy)
                     .Await());
    EXPECT_THAT(output, ElementsAre(0, 0, 0, 0, 0, 0));
  }
}

INSTANTIATE_TEST_SUITE_P(
    LoadedExecutableImplTest, LoadedExecutableImplTest,
    /*serialize=*/testing::Bool(),
    [](const ::testing::TestParamInfo<LoadedExecutableImplTest::ParamType>&
           info) {
      return std::string(info.param ? "SerializeAndLoad" : "DirectLoad");
    });

TEST(ExecutableTest, ExecutableSerialization) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  xla::ifrt::Compiler* compiler = client->GetDefaultCompiler();

  std::vector<xla::ifrt::Device*> devices = {
      client->addressable_devices().at(0), client->addressable_devices().at(1)};
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      CompileOnDevices(client.get(), compiler, module_add_sub, devices,
                       /*replicated=*/false, /*serialize=*/false));

  auto serialized_executable = loaded_executable->Serialize();
  if (absl::IsUnimplemented(serialized_executable.status())) {
    GTEST_SKIP() << "Serialization is not supported on this platform.";
  }
  TF_ASSERT_OK(serialized_executable);

  xla::ifrt::SerializedXlaExecutableMetadata metadata;
  tsl::protobuf::io::ArrayInputStream input_stream(
      serialized_executable->data(), serialized_executable->size());
  ASSERT_TRUE(google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      &metadata, &input_stream, nullptr));

  absl::StatusOr<std::shared_ptr<const xla::ifrt::ExecutableVersion>>
      executable_version = loaded_executable->executable_version();
  if (!absl::IsUnimplemented(executable_version.status())) {
    TF_ASSERT_OK(executable_version.status());
    TF_ASSERT_OK_AND_ASSIGN(
        auto xla_executable_version,
        xla::ifrt::ToXlaExecutableVersion(*std::move(executable_version)));
    TF_ASSERT_OK_AND_ASSIGN(auto serialized_xla_executable_version,
                            xla_executable_version->ToProto());
    EXPECT_THAT(metadata.executable_version(),
                EqualsProto(serialized_xla_executable_version));
  }

  EXPECT_EQ(metadata.computation_name(), "add_sub");

  int kNumOutputs = 2;
  TF_ASSERT_OK_AND_ASSIGN(auto output_memory_kinds,
                          loaded_executable->GetOutputMemoryKinds());
  ASSERT_EQ(output_memory_kinds.size(), 1);
  ASSERT_EQ(output_memory_kinds.at(0).size(), kNumOutputs);
  ASSERT_EQ(metadata.output_specs_size(), kNumOutputs);

  auto output_shardings = loaded_executable->GetOutputShardings();
  ASSERT_TRUE(output_shardings.has_value());
  ASSERT_EQ(output_shardings->size(), kNumOutputs);

  TF_ASSERT_OK_AND_ASSIGN(auto output_layouts,
                          loaded_executable->GetOutputLayouts());
  ASSERT_EQ(output_layouts.size(), kNumOutputs);

  for (int i = 0; i < kNumOutputs; ++i) {
    EXPECT_EQ(metadata.output_specs(i).memory_kind(),
              output_memory_kinds.at(0).at(i));
    EXPECT_EQ(static_cast<int32_t>(metadata.output_specs(i).dtype().kind()),
              static_cast<int32_t>(xla::ifrt::DType::kS32));
    EXPECT_THAT(metadata.output_specs(i).op_sharding(),
                tsl::proto_testing::EqualsProto(output_shardings->at(i)));

    // Verify that layout field behavior: if layout field is present,
    // it should match the expected layout. Otherwise it represents a default
    // layout.
    if (metadata.output_specs(i).has_layout()) {
      TF_ASSERT_OK_AND_ASSIGN(
          xla::ifrt::CustomLayoutRef output_layout,
          xla::ifrt::PjRtLayout::FromProto(metadata.output_specs(i).layout()));
      TF_ASSERT_OK_AND_ASSIGN(
          xla::ifrt::DType dtype,
          xla::ifrt::DType::FromProto(metadata.output_specs(i).dtype()));
      TF_ASSERT_OK_AND_ASSIGN(
          xla::ifrt::Shape ifrt_shard_shape,
          xla::ifrt::Shape::FromProto(metadata.output_specs(i).shard_shape()));
      TF_ASSERT_OK_AND_ASSIGN(
          auto pjrt_layout,
          xla::ifrt::ToPjRtLayout(dtype, ifrt_shard_shape, output_layout));
      EXPECT_EQ(pjrt_layout->ToString(), output_layouts[i]->ToString());
    }
  }

  int kNumParameters = 2;
  auto parameter_shardings = loaded_executable->GetParameterShardings();
  ASSERT_TRUE(parameter_shardings.has_value());
  ASSERT_EQ(parameter_shardings->size(), kNumParameters);
  ASSERT_EQ(metadata.parameter_specs_size(), kNumParameters);

  TF_ASSERT_OK_AND_ASSIGN(auto parameter_layouts,
                          loaded_executable->GetParameterLayouts());
  ASSERT_EQ(parameter_layouts.size(), kNumParameters);
  ASSERT_EQ(metadata.parameter_specs_size(), kNumParameters);

  TF_ASSERT_OK_AND_ASSIGN(auto donated_input_indices,
                          loaded_executable->GetDonatableInputIndices());
  absl::flat_hash_set<int> donated_input_indices_set(
      donated_input_indices.begin(), donated_input_indices.end());

  // Dummy shard shape and dtype for parameter layouts conversion.
  xla::ifrt::DType dummy_dtype(xla::ifrt::DType::kInvalid);
  xla::ifrt::Shape dummy_shape = xla::ifrt::Shape({});
  for (int i = 0; i < kNumParameters; ++i) {
    EXPECT_THAT(metadata.parameter_specs(i).op_sharding(),
                tsl::proto_testing::EqualsProto(parameter_shardings->at(i)));
    // Verify that layout field behavior: if layout field is present,
    // it should match the expected layout. Otherwise it represents a default
    // layout.
    if (metadata.parameter_specs(i).has_layout()) {
      TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::CustomLayoutRef parameter_layout,
                              xla::ifrt::PjRtLayout::FromProto(
                                  metadata.parameter_specs(i).layout()));
      TF_ASSERT_OK_AND_ASSIGN(
          auto pjrt_layout,
          xla::ifrt::ToPjRtLayout(dummy_dtype, dummy_shape, parameter_layout));
      EXPECT_EQ(pjrt_layout->ToString(), parameter_layouts[i]->ToString());
    }

    // Verify donated_input field
    bool expected_donated = donated_input_indices_set.contains(i);
    EXPECT_EQ(metadata.parameter_specs(i).donated_input(), expected_donated);
  }

  absl::string_view serialized_pjrt_executable = *serialized_executable;
  serialized_pjrt_executable.remove_prefix(input_stream.ByteCount());

  ASSERT_FALSE(serialized_pjrt_executable.empty());

  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          client->MakeDeviceList(devices));
  auto options = std::make_unique<xla::ifrt::XlaDeserializeExecutableOptions>();
  options->devices = device_list;
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized_executable,
                          client->GetDefaultCompiler()
                              ->DeserializeLoadedExecutable(
                                  *serialized_executable, std::move(options))
                              .Await());

  TF_ASSERT_OK_AND_ASSIGN(auto loaded_output_layouts,
                          loaded_executable->GetOutputLayouts());
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized_output_layouts,
                          deserialized_executable->GetOutputLayouts());
  ASSERT_EQ(loaded_output_layouts.size(), deserialized_output_layouts.size());
  for (int i = 0; i < loaded_output_layouts.size(); ++i) {
    EXPECT_EQ(loaded_output_layouts[i]->ToString(),
              deserialized_output_layouts[i]->ToString());
  }

  auto loaded_output_shardings = loaded_executable->GetOutputShardings();
  auto deserialized_output_shardings =
      deserialized_executable->GetOutputShardings();
  ASSERT_TRUE(loaded_output_shardings.has_value());
  ASSERT_TRUE(deserialized_output_shardings.has_value());
  ASSERT_EQ(loaded_output_shardings->size(),
            deserialized_output_shardings->size());
  for (int i = 0; i < loaded_output_shardings->size(); ++i) {
    EXPECT_THAT(
        (*loaded_output_shardings)[i],
        tsl::proto_testing::EqualsProto((*deserialized_output_shardings)[i]));
  }

  auto loaded_parameter_shardings = loaded_executable->GetParameterShardings();
  auto deserialized_parameter_shardings =
      deserialized_executable->GetParameterShardings();
  ASSERT_TRUE(loaded_parameter_shardings.has_value());
  ASSERT_TRUE(deserialized_parameter_shardings.has_value());
  ASSERT_EQ(loaded_parameter_shardings->size(),
            deserialized_parameter_shardings->size());
  for (int i = 0; i < loaded_parameter_shardings->size(); ++i) {
    EXPECT_THAT((*loaded_parameter_shardings)[i],
                tsl::proto_testing::EqualsProto(
                    (*deserialized_parameter_shardings)[i]));
  }
  EXPECT_EQ(deserialized_executable->name(), "add_sub");

  ASSERT_OK_AND_ASSIGN(
      xla::CompiledMemoryStats deserialized_compiled_memory_stats,
      deserialized_executable->GetCompiledMemoryStats());
  ASSERT_OK_AND_ASSIGN(xla::CompiledMemoryStats loaded_compiled_memory_stats,
                       loaded_executable->GetCompiledMemoryStats());

  // Temporary workaround for some implementations not round-tripping the
  // CompiledMemoryStats upon executable deserialization.
  loaded_compiled_memory_stats.serialized_buffer_assignment = "";
  loaded_compiled_memory_stats.peak_memory_in_bytes = 0;
  deserialized_compiled_memory_stats.serialized_buffer_assignment = "";
  deserialized_compiled_memory_stats.peak_memory_in_bytes = 0;

  EXPECT_THAT(deserialized_compiled_memory_stats.ToProto(),
              EqualsProto(loaded_compiled_memory_stats.ToProto()));

  // Execute the deserialized executable.
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  xla::ifrt::Shape shard_shape({1, 3});
  xla::ifrt::Shape shape({2, 3});
  std::vector<int32_t> data(6);
  absl::c_iota(data, 0);
  std::vector<xla::ifrt::ArrayRef> input_arrays;

  // Input 1 : [0, 1, 2, 3, 4, 5] sharded on device 0 and 1.
  xla::ifrt::ShardingRef shard_sharding0 =
      xla::ifrt::SingleDeviceSharding::Create(
          client->addressable_devices().at(0), xla::ifrt::MemoryKind());
  xla::ifrt::ShardingRef shard_sharding1 =
      xla::ifrt::SingleDeviceSharding::Create(
          client->addressable_devices().at(1), xla::ifrt::MemoryKind());
  xla::ifrt::ShardingRef input1_sharding =
      xla::ifrt::ConcreteEvenSharding::Create(
          device_list, xla::ifrt::MemoryKind(), shape, shard_shape,
          /*is_fully_replicated=*/false);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array_shard0,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shard_shape,
          /*byte_strides=*/std::nullopt, shard_sharding0,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto array_shard1,
      client->MakeArrayFromHostBuffer(
          data.data() + 3, dtype, shard_shape,
          /*byte_strides=*/std::nullopt, shard_sharding1,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));
  std::vector<xla::ifrt::ArrayRef> shards = {array_shard0, array_shard1};
  TF_ASSERT_OK_AND_ASSIGN(
      input_arrays.emplace_back(),
      client->AssembleArrayFromSingleDeviceArrays(
          dtype, shape, input1_sharding, absl::MakeSpan(shards),
          xla::ifrt::ArrayCopySemantics::kDonateInput,
          xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  // Input 2 : [0, 1, 2, 3, 4, 5] replicated on device 0 and 1.
  xla::ifrt::ShardingRef input2_sharding =
      xla::ifrt::ConcreteEvenSharding::Create(
          std::move(device_list), xla::ifrt::MemoryKind(), shape, shape,
          /*is_fully_replicated=*/true);
  TF_ASSERT_OK_AND_ASSIGN(
      input_arrays.emplace_back(),
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, input2_sharding,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));

  xla::ifrt::LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = true;
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::LoadedExecutable::ExecuteResult result,
                          deserialized_executable->Execute(
                              absl::MakeSpan(input_arrays), execute_options,
                              /*devices=*/std::nullopt));
  TF_ASSERT_OK(result.status.Await());
  EXPECT_THAT(result.outputs, SizeIs(2));

  {
    std::vector<int32_t> out_data(6);
    tsl::Future<> future = result.outputs[0]->CopyToHostBuffer(
        out_data.data(), /*byte_strides=*/std::nullopt,
        xla::ifrt::ArrayCopySemantics::kAlwaysCopy);
    TF_ASSERT_OK(future.Await());
    EXPECT_THAT(out_data, ElementsAre(0, 2, 4, 6, 8, 10));
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto output_shards,
        result.outputs[1]->DisassembleIntoSingleDeviceArrays(
            xla::ifrt::ArrayCopySemantics::kDonateInput,
            xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));
    ASSERT_THAT(output_shards, SizeIs(2));
    {
      std::vector<int32_t> out_data(3);
      auto future = output_shards[0]->CopyToHostBuffer(
          out_data.data(), /*byte_strides=*/std::nullopt,
          xla::ifrt::ArrayCopySemantics::kAlwaysCopy);
      TF_ASSERT_OK(future.Await());
      EXPECT_THAT(out_data, testing::Each(0));
    }
    {
      std::vector<float> out_data(3);
      auto future = output_shards[1]->CopyToHostBuffer(
          out_data.data(), /*byte_strides=*/std::nullopt,
          xla::ifrt::ArrayCopySemantics::kAlwaysCopy);
      TF_ASSERT_OK(future.Await());
      EXPECT_THAT(out_data, testing::Each(0));
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
