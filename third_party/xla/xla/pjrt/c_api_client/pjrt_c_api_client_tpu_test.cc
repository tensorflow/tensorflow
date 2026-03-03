/* Copyright 2025 The OpenXLA Authors.

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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_callback_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Pair;

using PjRtDeviceDimensionsAndInt = std::pair<PjRtDeviceDimensions, int32_t>;

// Helper to get a TPU topology description.
absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetTpuTopology() {
  return GetCApiTopology("tpu", "TPU v2:4x4");
}

TEST(PjRtCApiTopologyDescriptionTpuTest, IsSubsliceTopology) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  // The default TPU topology is not a subslice.
  EXPECT_THAT(topology->is_subslice_topology(), false);
}

TEST(PjRtCApiTopologyDescriptionTpuTest, SubsliceTopology) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  PjRtDeviceDimensions chips_per_host_bounds = {2, 2, 1};
  PjRtDeviceDimensions host_bounds = {1, 1, 1};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtTopologyDescription> subslice_topology,
      topology->Subslice(chips_per_host_bounds, host_bounds));
  EXPECT_THAT(subslice_topology->is_subslice_topology(), true);
  EXPECT_THAT(subslice_topology->DeviceDescriptions().size(), 8);
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ProcessCount) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  // Assuming a single process for a default test setup.
  EXPECT_THAT(topology->ProcessCount(), IsOkAndHolds(4));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ChipsPerProcess) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->ChipsPerProcess(), IsOkAndHolds(4));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, CoreCountOfDefaultTypePerChip) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  // TPU chips typically have 2 cores of the default type (TensorCores).
  EXPECT_THAT(topology->CoreCountOfDefaultTypePerChip(), IsOkAndHolds(2));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ToProto) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(PjRtTopologyDescriptionProto proto,
                          topology->ToProto());
  EXPECT_EQ(proto.platform_name(), "tpu");
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ChipCount) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->ChipCount(), IsOkAndHolds(16));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, CoreCountOfDefaultType) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->CoreCountOfDefaultType(), IsOkAndHolds(32));
}

TEST(PjRtCApiTopologyDescriptionTpuTest,
     LogicalDeviceCountOfDefaultTypePerProcess) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->LogicalDeviceCountOfDefaultTypePerProcess(),
              IsOkAndHolds(8));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, LogicalDeviceCountOfDefaultType) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->LogicalDeviceCountOfDefaultType(), IsOkAndHolds(32));
}

TEST(PjRtCApiTopologyDescriptionTpuTest,
     LogicalDeviceCountOfDefaultTypePerChip) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->LogicalDeviceCountOfDefaultTypePerChip(),
              IsOkAndHolds(2));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, CoreCountOfDefaultTypePerProcess) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->CoreCountOfDefaultTypePerProcess(), IsOkAndHolds(8));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ProcessIds) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(PjRtIdContainer<ProcessId> process_ids,
                          topology->ProcessIds());
  EXPECT_THAT(process_ids, ElementsAre(ProcessId(0), ProcessId(1), ProcessId(2),
                                       ProcessId(3)));
}

TEST(PjRtCApiTopologyDescriptionTpuTest,
     LogicalDeviceOfDefaultTypeIdsOnProcess) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtIdContainer<GlobalDeviceId> device_ids,
      topology->LogicalDeviceOfDefaultTypeIdsOnProcess(ProcessId(0)));
  EXPECT_THAT(device_ids, ElementsAre(GlobalDeviceId(0), GlobalDeviceId(1),
                                      GlobalDeviceId(2), GlobalDeviceId(3),
                                      GlobalDeviceId(8), GlobalDeviceId(9),
                                      GlobalDeviceId(10), GlobalDeviceId(11)));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ProcessIdAndIndexOnProcessForChip) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->ProcessIdAndIndexOnProcessForChip(GlobalChipId(2)),
              IsOkAndHolds(Pair(ProcessId(1), 0)));
}

TEST(PjRtCApiTopologyDescriptionTpuTest,
     ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType(
                  GlobalDeviceId(3)),
              IsOkAndHolds(Pair(ProcessId(0), 3)));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ProcessCoordFromId) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(PjRtDeviceDimensions coords,
                          topology->ProcessCoordFromId(ProcessId(2)));
  EXPECT_THAT(coords, (PjRtDeviceDimensions{0, 1, 0}));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ChipIdFromCoord) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->ChipIdFromCoord({1, 0, 0}),
              IsOkAndHolds(GlobalChipId(1)));
}

TEST(PjRtCApiTopologyDescriptionTpuTest,
     LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  EXPECT_THAT(topology->LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex(
                  {1, 1, 0}, 0),
              IsOkAndHolds(GlobalDeviceId(10)));
}

TEST(PjRtCApiTopologyDescriptionTpuTest,
     ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(
      const PjRtDeviceDimensionsAndInt& result,
      topology->ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(10)));
  EXPECT_THAT(absl::MakeConstSpan(result.first.data(), result.first.size()),
              ElementsAre(1, 1, 0));
  EXPECT_EQ(result.second, 0);
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ChipsPerProcessBounds) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(PjRtDeviceDimensions bounds,
                          topology->ChipsPerProcessBounds());
  EXPECT_THAT(bounds, (PjRtDeviceDimensions{2, 2, 1}));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ChipBounds) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(PjRtDeviceDimensions bounds, topology->ChipBounds());
  EXPECT_THAT(bounds, (PjRtDeviceDimensions{4, 4, 1}));
}

TEST(PjRtCApiTopologyDescriptionTpuTest, ProcessBounds) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  TF_ASSERT_OK_AND_ASSIGN(PjRtDeviceDimensions bounds,
                          topology->ProcessBounds());
  EXPECT_THAT(bounds, (PjRtDeviceDimensions{2, 2, 1}));
}

TEST(PjRtCApiClientTpuTest, RegisterAndInvokeCallback) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetXlaPjrtTpuClient());
  PjRtCApiClient* c_api_client = absl::down_cast<PjRtCApiClient*>(client.get());
  ASSERT_NE(c_api_client, nullptr);

  bool callback_executed = false;
  absl::Status callback_status;
  auto user_callback = [&](PJRT_Callback_PrefatalArgs* args) {
    callback_executed = true;
    callback_status = absl::Status(
        pjrt::PjrtErrorCodeToStatusCode(args->error_code),
        absl::string_view(args->error_message, args->error_message_size));
  };

  ASSERT_OK(
      (c_api_client->RegisterCallback<PJRT_Callback_PrefatalArgs,
                                      PJRT_Callback_PrefatalArgs_STRUCT_SIZE>(
          PJRT_Callback_Type::PJRT_Callback_Type_Prefatal, user_callback)));

  absl::Status error_status = absl::InternalError("Test error");
  PJRT_Callback_PrefatalArgs args;
  args.struct_size = PJRT_Callback_PrefatalArgs_STRUCT_SIZE;
  args.error_code = pjrt::StatusCodeToPjrtErrorCode(error_status.code());
  args.error_message = error_status.message().data();
  args.error_message_size = error_status.message().size();
  ASSERT_OK(c_api_client->InvokeCallbacks(
      PJRT_Callback_Type::PJRT_Callback_Type_Prefatal, &args));
  EXPECT_TRUE(callback_executed);
  EXPECT_EQ(callback_status, error_status);
}

TEST(PjRtCApiClientTpuTest, AbiVersion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetXlaPjrtTpuClient());
  PjRtCApiClient* c_api_client = absl::down_cast<PjRtCApiClient*>(client.get());
  ASSERT_NE(c_api_client, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtRuntimeAbiVersion> runtime_abi_version,
      client->RuntimeAbiVersion());

  TF_ASSERT_OK_AND_ASSIGN(auto runtime_proto, runtime_abi_version->ToProto());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtRuntimeAbiVersion> runtime_abi_version_from_proto,
      pjrt::CApiRuntimeAbiVersionFromProto(runtime_proto,
                                           c_api_client->pjrt_c_api()));

  // Test executable ABI version.
  ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnUnverifiedModule(R"(
HloModule Identity
ENTRY Identity() -> f32[2, 2] {
    ROOT %result = f32[2, 2] parameter(0)
})",
                                                                       {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  ASSERT_OK_AND_ASSIGN(auto executable,
                       client->CompileAndLoad(xla_computation, {}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutableAbiVersion> executable_abi_version,
      executable->GetExecutable()->GetAbiVersion());

  TF_ASSERT_OK_AND_ASSIGN(auto executable_proto,
                          executable_abi_version->ToProto());

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtExecutableAbiVersion>
                              executable_abi_version_from_proto,
                          pjrt::CApiExecutableAbiVersionFromProto(
                              executable_proto, c_api_client->pjrt_c_api()));

  EXPECT_OK(runtime_abi_version->IsCompatibleWith(
      *executable_abi_version_from_proto));
}

TEST(PjRtCApiClientTpuTest, DeviceAttributes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetXlaPjrtTpuClient());

  ASSERT_FALSE(client->addressable_devices().empty());
  for (PjRtDevice* device : client->addressable_devices()) {
    const auto& attributes = device->Attributes();
    EXPECT_TRUE(attributes.contains("physical_location"));
  }
}

TEST(PjRtCApiClientTpuTest, GetParameterAndOutputLayouts) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetXlaPjrtTpuClient());

  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  XlaBuilder builder("sum");
  auto inp_0 = Parameter(&builder, 0, shape, "input0");
  auto inp_1 = Parameter(&builder, 1, shape, "input1");
  auto sum = Add(inp_0, inp_1);
  auto computation = builder.Build(sum).value();

  CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client->CompileAndLoad(computation, options));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> parameter_layouts,
      executable->GetParameterLayouts());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const PjRtLayout>> output_layouts,
      executable->GetOutputLayouts());

  TF_ASSERT_OK_AND_ASSIGN(
      xla::Layout expected_layout,
      client->GetDefaultLayout(shape.element_type(), shape.dimensions()));

  // We expect parameter and output layouts to be a default layout because we
  // didn't specify any layouts in the HLO.
  EXPECT_EQ(parameter_layouts.size(), 2);
  for (const std::shared_ptr<const PjRtLayout>& parameter_layout :
       parameter_layouts) {
    EXPECT_EQ(parameter_layout->xla_layout(), expected_layout);
  }
  EXPECT_EQ(output_layouts.size(), 1);
  for (const std::shared_ptr<const PjRtLayout>& output_layout :
       output_layouts) {
    EXPECT_EQ(output_layout->xla_layout(), expected_layout);
  }
}

TEST(PjRtCApiClientTpuTest, CompileMlirModule) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetXlaPjrtTpuClient());
  constexpr char kProgram[] = "func.func @main() {return}";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtExecutable> executable,
                          client->Compile(*module, options));
  EXPECT_NE(executable.get(), nullptr);
}

TEST(PjRtCApiClientTpuTest, LoadExecutable) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetXlaPjrtTpuClient());
  constexpr char kProgram[] = "func.func @main() {return}";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtExecutable> executable,
                          client->Compile(*module, options));
  ASSERT_NE(executable.get(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> loaded_executable,
      client->Load(std::move(executable), LoadOptions{}));
  ASSERT_NE(loaded_executable.get(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
      loaded_executable->Execute(/*argument_handles=*/{{}}, ExecuteOptions()));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].size(), 0);
}

TEST(PjRtCApiClientTpuTest, LoadSameExecutableTwice) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetXlaPjrtTpuClient());
  constexpr char kProgram[] = "func.func @main() {return}";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(kProgram, context));
  CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<PjRtExecutable> executable,
                          client->Compile(*module, options));
  ASSERT_NE(executable.get(), nullptr);

  // Load the executable twice.
  {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtLoadedExecutable> loaded_executable,
        client->Load(executable, LoadOptions{}));
    ASSERT_NE(loaded_executable.get(), nullptr);

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
        loaded_executable->Execute(/*argument_handles=*/{{}},
                                   ExecuteOptions()));
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].size(), 0);
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtLoadedExecutable> loaded_executable,
        client->Load(executable, LoadOptions{}));
    ASSERT_NE(loaded_executable.get(), nullptr);

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
        loaded_executable->Execute(/*argument_handles=*/{{}},
                                   ExecuteOptions()));
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].size(), 0);
  }
}

}  // namespace
}  // namespace xla
