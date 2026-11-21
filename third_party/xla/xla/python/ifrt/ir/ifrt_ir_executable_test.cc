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

#include "xla/python/ifrt/ir/ifrt_ir_executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compilation/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/compilation/compiler.h"
#include "xla/python/ifrt/ir/compilation/mock_atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::SizeIs;
using ::testing::StrictMock;
using ::testing::UnorderedElementsAre;

// Non-IFRT-IR `Program` subtype for negative `Compiler::Compile` tests.
struct WrongTypeProgram : RTTIExtends<WrongTypeProgram, Program> {
  static char ID;  // NOLINT
};

char WrongTypeProgram::ID = 0;

// Non-IFRT-IR `CompileOptions` subtype for negative `Compiler::Compile` tests.
struct WrongTypeCompileOptions
    : RTTIExtends<WrongTypeCompileOptions, CompileOptions> {
  static char ID;  // NOLINT
};

char WrongTypeCompileOptions::ID = 0;

// Minimal concrete `Topology` for `IfrtIrProgramCompiler::Compile` unit tests.
class TestTopology final : public RTTIExtends<TestTopology, Topology> {
 public:
  absl::string_view platform_name() const override { return "cpu"; }
  absl::string_view platform_version() const override { return "test"; }
  PjRtPlatformId platform_id() const override { return 0; }
  const std::shared_ptr<const xla::PjRtTopologyDescription>& description()
      const override {
    return description_;
  }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    return {};
  }
  absl::StatusOr<xla::Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return xla::LayoutUtil::MakeDescendingLayout(dims.size());
  }
  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }
  const AttributeMap& Attributes() const override { return attributes_; }

  static char ID;  // NOLINT

 private:
  std::shared_ptr<const xla::PjRtTopologyDescription> description_;
  AttributeMap attributes_ = AttributeMap(AttributeMap::Map());
};

char TestTopology::ID = 0;

// Creates a mock unloaded `Executable` representing a compiled atom program.
// `static_name` must refer to storage that outlives the returned mock (e.g. a
// string literal).
ExecutableRef MakeMockUnloadedAtomExecutable(
    absl::string_view static_name,
    std::shared_ptr<const xla::PjRtLayout> layout,
    std::shared_ptr<xla::HloModule> hlo_module) {
  auto executable = std::make_shared<NiceMock<MockExecutable>>();
  ON_CALL(*executable, name()).WillByDefault(Return(static_name));
  ON_CALL(*executable, Fingerprint())
      .WillByDefault(Return(std::optional<std::string>("test_fp")));
  ON_CALL(*executable, Serialize())
      .WillByDefault(Return(std::string("serialized_unloaded_atom")));
  ON_CALL(*executable, num_devices()).WillByDefault(Return(2));
  ON_CALL(*executable, SizeOfGeneratedCodeInBytes()).WillByDefault(Return(64));
  ON_CALL(*executable, GetParameterShardings())
      .WillByDefault(Return(std::vector<xla::OpSharding>{xla::OpSharding()}));
  ON_CALL(*executable, GetOutputShardings())
      .WillByDefault(Return(std::vector<xla::OpSharding>{xla::OpSharding()}));
  if (layout == nullptr) {
    ON_CALL(*executable, GetParameterLayouts())
        .WillByDefault(
            Return(absl::UnimplementedError("Layouts not supported")));
    ON_CALL(*executable, GetOutputLayouts())
        .WillByDefault(
            Return(absl::UnimplementedError("Layouts not supported")));
  } else {
    ON_CALL(*executable, GetParameterLayouts())
        .WillByDefault(Return(
            std::vector<std::shared_ptr<const xla::PjRtLayout>>{layout}));
    ON_CALL(*executable, GetOutputLayouts())
        .WillByDefault(
            Return(std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                std::move(layout)}));
  }
  ON_CALL(*executable, GetOutputMemoryKinds())
      .WillByDefault(
          Return(std::vector<std::vector<absl::string_view>>{{"device"}}));
  if (hlo_module == nullptr) {
    ON_CALL(*executable, GetCompiledMemoryStats())
        .WillByDefault(
            Return(absl::UnimplementedError("Memory stats not supported")));
    ON_CALL(*executable, GetHloModules())
        .WillByDefault(
            Return(absl::UnimplementedError("HloModules not supported")));
    ON_CALL(*executable, GetCostAnalysis())
        .WillByDefault(
            Return(absl::UnimplementedError("Cost analysis not supported")));
  } else {
    xla::CompiledMemoryStats stats;
    stats.argument_size_in_bytes = 16;
    stats.output_size_in_bytes = 16;
    ON_CALL(*executable, GetCompiledMemoryStats()).WillByDefault(Return(stats));
    ON_CALL(*executable, GetHloModules())
        .WillByDefault(Return(std::vector<std::shared_ptr<xla::HloModule>>{
            std::move(hlo_module)}));
    AttributeMap::Map cost_map;
    cost_map.insert({"flops", AttributeMap::FloatValue(128.0f)});
    ON_CALL(*executable, GetCostAnalysis())
        .WillByDefault(Return(AttributeMap(std::move(cost_map))));
  }
  return executable;
}

class IfrtIrExecutableTest : public test_util::IfrtIrTestBase {
 protected:
  void SetUp() override {
    layout_ = std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(2));

    ON_CALL(device0_, Id()).WillByDefault(Return(DeviceId(10)));
    ON_CALL(device0_, PlatformName()).WillByDefault(Return("cpu"));
    ON_CALL(device1_, Id()).WillByDefault(Return(DeviceId(11)));
    ON_CALL(device1_, PlatformName()).WillByDefault(Return("cpu"));

    all_devices_ = {&device0_, &device1_};
    device_list_ = BasicDeviceList::Create(all_devices_);

    ON_CALL(client_, Attributes()).WillByDefault(ReturnRef(client_attributes_));
    ON_CALL(client_, LookupDevice(DeviceId(10)))
        .WillByDefault(Return(&device0_));
    ON_CALL(client_, LookupDevice(DeviceId(11)))
        .WillByDefault(Return(&device1_));
    ON_CALL(client_, MakeDeviceList(testing::_))
        .WillByDefault([](absl::Span<Device* const> devs) {
          return BasicDeviceList::Create(devs);
        });
    ON_CALL(client_, GetDefaultPjRtLayout(testing::_, testing::_, testing::_,
                                          testing::_))
        .WillByDefault(Return(layout_));
    ON_CALL(client_, DeleteValues(testing::_))
        .WillByDefault(Return(tsl::Future<>(absl::OkStatus())));
  }

  // Configures default compile-only (`CompileXla` and `CompileMpmdReshard`)
  // actions on `compiler`. With `StrictMock`, tests must still register
  // explicit `EXPECT_CALL`s for expected compilations, and any unexpected
  // `CompileAndLoad*` call fails the test.
  void ConfigureDefaultCompileOnlyActions(MockAtomProgramCompiler& compiler) {
    ON_CALL(compiler, CompileXla)
        .WillByDefault(
            [this](std::unique_ptr<HloProgram> computation, xla::CompileOptions,
                   std::vector<tsl::RCReference<LoadedHostCallback>>) {
              auto hlo_module = std::make_shared<xla::HloModule>(
                  std::string(computation->name()), xla::HloModuleConfig());
              ExecutableRef unloaded_executable =
                  MakeMockUnloadedAtomExecutable("fake_xla_atom", layout_,
                                                 std::move(hlo_module));
              auto load_options = std::make_unique<const LoadOptions>(
                  device_list_, /*outputs_bundle_slice_sizes=*/std::nullopt);
              return std::make_shared<const CompiledAtomProgram>(
                  CompiledAtomProgram{
                      /*executable=*/std::move(unloaded_executable),
                      /*load_options=*/std::move(load_options),
                  });
            });
    ON_CALL(compiler, CompileMpmdReshard)
        .WillByDefault([this](std::vector<DType>, std::vector<Shape>,
                              std::vector<IfrtArrayType>,
                              std::vector<IfrtArrayType>) {
          ExecutableRef unloaded_reshard = MakeMockUnloadedAtomExecutable(
              "mpmd_reshard", /*layout=*/nullptr, /*hlo_module=*/nullptr);
          auto load_options = std::make_unique<const LoadOptions>(
              device_list_, /*outputs_bundle_slice_sizes=*/std::nullopt);
          return std::make_shared<const CompiledAtomProgram>(
              CompiledAtomProgram{
                  /*executable=*/std::move(unloaded_reshard),
                  /*load_options=*/std::move(load_options),
              });
        });
  }

  std::unique_ptr<StrictMock<MockAtomProgramCompiler>>
  MakeUniqueStrictAtomCompiler() {
    auto compiler = std::make_unique<StrictMock<MockAtomProgramCompiler>>();
    ConfigureDefaultCompileOnlyActions(*compiler);
    return compiler;
  }

  std::shared_ptr<StrictMock<MockAtomProgramCompiler>>
  MakeSharedStrictAtomCompiler() {
    auto compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
    ConfigureDefaultCompileOnlyActions(*compiler);
    return compiler;
  }

  // Creates a mock loaded atom executable with serialization and layout stubs.
  std::shared_ptr<NiceMock<MockLoadedExecutable>> MakeLoadedAtomExecutable(
      absl::string_view module_name) {
    auto executable = std::make_shared<NiceMock<MockLoadedExecutable>>();
    auto hlo_module = std::make_shared<xla::HloModule>(std::string(module_name),
                                                       xla::HloModuleConfig());
    xla::OpSharding tile_sharding =
        xla::HloSharding::IotaTile({2, 1}).ToProto();
    ON_CALL(*executable, client()).WillByDefault(Return(&client_));
    ON_CALL(*executable, devices())
        .WillByDefault(Return(std::optional<DeviceListRef>(device_list_)));
    ON_CALL(*executable, Serialize())
        .WillByDefault(Return(absl::StrCat("serialized_", module_name)));
    ON_CALL(*executable, GetParameterShardings())
        .WillByDefault(Return(
            std::optional<std::vector<xla::OpSharding>>{{tile_sharding}}));
    ON_CALL(*executable, GetOutputShardings())
        .WillByDefault(Return(
            std::optional<std::vector<xla::OpSharding>>{{tile_sharding}}));
    ON_CALL(*executable, GetParameterLayouts())
        .WillByDefault(Return(
            std::vector<std::shared_ptr<const xla::PjRtLayout>>{layout_}));
    ON_CALL(*executable, GetOutputLayouts())
        .WillByDefault(Return(
            std::vector<std::shared_ptr<const xla::PjRtLayout>>{layout_}));
    ON_CALL(*executable, GetHloModules())
        .WillByDefault(
            Return(std::vector<std::shared_ptr<xla::HloModule>>{hlo_module}));
    return executable;
  }

  AttributeMap client_attributes_ = AttributeMap(AttributeMap::Map());
  NiceMock<MockClient> client_;
  NiceMock<MockDevice> device0_;
  NiceMock<MockDevice> device1_;
  std::vector<Device*> all_devices_;
  DeviceListRef device_list_;
  std::shared_ptr<const xla::PjRtLayout> layout_;
};

TEST_F(IfrtIrExecutableTest,
       CompilesWithoutLoadingAndSupportsConcurrentMultiLoadSerialization) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @compile_only_module {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @bound_op(%arg0) : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0,1]
        : (!array) -> !array
    return %1 : !array
  }

  ifrt.LoadedExecutable @bound_op on devices [0,1] : (!array) -> !array

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  std::unique_ptr<StrictMock<MockAtomProgramCompiler>> atom_compiler =
      MakeUniqueStrictAtomCompiler();
  EXPECT_CALL(*atom_compiler, CompileXla).Times(1);
  IfrtIrProgramCompiler compiler(
      &client_,
      [&](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::move(atom_compiler);
      });

  std::shared_ptr<NiceMock<MockLoadedExecutable>> bound_mock =
      MakeLoadedAtomExecutable("bound_op");
  LoadedExecutableRef bound_executable = bound_mock;
  absl::flat_hash_map<std::string, LoadedExecutableRef> bindings;
  bindings.insert({"bound_op", bound_executable});
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)}, std::move(bindings));

  TestTopology topology;
  ASSERT_OK_AND_ASSIGN(ExecutableRef executable,
                       compiler
                           .Compile(std::make_unique<IfrtIRProgram>(*module),
                                    topology, std::move(compile_options))
                           .Await());

  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_EQ(ifrt_ir_executable->name(), "compile_only_module");
  EXPECT_EQ(ifrt_ir_executable->mlir_module().getName(), "compile_only_module");
  EXPECT_EQ(ifrt_ir_executable->devices(), device_list_);
  EXPECT_THAT(ifrt_ir_executable->compile_options().device_assignments,
              ElementsAre(DeviceId(10), DeviceId(11)));
  EXPECT_EQ(ifrt_ir_executable->num_devices(), 2);
  EXPECT_EQ(ifrt_ir_executable->SizeOfGeneratedCodeInBytes(), -1);
  ASSERT_OK_AND_ASSIGN(std::optional<std::string> fingerprint,
                       ifrt_ir_executable->Fingerprint());
  EXPECT_EQ(fingerprint, std::nullopt);
  EXPECT_THAT(ifrt_ir_executable->Serialize(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Serialize is not implemented")));
  EXPECT_THAT(ifrt_ir_executable->GetCompiledMemoryStats(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("GetCompiledMemoryStats")));
  EXPECT_THAT(
      ifrt_ir_executable->GetCostAnalysis(),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("GetCostAnalysis")));
  EXPECT_THAT(ifrt_ir_executable->donatable_input_indices(), ElementsAre(0));
  EXPECT_THAT(ifrt_ir_executable->bound_atom_program_executables(), SizeIs(1));
  ASSERT_THAT(ifrt_ir_executable->compiled_atom_programs(), SizeIs(1));
  const std::optional<std::vector<xla::OpSharding>> param_shardings =
      ifrt_ir_executable->GetParameterShardings();
  ASSERT_TRUE(param_shardings.has_value());
  ASSERT_THAT(*param_shardings, SizeIs(1));
  ASSERT_OK_AND_ASSIGN(xla::HloSharding param_hlo_sharding,
                       xla::HloSharding::FromProto((*param_shardings)[0]));
  EXPECT_EQ(param_hlo_sharding, xla::HloSharding::IotaTile({2, 1}));

  const std::optional<std::vector<xla::OpSharding>> output_shardings =
      ifrt_ir_executable->GetOutputShardings();
  ASSERT_TRUE(output_shardings.has_value());
  ASSERT_THAT(*output_shardings, SizeIs(1));
  ASSERT_OK_AND_ASSIGN(xla::HloSharding output_hlo_sharding,
                       xla::HloSharding::FromProto((*output_shardings)[0]));
  EXPECT_EQ(output_hlo_sharding, xla::HloSharding::IotaTile({2, 1}));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> param_layouts,
      ifrt_ir_executable->GetParameterLayouts());
  EXPECT_THAT(param_layouts, ElementsAre(layout_));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts,
      ifrt_ir_executable->GetOutputLayouts());
  EXPECT_THAT(output_layouts, ElementsAre(layout_));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<absl::string_view>> output_memory_kinds,
      ifrt_ir_executable->GetOutputMemoryKinds());
  ASSERT_THAT(output_memory_kinds, SizeIs(1));
  EXPECT_THAT(output_memory_kinds.front(), ElementsAre("device"));

  const auto& [compiled_atom_name, compiled_atom] =
      *ifrt_ir_executable->compiled_atom_programs().begin();
  ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<xla::HloModule>> hlo_modules,
                       ifrt_ir_executable->GetHloModules());
  ASSERT_THAT(hlo_modules, SizeIs(2));
  const std::vector<std::string> hlo_module_names = {hlo_modules[0]->name(),
                                                     hlo_modules[1]->name()};
  EXPECT_THAT(hlo_module_names, UnorderedElementsAre("bound_op", "add_one"));

  ASSERT_NE(compiled_atom, nullptr);
  EXPECT_NE(compiled_atom->executable, nullptr);
  ASSERT_NE(compiled_atom->load_options, nullptr);
  EXPECT_EQ(compiled_atom->load_options->devices, device_list_);
  std::shared_ptr<NiceMock<MockLoadedExecutable>> loaded_add_one_1 =
      MakeLoadedAtomExecutable("loaded_add_one_1");
  std::shared_ptr<NiceMock<MockLoadedExecutable>> loaded_add_one_2 =
      MakeLoadedAtomExecutable("loaded_add_one_2");
  AtomExecutableMap load1_atoms;
  load1_atoms.insert({compiled_atom_name, loaded_add_one_1});
  AtomExecutableMap load2_atoms;
  load2_atoms.insert({compiled_atom_name, loaded_add_one_2});

  absl::StatusOr<std::shared_ptr<CompiledIfrtIrProgram>>
      compiled_program_1_status;
  absl::StatusOr<std::shared_ptr<CompiledIfrtIrProgram>>
      compiled_program_2_status;
  {
    constexpr int kNumConcurrentLoadThreads = 2;
    absl::Barrier load_start_barrier(kNumConcurrentLoadThreads);
    // `ifrt_ir_executable`, `client_`, `load1_atoms`, `load2_atoms`,
    // `load_start_barrier`, and the result variables outlive `load_thread_1`
    // and `load_thread_2` because `~Thread()` joins on scope exit.
    std::unique_ptr<tsl::Thread> load_thread_1 =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "create_compiled_program_1", [&]() {
              load_start_barrier.Block();
              compiled_program_1_status =
                  ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                      &client_, std::move(load1_atoms));
            }));
    std::unique_ptr<tsl::Thread> load_thread_2 =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "create_compiled_program_2", [&]() {
              load_start_barrier.Block();
              compiled_program_2_status =
                  ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                      &client_, std::move(load2_atoms));
            }));
  }
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<CompiledIfrtIrProgram> compiled_program_1,
      std::move(compiled_program_1_status));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<CompiledIfrtIrProgram> compiled_program_2,
      std::move(compiled_program_2_status));

  // Release `IfrtIrExecutable` before using or destroying the loaded programs
  // to verify each `CompiledIfrtIrProgram` owns an independent `MLIRContext`.
  executable.reset();

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_executable_1,
      IfrtIrLoadedExecutable::Create(&client_, compiled_program_1));
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_executable_2,
      IfrtIrLoadedExecutable::Create(&client_, compiled_program_2));

  absl::StatusOr<std::string> serialized_1;
  absl::StatusOr<std::string> serialized_2;
  {
    constexpr int kNumConcurrentSerializeThreads = 2;
    absl::Barrier serialize_start_barrier(kNumConcurrentSerializeThreads);
    // `loaded_executable_1`, `loaded_executable_2`, `serialize_start_barrier`,
    // `serialized_1`, and `serialized_2` outlive `thread_1` and `thread_2`
    // because `~Thread()` joins on scope exit.
    std::unique_ptr<tsl::Thread> thread_1 =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "serialize_1", [&]() {
              serialize_start_barrier.Block();
              serialized_1 = loaded_executable_1->Serialize();
            }));
    std::unique_ptr<tsl::Thread> thread_2 =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "serialize_2", [&]() {
              serialize_start_barrier.Block();
              serialized_2 = loaded_executable_2->Serialize();
            }));
  }
  ASSERT_OK(serialized_1);
  ASSERT_OK(serialized_2);
  EXPECT_FALSE(serialized_1->empty());
  EXPECT_FALSE(serialized_2->empty());

  ArrayRef input_1 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef bound_out_1 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef final_out_1 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef input_2 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef bound_out_2 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef final_out_2 = tsl::MakeRef<NiceMock<MockArray>>();

  EXPECT_CALL(*bound_mock,
              Execute(ElementsAre(input_1), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{bound_out_1},
      }));
  EXPECT_CALL(*loaded_add_one_1,
              Execute(ElementsAre(bound_out_1), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{final_out_1},
      }));
  EXPECT_CALL(*bound_mock,
              Execute(ElementsAre(input_2), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{bound_out_2},
      }));
  EXPECT_CALL(*loaded_add_one_2,
              Execute(ElementsAre(bound_out_2), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{final_out_2},
      }));

  LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = true;
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult exec_result_1,
      loaded_executable_1->Execute(absl::MakeSpan(&input_1, 1), execute_options,
                                   /*devices=*/std::nullopt));
  ASSERT_OK(exec_result_1.status.Await());
  EXPECT_THAT(exec_result_1.outputs, ElementsAre(final_out_1));

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult exec_result_2,
      loaded_executable_2->Execute(absl::MakeSpan(&input_2, 1), execute_options,
                                   /*devices=*/std::nullopt));
  ASSERT_OK(exec_result_2.status.Await());
  EXPECT_THAT(exec_result_2.outputs, ElementsAre(final_out_2));

  // Clearing `loaded_exec_binding` on load 1 must not mutate load 2's bindings.
  compiled_program_1->compile_options->loaded_exec_binding.clear();
  EXPECT_THAT(compiled_program_2->compile_options->loaded_exec_binding,
              SizeIs(1));
}

TEST_F(IfrtIrExecutableTest, CompilesMpmdReshardWithoutLoading) {
  constexpr absl::string_view kSource = R"mlir(
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                      [0]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
module @mpmd_reshard_module {
  func.func @main(%arg0: !array0) -> (!array1) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0,1]
        : (!array1) -> !array1
    return %1 : !array1
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  std::shared_ptr<StrictMock<MockAtomProgramCompiler>> atom_compiler =
      MakeSharedStrictAtomCompiler();
  EXPECT_CALL(*atom_compiler, CompileXla).Times(1);
  EXPECT_CALL(*atom_compiler, CompileMpmdReshard).Times(1);
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)});

  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(std::make_unique<IfrtIRProgram>(*module),
                               std::move(compile_options), &client_,
                               std::move(atom_compiler))
          .Await());

  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_TRUE(ifrt_ir_executable->bound_atom_program_executables().empty());
  ASSERT_THAT(ifrt_ir_executable->compiled_atom_programs(), SizeIs(2));
  EXPECT_THAT(ifrt_ir_executable->GetParameterLayouts(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Layouts not supported")));
  EXPECT_THAT(ifrt_ir_executable->GetHloModules(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("HloModules not supported")));

  AtomExecutableMap partial_loaded_atoms;
  partial_loaded_atoms.insert(
      {ifrt_ir_executable->compiled_atom_programs().begin()->first,
       MakeLoadedAtomExecutable("partial_atom")});
  EXPECT_THAT(ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                  &client_, std::move(partial_loaded_atoms)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 2 loaded atom executables")));

  AtomExecutableMap loaded_atoms;
  for (const auto& [name, unused_compiled_atom] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       ifrt_ir_executable->compiled_atom_programs()) {
    loaded_atoms.insert({name, MakeLoadedAtomExecutable(name)});
  }
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<CompiledIfrtIrProgram> compiled_program,
                       ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                           &client_, std::move(loaded_atoms)));
  EXPECT_THAT(*compiled_program->atom_program_executables, SizeIs(2));
}

TEST_F(IfrtIrExecutableTest, RejectsInvalidCreateCompiledIfrtIrProgramInputs) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @validation_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  std::shared_ptr<StrictMock<MockAtomProgramCompiler>> atom_compiler =
      MakeSharedStrictAtomCompiler();
  EXPECT_CALL(*atom_compiler, CompileXla).Times(1);
  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_, std::move(atom_compiler))
          .Await());
  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  ASSERT_THAT(ifrt_ir_executable->compiled_atom_programs(), SizeIs(1));
  const std::string compiled_atom_name =
      ifrt_ir_executable->compiled_atom_programs().begin()->first;

  NiceMock<MockClient> other_client;
  AtomExecutableMap other_client_atoms;
  other_client_atoms.insert(
      {compiled_atom_name, MakeLoadedAtomExecutable("loaded_add_one")});
  EXPECT_THAT(ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                  &other_client, std::move(other_client_atoms)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("different Client")));

  AtomExecutableMap wrong_name_atoms;
  wrong_name_atoms.insert(
      {"unexpected_atom", MakeLoadedAtomExecutable("unexpected")});
  EXPECT_THAT(ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                  &client_, std::move(wrong_name_atoms)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected atom executable")));

  AtomExecutableMap null_atoms;
  null_atoms.insert({compiled_atom_name, nullptr});
  EXPECT_THAT(ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                  &client_, std::move(null_atoms)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must not be null")));
}

TEST_F(IfrtIrExecutableTest,
       OwnsLoweredModuleAfterCallerDestroysInputModulePostCompile) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @ephemeral_caller_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  std::shared_ptr<StrictMock<MockAtomProgramCompiler>> atom_compiler =
      MakeSharedStrictAtomCompiler();
  EXPECT_CALL(*atom_compiler, CompileXla).Times(1);
  ExecutableRef executable;
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    ASSERT_OK_AND_ASSIGN(
        executable, IfrtIrExecutable::Create(
                        std::make_unique<IfrtIRProgram>(*module),
                        std::make_unique<IfrtIRCompileOptions>(
                            std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                        &client_, std::move(atom_compiler))
                        .Await());
  }

  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_TRUE(ifrt_ir_executable->program()->OwnsMlirContext());
  ASSERT_THAT(ifrt_ir_executable->compiled_atom_programs(), SizeIs(1));
  const std::string compiled_atom_name =
      ifrt_ir_executable->compiled_atom_programs().begin()->first;
  AtomExecutableMap loaded_atoms;
  loaded_atoms.insert(
      {compiled_atom_name, MakeLoadedAtomExecutable("loaded_add_one")});

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<CompiledIfrtIrProgram> compiled_program,
                       ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                           &client_, std::move(loaded_atoms)));
  EXPECT_NE(compiled_program, nullptr);
}

TEST_F(IfrtIrExecutableTest,
       SupportsPreBoundMpmdReshardExecutableWithoutSymbolCollision) {
  constexpr absl::string_view kSource = R"mlir(
!in_array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                        [0,1]>
!out_array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                         [0,1], memory_kind = "pinned_host">
module @bound_mpmd_reshard_module {
  func.func @main(%arg0: !in_array) -> (!out_array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @bound_reshard(%arg0)
        {ifrt.module_type = "mpmd_reshard"} : (!in_array) -> !out_array
    return %0 : !out_array
  }

  ifrt.LoadedExecutable @bound_reshard on devices [0,1]
      : (!in_array) -> !out_array
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  absl::flat_hash_map<std::string, LoadedExecutableRef> bindings;
  bindings.insert({"bound_reshard", MakeLoadedAtomExecutable("bound_reshard")});
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)}, std::move(bindings));

  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(
          std::make_unique<IfrtIRProgram>(*module), std::move(compile_options),
          &client_, std::make_shared<StrictMock<MockAtomProgramCompiler>>())
          .Await());
  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_TRUE(ifrt_ir_executable->compiled_atom_programs().empty());
  EXPECT_THAT(ifrt_ir_executable->bound_atom_program_executables(), SizeIs(1));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<absl::string_view>> output_memory_kinds,
      ifrt_ir_executable->GetOutputMemoryKinds());
  ASSERT_THAT(output_memory_kinds, SizeIs(1));
  EXPECT_THAT(output_memory_kinds.front(), ElementsAre("pinned_host"));

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<CompiledIfrtIrProgram> compiled_program,
                       ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                           &client_, /*newly_loaded_atom_executables=*/{}));
  EXPECT_THAT(*compiled_program->atom_program_executables, SizeIs(1));
}

TEST_F(IfrtIrExecutableTest, RejectsNullLoadedExecBinding) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @null_binding_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  IfrtIrProgramCompiler compiler(
      &client_,
      [](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::make_unique<StrictMock<MockAtomProgramCompiler>>();
      });
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)},
      AtomExecutableMap{{"unused_bound", nullptr}});
  TestTopology topology;
  EXPECT_THAT(compiler
                  .Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                           std::move(compile_options))
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Bound executable `unused_bound` must not be "
                                 "null")));
}

TEST_F(IfrtIrExecutableTest, RejectsUnboundCallLoadedExecutable) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @unbound_call_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @missing_bound(%arg0)
        : (!array) -> !array
    return %0 : !array
  }

  ifrt.LoadedExecutable @missing_bound on devices [0,1] : (!array) -> !array
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  EXPECT_THAT(
      IfrtIrExecutable::Create(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_, std::make_shared<StrictMock<MockAtomProgramCompiler>>())
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Could not find bound or compiled executable "
                         "for `missing_bound`")));
}

TEST_F(IfrtIrExecutableTest,
       ResetsPartialParameterAndOutputLayoutsWhenOutputLayoutResolutionFails) {
  constexpr absl::string_view kSource = R"mlir(
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                      [0]>
module @partial_layout_reset_module {
  func.func @main(%arg0: !array0) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.Reshard(%0) after %ctrl_0 : (!array0) -> !array1
    return %0, %1 : !array0, !array1
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  std::shared_ptr<StrictMock<MockAtomProgramCompiler>> atom_compiler =
      MakeSharedStrictAtomCompiler();
  EXPECT_CALL(*atom_compiler, CompileXla).Times(1);
  EXPECT_CALL(*atom_compiler, CompileMpmdReshard).Times(1);
  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_, std::move(atom_compiler))
          .Await());
  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_THAT(ifrt_ir_executable->layout_status(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Layouts not supported")));
  ASSERT_THAT(ifrt_ir_executable->in_specs(), SizeIs(1));
  EXPECT_EQ(ifrt_ir_executable->in_specs()[0].layout, nullptr);
  ASSERT_THAT(ifrt_ir_executable->out_specs(), SizeIs(2));
  EXPECT_EQ(ifrt_ir_executable->out_specs()[0].layout, nullptr);
  EXPECT_EQ(ifrt_ir_executable->out_specs()[1].layout, nullptr);
  EXPECT_THAT(ifrt_ir_executable->GetParameterLayouts(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Layouts not supported")));
  EXPECT_THAT(ifrt_ir_executable->GetOutputLayouts(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Layouts not supported")));
}

TEST_F(IfrtIrExecutableTest,
       PreservesUserContextDuringAsyncCompileAndCompileAndLoad) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @user_context_module {
  func.func @main(%arg0: !array, %unused_arg1: !array) -> (!array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  // 1. Compile-only `Compiler::Compile(program, topology, options)`.
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    UserContextRef expected_context = BasicUserContext::Create("compile_ctx");
    UserContextRef observed_atom_context;
    UserContextRef observed_layout_context;
    EXPECT_CALL(client_, GetDefaultPjRtLayout(testing::_, testing::_,
                                              testing::_, testing::_))
        .WillOnce([&](DType, absl::Span<const int64_t>, Device*, MemoryKind) {
          observed_layout_context = UserContextScope::current();
          return layout_;
        });

    using CompiledAtomPtr =
        absl_nonnull std::shared_ptr<const CompiledAtomProgram>;
    std::pair<tsl::Promise<CompiledAtomPtr>, tsl::Future<CompiledAtomPtr>>
        atom_promise_and_future = tsl::MakePromise<CompiledAtomPtr>();
    absl::Notification compile_xla_called;
    auto atom_compiler =
        std::make_unique<StrictMock<MockAtomProgramCompiler>>();
    EXPECT_CALL(*atom_compiler, CompileXla)
        .WillOnce([&](std::unique_ptr<HloProgram>, xla::CompileOptions,
                      std::vector<tsl::RCReference<LoadedHostCallback>>) {
          observed_atom_context = UserContextScope::current();
          compile_xla_called.Notify();
          return std::move(atom_promise_and_future.second);
        });
    IfrtIrProgramCompiler compiler(
        &client_,
        [&](const IfrtIRCompileOptions& unused_opts)
            -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
          return std::move(atom_compiler);
        });
    TestTopology topology;
    tsl::Future<ExecutableRef> compile_future;
    {
      UserContextScope user_context_scope(expected_context);
      compile_future = compiler.Compile(
          std::make_unique<IfrtIRProgram>(*module), topology,
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}));
    }
    compile_xla_called.WaitForNotification();
    EXPECT_EQ(observed_atom_context, expected_context);

    {
      // Fulfill the atom compilation promise from a background thread with no
      // active `UserContextScope` to verify `JoinFutures(...).Map(...)`
      // restores `expected_context`. `layout_` and `device_list_` outlive
      // `fulfill_thread` because `~Thread()` joins on scope exit.
      std::unique_ptr<tsl::Thread> fulfill_thread =
          absl::WrapUnique(tsl::Env::Default()->StartThread(
              tsl::ThreadOptions(), "fulfill_atom_compile",
              [&,
               promise = std::move(atom_promise_and_future.first)]() mutable {
                ExecutableRef unloaded_executable =
                    MakeMockUnloadedAtomExecutable("fake_xla_atom", layout_,
                                                   /*hlo_module=*/nullptr);
                auto load_options = std::make_unique<const LoadOptions>(
                    device_list_, /*outputs_bundle_slice_sizes=*/std::nullopt);
                promise.Set(std::make_shared<const CompiledAtomProgram>(
                    CompiledAtomProgram{
                        /*executable=*/std::move(unloaded_executable),
                        /*load_options=*/std::move(load_options),
                    }));
              }));
    }

    ASSERT_OK_AND_ASSIGN(ExecutableRef executable, compile_future.Await());
    EXPECT_NE(executable, nullptr);
    EXPECT_EQ(observed_layout_context, expected_context);
  }

  // 2. Eager `Compiler::CompileAndLoad(program, options)`.
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    UserContextRef expected_context =
        BasicUserContext::Create("compile_and_load_ctx");
    UserContextRef observed_atom_context;
    UserContextRef observed_layout_context;
    EXPECT_CALL(client_, GetDefaultPjRtLayout(testing::_, testing::_,
                                              testing::_, testing::_))
        .WillOnce([&](DType, absl::Span<const int64_t>, Device*, MemoryKind) {
          observed_layout_context = UserContextScope::current();
          return layout_;
        });

    std::pair<tsl::Promise<LoadedExecutableRef>,
              tsl::Future<LoadedExecutableRef>>
        atom_promise_and_future = tsl::MakePromise<LoadedExecutableRef>();
    absl::Notification compile_and_load_xla_called;
    auto atom_compiler =
        std::make_unique<StrictMock<MockAtomProgramCompiler>>();
    EXPECT_CALL(*atom_compiler, CompileAndLoadXla)
        .WillOnce([&](std::unique_ptr<HloProgram>, xla::CompileOptions,
                      std::vector<tsl::RCReference<LoadedHostCallback>>) {
          observed_atom_context = UserContextScope::current();
          compile_and_load_xla_called.Notify();
          return std::move(atom_promise_and_future.second);
        });
    IfrtIrProgramCompiler compiler(
        &client_,
        [&](const IfrtIRCompileOptions& unused_opts)
            -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
          return std::move(atom_compiler);
        });
    tsl::Future<LoadedExecutableRef> compile_and_load_future;
    {
      UserContextScope user_context_scope(expected_context);
      compile_and_load_future = compiler.CompileAndLoad(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}));
    }
    compile_and_load_xla_called.WaitForNotification();
    EXPECT_EQ(observed_atom_context, expected_context);

    {
      // `this` outlives `fulfill_thread` because `~Thread()` joins on scope
      // exit.
      std::unique_ptr<tsl::Thread> fulfill_thread =
          absl::WrapUnique(tsl::Env::Default()->StartThread(
              tsl::ThreadOptions(), "fulfill_atom_compile_and_load",
              [this,
               promise = std::move(atom_promise_and_future.first)]() mutable {
                promise.Set(
                    LoadedExecutableRef(MakeLoadedAtomExecutable("add_one")));
              }));
    }

    ASSERT_OK_AND_ASSIGN(LoadedExecutableRef loaded_executable,
                         compile_and_load_future.Await());
    ASSERT_NE(loaded_executable, nullptr);
    EXPECT_EQ(observed_layout_context, expected_context);
    EXPECT_EQ(loaded_executable->user_context(), expected_context);
  }
}

TEST_F(IfrtIrExecutableTest, PropagatesAsyncAtomCompilationError) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @async_atom_error_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  using CompiledAtomPtr =
      absl_nonnull std::shared_ptr<const CompiledAtomProgram>;
  std::pair<tsl::Promise<CompiledAtomPtr>, tsl::Future<CompiledAtomPtr>>
      atom_promise_and_future = tsl::MakePromise<CompiledAtomPtr>();
  absl::Notification compile_xla_called;
  auto atom_compiler = std::make_unique<StrictMock<MockAtomProgramCompiler>>();
  EXPECT_CALL(*atom_compiler, CompileXla)
      .WillOnce([&](std::unique_ptr<HloProgram>, xla::CompileOptions,
                    std::vector<tsl::RCReference<LoadedHostCallback>>) {
        compile_xla_called.Notify();
        return std::move(atom_promise_and_future.second);
      });
  IfrtIrProgramCompiler compiler(
      &client_,
      [&](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::move(atom_compiler);
      });
  TestTopology topology;
  tsl::Future<ExecutableRef> compile_future =
      compiler.Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                       std::make_unique<IfrtIRCompileOptions>(
                           std::vector<DeviceId>{DeviceId(10), DeviceId(11)}));
  compile_xla_called.WaitForNotification();

  {
    // The promise is moved into `fulfill_thread` and joined on scope exit.
    std::unique_ptr<tsl::Thread> fulfill_thread =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "fail_atom_compile",
            [promise = std::move(atom_promise_and_future.first)]() mutable {
              promise.Set(absl::InternalError("atom compile boom"));
            }));
  }

  EXPECT_THAT(compile_future.Await(), StatusIs(absl::StatusCode::kInternal,
                                               HasSubstr("atom compile boom")));
}

TEST_F(IfrtIrExecutableTest, CompileRejectsNullOrWrongTypeProgram) {
  IfrtIrProgramCompiler compiler(
      &client_,
      [](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::make_unique<StrictMock<MockAtomProgramCompiler>>();
      });
  TestTopology topology;
  EXPECT_THAT(
      compiler
          .Compile(/*program=*/nullptr, topology,
                   std::make_unique<IfrtIRCompileOptions>(
                       std::vector<DeviceId>{DeviceId(10), DeviceId(11)}))
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("requires an IFRT IR program")));
  EXPECT_THAT(
      compiler
          .Compile(std::make_unique<WrongTypeProgram>(), topology,
                   std::make_unique<IfrtIRCompileOptions>(
                       std::vector<DeviceId>{DeviceId(10), DeviceId(11)}))
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("requires an IFRT IR program")));
}

TEST_F(IfrtIrExecutableTest, CompileRejectsNullOrWrongTypeCompileOptions) {
  constexpr absl::string_view kSource = R"mlir(
module @null_options_module {
  func.func @main() attributes {ifrt.function} {
    return
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  IfrtIrProgramCompiler compiler(
      &client_,
      [](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::make_unique<StrictMock<MockAtomProgramCompiler>>();
      });
  TestTopology topology;
  EXPECT_THAT(compiler
                  .Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                           /*options=*/nullptr)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("options must be IfrtIRCompileOptions")));
  EXPECT_THAT(compiler
                  .Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                           std::make_unique<WrongTypeCompileOptions>())
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("options must be IfrtIRCompileOptions")));
}

TEST_F(IfrtIrExecutableTest, CompilePropagatesAtomProgramCompilerFactoryError) {
  constexpr absl::string_view kSource = R"mlir(
module @factory_error_module {
  func.func @main() attributes {ifrt.function} {
    return
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  IfrtIrProgramCompiler compiler(
      &client_,
      [](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return absl::FailedPreconditionError("atom compiler factory failed");
      });
  TestTopology topology;
  EXPECT_THAT(
      compiler
          .Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                   std::make_unique<IfrtIRCompileOptions>(
                       std::vector<DeviceId>{DeviceId(10), DeviceId(11)}))
          .Await(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("atom compiler factory failed")));
}

TEST_F(IfrtIrExecutableTest,
       CreateRejectsNullModuleAndInvalidMainFunctionAtCompileTime) {
  auto atom_compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  auto make_options = []() {
    return std::make_unique<IfrtIRCompileOptions>(
        std::vector<DeviceId>{DeviceId(10), DeviceId(11)});
  };

  EXPECT_THAT(IfrtIrExecutable::Create(std::make_unique<IfrtIRProgram>(),
                                       make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("non-null MLIR module")));
  EXPECT_THAT(
      CompiledIfrtIrProgram::Create(std::make_unique<IfrtIRProgram>(),
                                    make_options(), &client_, atom_compiler)
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("non-null MLIR module")));

  constexpr absl::string_view kPrivateMainSource = R"mlir(
module @private_main_module {
  func.func private @main() attributes {ifrt.function} {
    return
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> private_main_module,
                       LoadFromSource(kPrivateMainSource));
  EXPECT_THAT(IfrtIrExecutable::Create(
                  std::make_unique<IfrtIRProgram>(*private_main_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a public IFRT function")));
  EXPECT_THAT(CompiledIfrtIrProgram::Create(
                  std::make_unique<IfrtIRProgram>(*private_main_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a public IFRT function")));

  constexpr absl::string_view kNonIfrtMainSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @non_ifrt_main_module {
  func.func @main(%arg0: !array) -> (!array) {
    return %arg0 : !array
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> non_ifrt_module,
                       LoadFromSource(kNonIfrtMainSource));
  EXPECT_THAT(IfrtIrExecutable::Create(
                  std::make_unique<IfrtIRProgram>(*non_ifrt_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a public IFRT function")));
  EXPECT_THAT(CompiledIfrtIrProgram::Create(
                  std::make_unique<IfrtIRProgram>(*non_ifrt_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a public IFRT function")));

  constexpr absl::string_view kMissingMainSource = R"mlir(
module @missing_main_module {
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> missing_main_module,
                       LoadFromSource(kMissingMainSource));
  EXPECT_THAT(IfrtIrExecutable::Create(
                  std::make_unique<IfrtIRProgram>(*missing_main_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a public IFRT function")));
  EXPECT_THAT(CompiledIfrtIrProgram::Create(
                  std::make_unique<IfrtIRProgram>(*missing_main_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a public IFRT function")));

  constexpr absl::string_view kMultiBlockMainSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @multi_block_main_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    return %arg0 : !array
  ^bb1(%arg1: !array):
    return %arg1 : !array
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> multi_block_module,
                       LoadFromSource(kMultiBlockMainSource));
  EXPECT_THAT(IfrtIrExecutable::Create(
                  std::make_unique<IfrtIRProgram>(*multi_block_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must have a single block")));
  EXPECT_THAT(CompiledIfrtIrProgram::Create(
                  std::make_unique<IfrtIRProgram>(*multi_block_module),
                  make_options(), &client_, atom_compiler)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must have a single block")));
}

TEST_F(IfrtIrExecutableTest, CreateRejectsUnsupportedOpInMainAtCompileTime) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @unsupported_after_op_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %ctrl = "ifrt.After"(%arg0) : (!array) -> !ifrt.control
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) after %ctrl on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  auto atom_compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    EXPECT_THAT(IfrtIrExecutable::Create(
                    std::make_unique<IfrtIRProgram>(*module),
                    std::make_unique<IfrtIRCompileOptions>(
                        std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                    &client_, atom_compiler)
                    .Await(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Found unsupported op `ifrt.After`")));
  }
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    EXPECT_THAT(CompiledIfrtIrProgram::Create(
                    std::make_unique<IfrtIRProgram>(*module),
                    std::make_unique<IfrtIRCompileOptions>(
                        std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                    &client_, atom_compiler)
                    .Await(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Found unsupported op `ifrt.After`")));
  }
}

TEST_F(IfrtIrExecutableTest,
       CreateRejectsInvalidIntermediateMainOpDevicesAndRemapAtCompileTime) {
  auto atom_compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  auto make_options = []() {
    return std::make_unique<IfrtIRCompileOptions>(
        std::vector<DeviceId>{DeviceId(10), DeviceId(11)});
  };

  constexpr absl::string_view kOutOfBoundsCopySource = R"mlir(
!in_out_array = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!oob_array = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<2x1 to [0] on 2>, [0,2]>
module @oob_intermediate_copy_module {
  func.func @main(%arg0: !in_out_array) -> (!in_out_array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!in_out_array) -> (!oob_array)
    %1, %ctrl_1 = ifrt.CopyArrays(%0) after %ctrl_0
        : (!oob_array) -> (!in_out_array)
    return %1 : !in_out_array
  }
}
  )mlir";
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kOutOfBoundsCopySource));
    EXPECT_THAT(
        IfrtIrExecutable::Create(std::make_unique<IfrtIRProgram>(*module),
                                 make_options(), &client_, atom_compiler)
            .Await(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Logical device id 2 is out of range")));
  }
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kOutOfBoundsCopySource));
    EXPECT_THAT(
        CompiledIfrtIrProgram::Create(std::make_unique<IfrtIRProgram>(*module),
                                      make_options(), &client_, atom_compiler)
            .Await(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Logical device id 2 is out of range")));
  }

  constexpr absl::string_view kOutOfBoundsBitcastSource = R"mlir(
!in_out_array = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!oob_array = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<2x1 to [0] on 2>, [0,2]>
module @oob_intermediate_bitcast_module {
  func.func @main(%arg0: !in_out_array {ifrt.donated}) -> (!in_out_array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.BitcastArrays(%arg0) <donated = true>
        : (!in_out_array) -> (!oob_array)
    %1, %ctrl_1 = ifrt.BitcastArrays(%0) after %ctrl_0 <donated = true>
        : (!oob_array) -> (!in_out_array)
    return %1 : !in_out_array
  }
}
  )mlir";
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> oob_bitcast_module,
                         LoadFromSource(kOutOfBoundsBitcastSource));
    EXPECT_THAT(IfrtIrExecutable::Create(
                    std::make_unique<IfrtIRProgram>(*oob_bitcast_module),
                    make_options(), &client_, atom_compiler)
                    .Await(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Logical device id 2 is out of range")));
  }
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> oob_bitcast_module,
                         LoadFromSource(kOutOfBoundsBitcastSource));
    EXPECT_THAT(CompiledIfrtIrProgram::Create(
                    std::make_unique<IfrtIRProgram>(*oob_bitcast_module),
                    make_options(), &client_, atom_compiler)
                    .Await(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Logical device id 2 is out of range")));
  }

  constexpr absl::string_view kOutOfBoundsRemapSource = R"mlir(
!valid_array = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!oob_array = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
module @oob_intermediate_remap_module {
  func.func @main(%arg0: !valid_array) -> (!valid_array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!valid_array) -> (!oob_array)
    %1, %ctrl_1 = ifrt.RemapArrays(%0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        after %ctrl_0 : (!oob_array) -> (!valid_array)
    return %1 : !valid_array
  }
}
  )mlir";
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> oob_remap_module,
                         LoadFromSource(kOutOfBoundsRemapSource));
    EXPECT_THAT(IfrtIrExecutable::Create(
                    std::make_unique<IfrtIRProgram>(*oob_remap_module),
                    make_options(), &client_, atom_compiler)
                    .Await(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Logical device id 2 is out of range")));
  }
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> oob_remap_module,
                         LoadFromSource(kOutOfBoundsRemapSource));
    EXPECT_THAT(CompiledIfrtIrProgram::Create(
                    std::make_unique<IfrtIRProgram>(*oob_remap_module),
                    make_options(), &client_, atom_compiler)
                    .Await(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Logical device id 2 is out of range")));
  }
}

TEST_F(IfrtIrExecutableTest,
       CreateRejectsNullCompiledAtomProgramFromAtomCompiler) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @null_compiled_atom_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  const std::vector<std::shared_ptr<const CompiledAtomProgram>>
      malformed_atoms = {
          nullptr,
          std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
              /*executable=*/nullptr,
              /*load_options=*/
              std::make_unique<const LoadOptions>(
                  device_list_, /*outputs_bundle_slice_sizes=*/std::nullopt),
          }),
          std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
              /*executable=*/MakeMockUnloadedAtomExecutable(
                  "fake_xla_atom", layout_, /*hlo_module=*/nullptr),
              /*load_options=*/nullptr,
          }),
      };

  for (const std::shared_ptr<const CompiledAtomProgram>& malformed_atom :
       malformed_atoms) {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    auto atom_compiler =
        std::make_shared<StrictMock<MockAtomProgramCompiler>>();
    EXPECT_CALL(*atom_compiler, CompileXla).WillOnce(Return(malformed_atom));
    EXPECT_THAT(IfrtIrExecutable::Create(
                    std::make_unique<IfrtIRProgram>(*module),
                    std::make_unique<IfrtIRCompileOptions>(
                        std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                    &client_, std::move(atom_compiler))
                    .Await(),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("returned null compiled atom program")));
  }
}

TEST_F(IfrtIrExecutableTest,
       VerifiesBoundExternalExecutableBeforeDispatchingAtomCompilations) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @mismatched_bound_exec_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    %1, %ctrl_1 = ifrt.CallLoadedExecutable @bound_op(%0) after %ctrl_0
        : (!array) -> !array
    return %1 : !array
  }

  ifrt.LoadedExecutable @bound_op on devices [0,1] : (!array) -> !array

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  auto single_device_bound = std::make_shared<NiceMock<MockLoadedExecutable>>();
  ON_CALL(*single_device_bound, devices())
      .WillByDefault(Return(
          std::optional<DeviceListRef>(BasicDeviceList::Create({&device0_}))));
  AtomExecutableMap bindings;
  bindings.insert({"bound_op", single_device_bound});

  auto atom_compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  EXPECT_THAT(IfrtIrExecutable::Create(
                  std::make_unique<IfrtIRProgram>(*module),
                  std::make_unique<IfrtIRCompileOptions>(
                      std::vector<DeviceId>{DeviceId(10), DeviceId(11)},
                      std::move(bindings)),
                  &client_, std::move(atom_compiler))
                  .Await(),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("expects an executable with 2 devices")));
}

TEST_F(IfrtIrExecutableTest,
       CompileAndCompileAndLoadDrainInFlightAtomFuturesOnLoweringError) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @drain_inflight_atoms_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0,1]
        {ifrt.compile_options_key = "bad_override"} : (!array) -> !array
    return %1 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  auto make_options_with_bad_override = []() {
    auto overrides = std::make_shared<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>();
    overrides->insert(
        {"bad_override", std::make_unique<WrongTypeCompileOptions>()});
    return std::make_unique<IfrtIRCompileOptions>(
        std::vector<DeviceId>{DeviceId(10), DeviceId(11)}, AtomExecutableMap{},
        std::move(overrides));
  };

  // 1. `IfrtIrProgramCompiler::Compile`.
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    using CompiledAtomPtr =
        absl_nonnull std::shared_ptr<const CompiledAtomProgram>;
    auto [atom_promise, atom_future] = tsl::MakePromise<CompiledAtomPtr>();
    auto atom_compiler =
        std::make_unique<StrictMock<MockAtomProgramCompiler>>();
    EXPECT_CALL(*atom_compiler, CompileXla)
        .WillOnce(
            [future = std::move(atom_future)](
                std::unique_ptr<HloProgram>, xla::CompileOptions,
                std::vector<tsl::RCReference<LoadedHostCallback>>) mutable {
              return std::move(future);
            });
    IfrtIrProgramCompiler compiler(
        &client_,
        [&](const IfrtIRCompileOptions& unused_opts)
            -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
          return std::move(atom_compiler);
        });
    TestTopology topology;
    tsl::Future<ExecutableRef> compile_future =
        compiler.Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                         make_options_with_bad_override());
    EXPECT_FALSE(compile_future.IsReady());

    {
      // `layout_` and `device_list_` outlive `fulfill_thread` because
      // `~Thread()` joins on scope exit.
      std::unique_ptr<tsl::Thread> fulfill_thread =
          absl::WrapUnique(tsl::Env::Default()->StartThread(
              tsl::ThreadOptions(), "fulfill_drained_compile",
              [this, promise = std::move(atom_promise)]() mutable {
                ExecutableRef unloaded_executable =
                    MakeMockUnloadedAtomExecutable("fake_xla_atom", layout_,
                                                   /*hlo_module=*/nullptr);
                auto load_options = std::make_unique<const LoadOptions>(
                    device_list_, /*outputs_bundle_slice_sizes=*/std::nullopt);
                promise.Set(std::make_shared<const CompiledAtomProgram>(
                    CompiledAtomProgram{
                        /*executable=*/std::move(unloaded_executable),
                        /*load_options=*/std::move(load_options),
                    }));
              }));
    }

    EXPECT_THAT(compile_future.Await(),
                StatusIs(absl::StatusCode::kUnknown,
                         HasSubstr("not of type `XlaCompileOptions`")));
  }

  // 2. `IfrtIrProgramCompiler::CompileAndLoad`.
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    auto [atom_promise, atom_future] = tsl::MakePromise<LoadedExecutableRef>();
    auto atom_compiler =
        std::make_unique<StrictMock<MockAtomProgramCompiler>>();
    EXPECT_CALL(*atom_compiler, CompileAndLoadXla)
        .WillOnce(
            [future = std::move(atom_future)](
                std::unique_ptr<HloProgram>, xla::CompileOptions,
                std::vector<tsl::RCReference<LoadedHostCallback>>) mutable {
              return std::move(future);
            });
    IfrtIrProgramCompiler compiler(
        &client_,
        [&](const IfrtIRCompileOptions& unused_opts)
            -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
          return std::move(atom_compiler);
        });
    tsl::Future<LoadedExecutableRef> compile_and_load_future =
        compiler.CompileAndLoad(std::make_unique<IfrtIRProgram>(*module),
                                make_options_with_bad_override());
    EXPECT_FALSE(compile_and_load_future.IsReady());

    {
      // `this` outlives `fulfill_thread` because `~Thread()` joins on scope
      // exit.
      std::unique_ptr<tsl::Thread> fulfill_thread =
          absl::WrapUnique(tsl::Env::Default()->StartThread(
              tsl::ThreadOptions(), "fulfill_drained_compile_and_load",
              [this, promise = std::move(atom_promise)]() mutable {
                promise.Set(
                    LoadedExecutableRef(MakeLoadedAtomExecutable("add_one")));
              }));
    }

    EXPECT_THAT(compile_and_load_future.Await(),
                StatusIs(absl::StatusCode::kUnknown,
                         HasSubstr("not of type `XlaCompileOptions`")));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
