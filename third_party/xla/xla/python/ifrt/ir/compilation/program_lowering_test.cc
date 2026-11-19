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

#include "xla/python/ifrt/ir/compilation/program_lowering.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

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

using LayoutPtr = std::shared_ptr<const xla::PjRtLayout>;

class MockAtomProgramCompiler : public AtomProgramCompiler {
 public:
  MOCK_METHOD(
      tsl::Future<LoadedExecutableRef>, CompileAndLoadXla,
      (std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
       std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks),
      (override));
  MOCK_METHOD(
      tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>,
      CompileXla,
      (std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
       std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks),
      (override));
  MOCK_METHOD(tsl::Future<LoadedExecutableRef>, CompileAndLoadMpmdReshard,
              (std::vector<DType> dtypes, std::vector<Shape> shapes,
               std::vector<IfrtArrayType> in_array_types,
               std::vector<IfrtArrayType> out_array_types),
              (override));
  MOCK_METHOD(
      tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>,
      CompileMpmdReshard,
      (std::vector<DType> dtypes, std::vector<Shape> shapes,
       std::vector<IfrtArrayType> in_array_types,
       std::vector<IfrtArrayType> out_array_types),
      (override));
};

class ProgramLoweringTest : public test_util::IfrtIrTestBase {
 protected:
  void SetUp() override {
    default_layout_ =
        std::make_shared<xla::PjRtLayout>(xla::LayoutUtil::MakeLayout({1, 0}));
    param_layout_ =
        std::make_shared<xla::PjRtLayout>(xla::LayoutUtil::MakeLayout({0, 1}));
    output_layout_ = std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeLayout({1, 0}, /*tiles=*/{xla::Tile({2, 2})}));
    CHECK_NE(*default_layout_, *param_layout_);
    CHECK_NE(*default_layout_, *output_layout_);
    CHECK_NE(*param_layout_, *output_layout_);

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
        .WillByDefault(Return(default_layout_));
  }

  AttributeMap client_attributes_ = AttributeMap(AttributeMap::Map());
  NiceMock<MockClient> client_;
  NiceMock<MockDevice> device0_;
  NiceMock<MockDevice> device1_;
  std::vector<Device*> all_devices_;
  DeviceListRef device_list_;
  LayoutPtr default_layout_;
  LayoutPtr param_layout_;
  LayoutPtr output_layout_;
};

TEST_F(ProgramLoweringTest,
       DefersAtomProgramLoadingClonesContextAndPopulatesLayouts) {
  constexpr absl::string_view kSource = R"mlir(
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                      [0]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
module @deferred_lowering_module {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1)
      attributes {ifrt.function} {
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

  auto reshard_exec = std::make_shared<NiceMock<MockExecutable>>();
  ON_CALL(*reshard_exec, GetParameterLayouts())
      .WillByDefault(Return(std::vector<LayoutPtr>{param_layout_}));
  auto xla_exec = std::make_shared<NiceMock<MockExecutable>>();
  ON_CALL(*xla_exec, GetOutputLayouts())
      .WillByDefault(Return(std::vector<LayoutPtr>{output_layout_}));

  auto compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  EXPECT_CALL(*compiler, CompileMpmdReshard)
      .WillOnce(Return(std::make_shared<const CompiledAtomProgram>(
          CompiledAtomProgram{reshard_exec, std::make_unique<LoadOptions>()})));
  EXPECT_CALL(*compiler, CompileXla)
      .WillOnce(Return(std::make_shared<const CompiledAtomProgram>(
          CompiledAtomProgram{xla_exec, std::make_unique<LoadOptions>()})));

  absl::StatusOr<LoweredIfrtIrProgramState> lowered_state;
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    lowered_state = LowerIfrtIrProgram(
        std::make_unique<IfrtIRProgram>(*module),
        std::make_unique<IfrtIRCompileOptions>(
            std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
        &client_, compiler, AtomProgramLoadingMode::kDeferred);
  }
  ASSERT_OK(lowered_state);
  EXPECT_TRUE(lowered_state->ifrt_ir_program->OwnsMlirContext());
  EXPECT_EQ(lowered_state->atom_executable_future_map, nullptr);
  EXPECT_THAT(lowered_state->donatable_input_indices, ElementsAre(0));
  ASSERT_NE(lowered_state->compiled_atom_program_future_map, nullptr);
  EXPECT_THAT(*lowered_state->compiled_atom_program_future_map, SizeIs(2));

  CompiledAtomProgramMap compiled_atoms;
  for (const auto& [name, future] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       *lowered_state->compiled_atom_program_future_map) {
    ASSERT_OK_AND_ASSIGN(
        absl_nonnull std::shared_ptr<const CompiledAtomProgram> compiled_atom,
        future.Await());
    compiled_atoms.insert({name, std::move(compiled_atom)});
  }
  AtomExecutableMap empty_bound_executables;
  ASSERT_OK(PopulateIfrtIrProgramLayouts(
      lowered_state->ifrt_ir_program->mlir_module, &client_,
      empty_bound_executables, &compiled_atoms,
      absl::MakeSpan(lowered_state->in_specs),
      absl::MakeSpan(lowered_state->out_specs), lowered_state->device_list));
  ASSERT_THAT(lowered_state->in_specs, SizeIs(1));
  EXPECT_EQ(*lowered_state->in_specs[0].layout, *param_layout_);
  ASSERT_THAT(lowered_state->out_specs, SizeIs(1));
  EXPECT_EQ(*lowered_state->out_specs[0].layout, *output_layout_);
}

TEST_F(ProgramLoweringTest,
       EagerLoweringDispatchesCompileAndLoadAndBorrowsExternalMlirContext) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @eager_lowering_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
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
  auto loaded_exec = std::make_shared<NiceMock<MockLoadedExecutable>>();
  ON_CALL(*loaded_exec, GetParameterLayouts())
      .WillByDefault(Return(std::vector<LayoutPtr>{param_layout_}));
  ON_CALL(*loaded_exec, GetOutputLayouts())
      .WillByDefault(Return(std::vector<LayoutPtr>{output_layout_}));

  auto compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  EXPECT_CALL(*compiler, CompileAndLoadXla)
      .WillOnce(Return(LoadedExecutableRef(loaded_exec)));

  ASSERT_OK_AND_ASSIGN(
      LoweredIfrtIrProgramState state,
      LowerIfrtIrProgram(std::make_unique<IfrtIRProgram>(*module),
                         std::make_unique<IfrtIRCompileOptions>(
                             std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                         &client_, compiler, AtomProgramLoadingMode::kEager));
  EXPECT_FALSE(state.ifrt_ir_program->OwnsMlirContext());
  EXPECT_EQ(state.compiled_atom_program_future_map, nullptr);
  ASSERT_NE(state.atom_executable_future_map, nullptr);
  ASSERT_THAT(*state.atom_executable_future_map, SizeIs(1));

  const auto& [atom_name, atom_future] =
      *state.atom_executable_future_map->begin();
  ASSERT_OK_AND_ASSIGN(LoadedExecutableRef executable, atom_future.Await());
  AtomExecutableMap loaded_atoms;
  loaded_atoms.insert({atom_name, std::move(executable)});

  ASSERT_OK(PopulateIfrtIrProgramLayouts(
      state.ifrt_ir_program->mlir_module, &client_, loaded_atoms,
      /*compiled_atom_programs=*/nullptr, absl::MakeSpan(state.in_specs),
      absl::MakeSpan(state.out_specs), state.device_list));
  ASSERT_THAT(state.in_specs, SizeIs(1));
  EXPECT_EQ(*state.in_specs[0].layout, *param_layout_);
  ASSERT_THAT(state.out_specs, SizeIs(1));
  EXPECT_EQ(*state.out_specs[0].layout, *output_layout_);
}

TEST_F(ProgramLoweringTest,
       PopulatesLayoutsAcrossCopyArraysWithCompiledAtomPrograms) {
  constexpr absl::string_view kSource = R"mlir(
!dev_array = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                         memory_kind = "device">
!host_array = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                          memory_kind = "unpinned_host">
module @copy_arrays_layout_module {
  func.func @main(%arg0: !dev_array) -> (!dev_array, !host_array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!dev_array) -> !dev_array
    %dev_copy, %ctrl_1 = ifrt.CopyArrays(%0) : (!dev_array) -> !dev_array
    %host_copy, %ctrl_2 = ifrt.CopyArrays(%0) : (!dev_array) -> !host_array
    return %dev_copy, %host_copy : !dev_array, !host_array
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
  auto xla_exec = std::make_shared<NiceMock<MockExecutable>>();
  ON_CALL(*xla_exec, GetParameterLayouts())
      .WillByDefault(Return(std::vector<LayoutPtr>{param_layout_}));
  ON_CALL(*xla_exec, GetOutputLayouts())
      .WillByDefault(Return(std::vector<LayoutPtr>{output_layout_}));

  auto compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  EXPECT_CALL(*compiler, CompileXla)
      .WillOnce(Return(std::make_shared<const CompiledAtomProgram>(
          CompiledAtomProgram{xla_exec, std::make_unique<LoadOptions>()})));

  ASSERT_OK_AND_ASSIGN(
      LoweredIfrtIrProgramState state,
      LowerIfrtIrProgram(std::make_unique<IfrtIRProgram>(*module),
                         std::make_unique<IfrtIRCompileOptions>(
                             std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                         &client_, compiler,
                         AtomProgramLoadingMode::kDeferred));
  ASSERT_NE(state.compiled_atom_program_future_map, nullptr);
  ASSERT_THAT(*state.compiled_atom_program_future_map, SizeIs(1));

  const auto& [atom_name, atom_future] =
      *state.compiled_atom_program_future_map->begin();
  ASSERT_OK_AND_ASSIGN(
      absl_nonnull std::shared_ptr<const CompiledAtomProgram> compiled_atom,
      atom_future.Await());
  CompiledAtomProgramMap compiled_atoms;
  compiled_atoms.insert({atom_name, std::move(compiled_atom)});

  AtomExecutableMap empty_bound_executables;
  ASSERT_OK(PopulateIfrtIrProgramLayouts(
      state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
      &compiled_atoms, absl::MakeSpan(state.in_specs),
      absl::MakeSpan(state.out_specs), state.device_list));
  ASSERT_THAT(state.out_specs, SizeIs(2));
  EXPECT_EQ(*state.out_specs[0].layout, *output_layout_);
  EXPECT_EQ(*state.out_specs[1].layout, *default_layout_);
}

TEST_F(ProgramLoweringTest, RejectsNullLoadedExecBindingEntry) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @null_binding_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    return %arg0 : !array
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  auto compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  EXPECT_THAT(
      LowerIfrtIrProgram(std::make_unique<IfrtIRProgram>(*module),
                         std::make_unique<IfrtIRCompileOptions>(
                             std::vector<DeviceId>{DeviceId(10), DeviceId(11)},
                             AtomExecutableMap{{"null_bound", nullptr}}),
                         &client_, compiler, AtomProgramLoadingMode::kDeferred),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Bound executable `null_bound` must not be null")));
  EXPECT_THAT(
      LowerIfrtIrProgram(std::make_unique<IfrtIRProgram>(*module),
                         std::make_unique<IfrtIRCompileOptions>(
                             std::vector<DeviceId>{DeviceId(10), DeviceId(11)},
                             AtomExecutableMap{{"null_bound", nullptr}}),
                         &client_, compiler, AtomProgramLoadingMode::kEager),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Bound executable `null_bound` must not be null")));
}

TEST_F(ProgramLoweringTest,
       RejectsOutOfBoundsOrNullAtomLayoutsInPopulateIfrtIrProgramLayouts) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @invalid_layouts_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @identity(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  auto atom_exec = std::make_shared<NiceMock<MockExecutable>>();
  auto compiler = std::make_shared<StrictMock<MockAtomProgramCompiler>>();
  EXPECT_CALL(*compiler, CompileXla)
      .WillOnce(Return(std::make_shared<const CompiledAtomProgram>(
          CompiledAtomProgram{atom_exec, std::make_unique<LoadOptions>()})));

  ASSERT_OK_AND_ASSIGN(
      LoweredIfrtIrProgramState state,
      LowerIfrtIrProgram(std::make_unique<IfrtIRProgram>(*module),
                         std::make_unique<IfrtIRCompileOptions>(
                             std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                         &client_, compiler,
                         AtomProgramLoadingMode::kDeferred));
  ASSERT_NE(state.compiled_atom_program_future_map, nullptr);
  ASSERT_THAT(*state.compiled_atom_program_future_map, SizeIs(1));
  const auto& [atom_name, atom_future] =
      *state.compiled_atom_program_future_map->begin();
  ASSERT_OK_AND_ASSIGN(
      absl_nonnull std::shared_ptr<const CompiledAtomProgram> compiled_atom,
      atom_future.Await());
  AtomExecutableMap empty_bound_executables;

  // Missing atom executable -> FailedPreconditionError.
  CompiledAtomProgramMap empty_compiled_map;
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &empty_compiled_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Could not find SPMD executable")));

  CompiledAtomProgramMap compiled_map;
  compiled_map.insert({atom_name, std::move(compiled_atom)});

  // Empty parameter layouts vector -> out-of-range error instead of OOB read.
  EXPECT_CALL(*atom_exec, GetParameterLayouts())
      .WillOnce(Return(std::vector<LayoutPtr>{}));
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &compiled_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Parameter operand 0 out of range")));

  // Null parameter layout entry -> InternalError instead of null dereference.
  EXPECT_CALL(*atom_exec, GetParameterLayouts())
      .WillOnce(Return(std::vector<LayoutPtr>{nullptr}));
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &compiled_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("Parameter 0 layout")));

  // Empty output layouts vector -> out-of-range error instead of OOB read.
  EXPECT_CALL(*atom_exec, GetParameterLayouts())
      .WillRepeatedly(Return(std::vector<LayoutPtr>{param_layout_}));
  EXPECT_CALL(*atom_exec, GetOutputLayouts())
      .WillOnce(Return(std::vector<LayoutPtr>{}));
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &compiled_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Output 0 out of range")));

  // Null output layout entry -> InternalError instead of falling back to
  // BuildDefaultLayout.
  EXPECT_CALL(*atom_exec, GetOutputLayouts())
      .WillOnce(Return(std::vector<LayoutPtr>{nullptr}));
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &compiled_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("Output 0 layout")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
