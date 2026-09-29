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

#include "xla/pjrt/interpreter/interpreter_pjrt_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/interpreter/interpreter_client.h"
#include "xla/pjrt/interpreter/interpreter_topology_description.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_topology_description_registry.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

constexpr absl::string_view kProgram = R"(HloModule Computation

ENTRY Computation() -> s32[] {
  ROOT result = s32[] constant(2)
})";

constexpr absl::string_view kMlirProgram = R"mlir(
  module {
    func.func @main() -> tensor<i32> {
      %0 = mhlo.constant dense<2> : tensor<i32>
      return %0 : tensor<i32>
    }
  })mlir";

using InterpreterPjrtCompilerTest = HloHardwareIndependentTestBase;

TEST_F(InterpreterPjrtCompilerTest, CompileXlaComputationSuccess) {
  CompileOptions options;
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kProgram));
  XlaComputation computation(module->ToProto());

  InterpreterTopologyDescription topology;
  InterpreterPjRtCompiler compiler;
  ASSERT_OK_AND_ASSIGN(auto executable,
                       compiler.Compile(options, computation, topology,
                                        /*client=*/nullptr));
  EXPECT_NE(executable, nullptr);
}

TEST_F(InterpreterPjrtCompilerTest, CompileMlirOpSuccess) {
  CompileOptions options;
  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::func::FuncDialect, mlir::mhlo::MhloDialect>();
  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlirProgram, context.get());

  InterpreterTopologyDescription topology;
  InterpreterPjRtCompiler compiler;
  ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(
          options,
          MaybeOwningMlirModule(std::move(context), std::move(mlir_module)),
          topology, /*client=*/nullptr));
  EXPECT_NE(executable, nullptr);
}

TEST_F(InterpreterPjrtCompilerTest, DeserializePjRtTopologyDescriptionSuccess) {
  InterpreterPjRtCompiler compiler;
  InterpreterTopologyDescription topology;
  ASSERT_OK_AND_ASSIGN(auto serialized_topology, topology.ToProto());
  ASSERT_OK_AND_ASSIGN(auto deserialized,
                       compiler.DeserializePjRtTopologyDescription(
                           serialized_topology.SerializeAsString()));
  EXPECT_EQ(deserialized->platform_name(), xla::InterpreterName());
  EXPECT_EQ(deserialized->platform_id(), xla::InterpreterId());
}

TEST_F(InterpreterPjrtCompilerTest, CompileClientlessAndExecuteOnClient) {
  CompileOptions options;
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kProgram));
  XlaComputation computation(module->ToProto());

  InterpreterTopologyDescription topology;
  InterpreterPjRtCompiler compiler;
  ASSERT_OK_AND_ASSIGN(auto executable,
                       compiler.Compile(options, computation, topology,
                                        /*client=*/nullptr));
  ASSERT_NE(executable, nullptr);

  InterpreterClient client;
  LoadOptions load_options;
  ASSERT_OK_AND_ASSIGN(auto loaded_executable,
                       client.Load(std::move(executable), load_options));
  ASSERT_NE(loaded_executable, nullptr);

  ExecuteOptions exec_options;
  ASSERT_OK_AND_ASSIGN(auto results,
                       loaded_executable->Execute({{}}, exec_options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  Literal result_literal(ShapeUtil::MakeShape(S32, {}));
  TF_ASSERT_OK(results[0][0]->ToLiteralSync(&result_literal));
  EXPECT_TRUE(LiteralTestUtil::Equal(result_literal,
                                     LiteralUtil::CreateR0<int32_t>(2)));
}

TEST_F(InterpreterPjrtCompilerTest, CompileViaPjRtCompileAndRegistryLookup) {
  CompileOptions options;
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kProgram));
  XlaComputation computation(module->ToProto());

  InterpreterTopologyDescription topology;
  ASSERT_OK_AND_ASSIGN(auto executable,
                       PjRtCompile(options, computation, topology));
  ASSERT_NE(executable, nullptr);

  InterpreterClient client;
  LoadOptions load_options;
  ASSERT_OK_AND_ASSIGN(auto loaded_executable,
                       client.Load(std::move(executable), load_options));
  ASSERT_NE(loaded_executable, nullptr);

  ExecuteOptions exec_options;
  ASSERT_OK_AND_ASSIGN(auto results,
                       loaded_executable->Execute({{}}, exec_options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  Literal result_literal(ShapeUtil::MakeShape(S32, {}));
  TF_ASSERT_OK(results[0][0]->ToLiteralSync(&result_literal));
  EXPECT_TRUE(LiteralTestUtil::Equal(result_literal,
                                     LiteralUtil::CreateR0<int32_t>(2)));
}

TEST_F(InterpreterPjrtCompilerTest, SerializationRoundTrip) {
  CompileOptions options;
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kProgram));
  XlaComputation computation(module->ToProto());

  InterpreterTopologyDescription topology;
  InterpreterPjRtCompiler compiler;
  ASSERT_OK_AND_ASSIGN(auto executable,
                       compiler.Compile(options, computation, topology,
                                        /*client=*/nullptr));
  ASSERT_NE(executable, nullptr);

  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       executable->SerializeExecutable());
  EXPECT_FALSE(serialized.empty());

  InterpreterClient client;
  ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      client.LoadSerializedExecutable(serialized, std::nullopt, LoadOptions()));
  ASSERT_NE(loaded_executable, nullptr);

  ExecuteOptions exec_options;
  ASSERT_OK_AND_ASSIGN(auto results,
                       loaded_executable->Execute({{}}, exec_options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  Literal result_literal(ShapeUtil::MakeShape(S32, {}));
  TF_ASSERT_OK(results[0][0]->ToLiteralSync(&result_literal));
  EXPECT_TRUE(LiteralTestUtil::Equal(result_literal,
                                     LiteralUtil::CreateR0<int32_t>(2)));
}

TEST_F(InterpreterPjrtCompilerTest, GlobalTopologyDeserializerRegistry) {
  InterpreterTopologyDescription topology;
  ASSERT_OK_AND_ASSIGN(auto serialized_topology, topology.ToProto());
  ASSERT_OK_AND_ASSIGN(auto deserialized,
                       PjRtTopologyDescriptionFromProto(serialized_topology));
  EXPECT_EQ(deserialized->platform_name(), xla::InterpreterName());
  EXPECT_EQ(deserialized->platform_id(), xla::InterpreterId());
}

TEST_F(InterpreterPjrtCompilerTest, CompileViaClientCompileAndLoadDirect) {
  InterpreterClient client;
  const Shape shape = ShapeUtil::MakeShape(S32, {4});
  XlaBuilder builder("test");
  Add(Parameter(&builder, 0, shape, "parameter0"),
      ConstantR1(&builder, absl::Span<const int32_t>{1, 1, 1, 1}));
  ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  ASSERT_OK_AND_ASSIGN(auto loaded_executable,
                       client.CompileAndLoad(computation, CompileOptions()));
  ASSERT_NE(loaded_executable, nullptr);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> argument,
      client.BufferFromHostLiteral(
          LiteralUtil::CreateR1(absl::Span<const int32_t>{1, 2, 3, 4}),
          client.memory_spaces().front()));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
      loaded_executable->Execute({{argument.get()}}, ExecuteOptions()));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results.front().size(), 1);
  Literal result_literal(shape);
  TF_ASSERT_OK(results.front().front()->ToLiteralSync(&result_literal));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      result_literal,
      LiteralUtil::CreateR1(absl::Span<const int32_t>{2, 3, 4, 5})));
}

TEST_F(InterpreterPjrtCompilerTest, DeviceLookupAndBufferPlacement) {
  InterpreterClient client;
  ASSERT_OK_AND_ASSIGN(PjRtDevice * device,
                       client.LookupDevice(GlobalDeviceId(0)));
  EXPECT_EQ(device, client.devices().front());

  EXPECT_FALSE(client.LookupDevice(GlobalDeviceId(1)).ok());
  EXPECT_FALSE(client.LookupDevice(GlobalDeviceId(-1)).ok());

  ASSERT_OK_AND_ASSIGN(PjRtDevice * addr_device,
                       client.LookupAddressableDevice(LocalDeviceId(0)));
  EXPECT_EQ(addr_device, client.addressable_devices().front());

  EXPECT_FALSE(client.LookupAddressableDevice(LocalDeviceId(1)).ok());
  EXPECT_FALSE(client.LookupAddressableDevice(LocalDeviceId(-1)).ok());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client.BufferFromHostLiteral(
          LiteralUtil::CreateR1(absl::Span<const int32_t>{1, 2, 3}), nullptr));
  EXPECT_EQ(buffer->device(), client.devices().front());
  EXPECT_EQ(buffer->memory_space(), client.memory_spaces().front());
}

TEST_F(InterpreterPjrtCompilerTest, TupledArgumentsRepeatedExecution) {
  InterpreterClient client;
  const Shape elem_shape = ShapeUtil::MakeShape(S32, {2});
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({elem_shape, elem_shape});
  XlaBuilder builder("test_tuple");
  auto p = Parameter(&builder, 0, tuple_shape, "p");
  auto p0 = GetTupleElement(p, 0);
  auto p1 = GetTupleElement(p, 1);
  Add(p0, p1);
  ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  CompileOptions compile_options;
  compile_options.parameter_is_tupled_arguments = true;
  ASSERT_OK_AND_ASSIGN(auto loaded_executable,
                       client.CompileAndLoad(computation, compile_options));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> arg0,
      client.BufferFromHostLiteral(
          LiteralUtil::CreateR1(absl::Span<const int32_t>{10, 20}), nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> arg1,
      client.BufferFromHostLiteral(
          LiteralUtil::CreateR1(absl::Span<const int32_t>{1, 2}), nullptr));

  // First execution
  ASSERT_OK_AND_ASSIGN(
      auto results1,
      loaded_executable->Execute({{arg0.get(), arg1.get()}}, ExecuteOptions()));
  Literal res1(elem_shape);
  TF_ASSERT_OK(results1.front().front()->ToLiteralSync(&res1));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      res1, LiteralUtil::CreateR1(absl::Span<const int32_t>{11, 22})));

  // Second execution with same argument buffers (verifies buffers were not
  // invalidated)
  ASSERT_OK_AND_ASSIGN(
      auto results2,
      loaded_executable->Execute({{arg0.get(), arg1.get()}}, ExecuteOptions()));
  Literal res2(elem_shape);
  TF_ASSERT_OK(results2.front().front()->ToLiteralSync(&res2));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      res2, LiteralUtil::CreateR1(absl::Span<const int32_t>{11, 22})));
}

}  // namespace
}  // namespace xla
