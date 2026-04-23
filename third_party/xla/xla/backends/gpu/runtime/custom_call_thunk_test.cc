/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/custom_call_thunk.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/type_registry.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
struct TestState {
  std::string value;
};

struct NonSerializableTestState {
  int value;
};

struct FailingSerializableTestState {
  int value;
};

struct FailingDeserializableTestState {
  int value;
};
}  // namespace xla::gpu

namespace xla::ffi {
template <>
struct TypeRegistry::SerDes<xla::gpu::TestState> : public std::true_type {
  static absl::StatusOr<std::string> Serialize(
      const xla::gpu::TestState& value) {
    return value.value;
  }
  static absl::StatusOr<std::unique_ptr<xla::gpu::TestState>> Deserialize(
      absl::string_view data) {
    return std::make_unique<xla::gpu::TestState>(
        xla::gpu::TestState{std::string(data)});
  }
};

template <>
struct TypeRegistry::SerDes<xla::gpu::FailingSerializableTestState>
    : public std::true_type {
  static absl::StatusOr<std::string> Serialize(
      const xla::gpu::FailingSerializableTestState& value) {
    return absl::InternalError("Serialization failed");
  }
  static absl::StatusOr<std::unique_ptr<xla::gpu::FailingSerializableTestState>>
  Deserialize(absl::string_view data) {
    return std::make_unique<xla::gpu::FailingSerializableTestState>(
        xla::gpu::FailingSerializableTestState{0});
  }
};

template <>
struct TypeRegistry::SerDes<xla::gpu::FailingDeserializableTestState>
    : public std::true_type {
  static absl::StatusOr<std::string> Serialize(
      const xla::gpu::FailingDeserializableTestState& value) {
    return "some state";
  }
  static absl::StatusOr<
      std::unique_ptr<xla::gpu::FailingDeserializableTestState>>
  Deserialize(absl::string_view data) {
    return absl::InternalError("Deserialization failed");
  }
};
}  // namespace xla::ffi

namespace xla::gpu {
namespace {
using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::HasSubstr;

static absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
  ASSIGN_OR_RETURN(auto* platform, se::PlatformManager::PlatformWithName(name));
  return platform->ExecutorForDevice(0);
}

// A simple callback function that always returns an error.
absl::Status ReturnError() {
  return absl::UnknownError("Custom call was executed!");
}

XLA_FFI_DEFINE_HANDLER(kReturnError, ReturnError, ffi::Ffi::Bind(),
                       {ffi::Traits::kCmdBufferCompatible});

constexpr absl::string_view kReturnErrorCustomCallName =
    "__xla_test$$return_error";

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kReturnErrorCustomCallName,
                         "CUDA", kReturnError);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kReturnErrorCustomCallName,
                         "ROCM", kReturnError);

TEST(CustomCallThunkTest, ResolvesFFICustomCall) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(),
          /*target_name=*/std::string(kReturnErrorCustomCallName),
          /*operands=*/{},
          /*results=*/{}, /*attributes=*/{},
          /*called_computation=*/nullptr,
          /*platform_name=*/executor->GetPlatform()->Name(),
          /*gpu_compute_capability=*/
          stream->parent()->GetDeviceDescription().gpu_compute_capability()));

  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations empty_unused_allocations({}, 0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), empty_unused_allocations,
      /*stream=*/stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);
  EXPECT_THAT(thunk->ExecuteOnStream(params),
              StatusIs(absl::StatusCode::kUnknown,
                       HasSubstr("Custom call was executed!")));
}

TEST(CustomCallThunkTest, CustomCallWithOwnedHandlers) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  int instantiate_calls = 0;
  int prepare_calls = 0;
  int initialize_calls = 0;
  int execute_calls = 0;
  CustomCallThunk::OwnedHandlerBundle bundle;
  bundle.instantiate =
      ffi::Ffi::Bind<ffi::ExecutionStage::kInstantiate>().To([&]() {
        ++instantiate_calls;
        return absl::OkStatus();
      });
  bundle.prepare = ffi::Ffi::Bind<ffi::ExecutionStage::kPrepare>().To([&]() {
    ++prepare_calls;
    return absl::OkStatus();
  });
  bundle.initialize =
      ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>().To([&]() {
        ++initialize_calls;
        return absl::OkStatus();
      });
  bundle.execute = ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>().To([&]() {
    ++execute_calls;
    return absl::OkStatus();
  });

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({}, 0, &allocator);

  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(buffer_allocations);

  ScratchMemoryRequests scratch_memory_requests;

  Thunk::PrepareParams prepare_params{
      &collective_params,       &clique_requests, &memory_requests,
      &scratch_memory_requests, executor,         &buffer_allocations};

  Thunk::InitializeParams initialize_params;
  initialize_params.stream = stream.get();
  initialize_params.buffer_allocations = &buffer_allocations;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), buffer_allocations, stream.get(),
      stream.get(), nullptr, nullptr, nullptr);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), "target_name", std::move(bundle),
          /*operands=*/{},
          /*results=*/{}, /*attributes=*/{},
          /*called_computation=*/nullptr, se::GpuComputeCapability()));
  EXPECT_EQ(instantiate_calls, 1);
  EXPECT_EQ(prepare_calls, 0);
  EXPECT_EQ(initialize_calls, 0);
  EXPECT_EQ(execute_calls, 0);

  EXPECT_THAT(thunk->Prepare(prepare_params), IsOk());
  EXPECT_EQ(instantiate_calls, 1);
  EXPECT_EQ(prepare_calls, 1);
  EXPECT_EQ(initialize_calls, 0);
  EXPECT_EQ(execute_calls, 0);

  EXPECT_THAT(thunk->Initialize(initialize_params), IsOk());
  EXPECT_EQ(instantiate_calls, 1);
  EXPECT_EQ(prepare_calls, 1);
  EXPECT_EQ(initialize_calls, 1);
  EXPECT_EQ(execute_calls, 0);

  EXPECT_THAT(thunk->ExecuteOnStream(execute_params), IsOk());
  EXPECT_EQ(initialize_calls, 1);
  EXPECT_EQ(instantiate_calls, 1);
  EXPECT_EQ(prepare_calls, 1);
  EXPECT_EQ(execute_calls, 1);
}

TEST(CustomCallThunkTest, CustomCallWithOwnedHandlersWithoutOptionalOnes) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  int execute_calls = 0;
  CustomCallThunk::OwnedHandlerBundle bundle;
  bundle.execute = ffi::Ffi::Bind().To([&]() {
    ++execute_calls;
    return absl::OkStatus();
  });

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({}, 0, &allocator);

  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(buffer_allocations);
  ScratchMemoryRequests scratch_memory_requests;

  Thunk::PrepareParams prepare_params{
      &collective_params,       &clique_requests, &memory_requests,
      &scratch_memory_requests, executor,         &buffer_allocations};

  Thunk::InitializeParams initialize_params = Thunk::InitializeParams{};
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), buffer_allocations, stream.get(),
      stream.get(), nullptr, nullptr, nullptr);

  // Optional handlers are null and shouldn't be invoked.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(), "target_name", std::move(bundle),
          /*operands=*/{},
          /*results=*/{}, /*attributes=*/{},
          /*called_computation=*/nullptr, se::GpuComputeCapability()));
  EXPECT_THAT(thunk->Prepare(prepare_params), IsOk());
  EXPECT_THAT(thunk->Initialize(initialize_params), IsOk());
  EXPECT_THAT(thunk->ExecuteOnStream(execute_params), IsOk());
  EXPECT_EQ(execute_calls, 1);
}

TEST(CustomCallThunkTest, CustomCallWithOwnedHandlersWithoutExecute) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  CustomCallThunk::OwnedHandlerBundle bundle;  // all handlers null
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), BufferAllocations({}, 0, &allocator),
      stream.get(), stream.get(), nullptr, nullptr, nullptr);

  EXPECT_THAT(CustomCallThunk::Create(
                  Thunk::ThunkInfo(), "target_name", std::move(bundle),
                  /*operands=*/{},
                  /*results=*/{}, /*attributes=*/{},
                  /*called_computation=*/nullptr, se::GpuComputeCapability()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// A simple callback function that expects specific arguments.
absl::Status VerifyCallbackArguments(int my_attribute,
                                     ffi::AnyBuffer my_operand,
                                     ffi::Result<ffi::AnyBuffer> my_result,
                                     const HloComputation* called_computation,
                                     xla::gpu::TestState* state) {
  EXPECT_EQ(my_attribute, 42);
  EXPECT_EQ(my_operand.element_type(), xla::PrimitiveType::U8);
  EXPECT_EQ(my_operand.device_memory().opaque(),
            absl::bit_cast<void*>(static_cast<intptr_t>(0xDEADBEEF)));
  EXPECT_EQ(my_result->element_type(), xla::PrimitiveType::U16);
  EXPECT_EQ(my_result->device_memory().opaque(),
            absl::bit_cast<void*>(static_cast<intptr_t>(0xABCDEF)));
  EXPECT_EQ(called_computation->name(), "test_computation");
  EXPECT_EQ(state->value, "some state");
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kVerifyCallbackArguments, VerifyCallbackArguments,
                       ffi::Ffi::Bind()
                           .Attr<int>("my_attribute")
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::CalledComputation>()
                           .Ctx<ffi::State<xla::gpu::TestState>>(),
                       {ffi::Traits::kCmdBufferCompatible});

constexpr absl::string_view kVerifyCallbackArgumentsCustomCallName =
    "__xla_test$$verify_callback_arguments";
constexpr absl::string_view kTestPlatformName = "TEST_PLATFORM";

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         kVerifyCallbackArgumentsCustomCallName,
                         kTestPlatformName, kVerifyCallbackArguments);

TEST(CustomCallThunkTest, ProtoConversion) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  HloModuleConfig config;
  HloModule hlo_module("test_module", config);
  HloComputation::Builder builder("test_computation");
  // This instruction is pretty arbitrary, we just need a non-empty computation.
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {42}), "parameter"));
  hlo_module.AddEntryComputation(builder.Build());

  BufferAllocation alloc0{0, 1024, 0};
  BufferAllocation alloc1{1, 1024, 0};
  ShapedSlice operand_slice{BufferAllocation::Slice{&alloc0, 0, 1024},
                            ShapeUtil::MakeShape(U8, {1024})};
  ShapedSlice result_slice{BufferAllocation::Slice{&alloc1, 0, 1024},
                           ShapeUtil::MakeShape(U16, {512})};

  auto execution_state = std::make_unique<ffi::ExecutionState>();
  ASSERT_THAT(execution_state->Set(
                  std::make_unique<TestState>(TestState{"some state"})),
              IsOk());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> original_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(),
          /*target_name=*/std::string(kVerifyCallbackArgumentsCustomCallName),
          /*operands=*/{operand_slice},
          /*results=*/{result_slice}, /*attributes=*/{{"my_attribute", 42}},
          hlo_module.entry_computation(),
          /*platform_name=*/kTestPlatformName,
          /*gpu_compute_capability=*/{}, std::move(execution_state)));
  ASSERT_OK_AND_ASSIGN(ThunkProto proto, original_thunk->ToProto());
  ASSERT_TRUE(proto.has_custom_call_thunk());
  ASSERT_TRUE(proto.custom_call_thunk().has_execution_state());
  original_thunk.reset();

  std::array allocations = {alloc0, alloc1};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> new_thunk,
      CustomCallThunk::FromProto(Thunk::ThunkInfo(), proto.custom_call_thunk(),
                                 allocations, &hlo_module, kTestPlatformName,
                                 /*gpu_compute_capability=*/{},
                                 /*cpu_target_machine_options=*/std::nullopt));

  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations device_allocations(
      {stream_executor::DeviceAddressBase(
           absl::bit_cast<void*>(static_cast<intptr_t>(0xDEADBEEF)), 1024),
       stream_executor::DeviceAddressBase(
           absl::bit_cast<void*>(static_cast<intptr_t>(0xABCDEF)), 1024)},
      0, &allocator);
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), device_allocations,
      /*stream=*/stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);
  EXPECT_THAT(new_thunk->ExecuteOnStream(params), IsOk());
}

TEST(CustomCallThunkTest, DeserializationFailsWithMissingHloModule) {
  CustomCallThunkProto proto =
      tsl::proto_testing::ParseTextProtoOrDie<CustomCallThunkProto>(
          R"pb(
            target_name: "__xla_test$$verify_callback_arguments"
            api_version: API_VERSION_TYPED_FFI
            called_computation: "called_computation"
          )pb");

  HloModuleConfig config;
  HloModule hlo_module("test_module", config);
  HloComputation::Builder builder("not_called_computation");
  // This instruction is pretty arbitrary, we just need a non-empty computation.
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {42}), "parameter"));
  hlo_module.AddEntryComputation(builder.Build());

  EXPECT_THAT(
      CustomCallThunk::FromProto(Thunk::ThunkInfo(), proto,
                                 /*buffer_allocations=*/{}, &hlo_module,
                                 /*platform_name=*/kTestPlatformName,
                                 /*gpu_compute_capability=*/{},
                                 /*cpu_target_machine_options=*/std::nullopt),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CustomCallThunkTest, DeserializationFailsGracefully) {
  // We use a custom call name that is registered (so that finding the handler
  // succeeds) but we provide an execution state that fails deserialization.
  // The thunk creation should succeed, but the execution state should be null.
  CustomCallThunkProto proto =
      tsl::proto_testing::ParseTextProtoOrDie<CustomCallThunkProto>(
          R"pb(
            target_name: "__xla_test$$verify_callback_arguments"
            api_version: API_VERSION_TYPED_FFI
            called_computation: "called_computation"
            execution_state {
              type_name: "xla::gpu::FailingDeserializableTestState"
              state: "some state"
            }
          )pb");

  HloModuleConfig config;
  HloModule hlo_module("test_module", config);
  HloComputation::Builder builder("called_computation");
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {42}), "parameter"));
  hlo_module.AddEntryComputation(builder.Build());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::FromProto(Thunk::ThunkInfo(), proto,
                                 /*buffer_allocations=*/{}, &hlo_module,
                                 /*platform_name=*/kTestPlatformName,
                                 /*gpu_compute_capability=*/{},
                                 /*cpu_target_machine_options=*/std::nullopt));
  // If deserialization fails, we fall back to an empty execution state.
  EXPECT_NE(thunk->execution_state(), nullptr);
  EXPECT_FALSE(thunk->execution_state()->IsSet());
}

TEST(CustomCallThunkTest, RoundtripWithNonSerializableExecutionState) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  HloModuleConfig config;
  HloModule hlo_module("test_module", config);
  HloComputation::Builder builder("test_computation");
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {42}), "parameter"));
  hlo_module.AddEntryComputation(builder.Build());

  auto execution_state = std::make_unique<ffi::ExecutionState>();
  ASSERT_THAT(execution_state->Set(std::make_unique<NonSerializableTestState>(
                  NonSerializableTestState{42})),
              IsOk());
  EXPECT_FALSE(execution_state->IsSerializable());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> original_thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(),
          /*target_name=*/std::string(kVerifyCallbackArgumentsCustomCallName),
          /*operands=*/{},
          /*results=*/{}, /*attributes=*/{}, hlo_module.entry_computation(),
          /*platform_name=*/kTestPlatformName,
          /*gpu_compute_capability=*/{}, std::move(execution_state)));

  ASSERT_OK_AND_ASSIGN(ThunkProto proto, original_thunk->ToProto());
  ASSERT_TRUE(proto.has_custom_call_thunk());
  EXPECT_FALSE(proto.custom_call_thunk().has_execution_state());

  original_thunk.reset();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> new_thunk,
      CustomCallThunk::FromProto(
          Thunk::ThunkInfo(), proto.custom_call_thunk(),
          /*buffer_allocations=*/{}, &hlo_module, kTestPlatformName,
          executor->GetDeviceDescription().gpu_compute_capability(),
          /*cpu_target_machine_options=*/std::nullopt));

  EXPECT_NE(new_thunk->execution_state(), nullptr);
  EXPECT_FALSE(new_thunk->execution_state()->IsSet());
}

TEST(CustomCallThunkTest, SerializationFails) {
  HloModuleConfig config;
  HloModule hlo_module("test_module", config);
  HloComputation::Builder builder("test_computation");
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(U32, {42}), "parameter"));
  hlo_module.AddEntryComputation(builder.Build());

  auto execution_state = std::make_unique<ffi::ExecutionState>();
  ASSERT_OK(execution_state->Set(std::make_unique<FailingSerializableTestState>(
      FailingSerializableTestState{42})));
  EXPECT_TRUE(execution_state->IsSerializable());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(),
          /*target_name=*/std::string(kVerifyCallbackArgumentsCustomCallName),
          /*operands=*/{},
          /*results=*/{}, /*attributes=*/{}, hlo_module.entry_computation(),
          /*platform_name=*/kTestPlatformName,
          /*gpu_compute_capability=*/{}, std::move(execution_state)));

  EXPECT_THAT(thunk->ToProto(), StatusIs(absl::StatusCode::kInternal,
                                         HasSubstr("Serialization failed")));
}

TEST(CustomCallThunkTest, ParseFFIProtoWithNonUtf8Attribute) {
  // This test ensures that FFI attributes can contain non-UTF-8 data, and
  // these will be correctly parsed (and not fail).

  CustomCallThunkProto proto =
      tsl::proto_testing::ParseTextProtoOrDie<CustomCallThunkProto>(
          R"pb(
            target_name: "__xla_test$$return_error"
            api_version: API_VERSION_TYPED_FFI
            attributes {
              attrs {
                key: "my_string_attr"
                value { str: "\xfe" }
              }
            }
          )pb");

  std::string serialized_to_wire_format;
  proto.SerializeToString(&serialized_to_wire_format);

  CustomCallThunkProto reconstructed_proto;
  EXPECT_TRUE(reconstructed_proto.ParseFromString(serialized_to_wire_format));
}

static bool passes_cpu_target_machine_options_instantiate_called = false;

absl::Status VerifyCpuTargetMachineOptionsInstantiate(
    const xla::cpu::TargetMachineOptions* opts) {
  passes_cpu_target_machine_options_instantiate_called = true;
  EXPECT_NE(opts, nullptr);
  if (opts != nullptr) {
    EXPECT_EQ(opts->triple(), "test-triple");
    EXPECT_EQ(opts->cpu(), "test-cpu");
  }
  return absl::OkStatus();
}

// Execute stage is required for CustomCallThunk, so we do an empty one.
absl::Status VerifyCpuTargetMachineOptionsExecute() { return absl::OkStatus(); }

XLA_FFI_DEFINE_HANDLER(kVerifyCpuTargetMachineOptionsExecute,
                       VerifyCpuTargetMachineOptionsExecute, ffi::Ffi::Bind());

XLA_FFI_DEFINE_HANDLER(kVerifyCpuTargetMachineOptionsInstantiate,
                       VerifyCpuTargetMachineOptionsInstantiate,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kInstantiate>()
                           .Ctx<ffi::CpuTargetMachineOptions>());

constexpr absl::string_view kVerifyCpuTargetMachineOptionsCustomCallName =
    "__xla_test$$verify_cpu_target_machine_options";

XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(), kVerifyCpuTargetMachineOptionsCustomCallName, "CUDA",
    {kVerifyCpuTargetMachineOptionsInstantiate, nullptr, nullptr,
     kVerifyCpuTargetMachineOptionsExecute},
    static_cast<uint32_t>(ffi::Traits::kCmdBufferCompatible));
XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(), kVerifyCpuTargetMachineOptionsCustomCallName, "ROCM",
    {kVerifyCpuTargetMachineOptionsInstantiate, nullptr, nullptr,
     kVerifyCpuTargetMachineOptionsExecute},
    static_cast<uint32_t>(ffi::Traits::kCmdBufferCompatible));

TEST(CustomCallThunkTest, PassesCpuTargetMachineOptionsToInstantiate) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  xla::cpu::TargetMachineOptions options("test-triple", "test-cpu", "");
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> thunk,
      CustomCallThunk::Create(
          Thunk::ThunkInfo(),
          /*target_name=*/
          std::string(kVerifyCpuTargetMachineOptionsCustomCallName),
          /*operands=*/{},
          /*results=*/{}, /*attributes=*/{},
          /*called_computation=*/nullptr, executor->GetPlatform()->Name(),
          se::GpuComputeCapability(),
          /*execution_state=*/nullptr,
          /*cpu_target_machine_options=*/options));

  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);

  // Also check that FromProto restores the CPU target machine options.
  // We clear the execution state from the proto to force a re-instantiation.
  passes_cpu_target_machine_options_instantiate_called = false;
  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk->ToProto());
  proto.mutable_custom_call_thunk()->clear_execution_state();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CustomCallThunk> new_thunk,
      CustomCallThunk::FromProto(
          Thunk::ThunkInfo(), proto.custom_call_thunk(),
          /*buffer_allocations=*/{},
          /*hlo_module=*/nullptr, executor->GetPlatform()->Name(),
          executor->GetDeviceDescription().gpu_compute_capability(),
          /*cpu_target_machine_options=*/options));

  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

//===----------------------------------------------------------------------===//
// Command buffer tests (Record)
//===----------------------------------------------------------------------===//

// Cmd-buffer-compatible FFI handler that copies `src` to `dst` by issuing a
// D2D memcpy on the provided stream. Used by the Record tests below so that
// TracedCommand::Record can trace real device work onto the command buffer.
static absl::Status CmdBufferMemcpy(se::Stream* stream, ffi::AnyBuffer src,
                                    ffi::Result<ffi::AnyBuffer> dst) {
  se::DeviceAddressBase dst_mem = dst->device_memory();
  se::DeviceAddressBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kCmdBufferMemcpy, CmdBufferMemcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>(),
                       {ffi::Traits::kCmdBufferCompatible});

constexpr absl::string_view kCmdBufferMemcpyCustomCallName =
    "__xla_test$$cmd_buffer_memcpy";

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kCmdBufferMemcpyCustomCallName,
                         "CUDA", kCmdBufferMemcpy);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kCmdBufferMemcpyCustomCallName,
                         "ROCM", kCmdBufferMemcpy);

// Builds a CustomCallThunk that copies operand 0 into result 0.
static absl::StatusOr<std::unique_ptr<CustomCallThunk>> MakeMemcpyCustomCall(
    se::StreamExecutor* executor, const BufferAllocation& src_alloc,
    const BufferAllocation& dst_alloc, int64_t byte_length) {
  ShapedSlice src_slice{BufferAllocation::Slice{&src_alloc, 0, byte_length},
                        ShapeUtil::MakeShape(U8, {byte_length})};
  ShapedSlice dst_slice{BufferAllocation::Slice{&dst_alloc, 0, byte_length},
                        ShapeUtil::MakeShape(U8, {byte_length})};
  return CustomCallThunk::Create(
      Thunk::ThunkInfo(),
      /*target_name=*/std::string(kCmdBufferMemcpyCustomCallName),
      /*operands=*/{src_slice},
      /*results=*/{dst_slice}, /*attributes=*/{},
      /*called_computation=*/nullptr,
      /*platform_name=*/executor->GetPlatform()->Name(),
      /*gpu_compute_capability=*/
      executor->GetDeviceDescription().gpu_compute_capability());
}

TEST(CustomCallThunkTest, RecordCommandBuffer) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  constexpr int64_t kByteLength = 16;
  se::DeviceAddress<uint8_t> src =
      executor->AllocateArray<uint8_t>(kByteLength, 0);
  se::DeviceAddress<uint8_t> dst =
      executor->AllocateArray<uint8_t>(kByteLength, 0);
  std::vector<uint8_t> host_src(kByteLength, 0x5A);
  TF_ASSERT_OK(stream->Memcpy(&src, host_src.data(), kByteLength));
  TF_ASSERT_OK(stream->MemZero(&dst, kByteLength));

  BufferAllocation src_alloc{0, kByteLength, 0};
  BufferAllocation dst_alloc{1, kByteLength, 0};
  ASSERT_OK_AND_ASSIGN(
      auto thunk,
      MakeMemcpyCustomCall(executor, src_alloc, dst_alloc, kByteLength));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({src, dst}, 0, &allocator);

  Thunk::InitializeParams init_params;
  init_params.executor = executor;
  init_params.stream = stream.get();
  init_params.buffer_allocations = &buffer_allocations;
  TF_ASSERT_OK(thunk->Initialize(init_params));

  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk->Record(execute_params, record_params,
                    Command::RecordCreate{/*dependencies=*/{}},
                    command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<uint8_t> host_dst(kByteLength, 0);
  TF_ASSERT_OK(stream->Memcpy(host_dst.data(), dst, kByteLength));
  EXPECT_EQ(host_dst, host_src);
}

TEST(CustomCallThunkTest, RecordCommandBufferUpdate) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  constexpr int64_t kByteLength = 16;
  se::DeviceAddress<uint8_t> src =
      executor->AllocateArray<uint8_t>(kByteLength, 0);
  se::DeviceAddress<uint8_t> dst_first =
      executor->AllocateArray<uint8_t>(kByteLength, 0);
  se::DeviceAddress<uint8_t> dst_second =
      executor->AllocateArray<uint8_t>(kByteLength, 0);
  std::vector<uint8_t> host_src(kByteLength, 0x3C);
  TF_ASSERT_OK(stream->Memcpy(&src, host_src.data(), kByteLength));
  TF_ASSERT_OK(stream->MemZero(&dst_first, kByteLength));
  TF_ASSERT_OK(stream->MemZero(&dst_second, kByteLength));

  BufferAllocation src_alloc{0, kByteLength, 0};
  BufferAllocation dst_alloc{1, kByteLength, 0};
  ASSERT_OK_AND_ASSIGN(
      auto thunk,
      MakeMemcpyCustomCall(executor, src_alloc, dst_alloc, kByteLength));

  se::StreamExecutorAddressAllocator allocator(executor);

  BufferAllocations allocs_first({src, dst_first}, 0, &allocator);
  Thunk::InitializeParams init_params;
  init_params.executor = executor;
  init_params.stream = stream.get();
  init_params.buffer_allocations = &allocs_first;
  TF_ASSERT_OK(thunk->Initialize(init_params));

  Thunk::ExecuteParams params_first = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), allocs_first, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk->Record(params_first, record_params,
                    Command::RecordCreate{/*dependencies=*/{}},
                    command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<uint8_t> host_first(kByteLength, 0);
  TF_ASSERT_OK(stream->Memcpy(host_first.data(), dst_first, kByteLength));
  EXPECT_EQ(host_first, host_src);

  // Update with a different destination allocation and re-submit.
  BufferAllocations allocs_second({src, dst_second}, 0, &allocator);
  Thunk::ExecuteParams params_second = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), allocs_second, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk->Record(params_second, record_params, Command::RecordUpdate{cmd},
                    command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<uint8_t> host_second(kByteLength, 0);
  TF_ASSERT_OK(stream->Memcpy(host_second.data(), dst_second, kByteLength));
  EXPECT_EQ(host_second, host_src);
}

}  // namespace
}  // namespace xla::gpu
