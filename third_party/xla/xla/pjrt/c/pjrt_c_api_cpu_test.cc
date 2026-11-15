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
#include "xla/pjrt/c/pjrt_c_api_cpu.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_test.h"

namespace pjrt {
namespace {

const bool kUnused = (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); },
                                                  /*platform_name=*/"cpu"),
                      true);

TEST(PjRtCApiCpuTest, CreateClientWithCreateOptions) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "cpu_device_count";
  create_option.name_size = strlen("cpu_device_count");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  create_option.int64_value = 16;
  create_option.value_size = 1;

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_EQ(error, nullptr);

  PJRT_Client_Devices_Args dev_args;
  dev_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  dev_args.extension_start = nullptr;
  dev_args.client = args.client;
  error = api_->PJRT_Client_Devices(&dev_args);
  ASSERT_EQ(error, nullptr);
  ASSERT_EQ(dev_args.num_devices, 16);

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = args.client;
  PJRT_Error* destroy_error = GetPjrtApi()->PJRT_Client_Destroy(&destroy_args);
  ASSERT_EQ(destroy_error, nullptr);
}

TEST(PjRtCApiCpuTest, CreateClientWithInvalidCreateOptions) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "cpu_device_count";
  create_option.name_size = strlen("cpu_device_count");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kString;
  create_option.string_value = "invalid_int";
  create_option.value_size = strlen("invalid_int");

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_NE(error, nullptr);

  PJRT_Error_GetCode_Args code_args;
  code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  code_args.extension_start = nullptr;
  code_args.error = error;
  ASSERT_EQ(api_->PJRT_Error_GetCode(&code_args), nullptr);
  EXPECT_EQ(code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);

  PJRT_Error_Message_Args message_args;
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.extension_start = nullptr;
  message_args.error = error;
  api_->PJRT_Error_Message(&message_args);
  EXPECT_NE(
      strstr(message_args.message,
             "Option passed to PJRT_Client_Create with name cpu_device_count "
             "has type index 0 but expected type index is 1"),
      nullptr);

  PJRT_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.error = error;
  api_->PJRT_Error_Destroy(&destroy_args);
}

TEST(PjRtCApiCpuTest, CreateClientWithUnexpectedCreateOptions) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "unexpected_option";
  create_option.name_size = strlen("unexpected_option");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  create_option.int64_value = 16;
  create_option.value_size = 1;

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_NE(error, nullptr);

  PJRT_Error_GetCode_Args code_args;
  code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  code_args.extension_start = nullptr;
  code_args.error = error;
  ASSERT_EQ(api_->PJRT_Error_GetCode(&code_args), nullptr);
  EXPECT_EQ(code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);

  PJRT_Error_Message_Args message_args;
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.extension_start = nullptr;
  message_args.error = error;
  api_->PJRT_Error_Message(&message_args);
  EXPECT_NE(strstr(message_args.message,
                   "Unexpected option name passed to PJRT_Client_Create: "
                   "unexpected_option"),
            nullptr);

  PJRT_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.error = error;
  api_->PJRT_Error_Destroy(&destroy_args);
}

TEST(PjRtCApiCpuTest, CreateClientWithAsynchronousOption) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "asynchronous";
  create_option.name_size = strlen("asynchronous");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kBool;
  create_option.bool_value = true;
  create_option.value_size = 1;

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(args.client, nullptr);

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = args.client;
  ASSERT_EQ(api_->PJRT_Client_Destroy(&destroy_args), nullptr);
}

TEST(PjRtCApiCpuTest, CreateClientWithProcessIdOption) {
  const PJRT_Api* api_ = GetPjrtApi();

  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;

  PJRT_NamedValue create_option;
  create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  create_option.extension_start = nullptr;
  create_option.name = "process_id";
  create_option.name_size = strlen("process_id");
  create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  create_option.int64_value = 16;
  create_option.value_size = 1;

  args.create_options = &create_option;
  args.num_options = 1;
  args.client = nullptr;
  PJRT_Error* error = api_->PJRT_Client_Create(&args);
  ASSERT_EQ(error, nullptr);

  PJRT_Client_ProcessIndex_Args dev_args;
  dev_args.struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE;
  dev_args.extension_start = nullptr;
  dev_args.client = args.client;
  error = api_->PJRT_Client_ProcessIndex(&dev_args);
  ASSERT_EQ(error, nullptr);
  ASSERT_EQ(dev_args.process_index, 16);

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = args.client;
  PJRT_Error* destroy_error = GetPjrtApi()->PJRT_Client_Destroy(&destroy_args);
  ASSERT_EQ(destroy_error, nullptr);
}

void AwaitAndDestroyEvent(const PJRT_Api* api, PJRT_Event* event) {
  if (event == nullptr) {
    return;
  }
  PJRT_Event_Await_Args await_args{};
  await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await_args.event = event;
  EXPECT_EQ(api->PJRT_Event_Await(&await_args), nullptr);

  PJRT_Event_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy_args.event = event;
  EXPECT_EQ(api->PJRT_Event_Destroy(&destroy_args), nullptr);
}

void DestroyBuffer(const PJRT_Api* api, PJRT_Buffer* buffer) {
  if (buffer == nullptr) {
    return;
  }
  PJRT_Buffer_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  destroy_args.buffer = buffer;
  EXPECT_EQ(api->PJRT_Buffer_Destroy(&destroy_args), nullptr);
}

struct CpuCApiTestContext {
  const PJRT_Api* api = GetPjrtApi();
  PJRT_Client* client = nullptr;
  PJRT_Device* device = nullptr;
  PJRT_Buffer* input_buffer = nullptr;
  PJRT_LoadedExecutable* executable = nullptr;
  std::array<float, 4> host_input = {10.0f, 20.0f, 30.0f, 40.0f};
  int64_t dim = 4;

  void SetUp() {
    PJRT_NamedValue create_option{};
    create_option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    create_option.name = "cpu_device_count";
    create_option.name_size = strlen("cpu_device_count");
    create_option.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
    create_option.int64_value = 1;
    create_option.value_size = 1;

    PJRT_Client_Create_Args create_args{};
    create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &create_option;
    create_args.num_options = 1;
    ASSERT_EQ(api->PJRT_Client_Create(&create_args), nullptr);
    client = create_args.client;

    PJRT_Client_AddressableDevices_Args dev_args{};
    dev_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    dev_args.client = client;
    ASSERT_EQ(api->PJRT_Client_AddressableDevices(&dev_args), nullptr);
    ASSERT_GE(dev_args.num_addressable_devices, 1u);
    device = dev_args.addressable_devices[0];

    PJRT_Client_BufferFromHostBuffer_Args buf_args{};
    buf_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    buf_args.client = client;
    buf_args.data = host_input.data();
    buf_args.type = PJRT_Buffer_Type_F32;
    buf_args.dims = &dim;
    buf_args.num_dims = 1;
    buf_args.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    buf_args.device = device;
    ASSERT_EQ(api->PJRT_Client_BufferFromHostBuffer(&buf_args), nullptr);
    AwaitAndDestroyEvent(api, buf_args.done_with_host_buffer);
    input_buffer = buf_args.buffer;

    static constexpr char kMlirProgram[] =
        "module {\n"
        "  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n"
        "    %0 = mhlo.constant dense<1.000000e+00> : tensor<4xf32>\n"
        "    %1 = mhlo.add %arg0, %0 : tensor<4xf32>\n"
        "    return %1 : tensor<4xf32>\n"
        "  }\n"
        "}\n";
    PJRT_Program program{};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.code = const_cast<char*>(kMlirProgram);
    program.code_size = sizeof(kMlirProgram) - 1;
    program.format = "mlir";
    program.format_size = 4;

    static constexpr char kCompileOptions[] = "\x1a\x04\x20\x01\x28\x01";
    PJRT_Client_Compile_Args compile_args{};
    compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = client;
    compile_args.program = &program;
    compile_args.compile_options = kCompileOptions;
    compile_args.compile_options_size = sizeof(kCompileOptions) - 1;
    ASSERT_EQ(api->PJRT_Client_Compile(&compile_args), nullptr);
    executable = compile_args.executable;
  }

  ~CpuCApiTestContext() {
    if (executable != nullptr) {
      PJRT_LoadedExecutable_Destroy_Args exec_destroy{};
      exec_destroy.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
      exec_destroy.executable = executable;
      EXPECT_EQ(api->PJRT_LoadedExecutable_Destroy(&exec_destroy), nullptr);
    }
    DestroyBuffer(api, input_buffer);
    if (client != nullptr) {
      PJRT_Client_Destroy_Args client_destroy{};
      client_destroy.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
      client_destroy.client = client;
      EXPECT_EQ(api->PJRT_Client_Destroy(&client_destroy), nullptr);
    }
  }

  // Constructs a valid baseline PJRT_Buffer_ToHostBuffer_Args configured to
  // copy all 4 floats (16 bytes) from src into dst using the default
  // (nullptr) host layout.
  PJRT_Buffer_ToHostBuffer_Args MakeValidToHostBufferArgs(
      PJRT_Buffer* src, std::array<float, 4>* dst) const {
    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = src;
    args.host_layout = nullptr;
    args.dst = dst != nullptr ? dst->data() : nullptr;
    args.dst_size = dst != nullptr ? sizeof(*dst) : 0;
    args.event = nullptr;
    return args;
  }

  // Constructs a valid baseline PJRT_ExecuteOptions with no callbacks or
  // non-donatable input overrides.
  PJRT_ExecuteOptions MakeValidExecuteOptions() const {
    PJRT_ExecuteOptions options{};
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    options.use_major_to_minor_data_layout_for_callbacks = true;
    return options;
  }

  // Constructs a valid baseline PJRT_LoadedExecutable_Execute_Args for a
  // 1-device, 1-argument multi-device launch (execute_device = nullptr,
  // device_complete_events = nullptr).
  PJRT_LoadedExecutable_Execute_Args MakeValidExecuteArgs(
      PJRT_ExecuteOptions* options, PJRT_Buffer* const* const* argument_lists,
      PJRT_Buffer** const* output_lists) const {
    PJRT_LoadedExecutable_Execute_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args.executable = executable;
    args.options = options;
    args.argument_lists = argument_lists;
    args.num_devices = 1;
    args.num_args = 1;
    args.output_lists = output_lists;
    args.device_complete_events = nullptr;
    args.execute_device = nullptr;
    return args;
  }

  std::array<float, 4> ReadBuffer(PJRT_Buffer* buffer) const {
    std::array<float, 4> dst = {0.0f, 0.0f, 0.0f, 0.0f};
    PJRT_Buffer_ToHostBuffer_Args args =
        MakeValidToHostBufferArgs(buffer, &dst);
    EXPECT_EQ(api->PJRT_Buffer_ToHostBuffer(&args), nullptr);
    AwaitAndDestroyEvent(api, args.event);
    return dst;
  }
};

TEST(PjRtCApiCpuTest, ExecuteSingleAndMultiDevicePaths) {
  CpuCApiTestContext ctx;
  ASSERT_NO_FATAL_FAILURE(ctx.SetUp());

  PJRT_ExecuteOptions options = ctx.MakeValidExecuteOptions();
  PJRT_Buffer* arg_ptrs[1] = {ctx.input_buffer};
  PJRT_Buffer* const* arg_lists[1] = {arg_ptrs};

  // 1. Baseline multi-device path (execute_device = nullptr,
  // device_complete_events = nullptr): use the unmodified valid execute args
  // to run Populate2DCppBuffers without allocating completion event futures.
  PJRT_Buffer* multi_out_ptrs[1] = {nullptr};
  PJRT_Buffer** multi_out_lists[1] = {multi_out_ptrs};
  PJRT_LoadedExecutable_Execute_Args multi_args =
      ctx.MakeValidExecuteArgs(&options, arg_lists, multi_out_lists);
  ASSERT_EQ(ctx.api->PJRT_LoadedExecutable_Execute(&multi_args), nullptr);
  ASSERT_NE(multi_out_ptrs[0], nullptr);
  const std::array<float, 4> expected = {11.0f, 21.0f, 31.0f, 41.0f};
  EXPECT_EQ(ctx.ReadBuffer(multi_out_ptrs[0]), expected);
  DestroyBuffer(ctx.api, multi_out_ptrs[0]);

  // 2. Single-device portable path with completion event: start from the valid
  // execute args and set execute_device = ctx.device (switching from
  // Execute to ExecutePortable / Populate1DCppBuffers) and
  // device_complete_events = &complete_event (enabling returned_futures).
  PJRT_Buffer* single_out_ptrs[1] = {nullptr};
  PJRT_Buffer** single_out_lists[1] = {single_out_ptrs};
  PJRT_Event* complete_event = nullptr;
  PJRT_LoadedExecutable_Execute_Args single_args =
      ctx.MakeValidExecuteArgs(&options, arg_lists, single_out_lists);
  single_args.execute_device = ctx.device;
  single_args.device_complete_events = &complete_event;
  ASSERT_EQ(ctx.api->PJRT_LoadedExecutable_Execute(&single_args), nullptr);
  AwaitAndDestroyEvent(ctx.api, complete_event);
  ASSERT_NE(single_out_ptrs[0], nullptr);
  EXPECT_EQ(ctx.ReadBuffer(single_out_ptrs[0]), expected);
  DestroyBuffer(ctx.api, single_out_ptrs[0]);

  // 3. Single-device path with options.context and device_complete_events =
  // nullptr: start from the valid execute args, set execute_device =
  // ctx.device, attach a PJRT_ExecuteContext to options.context, and destroy
  // the C wrapper immediately after PJRT_LoadedExecutable_Execute returns so
  // the single-device OnReady callback must keep execute_context alive without
  // writing to a null device_complete_events pointer.
  PJRT_ExecuteContext_Create_Args ctx_create_args{};
  ctx_create_args.struct_size = PJRT_ExecuteContext_Create_Args_STRUCT_SIZE;
  ASSERT_EQ(ctx.api->PJRT_ExecuteContext_Create(&ctx_create_args), nullptr);
  PJRT_ExecuteOptions ctx_options = ctx.MakeValidExecuteOptions();
  ctx_options.context = ctx_create_args.context;

  PJRT_Buffer* single_ctx_out_ptrs[1] = {nullptr};
  PJRT_Buffer** single_ctx_out_lists[1] = {single_ctx_out_ptrs};
  PJRT_LoadedExecutable_Execute_Args single_ctx_args =
      ctx.MakeValidExecuteArgs(&ctx_options, arg_lists, single_ctx_out_lists);
  single_ctx_args.execute_device = ctx.device;
  ASSERT_EQ(ctx.api->PJRT_LoadedExecutable_Execute(&single_ctx_args), nullptr);

  PJRT_ExecuteContext_Destroy_Args ctx_destroy_args{};
  ctx_destroy_args.struct_size = PJRT_ExecuteContext_Destroy_Args_STRUCT_SIZE;
  ctx_destroy_args.context = ctx_create_args.context;
  ASSERT_EQ(ctx.api->PJRT_ExecuteContext_Destroy(&ctx_destroy_args), nullptr);

  ASSERT_NE(single_ctx_out_ptrs[0], nullptr);
  EXPECT_EQ(ctx.ReadBuffer(single_ctx_out_ptrs[0]), expected);
  DestroyBuffer(ctx.api, single_ctx_out_ptrs[0]);
}

}  // namespace
}  // namespace pjrt
