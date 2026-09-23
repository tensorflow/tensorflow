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

#include "xla/ffi/record_ffi.h"

#include <cstdint>
#include <initializer_list>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Eq;

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

TEST(FfiTest, RecordStageCreateAndUpdateCommands) {
  struct DummyCommand {};

  struct RecordState {
    DummyCommand launch{};
    DummyCommand memcpy{};
    DummyCommand empty{};

    const XLA_FFI_Command* launch_cmd =
        reinterpret_cast<const XLA_FFI_Command*>(&launch);
    const XLA_FFI_Command* memcpy_cmd =
        reinterpret_cast<const XLA_FFI_Command*>(&memcpy);
    const XLA_FFI_Command* empty_cmd =
        reinterpret_cast<const XLA_FFI_Command*>(&empty);

    bool launch_created = false;
    bool launch_updated = false;
    bool memcpy_created = false;
    bool memcpy_updated = false;
    bool empty_created = false;
  };

  RecordState state;
  XLA_FFI_RecordApi dummy_record_api = {
      /*create_launch=*/+[](XLA_FFI_RecordContext* ctx, const char*,
                            const void*, int64_t, XLA_FFI_SourceFormat,
                            XLA_FFI_LaunchDims, uint32_t, int32_t,
                            const XLA_FFI_KernelArgs*,
                            const XLA_FFI_Command* const*, uint32_t,
                            const XLA_FFI_Command** out_command)
                             -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<RecordState*>(ctx);
        *out_command = s->launch_cmd;
        return nullptr;
      },
      /*update_launch=*/
      +[](XLA_FFI_RecordContext* ctx, const XLA_FFI_Command* cmd,
          const XLA_FFI_KernelArgs*) -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<RecordState*>(ctx);
        EXPECT_EQ(cmd, s->launch_cmd);
        return nullptr;
      },
      /*create_memcpy_d2d=*/
      +[](XLA_FFI_RecordContext* ctx, void*, void*, int64_t,
          const XLA_FFI_Command* const*, uint32_t,
          const XLA_FFI_Command** out_command) -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<RecordState*>(ctx);
        *out_command = s->memcpy_cmd;
        return nullptr;
      },
      /*update_memcpy_d2d=*/
      +[](XLA_FFI_RecordContext* ctx, const XLA_FFI_Command* cmd, void*, void*,
          int64_t) -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<RecordState*>(ctx);
        EXPECT_EQ(cmd, s->memcpy_cmd);
        return nullptr;
      },
      /*request_stream_capture=*/
      +[](XLA_FFI_RecordContext*) -> XLA_FFI_Error* { return nullptr; },
      /*create_empty_command=*/
      +[](XLA_FFI_RecordContext* ctx, const XLA_FFI_Command* const*, uint32_t,
          const XLA_FFI_Command** out_command) -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<RecordState*>(ctx);
        *out_command = s->empty_cmd;
        return nullptr;
      },
  };

  auto handler = Ffi::BindRecord().Ctx<Extension<RecordExtension>>().To(
      [&](RecordContext ctx) -> absl::Status {
        RecordAction action = ctx.action();
        if (action == RecordAction::kCreate) {
          XLA_FFI_LaunchDims dims{{1, 1, 1}, {1, 1, 1}};
          auto launch_cmd = ctx.CreateLaunch(
              /*kernel_name=*/"dummy_kernel", /*kernel_data=*/"dummy_ptx",
              /*kernel_size=*/9, /*format=*/SourceFormat::kPtx,
              /*launch_dims=*/dims, /*shared_mem_bytes=*/0,
              /*uses_pdl=*/false, /*args=*/std::initializer_list<KernelArg>{});
          EXPECT_THAT(launch_cmd, IsOkAndHolds(Eq(state.launch_cmd)));
          state.launch_created = true;

          int src = 0, dst = 0;
          auto memcpy_cmd = ctx.CreateMemcpyD2D(
              /*dst=*/&dst, /*src=*/&src, /*size=*/sizeof(int));
          EXPECT_THAT(memcpy_cmd, IsOkAndHolds(Eq(state.memcpy_cmd)));
          state.memcpy_created = true;

          auto empty_cmd = ctx.CreateEmptyCommand();
          EXPECT_THAT(empty_cmd, IsOkAndHolds(Eq(state.empty_cmd)));
          state.empty_created = true;

          EXPECT_EQ(ctx.commands().size(), 3);
        } else if (action == RecordAction::kUpdate) {
          auto update_launch_st = ctx.UpdateLaunch(
              /*command=*/state.launch_cmd,
              /*args=*/std::initializer_list<KernelArg>{});
          EXPECT_OK(update_launch_st);
          state.launch_updated = true;

          int src = 0, dst = 0;
          auto update_memcpy_st = ctx.UpdateMemcpyD2D(
              /*command=*/state.memcpy_cmd, /*dst=*/&dst, /*src=*/&src,
              /*size=*/sizeof(int));
          EXPECT_OK(update_memcpy_st);
          state.memcpy_updated = true;
        }
        return absl::OkStatus();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  const XLA_FFI_Command* commands_storage[10] = {nullptr};
  int64_t num_commands = 0;

  // Create a record.
  {
    XLA_FFI_RecordFrame record_frame = {
        /*record_ctx=*/reinterpret_cast<XLA_FFI_RecordContext*>(&state),
        /*api=*/&dummy_record_api,
        /*action=*/XLA_FFI_RecordAction_Create,
        /*commands=*/commands_storage,
        /*num_commands=*/&num_commands,
        /*max_commands=*/10};
    RecordExtension::CExtension record_frame_ext =
        BuildRecordCExtension(&record_frame);

    InvokeContext context;
    context.extension_start = &record_frame_ext.extension_base;

    auto status =
        Invoke(Api(), *handler, call_frame, context, ExecutionStage::kRecord);
    ASSERT_OK(status);
    EXPECT_TRUE(state.launch_created);
    EXPECT_TRUE(state.memcpy_created);
    EXPECT_TRUE(state.empty_created);
    EXPECT_EQ(num_commands, 3);
    EXPECT_FALSE(state.launch_updated);
    EXPECT_FALSE(state.memcpy_updated);
  }

  // Update the record.
  {
    XLA_FFI_RecordFrame record_frame = {
        /*record_ctx=*/reinterpret_cast<XLA_FFI_RecordContext*>(&state),
        /*api=*/&dummy_record_api,
        /*action=*/XLA_FFI_RecordAction_Update,
        /*commands=*/commands_storage,
        /*num_commands=*/&num_commands,
        /*max_commands=*/10};
    RecordExtension::CExtension record_frame_ext =
        BuildRecordCExtension(&record_frame);

    InvokeContext context;
    context.extension_start = &record_frame_ext.extension_base;

    auto status =
        Invoke(Api(), *handler, call_frame, context, ExecutionStage::kRecord);
    ASSERT_OK(status);
    EXPECT_TRUE(state.launch_updated);
    EXPECT_TRUE(state.memcpy_updated);
    EXPECT_EQ(num_commands, 3);
  }
}

TEST(FfiTest, CreateLaunchWithFunctionPtr) {
  struct DummyCommand {};
  DummyCommand launch{};
  const XLA_FFI_Command* expected_cmd =
      reinterpret_cast<const XLA_FFI_Command*>(&launch);

  struct CaptureState {
    const XLA_FFI_Command* launch_cmd;
    const void* kernel_data = nullptr;
    int64_t kernel_size = -1;
    XLA_FFI_SourceFormat format = XLA_FFI_SourceFormat_PTX;
  } state{expected_cmd};

  const void* expected_func_ptr = reinterpret_cast<const void*>(0x12345678);

  XLA_FFI_RecordApi dummy_record_api = {};
  dummy_record_api.create_launch =
      +[](XLA_FFI_RecordContext* ctx, const char*, const void* kernel_data,
          int64_t kernel_size, XLA_FFI_SourceFormat format, XLA_FFI_LaunchDims,
          uint32_t, int32_t, const XLA_FFI_KernelArgs*,
          const XLA_FFI_Command* const*, uint32_t,
          const XLA_FFI_Command** out_command) -> XLA_FFI_Error* {
    auto* s = reinterpret_cast<CaptureState*>(ctx);
    s->kernel_data = kernel_data;
    s->kernel_size = kernel_size;
    s->format = format;
    *out_command = s->launch_cmd;
    return nullptr;
  };

  auto handler = Ffi::BindRecord().Ctx<Extension<RecordExtension>>().To(
      [&](RecordContext ctx) -> absl::Status {
        XLA_FFI_LaunchDims dims{{1, 1, 1}, {1, 1, 1}};
        auto null_cmd = ctx.CreateLaunch(
            /*kernel_name=*/"null_func", /*kernel_data=*/nullptr,
            /*kernel_size=*/0, /*format=*/SourceFormat::kFunctionPtr,
            /*launch_dims=*/dims, /*shared_mem_bytes=*/0,
            /*uses_pdl=*/false, /*args=*/std::initializer_list<KernelArg>{});
        EXPECT_THAT(null_cmd,
                    absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

        auto launch_cmd = ctx.CreateLaunch(
            /*kernel_name=*/"func_ptr_kernel",
            /*kernel_data=*/expected_func_ptr, /*kernel_size=*/0,
            /*format=*/SourceFormat::kFunctionPtr, /*launch_dims=*/dims,
            /*shared_mem_bytes=*/0, /*uses_pdl=*/false,
            /*args=*/std::initializer_list<KernelArg>{});
        EXPECT_THAT(launch_cmd, IsOkAndHolds(Eq(expected_cmd)));
        return absl::OkStatus();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  const XLA_FFI_Command* commands_storage[10] = {nullptr};
  int64_t num_commands = 0;

  XLA_FFI_RecordFrame record_frame = {
      /*record_ctx=*/reinterpret_cast<XLA_FFI_RecordContext*>(&state),
      /*api=*/&dummy_record_api,
      /*action=*/XLA_FFI_RecordAction_Create,
      /*commands=*/commands_storage,
      /*num_commands=*/&num_commands,
      /*max_commands=*/10};
  RecordExtension::CExtension record_frame_ext =
      BuildRecordCExtension(&record_frame);

  InvokeContext context;
  context.extension_start = &record_frame_ext.extension_base;

  ASSERT_OK(
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kRecord));
  EXPECT_EQ(state.kernel_data, expected_func_ptr);
  EXPECT_EQ(state.kernel_size, 0);
  EXPECT_EQ(state.format, XLA_FFI_SourceFormat_FUNCTION_PTR);
  EXPECT_EQ(num_commands, 1);
}

TEST(FfiTest, RecordExtensionVersionMismatch) {
  auto handler = Ffi::BindRecord().Ctx<Extension<RecordExtension>>().To(
      [&](RecordContext) -> absl::Status { return absl::OkStatus(); });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  XLA_FFI_RecordFrame record_frame = {};
  RecordExtension::CExtension record_frame_ext =
      BuildRecordCExtension(&record_frame);
  record_frame_ext.extension_base.id.minor_version = 1;

  InvokeContext context;
  context.extension_start = &record_frame_ext.extension_base;

  auto status =
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kRecord);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("Extension version mismatch for RecordExtension"));
}

}  // namespace
}  // namespace xla::ffi
