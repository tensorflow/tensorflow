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
                            XLA_FFI_LaunchDims, uint32_t,
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
          auto launch_cmd =
              ctx.CreateLaunch("dummy_kernel", nullptr, 0, SourceFormat::kPtx,
                               dims, 0, std::initializer_list<KernelArg>{});
          EXPECT_THAT(launch_cmd, IsOkAndHolds(Eq(state.launch_cmd)));
          state.launch_created = true;

          int src = 0, dst = 0;
          auto memcpy_cmd = ctx.CreateMemcpyD2D(&dst, &src, sizeof(int));
          EXPECT_THAT(memcpy_cmd, IsOkAndHolds(Eq(state.memcpy_cmd)));
          state.memcpy_created = true;

          auto empty_cmd = ctx.CreateEmptyCommand();
          EXPECT_THAT(empty_cmd, IsOkAndHolds(Eq(state.empty_cmd)));
          state.empty_created = true;

          EXPECT_EQ(ctx.commands().size(), 3);
        } else if (action == RecordAction::kUpdate) {
          auto update_launch_st = ctx.UpdateLaunch(
              state.launch_cmd, std::initializer_list<KernelArg>{});
          EXPECT_OK(update_launch_st);
          state.launch_updated = true;

          int src = 0, dst = 0;
          auto update_memcpy_st =
              ctx.UpdateMemcpyD2D(state.memcpy_cmd, &dst, &src, sizeof(int));
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
        reinterpret_cast<XLA_FFI_RecordContext*>(&state),
        &dummy_record_api,
        XLA_FFI_RecordAction_Create,
        commands_storage,
        &num_commands,
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
        reinterpret_cast<XLA_FFI_RecordContext*>(&state),
        &dummy_record_api,
        XLA_FFI_RecordAction_Update,
        commands_storage,
        &num_commands,
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

}  // namespace
}  // namespace xla::ffi
