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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

class RecordStageTest : public ::testing::TestWithParam<bool> {};

TEST_P(RecordStageTest, CreateAndUpdateCommands) {
  const bool use_kernel = GetParam();
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

    XLA_FFI_RecordKernel kernel{};
    XLA_FFI_KernelLaunchParams params{};
    std::vector<const XLA_FFI_Command*> deps;

    bool launch_created = false;
    bool launch_updated = false;
    bool memcpy_created = false;
    bool memcpy_updated = false;
    bool empty_created = false;
  };

  RecordState state;
  void* expected_func_ptr = reinterpret_cast<void*>(0x12345678);
  const auto* expected_kernel =
      reinterpret_cast<const XLA_FFI_Kernel*>(0x87654321);

  XLA_FFI_RecordApi dummy_record_api = {
      /*create_launch=*/+[](XLA_FFI_RecordContext* ctx,
                            const XLA_FFI_RecordKernel* kernel,
                            const XLA_FFI_KernelLaunchParams* launch_params,
                            const XLA_FFI_Command* const* dependencies,
                            uint32_t num_dependencies,
                            const XLA_FFI_Command** out_command)
                             -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<RecordState*>(ctx);
        s->kernel = *kernel;
        s->params = *launch_params;
        s->deps.assign(dependencies, dependencies + num_dependencies);
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
          auto empty_cmd = ctx.CreateEmptyCommand();
          if (!empty_cmd.ok()) {
            return empty_cmd.status();
          }
          const XLA_FFI_Command* dep = *empty_cmd;
          EXPECT_EQ(dep, state.empty_cmd);
          state.empty_created = true;

          XLA_FFI_LaunchDims dims{{1, 1, 1}, {1, 1, 1}};
          std::initializer_list<KernelArg> args{};
          if (use_kernel) {
            EXPECT_THAT(ctx.CreateLaunch(/*kernel=*/nullptr, dims,
                                         /*shared_mem_bytes=*/0, args),
                        StatusIs(absl::StatusCode::kInvalidArgument));
          } else {
            EXPECT_THAT(ctx.CreateLaunch(
                            "null_func", /*function=*/nullptr, dims,
                            /*shared_mem_bytes=*/0, /*uses_pdl=*/false, args),
                        StatusIs(absl::StatusCode::kInvalidArgument));
            EXPECT_THAT(ctx.CreateLaunch(
                            "bad_fmt", expected_func_ptr,
                            /*kernel_size=*/0, SourceFormat::kCubin, dims,
                            /*shared_mem_bytes=*/0, /*uses_pdl=*/false, args),
                        StatusIs(absl::StatusCode::kInvalidArgument));
            EXPECT_THAT(ctx.CreateLaunch(
                            "bad_sz", expected_func_ptr, /*kernel_size=*/8,
                            SourceFormat::kFunctionPtr, dims,
                            /*shared_mem_bytes=*/0, /*uses_pdl=*/false, args),
                        StatusIs(absl::StatusCode::kInvalidArgument));
          }

          auto launch_cmd =
              use_kernel
                  ? ctx.CreateLaunch(expected_kernel, dims,
                                     /*shared_mem_bytes=*/256, args,
                                     /*dependencies=*/{dep})
                  : ctx.CreateLaunch("dummy_kernel", expected_func_ptr, dims,
                                     /*shared_mem_bytes=*/256,
                                     /*uses_pdl=*/false, args,
                                     /*dependencies=*/{dep});
          EXPECT_THAT(launch_cmd, IsOkAndHolds(Eq(state.launch_cmd)));
          state.launch_created = true;

          int src = 0, dst = 0;
          auto memcpy_cmd = ctx.CreateMemcpyD2D(
              /*dst=*/&dst, /*src=*/&src, /*size=*/sizeof(int));
          EXPECT_THAT(memcpy_cmd, IsOkAndHolds(Eq(state.memcpy_cmd)));
          state.memcpy_created = true;

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
    EXPECT_EQ(state.kernel.struct_size, XLA_FFI_RecordKernel_STRUCT_SIZE);
    EXPECT_EQ(state.kernel.kernel, use_kernel ? expected_kernel : nullptr);
    EXPECT_EQ(state.kernel.function, use_kernel ? nullptr : expected_func_ptr);
    EXPECT_EQ(absl::string_view(state.kernel.function_name.ptr,
                                state.kernel.function_name.len),
              use_kernel ? "" : "dummy_kernel");
    EXPECT_EQ(state.params.shared_mem_bytes, 256u);
    EXPECT_EQ(state.params.uses_pdl, 0);
    EXPECT_THAT(state.deps, ElementsAre(state.empty_cmd));
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

INSTANTIATE_TEST_SUITE_P(All, RecordStageTest, ::testing::Bool());

TEST(FfiTest, RecordCreateLaunchChecksCapacity) {
  struct State {
    int calls = 0;
    int dummy = 0;
  } state;
  const auto* kernel = reinterpret_cast<const XLA_FFI_Kernel*>(0x87654321);

  absl::Status launch_status;
  auto handler = Ffi::BindRecord().Ctx<Extension<RecordExtension>>().To(
      [&](RecordContext ctx) -> absl::Status {
        launch_status = ctx.CreateLaunch(kernel, XLA_FFI_LaunchDims{},
                                         /*shared_mem_bytes=*/0,
                                         std::initializer_list<KernelArg>{})
                            .status();
        return absl::OkStatus();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto run = [&](const XLA_FFI_RecordApi* api, int64_t max_commands) {
    const XLA_FFI_Command* commands_storage[1] = {nullptr};
    int64_t num_commands = 0;
    XLA_FFI_RecordFrame record_frame = {
        /*record_ctx=*/reinterpret_cast<XLA_FFI_RecordContext*>(&state),
        /*api=*/api,
        /*action=*/XLA_FFI_RecordAction_Create,
        /*commands=*/commands_storage,
        /*num_commands=*/&num_commands,
        /*max_commands=*/max_commands};
    RecordExtension::CExtension record_frame_ext =
        BuildRecordCExtension(&record_frame);
    InvokeContext context;
    context.extension_start = &record_frame_ext.extension_base;
    ASSERT_OK(
        Invoke(Api(), *handler, call_frame, context, ExecutionStage::kRecord));
  };

  // With no room left, the backend is never called.
  XLA_FFI_RecordApi api{};
  api.create_launch =
      +[](XLA_FFI_RecordContext* ctx, const XLA_FFI_RecordKernel*,
          const XLA_FFI_KernelLaunchParams*, const XLA_FFI_Command* const*,
          uint32_t, const XLA_FFI_Command** out_command) -> XLA_FFI_Error* {
    auto* s = reinterpret_cast<State*>(ctx);
    ++s->calls;
    *out_command = reinterpret_cast<const XLA_FFI_Command*>(&s->dummy);
    return nullptr;
  };
  run(&api, /*max_commands=*/0);
  EXPECT_THAT(launch_status, StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(state.calls, 0);

  run(&api, /*max_commands=*/1);
  EXPECT_OK(launch_status);
  EXPECT_EQ(state.calls, 1);
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
