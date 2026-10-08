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

#include "xla/ffi/api/record_ffi.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

TEST(FfiTest, RecordStage) {
  bool called = false;

  auto handler = Ffi::BindRecord().Ctx<Extension<RecordExtension>>().To(
      [&](RecordContext ctx) {
        called = true;
        EXPECT_EQ(ctx.action(), RecordAction::kCreate);
        EXPECT_EQ(ctx.commands().capacity(), 10);
        EXPECT_EQ(ctx.commands().size(), 0);
        auto cmd = ctx.CreateLaunch(
            /*kernel_name=*/"test", /*kernel_data=*/"dummy_ptx",
            /*kernel_size=*/9, /*format=*/SourceFormat::kPtx,
            /*launch_dims=*/{{1, 1, 1}, {1, 1, 1}}, /*shared_mem_bytes=*/0,
            /*uses_pdl=*/false, /*args=*/std::vector<KernelArg>{});
        return Error::Success();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  static constexpr auto kMockApi = [](auto... args) -> XLA_FFI_Error* {
    return nullptr;
  };
  XLA_FFI_RecordApi mock_api{
      /*.create_launch=*/kMockApi,
      /*.create_memcpy_d2d=*/kMockApi,
      /*.update_memcpy_d2d=*/kMockApi,
      /*.request_stream_capture=*/kMockApi,
      /*.create_empty_command=*/kMockApi,
  };

  const XLA_FFI_Command* commands_storage[10] = {nullptr};
  int64_t num_commands = 0;

  XLA_FFI_RecordFrame record_frame = {/*record_ctx=*/nullptr,
                                      /*api=*/&mock_api,
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
  EXPECT_TRUE(called);
}

TEST(FfiTest, CreateLaunchWithFunctionPtr) {
  struct CaptureState {
    const void* kernel_data = nullptr;
    int64_t kernel_size = -1;
    XLA_FFI_SourceFormat format = XLA_FFI_SourceFormat_PTX;
    const XLA_FFI_Command* dummy_cmd =
        reinterpret_cast<const XLA_FFI_Command*>(0x1);
  } state;

  const void* expected_func_ptr = reinterpret_cast<const void*>(0xDEADBEEF);

  auto handler = Ffi::BindRecord().Ctx<Extension<RecordExtension>>().To(
      [&](RecordContext ctx) {
        auto null_cmd = ctx.CreateLaunch(
            /*kernel_name=*/"null_func", /*kernel_data=*/nullptr,
            /*kernel_size=*/0, /*format=*/SourceFormat::kFunctionPtr,
            /*launch_dims=*/{{1, 1, 1}, {1, 1, 1}}, /*shared_mem_bytes=*/0,
            /*uses_pdl=*/false, /*args=*/std::vector<KernelArg>{});
        EXPECT_FALSE(null_cmd.has_value());

        auto cmd = ctx.CreateLaunch(
            /*kernel_name=*/"test_func_ptr", /*kernel_data=*/expected_func_ptr,
            /*kernel_size=*/0, /*format=*/SourceFormat::kFunctionPtr,
            /*launch_dims=*/{{1, 1, 1}, {1, 1, 1}}, /*shared_mem_bytes=*/0,
            /*uses_pdl=*/false, /*args=*/std::vector<KernelArg>{});
        EXPECT_TRUE(cmd.has_value());
        EXPECT_EQ(*cmd, state.dummy_cmd);
        return Error::Success();
      });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  static constexpr auto kMockApi = [](auto... args) -> XLA_FFI_Error* {
    return nullptr;
  };
  XLA_FFI_RecordApi mock_api{
      /*.create_launch=*/+[](XLA_FFI_RecordContext* ctx, const char*,
                             const void* kernel_data, int64_t kernel_size,
                             XLA_FFI_SourceFormat format, XLA_FFI_LaunchDims,
                             uint32_t, int32_t, const XLA_FFI_KernelArgs*,
                             const XLA_FFI_Command* const*, uint32_t,
                             const XLA_FFI_Command** out_command)
                              -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<CaptureState*>(ctx);
        s->kernel_data = kernel_data;
        s->kernel_size = kernel_size;
        s->format = format;
        *out_command = s->dummy_cmd;
        return nullptr;
      },
      /*.update_launch=*/kMockApi,
      /*.create_memcpy_d2d=*/kMockApi,
      /*.update_memcpy_d2d=*/kMockApi,
      /*.request_stream_capture=*/kMockApi,
      /*.create_empty_command=*/kMockApi,
  };

  const XLA_FFI_Command* commands_storage[10] = {nullptr};
  int64_t num_commands = 0;

  XLA_FFI_RecordFrame record_frame = {
      /*record_ctx=*/reinterpret_cast<XLA_FFI_RecordContext*>(&state),
      /*api=*/&mock_api,
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
      [&](RecordContext) { return Error::Success(); });

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
