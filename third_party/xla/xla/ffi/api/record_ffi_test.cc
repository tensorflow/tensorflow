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
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/kernel_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

using ::testing::ElementsAre;

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

class RecordStageTest : public ::testing::TestWithParam<bool> {};

TEST_P(RecordStageTest, CreateLaunch) {
  const bool use_kernel = GetParam();
  struct CaptureState {
    XLA_FFI_RecordKernel kernel{};
    XLA_FFI_KernelLaunchParams params{};
    std::vector<const XLA_FFI_Command*> deps;
    const XLA_FFI_Command* dep_cmd =
        reinterpret_cast<const XLA_FFI_Command*>(0x2);
    const XLA_FFI_Command* dummy_cmd =
        reinterpret_cast<const XLA_FFI_Command*>(0x1);
  } state;

  void* expected_func_ptr = reinterpret_cast<void*>(0xDEADBEEF);
  const auto* expected_kernel =
      reinterpret_cast<const XLA_FFI_Kernel*>(0x87654321);
  bool called = false;

  auto handler = Ffi::BindRecord().Ctx<Extension<RecordExtension>>().To(
      [&](RecordContext ctx) {
        called = true;
        EXPECT_EQ(ctx.action(), RecordAction::kCreate);
        EXPECT_EQ(ctx.commands().capacity(), 10);
        EXPECT_EQ(ctx.commands().size(), 0);

        XLA_FFI_LaunchDims dims{{1, 1, 1}, {1, 1, 1}};
        std::vector<KernelArg> args;
        if (use_kernel) {
          EXPECT_FALSE(ctx.CreateLaunch(/*kernel=*/nullptr, dims,
                                        /*shared_mem_bytes=*/0, args)
                           .has_value());
        } else {
          EXPECT_FALSE(ctx.CreateLaunch("null_func", /*function=*/nullptr, dims,
                                        /*shared_mem_bytes=*/0,
                                        /*uses_pdl=*/false, args)
                           .has_value());
          EXPECT_FALSE(ctx.CreateLaunch("bad_fmt", expected_func_ptr,
                                        /*kernel_size=*/0, SourceFormat::kCubin,
                                        dims, /*shared_mem_bytes=*/0,
                                        /*uses_pdl=*/false, args)
                           .has_value());
          EXPECT_FALSE(ctx.CreateLaunch("bad_sz", expected_func_ptr,
                                        /*kernel_size=*/8,
                                        SourceFormat::kFunctionPtr, dims,
                                        /*shared_mem_bytes=*/0,
                                        /*uses_pdl=*/false, args)
                           .has_value());
        }

        auto cmd = use_kernel
                       ? ctx.CreateLaunch(expected_kernel, dims,
                                          /*shared_mem_bytes=*/128, args,
                                          /*dependencies=*/{state.dep_cmd})
                       : ctx.CreateLaunch("test", expected_func_ptr, dims,
                                          /*shared_mem_bytes=*/128,
                                          /*uses_pdl=*/false, args,
                                          /*dependencies=*/{state.dep_cmd});
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
      /*.create_launch=*/+[](XLA_FFI_RecordContext* ctx,
                             const XLA_FFI_RecordKernel* kernel,
                             const XLA_FFI_KernelLaunchParams* launch_params,
                             const XLA_FFI_Command* const* dependencies,
                             uint32_t num_dependencies,
                             const XLA_FFI_Command** out_command)
                              -> XLA_FFI_Error* {
        auto* s = reinterpret_cast<CaptureState*>(ctx);
        s->kernel = *kernel;
        s->params = *launch_params;
        s->deps.assign(dependencies, dependencies + num_dependencies);
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

  auto status =
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kRecord);

  ASSERT_OK(status);
  EXPECT_TRUE(called);
  EXPECT_EQ(state.kernel.struct_size, XLA_FFI_RecordKernel_STRUCT_SIZE);
  EXPECT_EQ(state.kernel.kernel, use_kernel ? expected_kernel : nullptr);
  EXPECT_EQ(state.kernel.function, use_kernel ? nullptr : expected_func_ptr);
  EXPECT_EQ(std::string_view(state.kernel.function_name.ptr,
                             state.kernel.function_name.len),
            use_kernel ? "" : "test");
  EXPECT_EQ(state.params.shared_mem_bytes, 128u);
  EXPECT_EQ(state.params.uses_pdl, 0);
  EXPECT_THAT(state.deps, ElementsAre(state.dep_cmd));
  EXPECT_EQ(num_commands, 1);
}

INSTANTIATE_TEST_SUITE_P(All, RecordStageTest, ::testing::Bool());

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
