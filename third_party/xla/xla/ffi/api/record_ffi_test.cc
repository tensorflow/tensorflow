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
        auto cmd = ctx.CreateLaunch("test", nullptr, 0, SourceFormat::kPtx,
                                    {{1, 1, 1}, {1, 1, 1}}, 0,
                                    std::vector<KernelArg>{});
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
                                      &mock_api,
                                      XLA_FFI_RecordAction_Create,
                                      commands_storage,
                                      &num_commands,
                                      10};

  RecordExtension::CExtension record_frame_ext =
      BuildRecordCExtension(&record_frame);

  InvokeContext context;
  context.extension_start = &record_frame_ext.extension_base;

  auto status =
      Invoke(Api(), *handler, call_frame, context, ExecutionStage::kRecord);

  ASSERT_OK(status);
  EXPECT_TRUE(called);
}

}  // namespace
}  // namespace xla::ffi
