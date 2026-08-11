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

#include "xla/backends/gpu/runtime/record_ffi.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"
#include "xla/ffi/record_ffi.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

struct KernelBinary {
  absl::Span<const uint8_t> bytes;
  ffi::SourceFormat format;

  const void* data() const { return bytes.data(); }
  size_t size() const { return bytes.size(); }
};

// Select the kernel bytes based on the platform.
// Selects the device binary (CUBIN on CUDA, HSACO on ROCm) via the fatbin.
// The bytes must outlive the handler, so the fatbin is kept in a static
// variable with a stable address that is also used as the kernel cache key.
absl::StatusOr<KernelBinary> GetKernelSpec() {
  static const absl::NoDestructor<absl::StatusOr<std::vector<uint8_t>>> kFatbin(
      []() -> absl::StatusOr<std::vector<uint8_t>> {
        ABSL_ASSIGN_OR_RETURN(auto fatbin,
                         stream_executor::gpu::GetGpuTestKernelsFatbin(
                             stream_executor::GpuPlatformName()));
        return fatbin;
      }());
  ABSL_ASSIGN_OR_RETURN(auto const& fatbin, *kFatbin);
  return KernelBinary{/*.bytes=*/fatbin,
                      /*.format=*/ffi::SourceFormat::kCubin};
}

absl::Status RecordFfiHandler(ffi::RecordContext record_ctx,
                              ffi::AnyBuffer input_0, ffi::AnyBuffer input_1,
                              ffi::AnyBuffer buffer_scratch,
                              ffi::AnyBuffer result) {
  void* in0 = input_0.untyped_data();
  void* in1 = input_1.untyped_data();
  void* scratch = buffer_scratch.untyped_data();
  void* res = result.untyped_data();
  size_t size = 8 * sizeof(int32_t);

  const auto action = record_ctx.action();
  if (action == ffi::RecordAction::kCreate) {
    ABSL_ASSIGN_OR_RETURN(const XLA_FFI_Command* memcpy_d2d,
                     record_ctx.CreateMemcpyD2D(scratch, in0, size));
    ABSL_ASSIGN_OR_RETURN(KernelBinary binary, GetKernelSpec());
    ABSL_RETURN_IF_ERROR(
        record_ctx
            .CreateLaunch(
                "AddI32", binary.data(), binary.size(), binary.format,
                /*launch_dims=*/{{1, 1, 1}, {8, 1, 1}}, /*shared_mem_bytes=*/0,
                std::vector<ffi::KernelArg>{ffi::DevicePointer{scratch},
                                            ffi::DevicePointer{in1},
                                            ffi::DevicePointer{res}},
                /*dependencies=*/{memcpy_d2d})
            .status());
  } else if (action == ffi::RecordAction::kUpdate) {
    auto cmds = record_ctx.commands();
    TF_RET_CHECK(cmds.size() == 2) << "Expected 2 commands recorded.";

    ABSL_RETURN_IF_ERROR(record_ctx.UpdateMemcpyD2D(cmds[0], scratch, in0, size));
    ABSL_RETURN_IF_ERROR(record_ctx.UpdateLaunch(
        cmds[1], std::vector<ffi::KernelArg>{ffi::DevicePointer{scratch},
                                             ffi::DevicePointer{in1},
                                             ffi::DevicePointer{res}}));
  }
  return absl::OkStatus();
}

struct DeviceMemoryBundle {
  se::DeviceAddress<int32_t> a;
  se::DeviceAddress<int32_t> b;
  se::DeviceAddress<int32_t> scratch;
  se::DeviceAddress<int32_t> c;

  void Allocate(se::StreamExecutor* executor, size_t size) {
    for (auto* dev_ptr : {&a, &b, &scratch, &c}) {
      *dev_ptr = executor->AllocateArray<int32_t>(size);
    }
  }

  void AddAsBuffers(ffi::CallFrameBuilder& builder) const {
    for (auto& buffer : {a, b, scratch, c}) {
      builder.AddBufferArg(buffer, PrimitiveType::S32, {8});
    }
  }
};

TEST(RecordFfiTest, KernelLaunchBoundFfi) {
  ASSERT_OK_AND_ASSIGN(auto platform,
                       stream_executor::PlatformManager::PlatformWithName(
                           stream_executor::GpuPlatformName()));
  ASSERT_OK_AND_ASSIGN(auto* executor, platform->ExecutorForDevice(0));
  const auto* cuda_cc = executor->GetDeviceDescription()
                            .gpu_compute_capability()
                            .cuda_compute_capability();
  if (cuda_cc && !cuda_cc->IsAtLeastAmpere()) {
    GTEST_SKIP() << "Skipping test for compute capability less than Ampere.";
  }
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> b = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<int32_t> c(8, 0);
  DeviceMemoryBundle initial_memory;
  initial_memory.Allocate(executor, 8);
  ASSERT_OK(
      stream->Memcpy(&initial_memory.a, a.data(), a.size() * sizeof(int32_t)));
  ASSERT_OK(
      stream->Memcpy(&initial_memory.b, b.data(), b.size() * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&initial_memory.scratch, 8 * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&initial_memory.c, c.size() * sizeof(int32_t)));

  ASSERT_OK_AND_ASSIGN(auto cmd_buffer,
                       executor->CreateCommandBuffer(
                           stream_executor::CommandBuffer::Mode::kPrimary));

  const XLA_FFI_RecordApi* ffi_api = GetXlaFfiRecordApi();
  int64_t num_commands = 0;
  const XLA_FFI_Command* commands_storage[2] = {nullptr, nullptr};

  XLA_FFI_RecordContext record_ctx_c = {cmd_buffer.get(), executor, {}, false};
  XLA_FFI_RecordFrame record_frame = {
      &record_ctx_c,    ffi_api,       XLA_FFI_RecordAction_Create,
      commands_storage, &num_commands, 2};
  auto record_extension = ffi::BuildRecordCExtension(&record_frame);

  ffi::InvokeContext invoke_context = {};
  invoke_context.extension_start = &record_extension.extension_base;

  ffi::CallFrameBuilder builder(/*num_args=*/4, /*num_rets=*/0);
  initial_memory.AddAsBuffers(builder);

  ffi::CallFrame call_frame = builder.Build();

  std::unique_ptr<ffi::Ffi> handler =
      ffi::Ffi::BindRecord()
          .Ctx<ffi::Extension<ffi::RecordExtension>>()
          .Arg<ffi::AnyBuffer>()
          .Arg<ffi::AnyBuffer>()
          .Arg<ffi::AnyBuffer>()
          .Arg<ffi::AnyBuffer>()
          .To(RecordFfiHandler);
  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), *handler, call_frame,
                        invoke_context, ffi::ExecutionStage::kRecord));
  ASSERT_OK(cmd_buffer->Finalize());
  ASSERT_OK(cmd_buffer->Submit(stream.get()));

  ASSERT_OK(
      stream->Memcpy(c.data(), initial_memory.c, c.size() * sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<int32_t> expected_create = {11, 22, 33, 44, 55, 66, 77, 88};
  EXPECT_EQ(c, expected_create);

  // --- UPDATE PHASE ---
  std::vector<int32_t> a_new = {100, 200, 300, 400, 500, 600, 700, 800};
  std::vector<int32_t> b_new = {1, 2, 3, 4, 5, 6, 7, 8};
  DeviceMemoryBundle update_memory;
  update_memory.Allocate(executor, 8);
  ASSERT_OK(stream->Memcpy(&update_memory.a, a_new.data(),
                           a_new.size() * sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&update_memory.b, b_new.data(),
                           b_new.size() * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&update_memory.scratch, 8 * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&update_memory.c, c.size() * sizeof(int32_t)));

  record_frame.action = XLA_FFI_RecordAction_Update;
  ASSERT_OK(
      call_frame.UpdateWithBuffers({update_memory.a, update_memory.b,
                                    update_memory.scratch, update_memory.c},
                                   /*rets=*/{}));
  ffi::InvokeContext update_context;
  update_context.extension_start = &record_extension.extension_base;
  ASSERT_OK(cmd_buffer->Update());  // Begin command buffer update.
  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), *handler, call_frame,
                        update_context, ffi::ExecutionStage::kRecord));
  ASSERT_OK(cmd_buffer->Finalize());
  ASSERT_OK(cmd_buffer->Submit(stream.get()));
  ASSERT_OK(
      stream->Memcpy(c.data(), update_memory.c, c.size() * sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<int32_t> expected_update = {101, 202, 303, 404,
                                          505, 606, 707, 808};
  EXPECT_EQ(c, expected_update);
}

}  // namespace
}  // namespace xla::gpu
