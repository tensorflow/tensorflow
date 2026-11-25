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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/ffi_kernel_registry.h"
#include "xla/backends/gpu/runtime/multi_gpu_barrier.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/ffi/api/kernel_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/ffi/invoke.h"
#include "xla/ffi/record_ffi.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/mock_command_buffer.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::StatusOr<void*> LoadNativeFunctionPtr(se::StreamExecutor* executor,
                                            absl::Span<const uint8_t> binary,
                                            absl::string_view name);

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::InSequence;
using ::testing::Matcher;
using ::testing::Ref;
using ::testing::Return;

struct KernelBinary {
  absl::Span<const uint8_t> bytes;

  const void* data() const { return bytes.data(); }
  size_t size() const { return bytes.size(); }
};

// Select the kernel bytes based on the platform.
// Selects the device binary (CUBIN on CUDA, HSACO on ROCm) via the fatbin.
// The bytes must outlive the handler, so the fatbin is kept in a static
// variable with a stable address.
absl::StatusOr<KernelBinary> GetKernelSpec() {
  static const absl::NoDestructor<absl::StatusOr<std::vector<uint8_t>>> kFatbin(
      []() -> absl::StatusOr<std::vector<uint8_t>> {
        ABSL_ASSIGN_OR_RETURN(auto fatbin,
                         stream_executor::gpu::GetGpuTestKernelsFatbin(
                             stream_executor::GpuPlatformName()));
        return fatbin;
      }());
  ABSL_RETURN_IF_ERROR(kFatbin->status());
  return KernelBinary{/*.bytes=*/kFatbin->value()};
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

void RunRecordFfiTest(se::StreamExecutor* executor, ffi::Ffi& handler) {
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> b = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<int32_t> c(8, 0);
  DeviceMemoryBundle initial_memory;
  initial_memory.Allocate(executor, 8);
  se::DeviceAddress<int32_t> ext_src = executor->AllocateArray<int32_t>(8);
  ASSERT_OK(stream->Memcpy(&ext_src, a.data(), a.size() * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&initial_memory.a, a.size() * sizeof(int32_t)));
  ASSERT_OK(
      stream->Memcpy(&initial_memory.b, b.data(), b.size() * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&initial_memory.scratch, 8 * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&initial_memory.c, c.size() * sizeof(int32_t)));

  ASSERT_OK_AND_ASSIGN(auto cmd_buffer,
                       executor->CreateCommandBuffer(
                           stream_executor::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* ext_dep,
      cmd_buffer->CreateMemcpyD2D(&initial_memory.a, ext_src,
                                  a.size() * sizeof(int32_t), {}));

  const XLA_FFI_RecordApi* ffi_api = GetXlaFfiRecordApi();
  int64_t num_commands = 0;
  const XLA_FFI_Command* commands_storage[2] = {nullptr, nullptr};

  XLA_FFI_RecordContext record_ctx_c = {
      cmd_buffer.get(), executor, {ext_dep}, false};
  XLA_FFI_RecordFrame record_frame = {
      &record_ctx_c,    ffi_api,       XLA_FFI_RecordAction_Create,
      commands_storage, &num_commands, 2};
  auto record_extension = ffi::BuildRecordCExtension(&record_frame);

  ffi::InvokeContext invoke_context = {};
  invoke_context.extension_start = &record_extension.extension_base;

  ffi::CallFrameBuilder builder(/*num_args=*/4, /*num_rets=*/0);
  initial_memory.AddAsBuffers(builder);

  ffi::CallFrame call_frame = builder.Build();

  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), handler, call_frame,
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
  se::DeviceAddress<int32_t> update_ext_src =
      executor->AllocateArray<int32_t>(8);
  ASSERT_OK(stream->Memcpy(&update_ext_src, a_new.data(),
                           a_new.size() * sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&update_memory.a, a_new.size() * sizeof(int32_t)));
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
  ASSERT_OK(cmd_buffer->UpdateMemcpyD2D(ext_dep, &update_memory.a,
                                        update_ext_src,
                                        a_new.size() * sizeof(int32_t)));
  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), handler, call_frame,
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

template <typename CreateLaunchFn>
std::unique_ptr<ffi::Ffi> MakeAddI32RecordHandler(
    CreateLaunchFn create_launch) {
  return ffi::Ffi::BindRecord()
      .Ctx<ffi::Extension<ffi::RecordExtension>>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .Arg<ffi::AnyBuffer>()
      .To([create_launch = std::move(create_launch)](
              ffi::RecordContext record_ctx, ffi::AnyBuffer input_0,
              ffi::AnyBuffer input_1, ffi::AnyBuffer buffer_scratch,
              ffi::AnyBuffer result) -> absl::Status {
        void* in0 = input_0.untyped_data();
        void* in1 = input_1.untyped_data();
        void* scratch = buffer_scratch.untyped_data();
        void* res = result.untyped_data();
        size_t size = 8 * sizeof(int32_t);
        std::vector<ffi::KernelArg> args = {ffi::DevicePointer{scratch},
                                            ffi::DevicePointer{in1},
                                            ffi::DevicePointer{res}};

        if (record_ctx.action() == ffi::RecordAction::kCreate) {
          ABSL_ASSIGN_OR_RETURN(const XLA_FFI_Command* memcpy_d2d,
                           record_ctx.CreateMemcpyD2D(scratch, in0, size));
          return create_launch(record_ctx, args, memcpy_d2d);
        }
        if (record_ctx.action() == ffi::RecordAction::kUpdate) {
          auto cmds = record_ctx.commands();
          TF_RET_CHECK(cmds.size() == 2) << "Expected 2 commands recorded.";
          ABSL_RETURN_IF_ERROR(
              record_ctx.UpdateMemcpyD2D(cmds[0], scratch, in0, size));
          return record_ctx.UpdateLaunch(cmds[1], args);
        }
        return absl::OkStatus();
      });
}

TEST(RecordFfiTest, KernelLaunchWithCuFunc) {
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

  ASSERT_OK_AND_ASSIGN(KernelBinary binary, GetKernelSpec());
  absl::StatusOr<void*> func_ptr =
      LoadNativeFunctionPtr(executor, binary.bytes, "AddI32");
  if (absl::IsUnimplemented(func_ptr.status())) {
    GTEST_SKIP() << func_ptr.status().message();
  }
  ASSERT_OK(func_ptr.status());

  std::unique_ptr<ffi::Ffi> handler = MakeAddI32RecordHandler(
      [func_ptr = *func_ptr](ffi::RecordContext& record_ctx,
                             const std::vector<ffi::KernelArg>& args,
                             const XLA_FFI_Command* memcpy_d2d) {
        return record_ctx
            .CreateLaunch("AddI32", func_ptr,
                          /*launch_dims=*/{{1, 1, 1}, {8, 1, 1}},
                          /*shared_mem_bytes=*/1024, /*uses_pdl=*/false, args,
                          /*dependencies=*/{memcpy_d2d})
            .status();
      });
  RunRecordFfiTest(executor, *handler);
}

TEST(RecordFfiTest, KernelLaunchWithRegisteredKernel) {
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

  ASSERT_OK_AND_ASSIGN(KernelBinary binary, GetKernelSpec());
  FfiKernelRegistry registry(platform->VisibleDeviceCount());
  ASSERT_OK_AND_ASSIGN(
      FfiKernelRegistry::RegisterResult registered,
      registry.Register(executor, "AddI32", binary.data(), binary.size(),
                        XLA_FFI_SourceFormat_CUBIN, /*use_pdl=*/false,
                        /*required_features=*/0));
  const XLA_FFI_Kernel* kernel = registered.kernel;

  std::unique_ptr<ffi::Ffi> handler =
      MakeAddI32RecordHandler([kernel](ffi::RecordContext& record_ctx,
                                       const std::vector<ffi::KernelArg>& args,
                                       const XLA_FFI_Command* memcpy_d2d) {
        return record_ctx
            .CreateLaunch(kernel,
                          /*launch_dims=*/{{1, 1, 1}, {8, 1, 1}},
                          /*shared_mem_bytes=*/1024, args,
                          /*dependencies=*/{memcpy_d2d})
            .status();
      });
  RunRecordFfiTest(executor, *handler);
}

class FakeSymmetricMemory : public xla::SymmetricMemory {
 public:
  se::DeviceAddressBase addr() const override {
    return se::DeviceAddressBase();
  }
  std::string ToString() const override { return "FakeSymmetricMemory"; }
  PackedKernelArg PackKernelArg() const override { return nullptr; }
};

TEST(RecordFfiTest, RecordMultiGpuBarrierWithNccl) {
  ASSERT_OK_AND_ASSIGN(auto platform,
                       stream_executor::PlatformManager::PlatformWithName(
                           stream_executor::GpuPlatformName()));
  ASSERT_OK_AND_ASSIGN(auto* executor, platform->ExecutorForDevice(0));
  if (!executor->GetDeviceDescription()
           .gpu_compute_capability()
           .cuda_compute_capability()) {
    GTEST_SKIP() << "MultiGpuBarrierWithNcclKernel is only supported on CUDA.";
  }
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  FakeSymmetricMemory sym_mem;
  ASSERT_OK_AND_ASSIGN(auto kernel,
                       LoadMultiGpuBarrierWithNcclKernel(executor));
  std::unique_ptr<ffi::Ffi> handler =
      ffi::Ffi::BindRecord()
          .Ctx<ffi::Extension<ffi::RecordExtension>>()
          .Arg<ffi::AnyBuffer>()
          .To([&kernel, &sym_mem](ffi::RecordContext record_ctx,
                                  ffi::AnyBuffer counter) -> absl::Status {
            const XLA_FFI_Command* cmd =
                record_ctx.action() == ffi::RecordAction::kUpdate
                    ? record_ctx.commands()[0]
                    : nullptr;
            return RecordMultiGpuBarrierWithNccl(kernel, record_ctx,
                                                 /*num_devices=*/0, RankId(0),
                                                 &sym_mem,
                                                 counter.device_memory(), cmd)
                .status();
          });

  se::DeviceAddress<uint32_t> counter_1 = executor->AllocateArray<uint32_t>(1);
  ASSERT_OK(stream->MemZero(&counter_1, sizeof(uint32_t)));

  ASSERT_OK_AND_ASSIGN(auto cmd_buffer,
                       executor->CreateCommandBuffer(
                           stream_executor::CommandBuffer::Mode::kPrimary));

  const XLA_FFI_RecordApi* ffi_api = GetXlaFfiRecordApi();
  int64_t num_commands = 0;
  const XLA_FFI_Command* commands_storage[1] = {nullptr};

  XLA_FFI_RecordContext record_ctx_c = {cmd_buffer.get(), executor, {}, false};
  XLA_FFI_RecordFrame record_frame = {
      &record_ctx_c,    ffi_api,       XLA_FFI_RecordAction_Create,
      commands_storage, &num_commands, 1};
  auto record_extension = ffi::BuildRecordCExtension(&record_frame);

  ffi::InvokeContext invoke_context = {};
  invoke_context.extension_start = &record_extension.extension_base;

  ffi::CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(counter_1, PrimitiveType::U32, {1});
  ffi::CallFrame call_frame = builder.Build();

  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), *handler, call_frame,
                        invoke_context, ffi::ExecutionStage::kRecord));
  ASSERT_OK(cmd_buffer->Finalize());
  ASSERT_OK(cmd_buffer->Submit(stream.get()));

  uint32_t host_counter = 0;
  ASSERT_OK(stream->Memcpy(&host_counter, counter_1, sizeof(uint32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(host_counter, 1);

  // Update with new buffer.
  se::DeviceAddress<uint32_t> counter_2 = executor->AllocateArray<uint32_t>(1);
  ASSERT_OK(stream->MemZero(&counter_2, sizeof(uint32_t)));

  record_frame.action = XLA_FFI_RecordAction_Update;
  ASSERT_OK(call_frame.UpdateWithBuffers({counter_2}, /*rets=*/{}));
  ASSERT_OK(cmd_buffer->Update());
  ASSERT_OK(ffi::Invoke(ffi::GetXlaFfiApi(), *handler, call_frame,
                        invoke_context, ffi::ExecutionStage::kRecord));
  ASSERT_OK(cmd_buffer->Finalize());
  ASSERT_OK(cmd_buffer->Submit(stream.get()));

  host_counter = 0;
  ASSERT_OK(stream->Memcpy(&host_counter, counter_2, sizeof(uint32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(host_counter, 1);
}

TEST(RecordFfiTest, KernelLaunchRejectsInvalidRecordKernel) {
  ASSERT_OK_AND_ASSIGN(auto platform,
                       stream_executor::PlatformManager::PlatformWithName(
                           stream_executor::GpuPlatformName()));
  ASSERT_OK_AND_ASSIGN(auto* executor, platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(auto cmd_buffer,
                       executor->CreateCommandBuffer(
                           stream_executor::CommandBuffer::Mode::kPrimary));

  const XLA_FFI_RecordApi* ffi_api = GetXlaFfiRecordApi();
  XLA_FFI_RecordContext record_ctx = {cmd_buffer.get(), executor, {}, false};
  XLA_FFI_KernelArgs empty_args = {nullptr, 0};
  XLA_FFI_KernelLaunchParams launch_params = {
      XLA_FFI_KernelLaunchParams_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      /*launch_dims=*/{{1, 1, 1}, {1, 1, 1}},
      /*shared_mem_bytes=*/0,
      /*uses_pdl=*/0,
      &empty_args,
  };
  const auto* dummy_kernel =
      reinterpret_cast<const XLA_FFI_Kernel*>(0x87654321);
  void* dummy_fn = reinterpret_cast<void*>(0x12345678);
  const XLA_FFI_Command* out_cmd = nullptr;

  auto call = [&](XLA_FFI_RecordContext* ctx,
                  const XLA_FFI_RecordKernel& kernel,
                  const XLA_FFI_Command** out) -> absl::Status {
    return ffi::TakeError(
        ffi_api->create_launch(ctx, &kernel, &launch_params, nullptr, 0, out));
  };

  XLA_FFI_RecordKernel both_set = {XLA_FFI_RecordKernel_STRUCT_SIZE,
                                   nullptr,
                                   dummy_kernel,
                                   dummy_fn,
                                   {nullptr, 0}};
  EXPECT_THAT(call(nullptr, both_set, &out_cmd),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(call(&record_ctx, both_set, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(call(&record_ctx, both_set, &out_cmd),
              StatusIs(absl::StatusCode::kInvalidArgument));

  XLA_FFI_RecordKernel neither_set = {XLA_FFI_RecordKernel_STRUCT_SIZE,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      {nullptr, 0}};
  EXPECT_THAT(call(&record_ctx, neither_set, &out_cmd),
              StatusIs(absl::StatusCode::kInvalidArgument));

  XLA_FFI_RecordKernel null_name_ptr = {XLA_FFI_RecordKernel_STRUCT_SIZE,
                                        nullptr,
                                        nullptr,
                                        dummy_fn,
                                        {nullptr, 4}};
  EXPECT_THAT(call(&record_ctx, null_name_ptr, &out_cmd),
              StatusIs(absl::StatusCode::kInvalidArgument));

  XLA_FFI_RecordKernel valid_fn = {XLA_FFI_RecordKernel_STRUCT_SIZE,
                                   nullptr,
                                   nullptr,
                                   dummy_fn,
                                   {nullptr, 0}};
  EXPECT_THAT(
      ffi::TakeError(ffi_api->create_launch(
          &record_ctx, &valid_fn, &launch_params, nullptr, 1, &out_cmd)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  const XLA_FFI_Command* null_dep = nullptr;
  EXPECT_THAT(
      ffi::TakeError(ffi_api->create_launch(
          &record_ctx, &valid_fn, &launch_params, &null_dep, 1, &out_cmd)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

class FakeCommand : public se::CommandBuffer::Command {};

class FakeKernel : public se::Kernel {
 public:
  unsigned Arity() const override { return 0; }
  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(se::ThreadDim,
                                                      size_t) const override {
    return 1;
  }
  absl::Status Launch(const se::ThreadDim&, const se::BlockDim&,
                      const std::optional<se::ClusterDim>&, se::Stream*,
                      const se::KernelArgs&) override {
    return absl::OkStatus();
  }
};

TEST(RecordFfiTest, RecordCommandsResolveContextAndExplicitDependencies) {
  se::MockStreamExecutor mock_executor;
  ON_CALL(mock_executor, device_ordinal).WillByDefault(Return(0));
  EXPECT_CALL(mock_executor, LoadKernel).WillOnce([] {
    return std::make_unique<FakeKernel>();
  });

  FfiKernelRegistry registry(/*num_devices=*/1);
  constexpr char kDummyCubin[] = "cubin";
  ASSERT_OK_AND_ASSIGN(
      FfiKernelRegistry::RegisterResult registered,
      registry.Register(&mock_executor, "k0", kDummyCubin, sizeof(kDummyCubin),
                        XLA_FFI_SourceFormat_CUBIN, /*use_pdl=*/false,
                        /*required_features=*/0));
  const XLA_FFI_Kernel* kernel = registered.kernel;
  ASSERT_OK_AND_ASSIGN(se::Kernel * se_kernel,
                       FindDeviceKernel(kernel, /*device_ordinal=*/0));

  se::MockCommandBuffer mock_cmd_buffer;
  FakeCommand ext_dep_0;
  FakeCommand ext_dep_1;
  FakeCommand memcpy_root_se_cmd;
  FakeCommand empty_root_se_cmd;
  FakeCommand launch_root_se_cmd;
  FakeCommand launch_chained_se_cmd;
  FakeCommand memcpy_chained_se_cmd;
  FakeCommand empty_join_se_cmd;

  XLA_FFI_RecordContext record_ctx_c = {
      &mock_cmd_buffer, &mock_executor, {&ext_dep_0, &ext_dep_1}, false};
  const XLA_FFI_Command* commands_storage[8] = {};
  int64_t num_commands = 0;
  XLA_FFI_RecordFrame record_frame = {
      &record_ctx_c,    GetXlaFfiRecordApi(), XLA_FFI_RecordAction_Create,
      commands_storage, &num_commands,        8};
  ffi::RecordContext record_ctx(ffi::GetXlaFfiApi(), &record_frame);

  void* dummy_dst = reinterpret_cast<void*>(0x1000);
  void* dummy_src = reinterpret_cast<void*>(0x2000);
  constexpr size_t kCopyBytes = 32;
  const XLA_FFI_LaunchDims dims = {{1, 1, 1}, {1, 1, 1}};
  const std::vector<ffi::KernelArg> empty_args;

  {
    InSequence seq;
    EXPECT_CALL(
        mock_cmd_buffer,
        CreateMemcpyD2D(_, _, kCopyBytes, ElementsAre(&ext_dep_0, &ext_dep_1)))
        .WillOnce(Return(&memcpy_root_se_cmd));
    EXPECT_CALL(mock_cmd_buffer,
                CreateEmptyCmd(ElementsAre(&ext_dep_0, &ext_dep_1), _))
        .WillOnce(Return(&empty_root_se_cmd));
    EXPECT_CALL(
        mock_cmd_buffer,
        CreateLaunch(_, _, _, Matcher<const se::Kernel&>(Ref(*se_kernel)), _,
                     ElementsAre(&ext_dep_0, &ext_dep_1), _))
        .WillOnce(Return(&launch_root_se_cmd));
    EXPECT_CALL(
        mock_cmd_buffer,
        CreateLaunch(_, _, _, Matcher<const se::Kernel&>(Ref(*se_kernel)), _,
                     ElementsAre(&memcpy_root_se_cmd), _))
        .WillOnce(Return(&launch_chained_se_cmd));
    EXPECT_CALL(
        mock_cmd_buffer,
        CreateMemcpyD2D(_, _, kCopyBytes, ElementsAre(&launch_root_se_cmd)))
        .WillOnce(Return(&memcpy_chained_se_cmd));
    EXPECT_CALL(
        mock_cmd_buffer,
        CreateEmptyCmd(
            ElementsAre(&launch_chained_se_cmd, &memcpy_chained_se_cmd), _))
        .WillOnce(Return(&empty_join_se_cmd));
  }

  ASSERT_OK_AND_ASSIGN(
      const XLA_FFI_Command* memcpy_root_cmd,
      record_ctx.CreateMemcpyD2D(dummy_dst, dummy_src, kCopyBytes));
  EXPECT_THAT(record_ctx.CreateEmptyCommand(), IsOk());
  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Command* launch_root_cmd,
                       record_ctx.CreateLaunch(
                           kernel, dims, /*shared_mem_bytes=*/0, empty_args));
  ASSERT_OK_AND_ASSIGN(
      const XLA_FFI_Command* launch_chained_cmd,
      record_ctx.CreateLaunch(kernel, dims, /*shared_mem_bytes=*/0, empty_args,
                              /*dependencies=*/{memcpy_root_cmd}));
  ASSERT_OK_AND_ASSIGN(
      const XLA_FFI_Command* memcpy_chained_cmd,
      record_ctx.CreateMemcpyD2D(dummy_dst, dummy_src, kCopyBytes,
                                 /*dependencies=*/{launch_root_cmd}));
  EXPECT_THAT(record_ctx.CreateEmptyCommand(
                  /*dependencies=*/{launch_chained_cmd, memcpy_chained_cmd}),
              IsOk());
}

TEST(RecordFfiTest, MemcpyD2DAndEmptyCommandRejectInvalidArguments) {
  se::MockStreamExecutor mock_executor;
  se::MockCommandBuffer mock_cmd_buffer;
  XLA_FFI_RecordContext record_ctx = {
      &mock_cmd_buffer, &mock_executor, {}, false};
  const XLA_FFI_RecordApi* ffi_api = GetXlaFfiRecordApi();
  const XLA_FFI_Command* out_cmd = nullptr;
  const XLA_FFI_Command* null_dep = nullptr;
  void* dummy_dst = reinterpret_cast<void*>(0x1000);
  void* dummy_src = reinterpret_cast<void*>(0x2000);

  EXPECT_THAT(ffi::TakeError(ffi_api->create_memcpy_d2d(
                  nullptr, dummy_dst, dummy_src, 32, nullptr, 0, &out_cmd)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ctx and out_command")));
  EXPECT_THAT(ffi::TakeError(ffi_api->create_memcpy_d2d(
                  &record_ctx, dummy_dst, dummy_src, 32, nullptr, 0, nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ctx and out_command")));
  EXPECT_THAT(ffi::TakeError(ffi_api->create_memcpy_d2d(
                  &record_ctx, dummy_dst, dummy_src, 32, nullptr, 1, &out_cmd)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("dependencies must not be null")));
  EXPECT_THAT(
      ffi::TakeError(ffi_api->create_memcpy_d2d(
          &record_ctx, dummy_dst, dummy_src, 32, &null_dep, 1, &out_cmd)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("dependency command must not be null")));

  EXPECT_THAT(ffi::TakeError(
                  ffi_api->create_empty_command(nullptr, nullptr, 0, &out_cmd)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ctx and out_command")));
  EXPECT_THAT(ffi::TakeError(ffi_api->create_empty_command(&record_ctx, nullptr,
                                                           0, nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ctx and out_command")));
  EXPECT_THAT(ffi::TakeError(ffi_api->create_empty_command(&record_ctx, nullptr,
                                                           1, &out_cmd)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("dependencies must not be null")));
  EXPECT_THAT(ffi::TakeError(ffi_api->create_empty_command(
                  &record_ctx, &null_dep, 1, &out_cmd)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("dependency command must not be null")));
}

}  // namespace
}  // namespace xla::gpu
