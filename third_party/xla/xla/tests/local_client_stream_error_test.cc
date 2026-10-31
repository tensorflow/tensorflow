/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/local_client_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class LocalClientStreamErrorTest : public LocalClientTestBase {};

struct KernelHolder {
  std::unique_ptr<se::Kernel> kernel;
};

absl::Status IllegalAccess(se::Stream* stream, KernelHolder* holder) {
  static constexpr absl::string_view kPtx = R"(
    .version 4.2
    .target sm_50
    .address_size 64
    .visible .entry IllegalAccess() {
      .reg .u64 %addr;
      mov.u64 %addr, 0x0;
      st.u64 [%addr], %addr;
      ret;
    }
  )";
  if (holder->kernel == nullptr) {
    TF_ASSIGN_OR_RETURN(
        holder->kernel,
        gpu::CreateKernel("IllegalAccess", 0, kPtx, stream->parent(), 0));
  }
  return gpu::ExecuteKernelOnStream(*holder->kernel, {},
                                    xla::gpu::LaunchDimensions(1, 1),
                                    std::nullopt, stream);
}

XLA_FFI_DEFINE_HANDLER(
    kIllegalAccess, IllegalAccess,
    ffi::Ffi::Bind().Ctx<ffi::Stream>().Ctx<ffi::UserData<KernelHolder>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "IllegalAccess", "CUDA",
                         kIllegalAccess);

class AsyncDeallocationAllocator : public se::DeviceAddressAllocator {
 public:
  using se::DeviceAddressAllocator::Allocate;

  explicit AsyncDeallocationAllocator(se::DeviceAddressAllocator* underlying)
      : se::DeviceAddressAllocator(underlying->platform()),
        underlying_(underlying) {}

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    TF_ASSIGN_OR_RETURN(
        auto mem, underlying_->Allocate(device_ordinal, size, retry_on_failure,
                                        memory_space));
    se::DeviceAddress<uint8_t> raw_addr = mem.Release();
    return se::ScopedDeviceAddress<uint8_t>(raw_addr, device_ordinal, this);
  }

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceAddressBase mem) override {
    if (tracked_ptr_ != nullptr && mem.opaque() == tracked_ptr_) {
      if (deallocated_) {
        LOG(FATAL) << "Buffer deallocated more than once";
      }
      deallocated_ = true;
    }
    return underlying_->Deallocate(device_ordinal, mem);
  }

  bool AllowsAsynchronousDeallocation() const override { return true; }

  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return underlying_->GetStream(device_ordinal);
  }

  void Track(void* ptr) { tracked_ptr_ = ptr; }

  bool deallocated() const { return deallocated_; }

 private:
  se::DeviceAddressAllocator* underlying_;
  void* tracked_ptr_ = nullptr;
  bool deallocated_ = false;
};

TEST_F(LocalClientStreamErrorTest, DonatedBufferCleanupOnStreamError) {
  static constexpr absl::string_view kProgram = R"(
    HloModule local_illegal_access
    ENTRY main {
      p0 = f32[4] parameter(0)
      ROOT %custom-call = f32[4] custom-call(p0),
                          custom_call_target="IllegalAccess",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  xla::XlaComputation xla_computation(hlo_module->ToProto());

  Shape argument_layout = ShapeUtil::MakeShapeWithDenseLayout(F32, {4}, {0});
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      local_client_->Compile(xla_computation, {&argument_layout},
                             ExecutableBuildOptions()));
  ASSERT_EQ(executables.size(), 1);

  AsyncDeallocationAllocator allocator(
      local_client_->backend().memory_allocator());
  TF_ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> initial_mem,
      allocator.Allocate(0, ShapeUtil::ByteSizeOf(argument_layout)));
  se::DeviceAddress<uint8_t> raw_addr = initial_mem.Release();
  allocator.Track(raw_addr.opaque());

  std::vector<ExecutionInput> arguments;
  arguments.emplace_back(argument_layout);
  se::ScopedDeviceAddress<uint8_t> donated_mem(raw_addr, 0, &allocator);
  arguments[0].SetBuffer({}, MaybeOwningDeviceAddress(std::move(donated_mem)));
  arguments[0].SetUnownedIndex({});

  ffi::ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Emplace<KernelHolder>());

  ExecutableRunOptions run_options;
  run_options.set_allocator(&allocator);
  run_options.set_ffi_execution_context(&execution_context);

  absl::StatusOr<ExecutionOutput> result =
      executables[0]->Run(std::move(arguments), run_options);

  EXPECT_THAT(result.status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("CUDA_ERROR_ILLEGAL_ADDRESS")));

  // Ensure that the donated buffer is deallocated cleanly exactly once by
  // the caller when execution encounters a stream error.
  allocator.Deallocate(0, raw_addr).IgnoreError();
}

}  // namespace
}  // namespace xla
