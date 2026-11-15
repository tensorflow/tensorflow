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

#include "xla/backends/gpu/runtime/ffi_kernel_registry.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/api/kernel_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/kernel_registry_c_api.h"
#include "xla/ffi/kernel_registry_ffi.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

constexpr XLA_FFI_LaunchDims kOneBlockOf4Threads = {
    {1, 1, 1}, {4, 1, 1}, {0, 0, 0}};

class FfiKernelRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(platform_,
                         stream_executor::PlatformManager::PlatformWithName(
                             stream_executor::GpuPlatformName()));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    ASSERT_OK_AND_ASSIGN(fatbin_, stream_executor::gpu::GetGpuTestKernelsFatbin(
                                      stream_executor::GpuPlatformName()));
    registry_ = std::make_unique<FfiKernelRegistry>(
        /*num_devices=*/platform_->VisibleDeviceCount());
    const std::vector<int32_t> a = {1, 2, 3, 4};
    const std::vector<int32_t> b = {10, 20, 30, 40};
    a_ = executor_->AllocateArray<int32_t>(4);
    b_ = executor_->AllocateArray<int32_t>(4);
    c_ = executor_->AllocateArray<int32_t>(4);
    ASSERT_OK(stream_->Memcpy(&a_, a.data(), 4 * sizeof(int32_t)));
    ASSERT_OK(stream_->Memcpy(&b_, b.data(), 4 * sizeof(int32_t)));
    ASSERT_OK(stream_->BlockHostUntilDone());
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      ASSERT_OK(stream_->BlockHostUntilDone());
    }
    if (executor_ != nullptr) {
      executor_->Deallocate(&a_);
      executor_->Deallocate(&b_);
      executor_->Deallocate(&c_);
    }
  }

  // Registers a kernel through the FFI API, as a handler would in Initialize.
  absl::StatusOr<const XLA_FFI_Kernel*> Register(absl::string_view name,
                                                 const void* data, size_t size,
                                                 ffi::SourceFormat format,
                                                 bool uses_pdl) {
    KernelRegistrationExtension extension(registry_.get(), executor_);
    return ffi::InitializeKernelRegistry(/*api=*/nullptr, &extension.raw())
        .Register(name, data, size, format, uses_pdl);
  }

  absl::StatusOr<const XLA_FFI_Kernel*> RegisterFatbinKernel(
      absl::string_view name, bool uses_pdl) {
    return Register(name, fatbin_.data(), fatbin_.size(),
                    ffi::SourceFormat::kCubin, uses_pdl);
  }

  // Launches `kernel` on four elements with args (first, b, c) through the FFI
  // API, as a handler would in Execute, and returns c.
  absl::StatusOr<std::vector<int32_t>> Run(const XLA_FFI_Kernel* kernel,
                                           ffi::KernelArg first,
                                           uint32_t shared_mem_bytes) {
    ABSL_RETURN_IF_ERROR(stream_->MemZero(&c_, 4 * sizeof(int32_t)));
    XLA_FFI_KernelRegistry_Extension extension =
        MakeKernelLaunchExtension(stream_.get());
    const std::vector<ffi::KernelArg> args = {first,
                                              ffi::DevicePointer{b_.opaque()},
                                              ffi::DevicePointer{c_.opaque()}};
    ABSL_RETURN_IF_ERROR(ffi::ExecuteKernelRegistry(/*api=*/nullptr, &extension)
                        .Launch(kernel, kOneBlockOf4Threads, shared_mem_bytes,
                                /*uses_pdl=*/false, args));
    std::vector<int32_t> c(4, 0);
    ABSL_RETURN_IF_ERROR(stream_->Memcpy(c.data(), c_, 4 * sizeof(int32_t)));
    ABSL_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    return c;
  }

  stream_executor::Platform* platform_ = nullptr;
  stream_executor::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<stream_executor::Stream> stream_;
  std::vector<uint8_t> fatbin_;
  std::unique_ptr<FfiKernelRegistry> registry_;
  stream_executor::DeviceAddress<int32_t> a_, b_, c_;
};

// The registry keeps its own copy of the binary: the caller's buffer can be
// freed right after Register, and the same binary from another buffer reuses
// the handle.
TEST_F(FfiKernelRegistryTest, RegisterCopiesAndDedupsBinary) {
  auto first_copy = std::make_unique<std::vector<uint8_t>>(fatbin_);
  ASSERT_OK_AND_ASSIGN(
      const XLA_FFI_Kernel* kernel,
      Register("AddI32", first_copy->data(), first_copy->size(),
               ffi::SourceFormat::kCubin, /*uses_pdl=*/false));
  first_copy.reset();
  EXPECT_THAT(RegisterFatbinKernel("AddI32", /*uses_pdl=*/false),
              absl_testing::IsOkAndHolds(kernel));

  EXPECT_THAT(FindDeviceKernel(kernel, executor_->device_ordinal()), IsOk());
  EXPECT_THAT(FindDeviceKernel(kernel, 999),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(Run(kernel, ffi::DevicePointer{a_.opaque()},
                  /*shared_mem_bytes=*/0),
              absl_testing::IsOkAndHolds(ElementsAre(11, 22, 33, 44)));
}

TEST_F(FfiKernelRegistryTest, RegisterAndLaunchPtxKernel) {
  if (executor_->GetDeviceDescription().gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "PTX kernels are not supported on ROCm.";
  }
  stream_executor::KernelLoaderSpec spec =
      stream_executor::gpu::GetAddI32PtxKernelSpec();
  absl::string_view ptx = spec.cuda_ptx_in_memory()->ptx;
  // Append a non-NUL byte right after the slice to verify Register NUL-
  // terminates its owned PTX copy.
  std::vector<char> un_terminated_ptx(ptx.begin(), ptx.end());
  un_terminated_ptx.push_back('X');

  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Kernel* kernel,
                       Register("AddI32", un_terminated_ptx.data(), ptx.size(),
                                ffi::SourceFormat::kPtx, /*uses_pdl=*/false));
  EXPECT_THAT(Run(kernel, ffi::DevicePointer{a_.opaque()},
                  /*shared_mem_bytes=*/0),
              absl_testing::IsOkAndHolds(ElementsAre(11, 22, 33, 44)));
}

TEST_F(FfiKernelRegistryTest, LaunchWithHostValue) {
  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Kernel* kernel,
                       RegisterFatbinKernel("IncI32", /*uses_pdl=*/false));
  const int32_t increment = 5;
  EXPECT_THAT(Run(kernel, ffi::HostValue{&increment, sizeof(increment)},
                  /*shared_mem_bytes=*/0),
              absl_testing::IsOkAndHolds(ElementsAre(15, 25, 35, 45)));
}

// DynShmemKernel reverses an n_cols x n_rows byte buffer through dynamic
// shared memory, so this fails if `shared_mem_bytes` is not forwarded.
TEST_F(FfiKernelRegistryTest, LaunchWithDynamicSharedMemory) {
  ASSERT_OK_AND_ASSIGN(
      const XLA_FFI_Kernel* kernel,
      RegisterFatbinKernel("DynShmemKernel", /*uses_pdl=*/false));
  constexpr uint32_t kCols = 4, kRows = 2;
  const std::vector<uint8_t> input = {1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_OK(stream_->Memcpy(&a_, input.data(), input.size()));
  ASSERT_OK(stream_->Memcpy(&b_, &kCols, sizeof(kCols)));
  ASSERT_OK(stream_->Memcpy(&c_, &kRows, sizeof(kRows)));

  XLA_FFI_KernelRegistry_Extension extension =
      MakeKernelLaunchExtension(stream_.get());
  const std::vector<ffi::KernelArg> args = {ffi::DevicePointer{a_.opaque()},
                                            ffi::DevicePointer{b_.opaque()},
                                            ffi::DevicePointer{c_.opaque()}};
  ASSERT_OK(ffi::ExecuteKernelRegistry(/*api=*/nullptr, &extension)
                .Launch(kernel, kOneBlockOf4Threads,
                        /*shared_mem_bytes=*/kCols * kRows,
                        /*uses_pdl=*/false, args));

  std::vector<uint8_t> output(input.size(), 0);
  ASSERT_OK(stream_->Memcpy(output.data(), a_, output.size()));
  ASSERT_OK(stream_->BlockHostUntilDone());
  EXPECT_THAT(output, ElementsAre(8, 7, 6, 5, 4, 3, 2, 1));
}

TEST_F(FfiKernelRegistryTest, RegisterRejectsInvalidSpec) {
  for (const auto& bad :
       {Register("AddI32", fatbin_.data(), 0, ffi::SourceFormat::kCubin, false),
        Register("", fatbin_.data(), fatbin_.size(), ffi::SourceFormat::kCubin,
                 false),
        Register("AddI32", fatbin_.data(), fatbin_.size(),
                 ffi::SourceFormat::kFunctionPtr, false)}) {
    EXPECT_THAT(bad, StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(FfiKernelRegistryHelpersTest, PackKernelArgsValidatesArgs) {
  int32_t value = 7;
  XLA_FFI_KernelArg null_host = {nullptr, 4, XLA_FFI_KernelArgType_HostValue};
  XLA_FFI_KernelArg sized_ptr = {&value, 4, XLA_FFI_KernelArgType_DevicePtr};
  XLA_FFI_KernelArg host = {&value, 4, XLA_FFI_KernelArgType_HostValue};
  for (XLA_FFI_KernelArgs bad :
       {XLA_FFI_KernelArgs{nullptr, -1}, XLA_FFI_KernelArgs{nullptr, 1},
        XLA_FFI_KernelArgs{&null_host, 1}, XLA_FFI_KernelArgs{&sized_ptr, 1}}) {
    EXPECT_THAT(PackKernelArgs(&bad, /*shared_mem_bytes=*/0),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  XLA_FFI_KernelArgs ok = {&host, 1};
  ASSERT_OK_AND_ASSIGN(auto packed,
                       PackKernelArgs(&ok, /*shared_mem_bytes=*/64));
  EXPECT_EQ(packed->number_of_arguments(), 2);  // The value and shared bytes.
  EXPECT_EQ(packed->number_of_shared_bytes(), 64);
}

TEST(FfiKernelRegistryHelpersTest, ConvertFfiLaunchDimsValidatesDims) {
  for (XLA_FFI_LaunchDims bad :
       {XLA_FFI_LaunchDims{{0, 1, 1}, {4, 1, 1}, {}},
        XLA_FFI_LaunchDims{{1, 1, 1}, {-4, 1, 1}, {}},
        XLA_FFI_LaunchDims{{16, 1, 1}, {4, 1, 1}, {16, 0, 0}}}) {
    EXPECT_THAT(ConvertFfiLaunchDims(bad),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  ASSERT_OK_AND_ASSIGN(
      FfiLaunchDims dims,
      ConvertFfiLaunchDims({{16, 1, 1}, {4, 1, 1}, {2, 1, 1}}));
  ASSERT_TRUE(dims.cluster.has_value());
  EXPECT_EQ(dims.cluster->x, 2);
}

// Prepare and Initialize may only register kernels; Execute may only launch
// them.
TEST_F(FfiKernelRegistryTest, ExtensionOnlyExposesCallbacksForItsStage) {
  KernelRegistrationExtension registration(registry_.get(), executor_);
  EXPECT_NE(registration.raw().api->register_kernel, nullptr);
  EXPECT_EQ(registration.raw().api->launch_kernel, nullptr);

  XLA_FFI_KernelRegistry_Extension launch =
      MakeKernelLaunchExtension(stream_.get());
  EXPECT_EQ(launch.api->register_kernel, nullptr);
  EXPECT_NE(launch.api->launch_kernel, nullptr);
}

// PDL is fixed at registration.
TEST_F(FfiKernelRegistryTest, UsesPdlIsFixedAtRegistration) {
  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Kernel* pdl_kernel,
                       RegisterFatbinKernel("AddI32", /*uses_pdl=*/true));
  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Kernel* no_pdl_kernel,
                       RegisterFatbinKernel("AddI32", /*uses_pdl=*/false));
  EXPECT_NE(pdl_kernel, no_pdl_kernel);
  const int device = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(stream_executor::Kernel * se_pdl,
                       FindDeviceKernel(pdl_kernel, device));
  ASSERT_OK_AND_ASSIGN(stream_executor::Kernel * se_no_pdl,
                       FindDeviceKernel(no_pdl_kernel, device));
  EXPECT_TRUE(se_pdl->use_pdl());
  EXPECT_FALSE(se_no_pdl->use_pdl());
}

class FakeKernel : public stream_executor::Kernel {
 public:
  unsigned Arity() const override { return 0; }
  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      stream_executor::ThreadDim, size_t) const override {
    return 1;
  }
  absl::Status Launch(const stream_executor::ThreadDim&,
                      const stream_executor::BlockDim&,
                      const std::optional<stream_executor::ClusterDim>&,
                      stream_executor::Stream*,
                      const stream_executor::KernelArgs&) override {
    return absl::OkStatus();
  }
};

TEST(FfiKernelRegistryCachedTest, PerDeviceCacheSkipsOrChecksBytesOnFallback) {
  using ::testing::Return;
  FfiKernelRegistry registry(/*num_devices=*/2);
  stream_executor::MockStreamExecutor e0, e1;
  ON_CALL(e0, device_ordinal).WillByDefault(Return(0));
  ON_CALL(e1, device_ordinal).WillByDefault(Return(1));
  EXPECT_CALL(e0, LoadKernel).Times(2).WillRepeatedly([] {
    return std::make_unique<FakeKernel>();
  });
  EXPECT_CALL(e1, LoadKernel).Times(2).WillRepeatedly([] {
    return std::make_unique<FakeKernel>();
  });

  constexpr char kBytesA[] = "cubin_bytes_a";
  constexpr char kBytesB[] = "cubin_bytes_b";
  static_assert(sizeof(kBytesA) == sizeof(kBytesB));
  KernelRegistrationExtension ext0(&registry, &e0);
  KernelRegistrationExtension ext1(&registry, &e1);
  auto reg = [](const KernelRegistrationExtension& ext, absl::string_view name,
                const char* bytes, bool uses_pdl,
                const XLA_FFI_Kernel* cached) {
    return ffi::InitializeKernelRegistry(/*api=*/nullptr, &ext.raw())
        .Register(name, bytes, sizeof(kBytesA), ffi::SourceFormat::kCubin,
                  uses_pdl, cached);
  };
  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Kernel* kernel_a,
                       reg(ext0, "k", kBytesA, /*uses_pdl=*/true, nullptr));
  EXPECT_TRUE(registry.Contains(0, kernel_a));
  EXPECT_FALSE(registry.Contains(1, kernel_a));

  // Same-device hit with matching metadata skips reading binary bytes.
  EXPECT_THAT(reg(ext0, "k", kBytesB, /*uses_pdl=*/true, kernel_a),
              absl_testing::IsOkAndHolds(kernel_a));

  // Unseen device (e1) misses Contains, checks kBytesB, and registers kernel_b.
  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Kernel* kernel_b,
                       reg(ext1, "k", kBytesB, /*uses_pdl=*/true, kernel_a));
  EXPECT_NE(kernel_b, kernel_a);

  // Concurrent per-device calls load kernel_a once on e1 and reuse thereafter.
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "workers", 2);
    for (int t = 0; t < 2; ++t) {
      pool.Schedule([&, t] {
        const auto& ext = (t == 0) ? ext0 : ext1;
        for (int i = 0; i < 16; ++i) {
          EXPECT_THAT(reg(ext, "k", kBytesA, /*uses_pdl=*/true, kernel_a),
                      absl_testing::IsOkAndHolds(kernel_a));
        }
      });
    }
  }
  ASSERT_OK_AND_ASSIGN(stream_executor::Kernel * se1,
                       FindDeviceKernel(kernel_a, 1));
  EXPECT_TRUE(se1->use_pdl());

  // Metadata mismatch on e0 (uses_pdl=false) falls back and loads a 2nd kernel.
  ASSERT_OK_AND_ASSIGN(const XLA_FFI_Kernel* no_pdl,
                       reg(ext0, "k", kBytesA, /*uses_pdl=*/false, kernel_a));
  EXPECT_NE(no_pdl, kernel_a);

  // Foreign registry handle misses Contains and falls back cleanly.
  FfiKernelRegistry other_registry(/*num_devices=*/2);
  EXPECT_CALL(e0, LoadKernel).Times(1).WillOnce([] {
    return std::make_unique<FakeKernel>();
  });
  KernelRegistrationExtension other_ext(&other_registry, &e0);
  ASSERT_OK_AND_ASSIGN(
      const XLA_FFI_Kernel* other_kernel,
      reg(other_ext, "k", kBytesA, /*uses_pdl=*/true, kernel_a));
  EXPECT_NE(other_kernel, kernel_a);

  // Concurrent distinct registrations + Contains on the same device ordinal.
  EXPECT_CALL(e0, LoadKernel).Times(16).WillRepeatedly([] {
    return std::make_unique<FakeKernel>();
  });
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "same_device", 4);
    for (int t = 0; t < 4; ++t) {
      pool.Schedule([&, t] {
        for (int i = 0; i < 4; ++i) {
          std::string kname = "concurrent_" + std::to_string(t * 4 + i);
          EXPECT_THAT(reg(ext0, kname, kBytesA, /*uses_pdl=*/true, nullptr),
                      IsOk());
          EXPECT_TRUE(registry.Contains(0, kernel_a));
        }
      });
    }
  }
}

}  // namespace
}  // namespace xla::gpu
