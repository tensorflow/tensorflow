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

#include <atomic>
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
#include "xla/ffi/api/kernel_registry_api.h"
#include "xla/ffi/api/kernel_registry_c_api.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/ffi/kernel_registry_ffi.h"
#include "xla/service/gpu/launch_dimensions.h"
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
  absl::StatusOr<ffi::RegisterResult> Register(absl::string_view name,
                                               const void* data, size_t size,
                                               ffi::SourceFormat format,
                                               bool uses_pdl) {
    KernelRegistrationExtension extension(registry_.get(), executor_,
                                          /*pdl_enabled=*/true);
    return ffi::InitializeKernelRegistry(/*api=*/nullptr, &extension.raw())
        .Register(name, data, size, format, uses_pdl,
                  ffi::RequiredFeatures::None());
  }

  absl::StatusOr<ffi::RegisterResult> RegisterFatbinKernel(
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
    ABSL_RETURN_IF_ERROR(
        ffi::ExecuteKernelRegistry(/*api=*/nullptr, &extension)
            .Launch(kernel, kOneBlockOf4Threads, shared_mem_bytes, args));
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
      ffi::RegisterResult first,
      Register("AddI32", first_copy->data(), first_copy->size(),
               ffi::SourceFormat::kCubin, /*uses_pdl=*/false));
  EXPECT_FALSE(first.already_loaded);
  first_copy.reset();
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult second,
                       RegisterFatbinKernel("AddI32", /*uses_pdl=*/false));
  EXPECT_EQ(second.kernel, first.kernel);
  EXPECT_TRUE(second.already_loaded);

  EXPECT_THAT(FindDeviceKernel(first.kernel, executor_->device_ordinal()),
              IsOk());
  EXPECT_THAT(FindDeviceKernel(first.kernel, 999),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(Run(first.kernel, ffi::DevicePointer{a_.opaque()},
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

  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult registered,
                       Register("AddI32", un_terminated_ptx.data(), ptx.size(),
                                ffi::SourceFormat::kPtx, /*uses_pdl=*/false));
  EXPECT_THAT(Run(registered.kernel, ffi::DevicePointer{a_.opaque()},
                  /*shared_mem_bytes=*/0),
              absl_testing::IsOkAndHolds(ElementsAre(11, 22, 33, 44)));
}

TEST_F(FfiKernelRegistryTest, LaunchWithHostValue) {
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult registered,
                       RegisterFatbinKernel("IncI32", /*uses_pdl=*/false));
  const int32_t increment = 5;
  EXPECT_THAT(
      Run(registered.kernel, ffi::HostValue{&increment, sizeof(increment)},
          /*shared_mem_bytes=*/0),
      absl_testing::IsOkAndHolds(ElementsAre(15, 25, 35, 45)));
}

// DynShmemKernel reverses an n_cols x n_rows byte buffer through dynamic
// shared memory, so this fails if `shared_mem_bytes` is not forwarded.
TEST_F(FfiKernelRegistryTest, LaunchWithDynamicSharedMemory) {
  ASSERT_OK_AND_ASSIGN(
      ffi::RegisterResult registered,
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
                .Launch(registered.kernel, kOneBlockOf4Threads,
                        /*shared_mem_bytes=*/kCols * kRows, args));

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
  EXPECT_THAT(ConvertFfiLaunchDims({{16, 2, 1}, {4, 3, 1}, {0, 0, 0}}),
              absl_testing::IsOkAndHolds(
                  LaunchDimensions(stream_executor::BlockDim(16, 2, 1),
                                   stream_executor::ThreadDim(4, 3, 1))));
  EXPECT_THAT(ConvertFfiLaunchDims({{16, 2, 1}, {4, 3, 1}, {2, 1, 1}}),
              absl_testing::IsOkAndHolds(
                  LaunchDimensions(stream_executor::BlockDim(16, 2, 1),
                                   stream_executor::ThreadDim(4, 3, 1),
                                   stream_executor::ClusterDim(2, 1, 1))));
}

// Prepare and Initialize may only register kernels; Execute may only launch
// them.
TEST_F(FfiKernelRegistryTest, ExtensionOnlyExposesCallbacksForItsStage) {
  KernelRegistrationExtension registration(registry_.get(), executor_,
                                           /*pdl_enabled=*/true);
  EXPECT_NE(registration.raw().api->register_kernel, nullptr);
  EXPECT_EQ(registration.raw().api->launch_kernel, nullptr);

  XLA_FFI_KernelRegistry_Extension launch =
      MakeKernelLaunchExtension(stream_.get());
  EXPECT_EQ(launch.api->register_kernel, nullptr);
  EXPECT_NE(launch.api->launch_kernel, nullptr);
}

// PDL is fixed at registration.
TEST_F(FfiKernelRegistryTest, UsesPdlIsFixedAtRegistration) {
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult pdl_res,
                       RegisterFatbinKernel("AddI32", /*uses_pdl=*/true));
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult no_pdl_res,
                       RegisterFatbinKernel("AddI32", /*uses_pdl=*/false));
  EXPECT_NE(pdl_res.kernel, no_pdl_res.kernel);
  const int device = executor_->device_ordinal();
  ASSERT_OK_AND_ASSIGN(stream_executor::Kernel * se_pdl,
                       FindDeviceKernel(pdl_res.kernel, device));
  ASSERT_OK_AND_ASSIGN(stream_executor::Kernel * se_no_pdl,
                       FindDeviceKernel(no_pdl_res.kernel, device));
  EXPECT_TRUE(se_pdl->use_pdl());
  EXPECT_FALSE(se_no_pdl->use_pdl());

  KernelRegistrationExtension pdl_disabled_ext(registry_.get(), executor_,
                                               /*pdl_enabled=*/false);
  ASSERT_OK_AND_ASSIGN(
      ffi::RegisterResult gated_res,
      ffi::InitializeKernelRegistry(/*api=*/nullptr, &pdl_disabled_ext.raw())
          .Register("AddI32", fatbin_.data(), fatbin_.size(),
                    ffi::SourceFormat::kCubin, /*uses_pdl=*/true,
                    ffi::RequiredFeatures::None(), pdl_res.kernel));
  EXPECT_EQ(gated_res.kernel, no_pdl_res.kernel);
  EXPECT_TRUE(gated_res.already_loaded);
}

TEST_F(FfiKernelRegistryTest, RequiredFeaturesMultimemMatchesDeviceSupport) {
  KernelRegistrationExtension ext(registry_.get(), executor_,
                                  /*pdl_enabled=*/true);
  auto reg = [&](const XLA_FFI_Kernel* cached) {
    return ffi::InitializeKernelRegistry(/*api=*/nullptr, &ext.raw())
        .Register("AddI32", fatbin_.data(), fatbin_.size(),
                  ffi::SourceFormat::kCubin, /*uses_pdl=*/false,
                  ffi::RequiredFeatures::Multimem(), cached);
  };
  if (executor_->is_multicast_supported()) {
    ASSERT_OK_AND_ASSIGN(ffi::RegisterResult first, reg(nullptr));
    EXPECT_FALSE(first.already_loaded);
    ASSERT_OK_AND_ASSIGN(ffi::RegisterResult second, reg(first.kernel));
    EXPECT_EQ(second.kernel, first.kernel);
    EXPECT_TRUE(second.already_loaded);
  } else {
    EXPECT_THAT(reg(nullptr), StatusIs(absl::StatusCode::kFailedPrecondition));
  }
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
  KernelRegistrationExtension ext0(&registry, &e0, /*pdl_enabled=*/true);
  KernelRegistrationExtension ext1(&registry, &e1, /*pdl_enabled=*/true);
  auto reg = [](const KernelRegistrationExtension& ext, absl::string_view name,
                const char* bytes, bool uses_pdl,
                const XLA_FFI_Kernel* cached) {
    return ffi::InitializeKernelRegistry(/*api=*/nullptr, &ext.raw())
        .Register(name, bytes, sizeof(kBytesA), ffi::SourceFormat::kCubin,
                  uses_pdl, ffi::RequiredFeatures::None(), cached);
  };
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult res_a,
                       reg(ext0, "k", kBytesA, /*uses_pdl=*/true, nullptr));
  const XLA_FFI_Kernel* kernel_a = res_a.kernel;
  EXPECT_FALSE(res_a.already_loaded);
  EXPECT_TRUE(registry.Contains(0, kernel_a));
  EXPECT_FALSE(registry.Contains(1, kernel_a));

  // Same-device hit with matching metadata skips reading binary bytes.
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult fast_hit,
                       reg(ext0, "k", kBytesB, /*uses_pdl=*/true, kernel_a));
  EXPECT_EQ(fast_hit.kernel, kernel_a);
  EXPECT_TRUE(fast_hit.already_loaded);

  // Unseen device (e1) misses Contains, checks kBytesB, and registers kernel_b.
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult res_b,
                       reg(ext1, "k", kBytesB, /*uses_pdl=*/true, kernel_a));
  EXPECT_NE(res_b.kernel, kernel_a);
  EXPECT_FALSE(res_b.already_loaded);

  // Concurrent calls racing on the same unloaded device (e1) report
  // !already_loaded to exactly one caller, while pre-loaded e0 reports 0.
  std::atomic<int> e0_loads{0};
  std::atomic<int> e1_loads{0};
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "workers", 4);
    for (int t = 0; t < 4; ++t) {
      pool.Schedule([&, t] {
        const auto& ext = (t == 0) ? ext0 : ext1;
        auto& loads = (t == 0) ? e0_loads : e1_loads;
        for (int i = 0; i < 16; ++i) {
          ASSERT_OK_AND_ASSIGN(
              ffi::RegisterResult r,
              reg(ext, "k", kBytesA, /*uses_pdl=*/true, kernel_a));
          EXPECT_EQ(r.kernel, kernel_a);
          if (!r.already_loaded) {
            loads.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }
  }
  EXPECT_EQ(e0_loads.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(e1_loads.load(std::memory_order_relaxed), 1);
  ASSERT_OK_AND_ASSIGN(stream_executor::Kernel * se1,
                       FindDeviceKernel(kernel_a, 1));
  EXPECT_TRUE(se1->use_pdl());

  // Metadata mismatch on e0 (uses_pdl=false) falls back and loads a 2nd kernel.
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult no_pdl,
                       reg(ext0, "k", kBytesA, /*uses_pdl=*/false, kernel_a));
  EXPECT_NE(no_pdl.kernel, kernel_a);
  EXPECT_FALSE(no_pdl.already_loaded);

  // Foreign registry handle misses Contains and falls back cleanly.
  FfiKernelRegistry other_registry(/*num_devices=*/2);
  EXPECT_CALL(e0, LoadKernel).Times(1).WillOnce([] {
    return std::make_unique<FakeKernel>();
  });
  KernelRegistrationExtension other_ext(&other_registry, &e0,
                                        /*pdl_enabled=*/true);
  ASSERT_OK_AND_ASSIGN(
      ffi::RegisterResult other_kernel,
      reg(other_ext, "k", kBytesA, /*uses_pdl=*/true, kernel_a));
  EXPECT_NE(other_kernel.kernel, kernel_a);
  EXPECT_FALSE(other_kernel.already_loaded);

  // Missing or undersized result out-struct is rejected.
  XLA_FFI_KernelSpec spec = {XLA_FFI_KernelSpec_STRUCT_SIZE,
                             nullptr,
                             {"k", 1},
                             kBytesA,
                             sizeof(kBytesA),
                             XLA_FFI_SourceFormat_CUBIN};
  XLA_FFI_KernelRegistry_Register_Result undersized_result{};
  for (XLA_FFI_KernelRegistry_Register_Result* bad_result :
       {static_cast<XLA_FFI_KernelRegistry_Register_Result*>(nullptr),
        &undersized_result}) {
    XLA_FFI_KernelRegistry_Register_Args args{};
    args.struct_size = XLA_FFI_KernelRegistry_Register_Args_STRUCT_SIZE;
    args.state = ext0.raw().state;
    args.kernel_spec = &spec;
    args.result = bad_result;
    EXPECT_THAT(ffi::TakeError(ext0.raw().api->register_kernel(&args)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

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

TEST(FfiKernelRegistryCachedTest, RequiredFeaturesValidationRunsBeforeCache) {
  using ::testing::Return;
  FfiKernelRegistry registry(/*num_devices=*/2);
  stream_executor::MockStreamExecutor e0, e1;
  ON_CALL(e0, device_ordinal).WillByDefault(Return(0));
  ON_CALL(e0, is_multicast_supported).WillByDefault(Return(false));
  EXPECT_CALL(e0, LoadKernel).Times(1).WillOnce([] {
    return std::make_unique<FakeKernel>();
  });
  ON_CALL(e1, device_ordinal).WillByDefault(Return(1));
  ON_CALL(e1, is_multicast_supported).WillByDefault(Return(true));
  EXPECT_CALL(e1, LoadKernel).Times(1).WillOnce([] {
    return std::make_unique<FakeKernel>();
  });

  constexpr char kBytes[] = "cubin_bytes";
  KernelRegistrationExtension ext0(&registry, &e0, /*pdl_enabled=*/true);
  KernelRegistrationExtension ext1(&registry, &e1, /*pdl_enabled=*/true);
  auto reg = [&](const KernelRegistrationExtension& ext,
                 const XLA_FFI_Kernel* cached, ffi::RequiredFeatures features) {
    return ffi::InitializeKernelRegistry(/*api=*/nullptr, &ext.raw())
        .Register("k", kBytes, sizeof(kBytes), ffi::SourceFormat::kCubin,
                  /*uses_pdl=*/false, features, cached);
  };

  constexpr ffi::RequiredFeatures kUnknown{uint32_t{1} << 31};
  // Cold-path feature failures do not load the kernel.
  EXPECT_THAT(reg(ext0, nullptr, ffi::RequiredFeatures::Multimem()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(registry.Register(&e0, "k", kBytes, sizeof(kBytes),
                                XLA_FFI_SourceFormat_CUBIN, /*use_pdl=*/false,
                                XLA_FFI_KERNEL_FEATURE_MULTIMEM),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(reg(ext0, nullptr, ffi::RequiredFeatures::Multimem() | kUnknown),
              StatusIs(absl::StatusCode::kUnimplemented));
  // Malformed specs fail with INVALID_ARGUMENT even on unsupported devices.
  EXPECT_THAT(registry.Register(&e0, "", kBytes, sizeof(kBytes),
                                XLA_FFI_SourceFormat_CUBIN, /*use_pdl=*/false,
                                XLA_FFI_KERNEL_FEATURE_MULTIMEM),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Registering without required features loads the kernel once on e0.
  ASSERT_OK_AND_ASSIGN(ffi::RegisterResult loaded,
                       reg(ext0, nullptr, ffi::RequiredFeatures::None()));
  EXPECT_FALSE(loaded.already_loaded);

  // Fast-path calls with existing_kernel still validate required features.
  EXPECT_THAT(reg(ext0, loaded.kernel, ffi::RequiredFeatures::Multimem()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(reg(ext0, loaded.kernel, kUnknown),
              StatusIs(absl::StatusCode::kUnimplemented));

  // Supported device (e1) succeeds on cold, fast, and direct paths.
  ASSERT_OK_AND_ASSIGN(
      ffi::RegisterResult e1_loaded,
      reg(ext1, loaded.kernel, ffi::RequiredFeatures::Multimem()));
  EXPECT_EQ(e1_loaded.kernel, loaded.kernel);
  EXPECT_FALSE(e1_loaded.already_loaded);
  ASSERT_OK_AND_ASSIGN(
      ffi::RegisterResult e1_fast,
      reg(ext1, loaded.kernel, ffi::RequiredFeatures::Multimem()));
  EXPECT_EQ(e1_fast.kernel, loaded.kernel);
  EXPECT_TRUE(e1_fast.already_loaded);
  ASSERT_OK_AND_ASSIGN(
      FfiKernelRegistry::RegisterResult e1_direct,
      registry.Register(&e1, "k", kBytes, sizeof(kBytes),
                        XLA_FFI_SourceFormat_CUBIN, /*use_pdl=*/false,
                        XLA_FFI_KERNEL_FEATURE_MULTIMEM));
  EXPECT_EQ(e1_direct.kernel, loaded.kernel);
  EXPECT_TRUE(e1_direct.already_loaded);
}

}  // namespace
}  // namespace xla::gpu
