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

#include "xla/stream_executor/cuda/caching_compilation_provider.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/mock_compilation_provider.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace stream_executor::cuda {
namespace {

using ::testing::Return;
using ::tsl::testing::IsOkAndHolds;

TEST(CachingCompilationProviderTest, CachingCompileCallsWorks) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const Assembly kAssembly{std::vector<uint8_t>{0x01, 0x02, 0x03}};

  // We expect only one call to the underlying compilation provider due to
  // caching.
  EXPECT_CALL(*mock_compilation_provider, Compile)
      .Times(1)
      .WillOnce(Return(kAssembly));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  EXPECT_THAT(caching_compilation_provider.Compile(CudaComputeCapability{10, 0},
                                                   "ptx", CompilationOptions()),
              IsOkAndHolds(kAssembly));
  EXPECT_THAT(caching_compilation_provider.Compile(CudaComputeCapability{10, 0},
                                                   "ptx", CompilationOptions()),
              IsOkAndHolds(kAssembly));
}

TEST(CachingCompilationProviderTest,
     CachingCompileToRelocatableModuleCallsWorks) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const RelocatableModule kModule{std::vector<uint8_t>{0x01, 0x02, 0x03}};

  // We expect only one call to the underlying compilation provider due to
  // caching.
  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule)
      .Times(1)
      .WillOnce(Return(kModule));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx", CompilationOptions()),
              IsOkAndHolds(kModule));
  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx", CompilationOptions()),
              IsOkAndHolds(kModule));
}

TEST(CachingCompilationProviderTest, ComputeCapabilityMattersInCompileCall) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const Assembly kAssembly1{std::vector<uint8_t>{0x01, 0x02, 0x03}};
  const Assembly kAssembly2{std::vector<uint8_t>{0x04, 0x05, 0x06}};

  // We expect two calls to the underlying compilation provider due to different
  // compute capabilities.
  EXPECT_CALL(*mock_compilation_provider, Compile)
      .Times(2)
      .WillOnce(Return(kAssembly1))
      .WillOnce(Return(kAssembly2));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  EXPECT_THAT(caching_compilation_provider.Compile(CudaComputeCapability{10, 0},
                                                   "ptx", CompilationOptions()),
              IsOkAndHolds(kAssembly1));
  EXPECT_THAT(caching_compilation_provider.Compile(CudaComputeCapability{11, 0},
                                                   "ptx", CompilationOptions()),
              IsOkAndHolds(kAssembly2));
}

TEST(CachingCompilationProviderTest,
     ComputeCapabilityMattersInCompileToRelocatableModuleCall) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const RelocatableModule kModule1{std::vector<uint8_t>{0x01, 0x02, 0x03}};
  const RelocatableModule kModule2{std::vector<uint8_t>{0x04, 0x05, 0x06}};

  // We expect two calls to the underlying compilation provider due to different
  // compute capabilities.
  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule)
      .Times(2)
      .WillOnce(Return(kModule1))
      .WillOnce(Return(kModule2));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx", CompilationOptions()),
              IsOkAndHolds(kModule1));
  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{11, 0}, "ptx", CompilationOptions()),
              IsOkAndHolds(kModule2));
}

TEST(CachingCompilationProviderTest, PtxMattersInCompileCall) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const Assembly kAssembly1{std::vector<uint8_t>{0x01, 0x02, 0x03}};
  const Assembly kAssembly2{std::vector<uint8_t>{0x04, 0x05, 0x06}};

  // We expect two calls to the underlying compilation provider due to different
  // compute capabilities.
  EXPECT_CALL(*mock_compilation_provider, Compile)
      .Times(2)
      .WillOnce(Return(kAssembly1))
      .WillOnce(Return(kAssembly2));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  EXPECT_THAT(caching_compilation_provider.Compile(
                  CudaComputeCapability{10, 0}, "ptx1", CompilationOptions()),
              IsOkAndHolds(kAssembly1));
  EXPECT_THAT(caching_compilation_provider.Compile(
                  CudaComputeCapability{10, 0}, "ptx2", CompilationOptions()),
              IsOkAndHolds(kAssembly2));
}

TEST(CachingCompilationProviderTest,
     PtxMattersInCompileToRelocatableModuleCall) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const RelocatableModule kModule1{std::vector<uint8_t>{0x01, 0x02, 0x03}};
  const RelocatableModule kModule2{std::vector<uint8_t>{0x04, 0x05, 0x06}};

  // We expect two calls to the underlying compilation provider due to different
  // compute capabilities.
  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule)
      .Times(2)
      .WillOnce(Return(kModule1))
      .WillOnce(Return(kModule2));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx1", CompilationOptions()),
              IsOkAndHolds(kModule1));
  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx2", CompilationOptions()),
              IsOkAndHolds(kModule2));
}

TEST(CachingCompilationProviderTest, CompileOptionsMatterInCompileCall) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const Assembly kAssembly1{std::vector<uint8_t>{0x01, 0x02, 0x03}};
  const Assembly kAssembly2{std::vector<uint8_t>{0x04, 0x05, 0x06}};

  // We expect two calls to the underlying compilation provider due to different
  // compute capabilities.
  EXPECT_CALL(*mock_compilation_provider, Compile)
      .Times(2)
      .WillOnce(Return(kAssembly1))
      .WillOnce(Return(kAssembly2));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  CompilationOptions options1;
  EXPECT_THAT(caching_compilation_provider.Compile(CudaComputeCapability{10, 0},
                                                   "ptx", options1),
              IsOkAndHolds(kAssembly1));

  CompilationOptions options2;
  options2.cancel_if_reg_spill = true;
  EXPECT_THAT(caching_compilation_provider.Compile(CudaComputeCapability{10, 0},
                                                   "ptx", options2),
              IsOkAndHolds(kAssembly2));
}

TEST(CachingCompilationProviderTest,
     CompileOptionsMatterInCompileToRelocatableModuleCall) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const RelocatableModule kModule1{std::vector<uint8_t>{0x01, 0x02, 0x03}};
  const RelocatableModule kModule2{std::vector<uint8_t>{0x04, 0x05, 0x06}};

  // We expect two calls to the underlying compilation provider due to different
  // compute capabilities.
  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule)
      .Times(2)
      .WillOnce(Return(kModule1))
      .WillOnce(Return(kModule2));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  CompilationOptions options1;
  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx", options1),
              IsOkAndHolds(kModule1));

  CompilationOptions options2;
  options2.cancel_if_reg_spill = true;
  EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx", options2),
              IsOkAndHolds(kModule2));
}

TEST(CachingCompilationProviderTest, CompileAndLinkCachesCompilationStep) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  // We tell the caching provider that the delegates supports
  // CompileToRelocatableModule, so it can cache the compilation step.
  EXPECT_CALL(*mock_compilation_provider, SupportsCompileToRelocatableModule)
      .WillRepeatedly(Return(true));

  const RelocatableModule kRelocatableModule{
      std::vector<uint8_t>{0x01, 0x02, 0x03}};
  const RelocatableModule kPrecompiledRelocatableModule{
      std::vector<uint8_t>{0x00, 0x05, 0x07}};
  const Assembly kAssembly{std::vector<uint8_t>{0x04, 0x05, 0x06}};

  // We expect only one call to `CompileToRelocatableModule` due to caching.
  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule)
      .Times(1)
      .WillOnce(Return(kRelocatableModule));

  EXPECT_CALL(*mock_compilation_provider, CompileAndLink)
      .Times(2)
      .WillRepeatedly(Return(kAssembly));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  EXPECT_THAT(
      caching_compilation_provider.CompileAndLink(
          CudaComputeCapability{10, 0},
          {Ptx{"ptx"}, kPrecompiledRelocatableModule}, CompilationOptions()),
      IsOkAndHolds(kAssembly));
  EXPECT_THAT(
      caching_compilation_provider.CompileAndLink(
          CudaComputeCapability{10, 0},
          {Ptx{"ptx"}, kPrecompiledRelocatableModule}, CompilationOptions()),
      IsOkAndHolds(kAssembly));
}

TEST(CachingCompilationProviderTest, ParallelCompilationWorks) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const Assembly kAssembly{std::vector<uint8_t>{0x01, 0x02, 0x03}};

  // We expect only one call to the underlying compilation provider due to
  // caching.
  EXPECT_CALL(*mock_compilation_provider, Compile)
      .Times(1)
      .WillOnce(Return(kAssembly));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  // We spawn a hundred thread and schedule parallel calls to `Compile` on them.
  // This is not guaranteed to fail if something was broken, but since we also
  // run this test with thread sanitizer enabled, this should give us a reliable
  // signal whether the locking logic is bogus or not.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 100);

  for (int i = 0; i < pool.NumThreads(); ++i) {
    pool.Schedule([&]() {
      EXPECT_THAT(
          caching_compilation_provider.Compile(CudaComputeCapability{10, 0},
                                               "ptx", CompilationOptions()),
          IsOkAndHolds(kAssembly));
    });
  }
}

TEST(CachingCompilationProviderTest,
     ParallelCompilationToRelocatableModuleWorks) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const RelocatableModule kModule{std::vector<uint8_t>{0x01, 0x02, 0x03}};

  // We expect only one call to the underlying compilation provider due to
  // caching.
  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule)
      .Times(1)
      .WillOnce(Return(kModule));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  // We spawn a hundred thread and schedule parallel calls to `Compile` on them.
  // This is not guaranteed to fail if something was broken, but since we also
  // run this test with thread sanitizer enabled, this should give us a reliable
  // signal whether the locking logic is bogus or not.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 100);

  for (int i = 0; i < pool.NumThreads(); ++i) {
    pool.Schedule([&]() {
      EXPECT_THAT(
          caching_compilation_provider.CompileToRelocatableModule(
              CudaComputeCapability{10, 0}, "ptx", CompilationOptions()),
          IsOkAndHolds(kModule));
    });
  }
}

TEST(CachingCompilationProviderTest, CompilationInterlockWorks) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const Assembly kAssembly{std::vector<uint8_t>{0x01, 0x02, 0x03}};

  absl::Mutex mutex;
  bool compilation_started = false;
  bool compilation_supposed_to_be_done = false;

  EXPECT_CALL(*mock_compilation_provider, Compile)
      .WillOnce([&]() {
        absl::MutexLock lock(&mutex);
        compilation_started = true;
        mutex.Await(absl::Condition(&compilation_supposed_to_be_done));
        return kAssembly;
      })
      .WillOnce(Return(kAssembly));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

  pool.Schedule([&]() {
    EXPECT_THAT(caching_compilation_provider.Compile(
                    CudaComputeCapability{10, 0}, "ptx", CompilationOptions()),
                IsOkAndHolds(kAssembly));
  });
  pool.Schedule([&]() {
    {
      // We wait for the other compilation to start, so that the cache is in
      // pending state.
      absl::MutexLock lock(&mutex);
      mutex.Await(absl::Condition(&compilation_started));
    }
    // This call makes sure we mutate the cache while the other compilation is
    // still running.
    EXPECT_THAT(caching_compilation_provider.Compile(
                    CudaComputeCapability{10, 0}, "ptx2", CompilationOptions()),
                IsOkAndHolds(kAssembly));
    // Then we let the other compilation finish
    absl::MutexLock lock(&mutex);
    compilation_supposed_to_be_done = true;
  });
}

TEST(CachingCompilationProviderTest,
     CompilationToRelocatableModuleInterlockWorks) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  const RelocatableModule kModule{std::vector<uint8_t>{0x01, 0x02, 0x03}};

  absl::Mutex mutex;
  bool compilation_started = false;
  bool compilation_supposed_to_be_done = false;

  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule)
      .WillOnce([&]() {
        absl::MutexLock lock(&mutex);
        compilation_started = true;
        mutex.Await(absl::Condition(&compilation_supposed_to_be_done));
        return kModule;
      })
      .WillOnce(Return(kModule));

  CachingCompilationProvider caching_compilation_provider(
      std::move(mock_compilation_provider));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

  pool.Schedule([&]() {
    EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                    CudaComputeCapability{10, 0}, "ptx", CompilationOptions()),
                IsOkAndHolds(kModule));
  });
  pool.Schedule([&]() {
    {
      // We wait for the other compilation to start, so that the cache is in
      // pending state.
      absl::MutexLock lock(&mutex);
      mutex.Await(absl::Condition(&compilation_started));
    }
    // This call makes sure we mutate the cache while the other compilation is
    // still running.
    EXPECT_THAT(caching_compilation_provider.CompileToRelocatableModule(
                    CudaComputeCapability{10, 0}, "ptx2", CompilationOptions()),
                IsOkAndHolds(kModule));
    // Then we let the other compilation finish
    absl::MutexLock lock(&mutex);
    compilation_supposed_to_be_done = true;
  });
}

}  // namespace
}  // namespace stream_executor::cuda
