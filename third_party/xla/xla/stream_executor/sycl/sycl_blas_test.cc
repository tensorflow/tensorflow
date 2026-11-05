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

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor::sycl {
namespace {

using ::testing::ElementsAreArray;

class SyclBlasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(Platform * platform,
                         PlatformManager::PlatformWithId(kSyclPlatformId));
    ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    blas_ = executor_->AsBlas();
    ASSERT_NE(blas_, nullptr);
  }

  // Scales a vector of 4 elements with stride incx and checks that only the
  // strided elements change.
  template <typename ScalarT, typename T>
  void TestScal(ScalarT alpha, T base, int incx) {
    constexpr uint64_t kElemCount = 4;
    std::vector<T> host(1 + (kElemCount - 1) * incx);
    for (size_t i = 0; i < host.size(); ++i) {
      host[i] = base * static_cast<T>(i + 1);
    }
    std::vector<T> expected = host;
    for (size_t i = 0; i < host.size(); i += incx) {
      expected[i] = alpha * host[i];
    }

    const uint64_t size_bytes = host.size() * sizeof(T);
    DeviceAddress<T> device = executor_->AllocateArray<T>(host.size());
    ASSERT_FALSE(device.is_null());
    absl::Cleanup cleanup = [&] { executor_->Deallocate(&device); };
    ASSERT_THAT(stream_->Memcpy(&device, host.data(), size_bytes),
                absl_testing::IsOk());

    EXPECT_TRUE(
        blas_->DoBlasScal(stream_.get(), kElemCount, alpha, &device, incx));

    ASSERT_THAT(stream_->Memcpy(host.data(), device, size_bytes),
                absl_testing::IsOk());
    ASSERT_THAT(stream_->BlockHostUntilDone(), absl_testing::IsOk());
    EXPECT_THAT(host, ElementsAreArray(expected));
  }

  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
  blas::BlasSupport* blas_ = nullptr;
};

TEST_F(SyclBlasTest, DoBlasScalFloat) {
  TestScal<float, float>(0.5f, 1.0f, /*incx=*/1);
  TestScal<float, float>(0.5f, 1.0f, /*incx=*/2);
}

TEST_F(SyclBlasTest, DoBlasScalDouble) {
  TestScal<double, double>(0.5, 1.0, /*incx=*/1);
  TestScal<double, double>(0.5, 1.0, /*incx=*/2);
}

TEST_F(SyclBlasTest, DoBlasScalComplexFloatByFloat) {
  using C = std::complex<float>;
  TestScal<float, C>(0.5f, C(1.0f, -2.0f), /*incx=*/1);
  TestScal<float, C>(0.5f, C(1.0f, -2.0f), /*incx=*/2);
}

TEST_F(SyclBlasTest, DoBlasScalComplexDoubleByDouble) {
  using C = std::complex<double>;
  TestScal<double, C>(0.5, C(1.0, -2.0), /*incx=*/1);
  TestScal<double, C>(0.5, C(1.0, -2.0), /*incx=*/2);
}

TEST_F(SyclBlasTest, DoBlasScalComplexFloat) {
  using C = std::complex<float>;
  TestScal<C, C>(C(0.5f, 2.0f), C(1.0f, -2.0f), /*incx=*/1);
  TestScal<C, C>(C(0.5f, 2.0f), C(1.0f, -2.0f), /*incx=*/2);
}

TEST_F(SyclBlasTest, DoBlasScalComplexDouble) {
  using C = std::complex<double>;
  TestScal<C, C>(C(0.5, 2.0), C(1.0, -2.0), /*incx=*/1);
  TestScal<C, C>(C(0.5, 2.0), C(1.0, -2.0), /*incx=*/2);
}

}  // namespace
}  // namespace stream_executor::sycl
