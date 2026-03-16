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

#include "xla/stream_executor/cuda/cub_scan_kernel_cuda.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace se = stream_executor;

namespace stream_executor::cuda {
namespace {

class CubScanKernelCudaTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<
          xla::PrimitiveType, size_t, size_t, size_t, CubScanKind, bool>> {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<StreamExecutorAddressAllocator>(stream_->parent());
  }

 public:
  template <typename T>
  absl::Status RunCubScanTest(xla::PrimitiveType type, size_t vector_length,
                              size_t row_length, size_t col_length,
                              CubScanKind kind, bool is_reverse) {
    if (type == xla::PrimitiveType::BF16 && row_length > 128) {
      GTEST_SKIP() << "BF16 for row length > 128 has precision issues.",
          absl::OkStatus();
    }

    size_t num_elements = vector_length * row_length * col_length;
    std::vector<T> host_data, expected;
    host_data.reserve(num_elements);
    expected.reserve(num_elements);
    for (int i = 0; i < col_length; ++i) {
      T sum = {};
      for (int j = 0; j < row_length; ++j) {
        // Small values to avoid precision issues with small data types.
        T value = static_cast<T>((i + j) % 5);
        sum += value;
        host_data.push_back(value);
        expected.push_back(sum);
      }
    }

    // Get scratch size.
    TF_ASSIGN_OR_RETURN(size_t temp_bytes,
                        CubScanGetScratchSize(type, vector_length, row_length,
                                              col_length, kind, is_reverse));

    // Allocate device buffers
    se::DeviceAddress<T> device_data =
        executor_->AllocateArray<T>(num_elements);
    auto data_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_data); });
    se::DeviceAddressBase device_temp = executor_->Allocate(temp_bytes);
    auto temp_cleanup = absl::MakeCleanup([&]() {
      if (device_temp != nullptr) {
        executor_->Deallocate(&device_temp);
      }
    });

    // Copy data to device.
    size_t size_bytes = num_elements * sizeof(T);
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&device_data, host_data.data(), size_bytes));

    TF_RETURN_IF_ERROR(CubScanLaunchKernel(
        type, device_temp.opaque(), temp_bytes, device_data.opaque(),
        device_data.opaque(), vector_length, row_length, col_length, kind,
        is_reverse,
        static_cast<CUstream>(stream_->platform_specific_handle().stream)));

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(host_data.data(), device_data, size_bytes));

    if constexpr (std::is_same_v<T, float>) {
      EXPECT_THAT(host_data,
                  ::testing::Pointwise(::testing::FloatEq(), expected));
    } else if constexpr (std::is_same_v<T, double>) {
      EXPECT_THAT(host_data,
                  ::testing::Pointwise(::testing::DoubleEq(), expected));
    } else if constexpr (std::is_same_v<T, __half>) {
      for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(__half2float(host_data[i]), __half2float(expected[i]));
      }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(host_data[i]),
                        __bfloat162float(expected[i]));
      }
    } else {
      EXPECT_EQ(host_data, expected);
    }
    return absl::OkStatus();
  }

 private:
  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<StreamExecutorAddressAllocator> allocator_;
};

TEST_P(CubScanKernelCudaTest, TestPrefixSum) {
  auto impl = [&](auto value) {
    TF_EXPECT_OK(
        std::apply(&CubScanKernelCudaTest::RunCubScanTest<decltype(value)>,
                   std::tuple_cat(std::make_tuple(this), GetParam())));
  };
  switch (std::get<xla::PrimitiveType>(GetParam())) {
    case xla::PrimitiveType::BF16:
      return impl(__nv_bfloat16{});
    case xla::PrimitiveType::F16:
      return impl(__half{});
    case xla::PrimitiveType::F32:
      return impl(float{});
    case xla::PrimitiveType::F64:
      return impl(double{});
    case xla::PrimitiveType::S8:
      return impl(int8_t{});
    case xla::PrimitiveType::S16:
      return impl(int16_t());
    case xla::PrimitiveType::S32:
      return impl(int32_t{});
    case xla::PrimitiveType::S64:
      return impl(int64_t{});
    case xla::PrimitiveType::U8:
      return impl(uint8_t{});
    case xla::PrimitiveType::U16:
      return impl(uint16_t());
    case xla::PrimitiveType::U32:
      return impl(uint32_t());
    case xla::PrimitiveType::U64:
      return impl(uint64_t{});
    default:
      TF_EXPECT_OK(
          absl::InvalidArgumentError("Unsupported element type for CUB scan"));
  }
}

std::string CubScanKindToString(CubScanKind kind) {
  switch (kind) {
    case CubScanKind::kSum:
      return "Sum";
    case CubScanKind::kInvalid:
      return "Invalid";
  }
}

std::string ParametersToString(
    const ::testing::TestParamInfo<::testing::tuple<
        xla::PrimitiveType, size_t, size_t, size_t, CubScanKind, bool>>& data) {
  const auto& [type, vector_length, row_length, col_length, kind, is_reverse] =
      data.param;
  return absl::StrFormat("%s_%s_%dx%dx%d_%s", CubScanKindToString(kind),
                         is_reverse ? "reverse" : "forward", col_length,
                         row_length, vector_length,
                         xla::primitive_util::LowercasePrimitiveTypeName(type));
}

INSTANTIATE_TEST_SUITE_P(
    CubPrefixSumKernelCudaTestInstance, CubScanKernelCudaTest,
    ::testing::Combine(::testing::ValuesIn({xla::BF16, xla::F16, xla::F32,
                                            xla::F64, xla::S8, xla::S16,
                                            xla::S32, xla::S64, xla::U8,
                                            xla::U16, xla::U32, xla::U64}),
                       ::testing::Values(size_t{1}),
                       ::testing::ValuesIn<size_t>({1, 2, 3, 128, 511, 513}),
                       ::testing::ValuesIn<size_t>({1, 2, 3, 128, 511, 512}),
                       ::testing::Values(CubScanKind::kSum),
                       ::testing::Values(false)),
    ParametersToString);

}  // namespace
}  // namespace stream_executor::cuda
