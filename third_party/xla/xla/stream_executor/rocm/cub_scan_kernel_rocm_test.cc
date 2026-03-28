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

#include "xla/stream_executor/rocm/cub_scan_kernel_rocm.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/amd_detail/amd_hip_bfloat16.h"
#include "rocm/include/hip/amd_detail/amd_hip_fp16.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace se = stream_executor;

namespace stream_executor::rocm {
namespace {

// Host-side bf16 conversion helpers. hip_bfloat16 operators require hipcc,
// but this test is compiled with the host C++ compiler.
float Bf16ToFloat(hip_bfloat16 v) {
  uint16_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
  float result;
  std::memcpy(&result, &f32_bits, sizeof(result));
  return result;
}

hip_bfloat16 FloatToBf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  // Round to nearest even (same algorithm as hip_bfloat16's float_to_bfloat16).
  uint16_t lsb = (bits >> 16) & 1;
  bits += 0x7FFF + lsb;
  uint16_t bf16_bits = static_cast<uint16_t>(bits >> 16);
  hip_bfloat16 result;
  std::memcpy(&result, &bf16_bits, sizeof(result));
  return result;
}

class CubScanKernelRocmTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<
          xla::PrimitiveType, size_t, size_t, size_t, CubScanKind, bool>> {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("ROCM"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
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
    for (size_t i = 0; i < col_length; ++i) {
      T sum = {};
      for (size_t j = 0; j < row_length; ++j) {
        T value = static_cast<T>((i + j) % 5);
        sum += value;
        host_data.push_back(value);
        expected.push_back(sum);
      }
    }

    TF_ASSIGN_OR_RETURN(size_t temp_bytes,
                        CubScanGetScratchSize(type, vector_length, row_length,
                                              col_length, kind, is_reverse));

    se::DeviceAddress<T> device_data =
        executor_->AllocateArray<T>(num_elements);
    auto data_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_data); });
    se::DeviceAddress<uint8_t> device_temp =
        executor_->AllocateArray<uint8_t>(temp_bytes);
    auto temp_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_temp); });

    size_t size_bytes = num_elements * sizeof(T);
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&device_data, host_data.data(), size_bytes));

    TF_RETURN_IF_ERROR(CubScanLaunchKernel(
        type, device_temp.opaque(), temp_bytes, device_data.opaque(),
        device_data.opaque(), vector_length, row_length, col_length, kind,
        is_reverse,
        static_cast<hipStream_t>(stream_->platform_specific_handle().stream)));

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(host_data.data(), device_data, size_bytes));

    if constexpr (std::is_same_v<T, float>) {
      EXPECT_THAT(host_data,
                  ::testing::Pointwise(::testing::FloatEq(), expected));
    } else if constexpr (std::is_same_v<T, double>) {
      EXPECT_THAT(host_data,
                  ::testing::Pointwise(::testing::DoubleEq(), expected));
    } else {
      EXPECT_EQ(host_data, expected);
    }
    return absl::OkStatus();
  }

  // bf16 specialization: host-side conversion via memcpy.
  absl::Status RunCubScanTestBf16(xla::PrimitiveType type, size_t vector_length,
                                  size_t row_length, size_t col_length,
                                  CubScanKind kind, bool is_reverse) {
    if (row_length > 128) {
      GTEST_SKIP() << "BF16 for row length > 128 has precision issues.",
          absl::OkStatus();
    }

    size_t num_elements = vector_length * row_length * col_length;
    std::vector<hip_bfloat16> host_data;
    std::vector<float> expected_f;
    host_data.reserve(num_elements);
    expected_f.reserve(num_elements);
    for (size_t i = 0; i < col_length; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < row_length; ++j) {
        float value = static_cast<float>((i + j) % 5);
        hip_bfloat16 bf_value = FloatToBf16(value);
        float bf_value_f = Bf16ToFloat(bf_value);
        sum += bf_value_f;
        sum = Bf16ToFloat(FloatToBf16(sum));
        host_data.push_back(bf_value);
        expected_f.push_back(sum);
      }
    }

    TF_ASSIGN_OR_RETURN(size_t temp_bytes,
                        CubScanGetScratchSize(type, vector_length, row_length,
                                              col_length, kind, is_reverse));

    se::DeviceAddress<hip_bfloat16> device_data =
        executor_->AllocateArray<hip_bfloat16>(num_elements);
    auto data_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_data); });
    se::DeviceAddress<uint8_t> device_temp =
        executor_->AllocateArray<uint8_t>(temp_bytes);
    auto temp_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_temp); });

    size_t size_bytes = num_elements * sizeof(hip_bfloat16);
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&device_data, host_data.data(), size_bytes));

    TF_RETURN_IF_ERROR(CubScanLaunchKernel(
        type, device_temp.opaque(), temp_bytes, device_data.opaque(),
        device_data.opaque(), vector_length, row_length, col_length, kind,
        is_reverse,
        static_cast<hipStream_t>(stream_->platform_specific_handle().stream)));

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(host_data.data(), device_data, size_bytes));

    for (size_t i = 0; i < num_elements; ++i) {
      EXPECT_FLOAT_EQ(Bf16ToFloat(host_data[i]), expected_f[i]);
    }
    return absl::OkStatus();
  }

  // f16 specialization: host-side conversion via __half2float / __float2half.
  absl::Status RunCubScanTestF16(xla::PrimitiveType type, size_t vector_length,
                                 size_t row_length, size_t col_length,
                                 CubScanKind kind, bool is_reverse) {
    size_t num_elements = vector_length * row_length * col_length;
    std::vector<__half> host_data;
    std::vector<float> expected_f;
    host_data.reserve(num_elements);
    expected_f.reserve(num_elements);
    for (size_t i = 0; i < col_length; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < row_length; ++j) {
        float value = static_cast<float>((i + j) % 5);
        __half h_value = __float2half(value);
        float h_value_f = __half2float(h_value);
        sum += h_value_f;
        sum = __half2float(__float2half(sum));
        host_data.push_back(h_value);
        expected_f.push_back(sum);
      }
    }

    TF_ASSIGN_OR_RETURN(size_t temp_bytes,
                        CubScanGetScratchSize(type, vector_length, row_length,
                                              col_length, kind, is_reverse));

    se::DeviceAddress<__half> device_data =
        executor_->AllocateArray<__half>(num_elements);
    auto data_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_data); });
    se::DeviceAddress<uint8_t> device_temp =
        executor_->AllocateArray<uint8_t>(temp_bytes);
    auto temp_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_temp); });

    size_t size_bytes = num_elements * sizeof(__half);
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&device_data, host_data.data(), size_bytes));

    TF_RETURN_IF_ERROR(CubScanLaunchKernel(
        type, device_temp.opaque(), temp_bytes, device_data.opaque(),
        device_data.opaque(), vector_length, row_length, col_length, kind,
        is_reverse,
        static_cast<hipStream_t>(stream_->platform_specific_handle().stream)));

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(host_data.data(), device_data, size_bytes));

    for (size_t i = 0; i < num_elements; ++i) {
      EXPECT_FLOAT_EQ(__half2float(host_data[i]), expected_f[i]);
    }
    return absl::OkStatus();
  }

 private:
  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
};

TEST_P(CubScanKernelRocmTest, TestPrefixSum) {
  const auto& params = GetParam();
  auto type = std::get<xla::PrimitiveType>(params);
  switch (type) {
    case xla::PrimitiveType::BF16:
      TF_EXPECT_OK(std::apply(&CubScanKernelRocmTest::RunCubScanTestBf16,
                              std::tuple_cat(std::make_tuple(this), params)));
      return;
    case xla::PrimitiveType::F16:
      TF_EXPECT_OK(std::apply(&CubScanKernelRocmTest::RunCubScanTestF16,
                              std::tuple_cat(std::make_tuple(this), params)));
      return;
    default:
      break;
  }
  auto impl = [&](auto value) {
    TF_EXPECT_OK(
        std::apply(&CubScanKernelRocmTest::RunCubScanTest<decltype(value)>,
                   std::tuple_cat(std::make_tuple(this), params)));
  };
  switch (type) {
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
      FAIL() << "Unsupported element type for CUB scan";
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
    CubScanKernelRocmTestInstance, CubScanKernelRocmTest,
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
}  // namespace stream_executor::rocm
