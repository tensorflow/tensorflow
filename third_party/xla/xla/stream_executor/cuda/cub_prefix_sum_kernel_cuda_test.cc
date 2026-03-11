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

#include "xla/stream_executor/cuda/cub_prefix_sum_kernel_cuda.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_status.h"
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

class CubPrefixSumKernelCudaTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<
          std::tuple<xla::PrimitiveType, int, int, bool>> {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<StreamExecutorAddressAllocator>(stream_->parent());
  }

  template <typename T>
  absl::StatusOr<se::DeviceAddress<T>> CheckNotNull(
      se::DeviceAddress<T> device_memory, absl::string_view name) {
    if (device_memory.is_null()) {
      return absl::InternalError(
          absl::StrFormat("Device memory for %s is null", name));
    }
    return device_memory;
  }

  template <typename T>
  absl::Status ComputePrefixSumOnDevice(const std::vector<T>& input,
                                        std::vector<T>& output, size_t num_rows,
                                        size_t num_items, bool in_place) {
    // Setup device buffers
    TF_ASSIGN_OR_RETURN(
        se::DeviceAddress<T> device_input,
        CheckNotNull(executor_->AllocateArray<T>(input.size()), "input"));
    se::DeviceAddress<T> device_output;
    if (in_place) {
      device_output = device_input;
    } else {
      TF_ASSIGN_OR_RETURN(
          device_output,
          CheckNotNull(executor_->AllocateArray<T>(output.size()), "output"));
    }
    auto cleanup = absl::MakeCleanup([&]() {
      if (!in_place) {
        executor_->Deallocate(&device_output);
      }
      executor_->Deallocate(&device_input);
    });

    TF_RETURN_IF_ERROR(stream_->Memcpy(&device_input, input.data(),
                                       input.size() * sizeof(input[0])));
    // For large number of items, limit the number of threads per block to 512
    // to avoid running out of shared memory.
    size_t temp_bytes = 0;
    TF_RETURN_IF_ERROR(ToStatus(CubPrefixSum<T>(nullptr, temp_bytes, nullptr,
                                                nullptr, num_items, nullptr)));

    TF_ASSIGN_OR_RETURN(
        se::DeviceAddress<uint8_t> device_temp,
        CheckNotNull(executor_->AllocateArray<uint8_t>(temp_bytes), "temp"));
    auto temp_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_temp); });

    for (int i = 0; i < num_rows; ++i) {
      auto* d_in =
          reinterpret_cast<const T*>(device_input.opaque()) + i * num_items;
      auto* d_out =
          reinterpret_cast<T*>(device_output.opaque()) + i * num_items;
      TF_RETURN_IF_ERROR(ToStatus(CubPrefixSum<T>(
          device_temp.opaque(), temp_bytes, d_in, d_out, num_items,
          static_cast<CUstream>(stream_->platform_specific_handle().stream))));
    }

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    TF_RETURN_IF_ERROR(stream_->Memcpy(output.data(), device_output,
                                       output.size() * sizeof(output[0])));
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status CheckComputePrefixSumOnDevice(size_t num_rows, size_t num_items,
                                             bool in_place) {
    std::vector<T> input(num_rows * num_items);
    std::vector<T> output(input.size());
    std::vector<T> expected;
    expected.reserve(input.size());
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < num_items; ++j) {
        // We use only small values, otherwise we will get precision problems
        // with small data types.
        input[i * num_items + j] = static_cast<T>((i + j) % 5);
        expected.push_back(input[i * num_items + j]);
        if (j > 0) {
          expected.back() += expected[expected.size() - 2];
        }
      }
    }
    TF_RETURN_IF_ERROR(ComputePrefixSumOnDevice<T>(input, output, num_rows,
                                                   num_items, in_place));
    EXPECT_EQ(output, expected);
    return absl::OkStatus();
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<StreamExecutorAddressAllocator> allocator_;
};

TEST_P(CubPrefixSumKernelCudaTest, TestPrefixSum) {
  absl::Status status;
  const auto& [primitive_type, num_rows, num_items, in_place] = GetParam();
  switch (primitive_type) {
    case xla::BF16:
      if (num_items > 128) {
        GTEST_SKIP() << "Rounding errors";
      }
      status = CheckComputePrefixSumOnDevice<__nv_bfloat16>(num_rows, num_items,
                                                            in_place);
      break;
    case xla::F16:
      status =
          CheckComputePrefixSumOnDevice<__half>(num_rows, num_items, in_place);
      break;
    case xla::F32:
      status =
          CheckComputePrefixSumOnDevice<float>(num_rows, num_items, in_place);
      break;
    case xla::F64:
      status =
          CheckComputePrefixSumOnDevice<double>(num_rows, num_items, in_place);
      break;
    case xla::S8:
      status =
          CheckComputePrefixSumOnDevice<int8_t>(num_rows, num_items, in_place);
      break;
    case xla::S16:
      status =
          CheckComputePrefixSumOnDevice<int16_t>(num_rows, num_items, in_place);
      break;
    case xla::S32:
      status =
          CheckComputePrefixSumOnDevice<int32_t>(num_rows, num_items, in_place);
      break;
    case xla::S64:
      status =
          CheckComputePrefixSumOnDevice<int64_t>(num_rows, num_items, in_place);
      break;
    case xla::U8:
      status =
          CheckComputePrefixSumOnDevice<uint8_t>(num_rows, num_items, in_place);
      break;
    case xla::U16:
      status = CheckComputePrefixSumOnDevice<uint16_t>(num_rows, num_items,
                                                       in_place);
      break;
    case xla::U32:
      status = CheckComputePrefixSumOnDevice<uint32_t>(num_rows, num_items,
                                                       in_place);
      break;
    case xla::U64:
      status = CheckComputePrefixSumOnDevice<uint64_t>(num_rows, num_items,
                                                       in_place);
      break;
    default:
      status = absl::OkStatus();
  }
  TF_EXPECT_OK(status);
}

std::string ParametersToString(
    const ::testing::TestParamInfo<
        ::testing::tuple<xla::PrimitiveType, int, int, bool>>& data) {
  const auto& [primitive_type, num_rows, num_items, in_place] = data.param;
  return absl::StrFormat(
      "Prefix_Sum_%dx%d_%s%s", num_rows, num_items,
      xla::primitive_util::LowercasePrimitiveTypeName(primitive_type),
      in_place ? "_in_place" : "");
}

INSTANTIATE_TEST_SUITE_P(
    CubPrefixSumKernelCudaTestInstance, CubPrefixSumKernelCudaTest,
    ::testing::Combine(::testing::ValuesIn({xla::BF16, xla::F16, xla::F32,
                                            xla::F64, xla::S8, xla::S16,
                                            xla::S32, xla::S64, xla::U8,
                                            xla::U16, xla::U32, xla::U64}),
                       ::testing::ValuesIn({1, 2, 3, 128, 511, 512}),
                       ::testing::ValuesIn({1, 2, 3, 128, 511, 513}),
                       ::testing::ValuesIn({false, true})),
    ParametersToString);

}  // namespace
}  // namespace stream_executor::cuda
