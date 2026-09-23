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
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::gpu {
namespace {

class RocmBlasLtTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(platform_, PlatformManager::PlatformWithName("ROCM"));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    LOG(INFO) << "Device name: " << executor_->GetDeviceDescription().name();
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    ASSERT_OK_AND_ASSIGN(blas_lt_, gpu::BlasLt::Get(executor_));
  }

  // Column-major A (m x k) and B (k x n) are ones, C is zero, alpha=1, beta=0.
  // Each output element must equal k.
  // skip_if_unsupported skips only an empty algorithm list (SKU/library gap).
  // GetMatmulPlan failures are always test failures.
  template <typename InputT, typename OutputT>
  void RunGemm(blas::ComputationType compute_type, bool skip_if_unsupported) {
    xla::PrimitiveType input_primitive_type =
        xla::primitive_util::NativeToPrimitiveType<InputT>();
    xla::PrimitiveType output_primitive_type =
        xla::primitive_util::NativeToPrimitiveType<OutputT>();

    const int64_t m = 32, n = 32, k = 32;

    gpu::MatrixLayout a_layout(input_primitive_type, m, k,
                               gpu::MatrixLayout::Order::kColumnMajor);
    gpu::MatrixLayout b_layout(input_primitive_type, k, n,
                               gpu::MatrixLayout::Order::kColumnMajor);
    gpu::MatrixLayout c_layout(output_primitive_type, m, n,
                               gpu::MatrixLayout::Order::kColumnMajor);

    gpu::GemmConfig cfg = {
        a_layout,                         // lhs_layout
        b_layout,                         // rhs_layout
        c_layout,                         // c_layout
        c_layout,                         // output_layout
        {1.0, 0.0},                       // alpha
        0.0,                              // beta
        0,                                // compute_precision
        xla::PrecisionConfig::ALG_UNSET,  // precision_algorithm
        std::nullopt,                     // algorithm
        false,                            // grad_x
        false,                            // grad_y
        gpu::ScaleMode::kNone,            // scale_mode
        compute_type                      // compute_type
    };

    ASSERT_OK_AND_ASSIGN(auto plan, blas_lt_->GetMatmulPlan(
                                        cfg, gpu::BlasLt::Epilogue::kDefault));

    const size_t workspace_size = 32 * 1024 * 1024;  // 32 MB
    ASSERT_OK_AND_ASSIGN(auto algorithms,
                         plan->GetAlgorithms(128, workspace_size));
    if (skip_if_unsupported && algorithms.empty()) {
      GTEST_SKIP() << "hipBLASLt returned no algorithms for this dtype.";
    }
    ASSERT_FALSE(algorithms.empty());
    ASSERT_OK(plan->SetAlgorithm(algorithms[0]));

    std::vector<InputT> h_a(m * k, static_cast<InputT>(1));
    std::vector<InputT> h_b(k * n, static_cast<InputT>(1));

    DeviceAddress<InputT> d_a = executor_->AllocateArray<InputT>(h_a.size());
    DeviceAddress<InputT> d_b = executor_->AllocateArray<InputT>(h_b.size());
    DeviceAddress<OutputT> d_c = executor_->AllocateArray<OutputT>(m * n);
    DeviceAddressBase workspace = executor_->Allocate(workspace_size);
    DeviceAddressHandle a_keep(executor_, d_a);
    DeviceAddressHandle b_keep(executor_, d_b);
    DeviceAddressHandle c_keep(executor_, d_c);
    DeviceAddressHandle workspace_keep(executor_, workspace);

    ASSERT_OK(stream_->Memcpy(&d_a, h_a.data(), h_a.size() * sizeof(InputT)));
    ASSERT_OK(stream_->Memcpy(&d_b, h_b.data(), h_b.size() * sizeof(InputT)));
    ASSERT_OK(stream_->MemZero(&d_c, m * n * sizeof(OutputT)));

    gpu::BlasLt::MemoryArgs args{
        /*a=*/d_a,
        /*b=*/d_b,
        /*c=*/d_c,
        /*d=*/d_c,
        /*bias=*/DeviceAddressBase{},
        /*aux=*/DeviceAddressBase{},
        /*a_scale=*/DeviceAddressBase{},
        /*b_scale=*/DeviceAddressBase{},
        /*c_scale=*/DeviceAddressBase{},
        /*d_scale=*/DeviceAddressBase{},
        /*d_amax=*/{DeviceAddressBase{}},
        workspace,
        /*scratch_allocator=*/nullptr,
    };

    ASSERT_OK(plan->ExecuteOnStream(stream_.get(), args, nullptr));

    std::vector<OutputT> h_result(m * n);
    ASSERT_OK(stream_->Memcpy(h_result.data(), d_c,
                              h_result.size() * sizeof(OutputT)));
    ASSERT_OK(stream_->BlockHostUntilDone());

    for (int64_t i = 0; i < m * n; ++i) {
      if constexpr (std::is_floating_point_v<OutputT>) {
        ASSERT_FLOAT_EQ(h_result[i], static_cast<OutputT>(k))
            << "at index " << i;
      } else {
        ASSERT_EQ(h_result[i], static_cast<OutputT>(k)) << "at index " << i;
      }
    }
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
  gpu::BlasLt* blas_lt_ = nullptr;
};

TEST_F(RocmBlasLtTest, F32F32F32Gemm) {
  RunGemm<float, float>(blas::ComputationType::kF32,
                        /*skip_if_unsupported=*/false);
}

// hipBLASLt int8 coverage is SKU/library dependent (CUDA skips S8 below sm61).
TEST_F(RocmBlasLtTest, S8S8S32Gemm) {
  RunGemm<int8_t, int32_t>(blas::ComputationType::kI32,
                           /*skip_if_unsupported=*/true);
}

TEST_F(RocmBlasLtTest, S8S8F32Gemm) {
  RunGemm<int8_t, float>(blas::ComputationType::kF32,
                         /*skip_if_unsupported=*/true);
}

}  // namespace
}  // namespace stream_executor::gpu
