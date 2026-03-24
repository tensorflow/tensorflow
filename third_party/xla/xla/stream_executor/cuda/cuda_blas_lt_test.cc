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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::cuda {
namespace {

class CudaBlasLtTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(platform_, PlatformManager::PlatformWithName("CUDA"));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    LOG(INFO) << "Device name: " << executor_->GetDeviceDescription().name();
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    blas_lt_ = gpu::BlasLt::Get(stream_.get());
    ASSERT_NE(blas_lt_, nullptr);
  }

  template <typename OutputT>
  void RunS8S8Gemm(blas::ComputationType compute_type) {
    xla::PrimitiveType output_primitive_type =
        xla::primitive_util::NativeToPrimitiveType<OutputT>();
    auto* cuda_cc = executor_->GetDeviceDescription()
                        .gpu_compute_capability()
                        .cuda_compute_capability();
    // S8 Gemms are not supported on Tesla P100 (compute capability 6.0).
    if (!cuda_cc->IsAtLeast(6, 1)) {
      GTEST_SKIP() << "S8 GEMM requires compute capability 6.1 or higher.";
      return;
    }

    // Matrix dimensions.
    int64_t m = 32, n = 32, k = 32;

    // A: m x k, B: k x n, C/D: m x n
    std::vector<int8_t> h_a(m * k, 1);
    std::vector<int8_t> h_b(k * n, 1);
    std::vector<OutputT> h_c(m * n, static_cast<OutputT>(0));

    DeviceAddress<int8_t> d_a = executor_->AllocateArray<int8_t>(h_a.size());
    DeviceAddress<int8_t> d_b = executor_->AllocateArray<int8_t>(h_b.size());
    DeviceAddress<OutputT> d_c = executor_->AllocateArray<OutputT>(h_c.size());

    ASSERT_OK(stream_->Memcpy(&d_a, h_a.data(), h_a.size() * sizeof(int8_t)));
    ASSERT_OK(stream_->Memcpy(&d_b, h_b.data(), h_b.size() * sizeof(int8_t)));
    ASSERT_OK(stream_->MemZero(&d_c, h_c.size() * sizeof(OutputT)));

    gpu::MatrixLayout a_layout(xla::PrimitiveType::S8, m, k,
                               gpu::MatrixLayout::Order::kColumnMajor);
    gpu::MatrixLayout b_layout(xla::PrimitiveType::S8, k, n,
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

    uint32_t workspace_size = 32 * 1024 * 1024;  // 32 MB
    ASSERT_OK_AND_ASSIGN(
        auto algorithms,
        plan->GetAlgorithms(stream_.get(), 128, workspace_size));
    ASSERT_FALSE(algorithms.empty());
    ASSERT_OK(plan->SetAlgorithm(algorithms[0]));

    DeviceAddressBase workspace = executor_->Allocate(workspace_size);

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
        /*d_amax=*/DeviceAddressBase{},
        workspace,
        /*scratch_allocator=*/nullptr,
    };

    ASSERT_OK(plan->ExecuteOnStream(stream_.get(), args, nullptr));

    std::vector<OutputT> h_result(m * n);
    ASSERT_OK(stream_->Memcpy(h_result.data(), d_c,
                              h_result.size() * sizeof(OutputT)));
    ASSERT_OK(stream_->BlockHostUntilDone());

    for (int i = 0; i < m * n; ++i) {
      ASSERT_EQ(h_result[i], static_cast<OutputT>(k)) << "at index " << i;
    }

    executor_->Deallocate(&d_a);
    executor_->Deallocate(&d_b);
    executor_->Deallocate(&d_c);
    executor_->Deallocate(&workspace);
  }

  Platform* platform_;
  StreamExecutor* executor_;
  std::unique_ptr<Stream> stream_;
  gpu::BlasLt* blas_lt_;
};

TEST_F(CudaBlasLtTest, S8S8S32Gemm) {
  RunS8S8Gemm<int32_t>(blas::ComputationType::kI32);
}

TEST_F(CudaBlasLtTest, S8S8F32Gemm) {
  RunS8S8Gemm<float>(blas::ComputationType::kF32);
}

}  // namespace
}  // namespace stream_executor::cuda
