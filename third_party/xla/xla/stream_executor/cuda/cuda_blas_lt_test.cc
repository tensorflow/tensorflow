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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.h"
#include "xla/stream_executor/cuda/cublas_lt_matmul_algorithm.pb.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.h"
#include "xla/stream_executor/gpu/blas_lt_matmul_algorithm.pb.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::cuda {
namespace {

using ::absl_testing::StatusIs;
using ::tsl::proto_testing::EqualsProto;

class CudaBlasLtTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(platform_, PlatformManager::PlatformWithName("CUDA"));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    LOG(INFO) << "Device name: " << executor_->GetDeviceDescription().name();
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    ASSERT_OK_AND_ASSIGN(blas_lt_, gpu::BlasLt::Get(executor_));
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
    ASSERT_OK_AND_ASSIGN(auto algorithms,
                         plan->GetAlgorithms(128, workspace_size));
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
        /*d_amax=*/{DeviceAddressBase{}},
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

TEST_F(CudaBlasLtTest, AlgorithmProtoRoundTripAndValidation) {
  int64_t m = 32, k = 32;
  gpu::MatrixLayout layout(xla::PrimitiveType::F32, m, k,
                           gpu::MatrixLayout::Order::kColumnMajor);
  gpu::GemmConfig cfg = {
      layout,                           // lhs_layout
      layout,                           // rhs_layout
      layout,                           // c_layout
      layout,                           // output_layout
      {1.0, 0.0},                       // alpha
      0.0,                              // beta
      0,                                // compute_precision
      xla::PrecisionConfig::ALG_UNSET,  // precision_algorithm
      std::nullopt,                     // algorithm
      false,                            // grad_x
      false,                            // grad_y
      gpu::ScaleMode::kNone,            // scale_mode
      blas::ComputationType::kF32       // compute_type
  };

  ASSERT_OK_AND_ASSIGN(
      gpu::BlasLt::MatmulPlanPtr plan,
      blas_lt_->GetMatmulPlan(cfg, gpu::BlasLt::Epilogue::kDefault));

  uint32_t workspace_size = 32 * 1024 * 1024;
  ASSERT_OK_AND_ASSIGN(std::vector<gpu::BlasLt::MatmulAlgorithm> algorithms,
                       plan->GetAlgorithms(16, workspace_size));
  ASSERT_FALSE(algorithms.empty());
  ASSERT_OK_AND_ASSIGN(gpu::BlasLtMatmulAlgorithm blas_lt_algorithm,
                       plan->GetBlasLtMatmulAlgorithm(algorithms[0]));
  EXPECT_GT(blas_lt_algorithm.library_version, 0);

  gpu::BlasLtMatmulAlgorithmProto algorithm_proto = blas_lt_algorithm.ToProto();
  EXPECT_TRUE(algorithm_proto.has_cublaslt());
  ASSERT_OK_AND_ASSIGN(gpu::BlasLtMatmulAlgorithm from_proto,
                       gpu::BlasLtMatmulAlgorithm::FromProto(algorithm_proto));
  EXPECT_EQ(from_proto, blas_lt_algorithm);

  ASSERT_OK_AND_ASSIGN(gpu::BlasLt::MatmulAlgorithm reconstructed,
                       plan->GetMatmulAlgorithm(from_proto, workspace_size));
  ASSERT_OK_AND_ASSIGN(gpu::BlasLtMatmulAlgorithm reconstructed_algorithm,
                       plan->GetBlasLtMatmulAlgorithm(reconstructed));
  EXPECT_EQ(reconstructed_algorithm, blas_lt_algorithm);
  EXPECT_THAT(reconstructed_algorithm.ToProto(), EqualsProto(algorithm_proto));
  EXPECT_EQ(reconstructed.workspace_size, algorithms[0].workspace_size);
  EXPECT_OK(plan->SetAlgorithm(reconstructed));
  EXPECT_OK(plan->SetCachedAlgorithm(blas_lt_algorithm, workspace_size));

  // Missing cublaslt field in proto must return InvalidArgument.
  gpu::BlasLtMatmulAlgorithmProto empty_proto;
  EXPECT_THAT(gpu::BlasLtMatmulAlgorithm::FromProto(empty_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Negative unsigned-attribute fields must return InvalidArgument.
  gpu::BlasLtMatmulAlgorithmProto negative_tile_proto = algorithm_proto;
  negative_tile_proto.mutable_cublaslt()->set_tile_id(-1);
  EXPECT_THAT(gpu::BlasLtMatmulAlgorithm::FromProto(negative_tile_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Shape IDs exceeding uint16_t range must return InvalidArgument.
  gpu::BlasLtMatmulAlgorithmProto overflow_shape_proto = algorithm_proto;
  overflow_shape_proto.mutable_cublaslt()->set_inner_shape_id(UINT16_MAX + 1);
  EXPECT_THAT(gpu::BlasLtMatmulAlgorithm::FromProto(overflow_shape_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid algorithm ID must fail cublasLtMatmulAlgoInit.
  gpu::BlasLtMatmulAlgorithm bad_id_algorithm = blas_lt_algorithm;
  std::get<CublasLtMatmulAlgorithm>(bad_id_algorithm.algorithm).algo_id =
      -999999;
  EXPECT_THAT(plan->GetMatmulAlgorithm(bad_id_algorithm, workspace_size),
              StatusIs(absl::StatusCode::kInternal));

  // Invalid tile enum must fail cublasLtMatmulAlgoCheck.
  gpu::BlasLtMatmulAlgorithm bad_tile_algorithm = blas_lt_algorithm;
  std::get<CublasLtMatmulAlgorithm>(bad_tile_algorithm.algorithm).tile_id =
      999999;
  EXPECT_THAT(plan->GetMatmulAlgorithm(bad_tile_algorithm, workspace_size),
              StatusIs(absl::StatusCode::kInternal));

  // Mismatched numerical_impl_flags must return FailedPrecondition.
  gpu::BlasLtMatmulAlgorithm bad_flags_algorithm = blas_lt_algorithm;
  std::get<CublasLtMatmulAlgorithm>(bad_flags_algorithm.algorithm)
      .numerical_impl_flags ^= 1;
  EXPECT_THAT(plan->GetMatmulAlgorithm(bad_flags_algorithm, workspace_size),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace stream_executor::cuda
