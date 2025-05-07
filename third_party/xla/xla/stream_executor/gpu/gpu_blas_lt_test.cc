/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/stream_executor/gpu/gpu_blas_lt.h"

#include <optional>

#include "xla/stream_executor/blas.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::gpu {

// Helper to compare MatrixLayout structs.
void ExpectMatrixLayoutEq(const MatrixLayout& lhs, const MatrixLayout& rhs) {
  EXPECT_EQ(lhs.dtype, rhs.dtype);
  EXPECT_EQ(lhs.num_rows, rhs.num_rows);
  EXPECT_EQ(lhs.num_cols, rhs.num_cols);
  EXPECT_EQ(lhs.order, rhs.order);
  EXPECT_EQ(lhs.batch_size, rhs.batch_size);
  EXPECT_EQ(lhs.leading_dim_stride, rhs.leading_dim_stride);
  EXPECT_EQ(lhs.batch_stride, rhs.batch_stride);
  EXPECT_EQ(lhs.transpose, rhs.transpose);
}

// Helper to compare GemmConfig structs.
void ExpectGemmConfigEq(const GemmConfig& lhs, const GemmConfig& rhs) {
  ExpectMatrixLayoutEq(lhs.lhs_layout, rhs.lhs_layout);
  ExpectMatrixLayoutEq(lhs.rhs_layout, rhs.rhs_layout);
  ExpectMatrixLayoutEq(lhs.c_layout, rhs.c_layout);
  ExpectMatrixLayoutEq(lhs.output_layout, rhs.output_layout);
  EXPECT_EQ(lhs.alpha.real(), rhs.alpha.real());
  EXPECT_EQ(lhs.alpha.imag(), rhs.alpha.imag());
  EXPECT_EQ(lhs.beta, rhs.beta);
  EXPECT_EQ(lhs.compute_precision, rhs.compute_precision);
  EXPECT_EQ(lhs.precision_algorithm, rhs.precision_algorithm);
  EXPECT_EQ(lhs.algorithm, rhs.algorithm);
  EXPECT_EQ(lhs.grad_x, rhs.grad_x);
  EXPECT_EQ(lhs.grad_y, rhs.grad_y);
  EXPECT_EQ(lhs.compute_type, rhs.compute_type);
}

TEST(GemmConfigTest, ProtoConversion) {
  MatrixLayout layout(xla::PrimitiveType::F32, 16, 16,
                      MatrixLayout::Order::kRowMajor);
  GemmConfig original_config = {
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
      std::nullopt                      // compute_type
  };

  xla::GemmConfigProto proto = original_config.ToProto();
  ASSERT_OK_AND_ASSIGN(GemmConfig round_tripped_config,
                       GemmConfig::FromProto(proto));

  ExpectGemmConfigEq(original_config, round_tripped_config);
}

TEST(GemmConfigTest, ProtoConversionWithOptionals) {
  MatrixLayout layout_a(xla::PrimitiveType::BF16, 32, 64,
                        MatrixLayout::Order::kColumnMajor, 2, 64, 64 * 32 * 2,
                        blas::Transpose::kTranspose);
  MatrixLayout layout_b(xla::PrimitiveType::BF16, 64, 48,
                        MatrixLayout::Order::kRowMajor, 2, 48, 48 * 64 * 2,
                        blas::Transpose::kNoTranspose);
  MatrixLayout layout_c(xla::PrimitiveType::F32, 32, 48,
                        MatrixLayout::Order::kColumnMajor, 2, 48, 48 * 32 * 2,
                        blas::Transpose::kNoTranspose);

  GemmConfig original_config = {
      layout_a,                                   // lhs_layout
      layout_b,                                   // rhs_layout
      layout_c,                                   // c_layout
      layout_c,                                   // output_layout
      {0.5, 0.1},                                 // alpha
      1.5,                                        // beta
      1,                                          // compute_precision
      xla::PrecisionConfig::ALG_DOT_F32_F32_F32,  // precision_algorithm
      7,                                          // algorithm
      true,                                       // grad_x
      false,                                      // grad_y
      blas::ComputationType::kTF32AsF32           // compute_type
  };

  xla::GemmConfigProto proto = original_config.ToProto();
  ASSERT_OK_AND_ASSIGN(GemmConfig round_tripped_config,
                       GemmConfig::FromProto(proto));

  ExpectGemmConfigEq(original_config, round_tripped_config);
}

}  // namespace stream_executor::gpu
