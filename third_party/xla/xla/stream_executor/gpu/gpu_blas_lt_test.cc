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
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/descriptor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::gpu {
using absl_testing::StatusIs;
using ::testing::ValuesIn;

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
  EXPECT_EQ(lhs.scale_mode, rhs.scale_mode);
  EXPECT_EQ(lhs.compute_type, rhs.compute_type);
}

// Helper to compare GemmConfig structs.
void ExpectGroupedGemmConfigEq(const GroupedGemmConfig& lhs,
                               const GroupedGemmConfig& rhs) {
  EXPECT_EQ(lhs.m, rhs.m);
  EXPECT_EQ(lhs.n, rhs.n);
  EXPECT_EQ(lhs.k, rhs.k);
  EXPECT_EQ(lhs.batch_count, rhs.batch_count);
  EXPECT_EQ(lhs.group_count, rhs.group_count);
  EXPECT_EQ(lhs.lhs_leading_dim_stride, rhs.lhs_leading_dim_stride);
  EXPECT_EQ(lhs.rhs_leading_dim_stride, rhs.rhs_leading_dim_stride);
  EXPECT_EQ(lhs.c_leading_dim_stride, rhs.c_leading_dim_stride);
  EXPECT_EQ(lhs.output_leading_dim_stride, rhs.output_leading_dim_stride);
  EXPECT_EQ(lhs.trans_a, rhs.trans_a);
  EXPECT_EQ(lhs.trans_b, rhs.trans_b);
  EXPECT_EQ(lhs.must_swap_operands, rhs.must_swap_operands);
  EXPECT_EQ(lhs.alpha.real(), rhs.alpha.real());
  EXPECT_EQ(lhs.alpha.imag(), rhs.alpha.imag());
  EXPECT_EQ(lhs.beta, rhs.beta);
  EXPECT_EQ(lhs.type_a, rhs.type_a);
  EXPECT_EQ(lhs.type_b, rhs.type_b);
  EXPECT_EQ(lhs.type_c, rhs.type_c);
  EXPECT_EQ(lhs.type_d, rhs.type_d);
  EXPECT_EQ(lhs.stride_ragged_dim, rhs.stride_ragged_dim);
  EXPECT_EQ(lhs.stride_group_dim, rhs.stride_group_dim);
  EXPECT_EQ(lhs.c_stride_ragged_dim, rhs.c_stride_ragged_dim);
  EXPECT_EQ(lhs.output_stride_ragged_dim, rhs.output_stride_ragged_dim);
  EXPECT_EQ(lhs.precision_algorithm, rhs.precision_algorithm);
  EXPECT_EQ(lhs.compute_precision, rhs.compute_precision);
  EXPECT_EQ(lhs.ragged_mode, rhs.ragged_mode);
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
      ScaleMode::kNone,                 // scale_mode
      std::nullopt                      // compute_type
  };

  xla::GemmConfigProto proto = original_config.ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto round_tripped_config,
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
      ScaleMode::kNone,                           // scale_mode
      blas::ComputationType::kTF32AsF32           // compute_type
  };

  xla::GemmConfigProto proto = original_config.ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto round_tripped_config,
                          GemmConfig::FromProto(proto));

  ExpectGemmConfigEq(original_config, round_tripped_config);
}

TEST(GroupedGemmConfigTest, ProtoConversionWithOptionals) {
  GroupedGemmConfig original_config = {
      32,                                         // m
      48,                                         // n
      64,                                         // k
      1,                                          // batch count
      4,                                          // group count
      128,                                        // lhs_leading_dim_stride
      256,                                        // rhs_leading_dim_stride
      128,                                        // c_leading_dim_stride
      128,                                        // output_leading_dim_stride
      blas::Transpose::kTranspose,                // trans_a
      blas::Transpose::kNoTranspose,              // trans_b
      true,                                       // must_swap_operands
      {0.5, 0.1},                                 // alpha
      1.5,                                        // beta
      blas::DataType::kBF16,                      // type_a
      blas::DataType::kBF16,                      // type_b
      blas::DataType::kFloat,                     // type_c
      blas::DataType::kFloat,                     // type_d
      64,                                         // stride_ragged_dim
      64 * 48,                                    // stride_group_dim
      48,                                         // c_stride_ragged_dim
      48,                                         // output_stride_ragged_dim
      xla::PrecisionConfig::ALG_DOT_F32_F32_F32,  // precision_algorithm
      1,                                          // compute_precision
      RaggedDotMode::kRaggedNonContracting,       // ragged_mode
      blas::ComputationType::kTF32AsF32           // compute_type
  };

  xla::GroupedGemmConfigProto proto = original_config.ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto round_tripped_config,
                          GroupedGemmConfig::FromProto(proto));

  ExpectGroupedGemmConfigEq(original_config, round_tripped_config);
}

TEST(EpilogueTest, ToProtoSucceedsForValidValues) {
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kDefault),
            xla::BlasLtEpilogueProto::EPILOGUE_DEFAULT);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kReLU),
            xla::BlasLtEpilogueProto::EPILOGUE_RELU);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kBias),
            xla::BlasLtEpilogueProto::EPILOGUE_BIAS);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kBiasThenReLU),
            xla::BlasLtEpilogueProto::EPILOGUE_BIAS_THEN_RELU);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kGELU),
            xla::BlasLtEpilogueProto::EPILOGUE_GELU);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kSILU),
            xla::BlasLtEpilogueProto::EPILOGUE_SILU);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kSILUWithAux),
            xla::BlasLtEpilogueProto::EPILOGUE_SILU_WITH_AUX);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kGELUWithAux),
            xla::BlasLtEpilogueProto::EPILOGUE_GELU_WITH_AUX);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kBiasThenGELU),
            xla::BlasLtEpilogueProto::EPILOGUE_BIAS_THEN_GELU);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kBiasThenSILU),
            xla::BlasLtEpilogueProto::EPILOGUE_BIAS_THEN_SILU);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kBiasThenGELUWithAux),
            xla::BlasLtEpilogueProto::EPILOGUE_BIAS_THEN_GELU_WITH_AUX);
  EXPECT_EQ(BlasLt::EpilogueToProto(BlasLt::Epilogue::kBiasThenSILUWithAux),
            xla::BlasLtEpilogueProto::EPILOGUE_BIAS_THEN_SILU_WITH_AUX);
}

using EpilogueFromProtoTest =
    ::testing::TestWithParam<xla::BlasLtEpilogueProto>;

TEST_P(EpilogueFromProtoTest, SucceedsForValidValue) {
  TF_EXPECT_OK(BlasLt::EpilogueFromProto(GetParam()));
}

std::vector<xla::BlasLtEpilogueProto> EnumerateBlasLtEpilogueProtoValues() {
  const google::protobuf::EnumDescriptor* descriptor =
      xla::BlasLtEpilogueProto_descriptor();
  std::vector<xla::BlasLtEpilogueProto> values;
  values.reserve(descriptor->value_count());
  for (int i = 0; i < descriptor->value_count(); ++i) {
    values.push_back(
        static_cast<xla::BlasLtEpilogueProto>(descriptor->value(i)->number()));
  }
  return values;
}

std::string ToString(const xla::BlasLtEpilogueProto& proto) {
  const google::protobuf::EnumDescriptor* descriptor =
      xla::BlasLtEpilogueProto_descriptor();
  const google::protobuf::EnumValueDescriptor* value =
      descriptor->FindValueByNumber(proto);
  if (value == nullptr) {
    return absl::StrCat("Unknown(", proto, ")");
  }
  return std::string(value->name());
}

INSTANTIATE_TEST_SUITE_P(
    EpilogueFromProtoTests, EpilogueFromProtoTest,
    ValuesIn(EnumerateBlasLtEpilogueProtoValues()),
    [](const testing::TestParamInfo<xla::BlasLtEpilogueProto>& info) {
      return ToString(info.param);
    });

TEST(BlasLtTest, EpilogueFromProtoReturnsErrorForInvalidValues) {
  constexpr int kInvalidProtoValue = 123456789;
  EXPECT_FALSE(xla::BlasLtEpilogueProto_IsValid(kInvalidProtoValue));
  EXPECT_THAT(BlasLt::EpilogueFromProto(
                  static_cast<xla::BlasLtEpilogueProto>(kInvalidProtoValue)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RaggedDotModeTest, ToProtoSucceedsForValidValues) {
  EXPECT_EQ(RaggedDotModeToProto(RaggedDotMode::kRaggedNonContracting),
            xla::RaggedDotModeProto::RAGGED_NON_CONTRACTING);
  EXPECT_EQ(RaggedDotModeToProto(RaggedDotMode::kRaggedContracting),
            xla::RaggedDotModeProto::RAGGED_CONTRACTING);
  EXPECT_EQ(RaggedDotModeToProto(RaggedDotMode::kRaggedBatch),
            xla::RaggedDotModeProto::RAGGED_BATCH);
}

using RaggedDotModeFromProtoTest =
    ::testing::TestWithParam<xla::RaggedDotModeProto>;

TEST_P(RaggedDotModeFromProtoTest, SucceedsForValidValue) {
  TF_EXPECT_OK(RaggedDotModeFromProto(GetParam()));
}

std::vector<xla::RaggedDotModeProto> EnumerateRaggedDotModeValues() {
  const google::protobuf::EnumDescriptor* descriptor =
      xla::RaggedDotModeProto_descriptor();
  std::vector<xla::RaggedDotModeProto> values;
  values.reserve(descriptor->value_count());
  for (int i = 0; i < descriptor->value_count(); ++i) {
    values.push_back(
        static_cast<xla::RaggedDotModeProto>(descriptor->value(i)->number()));
  }
  return values;
}

std::string ToString(const xla::RaggedDotModeProto& proto) {
  const google::protobuf::EnumDescriptor* descriptor =
      xla::RaggedDotModeProto_descriptor();
  const google::protobuf::EnumValueDescriptor* value =
      descriptor->FindValueByNumber(proto);
  if (value == nullptr) {
    return absl::StrCat("Unknown(", proto, ")");
  }
  return std::string(value->name());
}

INSTANTIATE_TEST_SUITE_P(
    RaggedDotModeFromProtoTests, RaggedDotModeFromProtoTest,
    ValuesIn(EnumerateRaggedDotModeValues()),
    [](const testing::TestParamInfo<xla::RaggedDotModeProto>& info) {
      return ToString(info.param);
    });

}  // namespace stream_executor::gpu
