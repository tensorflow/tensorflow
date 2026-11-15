/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class UniformQuantizeOpsTest : public OpsTestBase {
 protected:
  struct QParams {
    QParams(int q_axis, int q_min_val, int q_max_val, std::vector<float> scale,
            std::vector<int32_t> zp)
        : q_axis(q_axis),
          q_min_val(q_min_val),
          q_max_val(q_max_val),
          scale(scale),
          zp(zp) {}
    int q_axis;
    int q_min_val;
    int q_max_val;
    std::vector<float> scale;
    std::vector<int32_t> zp;
  };

  template <typename Tin, typename Tout>
  void TestUniformQuantize(std::vector<Tin> &input, std::vector<int64_t> &shape,
                           QParams &q_params,
                           std::vector<Tout> &expected_result) {
    TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                     .Input(FakeInput(DataTypeToEnum<Tin>::v()))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("Tin", DataTypeToEnum<Tin>::v())
                     .Attr("Tout", DataTypeToEnum<Tout>::v())
                     .Attr("quantization_axis", q_params.q_axis)
                     .Attr("quantization_min_val", q_params.q_min_val)
                     .Attr("quantization_max_val", q_params.q_max_val)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    const int64_t scale_size = static_cast<int64_t>(q_params.scale.size());
    TensorShape scale_shape(scale_size > 1 ? TensorShape({scale_size})
                                           : TensorShape({}));
    const int64_t zp_size = static_cast<int64_t>(q_params.zp.size());
    TensorShape zp_shape(zp_size > 1 ? TensorShape({zp_size})
                                     : TensorShape({}));
    AddInputFromArray<Tin>(TensorShape(shape), input);
    AddInputFromArray<float>(scale_shape, q_params.scale);
    AddInputFromArray<int32_t>(zp_shape, q_params.zp);

    TF_ASSERT_OK(RunOpKernel());
    Tensor expected(allocator(), DataTypeToEnum<Tout>::v(), TensorShape(shape));

    test::FillValues<Tout>(&expected, expected_result);
    test::ExpectTensorEqual<Tout>(expected, *GetOutput(0));
  }
};

TEST_F(UniformQuantizeOpsTest, QuantizeInvalidQuantizationAxis) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", DT_QINT8)
                   .Attr("quantization_axis", -2)
                   .Attr("quantization_min_val", -127)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  // quantization_axis < -1.
  EXPECT_TRUE(absl::IsInvalidArgument(InitOp()));

  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", DT_QINT8)
                   .Attr("quantization_axis", 2)
                   .Attr("quantization_min_val", -127)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<int32_t>(TensorShape({}), {0});

  // quantization_axis >= input tensor rank.
  EXPECT_TRUE(absl::IsInvalidArgument(RunOpKernel()));
}

TEST_F(UniformQuantizeOpsTest, PerTensorQuantize) {
  std::vector<float> input{-27.0, -20.0, 0.0, 1.0, 5.0, 10.0};
  std::vector<int64_t> shape{2, 3};
  QParams qparams(-1, -127, 127, {0.25}, {-20});
  std::vector<qint8> expected{-127, -100, -20, -16, 0, 20};
  TestUniformQuantize<float, qint8>(input, shape, qparams, expected);
}

TEST_F(UniformQuantizeOpsTest, PerTensorQuint8Quantize) {
  std::vector<float> input{25.0, 20.5, 0.0, 0.5, 15.0, 13.5};
  std::vector<int64_t> shape{2, 3};
  QParams qparams(-1, 0, 255, {0.25}, {-4});
  std::vector<quint8> expected{96, 78, 0, 0, 56, 50};
  TestUniformQuantize<float, quint8>(input, shape, qparams, expected);
}

TEST_F(UniformQuantizeOpsTest, PerChannelQuantize) {
  std::vector<float> input{-27.0, -20.0, 0.0, 1.0, 5.0, 10.0};
  std::vector<int64_t> shape{2, 3};
  QParams qparams(0, -127, 127, {0.25, 0.5}, {-20, -10});
  std::vector<qint8> expected{-127, -100, -20, -8, 0, 10};
  TestUniformQuantize<float, qint8>(input, shape, qparams, expected);
}

TEST_F(UniformQuantizeOpsTest, PerChannelQuint8Quantize) {
  std::vector<float> input{25.0, 20.5, 0.0, 0.5, 15.0, 13.5};
  std::vector<int64_t> shape{2, 3};
  QParams qparams(0, 0, 255, {0.25, 0.5}, {-20, -10});
  std::vector<quint8> expected{80, 62, 0, 0, 20, 17};
  TestUniformQuantize<float, quint8>(input, shape, qparams, expected);
}

TEST_F(UniformQuantizeOpsTest, ValidScaleControlQuantizesExpectedValues) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", DT_QINT8)
                   .Attr("quantization_axis", -1)
                   .Attr("quantization_min_val", -127)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({3}), {-1.0f, 0.0f, 1.0f});
  AddInputFromArray<float>(TensorShape({}), {0.5f});
  AddInputFromArray<int32_t>(TensorShape({}), {0});

  TF_ASSERT_OK(RunOpKernel());
  ASSERT_NE(GetOutput(0), nullptr);
  Tensor expected(allocator(), DT_QINT8, TensorShape({3}));
  test::FillValues<qint8>(&expected, {-2, 0, 2});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

class UniformQuantizeInvalidScaleTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<std::tuple<DataType, int, float>> {};

TEST_P(UniformQuantizeInvalidScaleTest, RejectsInvalidScale) {
  const auto [output_type, axis, scale] = GetParam();
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", output_type)
                   .Attr("quantization_axis", axis)
                   .Attr("quantization_min_val", 0)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2}), {0.0f, 2.0f});
  if (axis == -1) {
    AddInputFromArray<float>(TensorShape({}), {scale});
    AddInputFromArray<int32_t>(TensorShape({}), {0});
  } else {
    // A valid first channel must not hide an invalid later channel.
    AddInputFromArray<float>(TensorShape({2}), {0.5f, scale});
    AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  }

  const auto status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "scales must be positive, finite, and not subnormal.");
}

INSTANTIATE_TEST_SUITE_P(
    ScalarAndPerChannel, UniformQuantizeInvalidScaleTest,
    ::testing::Combine(
        ::testing::Values(DT_QINT8, DT_QUINT8, DT_QINT32),
        ::testing::Values(-1, 0),
        ::testing::Values(1e-45f, 1e-39f, 3e-39f, 0.0f, -0.0f, -1.0f,
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN())));

TEST_F(UniformQuantizeOpsTest, PerTensorQint32ValidScale) {
  std::vector<float> input{-2.0f, 0.0f, 2.0f};
  std::vector<int64_t> shape{3};
  QParams qparams(-1, -127, 127, {0.5f}, {3});
  std::vector<qint32> expected{-1, 3, 7};
  TestUniformQuantize<float, qint32>(input, shape, qparams, expected);
  ASSERT_NE(GetOutput(0), nullptr);
}

TEST_F(UniformQuantizeOpsTest, PerChannelQint32ValidScales) {
  std::vector<float> input{-2.0f, 0.0f, 2.0f, -2.0f, 0.0f, 2.0f};
  std::vector<int64_t> shape{2, 3};
  QParams qparams(0, -127, 127, {0.5f, 1.0f}, {3, 4});
  std::vector<qint32> expected{-1, 3, 7, 2, 4, 6};
  TestUniformQuantize<float, qint32>(input, shape, qparams, expected);
  ASSERT_NE(GetOutput(0), nullptr);
}

TEST_F(UniformQuantizeOpsTest, SmallNormalValidScale) {
  std::vector<float> input{0.0f};
  std::vector<int64_t> shape{1};
  QParams qparams(-1, -127, 127, {std::numeric_limits<float>::min()}, {3});
  std::vector<qint8> expected{3};
  TestUniformQuantize<float, qint8>(input, shape, qparams, expected);
  ASSERT_NE(GetOutput(0), nullptr);
}

class UniformQuantizeBenignSaturationTest
    : public UniformQuantizeOpsTest,
      public ::testing::WithParamInterface<std::tuple<DataType, int>> {};

TEST_P(UniformQuantizeBenignSaturationTest, SaturatesToDeclaredBounds) {
  const auto [output_type, axis] = GetParam();
  std::vector<float> input{-10.0f, 10.0f};
  std::vector<int64_t> shape{2};
  const std::vector<float> scales =
      axis == -1 ? std::vector<float>{0.5f} : std::vector<float>{0.5f, 0.5f};
  const std::vector<int32_t> zero_points =
      axis == -1 ? std::vector<int32_t>{3} : std::vector<int32_t>{3, 3};
  QParams qparams(axis, 0, 20, scales, zero_points);
  if (output_type == DT_QINT8) {
    std::vector<qint8> expected{0, 20};
    TestUniformQuantize<float, qint8>(input, shape, qparams, expected);
  } else if (output_type == DT_QUINT8) {
    std::vector<quint8> expected{0, 20};
    TestUniformQuantize<float, quint8>(input, shape, qparams, expected);
  } else {
    std::vector<qint32> expected{0, 20};
    TestUniformQuantize<float, qint32>(input, shape, qparams, expected);
  }
  ASSERT_NE(GetOutput(0), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    ScalarAndPerChannel, UniformQuantizeBenignSaturationTest,
    ::testing::Combine(::testing::Values(DT_QINT8, DT_QUINT8, DT_QINT32),
                       ::testing::Values(-1, 0)));

TEST_F(UniformQuantizeOpsTest, LargeNormalValidScale) {
  const float scale = std::numeric_limits<float>::max();
  std::vector<float> input{-scale, 0.0f, scale};
  std::vector<int64_t> shape{3};
  QParams qparams(-1, -127, 127, {std::numeric_limits<float>::max()}, {3});
  std::vector<qint8> expected{2, 3, 4};
  TestUniformQuantize<float, qint8>(input, shape, qparams, expected);
  ASSERT_NE(GetOutput(0), nullptr);
}

TEST_F(UniformQuantizeOpsTest, RoundsHalfIntegersTowardPositiveInfinity) {
  std::vector<float> input{-1.5f, -0.5f, 0.5f, 1.5f};
  std::vector<int64_t> shape{4};
  QParams qparams(-1, -127, 127, {1.0f}, {0});
  std::vector<qint8> expected{-1, 0, 1, 2};
  TestUniformQuantize<float, qint8>(input, shape, qparams, expected);
  ASSERT_NE(GetOutput(0), nullptr);
}

class UniformQuantizeEmptyInputScaleTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<std::tuple<DataType, float>> {};

TEST_P(UniformQuantizeEmptyInputScaleTest, RejectsInvalidScale) {
  const auto [output_type, scale] = GetParam();
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantize")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tin", DT_FLOAT)
                   .Attr("Tout", output_type)
                   .Attr("quantization_axis", -1)
                   .Attr("quantization_min_val", 0)
                   .Attr("quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // Scale validity does not depend on whether there are values to quantize.
  AddInputFromArray<float>(TensorShape({0}), std::vector<float>{});
  AddInputFromArray<float>(TensorShape({}), {scale});
  AddInputFromArray<int32_t>(TensorShape({}), {0});

  const auto status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_EQ(status.message(),
            "scales must be positive, finite, and not subnormal.");
}

INSTANTIATE_TEST_SUITE_P(
    ScalarScales, UniformQuantizeEmptyInputScaleTest,
    ::testing::Combine(::testing::Values(DT_QINT8, DT_QUINT8, DT_QINT32),
                       ::testing::Values(1e-45f, 1e-39f, 3e-39f)));

// Shared helpers for the complete-invariant tests below.
class UniformQuantizeEdgeTest : public OpsTestBase {
 protected:
  absl::Status Init(DataType tout, int axis, int32_t min_val, int32_t max_val) {
    absl::Status status = NodeDefBuilder("test", "UniformQuantize")
                              .Input(FakeInput(DT_FLOAT))
                              .Input(FakeInput(DT_FLOAT))
                              .Input(FakeInput(DT_INT32))
                              .Attr("Tin", DT_FLOAT)
                              .Attr("Tout", tout)
                              .Attr("quantization_axis", axis)
                              .Attr("quantization_min_val", min_val)
                              .Attr("quantization_max_val", max_val)
                              .Finalize(node_def());
    if (!status.ok()) return status;
    return InitOp();
  }

  static bool MessageContains(const absl::Status& status,
                              const std::string& needle) {
    return std::string(status.message()).find(needle) != std::string::npos;
  }
};

int32_t OutputMin(DataType tout) {
  return tout == DT_QINT8    ? -128
         : tout == DT_QUINT8 ? 0
                             : std::numeric_limits<int32_t>::min();
}

int32_t OutputMax(DataType tout) {
  return tout == DT_QINT8    ? 127
         : tout == DT_QUINT8 ? 255
                             : std::numeric_limits<int32_t>::max();
}

// Reference for floor(x / scale + 0.5) + zero_point, clamped to [lo, hi].
// It is computed in double and never narrows before the clamp, so it does not
// share the float-to-int32 conversion that the kernel must avoid.
int32_t ReferenceQuantize(float x, float scale, int32_t zero_point, int32_t lo,
                          int32_t hi) {
  const double q =
      std::floor(static_cast<double>(x) / static_cast<double>(scale) + 0.5) +
      static_cast<double>(zero_point);
  return static_cast<int32_t>(
      std::min(std::max(q, static_cast<double>(lo)), static_cast<double>(hi)));
}

template <typename T>
void ExpectQuantized(const Tensor& actual, DataType tout,
                     const std::vector<int32_t>& expected_ints) {
  Tensor expected(tout,
                  TensorShape({static_cast<int64_t>(expected_ints.size())}));
  auto flat = expected.flat<T>();
  for (size_t i = 0; i < expected_ints.size(); ++i) {
    flat(i) = static_cast<T>(expected_ints[i]);
  }
  test::ExpectTensorEqual<T>(expected, actual);
}

void ExpectQuantizedOutput(const Tensor& actual, DataType tout,
                           const std::vector<int32_t>& expected_ints) {
  ASSERT_EQ(actual.dtype(), tout);
  if (tout == DT_QINT8) {
    ExpectQuantized<qint8>(actual, tout, expected_ints);
  } else if (tout == DT_QUINT8) {
    ExpectQuantized<quint8>(actual, tout, expected_ints);
  } else {
    ExpectQuantized<qint32>(actual, tout, expected_ints);
  }
}

// (a) Storage-shape conjuncts of the scalar and size checks.
TEST_F(UniformQuantizeEdgeTest, RankOneSingletonZeroPointWithScalarScales) {
  TF_ASSERT_OK(Init(DT_QINT8, -1, -127, 127));
  AddInputFromArray<float>(TensorShape({2}), {0.0f, 1.0f});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  const auto status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_TRUE(MessageContains(status, "zero_points must be scalar.")) << status;
}

TEST_F(UniformQuantizeEdgeTest, RankOneSingletonScalesWithRankOneZeroPoint) {
  TF_ASSERT_OK(Init(DT_QINT8, -1, -127, 127));
  AddInputFromArray<float>(TensorShape({2}), {0.0f, 1.0f});
  AddInputFromArray<float>(TensorShape({1}), {1.0f});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  const auto status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_TRUE(MessageContains(status, "scales must be scalar.")) << status;
}

TEST_F(UniformQuantizeEdgeTest, PerChannelScalesAndZeroPointsSizeMismatch) {
  TF_ASSERT_OK(Init(DT_QINT8, 0, -127, 127));
  AddInputFromArray<float>(TensorShape({2}), {0.0f, 1.0f});
  AddInputFromArray<float>(TensorShape({2}), {1.0f, 1.0f});
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 0, 0});
  const auto status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_TRUE(
      MessageContains(status, "scales and zero_points sizes must match."))
      << status;
}

// (c) Non-finite input values, scalar and per-channel, first and last element.
class UniformQuantizeNonFiniteInputTest
    : public UniformQuantizeEdgeTest,
      public ::testing::WithParamInterface<
          std::tuple<DataType, int, float, bool>> {};

TEST_P(UniformQuantizeNonFiniteInputTest, RejectsNonFiniteInput) {
  const auto [tout, axis, bad_value, bad_first] = GetParam();
  TF_ASSERT_OK(Init(tout, axis, OutputMin(tout), OutputMax(tout)));
  const std::vector<float> input =
      bad_first ? std::vector<float>{bad_value, 1.0f, 2.0f}
                : std::vector<float>{1.0f, 2.0f, bad_value};
  AddInputFromArray<float>(TensorShape({3}), input);
  if (axis == -1) {
    AddInputFromArray<float>(TensorShape({}), {1.0f});
    AddInputFromArray<int32_t>(TensorShape({}), {0});
  } else {
    AddInputFromArray<float>(TensorShape({3}), {1.0f, 1.0f, 1.0f});
    AddInputFromArray<int32_t>(TensorShape({3}), {0, 0, 0});
  }
  const auto status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_TRUE(MessageContains(status, "input values must be finite."))
      << status;
}

INSTANTIATE_TEST_SUITE_P(
    ScalarAndPerChannel, UniformQuantizeNonFiniteInputTest,
    ::testing::Combine(
        ::testing::Values(DT_QINT8, DT_QUINT8, DT_QINT32),
        ::testing::Values(-1, 0),
        ::testing::Values(std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()),
        ::testing::Bool()));

// (b), (d), (e): finite values whose intermediate result leaves int32, an
// extreme zero point, and a large normal scale with a subnormal reciprocal.
// Every case is compared against a double-precision reference that saturates
// to the declared quantization range.
struct NumericCase {
  float input;
  float scale;
  int32_t zero_point;
};

class UniformQuantizeNumericBoundaryTest
    : public UniformQuantizeEdgeTest,
      public ::testing::WithParamInterface<
          std::tuple<DataType, int, NumericCase>> {};

TEST_P(UniformQuantizeNumericBoundaryTest, SaturatesLikeWideReference) {
  const auto [tout, axis, c] = GetParam();
  const int32_t lo = OutputMin(tout);
  const int32_t hi = OutputMax(tout);
  TF_ASSERT_OK(Init(tout, axis, lo, hi));
  AddInputFromArray<float>(TensorShape({2}), {c.input, 0.0f});
  if (axis == -1) {
    AddInputFromArray<float>(TensorShape({}), {c.scale});
    AddInputFromArray<int32_t>(TensorShape({}), {c.zero_point});
  } else {
    AddInputFromArray<float>(TensorShape({2}), {c.scale, c.scale});
    AddInputFromArray<int32_t>(TensorShape({2}), {c.zero_point, c.zero_point});
  }
  TF_ASSERT_OK(RunOpKernel());
  ExpectQuantizedOutput(
      *GetOutput(0), tout,
      {ReferenceQuantize(c.input, c.scale, c.zero_point, lo, hi),
       ReferenceQuantize(0.0f, c.scale, c.zero_point, lo, hi)});
}

constexpr float kLargeFinite = 1e30f;
constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();
constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();

INSTANTIATE_TEST_SUITE_P(
    ScalarAndPerChannel, UniformQuantizeNumericBoundaryTest,
    ::testing::Combine(
        ::testing::Values(DT_QINT8, DT_QUINT8, DT_QINT32),
        ::testing::Values(-1, 0),
        ::testing::Values(
            // Positive controls with ordinary values.
            NumericCase{0.75f, 0.5f, 3}, NumericCase{-0.75f, 0.5f, 3},
            // (d) The product exceeds int32 although input and scale are
            // finite and the scale is valid.
            NumericCase{kLargeFinite, 1.0f, 0},
            NumericCase{-kLargeFinite, 1.0f, 0},
            // (d) The float product overflows to infinity.
            NumericCase{1e10f, std::numeric_limits<float>::min(), 0},
            NumericCase{-1e10f, std::numeric_limits<float>::min(), 0},
            // (e) The zero point addition leaves int32.
            NumericCase{1.0f, 1.0f, kInt32Max},
            NumericCase{-1.0f, 1.0f, kInt32Max},
            NumericCase{1.0f, 1.0f, kInt32Min},
            NumericCase{-1.0f, 1.0f, kInt32Min},
            // (b) Large normal scales have subnormal reciprocals.
            NumericCase{1e38f, 1e38f, 0}, NumericCase{-2e38f, 1e38f, 5},
            NumericCase{std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(), 0})));

// (b) With subnormals flushed to zero the float reciprocal of a large normal
// scale becomes zero. The result must still be the mathematically correct one.
TEST_F(UniformQuantizeEdgeTest, LargeScaleReciprocalFlushedToZero) {
  TF_ASSERT_OK(Init(DT_QINT32, -1, kInt32Min, kInt32Max));
  AddInputFromArray<float>(TensorShape({3}), {1e38f, 2e38f, -1e38f});
  AddInputFromArray<float>(TensorShape({}), {1e38f});
  AddInputFromArray<int32_t>(TensorShape({}), {0});
  {
    port::ScopedFlushDenormal flush;
    TF_ASSERT_OK(RunOpKernel());
  }
  ExpectQuantizedOutput(*GetOutput(0), DT_QINT32, {1, 2, -1});
}

TEST_F(UniformQuantizeEdgeTest, LargeScaleReciprocalFlushedToZeroPerChannel) {
  TF_ASSERT_OK(Init(DT_QINT32, 0, kInt32Min, kInt32Max));
  AddInputFromArray<float>(TensorShape({3}), {1e38f, 2e38f, -1e38f});
  AddInputFromArray<float>(TensorShape({3}), {1e38f, 1e38f, 1e38f});
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 0, 0});
  {
    port::ScopedFlushDenormal flush;
    TF_ASSERT_OK(RunOpKernel());
  }
  ExpectQuantizedOutput(*GetOutput(0), DT_QINT32, {1, 2, -1});
}

// (f), (g): quantization range preconditions checked at kernel construction.
struct RangeCase {
  DataType tout;
  int32_t min_val;
  int32_t max_val;
  const char* message;  // nullptr means the range is valid.
};

class UniformQuantizeRangeTest
    : public UniformQuantizeEdgeTest,
      public ::testing::WithParamInterface<std::tuple<RangeCase, int>> {};

TEST_P(UniformQuantizeRangeTest, ChecksRangeAtConstruction) {
  const auto [c, axis] = GetParam();
  const auto status = Init(c.tout, axis, c.min_val, c.max_val);
  if (c.message == nullptr) {
    TF_EXPECT_OK(status);
  } else {
    EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
    EXPECT_TRUE(MessageContains(status, c.message)) << status;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Ranges, UniformQuantizeRangeTest,
    ::testing::Combine(
        ::testing::Values(
            // (f) Unordered bounds for every output type.
            RangeCase{DT_QINT8, 10, 5, "quantization bounds must be ordered."},
            RangeCase{DT_QUINT8, 10, 5, "quantization bounds must be ordered."},
            RangeCase{DT_QINT32, 10, 5, "quantization bounds must be ordered."},
            RangeCase{DT_QINT32, kInt32Max, kInt32Min,
                      "quantization bounds must be ordered."},
            // (g) Bounds outside the output type, one bound at a time.
            RangeCase{DT_QINT8, -129, 127,
                      "quantization_min_val must fit Tout."},
            RangeCase{DT_QINT8, -128, 128,
                      "quantization_max_val must fit Tout."},
            RangeCase{DT_QUINT8, -1, 255,
                      "quantization_min_val must fit Tout."},
            RangeCase{DT_QUINT8, 0, 256, "quantization_max_val must fit Tout."},
            // Valid boundaries and a single-point range.
            RangeCase{DT_QINT8, -128, 127, nullptr},
            RangeCase{DT_QUINT8, 0, 255, nullptr},
            RangeCase{DT_QINT32, kInt32Min, kInt32Max, nullptr},
            RangeCase{DT_QINT8, 7, 7, nullptr}),
        ::testing::Values(-1, 0)));

}  // namespace tensorflow
