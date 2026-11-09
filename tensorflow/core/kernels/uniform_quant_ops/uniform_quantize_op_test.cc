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
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

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
}

TEST_F(UniformQuantizeOpsTest, PerChannelQint32ValidScales) {
  std::vector<float> input{-2.0f, 0.0f, 2.0f, -2.0f, 0.0f, 2.0f};
  std::vector<int64_t> shape{2, 3};
  QParams qparams(0, -127, 127, {0.5f, 1.0f}, {3, 4});
  std::vector<qint32> expected{-1, 3, 7, 2, 4, 6};
  TestUniformQuantize<float, qint32>(input, shape, qparams, expected);
}

TEST_F(UniformQuantizeOpsTest, SmallNormalValidScale) {
  std::vector<float> input{0.0f};
  std::vector<int64_t> shape{1};
  QParams qparams(-1, -127, 127, {std::numeric_limits<float>::min()}, {3});
  std::vector<qint8> expected{3};
  TestUniformQuantize<float, qint8>(input, shape, qparams, expected);
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
  EXPECT_EQ(status.message(),
            "scales must be positive, finite, and not subnormal.");
}

INSTANTIATE_TEST_SUITE_P(
    ScalarScales, UniformQuantizeEmptyInputScaleTest,
    ::testing::Combine(::testing::Values(DT_QINT8, DT_QUINT8, DT_QINT32),
                       ::testing::Values(1e-45f, 1e-39f, 3e-39f)));

}  // namespace tensorflow
