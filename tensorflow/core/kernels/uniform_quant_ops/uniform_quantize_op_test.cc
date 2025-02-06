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
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

class UniformQuantizeOpsTest : public OpsTestBase {
 protected:
  struct QParams {
    QParams(int q_axis, int q_min_val, int q_max_val, std::vector<float> scale,
            std::vector<int32> zp)
        : q_axis(q_axis),
          q_min_val(q_min_val),
          q_max_val(q_max_val),
          scale(scale),
          zp(zp) {}
    int q_axis;
    int q_min_val;
    int q_max_val;
    std::vector<float> scale;
    std::vector<int32> zp;
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
    AddInputFromArray<int32>(zp_shape, q_params.zp);

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
  AddInputFromArray<int32>(TensorShape({}), {0});

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

}  // namespace tensorflow
