/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

using GraphRunner = std::function<void(
    const Tensor& input, const Tensor& scale, const Tensor& offset,
    const Tensor& mean, const Tensor& variance,
    const float exponential_avg_factor, const bool is_training, Tensor* output,
    Tensor* batch_mean, Tensor* batch_var)>;

using GraphRunnerGrad = std::function<void(
    const Tensor& input, const Tensor& filter, const Tensor& y_backprop,
    const Tensor& scale, const Tensor& mean, const Tensor& variance,
    const Tensor& res_sp3, Tensor* output, Tensor* scale_backprop,
    Tensor* offset_backprop, bool disable_grappler_opts)>;

template <typename T>
class CommonTestUtilities : public OpsTestBase {
 public:
  void TestBody() {}

  static void VerifyTensorsClose(const float exponential_avg_factor,
                                 const bool is_training, const GraphRunner& run,
                                 const GraphRunner& run_mkl) {
    int batch = 1;
    int height = 10;
    int width = 10;
    int depth = 3;
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor input(dtype, {batch, height, width, depth});
    input.flat<T>() = input.flat<T>().template setRandom<random_gen_>();
    Tensor scale(dtype, {depth});
    scale.flat<T>() = scale.flat<T>().template setRandom<random_gen_>();
    Tensor offset(dtype, {depth});
    offset.flat<T>() = offset.flat<T>().template setRandom<random_gen_>();

    if (is_training && (exponential_avg_factor == 1.0)) {
      depth = 0;
    }
    Tensor mean(dtype, {depth});
    mean.flat<T>() = mean.flat<T>().template setRandom<random_gen_>();
    Tensor variance(dtype, {depth});
    variance.flat<T>() =
        variance.flat<T>().template setRandom<random_gen_>().abs();

    Tensor output;
    Tensor batch_mean;
    Tensor batch_var;
    Tensor mkl_output;
    Tensor mkl_batch_mean;
    Tensor mkl_batch_var;

    run(input, scale, offset, mean, variance, exponential_avg_factor,
        is_training, &output, &batch_mean, &batch_var);
    run_mkl(input, scale, offset, mean, variance, exponential_avg_factor,
            is_training, &mkl_output, &mkl_batch_mean, &mkl_batch_var);

    ASSERT_EQ(output.dtype(), mkl_output.dtype());
    ASSERT_EQ(output.shape(), mkl_output.shape());
    ASSERT_EQ(batch_mean.dtype(), mkl_batch_mean.dtype());
    ASSERT_EQ(batch_mean.shape(), mkl_batch_mean.shape());
    ASSERT_EQ(batch_var.dtype(), mkl_batch_var.dtype());
    ASSERT_EQ(batch_var.shape(), mkl_batch_var.shape());

    test::ExpectClose(output, mkl_output, 1e-5);
    test::ExpectClose(batch_mean, mkl_batch_mean, 1e-5);
    test::ExpectClose(batch_var, mkl_batch_var, 1e-5);
  }

  static void VerifyTensorsCloseForGrad(const float epsilon,
                                        const GraphRunnerGrad& run,
                                        const GraphRunnerGrad& run_mkl) {
    int batch = 2;
    int height = 8;
    int width = 8;
    int depth = 1;
    int filter_height = 3;
    int filter_width = 3;
    int in_channels = 1;
    int out_channels = 6;
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor input(dtype, {batch, height, width, depth});
    input.flat<T>() = input.flat<T>().template setRandom<random_gen_>();
    Tensor filter(dtype,
                  {filter_height, filter_width, in_channels, out_channels});
    filter.flat<T>() = filter.flat<T>().template setRandom<random_gen_>();

    Tensor y_backprop(dtype, {batch, height, width, out_channels});
    y_backprop.flat<T>() =
        y_backprop.flat<T>().template setRandom<random_gen_>();
    Tensor scale(dtype, {out_channels});
    scale.flat<T>() = scale.flat<T>().template setRandom<random_gen_>();
    Tensor mean(dtype, {out_channels});
    mean.flat<T>() = mean.flat<T>().template setRandom<random_gen_>();
    Tensor variance(dtype, {out_channels});
    variance.flat<T>() =
        variance.flat<T>().template setRandom<random_gen_>().abs();
    Tensor res_sp3(dtype, {out_channels});
    res_sp3.flat<T>() =
        res_sp3.flat<T>().template setRandom<random_gen_>().abs();

    Tensor output;
    Tensor scale_backprop;
    Tensor offset_backprop;
    Tensor mkl_output;
    Tensor mkl_scale_backprop;
    Tensor mkl_offset_backprop;

    run(input, filter, y_backprop, scale, mean, variance, res_sp3, &output,
        &scale_backprop, &offset_backprop, epsilon);

    run_mkl(input, filter, y_backprop, scale, mean, variance, res_sp3,
            &mkl_output, &mkl_scale_backprop, &mkl_offset_backprop, epsilon);

    ASSERT_EQ(output.dtype(), mkl_output.dtype());
    ASSERT_EQ(output.shape(), mkl_output.shape());
    ASSERT_EQ(scale_backprop.dtype(), mkl_scale_backprop.dtype());
    ASSERT_EQ(scale_backprop.shape(), mkl_scale_backprop.shape());
    ASSERT_EQ(offset_backprop.dtype(), mkl_offset_backprop.dtype());
    ASSERT_EQ(offset_backprop.shape(), mkl_offset_backprop.shape());

    test::ExpectClose(output, mkl_output, 1e-5);
    test::ExpectClose(scale_backprop, mkl_scale_backprop, 1e-5, /*rtol*/ 1e-5);
    test::ExpectClose(offset_backprop, mkl_offset_backprop, 1e-5);
  }

 private:
  using random_gen_ = Eigen::internal::NormalRandomGenerator<T>;
};

template <typename T>
class Conv2DOpTest : public OpsTestBase {
  void TestBody() {}

 public:
  void RunConv2D(const Tensor& input, const Tensor& filter, Tensor* output) {
    DataType dtype = DataTypeToEnum<T>::v();

    TF_EXPECT_OK(NodeDefBuilder("MklConv2D", "_MklNativeConv2D")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Attr("strides", {1, 1, 1, 1})
                     .Attr("padding", "SAME")
                     .Attr("data_format", "NHWC")
                     .Attr("_kernel", "MklNameChangeOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(input.shape(), input.flat<T>());
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    TF_ASSERT_OK(RunOpKernel());

    *output = *GetOutput(0);
  }
};

template <typename T>
class FusedBatchNormOpTest : public OpsTestBase {
 protected:
  void VerifyFusedBatchNorm(const float exponential_avg_factor,
                            const bool is_training) {
    const GraphRunner run = [this](const Tensor& input, const Tensor& scale,
                                   const Tensor& offset, const Tensor& mean,
                                   const Tensor& variance,
                                   const float exponential_avg_factor,
                                   const bool is_training, Tensor* output,
                                   Tensor* batch_mean, Tensor* batch_var) {
      auto root = tensorflow::Scope::NewRootScope();
      auto input_op =
          ops::Const(root.WithOpName("input"), Input::Initializer(input));
      auto scale_op =
          ops::Const(root.WithOpName("scale"), Input::Initializer(scale));
      auto offset_op =
          ops::Const(root.WithOpName("offset"), Input::Initializer(offset));
      auto mean_op =
          ops::Const(root.WithOpName("mean"), Input::Initializer(mean));
      auto var_op =
          ops::Const(root.WithOpName("variance"), Input::Initializer(variance));

      ops::FusedBatchNorm::Attrs attr;
      attr = attr.IsTraining(is_training);
      attr = attr.ExponentialAvgFactor(exponential_avg_factor);
      attr = attr.Epsilon(0.001);
      auto bn = ops::FusedBatchNorm(root.WithOpName("FusedBatchNorm"), input_op,
                                    scale_op, offset_op, mean_op, var_op, attr);
      auto y = ops::Identity(root.WithOpName("y"), bn.y);
      auto y_batch_mean =
          ops::Identity(root.WithOpName("y_batch_mean"), bn.batch_mean);
      auto y_batch_var =
          ops::Identity(root.WithOpName("y_batch_var"), bn.batch_variance);

      tensorflow::GraphDef graph;
      TF_ASSERT_OK(root.ToGraphDef(&graph));

      std::unique_ptr<tensorflow::Session> session(
          tensorflow::NewSession(tensorflow::SessionOptions()));
      TF_ASSERT_OK(session->Create(graph));

      std::vector<Tensor> output_tensors;
      TF_ASSERT_OK(session->Run({}, {"y", "y_batch_mean", "y_batch_var"}, {},
                                &output_tensors));

      *output = output_tensors[0];
      *batch_mean = output_tensors[1];
      *batch_var = output_tensors[2];
    };

    const GraphRunner run_mkl = [this](const Tensor& input, const Tensor& scale,
                                       const Tensor& offset, const Tensor& mean,
                                       const Tensor& variance,
                                       const float exponential_avg_factor,
                                       const bool is_training, Tensor* output,
                                       Tensor* batch_mean, Tensor* batch_var) {
      DataType dtype = DataTypeToEnum<T>::v();
      TF_EXPECT_OK(
          NodeDefBuilder("MklNativeFusedBatchNorm", "_MklNativeFusedBatchNorm")
              .Input(FakeInput(dtype))
              .Input(FakeInput(DT_FLOAT))
              .Input(FakeInput(DT_FLOAT))
              .Input(FakeInput(DT_FLOAT))
              .Input(FakeInput(DT_FLOAT))
              .Attr("exponential_avg_factor", exponential_avg_factor)
              .Attr("epsilon", 0.001)
              .Attr("is_training", is_training)
              .Attr("_kernel", "MklNameChangeOp")
              .Finalize(node_def()));
      TF_EXPECT_OK(InitOp());

      AddInputFromArray<T>(input.shape(), input.flat<T>());
      AddInputFromArray<float>(scale.shape(), scale.flat<float>());
      AddInputFromArray<float>(offset.shape(), offset.flat<float>());
      AddInputFromArray<float>(mean.shape(), mean.flat<float>());
      AddInputFromArray<float>(variance.shape(), variance.flat<float>());
      TF_ASSERT_OK(RunOpKernel());

      *output = *GetOutput(0);
      *batch_mean = *GetOutput(1);
      *batch_var = *GetOutput(2);
    };

    CommonTestUtilities<T>::VerifyTensorsClose(exponential_avg_factor,
                                               is_training, run, run_mkl);
  }

  void VerifyFusedBatchNormGradWithConv2D(const float epsilon) {
    const GraphRunnerGrad run =
        [this](const Tensor& input, const Tensor& filter,
               const Tensor& y_backprop, const Tensor& scale,
               const Tensor& mean, const Tensor& variance,
               const Tensor& res_sp3, Tensor* x_backprop_tensor,
               Tensor* scale_backprop_tensor, Tensor* offset_backprop_tensor,
               const float epsilon) {
          auto root = tensorflow::Scope::NewRootScope();

          auto input_op =
              ops::Const(root.WithOpName("input"), Input::Initializer(input));
          auto filter_op =
              ops::Const(root.WithOpName("filter"), Input::Initializer(filter));
          ops::Conv2D::Attrs conv_attr;
          conv_attr = conv_attr.DataFormat("NHWC");
          auto conv = ops::Conv2D(root.WithOpName("Conv"), input_op, filter_op,
                                  {1, 1, 1, 1}, "SAME", conv_attr);
          // -------------------------------------------------------------
          auto y_backprop_op = ops::Const(root.WithOpName("y_backprop"),
                                          Input::Initializer(y_backprop));
          auto scale_op =
              ops::Const(root.WithOpName("scale"), Input::Initializer(scale));
          auto mean_op =
              ops::Const(root.WithOpName("mean"), Input::Initializer(mean));
          auto var_op = ops::Const(root.WithOpName("variance"),
                                   Input::Initializer(variance));
          auto res_sp3_op = ops::Const(root.WithOpName("reserve_space_3"),
                                       Input::Initializer(res_sp3));
          ops::FusedBatchNormGradV3::Attrs bn_attr;
          bn_attr = bn_attr.IsTraining(true);
          bn_attr = bn_attr.Epsilon(epsilon);
          bn_attr = bn_attr.DataFormat("NHWC");
          auto bn = ops::FusedBatchNormGradV3(
              root.WithOpName("FusedBatchNormGrad"), y_backprop_op, conv,
              scale_op, mean_op, var_op, res_sp3_op, bn_attr);

          auto x_backprop =
              ops::Identity(root.WithOpName("x_backprop"), bn.x_backprop);
          auto scale_backprop = ops::Identity(root.WithOpName("scale_backprop"),
                                              bn.scale_backprop);
          auto offset_backprop = ops::Identity(
              root.WithOpName("offset_backprop"), bn.offset_backprop);

          tensorflow::GraphDef graph;
          TF_ASSERT_OK(root.ToGraphDef(&graph));

          tensorflow::SessionOptions session_options;
          std::unique_ptr<tensorflow::Session> session(
              tensorflow::NewSession(session_options));
          TF_ASSERT_OK(session->Create(graph));

          std::vector<Tensor> output_tensors;
          TF_ASSERT_OK(session->Run(
              {}, {"x_backprop", "scale_backprop", "offset_backprop"}, {},
              &output_tensors));

          *x_backprop_tensor = output_tensors[0];
          *scale_backprop_tensor = output_tensors[1];
          *offset_backprop_tensor = output_tensors[2];
        };

    const GraphRunnerGrad run_mkl =
        [this](const Tensor& input, const Tensor& filter,
               const Tensor& y_backprop, const Tensor& scale,
               const Tensor& mean, const Tensor& variance,
               const Tensor& res_sp3, Tensor* x_backprop_tensor,
               Tensor* scale_backprop_tensor, Tensor* offset_backprop_tensor,
               const float epsilon) {
          Tensor conv2d_output;
          Conv2DOpTest<T> conv2d_test;
          conv2d_test.RunConv2D(input, filter, &conv2d_output);

          DataType dtype = DataTypeToEnum<T>::v();
          TF_EXPECT_OK(NodeDefBuilder("MklFusedBatchNorm",
                                      "_MklNativeFusedBatchNormGradV3")
                           .Input(FakeInput(dtype))
                           .Input(FakeInput(dtype))
                           .Input(FakeInput(DT_FLOAT))
                           .Input(FakeInput(DT_FLOAT))
                           .Input(FakeInput(DT_FLOAT))
                           .Input(FakeInput(DT_FLOAT))
                           .Attr("epsilon", epsilon)
                           .Attr("is_training", true)
                           .Attr("data_format", "NHWC")
                           .Attr("_kernel", "MklNameChangeOp")
                           .Finalize(node_def()));
          TF_EXPECT_OK(InitOp());

          AddInputFromArray<T>(y_backprop.shape(), y_backprop.flat<T>());
          AddInputFromArray<T>(conv2d_output.shape(), conv2d_output.flat<T>());
          AddInputFromArray<float>(scale.shape(), scale.flat<float>());
          AddInputFromArray<float>(mean.shape(), mean.flat<float>());
          AddInputFromArray<float>(variance.shape(), variance.flat<float>());
          AddInputFromArray<float>(res_sp3.shape(), res_sp3.flat<float>());
          TF_ASSERT_OK(RunOpKernel());

          *x_backprop_tensor = *GetOutput(0);
          *scale_backprop_tensor = *GetOutput(1);
          *offset_backprop_tensor = *GetOutput(2);
        };

    CommonTestUtilities<T>::VerifyTensorsCloseForGrad(epsilon, run, run_mkl);
  }
};

TYPED_TEST_SUITE_P(FusedBatchNormOpTest);

TYPED_TEST_P(FusedBatchNormOpTest, Training) {
  const float exponential_avg_factor = 1.0;
  const bool is_training = true;
  this->VerifyFusedBatchNorm(exponential_avg_factor, is_training);
}

TYPED_TEST_P(FusedBatchNormOpTest, TrainingRunningMean) {
  const float exponential_avg_factor = 0.5;
  const bool is_training = true;
  this->VerifyFusedBatchNorm(exponential_avg_factor, is_training);
}

TYPED_TEST_P(FusedBatchNormOpTest, Inference) {
  const float exponential_avg_factor = 1.0;
  const bool is_training = false;
  this->VerifyFusedBatchNorm(exponential_avg_factor, is_training);
}

TYPED_TEST_P(FusedBatchNormOpTest, InferenceIgnoreAvgFactor) {
  const float exponential_avg_factor = 0.5;
  const bool is_training = false;
  this->VerifyFusedBatchNorm(exponential_avg_factor, is_training);
}

TYPED_TEST_P(FusedBatchNormOpTest, FusedBatchNormGradV3) {
  const float epsilon = 0.001;
  this->VerifyFusedBatchNormGradWithConv2D(epsilon);
}

REGISTER_TYPED_TEST_SUITE_P(FusedBatchNormOpTest, Training, TrainingRunningMean,
                            Inference, InferenceIgnoreAvgFactor,
                            FusedBatchNormGradV3);

using FusedBatchNormDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedBatchNormOpTest,
                               FusedBatchNormDataTypes);

}  // namespace tensorflow

#endif  // INTEL_MKL
