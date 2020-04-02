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
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

// Helper class for converting MKL tensors to TF tensors and comparing to
// expected values

static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

using GraphRunner = std::function<void(
    const Tensor& input, const Tensor& scale, const Tensor& offset,
    const Tensor& mean, const Tensor& variance,
    const float exponential_avg_factor, const bool is_training, Tensor* output,
    Tensor* batch_mean, Tensor* batch_var)>;

template <typename T>
class CommonTestUtilities : public OpsTestBase {
 public:
  void PerformConversion(DataType dtype, const Tensor& tensor,
                         const Tensor& mkl_meta_tensor, Tensor* output) {
    // Create an MKL to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // Mkl second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(tensor.shape(), tensor.flat<T>());
    AddInputFromArray<uint8>(mkl_meta_tensor.shape(),
                             mkl_meta_tensor.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    *output = *GetOutput(0);
  }

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

 private:
  using random_gen_ = Eigen::internal::NormalRandomGenerator<T>;
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
      TF_EXPECT_OK(NodeDefBuilder("MklFusedBatchNorm", "_MklFusedBatchNorm")
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(DT_FLOAT))
                       .Input(FakeInput(DT_FLOAT))
                       .Input(FakeInput(DT_FLOAT))
                       .Input(FakeInput(DT_FLOAT))
                       .Input(FakeInput(DT_UINT8))
                       .Input(FakeInput(DT_UINT8))
                       .Input(FakeInput(DT_UINT8))
                       .Input(FakeInput(DT_UINT8))
                       .Input(FakeInput(DT_UINT8))
                       .Attr("exponential_avg_factor", exponential_avg_factor)
                       .Attr("epsilon", 0.001)
                       .Attr("is_training", is_training)
                       .Attr("_kernel", "MklLayoutDependentOp")
                       .Finalize(node_def()));
      TF_EXPECT_OK(InitOp());

      AddInputFromArray<float>(input.shape(), input.flat<T>());
      AddInputFromArray<float>(scale.shape(), scale.flat<T>());
      AddInputFromArray<float>(offset.shape(), offset.flat<T>());
      AddInputFromArray<float>(mean.shape(), mean.flat<T>());
      AddInputFromArray<float>(variance.shape(), variance.flat<T>());
      for (int i = 0; i < 5; ++i)
        AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
      TF_ASSERT_OK(RunOpKernel());

      CommonTestUtilities<T> test_util;
      test_util.PerformConversion(dtype, *GetOutput(0), *GetOutput(5), output);

      CommonTestUtilities<T> test_util_mean;
      test_util_mean.PerformConversion(dtype, *GetOutput(1), *GetOutput(6),
                                       batch_mean);

      CommonTestUtilities<T> test_util_var;
      test_util_var.PerformConversion(dtype, *GetOutput(2), *GetOutput(7),
                                      batch_var);
    };

    CommonTestUtilities<T>::VerifyTensorsClose(exponential_avg_factor,
                                               is_training, run, run_mkl);
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

REGISTER_TYPED_TEST_SUITE_P(FusedBatchNormOpTest, Training, TrainingRunningMean,
                            Inference, InferenceIgnoreAvgFactor);

using FusedBatchNormDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedBatchNormOpTest,
                               FusedBatchNormDataTypes);

}  // namespace tensorflow

#endif  // INTEL_MKL
