/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

using BiasAddGraphRunner =
    std::function<void(const Tensor& input_data, const Tensor& filter_data,
                       const Tensor& bias_data, Tensor* out)>;

using FusedGraphRunner =
    std::function<void(const Tensor& input_data, const Tensor& filter_data,
                       const Tensor& bias_data,
                       const std::vector<string>& fused_ops, Tensor* out)>;

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

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor.
  static void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                          Tensor* output) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void ConvertAndCompare(DataType dtype, const Tensor& tensor,
                         const Tensor& mkl_meta_tensor,
                         const Tensor& expected) {
    Tensor output;
    PerformConversion(dtype, tensor, mkl_meta_tensor, &output);
    test::ExpectTensorNear<T>(expected, output, 1e-5);
  }

  void ConvertAndCompareIntegral(DataType dtype, const Tensor& tensor,
                                 const Tensor& mkl_meta_tensor,
                                 const Tensor& expected) {
    Tensor output;
    PerformConversion(dtype, tensor, mkl_meta_tensor, &output);
    test::ExpectTensorEqual<T>(expected, output);
  }
  void TestBody() {}

  static void VerifyBiasAddTensorsClose(int depth, int image_width,
                                        int image_height, int image_batch_count,
                                        int filter_size, int filter_count,
                                        const BiasAddGraphRunner& run_default,
                                        const BiasAddGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor image(dtype, {image_batch_count, image_height, image_width, depth});
    image.flat<T>() = image.flat<T>().template setRandom<random_gen_>();

    Tensor filter(dtype, {filter_size, filter_size, depth, filter_count});
    filter.flat<T>() = filter.flat<T>().template setRandom<random_gen_>();

    const int bias_size = filter_count;
    Tensor bias(dtype, {bias_size});
    bias.flat<T>() = bias.flat<T>().template setRandom<random_gen_>();

    Tensor conv_2d;
    Tensor fused_conv_2d;

    run_default(image, filter, bias, &conv_2d);
    run_fused(image, filter, bias, &fused_conv_2d);

    ASSERT_EQ(conv_2d.dtype(), fused_conv_2d.dtype());
    ASSERT_EQ(conv_2d.shape(), fused_conv_2d.shape());

    test::ExpectClose(conv_2d, fused_conv_2d, 1e-5);
  }

  static void VerifyFusedTensorsClose(int depth, int image_width,
                                      int image_height, int image_batch_count,
                                      int filter_size, int filter_count,
                                      int bias_size,
                                      const std::vector<string>& fused_ops,
                                      const FusedGraphRunner& run_default,
                                      const FusedGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor image(dtype, {image_batch_count, image_height, image_width, depth});
    image.flat<T>() = image.flat<T>().template setRandom<random_gen_>();

    Tensor filter(dtype, {filter_size, filter_size, depth, filter_count});
    filter.flat<T>() = filter.flat<T>().template setRandom<random_gen_>();

    Tensor bias(dtype, {bias_size});
    bias.flat<T>() = bias.flat<T>().template setRandom<random_gen_>();

    Tensor conv_2d;
    Tensor fused_conv_2d;

    run_default(image, filter, bias, fused_ops, &conv_2d);
    run_fused(image, filter, bias, fused_ops, &fused_conv_2d);

    ASSERT_EQ(conv_2d.dtype(), fused_conv_2d.dtype());
    ASSERT_EQ(conv_2d.shape(), fused_conv_2d.shape());

    test::ExpectClose(conv_2d, fused_conv_2d, 1e-5);
  }

  static void VerifyFusedMatrixClose(int depth, int batch, int weight_count,
                                     const std::vector<string>& fused_ops,
                                     const FusedGraphRunner& run_default,
                                     const FusedGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor input(dtype, {batch, depth});
    input.flat<T>() = input.flat<T>().template setRandom<random_gen_>();

    Tensor weight(dtype, {depth, weight_count});
    weight.flat<T>() = weight.flat<T>().template setRandom<random_gen_>();

    Tensor bias(dtype, {weight_count});
    bias.flat<T>() = bias.flat<T>().template setRandom<random_gen_>();

    Tensor output;
    Tensor fused_output;

    run_default(input, weight, bias, fused_ops, &output);
    run_fused(input, weight, bias, fused_ops, &fused_output);

    ASSERT_EQ(output.dtype(), fused_output.dtype());
    ASSERT_EQ(output.shape(), fused_output.shape());

    test::ExpectClose(output, fused_output, 1e-5);
  }

 private:
  using random_gen_ = Eigen::internal::NormalRandomGenerator<T>;
};

// Testing MKL's fused convolution ops

template <typename T>
class MklFusedConv2DOpTest : public OpsTestBase {
 protected:
  static constexpr int kDepth = 3;
  static constexpr int kImageWidth = 32;
  static constexpr int kImageHeight = 32;
  static constexpr int kImageBatchCount = 8;

  void RunConv2DUnfused(const Tensor& input_data, const Tensor& filter_data,
                        const Tensor& bias_data,
                        const std::vector<string>& fused_ops, Tensor* output,
                        int stride = 1) {
    auto root = tensorflow::Scope::NewRootScope();
    auto input_data_op =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output next_op = ops::Conv2D(
        root.WithOpName("conv"), input_data_op,
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, "SAME");

    string last_op = "";
    if (std::find(fused_ops.begin(), fused_ops.end(), "BiasAdd") !=
        fused_ops.end()) {
      last_op = "with_bias";
      next_op = ops::BiasAdd(
          root.WithOpName(last_op), next_op,
          ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));
    }

    if (std::find(fused_ops.begin(), fused_ops.end(), "Add") !=
        fused_ops.end()) {
      last_op = "with_add";
      next_op = ops::AddN(root.WithOpName("with_add"),
                          std::initializer_list<Input>{next_op, input_data_op});
    }

    if (std::find(fused_ops.begin(), fused_ops.end(), "Relu") !=
        fused_ops.end()) {
      last_op = "with_relu";
      next_op = ops::Relu(root.WithOpName(last_op), next_op);
    }

    if (std::find(fused_ops.begin(), fused_ops.end(), "Relu6") !=
        fused_ops.end()) {
      last_op = "with_relu6";
      next_op = ops::Relu6(root.WithOpName(last_op), next_op);
    }

    if (std::find(fused_ops.begin(), fused_ops.end(), "Elu") !=
        fused_ops.end()) {
      last_op = "with_elu";
      next_op = ops::Elu(root.WithOpName(last_op), next_op);
    }

    CommonTestUtilities<T>::RunAndFetch(root, last_op, output);
  }

  void RunMklFusedConv2DOp(const Tensor& image, const Tensor& filter,
                           const std::vector<Tensor>& args,
                           const std::vector<string>& fused_ops, Tensor* output,
                           int stride = 1) {
    DataType dtype = DataTypeToEnum<T>::v();
    int num_args = static_cast<int>(args.size());

    TF_EXPECT_OK(NodeDefBuilder("fused_conv_op", "_MklFusedConv2D")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(num_args, dtype))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(num_args, DT_UINT8))
                     .Attr("T", dtype)
                     .Attr("num_args", num_args)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", "SAME")
                     .Attr("fused_ops", fused_ops)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(image.shape(), image.flat<T>());
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    for (const Tensor& arg : args)
      AddInputFromArray<T>(arg.shape(), arg.flat<T>());
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    for (const Tensor& arg : args)
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expected results
    const Tensor& output_tensor = *GetOutput(0);
    // Index 2 will need to be changed if the number of outputs produced
    // by MklConv2D change.
    const Tensor& output_meta_tensor = *GetOutput(2);
    CommonTestUtilities<T> test_util;
    test_util.PerformConversion(dtype, output_tensor, output_meta_tensor,
                                output);
  }

  // Verifies computing unfused ops in a graph is identical to FusedConv2D.
  void VerifyFusedConv2D(int filter_size, int filter_count,
                         const std::vector<string>& fused_ops,
                         int depth = kDepth, int image_width = kImageWidth,
                         int image_height = kImageHeight,
                         int image_batch_count = kImageBatchCount) {
    const FusedGraphRunner run_default =
        [this](const Tensor& input_data, const Tensor& filter_data,
               const Tensor& bias_data, const std::vector<string>& fused_ops,
               Tensor* out) {
          RunConv2DUnfused(input_data, filter_data, bias_data, fused_ops, out);
        };

    const FusedGraphRunner run_fused =
        [this](const Tensor& input_data, const Tensor& filter_data,
               const Tensor& bias_data, const std::vector<string>& fused_ops,
               Tensor* out) {
          std::vector<Tensor> fused_input = {bias_data};
          if (std::find(fused_ops.begin(), fused_ops.end(), "Add") !=
              fused_ops.end()) {
            fused_input.push_back(input_data);
          }
          RunMklFusedConv2DOp(input_data, filter_data, fused_input, fused_ops,
                              out);
        };

    const int bias_size = filter_count;
    CommonTestUtilities<T>::VerifyFusedTensorsClose(
        depth, image_width, image_height, image_batch_count, filter_size,
        filter_count, bias_size, fused_ops, run_default, run_fused);
  }
};

template <typename T>
class MklFusedConv2DWithBiasOpTest : public MklFusedConv2DOpTest<T> {};

TYPED_TEST_CASE_P(MklFusedConv2DWithBiasOpTest);

// -------------------------------------------------------------------------- //
// Conv2D + BiasAdd + {Activation}                                            //
// -------------------------------------------------------------------------- //

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolution) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolution) {
  const int kFilterSize = 3;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolutionAndRelu) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Relu"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolutionAndRelu) {
  const int kFilterSize = 3;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Relu"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolutionAndRelu6) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Relu6"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolutionAndRelu6) {
  const int kFilterSize = 3;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Relu6"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolutionAndElu) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Elu"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolutionAndElu) {
  const int kFilterSize = 3;
  const int kFilterCount = 12;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Elu"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolutionAndAdd) {
  const int kFilterSize = 1;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Add"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolutionAndAdd) {
  const int kFilterSize = 3;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Add"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolutionAndAddRelu) {
  const int kFilterSize = 1;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount,
                          {"BiasAdd", "Add", "Relu"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolutionAndAddRelu) {
  const int kFilterSize = 3;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount,
                          {"BiasAdd", "Add", "Relu"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolutionAndAddRelu6) {
  const int kFilterSize = 1;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount,
                          {"BiasAdd", "Add", "Relu6"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolutionAndAddRelu6) {
  const int kFilterSize = 3;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount,
                          {"BiasAdd", "Add", "Relu6"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, OneByOneConvolutionAndAddElu) {
  const int kFilterSize = 1;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Add", "Elu"});
}

TYPED_TEST_P(MklFusedConv2DWithBiasOpTest, SpatialConvolutionAndAddElu) {
  const int kFilterSize = 3;
  const int kFilterCount = 3;
  this->VerifyFusedConv2D(kFilterSize, kFilterCount, {"BiasAdd", "Add", "Elu"});
}

REGISTER_TYPED_TEST_CASE_P(
    MklFusedConv2DWithBiasOpTest, OneByOneConvolution, SpatialConvolution,
    OneByOneConvolutionAndRelu, SpatialConvolutionAndRelu,
    OneByOneConvolutionAndRelu6, SpatialConvolutionAndRelu6,
    OneByOneConvolutionAndElu, SpatialConvolutionAndElu,
    OneByOneConvolutionAndAdd, SpatialConvolutionAndAdd,
    OneByOneConvolutionAndAddRelu, SpatialConvolutionAndAddRelu,
    OneByOneConvolutionAndAddRelu6, SpatialConvolutionAndAddRelu6,
    OneByOneConvolutionAndAddElu, SpatialConvolutionAndAddElu);

using MklFusedBiasAddDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, MklFusedConv2DWithBiasOpTest,
                              MklFusedBiasAddDataTypes);

// Testing MKL's fused depthwise convolution ops
template <typename T>
class MklFusedDepthwiseConv2DOpTest : public OpsTestBase {
 protected:
  static constexpr int kDepth = 3;
  static constexpr int kImageWidth = 32;
  static constexpr int kImageHeight = 32;
  static constexpr int kImageBatchCount = 8;

  void RunDepthwiseConv2DUnfused(const Tensor& input_data,
                                 const Tensor& filter_data,
                                 const Tensor& bias_data,
                                 const std::vector<string>& fused_ops,
                                 Tensor* output, int stride = 1) {
    auto root = tensorflow::Scope::NewRootScope();
    auto input_data_op =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output next_op = ops::DepthwiseConv2dNative(
        root.WithOpName("depthwise_conv"), input_data_op,
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, "SAME");

    string last_op = "";
    if (std::find(fused_ops.begin(), fused_ops.end(), "BiasAdd") !=
        fused_ops.end()) {
      last_op = "with_bias";
      next_op = ops::BiasAdd(
          root.WithOpName(last_op), next_op,
          ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));
    }

    if (std::find(fused_ops.begin(), fused_ops.end(), "Relu") !=
        fused_ops.end()) {
      last_op = "with_relu";
      next_op = ops::Relu(root.WithOpName(last_op), next_op);
    }

    if (std::find(fused_ops.begin(), fused_ops.end(), "Relu6") !=
        fused_ops.end()) {
      last_op = "with_relu6";
      next_op = ops::Relu6(root.WithOpName(last_op), next_op);
    }

    if (std::find(fused_ops.begin(), fused_ops.end(), "Elu") !=
        fused_ops.end()) {
      last_op = "with_elu";
      next_op = ops::Elu(root.WithOpName(last_op), next_op);
    }

    CommonTestUtilities<T>::RunAndFetch(root, last_op, output);
  }

  void RunMklFusedDepthwiseConv2DOp(const Tensor& image, const Tensor& filter,
                                    const std::vector<Tensor>& args,
                                    const std::vector<string>& fused_ops,
                                    Tensor* output, int stride = 1) {
    DataType dtype = DataTypeToEnum<T>::v();
    int num_args = static_cast<int>(args.size());

    TF_EXPECT_OK(NodeDefBuilder("fused_depthwise_conv_op",
                                "_MklFusedDepthwiseConv2dNative")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(num_args, dtype))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(num_args, DT_UINT8))
                     .Attr("T", dtype)
                     .Attr("num_args", num_args)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", "SAME")
                     .Attr("fused_ops", fused_ops)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(image.shape(), image.flat<T>());
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    for (const Tensor& arg : args)
      AddInputFromArray<T>(arg.shape(), arg.flat<T>());
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    for (const Tensor& arg : args)
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expected results
    const Tensor& output_tensor = *GetOutput(0);
    // Index 2 will need to be changed if the number of outputs produced
    // by MklDepthwiseConv2D change.
    const Tensor& output_meta_tensor = *GetOutput(2);
    CommonTestUtilities<T> test_util;
    test_util.PerformConversion(dtype, output_tensor, output_meta_tensor,
                                output);
  }

  // Verifies computing unfused ops in a graph is identical to
  // FusedDepthwiseConv2D.
  void VerifyFusedDepthwiseConv2D(int filter_size, int filter_count,
                                  int bias_size,
                                  const std::vector<string>& fused_ops,
                                  int depth = kDepth,
                                  int image_width = kImageWidth,
                                  int image_height = kImageHeight,
                                  int image_batch_count = kImageBatchCount) {
    const FusedGraphRunner run_default =
        [this](const Tensor& input_data, const Tensor& filter_data,
               const Tensor& bias_data, const std::vector<string>& fused_ops,
               Tensor* out) {
          RunDepthwiseConv2DUnfused(input_data, filter_data, bias_data,
                                    fused_ops, out);
        };

    const FusedGraphRunner run_fused =
        [this](const Tensor& input_data, const Tensor& filter_data,
               const Tensor& bias_data, const std::vector<string>& fused_ops,
               Tensor* out) {
          std::vector<Tensor> fused_input = {bias_data};
          RunMklFusedDepthwiseConv2DOp(input_data, filter_data, fused_input,
                                       fused_ops, out);
        };

    CommonTestUtilities<T>::VerifyFusedTensorsClose(
        depth, image_width, image_height, image_batch_count, filter_size,
        filter_count, bias_size, fused_ops, run_default, run_fused);
  }
};

template <typename T>
class MklFusedDepthwiseConv2DWithBiasOpTest
    : public MklFusedDepthwiseConv2DOpTest<T> {};

TYPED_TEST_SUITE_P(MklFusedDepthwiseConv2DWithBiasOpTest);

// -------------------------------------------------------------------------- //
// DepthwiseConv2D + BiasAdd + {Activation}                                   //
// -------------------------------------------------------------------------- //

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest, OneByOneConvolution) {
  const int kFilterSize = 1;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd"});
}

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest, SpatialConvolution) {
  const int kFilterSize = 3;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd"});
}

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest,
             OneByOneConvolutionAndRelu) {
  const int kFilterSize = 1;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd", "Relu"});
}

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest, SpatialConvolutionAndRelu) {
  const int kFilterSize = 3;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd", "Relu"});
}

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest,
             OneByOneConvolutionAndRelu6) {
  const int kFilterSize = 1;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd", "Relu6"});
}

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest,
             SpatialConvolutionAndRelu6) {
  const int kFilterSize = 3;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd", "Relu6"});
}

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest, OneByOneConvolutionAndElu) {
  const int kFilterSize = 1;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd", "Elu"});
}

TYPED_TEST_P(MklFusedDepthwiseConv2DWithBiasOpTest, SpatialConvolutionAndElu) {
  const int kFilterSize = 3;
  const int kFilterCount = 1;
  const int kBiasSize = 3;
  this->VerifyFusedDepthwiseConv2D(kFilterSize, kFilterCount, kBiasSize,
                                   {"BiasAdd", "Elu"});
}

REGISTER_TYPED_TEST_SUITE_P(
    MklFusedDepthwiseConv2DWithBiasOpTest, OneByOneConvolution,
    SpatialConvolution, OneByOneConvolutionAndRelu, SpatialConvolutionAndRelu,
    OneByOneConvolutionAndRelu6, SpatialConvolutionAndRelu6,
    OneByOneConvolutionAndElu, SpatialConvolutionAndElu);

using MklFusedBiasAddDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, MklFusedDepthwiseConv2DWithBiasOpTest,
                               MklFusedBiasAddDataTypes);

// Testing fusion of pad and convolution

class FusedPadConvOpTest : public OpsTestBase {
 public:
  template <typename T>
  void Run(DataType dtype, Tensor& image, Tensor& filter, Tensor& padding,
           Tensor& expected, const string data_format) {
    const int stride = 1;

    // Create a fused pad+conv2d node
    TF_EXPECT_OK(NodeDefBuilder("fused_pad_conv_op", "_MklPadWithConv2D")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(dtype))     // Filter
                     .Input(FakeInput(DT_INT32))  // Padding
                     .Input(FakeInput(DT_UINT8))  // MKl second tensor
                     .Input(FakeInput(DT_UINT8))  // MKl second tensor
                     .Input(FakeInput(DT_UINT8))  // MKl second tensor
                     .Attr("padding", "VALID")
                     .Attr("data_format", data_format)
                     .Attr("T", dtype)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    // Setting up inputs and execute
    AddInputFromArray<T>(image.shape(), image.flat<T>());
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    AddInputFromArray<int32>(padding.shape(), padding.flat<int32>());
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expected results
    const Tensor& first = *GetOutput(0);
    const Tensor& second = *GetOutput(2);
    CommonTestUtilities<T> test_util;
    test_util.ConvertAndCompare(dtype, first, second, expected);
  }
};

TEST_F(FusedPadConvOpTest, PaddingConvTest) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  Tensor image(DT_FLOAT, {image_batch_count, image_height, image_width, depth});
  test::FillValues<float>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  const int kFilterSize = 3;
  const int kFilterCount = 1;
  Tensor filter(DT_FLOAT, {kFilterSize, kFilterSize, depth, kFilterCount});
  test::FillValues<float>(&filter, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  const int padding_height = 4;
  const int padding_width = 2;
  Tensor padding(DT_INT32, {padding_height, padding_width});
  test::FillValues<int32>(&padding, {0, 0, 3, 4, 1, 2, 0, 0});

  Tensor expected(DT_FLOAT, TensorShape({1, 8, 5, 1}));
  test::FillValues<float>(
      &expected,
      {0,  0,   0,   0,   0,   24, 42,  60,  33,  12,  105, 150, 183, 95,
       32, 235, 312, 357, 178, 56, 187, 234, 261, 121, 32,  106, 126, 138,
       59, 12,  0,   0,   0,   0,  0,   0,   0,   0,   0,   0});

  Run<float>(DT_FLOAT, image, filter, padding, expected, "NHWC");
}

TEST_F(FusedPadConvOpTest, PaddingConvTestNchw) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  Tensor image(DT_FLOAT, {image_batch_count, depth, image_height, image_width});
  test::FillValues<float>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  const int kFilterSize = 3;
  const int kFilterCount = 1;
  Tensor filter(DT_FLOAT, {kFilterSize, kFilterSize, depth, kFilterCount});
  test::FillValues<float>(&filter, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  const int padding_height = 4;
  const int padding_width = 2;
  Tensor padding(DT_INT32, {padding_height, padding_width});
  test::FillValues<int32>(&padding, {0, 0, 0, 0, 3, 4, 1, 2});

  Tensor expected(DT_FLOAT, TensorShape({1, 1, 8, 5}));
  test::FillValues<float>(
      &expected,
      {0,  0,   0,   0,   0,   24, 42,  60,  33,  12,  105, 150, 183, 95,
       32, 235, 312, 357, 178, 56, 187, 234, 261, 121, 32,  106, 126, 138,
       59, 12,  0,   0,   0,   0,  0,   0,   0,   0,   0,   0});

  Run<float>(DT_FLOAT, image, filter, padding, expected, "NCHW");
}

class FilterCacheTest : public OpsTestBase {
 public:
  template <typename T>
  void Run(DataType dtype, Tensor& image, Tensor& filter, Tensor& expected,
           const bool is_filter_const) {
    const int stride = 1;

    TF_EXPECT_OK(NodeDefBuilder("conv2d_filter_cache", "_MklConv2D")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(dtype))     // Filter
                     .Input(FakeInput(DT_UINT8))  // MKl second tensor
                     .Input(FakeInput(DT_UINT8))  // MKl second tensor
                     .Attr("padding", "VALID")
                     .Attr("data_format", "NHWC")
                     .Attr("is_filter_const", is_filter_const)
                     .Attr("T", dtype)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    // Setting up inputs and execute
    AddInputFromArray<T>(image.shape(), image.flat<T>());
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

    TF_ASSERT_OK(RunOpKernel());

    // Compare outputs to expected results
    const Tensor& output = *GetOutput(0);
    const Tensor& output_layout = *GetOutput(2);
    CommonTestUtilities<T> conv_comp;
    conv_comp.ConvertAndCompare(dtype, output, output_layout, expected);

    // TODO(bhavanis): For now, we rely on internal performance tests to
    // determine if filter data is being cached and reused.
    // However, we still need to add a check here to determine if this is
    // still the case by inspecting the contents of the persistent tensor.
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expected results
    const Tensor& output_new = *GetOutput(0);
    const Tensor& output_layout_new = *GetOutput(2);
    CommonTestUtilities<T> conv_comp_new;
    conv_comp_new.ConvertAndCompare(dtype, output_new, output_layout_new,
                                    expected);
  }
};

TEST_F(FilterCacheTest, Conv2DFilterCacheTest) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  Tensor image(DT_FLOAT, {image_batch_count, image_height, image_width, depth});
  test::FillValues<float>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  const int kFilterSize = 3;
  const int kFilterCount = 1;
  Tensor filter(DT_FLOAT, {kFilterSize, kFilterSize, depth, kFilterCount});
  test::FillValues<float>(&filter, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  Tensor expected(DT_FLOAT, TensorShape({1, 1, 2, 1}));
  test::FillValues<float>(&expected, {312, 357});

  Run<float>(DT_FLOAT, image, filter, expected, true);
}

// Testing fusion of MatMul and BiasAdd
template <typename T>
class MklFusedMatMulOpTest : public OpsTestBase {
 protected:
  void VerifyFusedMatMul(const int kBatch, const int kInputChannel,
                         const int kOutputChannel,
                         const std::vector<string>& fused_ops) {
    const FusedGraphRunner run_default =
        [this](const Tensor& input, const Tensor& weight, const Tensor& bias,
               const std::vector<string>& fused_ops, Tensor* output) {
          auto root = tensorflow::Scope::NewRootScope();
          auto input_op =
              ops::Const(root.WithOpName("input"), Input::Initializer(input));
          Output next_op = ops::MatMul(root.WithOpName("matmul"), input_op,
                                       ops::Const(root.WithOpName("weight"),
                                                  Input::Initializer(weight)));

          string last_op = "";
          if (std::find(fused_ops.begin(), fused_ops.end(), "BiasAdd") !=
              fused_ops.end()) {
            last_op = "with_bias";
            next_op = ops::BiasAdd(
                root.WithOpName(last_op), next_op,
                ops::Const(root.WithOpName("bias"), Input::Initializer(bias)));
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Relu") !=
              fused_ops.end()) {
            last_op = "with_relu";
            next_op = ops::Relu(root.WithOpName(last_op), next_op);
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Relu6") !=
              fused_ops.end()) {
            last_op = "with_relu6";
            next_op = ops::Relu6(root.WithOpName(last_op), next_op);
          }

          if (std::find(fused_ops.begin(), fused_ops.end(), "Elu") !=
              fused_ops.end()) {
            last_op = "with_elu";
            next_op = ops::Elu(root.WithOpName(last_op), next_op);
          }

          CommonTestUtilities<T>::RunAndFetch(root, last_op, output);
        };

    const FusedGraphRunner run_fused =
        [this](const Tensor& input, const Tensor& weight, const Tensor& bias,
               const std::vector<string>& fused_ops, Tensor* output) {
          DataType dtype = DataTypeToEnum<T>::v();
          const int num_args = 1;

          TF_EXPECT_OK(NodeDefBuilder("MklFusedMatMul", "_MklFusedMatMul")
                           .Input(FakeInput(dtype))
                           .Input(FakeInput(dtype))
                           .Input(FakeInput(num_args, dtype))
                           .Input(FakeInput(DT_UINT8))
                           .Input(FakeInput(DT_UINT8))
                           .Input(FakeInput(num_args, DT_UINT8))
                           .Attr("T", dtype)
                           .Attr("transpose_a", false)
                           .Attr("transpose_b", false)
                           .Attr("num_args", num_args)
                           .Attr("fused_ops", fused_ops)
                           .Attr("epsilon", 0.0001)
                           .Attr("_kernel", "MklLayoutDependentOp")
                           .Finalize(node_def()));

          TF_EXPECT_OK(InitOp());

          AddInputFromArray<T>(input.shape(), input.flat<T>());
          AddInputFromArray<T>(weight.shape(), weight.flat<T>());
          AddInputFromArray<T>(bias.shape(), bias.flat<T>());
          // Add MKL meta input for input, filter and bias.
          AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
          AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
          AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

          TF_ASSERT_OK(RunOpKernel());

          const Tensor& output_tensor = *GetOutput(0);
          const Tensor& output_meta_tensor = *GetOutput(1);
          CommonTestUtilities<T> test_util;
          test_util.PerformConversion(dtype, output_tensor, output_meta_tensor,
                                      output);
        };

    CommonTestUtilities<T>::VerifyFusedMatrixClose(kInputChannel, kBatch,
                                                   kOutputChannel, fused_ops,
                                                   run_default, run_fused);
  }
};

TYPED_TEST_CASE_P(MklFusedMatMulOpTest);

TYPED_TEST_P(MklFusedMatMulOpTest, WithBias) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel, {"BiasAdd"});
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndRelu) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Relu"});
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndRelu6) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Relu6"});
}

TYPED_TEST_P(MklFusedMatMulOpTest, WithBiasAndElu) {
  const int batch = 3;
  const int input_channel = 4;
  const int output_channel = 5;

  this->VerifyFusedMatMul(batch, input_channel, output_channel,
                          {"BiasAdd", "Elu"});
}

REGISTER_TYPED_TEST_CASE_P(MklFusedMatMulOpTest,  //
                           WithBias,              //
                           WithBiasAndRelu,       //
                           WithBiasAndRelu6,      //
                           WithBiasAndElu);

using MklFusedMatMulDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, MklFusedMatMulOpTest,
                              MklFusedMatMulDataTypes);

// Test the performance of MklFusedMatMul weight cache.
// For the first time B matrix will be reordered and cached which will be
// used for subsequent runs
class MklFusedMatMulCacheTest : public OpsTestBase {};

TEST_F(MklFusedMatMulCacheTest, WeightCached) {
  const int num_args = 1;
  const std::vector<string>& fused_ops = {"BiasAdd"};

  TF_ASSERT_OK(NodeDefBuilder("MklFusedMatMul", "_MklFusedMatMul")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(num_args, DT_FLOAT))
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(num_args, DT_UINT8))
                   .Attr("T", DT_FLOAT)
                   .Attr("transpose_a", false)
                   .Attr("transpose_b", false)
                   .Attr("num_args", num_args)
                   .Attr("fused_ops", fused_ops)
                   .Attr("epsilon", 0.0001)
                   .Attr("_kernel", "MklLayoutDependentOp")
                   .Finalize(node_def()));

  TF_EXPECT_OK(InitOp());
  // The tensor shape of (1,3) is selected to allow the mkldnn expected
  // weight format to be made as OI rather than IO for BS > 1
  // A matrix is:
  // |  1 |  2 |  3 |
  AddInputFromArray<float>(TensorShape({1, 3}), {1, 2, 3});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<float>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  // Bias vector.
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  // Add MKL meta input for input, filter and bias.
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

  int64 start_time = Env::Default()->NowMicros();
  TF_ASSERT_OK(RunOpKernel());
  int64 end_time = Env::Default()->NowMicros();
  int64 total_duration_unopt = end_time - start_time;

  // Final result after Bias addition:
  // | 75 | 82 | 89 | 96 |
  Tensor expected(DT_FLOAT, TensorShape({1, 4}));
  test::FillValues<float>(&expected, {75, 82, 89, 96});

  const Tensor& output = *GetOutput(0);
  const Tensor& mkl_shape_tensor = *GetOutput(1);
  CommonTestUtilities<float> test_util;
  test_util.ConvertAndCompare(DT_FLOAT, output, mkl_shape_tensor, expected);

  // Test for the second time to use the cached weight
  start_time = Env::Default()->NowMicros();
  TF_ASSERT_OK(RunOpKernel());
  end_time = Env::Default()->NowMicros();
  int64 total_duration_opt = end_time - start_time;
  LOG(INFO) << " Time taken by first call : " << total_duration_unopt
            << ", Time taken after Caching : " << total_duration_opt;

  // Cached call should be at least 20% faster.
  EXPECT_LT(total_duration_opt, total_duration_unopt * 0.8);

  // Compare the result with expected result
  CommonTestUtilities<float> test_util_new;
  const Tensor& output_new = *GetOutput(0);
  const Tensor& mkl_shape_tensor_new = *GetOutput(1);
  test_util_new.ConvertAndCompare(DT_FLOAT, output_new, mkl_shape_tensor_new,
                                  expected);
}

class BiasCacheTest : public OpsTestBase {
 public:
  template <typename T>
  void Run(DataType dtype, Tensor& image, Tensor& filter, Tensor& bias,
           Tensor& min_input, Tensor& max_input, Tensor& min_filter,
           Tensor& max_filter, Tensor& min_output, Tensor& max_output,
           Tensor& expected, const bool is_filter_const) {
    const int stride = 1;

    TF_EXPECT_OK(
        NodeDefBuilder("quantized_conv2d_bias_cache",
                       "_MklQuantizedConv2DWithBiasAndReluAndRequantize")
            .Input(FakeInput(dtype))     // Input
            .Input(FakeInput(DT_QINT8))  // Filter
            .Input(FakeInput(DT_FLOAT))  // Bias
            .Input(FakeInput(DT_FLOAT))  // Min-input
            .Input(FakeInput(DT_FLOAT))  // Max-input
            .Input(FakeInput(DT_FLOAT))  // Min-filter
            .Input(FakeInput(DT_FLOAT))  // Max-filter
            .Input(FakeInput(DT_FLOAT))  // Min-output
            .Input(FakeInput(DT_FLOAT))  // Max-output
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Input(FakeInput(DT_UINT8))  // MKL second tensor
            .Attr("Tinput", DT_QUINT8)
            .Attr("Tfilter", DT_QINT8)
            .Attr("Tbias", DT_FLOAT)
            .Attr("T", DT_QINT8)
            .Attr("out_type", DT_QUINT8)
            .Attr("data_format", "NHWC")
            .Attr("strides", {1, stride, stride, 1})
            .Attr("is_filter_const", is_filter_const)
            .Attr("is_bias_const", true)
            .Attr("padding", "VALID")
            .Attr("_kernel", "QuantizedMklOp")
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    // Setting up inputs and execute
    AddInputFromArray<quint8>(image.shape(), image.flat<quint8>());
    AddInputFromArray<qint8>(filter.shape(), filter.flat<qint8>());
    AddInputFromArray<float>(bias.shape(), bias.flat<float>());
    AddInputFromArray<float>(min_input.shape(), min_input.flat<float>());
    AddInputFromArray<float>(max_input.shape(), max_input.flat<float>());
    AddInputFromArray<float>(min_filter.shape(), min_filter.flat<float>());
    AddInputFromArray<float>(max_filter.shape(), max_filter.flat<float>());
    AddInputFromArray<float>(min_output.shape(), min_output.flat<float>());
    AddInputFromArray<float>(max_output.shape(), max_output.flat<float>());
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

    TF_ASSERT_OK(RunOpKernel());

    // Compare outputs to expected results
    const Tensor& output = *GetOutput(0);
    const Tensor& output_layout = *GetOutput(3);
    CommonTestUtilities<quint8> conv_comp;
    conv_comp.ConvertAndCompareIntegral(dtype, output, output_layout, expected);

    // TODO(wenxi): For now, we rely on internal performance tests to
    // determine if filter data is being cached and reused.
    // However, we still need to add a check here to determine if this is
    // still the case by inspecting the contents of the persistent tensor.
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expected results
    const Tensor& output_new = *GetOutput(0);
    const Tensor& output_layout_new = *GetOutput(3);
    CommonTestUtilities<quint8> conv_comp_new;
    conv_comp_new.ConvertAndCompareIntegral(dtype, output_new,
                                            output_layout_new, expected);
  }
};

TEST_F(BiasCacheTest, Conv2DBiasCacheTest) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;

  Tensor image(DT_QUINT8,
               {image_batch_count, image_height, image_width, depth});
  test::FillValues<quint8>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  const int kFilterSize = 3;
  const int kFilterCount = 1;
  Tensor filter(DT_QINT8, {kFilterSize, kFilterSize, depth, kFilterCount});
  test::FillValues<qint8>(&filter, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  Tensor bias(DT_FLOAT, {kFilterCount});
  test::FillValues<float>(&bias, {1});

  Tensor min_input(DT_FLOAT, {1});
  test::FillValues<float>(&min_input, {1});

  Tensor max_input(DT_FLOAT, {1});
  test::FillValues<float>(&max_input, {1});

  Tensor min_filter(DT_FLOAT, {1});
  test::FillValues<float>(&min_filter, {1});

  Tensor max_filter(DT_FLOAT, {1});
  test::FillValues<float>(&max_filter, {1});

  Tensor min_output(DT_FLOAT, {1});
  test::FillValues<float>(&min_output, {1});

  Tensor max_output(DT_FLOAT, {1});
  test::FillValues<float>(&max_output, {1});

  Tensor expected(DT_QUINT8, TensorShape({1, 1, 2, 1}));
  test::FillValues<quint8>(&expected, {255, 255});

  Run<float>(DT_QUINT8, image, filter, bias, min_input, max_input, min_filter,
             max_filter, min_output, max_output, expected, true);
}

// Testing fusion of pad and fusedconv2d
template <typename T>
class MklPadWithFusedConv2DOpTest : public OpsTestBase {
 protected:
  static constexpr int kDepth = 3;
  static constexpr int kImageWidth = 30;
  static constexpr int kImageHeight = 28;
  static constexpr int kImageBatchCount = 8;

  // 0: top pad, 1: bottom pad, 2: left pad, 3: right pad
  int padding_list_[4];

  // Verifies that computing Pad+Conv2D+BiasAdd in a graph is identical to
  // FusedConv2D.
  void VerifyPadAndConv2DWithBias(int filter_size, int filter_count,
                                  int depth = kDepth,
                                  int image_width = kImageWidth,
                                  int image_height = kImageHeight,
                                  int image_batch_count = kImageBatchCount) {
    const BiasAddGraphRunner run_default = [this](const Tensor& input_data,
                                                  const Tensor& filter_data,
                                                  const Tensor& bias_data,
                                                  Tensor* out) {
      RunMklPadWithFusedConv2DAndBias(input_data, filter_data, bias_data, out);
    };

    const BiasAddGraphRunner run_fused =
        [this](const Tensor& input_data, const Tensor& filter_data,
               const Tensor& bias_data, Tensor* out) {
          RunMklFusedConv2DWithPadOp(input_data, filter_data, {bias_data},
                                     {"BiasAdd"}, out);
        };

    CommonTestUtilities<T>::VerifyBiasAddTensorsClose(
        depth, image_width, image_height, image_batch_count, filter_size,
        filter_count, run_default, run_fused);
  }

  // Verifies that computing Pad+Conv2D+BiasAdd+Relu in a graph is identical to
  // FusedConv2D.
  void VerifyPadAndConv2DWithBiasRelu(
      int filter_size, int filter_count, int depth = kDepth,
      int image_width = kImageWidth, int image_height = kImageHeight,
      int image_batch_count = kImageBatchCount) {
    const BiasAddGraphRunner run_default =
        [this](const Tensor& input_data, const Tensor& filter_data,
               const Tensor& bias_data, Tensor* out) {
          RunMklPadWithFusedConv2DAndBiasRelu(input_data, filter_data,
                                              bias_data, out);
        };

    const BiasAddGraphRunner run_fused =
        [this](const Tensor& input_data, const Tensor& filter_data,
               const Tensor& bias_data, Tensor* out) {
          RunMklFusedConv2DWithPadOp(input_data, filter_data, {bias_data},
                                     {"BiasAdd", "Relu"}, out);
        };

    CommonTestUtilities<T>::VerifyBiasAddTensorsClose(
        depth, image_width, image_height, image_batch_count, filter_size,
        filter_count, run_default, run_fused);
  }

  void RunMklPadWithFusedConv2DAndBias(const Tensor& input_data,
                                       const Tensor& filter_data,
                                       const Tensor& bias_data, Tensor* output,
                                       int stride = 1) {
    auto root = tensorflow::Scope::NewRootScope();

    // FusedConv2D only supports NHWC format so we use NHWC here.
    auto padding = ops::Const(root.WithOpName("padding"),
                              {0, 0, padding_list_[0], padding_list_[1],
                               padding_list_[2], padding_list_[3], 0, 0},
                              {4, 2});
    auto pad = ops::Pad(
        root.WithOpName("pad"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        padding);

    auto conv = ops::Conv2D(
        root.WithOpName("conv"), pad,
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, "VALID");

    auto with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), conv,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    CommonTestUtilities<T>::RunAndFetch(root, "with_bias", output);
  }

  void RunMklPadWithFusedConv2DAndBiasRelu(const Tensor& input_data,
                                           const Tensor& filter_data,
                                           const Tensor& bias_data,
                                           Tensor* output, int stride = 1) {
    auto root = tensorflow::Scope::NewRootScope();

    // FusedConv2D only supports NHWC format so we use NHWC here.
    auto padding = ops::Const(root.WithOpName("padding"),
                              {0, 0, padding_list_[0], padding_list_[1],
                               padding_list_[2], padding_list_[3], 0, 0},
                              {4, 2});
    auto pad = ops::Pad(
        root.WithOpName("pad"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        padding);

    auto conv = ops::Conv2D(
        root.WithOpName("conv"), pad,
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, "VALID");

    auto with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), conv,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    auto with_relu = ops::Relu(root.WithOpName("with_relu"), with_bias);

    CommonTestUtilities<T>::RunAndFetch(root, "with_relu", output);
  }

  void RunMklFusedConv2DWithPadOp(const Tensor& image, const Tensor& filter,
                                  const std::vector<Tensor>& args,
                                  const std::vector<string>& fused_ops,
                                  Tensor* output, int stride = 1) {
    DataType dtype = DataTypeToEnum<T>::v();
    const int num_args = static_cast<int>(args.size());
    Tensor padding(DT_INT32, {4, 2});
    test::FillValues<int32>(
        &padding, {0, 0, padding_list_[0], padding_list_[1], padding_list_[2],
                   padding_list_[3], 0, 0});

    TF_EXPECT_OK(NodeDefBuilder("pad_fused_conv_op", "_MklPadWithFusedConv2D")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(num_args, dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(DT_UINT8))
                     .Input(FakeInput(num_args, DT_UINT8))
                     .Input(FakeInput(DT_UINT8))
                     .Attr("T", dtype)
                     .Attr("num_args", num_args)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", "VALID")
                     .Attr("fused_ops", fused_ops)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());

    AddInputFromArray<T>(image.shape(), image.flat<T>());
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    for (const Tensor& arg : args)
      AddInputFromArray<T>(arg.shape(), arg.flat<T>());
    AddInputFromArray<int32>(padding.shape(), padding.flat<int32>());
    // Add MKL meta input for input, filter, pad and agrs.
    for (int i = 0; i < args.size() + 3; ++i)
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
    TF_ASSERT_OK(RunOpKernel());

    // Compare output to expected results
    const Tensor& output_tensor = *GetOutput(0);
    // Index 2 will need to be changed if the number of outputs produced
    // by MklConv2D change.
    const Tensor& output_meta_tensor = *GetOutput(2);
    CommonTestUtilities<T> test_util;
    test_util.PerformConversion(dtype, output_tensor, output_meta_tensor,
                                output);
  }

 public:
  void SetPaddingList(int top, int bottom, int left, int right) {
    padding_list_[0] = top;
    padding_list_[1] = bottom;
    padding_list_[2] = left;
    padding_list_[3] = right;
  }
};

TYPED_TEST_CASE_P(MklPadWithFusedConv2DOpTest);

TYPED_TEST_P(MklPadWithFusedConv2DOpTest, WithBiasAndRoundPad) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->SetPaddingList(2, 2, 1, 1);
  this->VerifyPadAndConv2DWithBias(kFilterSize, kFilterCount);
}

TYPED_TEST_P(MklPadWithFusedConv2DOpTest, WithBiasAndPartialPad) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->SetPaddingList(4, 0, 2, 0);
  this->VerifyPadAndConv2DWithBias(kFilterSize, kFilterCount);
}

TYPED_TEST_P(MklPadWithFusedConv2DOpTest, WithBiasReluAndRoundPad) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->SetPaddingList(2, 2, 1, 1);
  this->VerifyPadAndConv2DWithBiasRelu(kFilterSize, kFilterCount);
}

TYPED_TEST_P(MklPadWithFusedConv2DOpTest, WithBiasReluAndPartialPad) {
  const int kFilterSize = 1;
  const int kFilterCount = 12;
  this->SetPaddingList(4, 0, 2, 0);
  this->VerifyPadAndConv2DWithBiasRelu(kFilterSize, kFilterCount);
}

REGISTER_TYPED_TEST_CASE_P(MklPadWithFusedConv2DOpTest,  //
                           WithBiasAndRoundPad,          //
                           WithBiasAndPartialPad,        //
                           WithBiasReluAndRoundPad,      //
                           WithBiasReluAndPartialPad);

using MklPadWithFusedConv2DDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_CASE_P(Test, MklPadWithFusedConv2DOpTest,
                              MklPadWithFusedConv2DDataTypes);

}  // namespace tensorflow
#endif  // INTEL_MKL
