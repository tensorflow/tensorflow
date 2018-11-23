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

// Helper class for converting MKL tesnors to TF tensors and comparing to
// expected values

static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

class ConvMklToTF : public OpsTestBase {
 public:
  template <typename T>
  void ConvertAndCompare(DataType dtype, const Tensor& first,
                         const Tensor& second, const Tensor& expected) {
    // Create an MKL to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // Mkl second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(first.shape(), first.flat<T>());
    AddInputFromArray<uint8>(second.shape(), second.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    const Tensor& output = *GetOutput(0);
    test::ExpectTensorNear<T>(expected, output, 1e-5);
  }
  void TestBody(){};
};

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
                     .Attr("_kernel", "MklOp")
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
    ConvMklToTF conv_comp;
    conv_comp.ConvertAndCompare<T>(dtype, first, second, expected);
  }
};

TEST_F(FusedPadConvOpTest, PaddingConvTest) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  Tensor image(DT_FLOAT, {image_batch_count, image_height, image_width, depth});
  test::FillValues<float>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  const int filter_size = 3;
  const int filter_count = 1;
  Tensor filter(DT_FLOAT, {filter_size, filter_size, depth, filter_count});
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

  const int filter_size = 3;
  const int filter_count = 1;
  Tensor filter(DT_FLOAT, {filter_size, filter_size, depth, filter_count});
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
}  // namespace tensorflow
#endif  // INTEL_MKL
