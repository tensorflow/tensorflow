/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

struct ConvParametersPeer {
  template <typename T>
  bool ShouldIncludeWinogradNonfusedAlgoPreCudnn7() {
    return params.ShouldIncludeWinogradNonfusedAlgoPreCudnn7<T>();
  }

  ConvParameters params;
};

TEST(ConvParameters, WinogradNonfusedAlgoSize) {
  ConvParametersPeer conv_params_small = {{
      1,            // batch
      32,           // in_depths
      {{300,        // in_rows
        300}},      // in_cols
      FORMAT_NCHW,  // compute_data_format
      128,          // out_depths
      {{3,          // filter_rows
        3}},        // filter_cols
      {{1,          // dilation_rows
        1}},        // dilation_cols
      {{1,          // stride_rows
        1}},        // stride_cols
      {{0,          // padding_rows
        0}},        // padding_cols
      DT_FLOAT,     // tensor datatype
      0,            // device_id
  }};
  EXPECT_TRUE(
      conv_params_small.ShouldIncludeWinogradNonfusedAlgoPreCudnn7<float>());

  ConvParametersPeer conv_params_large = {{
      1,            // batch
      128,          // in_depths
      {{300,        // in_rows
        300}},      // in_cols
      FORMAT_NCHW,  // compute_data_format
      768,          // out_depths
      {{3,          // filter_rows
        3}},        // filter_cols
      {{1,          // dilation_rows
        1}},        // dilation_cols
      {{1,          // stride_rows
        1}},        // stride_cols
      {{0,          // padding_rows
        0}},        // padding_cols
      DT_FLOAT,     // tensor datatype
      0,            // device_id
  }};
  EXPECT_FALSE(
      conv_params_large.ShouldIncludeWinogradNonfusedAlgoPreCudnn7<float>());
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class FusedResizePadConvOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void HandwrittenConv(DataType dtype) {
    const int stride = 1;
    TF_EXPECT_OK(NodeDefBuilder("fused_resize_op", "FusedResizeAndPadConv2D")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(dtype))
                     .Attr("T", dtype)
                     .Attr("resize_align_corners", false)
                     .Attr("mode", "REFLECT")
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", "SAME")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    const int depth = 1;
    const int image_width = 4;
    const int image_height = 3;
    const int image_batch_count = 1;
    // The image matrix is:
    // |  1 |  2 |  3 |  4 |
    // |  5 |  6 |  7 |  8 |
    // |  9 | 10 | 11 | 12 |
    Tensor image(dtype, {image_batch_count, image_height, image_width, depth});
    test::FillValues<T>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // The filter matrix is:
    // | 1 | 4 | 7 |
    // | 2 | 5 | 8 |
    // | 3 | 6 | 9 |
    const int filter_size = 3;
    const int filter_count = 1;
    Tensor filter(dtype, {filter_size, filter_size, depth, filter_count});
    test::FillValues<T>(&filter, {1, 4, 7, 2, 5, 8, 3, 6, 9});

    const int resized_width = image_width;
    const int resized_height = image_height;

    const int top_padding = 0;
    const int bottom_padding = 0;
    const int left_padding = 0;
    const int right_padding = 0;

    AddInputFromArray<T>(image.shape(), image.flat<T>());
    AddInputFromArray<int32>(TensorShape({2}), {resized_height, resized_width});
    AddInputFromArray<int32>(
        TensorShape({4, 2}),
        {0, 0, top_padding, bottom_padding, left_padding, right_padding, 0, 0});
    AddInputFromArray<T>(filter.shape(), filter.flat<T>());
    TF_ASSERT_OK(RunOpKernel());

    // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
    // the input set to zero because we're using the 'SAME' padding mode.
    // The calculations behind the expected output are:
    // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)=105
    // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)=150
    // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)=183
    // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)=95
    // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)=235
    // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
    // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
    // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)=178
    // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)=187
    // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)=234
    // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)=261
    // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
    // This means we should end up with this matrix:
    // |  105  |  150  |  183  |   95  |
    // |  235  |  312  |  357  |  178  |
    // |  187  |  234  |  261  |  121  |
    const int expected_width = image_width;
    const int expected_height = image_height * filter_count;
    Tensor expected(dtype, TensorShape({image_batch_count, expected_height,
                                        expected_width, filter_count}));
    test::FillValues<T>(
        &expected, {105, 150, 183, 95, 235, 312, 357, 178, 187, 234, 261, 121});
    const Tensor& output = *GetOutput(0);
    test::ExpectTensorNear<T>(expected, output, 1e-5);
  }

  template <typename T>
  void CompareFusedAndSeparate(int input_width, int input_height,
                               int input_depth, int resize_width,
                               int resize_height, int y_padding, int x_padding,
                               int filter_size, int filter_count,
                               bool resize_align_corners,
                               const string& pad_mode, int stride,
                               const string& padding, DataType dtype) {
    Scope root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT,
                      TensorShape({1, input_height, input_width, input_depth}));
    test::FillIota<float>(&input_data, 1.0f);
    Output input =
        Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output casted_input = Cast(root.WithOpName("casted_input"), input, dtype);

    Tensor filter_data(DT_FLOAT, TensorShape({filter_size, filter_size,
                                              input_depth, filter_count}));
    test::FillIota<float>(&filter_data, 1.0f);
    Output filter =
        Const(root.WithOpName("filter"), Input::Initializer(filter_data));
    Output casted_filter =
        Cast(root.WithOpName("casted_filter"), filter, dtype);

    Output resize_size =
        Const(root.WithOpName("resize_size"), {resize_height, resize_width});
    Output resize =
        ResizeBilinear(root.WithOpName("resize"), input, resize_size,
                       ResizeBilinear::AlignCorners(resize_align_corners));
    // Bilinear resize only output float, cast it to dtype to match the input.
    Output casted_resize = Cast(root.WithOpName("cast"), resize, dtype);
    Output paddings =
        Const(root.WithOpName("paddings"),
              {{0, 0}, {y_padding, y_padding}, {x_padding, x_padding}, {0, 0}});
    Output mirror_pad = MirrorPad(root.WithOpName("mirror_pad"), casted_resize,
                                  paddings, pad_mode);
    Output conv = Conv2D(root.WithOpName("conv"), mirror_pad, casted_filter,
                         {1, stride, stride, 1}, padding);

    Output fused_conv = FusedResizeAndPadConv2D(
        root.WithOpName("fused_conv"), casted_input, resize_size, paddings,
        casted_filter, pad_mode, {1, stride, stride, 1}, padding,
        FusedResizeAndPadConv2D::ResizeAlignCorners(resize_align_corners));

    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {"conv"}, {}, &unfused_tensors));

    std::vector<Tensor> fused_tensors;
    TF_ASSERT_OK(session->Run({}, {"fused_conv"}, {}, &fused_tensors));

    test::ExpectClose(unfused_tensors[0], fused_tensors[0]);
  }

  template <typename T>
  void CompareFusedPadOnlyAndSeparate(int input_width, int input_height,
                                      int input_depth, int y_padding,
                                      int x_padding, int filter_size,
                                      int filter_count, const string& pad_mode,
                                      int stride, const string& padding,
                                      DataType dtype) {
    Scope root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT,
                      TensorShape({1, input_height, input_width, input_depth}));
    test::FillIota<float>(&input_data, 1.0f);
    Output input =
        Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output casted_input = Cast(root.WithOpName("casted_input"), input, dtype);

    Tensor filter_data(DT_FLOAT, TensorShape({filter_size, filter_size,
                                              input_depth, filter_count}));
    test::FillIota<float>(&filter_data, 1.0f);
    Output filter =
        Const(root.WithOpName("filter"), Input::Initializer(filter_data));
    Output casted_filter =
        Cast(root.WithOpName("casted_filter"), filter, dtype);

    Output paddings =
        Const(root.WithOpName("paddings"),
              {{0, 0}, {y_padding, y_padding}, {x_padding, x_padding}, {0, 0}});
    Output mirror_pad = MirrorPad(root.WithOpName("mirror_pad"), casted_input,
                                  paddings, pad_mode);
    Output conv = Conv2D(root.WithOpName("conv"), mirror_pad, casted_filter,
                         {1, stride, stride, 1}, padding);

    Output fused_conv = FusedPadConv2D(
        root.WithOpName("fused_conv"), casted_input, paddings, casted_filter,
        pad_mode, {1, stride, stride, 1}, padding);

    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {"conv"}, {}, &unfused_tensors));

    std::vector<Tensor> fused_tensors;
    TF_ASSERT_OK(session->Run({}, {"fused_conv"}, {}, &fused_tensors));

    test::ExpectClose(unfused_tensors[0], fused_tensors[0]);
  }
};

TEST_F(FusedResizePadConvOpTest, HandwrittenConvHalf) {
  HandwrittenConv<Eigen::half>(DT_HALF);
}

TEST_F(FusedResizePadConvOpTest, HandwrittenConvFloat) {
  HandwrittenConv<float>(DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, HandwrittenConvDouble) {
  HandwrittenConv<double>(DT_DOUBLE);
}

TEST_F(FusedResizePadConvOpTest, IdentityComparativeHalf) {
  CompareFusedAndSeparate<Eigen::half>(10, 10, 1, 10, 10, 0, 0, 1, 1, false,
                                       "REFLECT", 1, "SAME", DT_HALF);
}

TEST_F(FusedResizePadConvOpTest, IdentityComparativeFloat) {
  CompareFusedAndSeparate<float>(10, 10, 1, 10, 10, 0, 0, 1, 1, false,
                                 "REFLECT", 1, "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, IdentityComparativeDouble) {
  CompareFusedAndSeparate<double>(10, 10, 1, 10, 10, 0, 0, 1, 1, false,
                                  "REFLECT", 1, "SAME", DT_DOUBLE);
}

TEST_F(FusedResizePadConvOpTest, ConvOnlyComparative) {
  CompareFusedAndSeparate<float>(10, 10, 3, 10, 10, 0, 0, 4, 4, false,
                                 "REFLECT", 1, "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeOnlyComparative) {
  CompareFusedAndSeparate<float>(10, 10, 1, 20, 20, 0, 0, 1, 1, false,
                                 "REFLECT", 1, "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeAndConvComparative) {
  CompareFusedAndSeparate<float>(2, 2, 4, 4, 2, 0, 0, 2, 2, false, "REFLECT", 1,
                                 "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeAlignAndConvComparative) {
  CompareFusedAndSeparate<float>(2, 2, 4, 4, 2, 0, 0, 2, 2, true, "REFLECT", 1,
                                 "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeAndConvStridedComparative) {
  CompareFusedAndSeparate<float>(2, 2, 4, 4, 2, 0, 0, 2, 2, false, "REFLECT", 2,
                                 "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeAlignAndConvValidComparative) {
  CompareFusedAndSeparate<float>(2, 2, 4, 4, 2, 0, 0, 2, 2, true, "REFLECT", 1,
                                 "VALID", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, PadOnlyComparative) {
  CompareFusedAndSeparate<float>(4, 4, 1, 4, 4, 2, 2, 1, 1, false, "REFLECT", 1,
                                 "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, PadOnlyWithChannelsComparative) {
  CompareFusedAndSeparate<float>(4, 4, 3, 4, 4, 2, 2, 1, 1, false, "REFLECT", 1,
                                 "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeAndPadComparative) {
  CompareFusedAndSeparate<float>(4, 4, 1, 6, 6, 2, 2, 1, 1, false, "REFLECT", 1,
                                 "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, PadOnlySymmetricComparative) {
  CompareFusedAndSeparate<float>(4, 4, 1, 4, 4, 2, 2, 1, 1, false, "SYMMETRIC",
                                 1, "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeAndPadSymmetricComparative) {
  CompareFusedAndSeparate<float>(4, 4, 3, 6, 6, 2, 2, 1, 1, false, "SYMMETRIC",
                                 1, "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, ResizeAndPadSymmetricComparativeLarge) {
  CompareFusedAndSeparate<float>(1000, 1000, 3, 1006, 1006, 2, 2, 1, 1, false,
                                 "SYMMETRIC", 1, "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, NoResizeIdentityComparativeHalf) {
  CompareFusedPadOnlyAndSeparate<Eigen::half>(10, 10, 1, 0, 0, 1, 1, "REFLECT",
                                              1, "SAME", DT_HALF);
}

TEST_F(FusedResizePadConvOpTest, NoResizeIdentityComparativeFloat) {
  CompareFusedPadOnlyAndSeparate<float>(10, 10, 1, 0, 0, 1, 1, "REFLECT", 1,
                                        "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, NoResizeIdentityComparativeDouble) {
  CompareFusedPadOnlyAndSeparate<double>(10, 10, 1, 0, 0, 1, 1, "REFLECT", 1,
                                         "SAME", DT_DOUBLE);
}

TEST_F(FusedResizePadConvOpTest, NoResizeConvOnlyComparative) {
  CompareFusedPadOnlyAndSeparate<float>(10, 10, 3, 0, 0, 4, 4, "REFLECT", 1,
                                        "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, NoResizePadOnlyComparative) {
  CompareFusedPadOnlyAndSeparate<float>(4, 4, 1, 2, 2, 1, 1, "REFLECT", 1,
                                        "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, NoResizePadOnlyWithChannelsComparative) {
  CompareFusedPadOnlyAndSeparate<float>(4, 4, 3, 2, 2, 1, 1, "REFLECT", 1,
                                        "SAME", DT_FLOAT);
}

TEST_F(FusedResizePadConvOpTest, NoResizePadOnlySymmetricComparative) {
  CompareFusedPadOnlyAndSeparate<float>(4, 4, 1, 2, 2, 1, 1, "SYMMETRIC", 1,
                                        "SAME", DT_FLOAT);
}

class ConvOpTest : public OpsTestBase {
 protected:
  void HandwrittenConv() {
    const int stride = 1;
    TF_EXPECT_OK(NodeDefBuilder("conv_op", "Conv2D")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DT_FLOAT)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", "SAME")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    const int depth = 1;
    const int image_width = 4;
    const int image_height = 3;
    const int image_batch_count = 1;
    // The image matrix is:
    // |  1 |  2 |  3 |  4 |
    // |  5 |  6 |  7 |  8 |
    // |  9 | 10 | 11 | 12 |
    Tensor image(DT_FLOAT,
                 {image_batch_count, image_height, image_width, depth});
    test::FillValues<float>(&image, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // The filter matrix is:
    // | 1 | 4 | 7 |
    // | 2 | 5 | 8 |
    // | 3 | 6 | 9 |
    const int filter_size = 3;
    const int filter_count = 1;
    Tensor filter(DT_FLOAT, {filter_size, filter_size, depth, filter_count});
    test::FillValues<float>(&filter, {1, 4, 7, 2, 5, 8, 3, 6, 9});

    AddInputFromArray<float>(image.shape(), image.flat<float>());
    AddInputFromArray<float>(filter.shape(), filter.flat<float>());
    TF_ASSERT_OK(RunOpKernel());

    // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
    // the input set to zero because we're using the 'SAME' padding mode.
    // The calculations behind the expected output are:
    // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)=105
    // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)=150
    // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)=183
    // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)=95
    // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)=235
    // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
    // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
    // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)=178
    // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)=187
    // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)=234
    // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)=261
    // (1*7)+(4*8)+(7*0)+(2*11)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
    // This means we should end up with this matrix:
    // |  105  |  150  |  183  |   95  |
    // |  235  |  312  |  357  |  178  |
    // |  187  |  234  |  261  |  121  |
    const int expected_width = image_width;
    const int expected_height = image_height * filter_count;
    Tensor expected(DT_FLOAT, TensorShape({image_batch_count, expected_height,
                                           expected_width, filter_count}));
    test::FillValues<float>(
        &expected, {105, 150, 183, 95, 235, 312, 357, 178, 187, 234, 261, 121});
    const Tensor& output = *GetOutput(0);
    test::ExpectTensorNear<float>(expected, output, 1e-5);
  }

  void AnisotropicStrides() {
    const int stride_width = 3;
    const int stride_height = 1;
    TF_EXPECT_OK(NodeDefBuilder("conv_op", "Conv2D")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DT_FLOAT)
                     .Attr("strides", {1, stride_height, stride_width, 1})
                     .Attr("padding", "VALID")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    const int depth = 1;
    const int image_width = 6;
    const int image_height = 3;
    const int image_batch_count = 1;
    Tensor image(DT_FLOAT,
                 {image_batch_count, image_height, image_width, depth});
    test::FillValues<float>(&image, {
                                        3, 2, 1, -1, -2, -3,  //
                                        4, 3, 2, -2, -3, -4,  //
                                        5, 4, 3, -3, -4, -5,  //
                                    });
    const int filter_size = 2;
    const int filter_count = 1;
    Tensor filter(DT_FLOAT, {filter_size, filter_size, depth, filter_count});
    test::FillValues<float>(&filter, {
                                         1, 2,  //
                                         3, 4,  //
                                     });

    AddInputFromArray<float>(image.shape(), image.flat<float>());
    AddInputFromArray<float>(filter.shape(), filter.flat<float>());
    TF_ASSERT_OK(RunOpKernel());

    const int expected_width = 2;
    const int expected_height = 2;
    Tensor expected(DT_FLOAT, TensorShape({image_batch_count, expected_height,
                                           expected_width, filter_count}));
    test::FillValues<float>(&expected, {31, -23, 41, -33});
    const Tensor& output = *GetOutput(0);
    test::ExpectTensorNear<float>(expected, output, 1e-5);
  }
};

TEST_F(ConvOpTest, HandwrittenConv) { HandwrittenConv(); }

TEST_F(ConvOpTest, AnisotropicStride) { AnisotropicStrides(); }

template <typename T>
class FusedConv2DOpTest : public OpsTestBase {
 protected:
  static constexpr int kDepth = 3;
  static constexpr int kImageWidth = 32;
  static constexpr int kImageHeight = 32;
  static constexpr int kImageBatchCount = 8;

  using BiasAddGraphRunner =
      std::function<void(const Tensor& input_data, const Tensor& filter_data,
                         const Tensor& bias_data, Tensor* out)>;

  using BatchNormGraphRunner = std::function<void(
      const Tensor& input_data, const Tensor& filter_data,
      const Tensor& scale_data, const Tensor& offset_data,
      const Tensor& mean_data, const Tensor& variance_data, Tensor* out)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                   Tensor* output, bool allow_gpu_device,
                   const NodeDef* fetch_node = nullptr) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    if (fetch_node) {
      *graph.add_node() = *fetch_node;
    }

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    std::vector<DeviceAttributes> available_devices;
    TF_ASSERT_OK(session->ListDevices(&available_devices))
        << "Failed to get available session devices";

    // Check if session has an available GPU device.
    const bool has_gpu_device =
        absl::c_any_of(available_devices, [](const DeviceAttributes& device) {
          return device.device_type() == DEVICE_GPU;
        });

    // Some of the `FusedConv2D` fusion types are implemented only for CPU, and
    // in this test we don't want to compare GPU vs CPU numbers, so place all
    // nodes on CPU in this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const string device = place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void RunConv2DWithBias(const Tensor& input_data, const Tensor& filter_data,
                         const Tensor& bias_data, const std::string& padding,
                         const std::vector<int>& explicit_paddings,
                         Tensor* output, bool allow_gpu_device = false,
                         int stride = 1) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::Conv2D conv = ops::Conv2D(
        root.WithOpName("conv"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, padding,
        ops::Conv2D::Attrs().ExplicitPaddings(explicit_paddings));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), conv,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    RunAndFetch(root, "with_bias", output, allow_gpu_device);
  }

  void RunConv2DWithBiasAndActivation(
      const Tensor& input_data, const Tensor& filter_data,
      const Tensor& bias_data, const string& activation_type,
      const std::string& padding, const std::vector<int>& explicit_paddings,
      Tensor* output, bool allow_gpu_device = false, int stride = 1) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::Conv2D conv = ops::Conv2D(
        root.WithOpName("conv"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, padding,
        ops::Conv2D::Attrs().ExplicitPaddings(explicit_paddings));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), conv,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    if (activation_type == "Relu") {
      ops::Relu(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Relu6") {
      ops::Relu6(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Elu") {
      ops::Elu(root.WithOpName("with_activation"), with_bias);
    } else {
      ops::Identity(root.WithOpName("with_activation"), with_bias);
    }

    RunAndFetch(root, "with_activation", output, allow_gpu_device);
  }

  void RunConv2DWithBatchNorm(
      const Tensor& input_data, const Tensor& filter_data,
      const Tensor& scale_data, const Tensor& offset_data,
      const Tensor& mean_data, const Tensor& variance_data,
      const std::string& padding, const std::vector<int>& explicit_paddings,
      Tensor* output, bool allow_gpu_device = false, int stride = 1) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::Conv2D conv = ops::Conv2D(
        root.WithOpName("conv"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, padding,
        ops::Conv2D::Attrs().ExplicitPaddings(explicit_paddings));

    ops::FusedBatchNorm::Attrs attr;
    attr = attr.IsTraining(false);

    ops::FusedBatchNorm with_fused_batch_norm = ops::FusedBatchNorm(
        root.WithOpName("with_fused_batch_norm"), conv,
        ops::Const(root.WithOpName("scale"), Input::Initializer(scale_data)),
        ops::Const(root.WithOpName("offset"), Input::Initializer(offset_data)),
        ops::Const(root.WithOpName("mean"), Input::Initializer(mean_data)),
        ops::Const(root.WithOpName("var"), Input::Initializer(variance_data)),
        attr);

    RunAndFetch(root, "with_fused_batch_norm", output, allow_gpu_device);
  }

  void RunConv2DWithBatchNormAndActivation(
      const Tensor& input_data, const Tensor& filter_data,
      const Tensor& scale_data, const Tensor& offset_data,
      const Tensor& mean_data, const Tensor& variance_data,
      const string& activation_type, const std::string& padding,
      const std::vector<int>& explicit_paddings, Tensor* output,
      bool allow_gpu_device = false, int stride = 1) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::Conv2D conv = ops::Conv2D(
        root.WithOpName("conv"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, padding,
        ops::Conv2D::Attrs().ExplicitPaddings(explicit_paddings));

    ops::FusedBatchNorm::Attrs attr;
    attr = attr.IsTraining(false);

    ops::FusedBatchNorm with_fused_batch_norm = ops::FusedBatchNorm(
        root.WithOpName("with_fused_batch_norm"), conv,
        ops::Const(root.WithOpName("scale"), Input::Initializer(scale_data)),
        ops::Const(root.WithOpName("offset"), Input::Initializer(offset_data)),
        ops::Const(root.WithOpName("mean"), Input::Initializer(mean_data)),
        ops::Const(root.WithOpName("var"), Input::Initializer(variance_data)),
        attr);

    if (activation_type == "Relu") {
      ops::Relu(root.WithOpName("with_activation"), with_fused_batch_norm.y);
    } else if (activation_type == "Relu6") {
      ops::Relu6(root.WithOpName("with_activation"), with_fused_batch_norm.y);
    } else if (activation_type == "Elu") {
      ops::Elu(root.WithOpName("with_activation"), with_fused_batch_norm.y);
    } else {
      ops::Identity(root.WithOpName("with_activation"),
                    with_fused_batch_norm.y);
    }

    RunAndFetch(root, "with_activation", output, allow_gpu_device);
  }

  void RunFusedConv2DOp(const Tensor& input_data, const Tensor& filter_data,
                        const std::vector<Tensor>& args_data,
                        const std::vector<string>& fused_ops,
                        const std::string& padding,
                        const std::vector<int>& explicit_paddings,
                        Tensor* output, bool allow_gpu_device = false,
                        int stride = 1) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();
    int num_args = static_cast<int>(args_data.size());

    Output input =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output filter =
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data));

    std::vector<NodeDefBuilder::NodeOut> args;
    for (int i = 0; i < num_args; ++i) {
      Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                              Input::Initializer(args_data[i]));
      args.emplace_back(arg.name(), 0, dtype);
    }

    NodeDef fused_conv2d;
    TF_EXPECT_OK(NodeDefBuilder("fused_conv", "_FusedConv2D")
                     .Input({input.name(), 0, dtype})
                     .Input({filter.name(), 0, dtype})
                     .Input(args)
                     .Attr("num_args", num_args)
                     .Attr("T", dtype)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", padding)
                     .Attr("explicit_paddings", explicit_paddings)
                     .Attr("fused_ops", fused_ops)
                     .Finalize(&fused_conv2d));

    RunAndFetch(root, fused_conv2d.name(), output, allow_gpu_device,
                &fused_conv2d);
  }

  void VerifyBiasAddTensorsNear(int depth, int image_width, int image_height,
                                int image_batch_count, int filter_size,
                                int filter_count,
                                const BiasAddGraphRunner& run_default,
                                const BiasAddGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor image(dtype, {image_batch_count, image_height, image_width, depth});
    image.flat<T>() = image.flat<T>().setRandom();

    // Add some negative values to filter to properly test Relu.
    Tensor filter(dtype, {filter_size, filter_size, depth, filter_count});
    filter.flat<T>() = filter.flat<T>().setRandom();
    filter.flat<T>() -= filter.flat<T>().constant(static_cast<T>(0.5f));

    const int bias_size = filter_count;
    Tensor bias(dtype, {bias_size});
    bias.flat<T>() = bias.flat<T>().setRandom();
    bias.flat<T>() += bias.flat<T>().constant(static_cast<T>(0.5f));

    Tensor conv_2d;
    Tensor fused_conv_2d;

    run_default(image, filter, bias, &conv_2d);
    run_fused(image, filter, bias, &fused_conv_2d);

    ASSERT_EQ(conv_2d.dtype(), fused_conv_2d.dtype());
    ASSERT_EQ(conv_2d.shape(), fused_conv_2d.shape());

    // NOTE(intel-tf): When filter_size is equal to the input image size,
    // conv2d essentially is element-wise multiplication followed by
    // a full sum reduction, which causes larger numerical error
    // than usual cases.
    if (image_width == filter_size && image_height == filter_size) {
      test::ExpectClose(conv_2d, fused_conv_2d, /*atol=*/1e-4);
    } else {
      test::ExpectClose(conv_2d, fused_conv_2d, /*atol=*/1e-5);
    }
  }

  void VerifyFusedBatchNormTensorsNear(int depth, int image_width,
                                       int image_height, int image_batch_count,
                                       int filter_size, int filter_count,
                                       const BatchNormGraphRunner& run_default,
                                       const BatchNormGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor image(dtype, {image_batch_count, image_height, image_width, depth});
    image.flat<T>() = image.flat<T>().setRandom();

    // Add some negative values to filter to properly test Relu.
    Tensor filter(dtype, {filter_size, filter_size, depth, filter_count});
    filter.flat<T>() = filter.flat<T>().setRandom();
    filter.flat<T>() -= filter.flat<T>().constant(static_cast<T>(0.5f));

    const int scale_size = filter_count;

    Tensor scale(dtype, {scale_size});
    scale.flat<T>() = scale.flat<T>().setRandom();

    Tensor offset(dtype, {scale_size});
    offset.flat<T>() = offset.flat<T>().setRandom();

    Tensor mean(dtype, {scale_size});
    mean.flat<T>() = mean.flat<T>().setRandom();

    Tensor variance(dtype, {scale_size});
    variance.flat<T>() = variance.flat<T>().setRandom();
    variance.flat<T>() += variance.flat<T>().constant(static_cast<T>(0.5f));

    Tensor conv_2d;
    Tensor fused_conv_2d;

    run_default(image, filter, scale, offset, mean, variance, &conv_2d);
    run_fused(image, filter, scale, offset, mean, variance, &fused_conv_2d);

    ASSERT_EQ(conv_2d.dtype(), fused_conv_2d.dtype());
    ASSERT_EQ(conv_2d.shape(), fused_conv_2d.shape());

    // NOTE(intel-tf): When filter_size is equal to the input image size,
    // conv2d essentially is element-wise multiplication followed by
    // a full sum reduction, which causes larger numerical error
    // than usual cases.
    if (image_width == filter_size && image_height == filter_size) {
      test::ExpectClose(conv_2d, fused_conv_2d, /*atol=*/1e-4);
    } else {
      test::ExpectClose(conv_2d, fused_conv_2d, /*atol=*/1e-5);
    }
  }

  // Verifies that computing Conv2D+BiasAdd in a graph is identical to
  // FusedConv2D.
  void VerifyConv2DWithBias(int filter_size, int filter_count,
                            const std::vector<int>& explicit_paddings = {},
                            int depth = kDepth, int image_width = kImageWidth,
                            int image_height = kImageHeight,
                            int image_batch_count = kImageBatchCount) {
    std::string padding = explicit_paddings.empty() ? "SAME" : "EXPLICIT";
    const BiasAddGraphRunner run_default =
        [this, &explicit_paddings, padding](
            const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunConv2DWithBias(input_data, filter_data, bias_data, padding,
                            explicit_paddings, out);
        };

    const BiasAddGraphRunner run_fused =
        [this, explicit_paddings, padding](
            const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunFusedConv2DOp(input_data, filter_data, {bias_data}, {"BiasAdd"},
                           padding, explicit_paddings, out);
        };

    VerifyBiasAddTensorsNear(depth, image_width, image_height,
                             image_batch_count, filter_size, filter_count,
                             run_default, run_fused);
  }

  // Verifies that computing Conv2D+BiasAdd+{Activation} in a graph is identical
  // to FusedConv2D.
  void VerifyConv2DWithBiasAndActivation(
      const string& activation, int filter_size, int filter_count,
      const std::vector<int>& explicit_paddings = {}, int depth = kDepth,
      int image_width = kImageWidth, int image_height = kImageHeight,
      int image_batch_count = kImageBatchCount) {
    std::string padding = explicit_paddings.empty() ? "SAME" : "EXPLICIT";
    const BiasAddGraphRunner run_default =
        [this, &activation, &explicit_paddings, &padding](
            const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunConv2DWithBiasAndActivation(
              input_data, filter_data, bias_data, activation, padding,
              explicit_paddings, out,
              /*allow_gpu_device=*/activation == "Relu");
        };

    const BiasAddGraphRunner run_fused = [this, &activation, &explicit_paddings,
                                          padding](const Tensor& input_data,
                                                   const Tensor& filter_data,
                                                   const Tensor& bias_data,
                                                   Tensor* out) {
      RunFusedConv2DOp(input_data, filter_data, {bias_data},
                       {"BiasAdd", activation}, padding, explicit_paddings, out,
                       /*allow_gpu_device=*/activation == "Relu");
    };

    VerifyBiasAddTensorsNear(depth, image_width, image_height,
                             image_batch_count, filter_size, filter_count,
                             run_default, run_fused);
  }

  // Verifies that computing Conv2D+FusedBatchNorm in a graph is identical to
  // FusedConv2D.
  void VerifyConv2DWithBatchNorm(int filter_size, int filter_count,
                                 const std::vector<int>& explicit_paddings = {},
                                 int depth = kDepth,
                                 int image_width = kImageWidth,
                                 int image_height = kImageHeight,
                                 int image_batch_count = kImageBatchCount) {
    std::string padding = explicit_paddings.empty() ? "SAME" : "EXPLICIT";
    const BatchNormGraphRunner run_default =
        [this, explicit_paddings, padding](
            const Tensor& input_data, const Tensor& filter_data,
            const Tensor& scale_data, const Tensor& offset_data,
            const Tensor& mean_data, const Tensor& variance_data, Tensor* out) {
          RunConv2DWithBatchNorm(input_data, filter_data, scale_data,
                                 offset_data, mean_data, variance_data, padding,
                                 explicit_paddings, out);
        };

    const BatchNormGraphRunner run_fused =
        [this, explicit_paddings, padding](
            const Tensor& input_data, const Tensor& filter_data,
            const Tensor& scale_data, const Tensor& offset_data,
            const Tensor& mean_data, const Tensor& variance_data, Tensor* out) {
          RunFusedConv2DOp(input_data, filter_data,
                           {scale_data, offset_data, mean_data, variance_data},
                           {"FusedBatchNorm"}, padding, explicit_paddings, out);
        };

    VerifyFusedBatchNormTensorsNear(depth, image_width, image_height,
                                    image_batch_count, filter_size,
                                    filter_count, run_default, run_fused);
  }

  // Verifies that computing Conv2D+FusedBatchNorm+{Activation} in a graph is
  // identical to FusedConv2D.
  void VerifyConv2DWithBatchNormAndActivation(
      const string& activation, int filter_size, int filter_count,
      const std::vector<int>& explicit_paddings = {}, int depth = kDepth,
      int image_width = kImageWidth, int image_height = kImageHeight,
      int image_batch_count = kImageBatchCount) {
    std::string padding = explicit_paddings.empty() ? "SAME" : "EXPLICIT";
    const BatchNormGraphRunner run_default =
        [this, &activation, explicit_paddings, padding](
            const Tensor& input_data, const Tensor& filter_data,
            const Tensor& scale_data, const Tensor& offset_data,
            const Tensor& mean_data, const Tensor& variance_data, Tensor* out) {
          RunConv2DWithBatchNormAndActivation(
              input_data, filter_data, scale_data, offset_data, mean_data,
              variance_data, activation, padding, explicit_paddings, out);
        };

    const BatchNormGraphRunner run_fused =
        [this, &activation, explicit_paddings, padding](
            const Tensor& input_data, const Tensor& filter_data,
            const Tensor& scale_data, const Tensor& offset_data,
            const Tensor& mean_data, const Tensor& variance_data, Tensor* out) {
          RunFusedConv2DOp(input_data, filter_data,
                           {scale_data, offset_data, mean_data, variance_data},
                           {"FusedBatchNorm", activation}, padding,
                           explicit_paddings, out);
        };

    VerifyFusedBatchNormTensorsNear(depth, image_width, image_height,
                                    image_batch_count, filter_size,
                                    filter_count, run_default, run_fused);
  }
};

// Conv2D with BatchNorm can be tested only with `T=float`, because default
// `FusedBatchNorm` kernel supports only floats for scale, mean and variance.

template <typename T>
class FusedConv2DWithBiasOpTest : public FusedConv2DOpTest<T> {};
template <typename T>
class FusedConv2DWithBatchNormOpTest : public FusedConv2DOpTest<T> {};

TYPED_TEST_SUITE_P(FusedConv2DWithBiasOpTest);
TYPED_TEST_SUITE_P(FusedConv2DWithBatchNormOpTest);

// ROCm does not yet support the _FusedConv2D op,
// Therefore disable tests that check _FusedConv2D, when building with ROCm

#ifndef TENSORFLOW_USE_ROCM
// -------------------------------------------------------------------------- //
// Conv2D + BiasAdd + {Activation}                                            //
// -------------------------------------------------------------------------- //

TYPED_TEST_P(FusedConv2DWithBiasOpTest, OneByOneConvolution) {
  const int filter_size = 1;
  const int filter_count = 12;
  this->VerifyConv2DWithBias(filter_size, filter_count);
}

TYPED_TEST_P(FusedConv2DWithBiasOpTest, ImageSizeConvolution) {
  const int filter_size = TestFixture::kImageWidth;
  const int filter_count = 12;
  this->VerifyConv2DWithBias(filter_size, filter_count);
}

TYPED_TEST_P(FusedConv2DWithBiasOpTest, SpatialConvolution) {
  const int filter_size = 3;
  const int filter_count = 12;
  this->VerifyConv2DWithBias(filter_size, filter_count);
}

#ifndef INTEL_MKL
TYPED_TEST_P(FusedConv2DWithBiasOpTest, ExplicitPaddingConvolution) {
  const int filter_size = 3;
  const int filter_count = 12;
  this->VerifyConv2DWithBias(filter_size, filter_count,
                             /*explicit_paddings=*/{0, 0, 1, 2, 3, 4, 0, 0});
}
#endif

TYPED_TEST_P(FusedConv2DWithBiasOpTest, OneByOneConvolutionAndActivation) {
  const int filter_size = 1;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(activation, filter_size,
                                            filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBiasOpTest, ImageSizeConvolutionAndActivation) {
  const int filter_size = TestFixture::kImageWidth;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(activation, filter_size,
                                            filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBiasOpTest, SpatialConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(activation, filter_size,
                                            filter_count);
  }
}

#ifndef INTEL_MKL
TYPED_TEST_P(FusedConv2DWithBiasOpTest,
             ExplicitPaddingConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(
        activation, filter_size, filter_count,
        /*explicit_paddings=*/{0, 0, 1, 2, 3, 4, 0, 0});
  }
}
#endif

// -------------------------------------------------------------------------- //
// Conv2D + FusedBatchNorm + {Activation}                                     //
// -------------------------------------------------------------------------- //

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest, OneByOneConvolution) {
  const int filter_size = 1;
  const int filter_count = 12;
  this->VerifyConv2DWithBatchNorm(filter_size, filter_count);
}

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest, ImageSizeConvolution) {
  const int filter_size = TestFixture::kImageWidth;
  const int filter_count = 12;
  this->VerifyConv2DWithBatchNorm(filter_size, filter_count);
}

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest, SpatialConvolution) {
  const int filter_size = 3;
  const int filter_count = 12;
  this->VerifyConv2DWithBatchNorm(filter_size, filter_count);
}

#ifndef INTEL_MKL
TYPED_TEST_P(FusedConv2DWithBatchNormOpTest, ExplicitPaddingConvolution) {
  const int filter_size = 3;
  const int filter_count = 12;
  this->VerifyConv2DWithBatchNorm(
      filter_size, filter_count,
      /*explicit_paddings=*/{0, 0, 1, 2, 3, 4, 0, 0});
}
#endif

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest, OneByOneConvolutionAndActivation) {
  const int filter_size = 1;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBatchNormAndActivation(activation, filter_size,
                                                 filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest,
             ImageSizeConvolutionAndActivation) {
  const int filter_size = TestFixture::kImageWidth;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBatchNormAndActivation(activation, filter_size,
                                                 filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest, SpatialConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBatchNormAndActivation(activation, filter_size,
                                                 filter_count);
  }
}

#ifndef INTEL_MKL
TYPED_TEST_P(FusedConv2DWithBatchNormOpTest,
             ExplicitPaddingConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBatchNormAndActivation(
        activation, filter_size, filter_count,
        /*explicit_paddings=*/{0, 0, 1, 2, 3, 4, 0, 0});
  }
}
#endif

REGISTER_TYPED_TEST_SUITE_P(FusedConv2DWithBiasOpTest,          //
                            OneByOneConvolution,                //
                            ImageSizeConvolution,               //
                            SpatialConvolution,                 //
#ifndef INTEL_MKL
                            ExplicitPaddingConvolution,         //
#endif
                            OneByOneConvolutionAndActivation,   //
                            ImageSizeConvolutionAndActivation,  //
#ifndef INTEL_MKL
                            SpatialConvolutionAndActivation,    //
                            ExplicitPaddingConvolutionAndActivation);
#else
                            SpatialConvolutionAndActivation);
#endif

REGISTER_TYPED_TEST_SUITE_P(FusedConv2DWithBatchNormOpTest,     //
                            OneByOneConvolution,                //
                            ImageSizeConvolution,               //
                            SpatialConvolution,                 //
#ifndef INTEL_MKL
                            ExplicitPaddingConvolution,         //
#endif
                            OneByOneConvolutionAndActivation,   //
                            ImageSizeConvolutionAndActivation,  //
#ifndef INTEL_MKL
                            SpatialConvolutionAndActivation,    //
                            ExplicitPaddingConvolutionAndActivation);
#else
                            SpatialConvolutionAndActivation);
#endif

using FusedBiasAddDataTypes = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedConv2DWithBiasOpTest,
                               FusedBiasAddDataTypes);


#ifndef INTEL_MKL
using FusedBatchNormDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedConv2DWithBatchNormOpTest,
                               FusedBatchNormDataTypes);
#endif

#endif  // TENSORFLOW_USE_ROCM
}  // namespace tensorflow
