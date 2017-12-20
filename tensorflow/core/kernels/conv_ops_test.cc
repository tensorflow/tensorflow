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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/kernels/conv_ops_gpu.h"

namespace tensorflow {

#if GOOGLE_CUDA

TEST(ConvParameters, WinogradNonfusedAlgoSize) {
  ConvParameters conv_params_small = {
      1,         // batch
      32,        // in_depths
      {{300,     // in_rows
        300}},   // in_cols
      128,       // out_depths
      {{3,       // filter_rows
        3}},     // filter_cols
      {{1,       // dilation_rows
        1}},     // dilation_cols
      {{1,       // stride_rows
        1}},     // stride_cols
      {{0,       // padding_rows
        0}},     // padding_cols
      DT_FLOAT,  // tensor datatype
      0,         // device_id
  };
  EXPECT_TRUE(conv_params_small.ShouldIncludeWinogradNonfusedAlgo<float>());

  ConvParameters conv_params_large = {
      1,         // batch
      128,       // in_depths
      {{300,     // in_rows
        300}},   // in_cols
      768,       // out_depths
      {{3,       // filter_rows
        3}},     // filter_cols
      {{1,       // dilation_rows
        1}},     // dilation_cols
      {{1,       // stride_rows
        1}},     // stride_cols
      {{0,       // padding_rows
        0}},     // padding_cols
      DT_FLOAT,  // tensor datatype
      0,         // device_id
  };
  EXPECT_FALSE(conv_params_large.ShouldIncludeWinogradNonfusedAlgo<float>());
}

#endif  // GOOGLE_CUDA

class FusedResizePadConvOpTest : public OpsTestBase {
 protected:
  void HandwrittenConv() {
    const int stride = 1;
    TF_EXPECT_OK(NodeDefBuilder("fused_resize_op", "FusedResizeAndPadConv2D")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DT_FLOAT)
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

    const int resized_width = image_width;
    const int resized_height = image_height;

    const int top_padding = 0;
    const int bottom_padding = 0;
    const int left_padding = 0;
    const int right_padding = 0;

    AddInputFromArray<float>(image.shape(), image.flat<float>());
    AddInputFromArray<int32>(TensorShape({2}), {resized_height, resized_width});
    AddInputFromArray<int32>(
        TensorShape({4, 2}),
        {0, 0, top_padding, bottom_padding, left_padding, right_padding, 0, 0});
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
    // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
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

  void CompareFusedAndSeparate(int input_width, int input_height,
                               int input_depth, int resize_width,
                               int resize_height, int y_padding, int x_padding,
                               int filter_size, int filter_count,
                               bool resize_align_corners,
                               const string& pad_mode, int stride,
                               const string& padding) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT,
                      TensorShape({1, input_height, input_width, input_depth}));
    test::FillIota<float>(&input_data, 1.0f);
    Output input =
        Const(root.WithOpName("input"), Input::Initializer(input_data));

    Tensor filter_data(DT_FLOAT, TensorShape({filter_size, filter_size,
                                              input_depth, filter_count}));
    test::FillIota<float>(&filter_data, 1.0f);
    Output filter =
        Const(root.WithOpName("filter"), Input::Initializer(filter_data));

    Output resize_size =
        Const(root.WithOpName("resize_size"), {resize_height, resize_width});
    Output resize =
        ResizeBilinear(root.WithOpName("resize"), input, resize_size,
                       ResizeBilinear::AlignCorners(resize_align_corners));
    Output paddings =
        Const(root.WithOpName("paddings"),
              {{0, 0}, {y_padding, y_padding}, {x_padding, x_padding}, {0, 0}});
    Output mirror_pad =
        MirrorPad(root.WithOpName("mirror_pad"), resize, paddings, pad_mode);
    Output conv = Conv2D(root.WithOpName("conv"), mirror_pad, filter,
                         {1, stride, stride, 1}, padding);

    Output fused_conv = FusedResizeAndPadConv2D(
        root.WithOpName("fused_conv"), input, resize_size, paddings, filter,
        pad_mode, {1, stride, stride, 1}, padding,
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

    test::ExpectTensorNear<float>(unfused_tensors[0], fused_tensors[0], 1e-5);
  }

  void CompareFusedPadOnlyAndSeparate(int input_width, int input_height,
                                      int input_depth, int y_padding,
                                      int x_padding, int filter_size,
                                      int filter_count, const string& pad_mode,
                                      int stride, const string& padding) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT,
                      TensorShape({1, input_height, input_width, input_depth}));
    test::FillIota<float>(&input_data, 1.0f);
    Output input =
        Const(root.WithOpName("input"), Input::Initializer(input_data));

    Tensor filter_data(DT_FLOAT, TensorShape({filter_size, filter_size,
                                              input_depth, filter_count}));
    test::FillIota<float>(&filter_data, 1.0f);
    Output filter =
        Const(root.WithOpName("filter"), Input::Initializer(filter_data));

    Output paddings =
        Const(root.WithOpName("paddings"),
              {{0, 0}, {y_padding, y_padding}, {x_padding, x_padding}, {0, 0}});
    Output mirror_pad =
        MirrorPad(root.WithOpName("mirror_pad"), input, paddings, pad_mode);
    Output conv = Conv2D(root.WithOpName("conv"), mirror_pad, filter,
                         {1, stride, stride, 1}, padding);

    Output fused_conv =
        FusedPadConv2D(root.WithOpName("fused_conv"), input, paddings, filter,
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

    test::ExpectTensorNear<float>(unfused_tensors[0], fused_tensors[0], 1e-5);
  }
};

TEST_F(FusedResizePadConvOpTest, HandwrittenConv) { HandwrittenConv(); }

TEST_F(FusedResizePadConvOpTest, IdentityComparative) {
  CompareFusedAndSeparate(10, 10, 1, 10, 10, 0, 0, 1, 1, false, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ConvOnlyComparative) {
  CompareFusedAndSeparate(10, 10, 3, 10, 10, 0, 0, 4, 4, false, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeOnlyComparative) {
  CompareFusedAndSeparate(10, 10, 1, 20, 20, 0, 0, 1, 1, false, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeAndConvComparative) {
  CompareFusedAndSeparate(2, 2, 4, 4, 2, 0, 0, 2, 2, false, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeAlignAndConvComparative) {
  CompareFusedAndSeparate(2, 2, 4, 4, 2, 0, 0, 2, 2, true, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeAndConvStridedComparative) {
  CompareFusedAndSeparate(2, 2, 4, 4, 2, 0, 0, 2, 2, false, "REFLECT", 2,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeAlignAndConvValidComparative) {
  CompareFusedAndSeparate(2, 2, 4, 4, 2, 0, 0, 2, 2, true, "REFLECT", 1,
                          "VALID");
}

TEST_F(FusedResizePadConvOpTest, PadOnlyComparative) {
  CompareFusedAndSeparate(4, 4, 1, 4, 4, 2, 2, 1, 1, false, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, PadOnlyWithChannelsComparative) {
  CompareFusedAndSeparate(4, 4, 3, 4, 4, 2, 2, 1, 1, false, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeAndPadComparative) {
  CompareFusedAndSeparate(4, 4, 1, 6, 6, 2, 2, 1, 1, false, "REFLECT", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, PadOnlySymmetricComparative) {
  CompareFusedAndSeparate(4, 4, 1, 4, 4, 2, 2, 1, 1, false, "SYMMETRIC", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeAndPadSymmetricComparative) {
  CompareFusedAndSeparate(4, 4, 3, 6, 6, 2, 2, 1, 1, false, "SYMMETRIC", 1,
                          "SAME");
}

TEST_F(FusedResizePadConvOpTest, NoResizeIdentityComparative) {
  CompareFusedPadOnlyAndSeparate(10, 10, 1, 0, 0, 1, 1, "REFLECT", 1, "SAME");
}

TEST_F(FusedResizePadConvOpTest, NoResizeConvOnlyComparative) {
  CompareFusedPadOnlyAndSeparate(10, 10, 3, 0, 0, 4, 4, "REFLECT", 1, "SAME");
}

TEST_F(FusedResizePadConvOpTest, NoResizePadOnlyComparative) {
  CompareFusedPadOnlyAndSeparate(4, 4, 1, 2, 2, 1, 1, "REFLECT", 1, "SAME");
}

TEST_F(FusedResizePadConvOpTest, NoResizePadOnlyWithChannelsComparative) {
  CompareFusedPadOnlyAndSeparate(4, 4, 3, 2, 2, 1, 1, "REFLECT", 1, "SAME");
}

TEST_F(FusedResizePadConvOpTest, NoResizePadOnlySymmetricComparative) {
  CompareFusedPadOnlyAndSeparate(4, 4, 1, 2, 2, 1, 1, "SYMMETRIC", 1, "SAME");
}

TEST_F(FusedResizePadConvOpTest, ResizeAndPadSymmetricComparativeLarge) {
  CompareFusedAndSeparate(1000, 1000, 3, 1006, 1006, 2, 2, 1, 1, false,
                          "SYMMETRIC", 1, "SAME");
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
    // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
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

}  // namespace tensorflow
