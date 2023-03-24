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

#include <cmath>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

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
  static constexpr int kDepth = 4;
  static constexpr int kImageWidth = 32;
  static constexpr int kImageHeight = 32;
  static constexpr int kImageBatchCount = 8;

  static constexpr bool kIsInt8 =
      std::is_same<T, int8>::value || std::is_same<T, qint8>::value;

  using BiasAddGraphRunner =
      std::function<void(const Tensor& input_data, const Tensor& filter_data,
                         const Tensor& bias_data, Tensor* out)>;

  using BatchNormGraphRunner = std::function<void(
      const Tensor& input_data, const Tensor& filter_data,
      const Tensor& scale_data, const Tensor& offset_data,
      const Tensor& mean_data, const Tensor& variance_data, Tensor* out)>;

  // Checks if it is a GPU test not a CPU test
  static bool HasGpuDevice() {
    tensorflow::SessionOptions session_options;
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    std::vector<DeviceAttributes> available_devices;
    [&]() { TF_ASSERT_OK(session->ListDevices(&available_devices)); }();

    // Check if session has an available GPU device.
    const bool has_gpu_device =
        absl::c_any_of(available_devices, [](const DeviceAttributes& device) {
          return device.device_type() == DEVICE_GPU;
        });

    return has_gpu_device;
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  void RunAndFetch(const tensorflow::Scope& root, const std::string& fetch,
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

    // Check if there is an available GPU device.
    const bool has_gpu_device = HasGpuDevice();

    // Some of the `FusedConv2D` fusion types are implemented only for CPU, and
    // in this test we don't want to compare GPU vs CPU numbers, so place all
    // nodes on CPU in this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const std::string device =
        place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
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
    RunConv2DWithBiasAndActivation(input_data, filter_data, bias_data,
                                   std::nullopt, padding, explicit_paddings,
                                   output, allow_gpu_device, stride);
  }

  template <typename From, typename To>
  static Tensor Cast(
      const Tensor& from, const std::function<To(From)>& cast = [](From v) {
        return static_cast<To>(v);
      }) {
    Tensor to(DataTypeToEnum<To>::v(), from.shape());
    for (int i = 0; i < from.NumElements(); ++i) {
      to.flat<To>()(i) = cast(from.flat<From>()(i));
    }
    return to;
  }

  // Run unfused convolution with bias and optional activation. For every data
  // type the input tensor is in NHWC format and the input filter is in HWIO
  // format. This function converts int8 input data to float and converts float
  // result back to int8 in the case of int8 data type.
  void RunConv2DWithBiasAndActivation(
      Tensor input_data, Tensor filter_data, Tensor bias_data,
      std::optional<std::string> activation_type, const std::string& padding,
      const std::vector<int>& explicit_paddings, Tensor* output,
      bool allow_gpu_device = false, int stride = 1) {
    Scope root = tensorflow::Scope::NewRootScope();

    if (kIsInt8) {
      input_data = Cast<T, float>(input_data);
      filter_data = Cast<T, float>(filter_data);
      bias_data = Cast<T, float>(bias_data);
    }

    ops::Conv2D conv = ops::Conv2D(
        root.WithOpName("conv"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data)),
        {1, stride, stride, 1}, padding,
        ops::Conv2D::Attrs().ExplicitPaddings(explicit_paddings));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), conv,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    if (activation_type.has_value()) {
      if (*activation_type == "Relu") {
        ops::Relu(root.WithOpName("with_activation"), with_bias);
      } else if (*activation_type == "Relu6") {
        ops::Relu6(root.WithOpName("with_activation"), with_bias);
      } else if (*activation_type == "Elu") {
        ops::Elu(root.WithOpName("with_activation"), with_bias);
      } else if (*activation_type == "LeakyRelu") {
        ops::internal::LeakyRelu(root.WithOpName("with_activation"), with_bias);
      } else {
        ops::Identity(root.WithOpName("with_activation"), with_bias);
      }
    }

    RunAndFetch(root,
                activation_type.has_value() ? "with_activation" : "with_bias",
                output, allow_gpu_device);

    if (kIsInt8) {
      *output = Cast<float, T>(
          *output, [](float v) { return static_cast<T>(std::lround(v)); });
    }
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
    } else if (activation_type == "LeakyRelu") {
      ops::internal::LeakyRelu(root.WithOpName("with_activation"),
                               with_fused_batch_norm.y);
    } else {
      ops::Identity(root.WithOpName("with_activation"),
                    with_fused_batch_norm.y);
    }

    RunAndFetch(root, "with_activation", output, allow_gpu_device);
  }

  void RunFusedConv2DOp(Tensor input_data, Tensor filter_data,
                        std::vector<Tensor> args_data,
                        const std::vector<std::string>& fused_ops,
                        const std::string& padding,
                        const std::vector<int>& explicit_paddings,
                        Tensor* output, bool allow_gpu_device = false,
                        int stride = 1) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();

    // Check if there is an available GPU device.
    const bool has_gpu_device = HasGpuDevice();
    const bool has_extra_parameters = kIsInt8;
    const bool has_float_bias = kIsInt8;

    DataType dtype_args =
        has_float_bias ? DataTypeToEnum<float>::v() : DataTypeToEnum<T>::v();

    const int n = GetTensorDim(input_data, FORMAT_NHWC, 'N');
    const int h = GetTensorDim(input_data, FORMAT_NHWC, 'H');
    const int w = GetTensorDim(input_data, FORMAT_NHWC, 'W');
    const int kh = GetFilterDim(filter_data, FORMAT_HWIO, 'H');
    const int kw = GetFilterDim(filter_data, FORMAT_HWIO, 'W');
    const int ic = GetFilterDim(filter_data, FORMAT_HWIO, 'I');
    const int oc = GetFilterDim(filter_data, FORMAT_HWIO, 'O');
    const int v = (kIsInt8 && allow_gpu_device && has_gpu_device) ? 4 : 1;

    if (v > 1) {
      {
        TensorShape shape;
        TF_EXPECT_OK(
            ShapeFromFormatWithStatus(FORMAT_NCHW_VECT_C, n, h, w, ic, &shape));
        Tensor input_data_nchwv(dtype, shape);
        input_data_nchwv.tensor<T, 5>() =
            input_data.shaped<T, 5>({n, h, w, ic / v, v})
                .shuffle(Eigen::array<int, 5>{0, 3, 1, 2, 4});
        input_data = input_data_nchwv;
      }

      {
        // Convert the filter from HWIO to OIHW_VECT_I
        Tensor filter_data_oihwv(
            dtype,
            ShapeFromFilterTensorFormat(FORMAT_OIHW_VECT_I, kh, kw, ic, oc));

        filter_data_oihwv.tensor<T, 5>() =
            filter_data.shaped<T, 4>({kh, kw, ic, oc})
                .reshape(Eigen::array<int, 5>{kh, kw, ic / v, v, oc})
                .shuffle(Eigen::array<int, 5>{4, 2, 0, 1, 3});
        filter_data = filter_data_oihwv;
      }
    }

    if (has_float_bias) {
      // Convert the bias to float
      for (Tensor& arg_data : args_data) {
        TensorShape shape = arg_data.shape();
        Tensor arg_data_float = Tensor(dtype_args, shape);
        for (int index = 0; index < arg_data.NumElements(); index++) {
          int8 v = *(reinterpret_cast<int8*>(arg_data.data()) + index);
          *(reinterpret_cast<float*>(arg_data_float.data()) + index) =
              static_cast<float>(v);
        }
        arg_data = arg_data_float;
      }
    }

    int num_args = static_cast<int>(args_data.size());

    Output input =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output filter =
        ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data));

    std::vector<NodeDefBuilder::NodeOut> args;
    std::vector<DataType> args_dtypes;
    for (int i = 0; i < num_args; ++i) {
      Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                              Input::Initializer(args_data[i]));
      args.emplace_back(arg.name(), 0, dtype_args);
      args_dtypes.emplace_back(dtype_args);
    }

    Tensor side_input(dtype);
    if (has_extra_parameters) {
      // Create side_input
      Padding padding_type;
      ASSERT_TRUE(GetPaddingFromString(padding, &padding_type).ok());
      int64_t oh, oh_padding;
      ASSERT_TRUE(
          GetWindowedOutputSize(h, kh, stride, padding_type, &oh, &oh_padding)
              .ok());
      int64_t ow, ow_padding;
      ASSERT_TRUE(
          GetWindowedOutputSize(w, kw, stride, padding_type, &ow, &ow_padding)
              .ok());
      TensorShape shape;
      TF_EXPECT_OK(
          ShapeFromFormatWithStatus(FORMAT_NCHW_VECT_C, n, oh, ow, oc, &shape));
      side_input = Tensor(dtype, shape);
      side_input.flat<T>() = side_input.flat<T>().setConstant(0);
    }

    Tensor conv_input_scale(DT_FLOAT, {1});
    Tensor side_input_scale(DT_FLOAT, {1});
    std::vector<NodeDefBuilder::NodeOut> host_args;
    int num_host_args = 0;
    if (has_extra_parameters) {
      ++num_args;
      Output arg2 = ops::Const(root.WithOpName("side_input"),
                               Input::Initializer(side_input));
      args.emplace_back(arg2.name(), 0, dtype);
      args_dtypes.emplace_back(dtype);

      ++num_host_args;
      conv_input_scale.scalar<float>()() = 1;
      Output arg3 = ops::Const(root.WithOpName("conv_input_scale"),
                               Input::Initializer(conv_input_scale));
      host_args.emplace_back(arg3.name(), 0, DT_FLOAT);

      ++num_host_args;
      side_input_scale.scalar<float>()() = 1;
      Output arg4 = ops::Const(root.WithOpName("side_input_scale"),
                               Input::Initializer(side_input_scale));
      host_args.emplace_back(arg4.name(), 0, DT_FLOAT);
    }

    NodeDef fused_conv2d;
    TF_EXPECT_OK(NodeDefBuilder("fused_conv", "_FusedConv2D")
                     .Input({input.name(), 0, dtype})
                     .Input({filter.name(), 0, dtype})
                     .Input(args)
                     .Input(host_args)
                     .Attr("num_args", num_args)
                     .Attr("num_host_args", num_host_args)
                     .Attr("T", dtype)
                     .Attr("TArgs", args_dtypes)
                     .Attr("data_format", v > 1 ? "NCHW_VECT_C" : "NHWC")
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", padding)
                     .Attr("explicit_paddings", explicit_paddings)
                     .Attr("fused_ops", fused_ops)
                     .Finalize(&fused_conv2d));

    RunAndFetch(root, fused_conv2d.name(), output, allow_gpu_device,
                &fused_conv2d);

    if (v > 1) {
      // Convert the output from NCHW_VECT_C to NHWC
      const int oh = GetTensorDim(*output, FORMAT_NCHW_VECT_C, 'H');
      const int ow = GetTensorDim(*output, FORMAT_NCHW_VECT_C, 'W');
      TensorShape shape;
      TF_EXPECT_OK(
          ShapeFromFormatWithStatus(FORMAT_NHWC, n, oh, ow, oc, &shape));
      Tensor output_nhwc(dtype, shape);
      output_nhwc.tensor<T, 4>() =
          output->tensor<T, 5>()
              .shuffle(Eigen::array<int, 5>{0, 2, 3, 1, 4})
              .reshape(Eigen::array<int, 4>{n, oh, ow, oc});
      *output = output_nhwc;
    }
  }

  void ExpectMatch(const Tensor& x, const Tensor& y, double atol) {
    constexpr bool exact_match =
        std::is_same<T, int8>::value || std::is_same<T, qint8>::value;
    if (exact_match) {
      test::ExpectEqual(x, y);
    } else {
      test::ExpectClose(x, y, atol);
    }
  }

  void VerifyBiasAddTensorsNear(int depth, int image_width, int image_height,
                                int image_batch_count, int filter_size,
                                int filter_count,
                                const BiasAddGraphRunner& run_default,
                                const BiasAddGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    constexpr int int8_scale = 80;

    using ConvT = typename std::conditional<kIsInt8, int8, T>::type;
    DataType dtype_conv = DataTypeToEnum<ConvT>::v();

    TensorShape image_shape{image_batch_count, image_height, image_width,
                            depth};
    Tensor image_tmp(dtype_conv, image_shape);
    image_tmp.flat<ConvT>() = image_tmp.flat<ConvT>().setRandom();
    if (kIsInt8) {
      image_tmp.flat<ConvT>() /= image_tmp.flat<ConvT>().constant(int8_scale);
    }
    Tensor image(dtype, image_shape);
    ASSERT_TRUE(image.BitcastFrom(image_tmp, dtype, image_shape).ok());

    // Add some negative values to filter to properly test Relu.
    TensorShape filter_shape{filter_size, filter_size, depth, filter_count};
    Tensor filter_tmp(dtype_conv, filter_shape);
    filter_tmp.flat<ConvT>() = filter_tmp.flat<ConvT>().setRandom();
    if (kIsInt8) {
      filter_tmp.flat<ConvT>() /= filter_tmp.flat<ConvT>().constant(int8_scale);
    } else {
      filter_tmp.flat<ConvT>() -=
          filter_tmp.flat<ConvT>().constant(static_cast<ConvT>(0.5f));
    }
    Tensor filter(dtype, filter_shape);
    ASSERT_TRUE(filter.BitcastFrom(filter_tmp, dtype, filter_shape).ok());

    const int bias_size = filter_count;
    TensorShape bias_shape{bias_size};
    Tensor bias_tmp(dtype_conv, bias_shape);
    bias_tmp.flat<ConvT>() = bias_tmp.flat<ConvT>().setRandom();
    if (kIsInt8) {
      bias_tmp.flat<ConvT>() /= bias_tmp.flat<ConvT>().constant(int8_scale);
    } else {
      bias_tmp.flat<ConvT>() +=
          bias_tmp.flat<ConvT>().constant(static_cast<ConvT>(0.5f));
    }
    Tensor bias(dtype, bias_shape);
    ASSERT_TRUE(bias.BitcastFrom(bias_tmp, dtype, bias_shape).ok());

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
      ExpectMatch(conv_2d, fused_conv_2d, /*atol=*/1e-4);
    } else {
      ExpectMatch(conv_2d, fused_conv_2d, /*atol=*/1e-5);
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
    if (kIsInt8 && !explicit_paddings.empty()) {
      // This combination is not supported
      return;
    }

    std::string padding = explicit_paddings.empty() ? "SAME" : "EXPLICIT";
    const BiasAddGraphRunner run_default =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunConv2DWithBias(input_data, filter_data, bias_data, padding,
                            explicit_paddings, out);
        };

    const BiasAddGraphRunner run_fused =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunFusedConv2DOp(input_data, filter_data, {bias_data}, {"BiasAdd"},
                           padding, explicit_paddings, out,
                           /*allow_gpu_device=*/kIsInt8);
        };

    VerifyBiasAddTensorsNear(depth, image_width, image_height,
                             image_batch_count, filter_size, filter_count,
                             run_default, run_fused);
  }

  // Verifies that computing Conv2D+BiasAdd+{Activation} in a graph is identical
  // to FusedConv2D.
  void VerifyConv2DWithBiasAndActivation(
      const std::string& activation, int filter_size, int filter_count,
      const std::vector<int>& explicit_paddings = {}, int depth = kDepth,
      int image_width = kImageWidth, int image_height = kImageHeight,
      int image_batch_count = kImageBatchCount) {
    if (kIsInt8 && (activation != "Relu" || !explicit_paddings.empty())) {
      // These combinations are not supported
      return;
    }

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

static auto activations = {"Relu", "Relu6", "Elu", "LeakyRelu"};

TYPED_TEST_P(FusedConv2DWithBiasOpTest, OneByOneConvolutionAndActivation) {
  // Requires full precision Conv2D op
  tensorflow::enable_tensor_float_32_execution(false);
  const int filter_size = 1;
  const int filter_count = 12;
  for (const std::string& activation : activations) {
    this->VerifyConv2DWithBiasAndActivation(activation, filter_size,
                                            filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBiasOpTest, ImageSizeConvolutionAndActivation) {
  const int filter_size = TestFixture::kImageWidth;
  const int filter_count = 12;
  for (const std::string& activation : activations) {
    this->VerifyConv2DWithBiasAndActivation(activation, filter_size,
                                            filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBiasOpTest, SpatialConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const std::string& activation : activations) {
    this->VerifyConv2DWithBiasAndActivation(activation, filter_size,
                                            filter_count);
  }
}

#ifndef INTEL_MKL
TYPED_TEST_P(FusedConv2DWithBiasOpTest,
             ExplicitPaddingConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const std::string& activation : activations) {
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
  for (const std::string& activation : activations) {
    this->VerifyConv2DWithBatchNormAndActivation(activation, filter_size,
                                                 filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest,
             ImageSizeConvolutionAndActivation) {
  const int filter_size = TestFixture::kImageWidth;
  const int filter_count = 12;
  for (const std::string& activation : activations) {
    this->VerifyConv2DWithBatchNormAndActivation(activation, filter_size,
                                                 filter_count);
  }
}

TYPED_TEST_P(FusedConv2DWithBatchNormOpTest, SpatialConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const std::string& activation : activations) {
    this->VerifyConv2DWithBatchNormAndActivation(activation, filter_size,
                                                 filter_count);
  }
}

#ifndef INTEL_MKL
TYPED_TEST_P(FusedConv2DWithBatchNormOpTest,
             ExplicitPaddingConvolutionAndActivation) {
  const int filter_size = 3;
  const int filter_count = 12;
  for (const std::string& activation : activations) {
    this->VerifyConv2DWithBatchNormAndActivation(
        activation, filter_size, filter_count,
        /*explicit_paddings=*/{0, 0, 1, 2, 3, 4, 0, 0});
  }
}
#endif

#ifndef INTEL_MKL
REGISTER_TYPED_TEST_SUITE_P(FusedConv2DWithBiasOpTest,          //
                            OneByOneConvolution,                //
                            ImageSizeConvolution,               //
                            SpatialConvolution,                 //
                            ExplicitPaddingConvolution,         //
                            OneByOneConvolutionAndActivation,   //
                            ImageSizeConvolutionAndActivation,  //
                            SpatialConvolutionAndActivation,    //
                            ExplicitPaddingConvolutionAndActivation);

REGISTER_TYPED_TEST_SUITE_P(FusedConv2DWithBatchNormOpTest,     //
                            OneByOneConvolution,                //
                            ImageSizeConvolution,               //
                            SpatialConvolution,                 //
                            ExplicitPaddingConvolution,         //
                            OneByOneConvolutionAndActivation,   //
                            ImageSizeConvolutionAndActivation,  //
                            SpatialConvolutionAndActivation,    //
                            ExplicitPaddingConvolutionAndActivation);
#else
REGISTER_TYPED_TEST_SUITE_P(FusedConv2DWithBiasOpTest,          //
                            OneByOneConvolution,                //
                            ImageSizeConvolution,               //
                            SpatialConvolution,                 //
                            OneByOneConvolutionAndActivation,   //
                            ImageSizeConvolutionAndActivation,  //
                            SpatialConvolutionAndActivation);

REGISTER_TYPED_TEST_SUITE_P(FusedConv2DWithBatchNormOpTest,     //
                            OneByOneConvolution,                //
                            ImageSizeConvolution,               //
                            SpatialConvolution,                 //
                            OneByOneConvolutionAndActivation,   //
                            ImageSizeConvolutionAndActivation,  //
                            SpatialConvolutionAndActivation);
#endif

using FusedBiasAddDataTypes = ::testing::Types<float, double, int8, qint8>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedConv2DWithBiasOpTest,
                               FusedBiasAddDataTypes);

using FusedBatchNormDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedConv2DWithBatchNormOpTest,
                               FusedBatchNormDataTypes);

#endif  // TENSORFLOW_USE_ROCM
}  // namespace tensorflow
