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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class ResizeBilinearOpTest : public OpsTestBase {
 protected:
  ResizeBilinearOpTest() {
    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_op", "ResizeBilinear")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", false)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  const Tensor* AddRandomImageInput(const TensorShape& shape) {
    CHECK_GT(input_types_.size(), inputs_.size())
        << "Adding more inputs than types; perhaps you need to call MakeOp";
    CHECK_EQ(shape.dims(), 4) << "All images must have 4 dimensions.";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(device_->GetAllocator(AllocatorAttributes()),
                               DataTypeToEnum<float>::v(), shape);
    input->flat<float>().setRandom();
    tensors_.push_back(input);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]),
               DataTypeToEnum<float>::v());
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<float>::v());
      inputs_.push_back({nullptr, input});
    }
    return input;
  }

  // This is the straight forward unoptimized implementation of resize bilinear
  // We use this to confirm that the optimized version is exactly identical.
  void ResizeBilinearBaseline(TTypes<float, 4>::ConstTensor images,
                              TTypes<float, 4>::Tensor output) {
    const int batch = images.dimension(0);
    const int64 in_height = images.dimension(1);
    const int64 in_width = images.dimension(2);
    const int channels = images.dimension(3);

    ASSERT_EQ(batch, output.dimension(0));
    ASSERT_EQ(channels, output.dimension(3));

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    for (int b = 0; b < batch; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const float in_y = y * height_scale;
        const int64 top_y_index = static_cast<int64>(floorf(in_y));
        const int64 bottom_y_index =
            std::min(static_cast<int64>(ceilf(in_y)), in_height - 1);
        const float y_lerp = in_y - top_y_index;
        for (int64 x = 0; x < out_width; ++x) {
          const float in_x = x * width_scale;
          const int64 left_x_index = static_cast<int64>(floorf(in_x));
          const int64 right_x_index =
              std::min(static_cast<int64>(ceilf(in_x)), in_width - 1);
          const float x_lerp = in_x - left_x_index;
          for (int c = 0; c < channels; ++c) {
            const float top_left = images(b, top_y_index, left_x_index, c);
            const float top_right = images(b, top_y_index, right_x_index, c);
            const float bottom_left =
                images(b, bottom_y_index, left_x_index, c);
            const float bottom_right =
                images(b, bottom_y_index, right_x_index, c);
            const float top = top_left + (top_right - top_left) * x_lerp;
            const float bottom =
                bottom_left + (bottom_right - bottom_left) * x_lerp;
            output(b, y, x, c) = top + (bottom - top) * y_lerp;
          }
        }
      }
    }
  }

  void TestResize(int input_width, int input_height, int channels,
                  int output_width, int output_height) {
    const TensorShape shape({1, input_width, input_height, channels});
    const Tensor* input = AddRandomImageInput(shape);
    AddInputFromArray<int32>(TensorShape({2}), {output_width, output_height});
    TF_ASSERT_OK(RunOpKernel());

    std::unique_ptr<Tensor> expected(
        new Tensor(device_->GetAllocator(AllocatorAttributes()),
                   DataTypeToEnum<float>::v(),
                   TensorShape({1, output_width, output_height, channels})));
    ResizeBilinearBaseline(input->tensor<float, 4>(),
                           expected->tensor<float, 4>());
    test::ExpectTensorEqual<float>(*expected, *GetOutput(0));
  }
};

class ResizeBilinearOpAlignCornersTest : public OpsTestBase {
 protected:
  ResizeBilinearOpAlignCornersTest() {
    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_op", "ResizeBilinear")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", true)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinearRandom2x2To1x1) {
  const Tensor* input = AddRandomImageInput(TensorShape({1, 2, 2, 1}));
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor* output = GetOutput(0);
  std::unique_ptr<Tensor> expected(
      new Tensor(device_->GetAllocator(AllocatorAttributes()),
                 DataTypeToEnum<float>::v(), TensorShape({1, 1, 1, 1})));
  ResizeBilinearBaseline(input->tensor<float, 4>(),
                         expected->tensor<float, 4>());
  EXPECT_EQ(input->flat<float>()(0), output->flat<float>()(0));
  test::ExpectTensorEqual<float>(*expected.get(), *output);
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,        5.0f / 3,  2,
     7.0f / 3, 3,         10.0f / 3,
     3,        11.0f / 3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // The corners exactly align with the original corners, and we bilinear
  // interpolate the values in between.

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5,  2,
     2,  2.5,  3,
     3,  3.5,  4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,   2.5,
     5.5,   7});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,
     7,  9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear3x3To4x4) {
  // Input:
  //  1, 2, 3,
  //  4, 5, 6,
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1.75, 2.5, 3,
     3.25, 4, 4.75, 5.25,
     5.5, 6.25, 7, 7.5,
     7,  7.75, 8.5, 9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,        7.0f/3, 11.0f/3,
     19.0f/3, 23.0f/3, 27.0f/3,
     35.0f/3, 39.0f/3, 43.0f/3});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    { 1,  2.5,  4,
      7,  8.5, 10,
     13, 14.5, 16});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To3x3Batch2) {
  // Input:
  //  1, 2
  //  3, 4
  //
  // repeated twice
  AddInputFromArray<float>(TensorShape({2, 2, 2, 1}), {1, 2, 3, 4, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 5.0f/3, 2, 7.0f/3, 3, 10.0f/3, 3, 11.0f/3, 4,
     1, 5.0f/3, 2, 7.0f/3, 3, 10.0f/3, 3, 11.0f/3, 4
    });
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2x2To3x3x2) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 2}),
                           {1, -1, 2, -2, 3, -3, 4, -4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 2}));
  // clang-format off
  test::FillValues<float>(&expected,
    {
      1,       -1,
      5.0f/3,  -5.0f/3,
      2,       -2,
      7.0f/3,  -7.0f/3,
      3,       -3,
      10.0f/3, -10.0f/3,
      3,       -3,
      11.0f/3, -11.0f/3,
      4,       -4
    });
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5, 2, 2,
     2,  2.5, 3, 3,
     3,  3.5, 4, 4,
     3,  3.5, 4, 4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// similar_size case
TEST_F(ResizeBilinearOpTest, Test1_1c) { TestResize(183, 299, 1, 299, 299); }
TEST_F(ResizeBilinearOpTest, Test1_3c) { TestResize(183, 299, 3, 299, 299); }

// Significantly smaller: scale_up case
TEST_F(ResizeBilinearOpTest, Test2_1c) { TestResize(141, 186, 1, 299, 299); }
TEST_F(ResizeBilinearOpTest, Test2_3c) { TestResize(141, 186, 3, 299, 299); }

// Significantly larger: scale_down case
TEST_F(ResizeBilinearOpTest, Test3_1c) { TestResize(749, 603, 1, 299, 299); }
TEST_F(ResizeBilinearOpTest, Test3_3c) { TestResize(749, 603, 3, 299, 299); }

// Exactly the same size
TEST_F(ResizeBilinearOpTest, Test4_1c) { TestResize(299, 299, 1, 299, 299); }
TEST_F(ResizeBilinearOpTest, Test4_3c) { TestResize(299, 299, 3, 299, 299); }

// Slightly smaller: similar_size case
TEST_F(ResizeBilinearOpTest, Test5_1c) { TestResize(298, 297, 1, 299, 299); }
TEST_F(ResizeBilinearOpTest, Test5_3c) { TestResize(298, 297, 3, 299, 299); }

// Slightly bigger: similar_size case
TEST_F(ResizeBilinearOpTest, Test6_1c) { TestResize(304, 303, 1, 299, 299); }
TEST_F(ResizeBilinearOpTest, Test6_3c) { TestResize(304, 303, 3, 299, 299); }

TEST_F(ResizeBilinearOpTest, TestInvalidOutputSize) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      StringPiece(s.ToString())
          .contains("Invalid argument: output dimensions must be positive"))
      << s;
}

TEST_F(ResizeBilinearOpTest, TestInvalidInputShape) {
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Invalid argument: input must be 4-dimensional"))
      << s;
}

TEST_F(ResizeBilinearOpTest, TestInvalidSizeDim) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {4, 4});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Invalid argument: shape_t must be 1-dimensional"))
      << s;
}

TEST_F(ResizeBilinearOpTest, TestInvalidSizeElements) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {4, 4, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Invalid argument: shape_t must have two elements"))
      << s;
}

}  // namespace tensorflow
