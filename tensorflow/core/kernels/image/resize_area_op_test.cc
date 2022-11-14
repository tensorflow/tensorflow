/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class ResizeAreaOpTest : public OpsTestBase {
 protected:
  ResizeAreaOpTest() = default;

  void CreateOp(bool align_corners) {
    TF_EXPECT_OK(NodeDefBuilder("resize_area_op", "ResizeArea")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  const Tensor* SetRandomImageInput(const TensorShape& shape) {
    inputs_.clear();

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

 private:
  // This is the unoptimized implementation of ResizeArea.
  // We use this to confirm that the optimized version is exactly identical.
  void ResizeAreaBaseline(TTypes<float, 4>::ConstTensor input_data,
                          TTypes<float, 4>::Tensor output_data) {
    const int batch_size = input_data.dimension(0);
    const int64_t in_height = input_data.dimension(1);
    const int64_t in_width = input_data.dimension(2);
    const int channels = input_data.dimension(3);

    ASSERT_EQ(batch_size, output_data.dimension(0));
    ASSERT_EQ(channels, output_data.dimension(3));

    const int64_t out_height = output_data.dimension(1);
    const int64_t out_width = output_data.dimension(2);

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    // A temporary tensor for computing the sum.
    Tensor sum_tensor(DT_FLOAT, TensorShape({channels}));
    typename TTypes<float, 1>::Tensor sum_data = sum_tensor.vec<float>();

    // When using this algorithm for downsizing, the target pixel value is the
    // weighted average of all the source pixels. The weight is determined by
    // the contribution percentage of the source pixel.
    //
    // Let "scale" be "target_image_size/source_image_size". If 1/n of the
    // source pixel contributes to the target pixel, then the weight is (1/n *
    // scale); if the complete source pixel contributes to the target pixel,
    // then the weight is scale.
    //
    // To visualize the implementation, use one dimension as an example:
    // Resize in[4] to out[3].
    //   scale = 3/4 = 0.75
    //   out[0]: in[0] and 1/3 of in[1]
    //   out[1]: 2/3 of in[1] and 2/3 of in[2]
    //   out[2]: 1/3 of in[2] and in[1]
    // Hence, the output pixel values are:
    //   out[0] = (in[0] * 1.0 + in[1] * 1/3) * scale
    //   out[1] = (in[1] * 2/3 + in[2] * 2/3 * scale
    //   out[2] = (in[3] * 1/3 + in[3] * 1.0) * scale
    float scale = 1.0 / (height_scale * width_scale);
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const float in_y = y * height_scale;
        const float in_y1 = (y + 1) * height_scale;
        // The start and end height indices of all the cells that could
        // contribute to the target cell.
        int64_t y_start = std::floor(in_y);
        int64_t y_end = std::ceil(in_y1);

        for (int64_t x = 0; x < out_width; ++x) {
          const float in_x = x * width_scale;
          const float in_x1 = (x + 1) * width_scale;
          // The start and end width indices of all the cells that could
          // contribute to the target cell.
          int64_t x_start = std::floor(in_x);
          int64_t x_end = std::ceil(in_x1);

          sum_data.setConstant(0.0);
          for (int64_t i = y_start; i < y_end; ++i) {
            float scale_y = i < in_y
                                ? (i + 1 > in_y1 ? height_scale : i + 1 - in_y)
                                : (i + 1 > in_y1 ? in_y1 - i : 1.0);
            for (int64_t j = x_start; j < x_end; ++j) {
              float scale_x = j < in_x
                                  ? (j + 1 > in_x1 ? width_scale : j + 1 - in_x)
                                  : (j + 1 > in_x1 ? in_x1 - j : 1.0);
              for (int64_t c = 0; c < channels; ++c) {
#define BOUND(val, limit) \
  std::min(((limit)-int64_t{1}), (std::max(int64_t{0}, (val))))
                sum_data(c) +=
                    static_cast<float>(input_data(b, BOUND(i, in_height),
                                                  BOUND(j, in_width), c)) *
                    scale_y * scale_x * scale;
#undef BOUND
              }
            }
          }
          for (int64_t c = 0; c < channels; ++c) {
            output_data(b, y, x, c) = sum_data(c);
          }
        }
      }
    }
  }

 protected:
  void RunRandomTest(int in_height, int in_width, int target_height,
                     int target_width, int channels) {
    const Tensor* input =
        SetRandomImageInput(TensorShape({1, in_height, in_width, channels}));
    AddInputFromArray<int32>(TensorShape({2}), {target_height, target_width});

    TF_ASSERT_OK(RunOpKernel());
    std::unique_ptr<Tensor> expected(
        new Tensor(device_->GetAllocator(AllocatorAttributes()),
                   DataTypeToEnum<float>::v(),
                   TensorShape({1, target_height, target_width, channels})));
    ResizeAreaBaseline(input->tensor<float, 4>(), expected->tensor<float, 4>());
    test::ExpectTensorNear<float>(*expected, *GetOutput(0), 0.00001);
  }

  void RunManyRandomTests(int channels) {
    for (int in_w : {2, 4, 7, 20, 165}) {
      for (int in_h : {1, 3, 5, 8, 100, 233}) {
        for (int target_height : {1, 2, 3, 50, 113}) {
          for (int target_width : {target_height, target_height / 2 + 1}) {
            RunRandomTest(in_h, in_w, target_height, target_width, channels);
          }
        }
      }
    }
  }
};

TEST_F(ResizeAreaOpTest, TestAreaRandom141x186) {
  CreateOp(/*align_corners=*/false);
  RunRandomTest(141, 186, 299, 299, 3 /* channels */);
}

TEST_F(ResizeAreaOpTest, TestAreaRandom183x229) {
  CreateOp(/*align_corners=*/false);
  RunRandomTest(183, 229, 299, 299, 3 /* channels */);
}

TEST_F(ResizeAreaOpTest, TestAreaRandom749x603) {
  CreateOp(/*align_corners=*/false);
  RunRandomTest(749, 603, 299, 299, 3 /* channels */);
}

TEST_F(ResizeAreaOpTest, TestAreaRandom1x1) {
  CreateOp(/*align_corners=*/false);
  RunRandomTest(1, 1, 8, 8, 3 /* channels */);
}

TEST_F(ResizeAreaOpTest, TestAreaRandom1x1AlignCorners) {
  CreateOp(/*align_corners=*/true);
  RunRandomTest(1, 1, 8, 8, 3 /* channels */);
}

TEST_F(ResizeAreaOpTest, TestAreaRandomDataSeveralInputsSizes1Channel) {
  CreateOp(/*align_corners=*/false);
  RunManyRandomTests(1);
}

TEST_F(ResizeAreaOpTest, TestAreaRandomDataSeveralInputsSizes3Channels) {
  CreateOp(/*align_corners=*/false);
  RunManyRandomTests(3);
}

TEST_F(ResizeAreaOpTest, TestAreaRandomDataSeveralInputsSizes4Channels) {
  CreateOp(/*align_corners=*/false);
  RunManyRandomTests(4);
}

}  // namespace tensorflow
