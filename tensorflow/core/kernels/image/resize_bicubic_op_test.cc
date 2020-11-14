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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ResizeBicubicOpTest : public OpsTestBase {
 protected:
  ResizeBicubicOpTest() {
    TF_EXPECT_OK(NodeDefBuilder("resize_bicubic_op", "ResizeBicubic")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", false)
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
  static constexpr int64 kTableSize = (1 << 10);

  const float* InitCoeffsTable() {
    // Allocate and initialize coefficients table using Bicubic
    // convolution algorithm.
    // https://en.wikipedia.org/wiki/Bicubic_interpolation
    float* coeffs_tab = new float[(kTableSize + 1) * 2];
    static const double A = -0.75;
    for (int i = 0; i <= kTableSize; ++i) {
      float x = i * 1.0 / kTableSize;
      coeffs_tab[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
      x += 1.0;
      coeffs_tab[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    }
    return coeffs_tab;
  }

  const float* GetCoeffsTable() {
    // Static so that we initialize it on first use
    static const float* coeffs_tab = InitCoeffsTable();
    return coeffs_tab;
  }

  // Used in the baseline implementation
  inline int64 Bound(int64 val, int64 limit) {
    return std::min(limit - 1, std::max(int64{0}, val));
  }

  // Used in the baseline implementation
  inline void GetWeightsAndIndices(float scale, int64 out_loc, int64 limit,
                                   std::array<float, 4>* weights,
                                   std::array<int64, 4>* indices) {
    const int64 in_loc = scale * out_loc;
    const float delta = scale * out_loc - in_loc;
    const int64 offset = lrintf(delta * kTableSize);
    const float* coeffs_tab = GetCoeffsTable();
    *weights = {{coeffs_tab[offset * 2 + 1], coeffs_tab[offset * 2],
                 coeffs_tab[(kTableSize - offset) * 2],
                 coeffs_tab[(kTableSize - offset) * 2 + 1]}};
    *indices = {{Bound(in_loc - 1, limit), Bound(in_loc, limit),
                 Bound(in_loc + 1, limit), Bound(in_loc + 2, limit)}};
  }

  // Used in the baseline implementation
  inline float Interpolate1D(const std::array<float, 4>& weights,
                             const std::array<float, 4>& values) {
    return values[0] * weights[0] + values[1] * weights[1] +
           values[2] * weights[2] + values[3] * weights[3];
  }

  // This is the straight forward unoptimized implementation of resize bicubic
  // We use this to confirm that the optimized version is exactly identical.
  void ResizeBicubicBaseline(TTypes<float, 4>::ConstTensor images,
                             TTypes<float, 4>::Tensor output) {
    const int batch_size = images.dimension(0);
    const int64 in_height = images.dimension(1);
    const int64 in_width = images.dimension(2);
    const int channels = images.dimension(3);

    ASSERT_EQ(batch_size, output.dimension(0));
    ASSERT_EQ(channels, output.dimension(3));

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    std::array<float, 4> coeff = {{0.0, 0.0, 0.0, 0.0}};
    for (int64 b = 0; b < batch_size; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        std::array<float, 4> y_weights;
        std::array<int64, 4> y_indices;
        GetWeightsAndIndices(height_scale, y, in_height, &y_weights,
                             &y_indices);
        for (int64 x = 0; x < out_width; ++x) {
          std::array<float, 4> x_weights;
          std::array<int64, 4> x_indices;
          GetWeightsAndIndices(width_scale, x, in_width, &x_weights,
                               &x_indices);
          for (int64 c = 0; c < channels; ++c) {
            // Use a 4x4 patch to compute the interpolated output value at
            // (b, y, x, c).
            for (int64 i = 0; i < 4; ++i) {
              const std::array<float, 4> values = {
                  {static_cast<float>(images(b, y_indices[i], x_indices[0], c)),
                   static_cast<float>(images(b, y_indices[i], x_indices[1], c)),
                   static_cast<float>(images(b, y_indices[i], x_indices[2], c)),
                   static_cast<float>(
                       images(b, y_indices[i], x_indices[3], c))}};
              coeff[i] = Interpolate1D(x_weights, values);
            }
            output(b, y, x, c) = Interpolate1D(y_weights, coeff);
          }
        }
      }
    }
  }

 protected:
  void RunRandomTest(const int batch_size, const int64 in_height,
                     const int64 in_width, const int target_height,
                     const int target_width, int channels) {
    LOG(INFO) << "Running random test " << in_height << "x" << in_width << "x"
              << channels << " to " << target_height << "x" << target_width
              << "x" << channels;
    const Tensor* input = SetRandomImageInput(
        TensorShape({batch_size, in_height, in_width, channels}));
    AddInputFromArray<int32>(TensorShape({2}), {target_height, target_width});

    TF_ASSERT_OK(RunOpKernel());

    std::unique_ptr<Tensor> expected(new Tensor(
        device_->GetAllocator(AllocatorAttributes()),
        DataTypeToEnum<float>::v(),
        TensorShape({batch_size, target_height, target_width, channels})));

    ResizeBicubicBaseline(input->tensor<float, 4>(),
                          expected->tensor<float, 4>());
    // Note: the baseline implementation reduces first in the x direction, and
    // then in the y direction. The optimized version reduces first in the y
    // direction, and then the X direction. As a result, there may be
    // some slight floating point inaccuracies. We thus ensure we're within
    // 0.00001 of the previous implementation.
    test::ExpectTensorNear<float>(*expected, *GetOutput(0), 0.00001);
  }

  void RunManyRandomTests(int channels) {
    for (int batch_size : {1, 2, 5}) {
      for (int in_w : {2, 4, 7, 20, 165}) {
        for (int in_h : {1, 3, 5, 8, 100, 233}) {
          for (int target_height : {1, 2, 3, 50, 113}) {
            for (int target_width : {target_height, target_height / 2 + 1}) {
              RunRandomTest(batch_size, in_h, in_w, target_height, target_width,
                            channels);
            }
          }
        }
      }
    }
  }
};

TEST_F(ResizeBicubicOpTest, TestBicubic2x2To1x1) {
  // Input:
  // 1, 2
  // 3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBicubicOpTest, TestBicubic2x2To0x0) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});

  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "Invalid argument: output dimensions must be positive"))
      << s;
}

TEST_F(ResizeBicubicOpTest, TestBicubicRandom141x186) {
  RunRandomTest(2, 141, 186, 299, 299, 1 /* channels */);
  RunRandomTest(2, 141, 186, 299, 299, 3 /* channels */);
}

TEST_F(ResizeBicubicOpTest, TestBicubicRandom183x229) {
  RunRandomTest(2, 183, 229, 299, 299, 1 /* channels */);
  RunRandomTest(2, 183, 229, 299, 299, 3 /* channels */);
}

TEST_F(ResizeBicubicOpTest, TestBicubicRandom749x603) {
  RunRandomTest(2, 749, 603, 299, 299, 1 /* channels */);
  RunRandomTest(2, 749, 603, 299, 299, 3 /* channels */);
}

TEST_F(ResizeBicubicOpTest, TestAreaRandomDataSeveralInputsSizes1Channel) {
  RunManyRandomTests(1);
}

TEST_F(ResizeBicubicOpTest, TestAreaRandomDataSeveralInputsSizes3Channels) {
  RunManyRandomTests(3);
}

TEST_F(ResizeBicubicOpTest, TestAreaRandomDataSeveralInputsSizes4Channels) {
  RunManyRandomTests(4);
}

static Graph* ResizeBicubic(int batch_size, int size, int channels,
                            float scale_y = 0.3, float scale_x = 0.7) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, TensorShape({batch_size, size, size, channels}));
  input.flat<float>().setRandom();
  Tensor shape(DT_INT32, TensorShape({2}));
  auto shape_t = shape.flat<int32>();
  shape_t(0) = scale_y * size;
  shape_t(1) = scale_x * size;
  test::graph::Binary(g, "ResizeBicubic", test::graph::Constant(g, input),
                      test::graph::Constant(g, shape));
  return g;
}

#define BM_ResizeBicubicDev(BATCH, SIZE, CHANNELS)                            \
  static void BM_ResizeBicubic##_##BATCH##_##SIZE##_##CHANNELS(int iters) {   \
    testing::ItemsProcessed(static_cast<int64>(iters) * BATCH * SIZE * SIZE * \
                            CHANNELS);                                        \
    test::Benchmark("cpu", ResizeBicubic(BATCH, SIZE, CHANNELS)).Run(iters);  \
  }                                                                           \
  BENCHMARK(BM_ResizeBicubic##_##BATCH##_##SIZE##_##CHANNELS);

BM_ResizeBicubicDev(8, 32, 3);
BM_ResizeBicubicDev(8, 128, 3);
BM_ResizeBicubicDev(8, 512, 3);
BM_ResizeBicubicDev(8, 1024, 3);
BM_ResizeBicubicDev(16, 32, 3);
BM_ResizeBicubicDev(16, 128, 3);
BM_ResizeBicubicDev(16, 512, 3);
BM_ResizeBicubicDev(16, 1024, 3);
BM_ResizeBicubicDev(32, 32, 3);
BM_ResizeBicubicDev(32, 128, 3);
BM_ResizeBicubicDev(32, 512, 3);
BM_ResizeBicubicDev(32, 1024, 3);

#define BM_ResizeBicubicExpand(BATCH, SIZE, CHANNELS)                         \
  static void BM_ResizeBicubicExpand##_##BATCH##_##SIZE##_##CHANNELS(         \
      int iters) {                                                            \
    testing::ItemsProcessed(static_cast<int64>(iters) * BATCH * SIZE * SIZE * \
                            CHANNELS * 8 * 8);                                \
    test::Benchmark("cpu", ResizeBicubic(BATCH, SIZE, CHANNELS, 8, 8))        \
        .Run(iters);                                                          \
  }                                                                           \
  BENCHMARK(BM_ResizeBicubicExpand##_##BATCH##_##SIZE##_##CHANNELS);

BM_ResizeBicubicExpand(12, 48, 1);
BM_ResizeBicubicExpand(12, 48, 3);
BM_ResizeBicubicExpand(12, 48, 40);

}  // end namespace tensorflow
