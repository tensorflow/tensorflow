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
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/image/sampling_kernels.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
using Eigen::Vector2f;

class DynamicKernel {
 public:
  virtual ~DynamicKernel() {}
  virtual float Value(const float x) const = 0;
  virtual float Radius() const = 0;
};

// Wraps a sampling kernel in a common interface.
template <typename KernelType>
class TypedDynamicKernel : public DynamicKernel {
 public:
  explicit TypedDynamicKernel(const KernelType& kernel) : kernel_(kernel) {}
  float Value(const float x) const override { return kernel_(x); }
  float Radius() const override { return kernel_.Radius(); }
  const KernelType kernel_;
};

template <typename KernelType>
std::unique_ptr<const DynamicKernel> CreateKernel(const KernelType& kernel) {
  return std::make_unique<TypedDynamicKernel<KernelType>>(kernel);
}

std::unique_ptr<const DynamicKernel> Create(
    functor::SamplingKernelType kernel_type) {
  switch (kernel_type) {
    case functor::Lanczos1Kernel:
      return CreateKernel(functor::CreateLanczos1Kernel());
    case functor::Lanczos3Kernel:
      return CreateKernel(functor::CreateLanczos3Kernel());
    case functor::Lanczos5Kernel:
      return CreateKernel(functor::CreateLanczos5Kernel());
    case functor::GaussianKernel:
      return CreateKernel(functor::CreateGaussianKernel());
    case functor::BoxKernel:
      return CreateKernel(functor::CreateBoxKernel());
    case functor::TriangleKernel:
      return CreateKernel(functor::CreateTriangleKernel());
    case functor::KeysCubicKernel:
      return CreateKernel(functor::CreateKeysCubicKernel());
    case functor::MitchellCubicKernel:
      return CreateKernel(functor::CreateMitchellCubicKernel());
    default:
      LOG(FATAL) << "Unknown kernel type.";
      return nullptr;
  }
}

template <typename T>
inline const T& Clamp(const T& low, const T& high, const T& value) {
  return std::min(high, std::max(low, value));
}

// Samples from the image at the passed batch at pixel location sample_f with a
// kernel scaled by scale.
void Sample(const DynamicKernel& kernel, const bool antialias,
            TTypes<float, 4>::Tensor images, const int batch,
            const Vector2f& scale, const Vector2f& sample_f, float* dest) {
  const Vector2f kernel_scale(antialias ? std::max(scale.x(), 1.0f) : 1.0,
                              antialias ? std::max(scale.y(), 1.0f) : 1.0);

  const int64_t in_height = images.dimension(1);
  const int64_t in_width = images.dimension(2);
  const int channels = images.dimension(3);
  const int64_t y_span_start = Clamp(
      static_cast<int64_t>(0), in_height - 1,
      static_cast<int64_t>(
          std::ceil(sample_f.y() - kernel.Radius() * kernel_scale.y() - 0.5f)));
  const int64_t y_span_end =
      Clamp(static_cast<int64_t>(0), in_height - 1,
            static_cast<int64_t>(std::floor(
                sample_f.y() + kernel.Radius() * kernel_scale.y() - 0.5f))) +
      1;
  const int64_t x_span_start = Clamp(
      static_cast<int64_t>(0), in_width - 1,
      static_cast<int64_t>(
          std::ceil(sample_f.x() - kernel.Radius() * kernel_scale.x() - 0.5f)));

  const int64_t x_span_end =
      Clamp(static_cast<int64_t>(0), in_width - 1,
            static_cast<int64_t>(std::floor(
                sample_f.x() + kernel.Radius() * kernel_scale.x() - 0.5f))) +
      1;

  std::fill(dest, dest + channels, 0.0f);
  if (sample_f.x() < 0.0f || sample_f.y() < 0.0f || sample_f.x() > in_width ||
      sample_f.y() > in_height) {
    return;
  }
  const Vector2f one_over_kernel_scale(1.0f / kernel_scale.x(),
                                       1.0f / kernel_scale.y());
  float total_weight = 0.0f;
  for (int64_t y = y_span_start; y < y_span_end; ++y) {
    float y_kernel_pos = static_cast<float>(y) + 0.5f - sample_f.y();
    float y_weight = kernel.Value(y_kernel_pos * one_over_kernel_scale.y());
    for (int64_t x = x_span_start; x < x_span_end; ++x) {
      float x_kernel_pos = static_cast<float>(x) + 0.5f - sample_f.x();
      float x_weight = kernel.Value(x_kernel_pos * one_over_kernel_scale.x());
      float kernel_weight = y_weight * x_weight;
      total_weight += kernel_weight;
      for (int c = 0; c < channels; ++c) {
        dest[c] += static_cast<float>(images(batch, y, x, c)) * kernel_weight;
      }
    }
  }
  if (std::abs(total_weight) >= 1000.0f * std::numeric_limits<float>::min()) {
    CHECK_NE(total_weight, 0.0f) << y_span_start << "," << y_span_end << " "
                                 << x_span_start << "," << x_span_end;
    for (int c = 0; c < channels; ++c) {
      dest[c] /= total_weight;
    }
  }
}

// This is the straight forward unoptimized implementation of ScaleAndTranslate
// We use this to confirm that the optimized version is almost identical. The
// only difference will be small floating point differences, since this version
// does not to separable passes in x and y dimensions.
void ScaleAndTranslateBaseline(const DynamicKernel& kernel,
                               const bool antialias,
                               TTypes<float, 4>::Tensor images,
                               const Vector2f& orig_scale,
                               const Vector2f& orig_translate,
                               TTypes<float, 4>::Tensor output) {
  const Vector2f scale(1.0f / orig_scale[0], 1.0f / orig_scale[1]);
  const Vector2f translate(-orig_translate[0] / orig_scale[0],
                           -orig_translate[1] / orig_scale[1]);

  const int batch = images.dimension(0);
  const int channels = images.dimension(3);

  ASSERT_EQ(batch, output.dimension(0));
  ASSERT_EQ(channels, output.dimension(3));

  const int64_t out_height = output.dimension(1);
  const int64_t out_width = output.dimension(2);
  const int64_t in_height = images.dimension(1);
  const int64_t in_width = images.dimension(2);

  for (int b = 0; b < batch; ++b) {
    for (int64_t y = 0; y < out_height; ++y) {
      const float out_y_f = static_cast<float>(y) + 0.5;
      const float in_y_f = out_y_f * scale.y() + translate.y();
      for (int64_t x = 0; x < out_width; ++x) {
        const float out_x_f = static_cast<float>(x) + 0.5;
        const float in_x_f = out_x_f * scale.x() + translate.x();
        if (in_x_f < 0.0f || in_y_f < 0.0f || in_x_f > in_width ||
            in_y_f > in_height) {
          std::fill(&output(b, y, x, 0), &output(b, y, x + 1, 0), 0.0f);
        } else {
          Sample(kernel, antialias, images, b, scale, Vector2f(in_x_f, in_y_f),
                 &output(b, y, x, 0));
        }
      }
    }
  }
}

class ScaleAndTranslateOpTest : public OpsTestBase {
 protected:
  void CreateOp(const string& kernel_type_str, const bool antialias) {
    TF_EXPECT_OK(NodeDefBuilder("scale_and_translate_op", "ScaleAndTranslate")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("kernel_type", kernel_type_str)
                     .Attr("antialias", antialias)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    kernel_type_ = functor::SamplingKernelTypeFromString(kernel_type_str);
    antialias_ = antialias;
  }

  void SetCheckerboardImageInput(int batch_size, int num_row_squares,
                                 int num_col_squares, int square_size,
                                 int num_channels) {
    inputs_.clear();
    std::vector<float> data;
    const int64_t row_size = num_col_squares * square_size * num_channels;
    const int64_t image_size = num_row_squares * square_size * row_size;
    data.resize(batch_size * image_size);
    random::PhiloxRandom philox(42);
    random::SimplePhilox rnd(&philox);
    std::vector<float> col(num_channels);
    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < num_row_squares; ++y) {
        for (int x = 0; x < num_col_squares; ++x) {
          for (int n = 0; n < num_channels; ++n) {
            col[n] = rnd.RandFloat();
          }
          for (int r = y * square_size; r < (y + 1) * square_size; ++r) {
            auto it = data.begin() + b * image_size + r * row_size +
                      x * square_size * num_channels;
            for (int n = 0; n < square_size; ++n) {
              for (int chan = 0; chan < num_channels; ++chan, ++it) {
                *it = col[chan] * 255.0;
              }
            }
          }
        }
      }
    }
    AddInputFromArray<float>(
        TensorShape({batch_size, num_row_squares * square_size,
                     num_col_squares * square_size, num_channels}),
        data);
  }

  void RunTest(int output_image_height, int output_image_width,
               const Vector2f& scale, const Vector2f& translate) {
    AddInputFromArray<int32>(TensorShape({2}),
                             {output_image_height, output_image_width});
    AddInputFromArray<float>(TensorShape({2}), {scale[1], scale[0]});
    AddInputFromArray<float>(TensorShape({2}), {translate[1], translate[0]});
    absl::Status s = RunOpKernel();
    const int batch_size = GetOutput(0)->dim_size(0);
    const int channels = GetOutput(0)->dim_size(3);
    Tensor expected(allocator(), DT_FLOAT,
                    TensorShape({batch_size, output_image_height,
                                 output_image_width, channels}));

    std::unique_ptr<const DynamicKernel> kernel = Create(kernel_type_);
    ScaleAndTranslateBaseline(*kernel, antialias_,
                              mutable_input(0)->tensor<float, 4>(), scale,
                              translate, expected.tensor<float, 4>());
    constexpr double kAbs = 1e-2f;
    test::ExpectTensorNear<float>(expected, *GetOutput(0), kAbs);
  }

  functor::SamplingKernelType kernel_type_;
  bool antialias_;
};

TEST_F(ScaleAndTranslateOpTest, IdentityTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = kNumRowSquares * kSquareSize;
  constexpr int kOutputImageWidth = kNumColSquares * kSquareSize;
  const Vector2f kScale(1.0f, 1.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, UpsampleTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = kNumRowSquares * kSquareSize * 2;
  constexpr int kOutputImageWidth = kNumColSquares * kSquareSize * 2;
  const Vector2f kScale(2.0f, 2.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, DownsampleTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = kNumRowSquares * kSquareSize / 2;
  constexpr int kOutputImageWidth = kNumColSquares * kSquareSize / 2;
  const Vector2f kScale(0.5f, 0.5f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, AntiAliasedDownsampleToASinglePixelTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 1;
  constexpr int kOutputImageWidth = 1;
  const Vector2f kScale(1.0f / (kNumRowSquares * kSquareSize),
                        1.0f / (kNumColSquares * kSquareSize));
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, NonAntiAliasedDownsampleToASinglePixelTest) {
  CreateOp("lanczos3", false);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 16;
  constexpr int64_t kNumColSquares = 13;
  constexpr int64_t kSquareSize = 12;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 1;
  constexpr int kOutputImageWidth = 1;
  const Vector2f kScale(1.0f / (kNumRowSquares * kSquareSize),
                        1.0f / (kNumColSquares * kSquareSize));
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, UsampleFromASinglePixelTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 1;
  constexpr int64_t kNumColSquares = 1;
  constexpr int64_t kSquareSize = 1;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 10;
  constexpr int kOutputImageWidth = 17;
  const Vector2f kScale(17.0f, 10.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, NonAntialiasedUsampleFromASinglePixelTest) {
  CreateOp("lanczos3", false);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 1;
  constexpr int64_t kNumColSquares = 1;
  constexpr int64_t kSquareSize = 1;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 10;
  constexpr int kOutputImageWidth = 17;
  const Vector2f kScale(17.0f, 10.0f);
  const Vector2f kTranslate(0.0f, 0.0f);
  // Anti-aliasing shouldn't have any effect here, verify by comparing with the
  // ground truth with anti-aliasing turned on.
  antialias_ = true;
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, AntialiasedScaleAndTranslationTest) {
  CreateOp("lanczos3", true);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 11;
  constexpr int64_t kNumColSquares = 7;
  constexpr int64_t kSquareSize = 5;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 49;
  constexpr int kOutputImageWidth = 51;
  const Vector2f kScale(1.25f, 0.6f);
  const Vector2f kTranslate(4.1f, -3.1f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, NonAntialiasedScaleAndTranslationTest) {
  CreateOp("lanczos3", false);
  constexpr int64_t kBatchSize = 2;
  constexpr int64_t kNumRowSquares = 11;
  constexpr int64_t kNumColSquares = 7;
  constexpr int64_t kSquareSize = 5;
  constexpr int64_t kNumChannels = 3;
  SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                            kSquareSize, kNumChannels);
  constexpr int kOutputImageHeight = 49;
  constexpr int kOutputImageWidth = 51;
  const Vector2f kScale(1.25f, 0.6f);
  const Vector2f kTranslate(4.1f, -3.1f);
  RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
}

TEST_F(ScaleAndTranslateOpTest, TestKernelTypes) {
  const std::vector<string> kKernelTypes = {
      "lanczos1", "lanczos3",  "lanczos5",     "box",
      "triangle", "keyscubic", "mitchellcubic"};
  for (const string& kernel_type : kKernelTypes) {
    CreateOp(kernel_type, true);
    constexpr int64_t kBatchSize = 2;
    constexpr int64_t kNumRowSquares = 10;
    constexpr int64_t kNumColSquares = 11;
    constexpr int64_t kSquareSize = 1;
    constexpr int64_t kNumChannels = 3;
    SetCheckerboardImageInput(kBatchSize, kNumRowSquares, kNumColSquares,
                              kSquareSize, kNumChannels);
    constexpr int kOutputImageHeight = 9;
    constexpr int kOutputImageWidth = 11;
    const Vector2f kScale(1.9f, 1.9f);
    const Vector2f kTranslate(0.3f, 2.1f);
    RunTest(kOutputImageHeight, kOutputImageWidth, kScale, kTranslate);
  }
}

}  // namespace tensorflow
