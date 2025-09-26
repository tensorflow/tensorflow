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

#define EIGEN_USE_THREADS

#include <functional>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/audio_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace ops {
namespace {

TEST(SpectrogramOpTest, SimpleTest) {
  Scope root = Scope::NewRootScope();

  Tensor audio_tensor(DT_FLOAT, TensorShape({8, 1}));
  test::FillValues<float>(&audio_tensor,
                          {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f});

  Output audio_const_op = Const(root.WithOpName("audio_const_op"),
                                Input::Initializer(audio_tensor));

  AudioSpectrogram spectrogram_op =
      AudioSpectrogram(root.WithOpName("spectrogram_op"), audio_const_op, 8, 1);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run(ClientSession::FeedType(),
                           {spectrogram_op.spectrogram}, &outputs));

  const Tensor& spectrogram_tensor = outputs[0];

  EXPECT_EQ(3, spectrogram_tensor.dims());
  EXPECT_EQ(5, spectrogram_tensor.dim_size(2));
  EXPECT_EQ(1, spectrogram_tensor.dim_size(1));
  EXPECT_EQ(1, spectrogram_tensor.dim_size(0));

  test::ExpectTensorNear<float>(
      spectrogram_tensor,
      test::AsTensor<float>({0, 1, 2, 1, 0}, TensorShape({1, 1, 5})), 1e-3);
}

TEST(SpectrogramOpTest, SquaredTest) {
  Scope root = Scope::NewRootScope();

  Tensor audio_tensor(DT_FLOAT, TensorShape({8, 1}));
  test::FillValues<float>(&audio_tensor,
                          {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f});

  Output audio_const_op = Const(root.WithOpName("audio_const_op"),
                                Input::Initializer(audio_tensor));

  AudioSpectrogram spectrogram_op =
      AudioSpectrogram(root.WithOpName("spectrogram_op"), audio_const_op, 8, 1,
                       AudioSpectrogram::Attrs().MagnitudeSquared(true));

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run(ClientSession::FeedType(),
                           {spectrogram_op.spectrogram}, &outputs));

  const Tensor& spectrogram_tensor = outputs[0];

  EXPECT_EQ(3, spectrogram_tensor.dims());
  EXPECT_EQ(5, spectrogram_tensor.dim_size(2));
  EXPECT_EQ(1, spectrogram_tensor.dim_size(1));
  EXPECT_EQ(1, spectrogram_tensor.dim_size(0));

  test::ExpectTensorNear<float>(
      spectrogram_tensor,
      test::AsTensor<float>({0, 1, 4, 1, 0}, TensorShape({1, 1, 5})), 1e-3);
}

TEST(SpectrogramOpTest, MultichannelTest) {
  Scope root = Scope::NewRootScope();

  const int audio_size = 8;
  const int channel_size = 2;
  Tensor audio_tensor(DT_FLOAT, TensorShape({audio_size, channel_size}));
  test::FillValues<float>(
      &audio_tensor, {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f,
                      -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f});

  Output audio_const_op = Const(root.WithOpName("audio_const_op"),
                                Input::Initializer(audio_tensor));

  AudioSpectrogram spectrogram_op =
      AudioSpectrogram(root.WithOpName("spectrogram_op"), audio_const_op,
                       audio_size, channel_size);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run(ClientSession::FeedType(),
                           {spectrogram_op.spectrogram}, &outputs));

  const Tensor& spectrogram_tensor = outputs[0];

  EXPECT_EQ(3, spectrogram_tensor.dims());
  EXPECT_EQ(5, spectrogram_tensor.dim_size(2));
  EXPECT_EQ(1, spectrogram_tensor.dim_size(1));
  EXPECT_EQ(channel_size, spectrogram_tensor.dim_size(0));

  for (int channel = 0; channel < channel_size; channel++) {
    test::ExpectTensorNear<float>(
        spectrogram_tensor.SubSlice(channel),
        test::AsTensor<float>({0, 1, 2, 1, 0}, TensorShape({1, 5})), 1e-3);
  }
}

TEST(SpectrogramOpTest, InvalidWindowSize) {
  Scope root = Scope::NewRootScope();
  const int audio_size = 8;
  const int channel_size = 2;
  Tensor audio_tensor(DT_FLOAT, TensorShape({audio_size, channel_size}));
  test::FillValues<float>(
      &audio_tensor, {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f,
                      -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  Output audio_const_op = Const(root.WithOpName("audio_const_op"),
                                Input::Initializer(audio_tensor));
  AudioSpectrogram spectrogram_op =
      AudioSpectrogram(root.WithOpName("spectrogram_op"), audio_const_op,
                       /*window_size=*/1, /*stride=*/1);
  EXPECT_THAT(root.status(),
              absl_testing::StatusIs(tsl::error::Code::INVALID_ARGUMENT,
                                     ::testing::ContainsRegex("window size")));
}

TEST(SpectrogramOpTest, InvalidStride) {
  Scope root = Scope::NewRootScope();
  const int audio_size = 8;
  const int channel_size = 2;
  Tensor audio_tensor(DT_FLOAT, TensorShape({audio_size, channel_size}));
  test::FillValues<float>(
      &audio_tensor, {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f,
                      -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  Output audio_const_op = Const(root.WithOpName("audio_const_op"),
                                Input::Initializer(audio_tensor));
  AudioSpectrogram spectrogram_op =
      AudioSpectrogram(root.WithOpName("spectrogram_op"), audio_const_op,
                       /*window_size=*/2, /*stride=*/0);
  EXPECT_THAT(root.status(),
              absl_testing::StatusIs(tsl::error::Code::INVALID_ARGUMENT,
                                     ::testing::ContainsRegex("stride")));
}

}  // namespace
}  // namespace ops
}  // namespace tensorflow
