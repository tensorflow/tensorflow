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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/audio_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace ops {
namespace {

TEST(EncodeWavOpTest, EncodeWavTest) {
  Scope root = Scope::DisabledShapeInferenceScope();

  Tensor audio_tensor(DT_FLOAT, {4, 2});
  test::FillValues<float>(
      &audio_tensor, {0.0f, 0.5f, 1.0f, -1.0f, 0.25f, 0.75f, 1.25f, -0.5f});
  Output audio_op =
      Const(root.WithOpName("audio_op"), Input::Initializer(audio_tensor));

  Output sample_rate_op = Const(root.WithOpName("sample_rate_op"), 44100);

  EncodeWav encode_wav_op =
      EncodeWav(root.WithOpName("encode_wav_op"), audio_op, sample_rate_op);

  DecodeWav decode_wav_op =
      DecodeWav(root.WithOpName("decode_wav_op"), encode_wav_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run(ClientSession::FeedType(),
                           {decode_wav_op.audio, decode_wav_op.sample_rate},
                           &outputs));

  const Tensor& audio = outputs[0];
  const int sample_rate = outputs[1].flat<int32>()(0);

  EXPECT_EQ(2, audio.dims());
  EXPECT_EQ(2, audio.dim_size(1));
  EXPECT_EQ(4, audio.dim_size(0));
  EXPECT_NEAR(0.0f, audio.flat<float>()(0), 1e-4f);
  EXPECT_NEAR(0.5f, audio.flat<float>()(1), 1e-4f);
  EXPECT_NEAR(1.0f, audio.flat<float>()(2), 1e-4f);
  EXPECT_NEAR(-1.0f, audio.flat<float>()(3), 1e-4f);
  EXPECT_NEAR(0.25f, audio.flat<float>()(4), 1e-4f);
  EXPECT_NEAR(0.75f, audio.flat<float>()(5), 1e-4f);
  EXPECT_NEAR(1.0f, audio.flat<float>()(6), 1e-4f);
  EXPECT_NEAR(-0.5f, audio.flat<float>()(7), 1e-4f);
  EXPECT_EQ(44100, sample_rate);
}

}  // namespace
}  // namespace ops
}  // namespace tensorflow
