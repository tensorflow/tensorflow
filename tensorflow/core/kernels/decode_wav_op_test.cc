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

using namespace ops;  // NOLINT(build/namespaces)

TEST(DecodeWavOpTest, DecodeWavTest) {
  Scope root = Scope::NewRootScope();

  std::vector<uint8> wav_data = {
      'R',  'I',  'F', 'F', 44,  0,   0,   0,  // size of whole file - 8
      'W',  'A',  'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0,
      0,                   // size of fmt block - 8: 24 - 8
      1,    0,             // format: PCM (1)
      1,    0,             // channels: 1
      0x13, 0x37, 0,   0,  // sample rate: 14099
      0x26, 0x6e, 0,   0,  // byte rate: 2 * 14099
      2,    0,             // block align: NumChannels * BytesPerSample
      16,   0,             // bits per sample: 2 * 8
      'd',  'a',  't', 'a', 8,   0,   0,   0,  // size of payload: 8
      0,    0,                                 // first sample: 0
      0xff, 0x3f,                              // second sample: 16383
      0xff, 0x7f,  // third sample: 32767 (saturated)
      0x00, 0x80,  // fourth sample: -32768 (saturated)
  };
  Tensor content_tensor =
      test::AsScalar<string>(string(wav_data.begin(), wav_data.end()));
  Output content_op =
      Const(root.WithOpName("content_op"), Input::Initializer(content_tensor));

  DecodeWav decode_wav_op =
      DecodeWav(root.WithOpName("decode_wav_op"), content_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run(ClientSession::FeedType(),
                           {decode_wav_op.audio, decode_wav_op.sample_rate},
                           &outputs));

  const Tensor& audio = outputs[0];
  const int sample_rate = outputs[1].flat<int32>()(0);

  EXPECT_EQ(2, audio.dims());
  EXPECT_EQ(1, audio.dim_size(1));
  EXPECT_EQ(4, audio.dim_size(0));
  EXPECT_NEAR(0.0f, audio.flat<float>()(0), 1e-4f);
  EXPECT_NEAR(0.5f, audio.flat<float>()(1), 1e-4f);
  EXPECT_NEAR(1.0f, audio.flat<float>()(2), 1e-4f);
  EXPECT_NEAR(-1.0f, audio.flat<float>()(3), 1e-4f);
  EXPECT_EQ(14099, sample_rate);
}

}  // namespace tensorflow
