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

TEST(MfccOpTest, SimpleTest) {
  Scope root = Scope::DisabledShapeInferenceScope();

  Tensor spectrogram_tensor(DT_FLOAT, TensorShape({1, 1, 513}));
  test::FillIota<float>(&spectrogram_tensor, 1.0f);

  Output spectrogram_const_op = Const(root.WithOpName("spectrogram_const_op"),
                                      Input::Initializer(spectrogram_tensor));

  Output sample_rate_const_op =
      Const(root.WithOpName("sample_rate_const_op"), 22050);

  Mfcc mfcc_op = Mfcc(root.WithOpName("mfcc_op"), spectrogram_const_op,
                      sample_rate_const_op);

  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(
      session.Run(ClientSession::FeedType(), {mfcc_op.output}, &outputs));

  const Tensor& mfcc_tensor = outputs[0];

  EXPECT_EQ(3, mfcc_tensor.dims());
  EXPECT_EQ(13, mfcc_tensor.dim_size(2));
  EXPECT_EQ(1, mfcc_tensor.dim_size(1));
  EXPECT_EQ(1, mfcc_tensor.dim_size(0));

  test::ExpectTensorNear<float>(
      mfcc_tensor,
      test::AsTensor<float>(
          {29.13970072, -6.41568601, -0.61903012, -0.96778652, -0.26819878,
           -0.40907028, -0.15614748, -0.23203119, -0.10481487, -0.1543029,
           -0.0769791, -0.10806114, -0.06047613},
          TensorShape({1, 1, 13})),
      1e-3);
}

}  // namespace tensorflow
