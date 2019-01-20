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

#include "tensorflow/core/kernels/eigen_softmax.h"
#include "tensorflow/core/platform/test.h"

namespace Eigen {

namespace {
void EigenApprox(float a, float b) {
  ASSERT_TRUE(std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * 1e-3);
}
}  // namespace

TEST(EigenSoftmaxTest, Simple) {
  const int depth = 1024;
  const int batch = 32;
  const float beta = 1.2f;

  Tensor<float, 2> input(depth, batch);
  input = input.constant(11.0f) + input.random();

  Tensor<float, 2> reference(depth, batch);
  reference.setRandom();

  Eigen::array<int, 1> depth_dim;
  depth_dim[0] = 0;
  Eigen::array<int, 2> bcast;
  bcast[0] = depth;
  bcast[1] = 1;
  Tensor<float, 2>::Dimensions dims2d;
  dims2d[0] = 1;
  dims2d[1] = batch;
  reference =
      ((input -
        input.maximum(depth_dim).eval().reshape(dims2d).broadcast(bcast)) *
       beta)
          .exp();
  reference =
      reference /
      (reference.sum(depth_dim).eval().reshape(dims2d).broadcast(bcast));

  Tensor<float, 2> result = SoftMax(input, beta);

  for (int i = 0; i < depth; ++i) {
    for (int j = 0; j < batch; ++j) {
      EigenApprox(result(i, j), reference(i, j));
    }
  }
}

}  // namespace Eigen
