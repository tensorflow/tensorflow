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

#include "tensorflow/contrib/image/kernels/adjust_hsv_in_yiq_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class AdjustHsvInYiqOpTest : public OpsTestBase {
 protected:
};

TEST_F(AdjustHsvInYiqOpTest, IdentiyTransformMatrix) {
  Tensor matrix(allocator(), DT_FLOAT, TensorShape({9}));
  internal::compute_tranformation_matrix<9>(0.0, 1.0, 1.0,
                                            matrix.flat<float>().data());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({9}));
  test::FillValues<float>(&expected, {1, 0, 0, 0, 1, 0, 0, 0, 1});
  test::ExpectClose(matrix, expected);
}

TEST_F(AdjustHsvInYiqOpTest, ScaleValueTransformMatrix) {
  float scale_v = 2.3;
  Tensor matrix(allocator(), DT_FLOAT, TensorShape({9}));
  internal::compute_tranformation_matrix<9>(0.0, 1.0, scale_v,
                                            matrix.flat<float>().data());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({9}));
  test::FillValues<float>(&expected,
                          {scale_v, 0, 0, 0, scale_v, 0, 0, 0, scale_v});
  test::ExpectClose(matrix, expected);
}

}  // end namespace tensorflow
