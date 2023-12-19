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

#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

template <typename T>
struct BatchNormOpTest : public OpsTestBase {
  static constexpr auto TValueType = DataTypeToEnum<T>::value;

  void run_me() {
    TF_EXPECT_OK(
        NodeDefBuilder("batch_norm_op", "BatchNormWithGlobalNormalization")
            .Input(FakeInput(TValueType))
            .Input(FakeInput(TValueType))
            .Input(FakeInput(TValueType))
            .Input(FakeInput(TValueType))
            .Input(FakeInput(TValueType))
            .Attr("scale_after_normalization", false)
            .Attr("variance_epsilon", 0.001)
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOpWithGraphVersion(8));

    AddInputFromList<T>(TensorShape({1, 1, 6, 2}),
                        {1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6});
    AddInputFromList<T>(TensorShape({2}), {10, 20});
    AddInputFromList<T>(TensorShape({2}), {0.25, 0.5});
    AddInputFromList<T>(TensorShape({2}), {0.1, 0.6});
    AddInputFromList<T>(TensorShape({2}), {0.0, 0.0});

    TF_ASSERT_OK(RunOpKernel());

    double atol = TValueType == DT_FLOAT ? 0.01 : 0.1;

    Tensor expected(allocator(), TValueType, TensorShape({1, 1, 6, 2}));
    test::FillValues<T>(&expected,
                        {-17.86f, -22.00f, -15.87f, -20.59f, -13.87f, -19.18f,
                         -21.86f, -33.31f, -23.85f, -34.72f, -25.85f, -36.13f});
    test::ExpectTensorNear<T>(expected, *GetOutput(0), atol);
  }
};

TYPED_TEST_SUITE_P(BatchNormOpTest);

TYPED_TEST_P(BatchNormOpTest, Simple) { this->run_me(); }

REGISTER_TYPED_TEST_SUITE_P(BatchNormOpTest, Simple);

// TODO(ezhulenev): Add support for more data types.
using DataTypes = ::testing::Types<float, Eigen::half>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, BatchNormOpTest, DataTypes);

}  // namespace tensorflow
