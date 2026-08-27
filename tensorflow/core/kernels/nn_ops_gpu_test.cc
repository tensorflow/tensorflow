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

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <cstdint>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class MaxPoolingGpuTest : public OpsTestBase {};

TEST_F(MaxPoolingGpuTest, MaxPoolGradGradWithArgmaxOutOfBounds) {
  SetDevice(DEVICE_GPU,
            DeviceFactory::NewDevice("GPU", {}, "/job:a/replica:0/task:0"));
  DataType dtype = DT_FLOAT;
  TF_ASSERT_OK(NodeDefBuilder("maxpoolgradgradwithargmax_op",
                              "MaxPoolGradGradWithArgmax")
                   .Input(FakeInput(dtype))
                   .Input(FakeInput(dtype))
                   .Input(FakeInput(DT_INT64))
                   .Attr("ksize", {1, 1, 1, 1})
                   .Attr("strides", {1, 1, 1, 1})
                   .Attr("padding", "SAME")
                   .Attr("include_batch_in_index", false)
                   .Attr("T", dtype)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 1, 1}), {42.0});
  AddInputFromArray<float>(TensorShape({1, 1, 1, 1}), {1.0});
  AddInputFromArray<int64_t>(TensorShape({1, 1, 1, 1}), {-1});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {0.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow
