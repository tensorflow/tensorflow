/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

class UniformQuantizedClipByValueOpTest : public OpsTestBase {
 protected:
};

TEST_F(UniformQuantizedClipByValueOpTest, PerChannel) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedClipByValue")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("T", DT_QINT32)
          .Attr("quantization_axis", 1)
          .Attr("quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("quantization_max_val", static_cast<int32_t>(2147483647))
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({3}), {-1, -5, -1});
  AddInputFromArray<qint32>(TensorShape({3}), {1, 1, 5});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {-20, 0, 20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-1, -4, -1, 0, 1, 4});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedClipByValueOpTest, PerTensor) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedClipByValue")
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("T", DT_QINT32)
          .Attr("quantization_axis", -1)
          .Attr("quantization_min_val", static_cast<int32_t>(-2147483648))
          .Attr("quantization_max_val", static_cast<int32_t>(2147483647))
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({}), {-1});
  AddInputFromArray<qint32>(TensorShape({}), {1});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-1, -1, -1, 0, 1, 1});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

}  // namespace tensorflow
