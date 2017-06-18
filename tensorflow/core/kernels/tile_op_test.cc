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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class TileOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
    TF_ASSERT_OK(NodeDefBuilder("tile", "Tile")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(data_type))
                     .Attr("T", data_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(TileOpTest, Tile_1d) {
  MakeOp(DT_INT32);
  // Feed and run
  // [1, 2, 3]
  AddInputFromArray<int32>(TensorShape({3}), {1,2,3});
  AddInputFromArray<int32>(TensorShape({1}), {2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  // Should become
  // [1, 2, 3, 1, 2, 3]
  test::FillValues<int32>(
      &expected, {1,2,3,1,2,3});
  test::ExpectTensorEqual<int32>(expected, *output);
}

TEST_F(TileOpTest, Tile_2d) {
  MakeOp(DT_INT32);
  // Feed and run
  // [[1, 2, 3],
  //  [4, 5, 6]]
  AddInputFromArray<int32>(TensorShape({2,3}), {1,2,3,4,5,6});
  AddInputFromArray<int32>(TensorShape({2}), {2,2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  Tensor expected(allocator(), DT_INT32, TensorShape({4,6}));
  // Should become
  // [[1, 2, 3, 1, 2, 3],
  //  [4, 5, 6, 4, 5, 6],
  //  [1, 2, 3, 1, 2, 3],
  //  [4, 5, 6, 4, 5, 6],
  test::FillValues<int32>(
      &expected, {1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6});
  test::ExpectTensorEqual<int32>(expected, *output);
}

TEST_F(TileOpTest, Tile_2d_zero) {
  MakeOp(DT_INT32);
  // Feed and run
  // [[1, 2, 3],
  //  [4, 5, 6]]
  AddInputFromArray<int32>(TensorShape({2,3}), {1,2,3,4,5,6});
  AddInputFromArray<int32>(TensorShape({2}), {2,0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  Tensor expected(allocator(), DT_INT32, TensorShape({4,0}));
  // Should become
  // [],
  test::FillValues<int32>(
      &expected, {});
  test::ExpectTensorEqual<int32>(expected, *output);
}

TEST_F(TileOpTest, Tile_3d) {
  MakeOp(DT_INT32);
  // Feed and run
  // [[[1, 2, 3],
  //   [4, 5, 6]]]
  AddInputFromArray<int32>(TensorShape({1,2,3}),{1,2,3,4,5,6});
  AddInputFromArray<int32>(TensorShape({3}), {2,3,1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  Tensor expected(allocator(), DT_INT32, TensorShape({2,6,3}));
  // should become
  // [[[1, 2, 3],
  //   [4, 5, 6],
  //   [1, 2, 3],
  //   [4, 5, 6],
  //   [1, 2, 3],
  //   [4, 5, 6]],
  //  [[1, 2, 3],
  //   [4, 5, 6],
  //   [1, 2, 3],
  //   [4, 5, 6],
  //   [1, 2, 3],
  //   [4, 5, 6]]],
  test::FillValues<int32>(&expected,
                          {1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,
                           1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6});
  test::ExpectTensorEqual<int32>(expected, *output);
}

TEST_F(TileOpTest, Tile_9d) {
  MakeOp(DT_INT32);
  // Feed and run
  // [[[[[[[[[1, 2]]]]]]]]]
  AddInputFromArray<int32>(TensorShape({1,1,1,1,1,1,1,1,2}), {1,2});
  AddInputFromArray<int32>(TensorShape({9}), {2,1,1,1,1,1,1,2,1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  Tensor expected(allocator(), DT_INT32, TensorShape({2,1,1,1,1,1,1,2,2}));
  // should become
  // [[[[[[[[[1, 2],
  //         [1, 2]]]]]]]],
  //  [[[[[[[[1, 2],
  //         [1, 2]]]]]]]]]
  test::FillValues<int32>(&expected, {1,2,1,2,1,2,1,2});
  test::ExpectTensorEqual<int32>(expected, *output);
}

}  // namespace
}  // namespace tensorflow
