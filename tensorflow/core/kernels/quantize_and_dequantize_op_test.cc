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
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class QuantizeAndDequantizeTest : public OpsTestBase {};

// Convert a simple scalar tensor.
TEST_F(QuantizeAndDequantizeTest, Convert_scalar_tensor) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantize")
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Attr("input_min", 0.0)
          .Attr("input_max", 0.0)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1}), {-3.5});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&expected, {-3.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// Convert a 1D tensor with signed 8 bits.
TEST_F(QuantizeAndDequantizeTest, Convert_1D_tensor_with_int8) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantize")
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Attr("input_min", 0.0)
          .Attr("input_max", 0.0)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {-1, -0.5, 0, 0.3, 0.8, 0.555});

  // With int8, the tensor is quantized to {-127, -63, 0, 38, 102, 70}.
  // Scale is: 1/127
  // Then it is dequantized to {-1, -63.0/127, 0, 38.0/127, 102.0/127, 70.0/127}
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(
      &expected, {-1, -63.0 / 127, 0, 38.0 / 127, 102.0 / 127, 70.0 / 127});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 1D tensor with signed 4 bits.
TEST_F(QuantizeAndDequantizeTest, Convert_1D_tensor_with_int4) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantize")
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 4)
          .Attr("range_given", false)
          .Attr("input_min", 0.0)
          .Attr("input_max", 0.0)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {-1, -0.5, 0, 0.3, 0.8, 0.555});

  // With int4, the tensor is quantized to {-7, -3, 0, 2, 6, 4}.
  // Scale is: 1/7
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected,
                          {-1, -3.0 / 7, 0, 2.0 / 7, 6.0 / 7, 4.0 / 7});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 2D tensor with signed 8 bits with given range.
TEST_F(QuantizeAndDequantizeTest, Convert_2D_tensor_with_int8_range_given) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantize")
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Attr("input_min", -1.0)
          .Attr("input_max", 1.0)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // Note that the last two values are saturated.
  AddInputFromArray<float>(TensorShape({2, 4}),
                           {-0.8, -0.5, 0, 0.3, 0.8, 0.555, -2, 33});

  // Note that the range is given as [-1, 1].
  // With int8, the tensor is quantized to {-102, -63, 0, 38, 102, 70, -127,
  // 127}.
  // Scale is: 1/127
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 4}));
  test::FillValues<float>(&expected, {-102.0 / 127, -63.0 / 127, 0, 38.0 / 127,
                                      102.0 / 127, 70.0 / 127, -1, 1});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 4D tensor with unsigned 8 bits with given range.
TEST_F(QuantizeAndDequantizeTest, Convert_4D_tensor_with_uint8_range_given) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantize")
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", false)
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Attr("input_min", 0.0)
          .Attr("input_max", 1.0)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {-0.5, 0, 0.3, 0.8});

  // Note that the range is given as [0, 1].
  // With int8, the tensor is quantized to {0, 0, 77, 204}
  // Scale is: 1/255
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 1, 1}));
  test::FillValues<float>(&expected, {0, 0, 77.0 / 255, 204.0 / 255});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a tensor with all 0.
TEST_F(QuantizeAndDequantizeTest, Convert_tensor_with_all_0) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantize")
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", false)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Attr("input_min", 0.0)
          .Attr("input_max", 1.0)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {0, 0, 0, 0});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 1, 1}));
  test::FillValues<float>(&expected, {0, 0, 0, 0});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Range is invalid
TEST_F(QuantizeAndDequantizeTest, Invalid_range_given) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantize")
          .Input(FakeInput(DT_FLOAT))
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Attr("input_min", 2.0)
          .Attr("input_max", 1.0)
          .Finalize(node_def()));
  Status s = InitOp();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Invalid range: input_min 2 > input_max 1"))

      << s;
}

#define BM_SIMPLE_QUAN_DEQUAN(DEVICE)                     \
  static void BM_SIMPLE_QUAN_DEQUAN_##DEVICE(int iters) { \
    auto root = Scope::NewRootScope().ExitOnError();      \
    ops::QuantizeAndDequantize(root, {-3.5} /* input */); \
    TF_CHECK_OK(root.status());                           \
    Graph* g = new Graph(OpRegistry::Global());           \
    TF_CHECK_OK(root.ToGraph(g));                         \
    test::Benchmark(#DEVICE, g).Run(iters);               \
  }                                                       \
  BENCHMARK(BM_SIMPLE_QUAN_DEQUAN_##DEVICE);

BM_SIMPLE_QUAN_DEQUAN(cpu);
BM_SIMPLE_QUAN_DEQUAN(gpu);

}  // namespace
}  // namespace tensorflow
