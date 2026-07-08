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

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/version.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#endif

namespace tensorflow {

namespace {

class SparseToDenseTest : public OpsTestBase {
 protected:
  void MakeOp(int dim, DataType index_type, DataType value_type) {
    TF_ASSERT_OK(NodeDefBuilder("sparsetodense", "SparseToDense")
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(value_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseToDenseTest, OneD_OneValue) {
  MakeOp(1, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 3, 4});
  // output_shape
  AddInputFromArray<int32_t>(TensorShape({1}), {5});
  // sparse_values
  AddInputFromArray<float>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {5});
  test::FillValues<float>(&expected, {-2, 2, -2, 2, 2});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, OneD_OneValue_int64_double) {
  MakeOp(1, DT_INT64, DT_DOUBLE);

  // sparse_indices
  AddInputFromArray<int64_t>(TensorShape({3}), {1, 3, 4});
  // output_shape
  AddInputFromArray<int64_t>(TensorShape({1}), {5});
  // sparse_values
  AddInputFromArray<double>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<double>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, {5});
  test::FillValues<double>(&expected, {-2, 2, -2, 2, 2});
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, OneD_MultValues) {
  MakeOp(1, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32_t>({3}, {1, 3, 4});
  // output_shape
  AddInputFromArray<int32_t>({1}, {5});
  // sparse_values
  AddInputFromArray<float>({3}, {3, 4, 5});
  // default_value
  AddInputFromArray<float>({}, {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {5});
  test::FillValues<float>(&expected, {-2, 3, -2, 4, 5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, TwoD_OneValue) {
  MakeOp(2, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32_t>(TensorShape({3, 2}), {0, 1, 0, 2, 2, 3});
  // output_shape
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 4});
  // sparse_values
  AddInputFromArray<float>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 2>()(0, 1) = 2;
  expected.tensor<float, 2>()(0, 2) = 2;
  expected.tensor<float, 2>()(2, 3) = 2;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, TwoD_MultValues) {
  MakeOp(2, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32_t>(TensorShape({3, 2}), {0, 1, 0, 2, 2, 3});
  // output_shape
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 4});
  // sparse_values
  AddInputFromArray<float>(TensorShape({3}), {3, 4, 5});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 2>()(0, 1) = 3;
  expected.tensor<float, 2>()(0, 2) = 4;
  expected.tensor<float, 2>()(2, 3) = 5;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, ThreeD_OneValue) {
  MakeOp(3, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32_t>(TensorShape({3, 3}), {0, 1, 1, 0, 2, 0, 2, 3, 1});
  // output_shape
  AddInputFromArray<int32_t>(TensorShape({3}), {3, 4, 2});
  // sparse_values
  AddInputFromArray<float>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4, 2});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 3>()(0, 1, 1) = 2;
  expected.tensor<float, 3>()(0, 2, 0) = 2;
  expected.tensor<float, 3>()(2, 3, 1) = 2;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, ThreeD_MultValues) {
  MakeOp(3, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32_t>(TensorShape({3, 3}), {0, 1, 1, 0, 2, 0, 2, 3, 1});
  // output_shape
  AddInputFromArray<int32_t>(TensorShape({3}), {3, 4, 2});
  // sparse_values
  AddInputFromArray<float>(TensorShape({3}), {3, 4, 5});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4, 2});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 3>()(0, 1, 1) = 3;
  expected.tensor<float, 3>()(0, 2, 0) = 4;
  expected.tensor<float, 3>()(2, 3, 1) = 5;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, GPU_InvalidIndices) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  SetDevice(DEVICE_GPU,
            std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                "GPU", {}, "/job:a/replica:0/task:0")));

  // Test 1: Index out of bounds
  {
    MakeOp(1, DT_INT32, DT_FLOAT);
    // sparse_indices (6 is out of bounds for shape 5)
    AddInputFromArray<int32_t>(TensorShape({3}), {1, 6, 4});
    // output_shape
    AddInputFromArray<int32_t>(TensorShape({1}), {5});
    // sparse_values
    AddInputFromArray<float>(TensorShape({}), {2});
    // default_value
    AddInputFromArray<float>(TensorShape({}), {-2});

    absl::Status status = RunOpKernel();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("out of bounds"), std::string::npos);
  }

  // Test 2: Indices not sorted (out of order)
  {
    MakeOp(1, DT_INT32, DT_FLOAT);
    // sparse_indices (4 then 2 is out of order)
    AddInputFromArray<int32_t>(TensorShape({3}), {1, 4, 2});
    // output_shape
    AddInputFromArray<int32_t>(TensorShape({1}), {5});
    // sparse_values
    AddInputFromArray<float>(TensorShape({}), {2});
    // default_value
    AddInputFromArray<float>(TensorShape({}), {-2});

    absl::Status status = RunOpKernel();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("out of order"), std::string::npos);
  }

  // Test 3: Repeated indices
  {
    MakeOp(1, DT_INT32, DT_FLOAT);
    // sparse_indices (2 is repeated)
    AddInputFromArray<int32_t>(TensorShape({3}), {1, 2, 2});
    // output_shape
    AddInputFromArray<int32_t>(TensorShape({1}), {5});
    // sparse_values
    AddInputFromArray<float>(TensorShape({}), {2});
    // default_value
    AddInputFromArray<float>(TensorShape({}), {-2});

    absl::Status status = RunOpKernel();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(status.message().find("repeated"), std::string::npos);
  }
#endif
}

TEST(SparseToDenseFixTest, SourceCheck) {
  std::ifstream file(
      "third_party/tensorflow/core/kernels/sparse_to_dense_op_gpu.cu.cc");
  ASSERT_TRUE(file.is_open())
      << "Failed to open source file "
         "third_party/tensorflow/core/kernels/sparse_to_dense_op_gpu.cu.cc!";
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("valid_status_host_tensor"), std::string::npos)
      << "The UAF/pinned memory validation fix (valid_status_host_tensor) "
         "was not found in the GPU source file!";
}

}  // namespace

static void BM_SparseToDense(::testing::benchmark::State& state) {
  const int NDIM = state.range(0);
  const int N = state.range(1);

  // TODO(zhifengc): Switch to use kernel_benchmark_testlib.h

  const int IndexDim = (NDIM == 1) ? 0 : 1;

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  absl::InlinedVector<TensorValue, 4UL> inputs;

  // Create a dense tensor with dims [1, ..., 1, N]
  Tensor output_shape(DT_INT32, TensorShape({NDIM}));
  Tensor sparse_indices(DT_INT64, TensorShape({N, NDIM}));
  Tensor sparse_values(DT_FLOAT, TensorShape({N}));
  Tensor default_value(DT_FLOAT, TensorShape({}));
  auto output_shape_t = output_shape.vec<int32_t>();
  for (int d = 0; d < NDIM; ++d) {
    output_shape_t(d) = (d == IndexDim) ? N : 3;
  }

  auto sparse_indices_t = sparse_indices.matrix<int64_t>();
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < NDIM; ++d)
      sparse_indices_t(n, d) = (d == IndexDim) ? n : 0;
  }

  for (auto* ptr :
       {&sparse_indices, &output_shape, &sparse_values, &default_value}) {
    inputs.push_back({nullptr, ptr});
  }

  NodeDef sparse_node_def;
  TF_CHECK_OK(NodeDefBuilder("sparsetodense", "SparseToDense")
                  .Input(FakeInput(DT_INT32))
                  .Input(FakeInput(DT_INT32))
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&sparse_node_def));

  absl::Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), sparse_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  auto sparse_context = std::make_unique<OpKernelContext>(&params);
  op->Compute(sparse_context.get());
  for (auto s : state) {
    delete sparse_context->release_output(0).tensor;
    op->Compute(sparse_context.get());
    TF_ASSERT_OK(sparse_context->status());
  }

  // processing input, mainly
  int64_t bytes_per_iter = static_cast<int64_t>((N + N * NDIM) * sizeof(float));
  state.SetBytesProcessed(bytes_per_iter * state.iterations());
}

BENCHMARK(BM_SparseToDense)
    ->ArgPair(1, 10)
    ->ArgPair(1, 100)
    ->ArgPair(1, 1000)
    ->ArgPair(1, 10000)
    ->ArgPair(2, 10)
    ->ArgPair(2, 100)
    ->ArgPair(2, 1000)
    ->ArgPair(2, 10000)
    ->ArgPair(3, 10)
    ->ArgPair(3, 100)
    ->ArgPair(3, 1000)
    ->ArgPair(3, 10000)
    ->ArgPair(5, 10)
    ->ArgPair(5, 100)
    ->ArgPair(5, 1000)
    ->ArgPair(5, 10000);

// Trigger Copybara export.
}  // namespace tensorflow
