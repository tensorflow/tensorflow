/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#undef INTEL_MKL

#ifdef INTEL_MKL

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "mkldnn.hpp"
#include "tensorflow/core/util/mkl_util.h"

// Compare performance of default Tensorflow convolution kernels (Eigen) with
// MKL kernels on CPU.

// Before running these benchmarks configure OpenMP environment variables:
//   export KMP_BLOCKTIME=0
//   export OMP_NUM_THREADS=${num_threads}

namespace tensorflow {
static Tensor NonMklTensor() {
  MklDnnShape non_mkl_shape;
  non_mkl_shape.SetMklTensor(false);

  auto size = static_cast<int64>(non_mkl_shape.GetSerializeBufferSize());
  Tensor tensor(DT_UINT8, {size});

  non_mkl_shape.SerializeMklDnnShape(tensor.flat<uint8>().data(),
                                     size * sizeof(uint8));
  return tensor;
}

static Tensor GetRandomTensor(const TensorShape& shape) {
  Tensor tensor(DT_FLOAT, TensorShape(shape));
  tensor.flat<float>() = tensor.flat<float>().setRandom();
  return tensor;
}

#define CREATE_DEFAULT_FWD_OP(NODE_NAME, OP_NAME)                 \
  static Graph* NODE_NAME(const TensorShape& shape) {             \
    auto* graph = new Graph(OpRegistry::Global());                \
    Tensor input_t = GetRandomTensor(shape);                      \
    Node* input = test::graph::Constant(graph, input_t, "input"); \
    Node* op;                                                     \
    TF_CHECK_OK(NodeBuilder(graph->NewName(#NODE_NAME), #OP_NAME) \
                    .Input(input)                                 \
                    .Attr("T", DT_FLOAT)                          \
                    .Finalize(graph, &op));                       \
    return graph;                                                 \
  }
CREATE_DEFAULT_FWD_OP(Default_Tanh, Tanh)
CREATE_DEFAULT_FWD_OP(Default_Elu, Elu)
CREATE_DEFAULT_FWD_OP(Default_Relu, Relu)
CREATE_DEFAULT_FWD_OP(Default_Relu6, Relu6)
CREATE_DEFAULT_FWD_OP(Default_LeakyRelu, LeakyRelu)

#define CREATE_DEFAULT_BWD_OP(NODE_NAME, OP_NAME)                 \
  static Graph* NODE_NAME(const TensorShape& shape) {             \
    auto* graph = new Graph(OpRegistry::Global());                \
    Tensor input_t = GetRandomTensor(shape);                      \
    Node* input = test::graph::Constant(graph, input_t, "input"); \
    Tensor grad_t = GetRandomTensor(shape);                       \
    Node* grad = test::graph::Constant(graph, grad_t, "grad");    \
    Node* op;                                                     \
    TF_CHECK_OK(NodeBuilder(graph->NewName(#NODE_NAME), #OP_NAME) \
                    .Input(grad)                                  \
                    .Input(input)                                 \
                    .Attr("T", DT_FLOAT)                          \
                    .Finalize(graph, &op));                       \
    return graph;                                                 \
  }
CREATE_DEFAULT_BWD_OP(Default_TanhGrad, TanhGrad)
CREATE_DEFAULT_BWD_OP(Default_EluGrad, EluGrad)
CREATE_DEFAULT_BWD_OP(Default_ReluGrad, ReluGrad)
CREATE_DEFAULT_BWD_OP(Default_Relu6Grad, Relu6Grad)
CREATE_DEFAULT_BWD_OP(Default_LeakyReluGrad, LeakyReluGrad)

#define CREATE_MKL_FWD_OP(NODE_NAME, OP_NAME)                     \
  static Graph* NODE_NAME(const TensorShape& shape) {             \
    auto* graph = new Graph(OpRegistry::Global());                \
                                                                  \
    Tensor input_t = GetRandomTensor(shape);                      \
    Node* input = test::graph::Constant(graph, input_t, "input"); \
                                                                  \
    Node* not_mkl_shape =                                         \
        test::graph::Constant(graph, NonMklTensor(), "not_mkl");  \
                                                                  \
    Node* op;                                                     \
    TF_CHECK_OK(NodeBuilder(graph->NewName(#NODE_NAME), #OP_NAME) \
                    .Input(input)                                 \
                    .Input(not_mkl_shape)                         \
                    .Attr("T", DT_FLOAT)                          \
                    .Attr("_kernel", "MklLayoutDependentOp")      \
                    .Finalize(graph, &op));                       \
                                                                  \
    return graph;                                                 \
  }

CREATE_MKL_FWD_OP(Mkl_Tanh, _MklTanh)
CREATE_MKL_FWD_OP(Mkl_Elu, _MklElu)
CREATE_MKL_FWD_OP(Mkl_Relu, _MklRelu)
CREATE_MKL_FWD_OP(Mkl_Relu6, _MklRelu6)
CREATE_MKL_FWD_OP(Mkl_LeakyRelu, _MklLeakyRelu)

#define CREATE_MKL_BWD_OP(NODE_NAME, OP_NAME)                     \
  static Graph* NODE_NAME(const TensorShape& shape) {             \
    auto* graph = new Graph(OpRegistry::Global());                \
                                                                  \
    Tensor input_t = GetRandomTensor(shape);                      \
    Node* input = test::graph::Constant(graph, input_t, "input"); \
    Tensor grad_t = GetRandomTensor(shape);                       \
    Node* grad = test::graph::Constant(graph, grad_t, "grad");    \
                                                                  \
    Node* not_mkl_shape =                                         \
        test::graph::Constant(graph, NonMklTensor(), "not_mkl");  \
                                                                  \
    Node* op;                                                     \
    TF_CHECK_OK(NodeBuilder(graph->NewName(#NODE_NAME), #OP_NAME) \
                    .Input(grad)                                  \
                    .Input(input)                                 \
                    .Input(not_mkl_shape)                         \
                    .Input(not_mkl_shape)                         \
                    .Attr("T", DT_FLOAT)                          \
                    .Attr("_kernel", "MklLayoutDependentOp")      \
                    .Finalize(graph, &op));                       \
                                                                  \
    return graph;                                                 \
  }

CREATE_MKL_BWD_OP(Mkl_TanhGrad, _MklTanhGrad)
CREATE_MKL_BWD_OP(Mkl_EluGrad, _MklEluGrad)
CREATE_MKL_BWD_OP(Mkl_ReluGrad, _MklReluGrad)
CREATE_MKL_BWD_OP(Mkl_Relu6Grad, _MklRelu6Grad)
CREATE_MKL_BWD_OP(Mkl_LeakyReluGrad, _MklLeakyReluGrad)

#define BM_Activation(op, kind, A, B, C, D, type)                            \
  static void BM_##op##_##kind##_##type##_##A##_##B##_##C##_##D(int iters) { \
    int64 num_computed_elements = (A) * (B) * (C) * (D);                     \
    int64 flops_per_iter = num_computed_elements;                            \
    testing::ItemsProcessed(static_cast<int64>(iters) * flops_per_iter);     \
                                                                             \
    test::Benchmark(#type, kind##_##op({A, B, C, D})).Run(iters);            \
  }                                                                          \
  BENCHMARK(BM_##op##_##kind##_##type##_##A##_##B##_##C##_##D)

#define BM(op, A, B, C, D, type)                \
  BM_Activation(op, Default, A, B, C, D, type); \
  BM_Activation(op, Mkl, A, B, C, D, type);

#define TEST_ALL_SIZES(OP)       \
  BM(OP, 2, 4, 8, 16, cpu);      \
  BM(OP, 3, 5, 9, 17, cpu);      \
  BM(OP, 32, 64, 128, 256, cpu); \
  BM(OP, 33, 65, 129, 257, cpu);

TEST_ALL_SIZES(Tanh)
TEST_ALL_SIZES(TanhGrad)
TEST_ALL_SIZES(Relu)
TEST_ALL_SIZES(ReluGrad)
TEST_ALL_SIZES(Elu)
TEST_ALL_SIZES(EluGrad)
TEST_ALL_SIZES(Relu6)
TEST_ALL_SIZES(Relu6Grad)
TEST_ALL_SIZES(LeakyRelu)
TEST_ALL_SIZES(LeakyReluGrad)

}  // namespace tensorflow

#endif  // INTEL_MKL
