/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL
#include "tensorflow/core/graph/mkl_testlib.h"

#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace test {
namespace graph {

Node* oneDNNMatmul(Graph* g, Node* in0, Node* in1, bool transpose_a,
                   bool transpose_b) {
  Node* ret = nullptr;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_MklMatMul")
                  .Input(in0)
                  .Input(in1)
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Attr("_kernel", mkl_op_registry::kMklNameChangeOpLabel)
                  .Finalize(g, &ret));
  return ret;
}

Node* oneDNNSoftmax(Graph* g, Node* input) {
  Node* ret = nullptr;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_MklSoftmax")
                  .Input(input)
                  .Attr("_kernel", mkl_op_registry::kMklNameChangeOpLabel)
                  .Finalize(g, &ret));
  return ret;
}

}  // namespace graph
}  // namespace test
}  // namespace tensorflow

#endif  // INTEL_MKL
