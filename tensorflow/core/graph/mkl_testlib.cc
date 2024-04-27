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

Node* oneDNNSoftmax(Graph* g, Node* input) {
  Node* ret = nullptr;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_MklSoftmax")
                  .Input(input)
                  .Attr("_kernel", mkl_op_registry::kMklNameChangeOpLabel)
                  .Finalize(g, &ret));
  return ret;
}

#ifdef ENABLE_ONEDNN_V3
Node* oneDNNSparseCSRMatmul(Graph* g, Node* csr_matrix_t, Node* b) {
  Node* ret = nullptr;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_MklNativeSparseMatrixMatMul")
                  .Input(csr_matrix_t)
                  .Input(b)
                  .Attr("T", DT_FLOAT)
                  .Attr("_kernel", mkl_op_registry::kMklNameChangeOpLabel)
                  .Finalize(g, &ret));
  return ret;
}
#endif  // ENABLE_ONEDNN_V3

}  // namespace graph
}  // namespace test
}  // namespace tensorflow

#endif  // INTEL_MKL
