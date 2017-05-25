/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_REMOTE_FUSED_GRAPH_EXECUTE_OP_TEST_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_REMOTE_FUSED_GRAPH_EXECUTE_OP_TEST_UTILS_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// RemoteFusedGraphExecuteOpTestUtils is a set of utilities in tests for
// RemoteFusedGraphExecuteOp.
class RemoteFusedGraphExecuteOpTestUtils {
 public:
  static Output BuildAddOp(const Scope& scope, const Input& x, const Input& y);
  static Status BuildAddGraph(const string& name0, const float val0,
                              const string& name1, const float val1,
                              const string& name_out, GraphDef* graph_def);

  // BuildMultipleAddGraph builds the following graph
  //
  //  A         B         C         D         E
  //  |         |         |         |         |
  //  +----+----+         |         +----+----+
  //       |              |              |
  //       F             / \             G
  //       |            |   |           / \
  //       +-----+------+   +-----+----+   +
  //             |                |        |
  //             H                I        |
  //             |                |        |
  //             +-------+--------+        |
  //                     |                 |
  //                     J                 |
  //                     |                 |
  //                     +--------+--------+
  //                              |
  //                              K
  //
  static Status BuildMultipleAddGraph(GraphDef* graph_def);

 private:
  RemoteFusedGraphExecuteOpTestUtils() = delete;
  TF_DISALLOW_COPY_AND_ASSIGN(RemoteFusedGraphExecuteOpTestUtils);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_REMOTE_FUSED_GRAPH_EXECUTE_OP_TEST_UTILS_H_
