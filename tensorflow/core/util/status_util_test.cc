/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/status_util.h"

#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TestStatusUtil, ErrorFormatTagForNode) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("Foo", "NoOp").Finalize(&graph, &node));
  EXPECT_EQ(error_format_tag(*node, "${line}"), "^^node:Foo:${line}^^");
  EXPECT_EQ(error_format_tag(*node, "${file}:${line}"),
            "^^node:Foo:${file}:${line}^^");
}

}  // namespace
}  // namespace tensorflow
