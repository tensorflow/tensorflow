/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/graph_runner.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

using test::internal::ExpectEqual;

TEST(GraphRunnerTest, SingleConst) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0f);
  std::vector<Tensor> outputs;
  Status s = GraphRunner::Run(root.graph(), nullptr, Env::Default(), {},
                              {c.name()}, &outputs);
  TF_ASSERT_OK(s);
  ExpectEqual(42.0f, outputs[0].scalar<float>()());
}

TEST(GraphRunnerTest, MultiFetchConst) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0f);
  auto pi = ops::Const(root, 3.14f);
  std::vector<Tensor> outputs;
  Status s = GraphRunner::Run(root.graph(), nullptr, Env::Default(), {},
                              {c.name(), pi.name()}, &outputs);
  TF_ASSERT_OK(s);
  ExpectEqual(42.0f, outputs[0].scalar<float>()());
  ExpectEqual(3.14f, outputs[1].scalar<float>()());
}

TEST(GraphRunnerTest, FeedAndFetch) {
  Scope root = Scope::NewRootScope();
  auto p1 = ops::Placeholder(root.WithOpName("p1"), DT_FLOAT);
  auto p2 = ops::Placeholder(root.WithOpName("p2"), DT_FLOAT);
  auto add = ops::Add(root.WithOpName("add"), p1, p2);

  Tensor p1_data(DT_FLOAT, TensorShape({}));
  Tensor p2_data(DT_FLOAT, TensorShape({}));
  p1_data.scalar<float>()() = 1.0f;
  p2_data.scalar<float>()() = 2.0f;
  std::vector<std::pair<string, Tensor>> inputs = {{"p1:0", p1_data},
                                                   {"p2:0", p2_data}};

  std::vector<Tensor> outputs;
  Status s = GraphRunner::Run(root.graph(), nullptr, Env::Default(), inputs,
                              {"add:0"}, &outputs);
  TF_ASSERT_OK(s);
  ExpectEqual(3.0f, outputs[0].scalar<float>()());
}

}  // namespace
}  // namespace tensorflow
