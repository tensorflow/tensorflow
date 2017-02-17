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

#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

TEST(ClientSessionTest, Basic) {
  Scope root = Scope::NewRootScope();
  auto c = Const(root, {{1, 1}});
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({1, 1}, {1, 2}));
}

TEST(ClientSessionTest, Feed) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32);
  auto b = Placeholder(root, DT_INT32);
  auto c = Add(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({{a, 1}, {b, 41}}, {c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({42}, {}));
}

TEST(ClientSessionTest, Extend) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32);
  auto c = Add(root, a, {2, 2});
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({{a, {1, 1}}}, {c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({3, 3}, {2}));

  auto d = Add(root, c, {39, 39});
  outputs.clear();
  TF_EXPECT_OK(session.Run({{a, {-10, 1}}}, {d}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({31, 42}, {2}));
}

TEST(ClientSessionTest, MultiThreaded) {
  Scope root = Scope::NewRootScope();
  auto a = Add(root, {1, 2}, {3, 4});
  auto b = Mul(root, {1, 2}, {3, 4});
  ClientSession session(root);
  {
    thread::ThreadPool thread_pool(Env::Default(), "pool", 2);
    thread_pool.Schedule([&session, a]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run({a}, &outputs));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({4, 6}, {2}));
    });
    thread_pool.Schedule([&session, b]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run({b}, &outputs));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({3, 8}, {2}));
    });
  }
  auto c = Sub(root, b, a);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({-1, 2}, {2}));
}

}  // end namespace tensorflow
