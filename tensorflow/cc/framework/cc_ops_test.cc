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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/test_op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

Output Linear(const Scope& scope, Input x, Input w, Input b) {
  auto cop_scopes = scope.GetCompositeOpScopes("linear");
  auto m = MatMul(cop_scopes.child, x, w);
  return BiasAdd(cop_scopes.last, m, b);
}

void GetTensors(const Scope& scope, OutputList tensors,
                std::vector<Tensor>* out) {
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  GraphDef def;
  scope.graph()->ToGraphDef(&def);

  graph::SetDefaultDevice("/cpu:0", &def);

  TF_CHECK_OK(session->Create(def));
  std::vector<string> names;
  for (const auto& t : tensors) {
    names.push_back(strings::StrCat(t.node()->name(), ":", t.index()));
  }
  TF_CHECK_OK(session->Run({}, names, {}, out));
  TF_CHECK_OK(session->Close());
}

void GetTensor(const Scope& scope, Output tensor, Tensor* out) {
  std::vector<Tensor> outputs;
  GetTensors(scope, {tensor}, &outputs);
  *out = outputs[0];
}

void GetColocationConstraints(Output tensor, std::vector<string>* constraints) {
  constraints->clear();
  const auto& attrs = tensor.op().node()->def().attr();
  ASSERT_TRUE(attrs.find("_class") != attrs.end());
  auto loc = attrs.find("_class")->second;
  TF_EXPECT_OK(AttrValueHasType(loc, "list(string)"));
  if (loc.value_case() == AttrValue::kList && loc.list().s_size() > 0) {
    for (int i = 0; i < loc.list().s_size(); ++i) {
      if (loc.list().s(i).find("loc:@") == 0) {
        constraints->push_back(loc.list().s(i));
      }
    }
  }
}

}  // namespace

TEST(CCOpTest, Basic) {
  Scope root = Scope::NewRootScope();
  auto c = Const(root, {{1, 1}});
  // NOTE: The recommended style for constructing ops is
  // auto v = OpConstructor(t0, t1, ..);
  // Since the wrappers are implemented as one class per op, the following
  // style is also possible :
  // PrimitiveOp p(t0, t1, ...);
  // It's being used here ONLY to ensure that, that style is tested.
  MatMul m(root, c, {{41}, {1}});
  TF_EXPECT_OK(root.status());
  Tensor out;
  GetTensor(root, m, &out);
  test::ExpectTensorEqual<int>(out, test::AsTensor<int>({42}, {1, 1}));
}

TEST(CCOpTest, Attrs) {
  Scope root = Scope::NewRootScope();
  auto m = MatMul(root, {{1}, {1}}, {{41}, {1}}, MatMul::TransposeA(true));
  TF_EXPECT_OK(root.status());
  Tensor out;
  GetTensor(root, m, &out);
  test::ExpectTensorEqual<int>(out, test::AsTensor<int>({42}, {1, 1}));
}

TEST(CCOpTest, SplitConcat) {
  Scope root = Scope::NewRootScope();
  Split p(root, 0, {{1}, {2}}, 2);
  auto c = Concat(root, 0, {p[0], p[1]});
  TF_EXPECT_OK(root.status());
  Tensor out;
  GetTensor(root, c, &out);
  test::ExpectTensorEqual<int>(out, test::AsTensor<int>({1, 2}, {2, 1}));
}

TEST(CCOpTest, CompositeOp) {
  Scope root = Scope::NewRootScope();
  auto l = Linear(root.WithOpName("layer0"), {{10.0f, -3.0f}},
                  {{.8f, .5f}, {.1f, .6f}}, {-8.0f, 31.0f});
  TF_EXPECT_OK(root.status());
  EXPECT_EQ(l.node()->name(), "layer0");
  Tensor out;
  GetTensor(root, l, &out);
  test::ExpectClose(out, test::AsTensor<float>({-0.3, 34.2}, {1, 2}));
}

TEST(CCOpTest, MultiOutput) {
  Scope root = Scope::NewRootScope();
  auto u = Unique(root, {1, 2, 2, 4, 3, 2});
  std::vector<Tensor> outputs;
  GetTensors(root, {u.y, u.idx}, &outputs);
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({1, 2, 4, 3}));
  test::ExpectTensorEqual<int>(outputs[1],
                               test::AsTensor<int>({0, 1, 1, 2, 3, 1}));
}

TEST(CCOpTest, ExampleTrainer) {
  Scope root = Scope::NewRootScope();
  // a = [3 2; -1 0]
  auto a = Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // x = [1.0; 1.0]
  auto x = Const(root.WithOpName("x"), {{1.f}, {1.f}});
  // y = a * x
  auto y = MatMul(root.WithOpName("y"), a, x);
  // y2 = y.^2
  auto y2 = Square(root, y);
  // y2_sum = sum(y2)
  auto y2_sum = Sum(root, y2, 0);
  // y_norm = sqrt(y2_sum)
  auto y_norm = Sqrt(root, y2_sum);
  // y_normalized = y ./ y_norm
  auto y_normalized = Div(root.WithOpName("y_normalized"), y, y_norm);
  Tensor out;
  GetTensor(root, y_normalized, &out);
  test::ExpectTensorNear<float>(
      out, test::AsTensor<float>({0.98058069, -0.19611613}, {2, 1}), 1e-5);
}

TEST(CCOpTest, ThrowAwayOp) {
  Scope root = Scope::NewRootScope();
  ThrowAway1(root, 1, 2.3f, 1, 1, 1, ThrowAway1::Builder(42));
  ThrowAway2(root, ThrowAway2::ThrowAway2_(3).Scope(1));
  TF_EXPECT_OK(root.status());
}

TEST(CCOpTest, ControlDeps) {
  Scope root = Scope::NewRootScope();
  auto v = Variable(root, {}, DT_FLOAT);
  auto assign = Assign(root, v, 41.0f);
  Scope with_control_deps = root.WithControlDependencies(assign);
  auto add = Add(with_control_deps, v, 1.0f);
  Scope no_control_deps = with_control_deps.WithNoControlDependencies();
  auto sub = Sub(no_control_deps, 3.0f, 2.0f);
  auto is_inited =
      IsVariableInitialized(no_control_deps.WithControlDependencies(sub), v);

  TF_EXPECT_OK(root.status());

  std::vector<Tensor> out;

  GetTensors(root, {add}, &out);
  test::ExpectTensorNear<float>(out[0], test::AsTensor<float>({42.0f}, {}),
                                1e-5);

  out.clear();
  // Note : GetTensors creates a new session, so 'v' is uninitialized.
  // sub should have no control deps, so it should not cause the assign to run.
  // Hence is_inited should be false.
  GetTensors(root, {sub, is_inited}, &out);
  test::ExpectTensorNear<float>(out[0], test::AsTensor<float>({1.0f}, {}),
                                1e-5);
  test::ExpectTensorEqual<bool>(out[1], test::AsTensor<bool>({false}, {}));
}

TEST(CCOpTest, KernelLabel) {
  Scope root = Scope::NewRootScope();
  auto add = Add(root.WithKernelLabel("AddWithKernelLabel"), 1.0f, 2.0f);
  TF_EXPECT_OK(root.status());
  const auto& attrs = add.z.op().node()->def().attr();
  ASSERT_TRUE(attrs.find("_kernel") != attrs.end());
  auto kernel_attr = attrs.find("_kernel")->second;
  TF_EXPECT_OK(AttrValueHasType(kernel_attr, "string"));
  EXPECT_EQ(kernel_attr.s(), "AddWithKernelLabel");
}

TEST(CCOpTest, ColocateWith) {
  Scope root = Scope::NewRootScope();
  auto c1 = Const(root.WithOpName("c1"), 1);
  auto c2 = Const(root.WithOpName("c2").ColocateWith(c1), 2);
  std::vector<string> constraints;
  GetColocationConstraints(c2, &constraints);
  EXPECT_EQ(constraints[0], "loc:@c1");

  auto c3 = Const(root.WithOpName("c3").ColocateWith(c2), 3);
  GetColocationConstraints(c3, &constraints);
  EXPECT_EQ(constraints[0], "loc:@c1");

  auto a = Const(root.WithOpName("a"), 4);
  auto c4 = Const(root.WithOpName("c4").ColocateWith(a), 5);
  GetColocationConstraints(c4, &constraints);
  EXPECT_EQ(constraints[0], "loc:@a");

  auto c5 = Const(root.WithOpName("c5").ColocateWith(c3).ColocateWith(c4), 6);
  GetColocationConstraints(c5, &constraints);
  EXPECT_EQ(constraints[0], "loc:@a");
  EXPECT_EQ(constraints[1], "loc:@c1");

  Scope with_colocate = root.ColocateWith(c3).ColocateWith(c4);
  auto c6 = Const(with_colocate.WithOpName("c6").ClearColocation(), 7);
  const auto& attrs = c6.op().node()->def().attr();
  EXPECT_TRUE(attrs.find("_class") == attrs.end());
}

TEST(CCOpTest, TemplatedConst) {
  Scope root = Scope::NewRootScope();
  auto c1 = ops::Const<float>(root, {{3, 2}, {-1, 0}});
  TF_EXPECT_OK(root.status());

  Tensor out;
  GetTensor(root, c1, &out);
  test::ExpectTensorEqual<float>(
      out, test::AsTensor<float>({3.f, 2.f, -1.f, 0.f}, {2, 2}));

  auto c2 = ops::Const<string>(root, {{"this"}, {"is"}, {"a"}, {"constant"}});
  GetTensor(root, c2, &out);
  test::ExpectTensorEqual<string>(
      out, test::AsTensor<string>({"this", "is", "a", "constant"}, {4, 1}));
}

TEST(CCOpTest, EmptyConst) {
  Scope root = Scope::NewRootScope();

  auto c1 = ops::Const(root, {});
  TF_CHECK_OK(root.status());

  Tensor out;
  GetTensor(root, c1, &out);
  test::ExpectTensorEqual<float>(out, Tensor(DT_FLOAT, {0}));

  auto c2 = ops::Const(root, {{}});
  TF_CHECK_OK(root.status());
  GetTensor(root, c2, &out);
  test::ExpectTensorEqual<float>(out, Tensor(DT_FLOAT, {1, 0}));

  auto c3 = ops::Const(root, {{{}, {}}});
  TF_CHECK_OK(root.status());
  GetTensor(root, c3, &out);
  test::ExpectTensorEqual<float>(out, Tensor(DT_FLOAT, {1, 2, 0}));

  auto c4 = ops::Const<int>(root, {{{}}});
  TF_CHECK_OK(root.status());
  GetTensor(root, c4, &out);
  test::ExpectTensorEqual<int>(out, Tensor(DT_INT32, {1, 1, 0}));

  ops::Const(root, {{}, {{}}});
  EXPECT_FALSE(root.status().ok());
}

}  // namespace tensorflow
