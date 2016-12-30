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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"

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

class ConstantFoldingTest : public ::testing::Test {
 protected:
  ConstantFoldingTest() { Reset(); }
  void Reset() { g_.reset(new Graph(OpRegistry::Global())); }

  template <typename T>
  Node* Constant(gtl::ArraySlice<T> values, TensorShape shape) {
    return test::graph::Constant(g_.get(), test::AsTensor(values, shape));
  }

  template <typename T>
  Node* Constant(T v) {
    return test::graph::Constant(g_.get(), test::AsScalar(v));
  }

  template <typename T>
  void ExpectNodeClose(const Node* n, gtl::ArraySlice<T> values,
                       TensorShape shape) {
    EXPECT_TRUE(n->IsConstant());
    const TensorProto* tensor_proto;
    TF_EXPECT_OK(GetNodeAttr(n->def(), "value", &tensor_proto));
    DataType dtype;
    TF_EXPECT_OK(GetNodeAttr(n->def(), "dtype", &dtype));
    Tensor t(dtype);
    EXPECT_TRUE(t.FromProto(*tensor_proto));
    test::ExpectClose(t, test::AsTensor(values, shape));
  }

  template <typename T>
  void ExpectNodeEqual(const Node* n, gtl::ArraySlice<T> values,
                       TensorShape shape) {
    EXPECT_TRUE(n->IsConstant());
    const TensorProto* tensor_proto;
    TF_EXPECT_OK(GetNodeAttr(n->def(), "value", &tensor_proto));
    DataType dtype;
    TF_EXPECT_OK(GetNodeAttr(n->def(), "dtype", &dtype));
    Tensor t(dtype);
    EXPECT_TRUE(t.FromProto(*tensor_proto));
    test::ExpectTensorEqual<T>(t, test::AsTensor(values, shape));
  }

// Construct the following graph
/*
      s1  s2
      |    |
      m1   m2
      / \ / \
     a   b   c
*/
#define SIMPLE_GRAPH                                                  \
  Reset();                                                            \
  Graph* g = g_.get();                                                \
  Node* a = Constant<float>({1.0, 0.0, 0.0, 1.0}, {2, 2});            \
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});            \
  Node* c = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});            \
  g->AddControlEdge(g->source_node(), a);                             \
  g->AddControlEdge(g->source_node(), b);                             \
  g->AddControlEdge(g->source_node(), c);                             \
  Node* m1 = test::graph::Matmul(g, a, b, false, false);              \
  Node* s1 = test::graph::Send(g, m1, "m1", "sender", 0, "receiver"); \
  Node* m2 = test::graph::Matmul(g, b, c, false, false);              \
  Node* s2 = test::graph::Send(g, m2, "m2", "sender", 0, "receiver"); \
  g->AddControlEdge(s1, g->sink_node());                              \
  g->AddControlEdge(s2, g->sink_node());

  std::unique_ptr<Graph> g_;
};

TEST_F(ConstantFoldingTest, Basic) {
  SIMPLE_GRAPH;
  EXPECT_TRUE(DoConstantFolding(ConstantFoldingOptions{}, nullptr,
                                Env::Default(), nullptr, g));

  // Nodes s1 and s2 now should now have a constant input
  EXPECT_EQ(1, s1->num_inputs());
  ExpectNodeClose<float>(*(s1->in_nodes().begin()), {1.0, 2.0, 3.0, 4.0},
                         {2, 2});
  EXPECT_EQ(1, s2->num_inputs());
  ExpectNodeClose<float>(*(s2->in_nodes().begin()), {2.0, 1.0, 4.0, 3.0},
                         {2, 2});
}

TEST_F(ConstantFoldingTest, ConsiderFunction) {
  SIMPLE_GRAPH;
  ConstantFoldingOptions opts;
  // Do not allow constant folding of m2
  opts.consider = [m2](const Node* n) { return m2 != n; };
  EXPECT_TRUE(DoConstantFolding(opts, nullptr, Env::Default(), nullptr, g));

  // Node s1 now should now have a constant input
  EXPECT_EQ(1, s1->num_inputs());
  ExpectNodeClose<float>(*(s1->in_nodes().begin()), {1.0, 2.0, 3.0, 4.0},
                         {2, 2});
  // s2's input should still be m2
  EXPECT_EQ(1, s2->num_inputs());
  EXPECT_EQ(*(s2->in_nodes().begin()), m2);
}

TEST_F(ConstantFoldingTest, TestNoReplaceAnotherConstant) {
  SIMPLE_GRAPH;
  Node* d = Constant<float>({1.0, 0.0, 0.0, 1.0}, {2, 2});
  g->AddControlEdge(g->source_node(), d);
  Node* s3 = test::graph::Send(g, d, "d", "sender", 0, "receiver");
  g->AddControlEdge(s3, g->sink_node());
  EXPECT_TRUE(DoConstantFolding(ConstantFoldingOptions{}, nullptr,
                                Env::Default(), nullptr, g));

  // Nodes s3 should still have d as input
  EXPECT_EQ(1, s3->num_inputs());
  EXPECT_EQ(*(s3->in_nodes().begin()), d);
}

#undef SIMPLE_GRAPH

TEST_F(ConstantFoldingTest, TwoOutputs) {
  Reset();
  Graph* g = g_.get();
  Node* s0 = Constant<int>({1}, {1});
  Node* s1 = Constant<int>({2, 2}, {2});
  g->AddControlEdge(g->source_node(), s0);
  g->AddControlEdge(g->source_node(), s1);
  Node* b = test::graph::BroadcastGradientArgs(g, s0, s1);
  Node* b0 = test::graph::Send(g, test::graph::Identity(g, b, 0),
                               strings::StrCat(b->name(), "0"), "sender", 0,
                               "receiver");
  Node* b1 = test::graph::Send(g, test::graph::Identity(g, b, 1),
                               strings::StrCat(b->name(), "1"), "sender", 0,
                               "receiver");
  g->AddControlEdge(b0, g->sink_node());
  g->AddControlEdge(b1, g->sink_node());

  EXPECT_TRUE(DoConstantFolding(ConstantFoldingOptions{}, nullptr,
                                Env::Default(), nullptr, g));
  EXPECT_EQ(1, b0->num_inputs());
  ExpectNodeEqual<int>(*(b0->in_nodes().begin()), {0, 1}, {2});
  EXPECT_EQ(1, b1->num_inputs());
  ExpectNodeEqual<int>(*(b1->in_nodes().begin()), {}, {0});
}

TEST_F(ConstantFoldingTest, TwoOutputsFoldOneOutput) {
  Reset();
  Graph* g = g_.get();
  Node* s0 = Constant<int>({1}, {1});
  Node* s1 = Constant<int>({2, 2}, {2});
  g->AddControlEdge(g->source_node(), s0);
  g->AddControlEdge(g->source_node(), s1);
  Node* b = test::graph::BroadcastGradientArgs(g, s0, s1);
  Node* b0 = test::graph::Send(g, test::graph::Identity(g, b, 0),
                               strings::StrCat(b->name(), "0"), "sender", 0,
                               "receiver");
  Node* b1_ident = test::graph::Identity(g, b, 1);
  Node* b1 = test::graph::Send(g, b1_ident, strings::StrCat(b->name(), "1"),
                               "sender", 0, "receiver");
  g->AddControlEdge(b0, g->sink_node());
  g->AddControlEdge(b1, g->sink_node());

  ConstantFoldingOptions opts;
  opts.consider = [b1_ident](const Node* n) { return b1_ident != n; };
  EXPECT_TRUE(DoConstantFolding(opts, nullptr, Env::Default(), nullptr, g));
  // 0th output of b should have been folded.
  EXPECT_EQ(1, b0->num_inputs());
  ExpectNodeEqual<int>(*(b0->in_nodes().begin()), {0, 1}, {2});
  // 1st output of b should still be b1_ident. However, b1_ident's input must
  // have been replaced with a constant.
  EXPECT_EQ(1, b1->num_inputs());
  EXPECT_EQ(*(b1->in_nodes().begin()), b1_ident);

  EXPECT_EQ(1, b1_ident->num_inputs());
  ExpectNodeEqual<int>(*(b1_ident->in_nodes().begin()), {}, {0});
}

TEST_F(ConstantFoldingTest, TestNoReplaceLargeConstant) {
  Reset();
  Graph* g = g_.get();
  Node* s0 =
      Constant<int>(std::vector<int>(5 * 1024 * 256, 0), {5 * 1024 * 256});
  Node* s1 = Constant<int>(std::vector<int>(5 * 1024 * 256 + 1, 0),
                           {5 * 1024 * 256 + 1});
  Node* concat_dim = Constant<int>(0);
  g->AddControlEdge(g->source_node(), s0);
  g->AddControlEdge(g->source_node(), s1);
  // Concat s0 and s1. The resulting tensor would be of size 10M + 1 bytes
  Node* concat = test::graph::Concat(g, concat_dim, {s0, s1});
  Node* concat_send =
      test::graph::Send(g, concat, "concat_send", "sender", 0, "receiver");
  g->AddControlEdge(concat_send, g->sink_node());

  // The above concat should not have been constant folded.
  bool was_mutated;
  Status status =
      DoConstantFoldingWithStatus(ConstantFoldingOptions{}, nullptr,
                                  Env::Default(), nullptr, g, &was_mutated);
  EXPECT_FALSE(was_mutated);
  TF_EXPECT_OK(status);
}

TEST_F(ConstantFoldingTest, TestNoReplaceFunctionCall) {
  FunctionDefLibrary fdef_lib;
  *fdef_lib.add_function() = test::function::XTimesTwo();

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  g_.reset(new Graph(&flib_def));

  Graph* g = g_.get();
  Node* s =
      Constant<int>(std::vector<int>(5 * 1024 * 256, 0), {5 * 1024 * 256});
  g->AddControlEdge(g->source_node(), s);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("times_two", "XTimesTwo", g->op_registry())
                   .Input(s->name(), 0, DT_INT32)
                   .Finalize(&def));
  Status status;
  Node* times_two = g->AddNode(def, &status);
  TF_ASSERT_OK(status);

  Node* times_two_send = test::graph::Send(g, times_two, "times_two_send",
                                           "sender", 0, "receiver");
  g->AddControlEdge(times_two_send, g->sink_node());

  // The above function call should not have been constant folded.
  bool was_mutated;
  status =
      DoConstantFoldingWithStatus(ConstantFoldingOptions{}, nullptr,
                                  Env::Default(), nullptr, g, &was_mutated);
  EXPECT_FALSE(was_mutated);
  EXPECT_FALSE(status.ok());

  g_ = nullptr;
}

namespace {

const char kTestMemRegionName[] = "test://test";

class TestReadOnlyMemoryRegion : public ::tensorflow::ReadOnlyMemoryRegion {
 public:
  ~TestReadOnlyMemoryRegion() override = default;
  TestReadOnlyMemoryRegion(const void* data, uint64 length)
      : data_(data), length_(length) {}
  const void* data() override { return data_; }
  uint64 length() override { return length_; }

 protected:
  const void* data_;
  uint64 length_;
};

class TestTFFileSystem : public ::tensorflow::NullFileSystem {
 public:
  TestTFFileSystem()
      : ::tensorflow::NullFileSystem(),
        data_tensor_(test::AsTensor<double>({1., 2., 3., 4.}, {2, 2})) {}

  ::tensorflow::Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<::tensorflow::ReadOnlyMemoryRegion>* result) override {
    if (fname != kTestMemRegionName) {
      return ::tensorflow::errors::Unimplemented(
          "NewReadOnlyMemoryRegionFromFile unimplemented");
    }
    const ::tensorflow::StringPiece sp = data_tensor_.tensor_data();
    *result = std::unique_ptr<::tensorflow::ReadOnlyMemoryRegion>(
        new TestReadOnlyMemoryRegion(sp.data(), sp.size()));
    return ::tensorflow::Status::OK();
  }

 protected:
  ::tensorflow::Tensor data_tensor_;
};

// A test TF environent that checks that the environment was used.
class TestTFEnvironment : public ::tensorflow::EnvWrapper {
 public:
  using tf_base = ::tensorflow::EnvWrapper;
  TestTFEnvironment() : ::tensorflow::EnvWrapper(Default()) {}
  ::tensorflow::Status GetFileSystemForFile(
      const string& fname, ::tensorflow::FileSystem** result) override {
    was_used_ = true;
    if (fname == "test://test") {
      *result = &test_filesystem_;
      return ::tensorflow::Status::OK();
    }
    return tf_base::GetFileSystemForFile(fname, result);
  }
  bool was_used() const { return was_used_; }

 protected:
  TestTFFileSystem test_filesystem_;
  bool was_used_ = false;
};
}  // namespace

TEST_F(ConstantFoldingTest, TestImmutableConst) {
  Reset();
  Graph* g = g_.get();
  Scope root = Scope::NewRootScope();

  auto a = ops::ImmutableConst(root, DT_DOUBLE, {2, 2}, kTestMemRegionName);
  auto b = ops::Const<double>(root, {1.0, 2.0, 3.0, 4.0}, {2, 2});
  auto c = ops::RandomGamma(root, {2, 2}, 2.0);
  auto result1 = ops::MatMul(root, a, b);
  auto result2 = ops::MatMul(root, result1, c);
  TF_ASSERT_OK(root.ToGraph(g));
  TestTFEnvironment test_env;
  bool was_mutated;
  Status status =
      DoConstantFoldingWithStatus(ConstantFoldingOptions{}, nullptr,
                                  Env::Default(), nullptr, g, &was_mutated);
  EXPECT_FALSE(was_mutated);
  EXPECT_FALSE(status.ok());
  status = DoConstantFoldingWithStatus(ConstantFoldingOptions{}, nullptr,
                                       &test_env, nullptr, g, &was_mutated);
  EXPECT_TRUE(was_mutated);
  TF_EXPECT_OK(status);
}

}  // namespace
}  // namespace tensorflow
