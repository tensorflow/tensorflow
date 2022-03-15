/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class JitCompilationListener : public XlaActivityListener {
 public:
  Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) override {
    return Status::OK();
  }

  Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) override {
    used_persistent_cache_.push_back(
        jit_compilation_activity.used_persistent_cache());
    return Status::OK();
  }

  Status Listen(const XlaOptimizationRemark& optimization_remark) override {
    return Status::OK();
  }

  ~JitCompilationListener() override {}

  Status VerifyListenerHistory(bool expect_persistent_cache_use) {
    for (bool used_persistent_cache : used_persistent_cache_) {
      if (used_persistent_cache != expect_persistent_cache_use) {
        return errors::FailedPrecondition("Unexpected listener history.");
      }
    }
    return Status::OK();
  }

  void ClearListenerHistory() { used_persistent_cache_.clear(); }

 private:
  std::vector<bool> used_persistent_cache_;
};

class XlaCompilationCacheSerializeTest : public ::testing::Test {
 protected:
  XlaCompilationCacheSerializeTest() {
    auto listener = absl::make_unique<JitCompilationListener>();
    listener_ = listener.get();
    RegisterXlaActivityListener(std::move(listener));
  }

  JitCompilationListener* listener() const { return listener_; }

 private:
  JitCompilationListener* listener_;
};

// Creates a float tensor of linearly increasing values, starting from offset.
Tensor CreateInputTensor(const TensorShape& shape, float offset) {
  Tensor tensor(DT_FLOAT, shape);
  for (int64 i = 0; i < tensor.flat<float>().size(); ++i) {
    tensor.flat<float>()(i) = offset + i;
  }
  return tensor;
}

NodeDef MakeNode(
    absl::string_view name, absl::string_view op,
    absl::Span<const std::string> inputs,
    absl::Span<
        const std::pair<std::string, FunctionDefHelper::AttrValueWrapper>>
        attrs) {
  NodeDef node;
  node.set_name(name);
  node.set_op(op);
  for (const auto& input : inputs) node.add_input(input);
  for (const auto& attr : attrs)
    node.mutable_attr()->insert({attr.first, attr.second.proto});
  return node;
}

// Creates a graph for testing compilation cache serialization.
GraphDef BuildTestGraph(const PartialTensorShape& input_shape) {
  FunctionDef make_test_fn = FunctionDefHelper::Define(
      "TestFn", {"a:float", "b:float", "c:float"}, {"m:float"}, {},
      {{{"d"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
       {{"e"}, "Mul", {"d", "c"}, {{"T", DT_FLOAT}}},
       {{"f"}, "Add", {"e", "a"}, {{"T", DT_FLOAT}}},
       {{"g"}, "Mul", {"f", "b"}, {{"T", DT_FLOAT}}},
       // Force two clusters by excluding this node explicitly.
       {{"h"}, "Add", {"g", "f"}, {{"T", DT_FLOAT}, {"_XlaCompile", false}}},
       {{"i"}, "Add", {"h", "e"}, {{"T", DT_FLOAT}}},
       {{"j"}, "Add", {"i", "h"}, {{"T", DT_FLOAT}}},
       {{"k"}, "Add", {"j", "h"}, {{"T", DT_FLOAT}}},
       {{"l"}, "Add", {"k", "h"}, {{"T", DT_FLOAT}}},
       {{"m"}, "Identity", {"l"}, {{"T", DT_FLOAT}}}});

  GraphDef graph;
  *graph.mutable_library()->add_function() = make_test_fn;
  *graph.add_node() = MakeNode("a", "Placeholder", {},
                               {{"dtype", DT_FLOAT}, {"shape", input_shape}});
  *graph.add_node() = MakeNode("b", "Placeholder", {},
                               {{"dtype", DT_FLOAT}, {"shape", input_shape}});
  *graph.add_node() = MakeNode("c", "Placeholder", {},
                               {{"dtype", DT_FLOAT}, {"shape", input_shape}});
  *graph.add_node() = MakeNode("m", "TestFn", {"a", "b", "c"}, {});
  return graph;
}

TEST_F(XlaCompilationCacheSerializeTest, PersistentCacheTest) {
  GraphDef graph = BuildTestGraph({-1, 4});

  auto exec_with_batch = [&graph](int batch) {
    const TensorShape shape({batch, 4});

    // Compute the golden output tensor
    std::vector<Tensor> golden_output_tensors;
    {
      SessionOptions options;
      std::unique_ptr<Session> session(NewSession(options));
      TF_ASSERT_OK(session->Create(graph));
      RunOptions run_options;

      Tensor input_a = CreateInputTensor(shape, 0);
      Tensor input_b = CreateInputTensor(shape, shape.num_elements());
      Tensor input_c = CreateInputTensor(shape, 2 * shape.num_elements());
      TF_ASSERT_OK(session->Run(
          run_options,
          {std::make_pair("a", input_a), std::make_pair("b", input_b),
           std::make_pair("c", input_c)},
          {"m"}, {}, &golden_output_tensors, nullptr));
      TF_ASSERT_OK(session->Close());
    }

    // Compute the XLA compiled output
    std::vector<Tensor> output_tensors;
    {
      SessionOptions options;
      auto& opts =
          *options.config.mutable_graph_options()->mutable_optimizer_options();
      opts.set_global_jit_level(OptimizerOptions::ON_1);
      opts.set_cpu_global_jit(true);

      std::unique_ptr<Session> session(NewSession(options));
      TF_ASSERT_OK(session->Create(graph));
      RunOptions run_options;
      Tensor input_a = CreateInputTensor(shape, 0);
      Tensor input_b = CreateInputTensor(shape, shape.num_elements());
      Tensor input_c = CreateInputTensor(shape, 2 * shape.num_elements());
      TF_ASSERT_OK(session->Run(
          run_options,
          {std::make_pair("a", input_a), std::make_pair("b", input_b),
           std::make_pair("c", input_c)},
          {"m"}, {}, &output_tensors, nullptr));
      TF_ASSERT_OK(session->Close());
    }

    Tensor f32_input(DT_FLOAT, shape);
    for (int64 i = 0; i < f32_input.NumElements(); ++i) {
      EXPECT_NEAR(golden_output_tensors[0].flat<float>()(i),
                  output_tensors[0].flat<float>()(i), 1e-3);
    }
  };

  // Warmup the persistent cache(s) with multiple runs. 4 is a magic number to
  // detect non-determinism in TF when running the test.
  listener()->ClearListenerHistory();
  for (int b = 1; b < 4; ++b) {
    exec_with_batch(b);
  }
  TF_ASSERT_OK(
      listener()->VerifyListenerHistory(/*expect_persistent_cache_use=*/false));

  // Reset the cluster numbering between sessions so we can get the same
  // cluster numbering.
  testing::ResetClusterSequenceNumber();

  // Run again but these should all hit in the persistent cache.
  listener()->ClearListenerHistory();
  for (int b = 1; b < 4; ++b) {
    exec_with_batch(b);
  }
  TF_ASSERT_OK(
      listener()->VerifyListenerHistory(/*expect_persistent_cache_use=*/true));
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_deterministic_cluster_names = true;
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = tensorflow::testing::TmpDir();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
