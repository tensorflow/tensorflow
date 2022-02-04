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

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_compilation_cache_persistence.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

using EntryMap = std::map<std::string, XlaSerializedCacheEntry>;

class XlaCompilationCacheMockSaver : public XlaCompilationCacheSaver {
 public:
  explicit XlaCompilationCacheMockSaver(EntryMap& map) : saved_entries_(map) {}

  Status Save(const XlaSerializedCacheEntry& entry) override {
    std::cerr << "Saving " << entry.key().DebugString() << "\n";
    std::string serialized_key;
    if (!SerializeToStringDeterministic(entry.key(), &serialized_key)) {
      return errors::Unknown("Serialization failed.");
    }

    saved_entries_[serialized_key] = entry;
    return Status::OK();
  }

 private:
  EntryMap& saved_entries_;
};

class XlaCompilationCacheMockLoader : public XlaCompilationCacheLoader {
 public:
  explicit XlaCompilationCacheMockLoader(EntryMap& map) : saved_entries_(map) {}

  StatusOr<absl::optional<XlaSerializedCacheEntry>> TryLoad(
      const XlaSerializedCacheKey& key) override {
    std::cerr << "Loading " << key.DebugString() << "\n";
    std::string serialized_key;
    if (!SerializeToStringDeterministic(key, &serialized_key)) {
      return errors::Unknown("Serialization failed.");
    }

    auto it = saved_entries_.find(serialized_key);
    if (it == saved_entries_.end()) {
      return errors::NotFound("entry not found!");
    }
    return {it->second};
  }

 private:
  EntryMap& saved_entries_;
};

// Creates a float tensor of linearly increasing values, starting from offset.
Tensor CreateInputTensor(const TensorShape& shape, float offset) {
  Tensor tensor(DT_FLOAT, shape);
  for (int64 i = 0; i < tensor.flat<float>().size(); ++i) {
    tensor.flat<float>()(i) = offset + i;
  }
  return tensor;
}

// Creates a graph for testing compilation cache serialization.
std::unique_ptr<Graph> BuildTestGraph(const PartialTensorShape& input_shape) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* a = ops::SourceOp("Placeholder", builder.opts()
                                             .WithAttr("dtype", DT_FLOAT)
                                             .WithAttr("shape", input_shape)
                                             .WithName("A"));
  Node* b = ops::SourceOp("Placeholder", builder.opts()
                                             .WithAttr("dtype", DT_FLOAT)
                                             .WithAttr("shape", input_shape)
                                             .WithName("B"));
  Node* c = ops::SourceOp("Placeholder", builder.opts()
                                             .WithAttr("dtype", DT_FLOAT)
                                             .WithAttr("shape", input_shape)
                                             .WithName("C"));

  Node* d = ops::BinaryOp("Add", a, b, builder.opts().WithName("D"));
  Node* e = ops::BinaryOp("Mul", d, c, builder.opts().WithName("E"));
  Node* f = ops::BinaryOp("Add", e, a, builder.opts().WithName("F"));
  Node* g = ops::BinaryOp("Mul", f, b, builder.opts().WithName("G"));

  // Force two clusters by excluding this node explicitly.
  Node* h = ops::BinaryOp(
      "Add", g, f, builder.opts().WithName("H").WithAttr("_XlaCompile", false));

  Node* i = ops::BinaryOp("Add", h, e, builder.opts().WithName("I"));
  Node* j = ops::BinaryOp("Add", i, h, builder.opts().WithName("J"));
  Node* k = ops::BinaryOp("Add", j, h, builder.opts().WithName("K"));
  Node* m = ops::BinaryOp("Add", k, h, builder.opts().WithName("M"));

  ops::UnaryOp("Identity", m, builder.opts().WithName("N"));
  CHECK(GraphDefBuilderToGraph(builder, graph.get()).ok());
  return graph;
}

TEST(XlaCompilationCacheSerializeTest, SerializeCacheTest) {
  EntryMap saved_entries;
  TF_ASSERT_OK(RegisterXlaCompilationCacheSaver([&] {
    return std::make_unique<XlaCompilationCacheMockSaver>(saved_entries);
  }));

  GraphDef graph;
  BuildTestGraph({-1, 4})->ToGraphDef(&graph);

  auto exec_with_batch = [&graph](int batch) {
    const TensorShape shape({batch, 4});
    grappler::GrapplerItem item;
    item.graph = graph;

    Tensor f32_input(DT_FLOAT, shape);
    item.feed.emplace_back("A", f32_input);
    item.feed.emplace_back("B", f32_input);
    item.feed.emplace_back("C", f32_input);
    item.fetch.emplace_back("K");

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
          {std::make_pair("A", input_a), std::make_pair("B", input_b),
           std::make_pair("C", input_c)},
          {"N"}, {}, &golden_output_tensors, nullptr));
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
          {std::make_pair("A", input_a), std::make_pair("B", input_b),
           std::make_pair("C", input_c)},
          {"N"}, {}, &output_tensors, nullptr));
      TF_ASSERT_OK(session->Close());
    }

    for (int64 i = 0; i < f32_input.NumElements(); ++i) {
      EXPECT_NEAR(golden_output_tensors[0].flat<float>()(i),
                  output_tensors[0].flat<float>()(i), 1e-3);
    }
  };

  // Warmup the cache(s) with multiple runs
  for (int b = 1; b < 4; ++b) {
    exec_with_batch(b);
  }

  // Reset the cluster numbering between sessions so we can get the same
  // cluster numbering.
  testing::ResetClusterSequenceNumber();
  UnregisterXlaCompilationCacheSaver();

  // Register the loader here because it expects entries to exist.

  TF_ASSERT_OK(RegisterXlaCompilationCacheLoader([&] {
    return std::make_unique<XlaCompilationCacheMockLoader>(saved_entries);
  }));

  // Run again but these should all be cache hits
  for (int b = 1; b < 4; ++b) {
    exec_with_batch(b);
  }

  UnregisterXlaCompilationCacheLoader();
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_deterministic_cluster_names = true;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
