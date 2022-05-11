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

#include "tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h"

#include <string>

#include "absl/strings/match.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

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
  node.set_name(std::string(name));
  node.set_op(std::string(op));
  for (const auto& input : inputs) node.add_input(input);
  for (const auto& attr : attrs)
    node.mutable_attr()->insert({attr.first, attr.second.proto});
  return node;
}

}  // namespace

GraphDef XlaCompilationCacheSerializeTest::GetTestGraph(
    const PartialTensorShape& input_shape) {
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

Status XlaCompilationCacheSerializeTest::ExecuteWithBatch(const GraphDef& graph,
                                                          int batch) {
  const TensorShape shape({batch, 4});

  // Compute the golden output tensor
  std::vector<Tensor> golden_output_tensors;
  {
    SessionOptions options;
    std::unique_ptr<Session> session(NewSession(options));
    TF_RETURN_IF_ERROR(session->Create(graph));
    RunOptions run_options;

    Tensor input_a = CreateInputTensor(shape, 0);
    Tensor input_b = CreateInputTensor(shape, shape.num_elements());
    Tensor input_c = CreateInputTensor(shape, 2 * shape.num_elements());
    TF_RETURN_IF_ERROR(session->Run(
        run_options,
        {std::make_pair("a", input_a), std::make_pair("b", input_b),
         std::make_pair("c", input_c)},
        {"m"}, {}, &golden_output_tensors, nullptr));
    TF_RETURN_IF_ERROR(session->Close());
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
    TF_RETURN_IF_ERROR(session->Create(graph));
    RunOptions run_options;
    Tensor input_a = CreateInputTensor(shape, 0);
    Tensor input_b = CreateInputTensor(shape, shape.num_elements());
    Tensor input_c = CreateInputTensor(shape, 2 * shape.num_elements());
    TF_RETURN_IF_ERROR(session->Run(
        run_options,
        {std::make_pair("a", input_a), std::make_pair("b", input_b),
         std::make_pair("c", input_c)},
        {"m"}, {}, &output_tensors, nullptr));
    TF_RETURN_IF_ERROR(session->Close());
  }

  Tensor f32_input(DT_FLOAT, shape);
  for (int64 i = 0; i < f32_input.NumElements(); ++i) {
    EXPECT_NEAR(golden_output_tensors[0].flat<float>()(i),
                output_tensors[0].flat<float>()(i), 1e-3);
  }
  return Status::OK();
}

Status
XlaCompilationCacheSerializeTest::AlterPersistentCacheEntryHloModuleNames(
    absl::string_view persistent_cache_dir_path,
    absl::string_view file_prefix) {
  Env* env = Env::Default();
  std::vector<string> file_names;
  TF_RETURN_IF_ERROR(
      env->GetChildren(tensorflow::testing::TmpDir(), &file_names));

  bool altered = false;
  for (const auto& file_name : file_names) {
    if (absl::EndsWith(file_name, ".pb") &&
        absl::StartsWith(file_name, file_prefix)) {
      XlaSerializedCacheEntry entry;
      auto file_path = io::JoinPath(persistent_cache_dir_path, file_name);
      TF_RETURN_IF_ERROR(ReadTextOrBinaryProto(env, file_path, &entry));
      entry.mutable_hlo_module()->set_name(
          absl::StrCat(entry.hlo_module().name(), "_altered"));
      TF_RETURN_IF_ERROR(WriteBinaryProto(env, file_path, entry));
      altered = true;
    }
  }

  if (!altered) {
    return errors::NotFound(
        "Did not find any persistent XLA compilation cache entries to alter.");
  }
  return Status::OK();
}

}  // namespace tensorflow
