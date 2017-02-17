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
#include "tensorflow/contrib/util/convert_graphdef_memmapped_format_lib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"

namespace tensorflow {
namespace {

bool GraphHasImmutableConstNodes(const GraphDef& graph_def) {
  for (const auto& node : graph_def.node()) {
    if (node.op() == "ImmutableConst") {
      return true;
    }
  }
  return false;
}

TEST(ConvertGraphdefMemmappedFormatTest, ConvertModel) {
  const string dir = testing::TmpDir();
  const string filename_pb = io::JoinPath(dir, "graphdef.pb");

  // Create a simple graph and write it to filename_pb.
  constexpr int kTensorWidth = 4000;
  constexpr int kTensorHeight = 100;
  const TensorShape kTestTensorShape({kTensorWidth, kTensorHeight});
  const TensorShape kTestTensorShapeT({kTensorHeight, kTensorWidth});

  Tensor test_tensor1(DT_FLOAT, kTestTensorShape);
  test::FillFn<float>(&test_tensor1, [](int) -> float { return 2.0; });

  Tensor test_tensor2(DT_FLOAT, kTestTensorShapeT);
  test::FillFn<float>(&test_tensor2, [](int) -> float { return 3.0; });

  auto root = Scope::NewRootScope().ExitOnError();
  Output m = ops::MatMul(root, test_tensor1, test_tensor2);
  const string result_name = m.node()->name();

  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));
  string graph_def_serialized;
  graph_def.SerializeToString(&graph_def_serialized);
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), filename_pb, graph_def_serialized));

  const string filename_mmap = io::JoinPath(dir, "graphdef.mmap");
  TF_ASSERT_OK(ConvertConstantsToImmutable(filename_pb, filename_mmap, 10000));

  // Create and initialize MemmappedEnv from the converted file.
  MemmappedEnv memmapped_env(Env::Default());
  TF_ASSERT_OK(memmapped_env.InitializeFromFile(filename_mmap));

  // Load the graph and run calculations.
  SessionOptions session_options;
  session_options.env = &memmapped_env;
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  GraphDef loaded_graph_def;
  TF_ASSERT_OK(ReadBinaryProto(
      &memmapped_env, MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
      &loaded_graph_def));
  ASSERT_TRUE(GraphHasImmutableConstNodes(loaded_graph_def));

  TF_ASSERT_OK(session->Create(loaded_graph_def)) << "Can't create test graph";
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {result_name + ":0"}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs.front().flat<float>()(0), 2.0f * 3.0f * kTensorHeight);
  EXPECT_EQ(outputs.front().flat<float>()(1), 2.0f * 3.0f * kTensorHeight);
  EXPECT_EQ(outputs.front().flat<float>()(2), 2.0f * 3.0f * kTensorHeight);
}

TEST(ConvertGraphdefMemmappedFormatTest, NotSupportedTypesConvert) {
  // Create a graph with strings.
  const string dir = testing::TmpDir();
  const string filename_pb = io::JoinPath(dir, "string_graphdef.pb");

  constexpr int kTensorWidth = 4000;
  constexpr int kTensorHeight = 100;
  const TensorShape kTestTensorShape({kTensorWidth, kTensorHeight});
  Tensor test_tensor1(DT_STRING, kTestTensorShape);
  test::FillFn<string>(&test_tensor1, [](int) -> string { return "ABC"; });

  Tensor test_tensor2(DT_STRING, kTestTensorShape);
  test::FillFn<string>(&test_tensor2, [](int) -> string { return "XYZ"; });
  auto root = Scope::NewRootScope().ExitOnError();
  Output m = ops::Add(root, test_tensor1, test_tensor2);
  const string result_name = m.node()->name();

  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));
  string graph_def_serialized;
  graph_def.SerializeToString(&graph_def_serialized);
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), filename_pb, graph_def_serialized));

  const string filename_mmap = io::JoinPath(dir, "string_graphdef.mmap");
  TF_ASSERT_OK(ConvertConstantsToImmutable(filename_pb, filename_mmap, 1000));

  // Create and initialize MemmappedEnv from the converted file.
  MemmappedEnv memmapped_env(Env::Default());
  TF_ASSERT_OK(memmapped_env.InitializeFromFile(filename_mmap));

  // Load the graph and run calculations.
  SessionOptions session_options;
  session_options.env = &memmapped_env;
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  GraphDef loaded_graph_def;
  TF_ASSERT_OK(ReadBinaryProto(
      &memmapped_env, MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
      &loaded_graph_def));
  ASSERT_FALSE(GraphHasImmutableConstNodes(loaded_graph_def));
}

}  // namespace
}  // namespace tensorflow
