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

#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "google/protobuf/any.pb.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class GrapplerItemBuilderTest : public ::testing::Test {};

TEST_F(GrapplerItemBuilderTest, AssetFilepathOverrideTest) {
  MetaGraphDef meta_graph;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output var =
      ops::Variable(s.WithOpName("var"), TensorShape(), DataType::DT_FLOAT);
  Output filename_node =
      ops::Const(s.WithOpName("filename"), string("model"), TensorShape());
  Output tensor_name =
      ops::Const(s.WithOpName("tensorname"), string("var"), TensorShape());
  Output restore = ops::Restore(s.WithOpName("restore"), filename_node,
                                tensor_name, DataType::DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign"), var, restore);

  TF_CHECK_OK(s.ToGraphDef(meta_graph.mutable_graph_def()));

  string temp_dir = testing::TmpDir();

  Env *env = Env::Default();
  string filename =
      io::JoinPath(temp_dir, "grappler_item_builder_test_filename");
  env->DeleteFile(filename).IgnoreError();
  std::unique_ptr<WritableFile> file_to_write;
  TF_CHECK_OK(env->NewWritableFile(filename, &file_to_write));
  TF_CHECK_OK(file_to_write->Close());
  TF_CHECK_OK(env->FileExists(filename));
  LOG(INFO) << filename;

  AssetFileDef asset_file_def;
  *asset_file_def.mutable_tensor_info()->mutable_name() = "filename";
  *asset_file_def.mutable_filename() = "grappler_item_builder_test_filename";

  (*meta_graph.mutable_collection_def())["saved_model_assets"]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(asset_file_def);
  *((*meta_graph.mutable_collection_def())["train_op"]
        .mutable_node_list()
        ->add_value()) = "assign";

  ItemConfig cfg;
  cfg.assets_directory_override = temp_dir;

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, cfg);
  ASSERT_TRUE(item != nullptr);
  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "filename") {
      const auto iter = node.attr().find("value");
      ASSERT_TRUE(iter != node.attr().end());
      ASSERT_TRUE(iter->second.has_tensor());
      ASSERT_EQ(1, iter->second.tensor().string_val_size());

      string tensor_string_val = iter->second.tensor().string_val(0);
      EXPECT_EQ(tensor_string_val, filename);
    }
  }
}

TEST_F(GrapplerItemBuilderTest, AssetFilepathOverrideTest_FileNotAccessible) {
  MetaGraphDef meta_graph;

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output var =
      ops::Variable(s.WithOpName("var"), TensorShape(), DataType::DT_FLOAT);
  Output filename_node1 =
      ops::Const(s.WithOpName("filename1"), string("model1"), TensorShape());
  Output filename_node2 =
      ops::Const(s.WithOpName("filename2"), string("model2"), TensorShape());
  Output tensor_name =
      ops::Const(s.WithOpName("tensorname"), string("var"), TensorShape());
  Output restore1 = ops::Restore(s.WithOpName("restore1"), filename_node1,
                                 tensor_name, DataType::DT_FLOAT);
  Output restore2 = ops::Restore(s.WithOpName("restore2"), filename_node1,
                                 tensor_name, DataType::DT_FLOAT);
  Output assign1 = ops::Assign(s.WithOpName("assign1"), var, restore1);
  Output assign2 = ops::Assign(s.WithOpName("assign2"), var, restore2);

  TF_CHECK_OK(s.ToGraphDef(meta_graph.mutable_graph_def()));

  string temp_dir = testing::TmpDir();

  // Create the first AssetFileDef that has a valid file.
  Env *env = Env::Default();
  string filename1 =
      io::JoinPath(temp_dir, "grappler_item_builder_test_filename1");
  env->DeleteFile(filename1).IgnoreError();
  std::unique_ptr<WritableFile> file_to_write;
  TF_CHECK_OK(env->NewWritableFile(filename1, &file_to_write));
  TF_CHECK_OK(file_to_write->Close());
  TF_CHECK_OK(env->FileExists(filename1));

  AssetFileDef asset_file_def1;
  *asset_file_def1.mutable_tensor_info()->mutable_name() = "filename1";
  *asset_file_def1.mutable_filename() = "grappler_item_builder_test_filename1";

  // Create the second AssetFileDef that has not a valid file.
  string filename2 =
      io::JoinPath(temp_dir, "grappler_item_builder_test_filename1");
  env->DeleteFile(filename2).IgnoreError();
  EXPECT_FALSE(env->FileExists(filename2).ok());

  AssetFileDef asset_file_def2;
  *asset_file_def2.mutable_tensor_info()->mutable_name() = "filename2";
  *asset_file_def2.mutable_filename() = "grappler_item_builder_test_filename2";

  (*meta_graph.mutable_collection_def())["saved_model_assets"]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(asset_file_def1);
  (*meta_graph.mutable_collection_def())["saved_model_assets"]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(asset_file_def2);

  *((*meta_graph.mutable_collection_def())["train_op"]
        .mutable_node_list()
        ->add_value()) = "assign1";
  *((*meta_graph.mutable_collection_def())["train_op"]
        .mutable_node_list()
        ->add_value()) = "assign2";

  ItemConfig cfg;
  cfg.assets_directory_override = temp_dir;

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, cfg);
  ASSERT_TRUE(item == nullptr);
}

TEST_F(GrapplerItemBuilderTest, GraphWithFunctions) {
  MetaGraphDef meta_graph;
  // y = XTimesTwo(x)
  constexpr char device[] = "/cpu:0";
  *meta_graph.mutable_graph_def() = test::function::GDef(
      {test::function::NDef("x", "Const", {}, {{"dtype", DT_FLOAT}}, device),
       test::function::NDef("y", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}},
                            device)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  CollectionDef train_op;
  train_op.mutable_node_list()->add_value("y");
  (*meta_graph.mutable_collection_def())["train_op"] = train_op;

  ItemConfig cfg;

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, cfg);
  ASSERT_TRUE(item != nullptr);
}

TEST_F(GrapplerItemBuilderTest, GraphWithCustomOps) {
  MetaGraphDef meta_graph;
  // y = XTimesTwo(x)
  constexpr char device[] = "/cpu:0";
  *meta_graph.mutable_graph_def() = test::function::GDef(
      {test::function::NDef("x", "Const", {}, {{"dtype", DT_FLOAT}}, device),
       test::function::NDef("y", "CustomOp", {"x"}, {{"T", DT_FLOAT}}, device)},
      {});

  CollectionDef train_op;
  train_op.mutable_node_list()->add_value("y");
  (*meta_graph.mutable_collection_def())["train_op"] = train_op;

  ItemConfig cfg;

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, cfg);
  ASSERT_TRUE(item != nullptr);
}

TEST_F(GrapplerItemBuilderTest, FromGraphWithSignatureDef) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), 0);
  auto y = ops::Const(s.WithOpName("y"), 1);
  auto z = ops::Add(s.WithOpName("z"), x, y);

  MetaGraphDef meta_graph;
  TF_CHECK_OK(s.ToGraphDef(meta_graph.mutable_graph_def()));

  TensorInfo input, output;
  input.set_name("x");
  input.set_dtype(DT_FLOAT);
  output.set_name("z");
  SignatureDef serving_signature;
  (*serving_signature.mutable_inputs())["input"] = input;
  (*serving_signature.mutable_outputs())["output"] = output;
  (*meta_graph.mutable_signature_def())["serving"] = serving_signature;

  // It should be able to dedup the input and output with same names.
  TensorInfo input2, output2;
  input.set_name("x");
  input.set_dtype(DT_FLOAT);
  output.set_name("z");
  SignatureDef serving_signature2;
  (*serving_signature.mutable_inputs())["input2"] = input2;
  (*serving_signature.mutable_outputs())["output2"] = output2;
  (*meta_graph.mutable_signature_def())["serving2"] = serving_signature2;

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, ItemConfig());
  ASSERT_TRUE(item != nullptr);

  EXPECT_EQ(item->feed.size(), 1);
  EXPECT_EQ(item->fetch.size(), 1);
  EXPECT_EQ(item->feed[0].first, "x");
  EXPECT_EQ(item->fetch[0], "z");
}

TEST_F(GrapplerItemBuilderTest, FromGraphWithIncompleteSignatureDef) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), 0);
  auto y = ops::Const(s.WithOpName("y"), 1);

  MetaGraphDef meta_graph;
  TF_CHECK_OK(s.ToGraphDef(meta_graph.mutable_graph_def()));

  CollectionDef train_op;
  train_op.mutable_node_list()->add_value("y");
  (*meta_graph.mutable_collection_def())["train_op"] = train_op;

  TensorInfo input, output;
  input.set_name("x");
  input.set_dtype(DT_FLOAT);
  // Its coo_sparse proto is incomplete.
  output.mutable_coo_sparse()->set_values_tensor_name("z");
  SignatureDef serving_signature;
  (*serving_signature.mutable_inputs())["input"] = input;
  (*serving_signature.mutable_outputs())["output"] = output;
  (*meta_graph.mutable_signature_def())["serving"] = serving_signature;

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, ItemConfig());
  ASSERT_TRUE(item == nullptr);
}

TEST_F(GrapplerItemBuilderTest, FromGraphWithUnknownDimInSignatureInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto shape_1d = PartialTensorShape({-1});
  auto x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                            ops::Placeholder::Shape(shape_1d));
  auto y = ops::Const(s.WithOpName("y"), static_cast<float>(1.0));
  auto z = ops::Add(s.WithOpName("z"), x, y);

  MetaGraphDef meta_graph;
  TF_CHECK_OK(s.ToGraphDef(meta_graph.mutable_graph_def()));

  TensorInfo input, output;
  input.set_name("x");
  input.set_dtype(DT_FLOAT);
  shape_1d.AsProto(input.mutable_tensor_shape());
  output.set_name("z");

  SignatureDef serving_signature;
  (*serving_signature.mutable_inputs())["input"] = input;
  (*serving_signature.mutable_outputs())["output"] = output;
  (*meta_graph.mutable_signature_def())["serving"] = serving_signature;

  ItemConfig cfg;
  cfg.placeholder_unknown_output_shape_dim = 64;
  std::unique_ptr<GrapplerItem> item1 =
      GrapplerItemFromMetaGraphDef("0", meta_graph, cfg);
  ASSERT_TRUE(item1 != nullptr);

  ASSERT_EQ(item1->feed.size(), 1);
  EXPECT_EQ(item1->feed[0].second.NumElements(), 64);

  std::unique_ptr<GrapplerItem> item2 =
      GrapplerItemFromMetaGraphDef("0", meta_graph, ItemConfig());
  ASSERT_TRUE(item2 != nullptr);

  ASSERT_EQ(item2->feed.size(), 1);
  EXPECT_EQ(item2->feed[0].second.NumElements(), 1);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
