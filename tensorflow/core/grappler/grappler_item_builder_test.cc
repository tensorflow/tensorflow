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

// Create a sample graph with a symbolic gradient for sum.
void SampleSumSymbolicGradientGraphdef(
    GraphDef *def, CollectionDef *fetches,
    std::vector<string> *names_of_ops_of_inline) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto dummy_variable = Variable(scope, {2, 2}, DT_FLOAT);
  auto x = Const(scope, 1.0f);
  auto y = Const(scope, 2);
  auto z = Const(scope, 3.0f);
  TF_ASSERT_OK(scope.status());

  NameAttrList fn;
  fn.set_name("Sum");
  (*fn.mutable_attr())["T"].set_type(DT_FLOAT);
  auto g0 = SymbolicGradient(scope, std::initializer_list<Input>{x, y, z},
                             {DT_FLOAT, DT_INT32}, fn);

  // TODO(bsteiner): we should rewrite the feed/fetch nodes to reflect the
  // inlining that's done in the item builder
  // fetches->mutable_node_list()->add_value(g0[0].name());
  fetches->mutable_node_list()->add_value("SymbolicGradient/dx");
  fetches->mutable_node_list()->add_value("SymbolicGradient/dy_reshaped");

  TF_CHECK_OK(scope.ToGraphDef(def));

  // Add names of the ops that replace the Mul symbolic gradient during
  // inlining. This is for validation.
  *names_of_ops_of_inline = {
      "SymbolicGradient/dx",          "SymbolicGradient/tile_scaling",
      "SymbolicGradient/dy_reshaped", "SymbolicGradient/y_shape",
      "SymbolicGradient/x_shape",     "SymbolicGradient/stitch_idx0",
      "SymbolicGradient/x_rank",      "SymbolicGradient/stitch_val1",
      "SymbolicGradient/i_shape",     "SymbolicGradient/di",
      "SymbolicGradient/zero",        "SymbolicGradient/one"};
}

std::unique_ptr<GrapplerItem> CreateGrapplerItem(const GraphDef &def,
                                                 const CollectionDef &fetches) {
  MetaGraphDef meta_def;
  ItemConfig cfg;
  cfg.inline_functions = true;
  *meta_def.mutable_graph_def() = def;
  (*meta_def.mutable_collection_def())["train_op"] = fetches;
  return GrapplerItemFromMetaGraphDef("0", meta_def, cfg);
}

int CountSymbolicGradientOps(const std::unique_ptr<GrapplerItem> &item) {
  int n_symb_grads = 0;
  for (const auto &node : item->graph.node()) {
    if (node.op() == FunctionLibraryDefinition::kGradientOp) {
      n_symb_grads++;
    }
  }
  return n_symb_grads;
}

int CountOpsWithNames(const std::unique_ptr<GrapplerItem> &item,
                      const std::vector<string> &names) {
  std::set<string> names_set(names.begin(), names.end());
  int n_with_names = 0;
  for (const auto &node : item->graph.node()) {
    if (names_set.find(node.name()) != names_set.end()) {
      n_with_names++;
    }
  }
  return n_with_names;
}

TEST_F(GrapplerItemBuilderTest, SymbolicGradientInlining) {
  // Create sample sum symbolic gradient graph.
  GraphDef def;
  CollectionDef fetches;
  std::vector<string> ops_of_inline;
  SampleSumSymbolicGradientGraphdef(&def, &fetches, &ops_of_inline);

  // Create the inlined graph.
  std::unique_ptr<GrapplerItem> with_inline = CreateGrapplerItem(def, fetches);

  // For the inlined graph, there should be 0 symbolic gradient ops.
  EXPECT_EQ(0, CountSymbolicGradientOps(with_inline));

  // For the inlined graph, make sure all the required expanded op’s are in the
  // graph.
  EXPECT_EQ(ops_of_inline.size(),
            CountOpsWithNames(with_inline, ops_of_inline));
}

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
  cfg.inline_functions = false;

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, cfg);
  ASSERT_TRUE(item != nullptr);
}

TEST_F(GrapplerItemBuilderTest, FromSimpleFunctionDef) {
  const Tensor kTwo = test::AsScalar<int64>(2);
  FunctionDef func = FunctionDefHelper::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);
  FunctionDefLibrary library;
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);
  CHECK(item);
  EXPECT_EQ("XTimesTwo", item->id);
  EXPECT_EQ(4, item->graph.node_size());
  EXPECT_EQ(std::vector<string>({"y"}), item->fetch);
  EXPECT_EQ(1, item->feed.size());
  EXPECT_EQ("x", item->feed[0].first);

  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "x") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "two") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "scale") {
      EXPECT_EQ("Cast", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("DstT").type());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("two:0", node.input(0));
    } else if (node.name() == "y") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("scale:0", node.input(1));
    }
  }
}

TEST_F(GrapplerItemBuilderTest, FromFunctionDefWithMultiOutputNodes) {
  // Gradient graph for the Subtract operation
  std::vector<FunctionDefHelper::Node> nodes = {
      {{"sx"}, "Shape", {"x"}},
      {{"sy"}, "Shape", {"y"}},
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Neg", {"dz"}},
      {{"rx", "ry"}, "BroadcastGradientArgs", {"sx", "sy"}},
      {{"sum_gx"}, "Sum", {"gx", "rx"}},
      {{"dx"}, "Reshape", {"sum_gx", "sx"}},
      {{"sum_gy"}, "Sum", {"gy", "ry"}},
      {{"dy"}, "Reshape", {"sum_gy", "sy"}},
  };

  for (auto &n : nodes) {
    // "BroadcastGradientArgs" doesn't need any attrs.
    if (n.attr.empty() && n.op != "BroadcastGradientArgs") {
      n.attr = {{"T", "$T"}};
    }
  }
  FunctionDef func = FunctionDefHelper::Define(
      // Name
      "SubGrad",
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);
  FunctionDefLibrary library;
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);
  CHECK(item);
  EXPECT_EQ("SubGrad", item->id);
  EXPECT_EQ(12, item->graph.node_size());
  EXPECT_EQ(std::vector<string>({"dx", "dy"}), item->fetch);
  EXPECT_EQ(3, item->feed.size());
  EXPECT_EQ("x", item->feed[0].first);
  EXPECT_EQ("y", item->feed[1].first);
  EXPECT_EQ("dz", item->feed[2].first);

  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "x" || node.name() == "y" || node.name() == "dz") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "rx") {
      EXPECT_EQ("BroadcastGradientArgs", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("sx:0", node.input(0));
      EXPECT_EQ("sy:0", node.input(1));
    } else if (node.name() == "sum_gx") {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gx:0", node.input(0));
      EXPECT_EQ("rx:0", node.input(1));
    } else if (node.name() == "sum_gy") {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gy:0", node.input(0));
      EXPECT_EQ("rx:1", node.input(1));
    }
  }
}

TEST_F(GrapplerItemBuilderTest, FromFunctionDefWithNestedFuncs) {
  FunctionDefLibrary library;
  *library.add_function() = FunctionDefHelper::Define(
      // Name
      "Swap",
      // Args
      {"i0: T", "i1: T"},
      // Return values
      {"o0: T", "o1: T"},
      // Attr def
      {"T: {float, double}"},
      // Nodes
      {{{"o0"}, "Identity", {"i1"}, {{"T", "$T"}}},
       {{"o1"}, "Identity", {"i0"}, {{"T", "$T"}}}});

  FunctionDef func = FunctionDefHelper::Create(
      // Name
      "ManySwapsFirst",
      // Args
      {"x: float", "y: float"},
      // Return values
      {"o: float"},
      // attr def
      {},
      // Nodes
      // o = x*x + y*y.  Furthermore, The 1st swap depends on x2, and
      // y2 depends on the 2nd swap.  The 2nd swap has data dependency
      // on the 1st swap.
      {{{"a0"}, "Swap", {"x", "y"}, {{"T", DT_FLOAT}}, {"x2"}},
       {{"a1"}, "Swap", {"a0:o0:0", "a0:o1:0"}, {{"T", DT_FLOAT}}},
       {{"x2"}, "Mul", {"x", "x"}, {{"T", DT_FLOAT}}},
       {{"y2"}, "Mul", {"y", "y"}, {{"T", DT_FLOAT}}, {"a1"}},
       {{"o"}, "Add", {"x2:z:0", "y2:z:0"}, {{"T", DT_FLOAT}}}},
      {{"o", "o:z:0"}});

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);

  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "x" || node.name() == "y") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "a0") {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^x2", node.input(2));
    } else if (node.name() == "a1") {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("a0:0", node.input(0));
      EXPECT_EQ("a0:1", node.input(1));
    } else if (node.name() == "x2") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("x", node.input(1));
    } else if (node.name() == "y2") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^a1", node.input(2));
    } else if (node.name() == "o") {
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x2:0", node.input(0));
      EXPECT_EQ("y2:0", node.input(1));
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
