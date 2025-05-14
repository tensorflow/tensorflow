/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_saved_model_export.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_test_base.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace mlir::tf_quant::stablehlo {
namespace {

using ::tensorflow::AssetFileDef;
using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::tensorflow::SaverDef;
using ::tensorflow::quantization::ExportedModel;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

TEST(CreateExportedModelTest, CreateExportedModelBasicFieldsSet) {
  GraphDef graph_def{};
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(node { name: "foo" })pb", &graph_def));

  const ExportedModel exported_model = CreateExportedModelFromGraphDef(
      std::move(graph_def), "init_node_name", "checkpoint_dir",
      /*saver_def=*/std::nullopt,
      /*function_aliases=*/{}, /*asset_file_defs=*/{});
  ASSERT_THAT(exported_model.graph_def().node(), SizeIs(1));
  EXPECT_THAT(exported_model.graph_def().node()[0].name(), StrEq("foo"));

  EXPECT_THAT(exported_model.init_node_name(), StrEq("init_node_name"));
  EXPECT_THAT(exported_model.checkpoint_dir(), StrEq("checkpoint_dir"));
  EXPECT_FALSE(exported_model.has_saver_def());
  EXPECT_THAT(exported_model.function_aliases(), IsEmpty());
  EXPECT_THAT(exported_model.asset_file_defs(), IsEmpty());
}

TEST(CreateExportedModelTest, CreateExportedModelWithAddedFunctionAliases) {
  const ExportedModel exported_model = CreateExportedModelFromGraphDef(
      GraphDef(), /*init_node_name=*/"", /*checkpoint_dir=*/"",
      /*saver_def=*/std::nullopt,
      /*function_aliases=*/{{"func1", "alias1"}, {"func2", "alias2"}},
      /*asset_file_defs=*/{});
  ASSERT_THAT(exported_model.function_aliases(), SizeIs(2));
  EXPECT_TRUE(exported_model.function_aliases().contains("func1"));
  EXPECT_THAT(exported_model.function_aliases().at("func1"), StrEq("alias1"));
  EXPECT_TRUE(exported_model.function_aliases().contains("func2"));
  EXPECT_THAT(exported_model.function_aliases().at("func2"), StrEq("alias2"));
}

TEST(CreateExportedModelTest, CreateExportedModelWithAddedAssetFileDefs) {
  AssetFileDef asset1;
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(filename: "fname1")pb", &asset1));

  AssetFileDef asset2;
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(filename: "fname2")pb", &asset2));

  const ExportedModel exported_model = CreateExportedModelFromGraphDef(
      GraphDef(), /*init_node_name=*/"", /*checkpoint_dir=*/"",
      /*saver_def=*/std::nullopt, /*function_aliases=*/{},
      /*asset_file_defs=*/{asset1, asset2});
  ASSERT_THAT(exported_model.asset_file_defs(), SizeIs(2));
  EXPECT_THAT(exported_model.asset_file_defs()[0].filename(), StrEq("fname1"));
  EXPECT_THAT(exported_model.asset_file_defs()[1].filename(), StrEq("fname2"));
}

TEST(CreateExportedModelTest, CreateExportedModelWithAddedSaverDef) {
  SaverDef saver_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(filename_tensor_name: "my_file")pb", &saver_def));

  const ExportedModel exported_model = CreateExportedModelFromGraphDef(
      GraphDef(), /*init_node_name=*/"", /*checkpoint_dir=*/"", saver_def,
      /*function_aliases=*/{}, /*asset_file_defs=*/{});
  EXPECT_THAT(exported_model.saver_def().filename_tensor_name(), "my_file");
}

TEST(CreateSaverDefTest, CreateValidSaverDef) {
  // Needs to have a _Arg node with an attribute "tf_saved_model.index_path" =
  // ["__tf_file_prefix"].
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(node {
             name: "foo",
             op: "_Arg",
             attr {
               key: "tf_saved_model.index_path",
               value { list { s: "__tf_file_prefix" } }
             }
           })pb",
      &graph_def));

  // Restore op's name should start with "restore_op" and the save op's name
  // should start with "tf_quant__save_op".
  const std::vector<std::string> control_ret_node_names = {
      "restore_op_0", "tf_quant__save_op_0"};

  TF_ASSERT_OK_AND_ASSIGN(const std::optional<SaverDef> saver_def,
                          CreateSaverDef(control_ret_node_names, graph_def));
  ASSERT_NE(saver_def, std::nullopt);
  EXPECT_THAT(saver_def->version(), SaverDef::V2);
  EXPECT_THAT(saver_def->restore_op_name(), "restore_op_0");
  EXPECT_THAT(saver_def->filename_tensor_name(), "foo:0");
  EXPECT_THAT(saver_def->save_tensor_name(), "tf_quant__save_op_0:0");
}

TEST(CreateSaverDefTest, ReturnsNulloptIfNoSaverDefRelatedNodesExist) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::optional<SaverDef> saver_def,
      CreateSaverDef(/*control_ret_node_names=*/{}, GraphDef()));
  EXPECT_EQ(saver_def, std::nullopt);
}

TEST(CreateSaverDefTest, ReturnsErrorStatusIfSaverDefNodesPartiallyExist) {
  // An _Arg node missing the attribute "tf_saved_model.index_path" =
  // ["__tf_file_prefix"].
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(node { name: "foo", op: "_Arg" })pb", &graph_def));

  // Restore op's name should start with "restore_op" and the save op's name
  // should start with "tf_quant__save_op".
  const std::vector<std::string> control_ret_node_names = {
      "restore_op_0", "tf_quant__save_op_0"};

  const absl::StatusOr<std::optional<SaverDef>> saver_def =
      CreateSaverDef(control_ret_node_names, graph_def);
  EXPECT_THAT(
      saver_def,
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "should be either all empty strings or all non-empty strings")));
}

// Testing ConvertMlirModuleToExportedModel requires parsing MLIR string to
// ModuleOp.
using ConvertMlirModuleToExportedModelTest =
    ::mlir::tf_quant::QuantizationTestBase;

TEST_F(ConvertMlirModuleToExportedModelTest, SimpleGraphDefSet) {
  // Define a module a no-op main function.
  mlir::OwningOpRef<mlir::ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {tf_saved_model.semantics} {
      func.func @main(%arg: tensor<1x2xf32> {tf_saved_model.index_path = ["input_tensor:0"]}) -> (tensor<1x2xf32> {tf_saved_model.index_path = ["output_tensor:0"]}) attributes {tf.entry_function = {inputs = "input_tensor:0", outputs = "output_tensor:0"}, tf_saved_model.exported_names = ["main"]} {
        %0 = tf_executor.graph {
          tf_executor.fetch %arg : tensor<1x2xf32>
        }
        return %0 : tensor<1x2xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  const absl::StatusOr<ExportedModel> exported_model =
      ConvertMlirModuleToExportedModel(*module_op, /*checkpoint_dir=*/"",
                                       /*function_aliases=*/{},
                                       /*asset_file_defs=*/{});

  ASSERT_THAT(exported_model, IsOk());
  // There are 2 nodes in the graph, one for arg and another for retval.
  ASSERT_THAT(exported_model->graph_def().node(), SizeIs(2));

  // Match the `_Arg` node that corresponds to the argument of @main.
  const auto arg_node_itr =
      llvm::find_if(exported_model->graph_def().node(),
                    [](const NodeDef& node) { return node.op() == "_Arg"; });
  ASSERT_NE(arg_node_itr, exported_model->graph_def().node().end());
  EXPECT_THAT(arg_node_itr->name(), StrEq("input_tensor"));
  ASSERT_TRUE(arg_node_itr->attr().contains("tf_saved_model.index_path"));
  ASSERT_THAT(arg_node_itr->attr().at("tf_saved_model.index_path").list().s(),
              SizeIs(1));
  EXPECT_THAT(
      arg_node_itr->attr().at("tf_saved_model.index_path").list().s()[0],
      StrEq("input_tensor:0"));

  // Match the `_Retval` node that corresponds to the return value of @main.
  const auto retval_node_itr =
      llvm::find_if(exported_model->graph_def().node(),
                    [](const NodeDef& node) { return node.op() == "_Retval"; });
  ASSERT_NE(retval_node_itr, exported_model->graph_def().node().end());
  EXPECT_THAT(retval_node_itr->name(), StrEq("output_tensor"));
  ASSERT_TRUE(retval_node_itr->attr().contains("tf_saved_model.index_path"));
  ASSERT_THAT(
      retval_node_itr->attr().at("tf_saved_model.index_path").list().s(),
      SizeIs(1));
  EXPECT_THAT(
      retval_node_itr->attr().at("tf_saved_model.index_path").list().s()[0],
      StrEq("output_tensor:0"));
}

TEST_F(ConvertMlirModuleToExportedModelTest, CheckpointDirSet) {
  // Define a module a no-op main function.
  mlir::OwningOpRef<mlir::ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {tf_saved_model.semantics} {
      func.func @main() -> () attributes {tf_saved_model.exported_names = ["main"]} {
        tf_executor.graph {
          tf_executor.fetch
        }
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  const absl::StatusOr<ExportedModel> exported_model =
      ConvertMlirModuleToExportedModel(*module_op, "my_checkpoint_dir",
                                       /*function_aliases=*/{},
                                       /*asset_file_defs=*/{});

  ASSERT_THAT(exported_model, IsOk());
  EXPECT_THAT(exported_model->checkpoint_dir(), StrEq("my_checkpoint_dir"));
}

TEST_F(ConvertMlirModuleToExportedModelTest, FunctionAliasesSet) {
  // Define a module with 2 function calls, function_1 and function_2.
  mlir::OwningOpRef<mlir::ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {tf_saved_model.semantics} {
      func.func private @function_1() -> () attributes {tf._original_func_name = "__func_1"} {
        tf_executor.graph {
          %control_0 = tf_executor.island wraps "tf.NoOp"() : () -> ()
        }
        return
      }

      func.func private @function_2() -> () attributes {tf._original_func_name = "__func_2"} {
        tf_executor.graph {
          %control_0 = tf_executor.island wraps "tf.NoOp"() : () -> ()
        }
        return
      }

      func.func @main() -> () attributes {tf_saved_model.exported_names = ["main"]} {
        tf_executor.graph {
          %control_0 = tf_executor.island wraps "tf.PartitionedCall"() <{config = "", config_proto = "", executor_type = "", f = @function_1}> : () -> ()
          %control_1 = tf_executor.island wraps "tf.PartitionedCall"() <{config = "", config_proto = "", executor_type = "", f = @function_2}> : () -> ()
          tf_executor.fetch %control_0, %control_1 : !tf_executor.control, !tf_executor.control
        }
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  const absl::StatusOr<ExportedModel> exported_model =
      ConvertMlirModuleToExportedModel(
          *module_op, /*checkpoint_dir=*/"",
          /*function_aliases=*/
          {{"alias_1", "function_1"}, {"alias_2", "function_2"}},
          /*asset_file_defs=*/{});

  ASSERT_THAT(exported_model, IsOk());
  ASSERT_THAT(exported_model->function_aliases(), SizeIs(2));
  EXPECT_THAT(exported_model->function_aliases().at("alias_1"),
              StrEq("function_1"));
  EXPECT_THAT(exported_model->function_aliases().at("alias_2"),
              StrEq("function_2"));
}

TEST_F(ConvertMlirModuleToExportedModelTest, AssetFileDefSet) {
  // Define a module a no-op main function.
  mlir::OwningOpRef<mlir::ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {tf_saved_model.semantics} {
      func.func @main() -> () attributes {tf_saved_model.exported_names = ["main"]} {
        tf_executor.graph {
          tf_executor.fetch
        }
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  AssetFileDef asset_file_def{};
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(filename: "vocab_file.txt",
                                       tensor_info { name: "arg_0:0" })pb",
                                  &asset_file_def));
  const std::vector<AssetFileDef> asset_file_defs = {asset_file_def};

  const absl::StatusOr<ExportedModel> exported_model =
      ConvertMlirModuleToExportedModel(*module_op, /*checkpoint_dir=*/"",
                                       /*function_aliases=*/{},
                                       /*asset_file_defs=*/asset_file_defs);

  ASSERT_THAT(exported_model, IsOk());
  ASSERT_THAT(exported_model->asset_file_defs(), SizeIs(1));
  EXPECT_THAT(exported_model->asset_file_defs()[0].filename(),
              StrEq("vocab_file.txt"));
  EXPECT_THAT(exported_model->asset_file_defs()[0].tensor_info().name(),
              StrEq("arg_0:0"));
}

TEST_F(ConvertMlirModuleToExportedModelTest,
       InitNodeNameSetToLocOfControlOutput) {
  // Define a module that initializes a tf.HashTableV2 whose control output node
  // for the initialization is named "init_op_init_all_tables".
  mlir::OwningOpRef<mlir::ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() <{initializers = []}> : () -> ()
      "tf_saved_model.asset"() <{filename = "assets/vocab_file.txt", sym_name = "__tf_saved_model_asset0_vocab_file.txt"}> : () -> ()
      func.func @main(%arg1: tensor<!tf_type.string> {tf_saved_model.index_path = ["arg_0:0"]}) -> (tensor<1x2xf32> {tf_saved_model.index_path = ["output:0"]}) attributes {tf.entry_function = {inputs = "arg_0:0", outputs = "output:0"}, tf_saved_model.exported_names = ["main"]} {
        %0 = tf_executor.graph {
          %o_0, %c_0 = tf_executor.island wraps "tf.Const"() <{value = dense<1.0> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
          %o, %c = tf_executor.island wraps "tf.HashTableV2"() <{container = "", key_dtype = !tf_type.string, shared_name = "vocab_file.txt", use_node_name_sharing = false, value_dtype = i64}> {device = ""} : () -> tensor<!tf_type.resource>
          %c_9 = tf_executor.island wraps "tf.InitializeTableFromTextFileV2"(%o, %arg1) <{delimiter = "\09", key_index = -2 : i64, value_index = -1 : i64, vocab_size = -1 : i64}> {_has_manual_control_dependencies = true, device = ""} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
          // Location of this control output op becomes the name of the init_op.
          %c_10 = tf_executor.island(%c_9) wraps "tf.NoOp"() : () -> () loc("init_op_init_all_tables")
          tf_executor.fetch %o_0, %c_10 : tensor<1x2xf32>, !tf_executor.control
        }
        return %0 : tensor<1x2xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  const absl::StatusOr<ExportedModel> exported_model =
      ConvertMlirModuleToExportedModel(*module_op, /*checkpoint_dir=*/"",
                                       /*function_aliases=*/{},
                                       /*asset_file_defs=*/{});

  ASSERT_THAT(exported_model, IsOk());
  EXPECT_THAT(exported_model->init_node_name(),
              StrEq("init_op_init_all_tables"));

  // Match the init node, which is a NoOp that has control dependency to
  // HashTableV2 initialization. Fetching this node in TF Session will
  // initialize the hash table.
  const auto init_node_itr = llvm::find_if(
      exported_model->graph_def().node(), [](const NodeDef& node) {
        return node.name() == "init_op_init_all_tables";
      });
  ASSERT_NE(init_node_itr, exported_model->graph_def().node().end());
  EXPECT_THAT(init_node_itr->op(), StrEq("NoOp"));
  ASSERT_THAT(init_node_itr->input(), SizeIs(1));
  // "^" means control input.
  EXPECT_THAT(init_node_itr->input()[0],
              StrEq("^tf.InitializeTableFromTextFileV2"));
}

TEST_F(ConvertMlirModuleToExportedModelTest, InitNodeNotSetIfLocNameMismatch) {
  // Define a module that initializes a tf.HashTableV2 whose control output node
  // for the initialization is named "init_ok". Since the output control node
  // name does not begin with "init_op" the init node could not have been found
  // after the conversion.
  mlir::OwningOpRef<mlir::ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() <{initializers = []}> : () -> ()
      "tf_saved_model.asset"() <{filename = "assets/vocab_file.txt", sym_name = "__tf_saved_model_asset0_vocab_file.txt"}> : () -> ()
      func.func @main(%arg1: tensor<!tf_type.string> {tf_saved_model.index_path = ["arg_0:0"]}) -> (tensor<1x2xf32> {tf_saved_model.index_path = ["output:0"]}) attributes {tf.entry_function = {inputs = "arg_0:0", outputs = "output:0"}, tf_saved_model.exported_names = ["main"]} {
        %0 = tf_executor.graph {
          %output_0, %control_0 = tf_executor.island wraps "tf.Const"() <{value = dense<1.0> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
          %output_1, %control_1 = tf_executor.island wraps "tf.HashTableV2"() <{container = "", key_dtype = !tf_type.string, shared_name = "vocab_file.txt", use_node_name_sharing = false, value_dtype = i64}> {device = ""} : () -> tensor<!tf_type.resource>
          %control_2 = tf_executor.island wraps "tf.InitializeTableFromTextFileV2"(%output_1, %arg1) <{delimiter = "\09", key_index = -2 : i64, value_index = -1 : i64, vocab_size = -1 : i64}> {_has_manual_control_dependencies = true, device = ""} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
          // Location of this control output op becomes the name of the init_op.
          %control_3 = tf_executor.island(%control_2) wraps "tf.NoOp"() : () -> () loc("init_ok")
          tf_executor.fetch %output_0, %control_3 : tensor<1x2xf32>, !tf_executor.control
        }
        return %0 : tensor<1x2xf32>
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  const absl::StatusOr<ExportedModel> exported_model =
      ConvertMlirModuleToExportedModel(*module_op, /*checkpoint_dir=*/"",
                                       /*function_aliases=*/{},
                                       /*asset_file_defs=*/{});

  ASSERT_THAT(exported_model, IsOk());
  EXPECT_THAT(exported_model->init_node_name(), IsEmpty());
}

TEST_F(ConvertMlirModuleToExportedModelTest,
       ConversionFailureWhenNoMainFunction) {
  // Define a module a function whose name is not @main.
  mlir::OwningOpRef<mlir::ModuleOp> module_op = ParseModuleOpString(R"mlir(
    module attributes {tf_saved_model.semantics} {
      func.func @not_main() -> () attributes {tf_saved_model.exported_names = ["not_main"]} {
        tf_executor.graph {
          tf_executor.fetch
        }
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module_op);

  const absl::StatusOr<ExportedModel> exported_model =
      ConvertMlirModuleToExportedModel(*module_op, "my_checkpoint_dir",
                                       /*function_aliases=*/{},
                                       /*asset_file_defs=*/{});
  EXPECT_THAT(exported_model,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("entry function `main` must be present")));
}

}  // namespace
}  // namespace mlir::tf_quant::stablehlo
