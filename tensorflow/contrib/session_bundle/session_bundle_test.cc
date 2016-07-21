/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/contrib/session_bundle/session_bundle.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/contrib/session_bundle/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace serving {
namespace {

// Constants for the export file names.
const char kVariablesFilename[] = "export-00000-of-00001";
const char kMetaGraphDefFilename[] = "export.meta";

// Function used to rewrite a MetaGraphDef.
using MetaGraphDefTwiddler = std::function<void(MetaGraphDef*)>;

// Copy the base half_plus_two to `export_path`.
// Outputs the files using the passed names (typically the constants above).
// The Twiddler can be used to update the MetaGraphDef before output.
Status CopyExport(const string& export_path, const string& variables_filename,
                  const string& meta_graph_def_filename,
                  const MetaGraphDefTwiddler& twiddler) {
  TF_RETURN_IF_ERROR(Env::Default()->CreateDir(export_path));
  const string orig_path = test_util::TestSrcDirPath(
      "session_bundle/example/half_plus_two/00000123");
  {
    const string source =
        tensorflow::io::JoinPath(orig_path, kVariablesFilename);
    const string sink =
        tensorflow::io::JoinPath(export_path, variables_filename);

    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(Env::Default(), source, &data));
    TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), sink, data));
  }
  {
    const string source =
        tensorflow::io::JoinPath(orig_path, kMetaGraphDefFilename);
    const string sink =
        tensorflow::io::JoinPath(export_path, meta_graph_def_filename);

    tensorflow::MetaGraphDef graph_def;
    TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), source, &graph_def));
    twiddler(&graph_def);
    TF_RETURN_IF_ERROR(
        WriteStringToFile(Env::Default(), sink, graph_def.SerializeAsString()));
  }
  return Status::OK();
}

void BasicTest(const string& export_path) {
  tensorflow::SessionOptions options;
  SessionBundle bundle;
  TF_ASSERT_OK(LoadSessionBundleFromPath(options, export_path, &bundle));

  const string asset_path =
      tensorflow::io::JoinPath(export_path, kAssetsDirectory);
  // Validate the assets behavior.
  std::vector<Tensor> path_outputs;
  TF_ASSERT_OK(bundle.session->Run({}, {"filename1:0", "filename2:0"}, {},
                                   &path_outputs));
  ASSERT_EQ(2, path_outputs.size());
  // Validate the two asset file tensors are set by the init_op and include the
  // base_path and asset directory.
  test::ExpectTensorEqual<string>(
      test::AsTensor<string>(
          {tensorflow::io::JoinPath(asset_path, "hello1.txt")},
          TensorShape({})),
      path_outputs[0]);
  test::ExpectTensorEqual<string>(
      test::AsTensor<string>(
          {tensorflow::io::JoinPath(asset_path, "hello2.txt")},
          TensorShape({})),
      path_outputs[1]);

  // Validate the half plus two behavior.
  Tensor input = test::AsTensor<float>({0, 1, 2, 3}, TensorShape({4, 1}));

  // Recover the Tensor names of our inputs and outputs.
  Signatures signatures;
  TF_ASSERT_OK(GetSignatures(bundle.meta_graph_def, &signatures));
  ASSERT_TRUE(signatures.default_signature().has_regression_signature());
  const tensorflow::serving::RegressionSignature regression_signature =
      signatures.default_signature().regression_signature();

  const string input_name = regression_signature.input().tensor_name();
  const string output_name = regression_signature.output().tensor_name();

  std::vector<Tensor> outputs;
  TF_ASSERT_OK(
      bundle.session->Run({{input_name, input}}, {output_name}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  test::ExpectTensorEqual<float>(
      outputs[0], test::AsTensor<float>({2, 2.5, 3, 3.5}, TensorShape({4, 1})));
}

TEST(LoadSessionBundleFromPath, BasicTensorFlowContrib) {
  const string export_path = test_util::TestSrcDirPath(
      "session_bundle/example/half_plus_two/00000123");
  BasicTest(export_path);
}

TEST(LoadSessionBundleFromPath, BadExportPath) {
  const string export_path = test_util::TestSrcDirPath("/tmp/bigfoot");
  tensorflow::SessionOptions options;
  options.target = "local";
  SessionBundle bundle;
  const auto status = LoadSessionBundleFromPath(options, export_path, &bundle);
  ASSERT_FALSE(status.ok());
  const string msg = status.ToString();
  EXPECT_TRUE(msg.find("Not found") != std::string::npos) << msg;
}

class SessionBundleTest : public ::testing::Test {
 protected:
  // Copy the half_plus_two graph and apply the twiddler to rewrite the
  // MetaGraphDef.
  // Returns the path of the export.
  // ** Should only be called once per test **
  string SetupExport(MetaGraphDefTwiddler twiddler) {
    return SetupExport(twiddler, kVariablesFilename, kMetaGraphDefFilename);
  }
  // SetupExport that allows for the variables and meta_graph_def filenames
  // to be overridden.
  string SetupExport(MetaGraphDefTwiddler twiddler,
                     const string& variables_filename,
                     const string& meta_graph_def_filename) {
    // Construct a unique path name based on the test name.
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const string export_path = tensorflow::io::JoinPath(
        testing::TmpDir(),
        strings::StrCat(test_info->test_case_name(), test_info->name()));
    TF_CHECK_OK(CopyExport(export_path, variables_filename,
                           meta_graph_def_filename, twiddler));
    return export_path;
  }

  tensorflow::SessionOptions options_;
  SessionBundle bundle_;
  Status status_;
};

TEST_F(SessionBundleTest, Basic) {
  const string export_path = SetupExport([](MetaGraphDef*) {});
  BasicTest(export_path);
}

TEST_F(SessionBundleTest, UnshardedVariableFile) {
  // Test that we can properly read the variables when exported
  // without sharding.
  const string export_path =
      SetupExport([](MetaGraphDef*) {}, "export", kMetaGraphDefFilename);
  BasicTest(export_path);
}

TEST_F(SessionBundleTest, ServingGraph_Empty) {
  const string path = SetupExport([](MetaGraphDef* def) {
    (*def->mutable_collection_def())[kGraphKey].clear_any_list();
  });
  status_ = LoadSessionBundleFromPath(options_, path, &bundle_);
  EXPECT_FALSE(status_.ok());
  EXPECT_TRUE(StringPiece(status_.error_message())
                  .contains("Expected exactly one serving GraphDef"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, ServingGraphAny_IncorrectType) {
  const string path = SetupExport([](MetaGraphDef* def) {
    // Pack an unexpected type in the GraphDef Any.
    (*def->mutable_collection_def())[kGraphKey].clear_any_list();
    auto* any = (*def->mutable_collection_def())[kGraphKey]
                    .mutable_any_list()
                    ->add_value();
    any->PackFrom(AssetFile());
  });
  status_ = LoadSessionBundleFromPath(options_, path, &bundle_);
  EXPECT_FALSE(status_.ok());
  EXPECT_TRUE(StringPiece(status_.error_message())
                  .contains("Expected Any type_url for: tensorflow.GraphDef"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, ServingGraphAnyValue_Corrupted) {
  const string path = SetupExport([](MetaGraphDef* def) {
    // Pack an unexpected type in the GraphDef Any.
    (*def->mutable_collection_def())[kGraphKey].clear_any_list();
    auto* any = (*def->mutable_collection_def())[kGraphKey]
                    .mutable_any_list()
                    ->add_value();
    any->PackFrom(GraphDef());
    any->set_value("junk junk");
  });
  status_ = LoadSessionBundleFromPath(options_, path, &bundle_);
  EXPECT_FALSE(status_.ok());
  EXPECT_TRUE(StringPiece(status_.error_message()).contains("Failed to unpack"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, AssetFileAny_IncorrectType) {
  const string path = SetupExport([](MetaGraphDef* def) {
    // Pack an unexpected type in the AssetFile Any.
    (*def->mutable_collection_def())[kAssetsKey].clear_any_list();
    auto* any = (*def->mutable_collection_def())[kAssetsKey]
                    .mutable_any_list()
                    ->add_value();
    any->PackFrom(GraphDef());
  });
  status_ = LoadSessionBundleFromPath(options_, path, &bundle_);
  EXPECT_FALSE(status_.ok());
  EXPECT_TRUE(
      StringPiece(status_.error_message())
          .contains(
              "Expected asset Any type_url for: tensorflow.serving.AssetFile"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, AssetFileAny_ValueCorrupted) {
  const string path = SetupExport([](MetaGraphDef* def) {
    // Pack an unexpected type in the AssetFile Any.
    (*def->mutable_collection_def())[kAssetsKey].clear_any_list();
    auto* any = (*def->mutable_collection_def())[kAssetsKey]
                    .mutable_any_list()
                    ->add_value();
    any->PackFrom(AssetFile());
    any->set_value("junk junk");
  });
  status_ = LoadSessionBundleFromPath(options_, path, &bundle_);
  EXPECT_FALSE(status_.ok());
  EXPECT_TRUE(StringPiece(status_.error_message()).contains("Failed to unpack"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, InitOp_TooManyValues) {
  const string path = SetupExport([](MetaGraphDef* def) {
    // Pack multiple init ops in to the collection.
    (*def->mutable_collection_def())[kInitOpKey].clear_node_list();
    auto* node_list =
        (*def->mutable_collection_def())[kInitOpKey].mutable_node_list();
    node_list->add_value("foo");
    node_list->add_value("bar");
  });
  status_ = LoadSessionBundleFromPath(options_, path, &bundle_);
  EXPECT_FALSE(status_.ok());
  EXPECT_TRUE(StringPiece(status_.error_message())
                  .contains("Expected exactly one serving init op"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, PossibleExportDirectory) {
  const string export_path = SetupExport([](MetaGraphDef*) {});
  EXPECT_TRUE(IsPossibleExportDirectory(export_path));

  EXPECT_FALSE(
      IsPossibleExportDirectory(io::JoinPath(export_path, kAssetsDirectory)));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
