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

#include "tensorflow/contrib/session_bundle/session_bundle.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/contrib/session_bundle/test_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
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

// Constants for the export path and file-names.
const char kExportPath[] = "session_bundle/testdata/half_plus_two/00000123";
const char kExportCheckpointV2Path[] =
    "session_bundle/testdata/half_plus_two_ckpt_v2/00000123";
const char kMetaGraphDefFilename[] = "export.meta";
const char kVariablesFilename[] = "export-00000-of-00001";

// Function used to rewrite a MetaGraphDef.
using MetaGraphDefTwiddler = std::function<void(MetaGraphDef*)>;

// Copy the base half_plus_two to `export_path`.
// Outputs the files using the passed names (typically the constants above).
// The Twiddler can be used to update the MetaGraphDef before output.
Status CopyExport(const string& export_path, const string& variables_filename,
                  const string& meta_graph_def_filename,
                  const MetaGraphDefTwiddler& twiddler) {
  TF_RETURN_IF_ERROR(Env::Default()->CreateDir(export_path));
  const string orig_path = test_util::TestSrcDirPath(kExportPath);
  {
    const string source = io::JoinPath(orig_path, kVariablesFilename);
    const string sink = io::JoinPath(export_path, variables_filename);

    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(Env::Default(), source, &data));
    TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), sink, data));
  }
  {
    const string source = io::JoinPath(orig_path, kMetaGraphDefFilename);
    const string sink = io::JoinPath(export_path, meta_graph_def_filename);

    MetaGraphDef graph_def;
    TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), source, &graph_def));
    twiddler(&graph_def);
    TF_RETURN_IF_ERROR(
        WriteStringToFile(Env::Default(), sink, graph_def.SerializeAsString()));
  }
  return Status::OK();
}

string MakeSerializedExample(float x) {
  tensorflow::Example example;
  auto* feature_map = example.mutable_features()->mutable_feature();
  (*feature_map)["x"].mutable_float_list()->add_value(x);
  return example.SerializeAsString();
}

void CheckRegressionSignature(const Signatures& signatures,
                              const SessionBundle& bundle) {
  // Recover the Tensor names of our inputs and outputs.
  ASSERT_TRUE(signatures.default_signature().has_regression_signature());
  const RegressionSignature regression_signature =
      signatures.default_signature().regression_signature();

  const string input_name = regression_signature.input().tensor_name();
  const string output_name = regression_signature.output().tensor_name();

  // Validate the half plus two behavior.
  std::vector<string> serialized_examples;
  for (float x : {0, 1, 2, 3}) {
    serialized_examples.push_back(MakeSerializedExample(x));
  }
  Tensor input = test::AsTensor<string>(serialized_examples, TensorShape({4}));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(
      bundle.session->Run({{input_name, input}}, {output_name}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  test::ExpectTensorEqual<float>(
      outputs[0], test::AsTensor<float>({2, 2.5, 3, 3.5}, TensorShape({4, 1})));
}

void CheckNamedSignatures(const Signatures& signatures,
                          const SessionBundle& bundle) {
  // Recover the Tensor names of our inputs and outputs.
  const string input_name = signatures.named_signatures()
                                .at("inputs")
                                .generic_signature()
                                .map()
                                .at("x")
                                .tensor_name();
  const string output_name = signatures.named_signatures()
                                 .at("outputs")
                                 .generic_signature()
                                 .map()
                                 .at("y")
                                 .tensor_name();

  // Validate the half plus two behavior.
  Tensor input = test::AsTensor<float>({0, 1, 2, 3}, TensorShape({4, 1}));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(
      bundle.session->Run({{input_name, input}}, {output_name}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  test::ExpectTensorEqual<float>(
      outputs[0], test::AsTensor<float>({2, 2.5, 3, 3.5}, TensorShape({4, 1})));
}

void CheckSessionBundle(const string& export_path,
                        const SessionBundle& bundle) {
  const string asset_path = io::JoinPath(export_path, kAssetsDirectory);
  // Validate the assets behavior.
  std::vector<Tensor> path_outputs;
  TF_ASSERT_OK(bundle.session->Run({}, {"filename1:0", "filename2:0"}, {},
                                   &path_outputs));
  ASSERT_EQ(2, path_outputs.size());
  // Validate the two asset file tensors are set by the init_op and include the
  // base_path and asset directory.
  test::ExpectTensorEqual<string>(
      test::AsTensor<string>({io::JoinPath(asset_path, "hello1.txt")},
                             TensorShape({})),
      path_outputs[0]);
  test::ExpectTensorEqual<string>(
      test::AsTensor<string>({io::JoinPath(asset_path, "hello2.txt")},
                             TensorShape({})),
      path_outputs[1]);

  Signatures signatures;
  TF_ASSERT_OK(GetSignatures(bundle.meta_graph_def, &signatures));
  CheckRegressionSignature(signatures, bundle);
  CheckNamedSignatures(signatures, bundle);
}

void BasicTest(const string& export_path) {
  SessionOptions options;
  SessionBundle bundle;
  TF_ASSERT_OK(LoadSessionBundleFromPath(options, export_path, &bundle));
  CheckSessionBundle(export_path, bundle);
}

// Test for resource leaks when loading and unloading large numbers of
// SessionBundles. Concurrent with adding this test, we had a leak where the
// TensorFlow Session was not being closed, which leaked memory.
// TODO(b/31711147): Increase the SessionBundle ResourceLeakTest iterations and
// move outside of the test suite.
TEST(LoadSessionBundleFromPath, ResourceLeakTest) {
  const string export_path = test_util::TestSrcDirPath(kExportPath);
  for (int i = 0; i < 100; i++) {
    BasicTest(export_path);
  }
}

TEST(LoadSessionBundleFromPath, BasicTensorFlowContrib) {
  const string export_path = test_util::TestSrcDirPath(kExportPath);
  BasicTest(export_path);
}

TEST(LoadSessionBundleFromPath, BasicTestRunOptions) {
  const string export_path = test_util::TestSrcDirPath(kExportPath);

  // Use default session-options.
  SessionOptions session_options;

  // Setup run-options with full-traces.
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);

  SessionBundle bundle;
  TF_ASSERT_OK(LoadSessionBundleFromPathUsingRunOptions(
      session_options, run_options, export_path, &bundle));
  CheckSessionBundle(export_path, bundle);
}

TEST(LoadSessionBundleFromPath, BasicTestRunOptionsThreadPool) {
  const string export_path = test_util::TestSrcDirPath(kExportPath);
  const int32 threadpool_index = 1;

  // Setup session-options with separate thread-pools.
  SessionOptions session_options;
  session_options.config.add_session_inter_op_thread_pool();
  session_options.config.add_session_inter_op_thread_pool()->set_num_threads(2);

  // Setup run-options with the threadpool index to use.
  RunOptions run_options;
  run_options.set_inter_op_thread_pool(threadpool_index);

  SessionBundle bundle;
  TF_ASSERT_OK(LoadSessionBundleFromPathUsingRunOptions(
      session_options, run_options, export_path, &bundle));
  CheckSessionBundle(export_path, bundle);
}

TEST(LoadSessionBundleFromPath, BasicTestRunOptionsThreadPoolInvalid) {
  const string export_path = test_util::TestSrcDirPath(kExportPath);
  const int32 invalid_threadpool_index = 2;

  // Setup session-options with separate thread-pools.
  SessionOptions session_options;
  session_options.config.add_session_inter_op_thread_pool();
  session_options.config.add_session_inter_op_thread_pool()->set_num_threads(2);

  // Setup run-options with an invalid threadpool index.
  RunOptions run_options;
  run_options.set_inter_op_thread_pool(invalid_threadpool_index);

  SessionBundle bundle;
  Status status = LoadSessionBundleFromPathUsingRunOptions(
      session_options, run_options, export_path, &bundle);

  // Expect failed session run calls with invalid run-options.
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(StringPiece(status.error_message())
                  .contains("Invalid inter_op_thread_pool: 2"))
      << status.error_message();
}

TEST(LoadSessionBundleFromPath, BadExportPath) {
  const string export_path = test_util::TestSrcDirPath("/tmp/bigfoot");
  SessionOptions options;
  options.target = "local";
  SessionBundle bundle;
  const auto status = LoadSessionBundleFromPath(options, export_path, &bundle);
  ASSERT_FALSE(status.ok());
  const string msg = status.ToString();
  EXPECT_TRUE(msg.find("Not found") != std::string::npos) << msg;
}

TEST(CheckpointV2Test, LoadSessionBundleFromPath) {
  const string export_path = test_util::TestSrcDirPath(kExportCheckpointV2Path);
  BasicTest(export_path);
}

TEST(CheckpointV2Test, IsPossibleExportDirectory) {
  const string export_path = test_util::TestSrcDirPath(kExportCheckpointV2Path);
  EXPECT_TRUE(IsPossibleExportDirectory(export_path));
}

class SessionBundleTest : public ::testing::Test {
 protected:
  // Copy the half_plus_two graph and apply the twiddler to rewrite the
  // MetaGraphDef.
  // Returns the path of the export.
  // ** Should only be called once per test **
  string SetupExport(const MetaGraphDefTwiddler& twiddler) {
    return SetupExport(twiddler, kVariablesFilename, kMetaGraphDefFilename);
  }
  // SetupExport that allows for the variables and meta_graph_def filenames
  // to be overridden.
  string SetupExport(const MetaGraphDefTwiddler& twiddler,
                     const string& variables_filename,
                     const string& meta_graph_def_filename) {
    // Construct a unique path name based on the test name.
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const string export_path = io::JoinPath(
        testing::TmpDir(),
        strings::StrCat(test_info->test_case_name(), test_info->name()));
    TF_CHECK_OK(CopyExport(export_path, variables_filename,
                           meta_graph_def_filename, twiddler));
    return export_path;
  }

  SessionOptions options_;
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

TEST_F(SessionBundleTest, ServingGraphEmpty) {
  const string path = SetupExport([](MetaGraphDef* def) {
    (*def->mutable_collection_def())[kGraphKey].clear_any_list();
  });
  status_ = LoadSessionBundleFromPath(options_, path, &bundle_);
  EXPECT_FALSE(status_.ok());
  EXPECT_TRUE(StringPiece(status_.error_message())
                  .contains("Expected exactly one serving GraphDef"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, ServingGraphAnyIncorrectType) {
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

TEST_F(SessionBundleTest, ServingGraphAnyValueCorrupted) {
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

TEST_F(SessionBundleTest, AssetFileAnyIncorrectType) {
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
          .contains("Expected Any type_url for: tensorflow.serving.AssetFile"))
      << status_.error_message();
}

TEST_F(SessionBundleTest, AssetFileAnyValueCorrupted) {
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

TEST_F(SessionBundleTest, InitOpTooManyValues) {
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
