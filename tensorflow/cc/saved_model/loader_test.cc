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

#include "tensorflow/cc/saved_model/loader.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kTestDataPb[] = "cc/saved_model/testdata/half_plus_two";
constexpr char kTestDataPbTxt[] = "cc/saved_model/testdata/half_plus_two_pbtxt";
constexpr char kTestDataSharded[] =
    "cc/saved_model/testdata/half_plus_two_sharded";

class LoaderTest : public ::testing::Test {
 protected:
  LoaderTest() {}

  void CheckSavedModelBundle(const SavedModelBundle& bundle) {
    // Validate the half plus two behavior.
    Tensor input = test::AsTensor<float>({0, 1, 2, 3}, TensorShape({4, 1}));

    // Retrieve the regression signature from meta graph def.
    const auto signature_def_map = bundle.meta_graph_def.signature_def();
    const auto signature_def = signature_def_map.at("regression");

    const string input_name = signature_def.inputs().at("input").name();
    const string output_name = signature_def.outputs().at("output").name();

    std::vector<Tensor> outputs;
    TF_ASSERT_OK(bundle.session->Run({{input_name, input}}, {output_name}, {},
                                     &outputs));
    ASSERT_EQ(outputs.size(), 1);
    test::ExpectTensorEqual<float>(
        outputs[0],
        test::AsTensor<float>({2, 2.5, 3, 3.5}, TensorShape({4, 1})));
  }
};

TEST_F(LoaderTest, TagMatch) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPb);
  TF_ASSERT_OK(LoadSavedModel(export_dir, {kSavedModelTagServe},
                              session_options, run_options, &bundle));
  CheckSavedModelBundle(bundle);
}

TEST_F(LoaderTest, NoTagMatch) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPb);
  Status st = LoadSavedModel(export_dir, {"missing-tag"}, session_options,
                             run_options, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(
      StringPiece(st.error_message())
          .contains("Could not find meta graph def matching supplied tags."))
      << st.error_message();
}

TEST_F(LoaderTest, NoTagMatchMultiple) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPb);
  Status st = LoadSavedModel(export_dir, {kSavedModelTagServe, "missing-tag"},
                             session_options, run_options, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(
      StringPiece(st.error_message())
          .contains("Could not find meta graph def matching supplied tags."))
      << st.error_message();
}

TEST_F(LoaderTest, PbtxtFormat) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPbTxt);
  TF_ASSERT_OK(LoadSavedModel(export_dir, {kSavedModelTagServe},
                              session_options, run_options, &bundle));
  CheckSavedModelBundle(bundle);
}

TEST_F(LoaderTest, ShardedVariables) {
  SavedModelBundle bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  TF_ASSERT_OK(LoadSavedModel(export_dir, {kSavedModelTagServe},
                              session_options, run_options, &bundle));
  CheckSavedModelBundle(bundle);
}

TEST_F(LoaderTest, InvalidExportPath) {
  SavedModelBundle bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "missing-path");
  Status st = LoadSavedModel(export_dir, {kSavedModelTagServe}, session_options,
                             run_options, &bundle);
  EXPECT_FALSE(st.ok());
}

TEST_F(LoaderTest, MaybeSavedModelDirectory) {
  // Valid SavedModel directory.
  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPb);
  EXPECT_TRUE(MaybeSavedModelDirectory(export_dir));

  // Directory that does not exist.
  const string missing_export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "missing-path");
  EXPECT_FALSE(MaybeSavedModelDirectory(missing_export_dir));

  // Directory that exists but is an invalid SavedModel location.
  const string invalid_export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata");
  EXPECT_FALSE(MaybeSavedModelDirectory(invalid_export_dir));
}

}  // namespace
}  // namespace tensorflow
