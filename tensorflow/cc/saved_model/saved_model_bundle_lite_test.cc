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

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace {

constexpr char kTestDataPbTxt[] =
    "cc/saved_model/testdata/half_plus_two_pbtxt/00000123";
constexpr char kTestDataMainOp[] =
    "cc/saved_model/testdata/half_plus_two_main_op/00000123";
constexpr char kTestDataSharded[] =
    "cc/saved_model/testdata/half_plus_two/00000123";
constexpr char kTestDataInitOpV2[] =
    "cc/saved_model/testdata/half_plus_two_v2/00000123";

class LoaderTest : public ::testing::Test {
 protected:
  LoaderTest() {}

  string MakeSerializedExample(float x) {
    tensorflow::Example example;
    auto* feature_map = example.mutable_features()->mutable_feature();
    (*feature_map)["x"].mutable_float_list()->add_value(x);
    return example.SerializeAsString();
  }

  void ValidateAssets(const string& export_dir,
                      const SavedModelBundleLite& bundle) {
    const string asset_directory =
        io::JoinPath(export_dir, kSavedModelAssetsDirectory);
    const string asset_filename = "foo.txt";
    const string asset_filepath = io::JoinPath(asset_directory, asset_filename);
    TF_EXPECT_OK(Env::Default()->FileExists(asset_filepath));

    std::vector<Tensor> path_outputs;
    TF_ASSERT_OK(
        bundle.GetSession()->Run({}, {"filename_tensor:0"}, {}, &path_outputs));
    ASSERT_EQ(1, path_outputs.size());

    test::ExpectTensorEqual<tstring>(
        test::AsTensor<tstring>({"foo.txt"}, TensorShape({})), path_outputs[0]);
  }

  void CheckSavedModelBundle(const string& export_dir,
                             const SavedModelBundleLite& bundle) {
    ValidateAssets(export_dir, bundle);
    // Retrieve the regression signature from the bundle.
    const auto& signature_def = bundle.GetSignatures().at("regress_x_to_y");

    const string input_name = signature_def.inputs().at(kRegressInputs).name();
    const string output_name =
        signature_def.outputs().at(kRegressOutputs).name();

    std::vector<tstring> serialized_examples;
    for (float x : {0, 1, 2, 3}) {
      serialized_examples.push_back(MakeSerializedExample(x));
    }

    // Validate the half plus two behavior.
    Tensor input =
        test::AsTensor<tstring>(serialized_examples, TensorShape({4}));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(bundle.GetSession()->Run({{input_name, input}}, {output_name},
                                          {}, &outputs));
    ASSERT_EQ(outputs.size(), 1);
    test::ExpectTensorEqual<float>(
        outputs[0],
        test::AsTensor<float>({2, 2.5, 3, 3.5}, TensorShape({4, 1})));

    // Validate the `output_partition_graphs` is not supported.
    RunOptions run_options;
    run_options.set_output_partition_graphs(true);
    RunMetadata run_metadata;
    Status s =
        bundle.GetSession()->Run(run_options, {{input_name, input}},
                                 {output_name}, {}, &outputs, &run_metadata);
    ASSERT_TRUE(errors::IsInvalidArgument(s));
  }
};

// Test for resource leaks related to TensorFlow session closing requirements
// when loading and unloading large numbers of SavedModelBundles.
// TODO(sukritiramesh): Increase run iterations and move outside of the test
// suite.
TEST_F(LoaderTest, ResourceLeakTest) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  for (int i = 0; i < 100; ++i) {
    TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                                {kSavedModelTagServe}, &bundle));
    CheckSavedModelBundle(export_dir, bundle);
  }
}

TEST_F(LoaderTest, ExtendFailsTest) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  Status s = bundle.GetSession()->Extend({});
  ASSERT_TRUE(errors::IsUnimplemented(s));
}

TEST_F(LoaderTest, TagMatch) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

TEST_F(LoaderTest, NoTagMatch) {
  SavedModelBundleLite bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {"missing-tag"}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.error_message(),
      "Could not find meta graph def matching supplied tags: { missing-tag }"))
      << st.error_message();
}

TEST_F(LoaderTest, NoTagMatchMultiple) {
  SavedModelBundleLite bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe, "missing-tag"}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.error_message(),
      "Could not find meta graph def matching supplied tags: "))
      << st.error_message();
}

TEST_F(LoaderTest, SessionCreationFailure) {
  SavedModelBundleLite bundle;
  // Use invalid SessionOptions to cause session creation to fail.  Default
  // options work, so provide an invalid value for the target field.
  SessionOptions session_options;
  constexpr char kInvalidTarget[] = "invalid target";
  session_options.target = kInvalidTarget;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(st.error_message(), kInvalidTarget))
      << st.error_message();
}

TEST_F(LoaderTest, PbtxtFormat) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataPbTxt);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

TEST_F(LoaderTest, MainOpFormat) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataMainOp);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

TEST_F(LoaderTest, InvalidExportPath) {
  SavedModelBundleLite bundle;
  RunOptions run_options;
  SessionOptions session_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "missing-path");
  Status st = LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle);
  EXPECT_FALSE(st.ok());
}

TEST_F(LoaderTest, MaybeSavedModelDirectory) {
  // Valid SavedModel directory.
  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  EXPECT_TRUE(MaybeSavedModelDirectory(export_dir));

  // Directory that does not exist.
  const string missing_export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "missing-path");
  EXPECT_FALSE(MaybeSavedModelDirectory(missing_export_dir));

  // Directory that exists but is an invalid SavedModel location.
  const string invalid_export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model");
  EXPECT_FALSE(MaybeSavedModelDirectory(invalid_export_dir));
}

TEST_F(LoaderTest, SavedModelInitOpV2Format) {
  SavedModelBundleLite bundle;
  SessionOptions session_options;
  RunOptions run_options;

  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataInitOpV2);
  TF_ASSERT_OK(LoadSavedModel(session_options, run_options, export_dir,
                              {kSavedModelTagServe}, &bundle));
  CheckSavedModelBundle(export_dir, bundle);
}

}  // namespace
}  // namespace tensorflow
