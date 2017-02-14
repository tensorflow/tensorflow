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

#include "tensorflow/contrib/session_bundle/bundle_shim.h"

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/contrib/session_bundle/test_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace serving {
namespace internal {
namespace {

constexpr char kSessionBundlePath[] =
    "session_bundle/testdata/half_plus_two/00000123";
constexpr char kSessionBundleMetaGraphFilename[] = "export.meta";
constexpr char kSessionBundleVariablesFilename[] = "export-00000-of-00001";
constexpr char kSavedModelBundlePath[] =
    "cc/saved_model/testdata/half_plus_two/00000123";

string MakeSerializedExample(float x) {
  tensorflow::Example example;
  auto* feature_map = example.mutable_features()->mutable_feature();
  (*feature_map)["x"].mutable_float_list()->add_value(x);
  return example.SerializeAsString();
}

void ValidateHalfPlusTwo(const SavedModelBundle& saved_model_bundle,
                         const string& input_tensor_name,
                         const string& output_tensor_name) {
  // Validate the half plus two behavior.
  std::vector<string> serialized_examples;
  for (float x : {0, 1, 2, 3}) {
    serialized_examples.push_back(MakeSerializedExample(x));
  }
  Tensor input = test::AsTensor<string>(serialized_examples, TensorShape({4}));

  std::vector<Tensor> outputs;
  TF_ASSERT_OK(saved_model_bundle.session->Run(
      {{input_tensor_name, input}}, {output_tensor_name}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  test::ExpectTensorEqual<float>(
      outputs[0], test::AsTensor<float>({2, 2.5, 3, 3.5}, TensorShape({4, 1})));
}

void LoadAndValidateSavedModelBundle(const string& export_dir,
                                     const std::unordered_set<string>& tags,
                                     const string& signature_def_key) {
  SessionOptions session_options;
  RunOptions run_options;
  SavedModelBundle saved_model_bundle;
  TF_ASSERT_OK(LoadSessionBundleOrSavedModelBundle(
      session_options, run_options, export_dir, tags, &saved_model_bundle));
  const MetaGraphDef meta_graph_def = saved_model_bundle.meta_graph_def;
  const auto& signature_def_map = meta_graph_def.signature_def();

  const auto& regression_entry = signature_def_map.find(signature_def_key);
  ASSERT_FALSE(regression_entry == signature_def_map.end());
  SignatureDef regression_signature_def = regression_entry->second;

  EXPECT_EQ(1, regression_signature_def.inputs_size());
  ASSERT_FALSE(regression_signature_def.inputs().find(kRegressInputs) ==
               regression_signature_def.inputs().end());
  TensorInfo input_tensor_info =
      regression_signature_def.inputs().find(kRegressInputs)->second;
  EXPECT_EQ(1, regression_signature_def.outputs_size());
  // Ensure the TensorInfo has dtype populated.
  EXPECT_EQ(DT_STRING, input_tensor_info.dtype());

  ASSERT_FALSE(regression_signature_def.outputs().find(kRegressOutputs) ==
               regression_signature_def.outputs().end());
  TensorInfo output_tensor_info =
      regression_signature_def.outputs().find(kRegressOutputs)->second;
  // Ensure the TensorInfo has dtype populated.
  EXPECT_EQ(DT_FLOAT, output_tensor_info.dtype());
  ValidateHalfPlusTwo(saved_model_bundle, input_tensor_info.name(),
                      output_tensor_info.name());
}

// Helper function to validate that the SignatureDef found in the MetaGraphDef
// with the provided key has the expected string representation.
void ValidateSignatureDef(const MetaGraphDef& meta_graph_def, const string& key,
                          const string& expected_string_signature_def) {
  tensorflow::SignatureDef expected_signature;
  CHECK(protobuf::TextFormat::ParseFromString(expected_string_signature_def,
                                              &expected_signature));
  auto iter = meta_graph_def.signature_def().find(key);
  ASSERT_TRUE(iter != meta_graph_def.signature_def().end());
  EXPECT_EQ(expected_signature.DebugString(), iter->second.DebugString());
}

// Checks that the input map in a signature def is populated correctly.
TEST(BundleShimTest, AddInputToSignatureDef) {
  SignatureDef signature_def;
  const string tensor_name = "foo_tensor";
  const string map_key = "foo_key";

  // Build a map of tensor-name to dtype, for the unit-test.
  std::unordered_map<string, DataType> tensor_name_to_dtype;
  tensor_name_to_dtype[tensor_name] = tensorflow::DT_STRING;

  AddInputToSignatureDef(tensor_name, tensor_name_to_dtype, map_key,
                         &signature_def);
  EXPECT_EQ(1, signature_def.inputs_size());
  EXPECT_EQ(tensor_name, signature_def.inputs().find(map_key)->second.name());
}

// Checks that the output map in a signature def is populated correctly.
TEST(BundleShimTest, AddOutputToSignatureDef) {
  SignatureDef signature_def;
  const string tensor_name = "foo_tensor";
  const string map_key = "foo_key";

  // Build a map of tensor-name to dtype, for the unit-test.
  std::unordered_map<string, DataType> tensor_name_to_dtype;
  tensor_name_to_dtype[tensor_name] = tensorflow::DT_STRING;

  AddOutputToSignatureDef(tensor_name, tensor_name_to_dtype, map_key,
                          &signature_def);
  EXPECT_EQ(1, signature_def.outputs_size());
  EXPECT_EQ(tensor_name, signature_def.outputs().find(map_key)->second.name());
}

// Checks that no signature defs are added if the default signature is missing.
TEST(BundleShimTest, DefaultSignatureMissing) {
  MetaGraphDef meta_graph_def;
  // Signatures signatures;
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(0, meta_graph_def.signature_def_size());
}

// Checks that no signature defs are added if the default signature is empty.
TEST(BundleShimTest, DefaultSignatureEmpty) {
  Signatures signatures;
  signatures.mutable_default_signature();

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(0, meta_graph_def.signature_def_size());
}

// Checks the conversion to signature def for a regression default signature.
TEST(BundleShimTest, DefaultSignatureRegression) {
  Signatures signatures;
  RegressionSignature* regression_signature =
      signatures.mutable_default_signature()->mutable_regression_signature();
  regression_signature->mutable_input()->set_tensor_name("foo-input");
  regression_signature->mutable_output()->set_tensor_name("foo-output");
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(1, meta_graph_def.signature_def_size());
  const auto actual_signature_def =
      meta_graph_def.signature_def().find(kDefaultServingSignatureDefKey);
  EXPECT_EQ("foo-input", actual_signature_def->second.inputs()
                             .find(kRegressInputs)
                             ->second.name());
  EXPECT_EQ("foo-output", actual_signature_def->second.outputs()
                              .find(kRegressOutputs)
                              ->second.name());
  EXPECT_EQ(kRegressMethodName, actual_signature_def->second.method_name());
}

// Checks the conversion to signature def for a classification default
// signature.
TEST(BundleShimTest, DefaultSignatureClassification) {
  Signatures signatures;
  ClassificationSignature* classification_signature =
      signatures.mutable_default_signature()
          ->mutable_classification_signature();
  classification_signature->mutable_input()->set_tensor_name("foo-input");
  classification_signature->mutable_classes()->set_tensor_name("foo-classes");
  classification_signature->mutable_scores()->set_tensor_name("foo-scores");
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(1, meta_graph_def.signature_def_size());
  const auto actual_signature_def =
      meta_graph_def.signature_def().find(kDefaultServingSignatureDefKey);
  EXPECT_EQ("foo-input", actual_signature_def->second.inputs()
                             .find(kClassifyInputs)
                             ->second.name());
  EXPECT_EQ("foo-classes", actual_signature_def->second.outputs()
                               .find(kClassifyOutputClasses)
                               ->second.name());
  EXPECT_EQ("foo-scores", actual_signature_def->second.outputs()
                              .find(kClassifyOutputScores)
                              ->second.name());
  EXPECT_EQ(kClassifyMethodName, actual_signature_def->second.method_name());
}

// Checks that generic default signatures are not up converted.
TEST(BundleShimTest, DefaultSignatureGeneric) {
  TensorBinding input_binding;
  input_binding.set_tensor_name("foo-input");

  TensorBinding output_binding;
  output_binding.set_tensor_name("foo-output");

  Signatures signatures;
  GenericSignature* generic_signature =
      signatures.mutable_default_signature()->mutable_generic_signature();
  generic_signature->mutable_map()->insert({kPredictInputs, input_binding});
  generic_signature->mutable_map()->insert({kPredictOutputs, output_binding});

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(0, meta_graph_def.signature_def_size());
}

TEST(BundleShimTest, NamedRegressionSignatures) {
  Signatures signatures;

  RegressionSignature* foo_regression_signature =
      (*signatures.mutable_named_signatures())["foo"]
          .mutable_regression_signature();
  foo_regression_signature->mutable_input()->set_tensor_name("foo-input");
  foo_regression_signature->mutable_output()->set_tensor_name("foo-output");

  RegressionSignature* bar_regression_signature =
      (*signatures.mutable_named_signatures())["bar"]
          .mutable_regression_signature();
  bar_regression_signature->mutable_input()->set_tensor_name("bar-input");
  bar_regression_signature->mutable_output()->set_tensor_name("bar-output");

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  ASSERT_EQ(2, meta_graph_def.signature_def_size());

  ValidateSignatureDef(meta_graph_def, "foo",
                       "inputs { "
                       "  key: \"inputs\" "
                       "  value { "
                       "name: \"foo-input\" "
                       "  } "
                       "} "
                       "outputs { "
                       "  key: \"outputs\" "
                       "  value { "
                       "    name: \"foo-output\" "
                       "  } "
                       "} "
                       "method_name: \"tensorflow/serving/regress\" ");
  ValidateSignatureDef(meta_graph_def, "bar",
                       "inputs { "
                       "  key: \"inputs\" "
                       "  value { "
                       "name: \"bar-input\" "
                       "  } "
                       "} "
                       "outputs { "
                       "  key: \"outputs\" "
                       "  value { "
                       "    name: \"bar-output\" "
                       "  } "
                       "} "
                       "method_name: \"tensorflow/serving/regress\" ");
}

TEST(BundleShimTest, NamedClassificationSignatures) {
  Signatures signatures;

  ClassificationSignature* foo_classification_signature =
      (*signatures.mutable_named_signatures())["foo"]
          .mutable_classification_signature();
  foo_classification_signature->mutable_input()->set_tensor_name("foo-input");
  foo_classification_signature->mutable_classes()->set_tensor_name(
      "foo-classes");

  ClassificationSignature* bar_classification_signature =
      (*signatures.mutable_named_signatures())["bar"]
          .mutable_classification_signature();
  bar_classification_signature->mutable_input()->set_tensor_name("bar-input");
  bar_classification_signature->mutable_scores()->set_tensor_name("bar-scores");

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  ASSERT_EQ(2, meta_graph_def.signature_def_size());

  ValidateSignatureDef(meta_graph_def, "foo",
                       "inputs { "
                       "  key: \"inputs\" "
                       "  value { "
                       "name: \"foo-input\" "
                       "  } "
                       "} "
                       "outputs { "
                       "  key: \"classes\" "
                       "  value { "
                       "    name: \"foo-classes\" "
                       "  } "
                       "} "
                       "method_name: \"tensorflow/serving/classify\" ");
  ValidateSignatureDef(meta_graph_def, "bar",
                       "inputs { "
                       "  key: \"inputs\" "
                       "  value { "
                       "name: \"bar-input\" "
                       "  } "
                       "} "
                       "outputs { "
                       "  key: \"scores\" "
                       "  value { "
                       "    name: \"bar-scores\" "
                       "  } "
                       "} "
                       "method_name: \"tensorflow/serving/classify\" ");
}

// Checks the Predict SignatureDef created when the named signatures have
// `inputs` and `outputs`.
TEST(BundleShimTest, NamedSignatureGenericInputsAndOutputs) {
  TensorBinding input_binding;
  input_binding.set_tensor_name("foo-input");

  TensorBinding output_binding;
  output_binding.set_tensor_name("foo-output");

  Signatures signatures;
  GenericSignature* input_generic_signature =
      (*signatures.mutable_named_signatures())[kPredictInputs]
          .mutable_generic_signature();
  input_generic_signature->mutable_map()->insert({"foo-input", input_binding});

  GenericSignature* output_generic_signature =
      (*signatures.mutable_named_signatures())[kPredictOutputs]
          .mutable_generic_signature();
  output_generic_signature->mutable_map()->insert(
      {"foo-output", output_binding});

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(1, meta_graph_def.signature_def_size());
  const auto actual_signature_def =
      meta_graph_def.signature_def().find(kDefaultServingSignatureDefKey);
  ASSERT_FALSE(actual_signature_def == meta_graph_def.signature_def().end());
  ASSERT_FALSE(actual_signature_def->second.inputs().find("foo-input") ==
               actual_signature_def->second.inputs().end());
  EXPECT_EQ(
      "foo-input",
      actual_signature_def->second.inputs().find("foo-input")->second.name());
  ASSERT_FALSE(actual_signature_def->second.outputs().find("foo-output") ==
               actual_signature_def->second.outputs().end());
  EXPECT_EQ(
      "foo-output",
      actual_signature_def->second.outputs().find("foo-output")->second.name());
  EXPECT_EQ(kPredictMethodName, actual_signature_def->second.method_name());
}

// Checks that a signature def is not added if the named signatures is generic
// but does not have `inputs` and `outputs`.
TEST(BundleShimTest, NamedSignatureGenericNoInputsOrOutputs) {
  TensorBinding input_binding;
  input_binding.set_tensor_name("foo-input");

  TensorBinding output_binding;
  output_binding.set_tensor_name("foo-output");

  Signatures signatures;
  GenericSignature* generic_signature =
      (*signatures.mutable_named_signatures())["unknown"]
          .mutable_generic_signature();
  generic_signature->mutable_map()->insert({kPredictInputs, input_binding});
  generic_signature->mutable_map()->insert({kPredictOutputs, output_binding});

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(0, meta_graph_def.signature_def_size());
}

// Checks that a signature def is not added when the named signatures have only
// one of `inputs` and `outputs`.
TEST(BundleShimTest, NamedSignatureGenericOnlyInput) {
  TensorBinding input_binding;
  input_binding.set_tensor_name("foo-input");

  Signatures signatures;
  GenericSignature* input_generic_signature =
      (*signatures.mutable_named_signatures())[kPredictInputs]
          .mutable_generic_signature();
  input_generic_signature->mutable_map()->insert({"foo-input", input_binding});

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(0, meta_graph_def.signature_def_size());
}

// Tests up-conversion of Signatures to SignatureDefs when both `default` and
// `named` signatures are present.
TEST(BundleShimTest, DefaultAndNamedSignatureWithPredict) {
  Signatures signatures;

  // Build a generic signature corresponding to `inputs` and add it to the
  // Signatures to up-convert.
  TensorBinding input_binding;
  input_binding.set_tensor_name("foo-input");
  GenericSignature* input_generic_signature =
      (*signatures.mutable_named_signatures())[kPredictInputs]
          .mutable_generic_signature();
  input_generic_signature->mutable_map()->insert({"foo-input", input_binding});

  // Build a generic signature corresponding to `outputs` and add it to the
  // Signatures to up-convert.
  TensorBinding output_binding;
  output_binding.set_tensor_name("foo-output");
  GenericSignature* output_generic_signature =
      (*signatures.mutable_named_signatures())[kPredictOutputs]
          .mutable_generic_signature();
  output_generic_signature->mutable_map()->insert(
      {"foo-output", output_binding});

  // Build a regression signature and set it as the default signature.
  RegressionSignature* inputs_regression_signature =
      (*signatures.mutable_default_signature()).mutable_regression_signature();
  inputs_regression_signature->mutable_input()->set_tensor_name("bar-input");

  // Up-convert the available signatures to SignatureDefs.
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);
  TF_EXPECT_OK(ConvertSignaturesToSignatureDefs(&meta_graph_def));
  EXPECT_EQ(2, meta_graph_def.signature_def_size());

  // Verify that the default regression signature is converted to a
  // SignatureDef that corresponds to the kDefaultServingSignatureDefKey.
  const auto actual_signature_def_regress =
      meta_graph_def.signature_def().find(kDefaultServingSignatureDefKey);
  ASSERT_FALSE(actual_signature_def_regress ==
               meta_graph_def.signature_def().end());
  ASSERT_FALSE(
      actual_signature_def_regress->second.inputs().find(kRegressInputs) ==
      actual_signature_def_regress->second.inputs().end());

  // Verify that the `Predict` SignatureDef is created under a different key.
  const auto actual_signature_def_predict = meta_graph_def.signature_def().find(
      strings::StrCat(kDefaultServingSignatureDefKey, "_from_named"));
  ASSERT_FALSE(actual_signature_def_predict ==
               meta_graph_def.signature_def().end());
  ASSERT_FALSE(
      actual_signature_def_predict->second.inputs().find("foo-input") ==
      actual_signature_def_predict->second.inputs().end());
  EXPECT_EQ("foo-input",
            actual_signature_def_predict->second.inputs()
                .find("foo-input")
                ->second.name());
  ASSERT_FALSE(
      actual_signature_def_predict->second.outputs().find("foo-output") ==
      actual_signature_def_predict->second.outputs().end());
  EXPECT_EQ("foo-output",
            actual_signature_def_predict->second.outputs()
                .find("foo-output")
                ->second.name());
  EXPECT_EQ(kPredictMethodName,
            actual_signature_def_predict->second.method_name());
}

// Checks a basic up conversion for half plus two for SessionBundle.
TEST(BundleShimTest, BasicExportSessionBundle) {
  const std::unordered_set<string> tags = {"tag"};
  const string session_bundle_export_dir =
      test_util::TestSrcDirPath(kSessionBundlePath);
  LoadAndValidateSavedModelBundle(session_bundle_export_dir, tags,
                                  kDefaultServingSignatureDefKey);

  // Verify that the named signature is also present.
  SessionOptions session_options;
  RunOptions run_options;
  SavedModelBundle saved_model_bundle;
  TF_ASSERT_OK(LoadSessionBundleOrSavedModelBundle(session_options, run_options,
                                                   session_bundle_export_dir,
                                                   tags, &saved_model_bundle));
  const MetaGraphDef meta_graph_def = saved_model_bundle.meta_graph_def;
  const auto& signature_def_map = meta_graph_def.signature_def();
  bool found_named_signature = false;
  for (const auto& entry : signature_def_map) {
    const string& key = entry.first;
    const SignatureDef& signature_def = entry.second;

    // We're looking for the key that is *not* kDefaultServingSignatureDefKey.
    if (key == kDefaultServingSignatureDefKey) {
      continue;
    }
    found_named_signature = true;

    EXPECT_EQ(1, signature_def.inputs_size());
    const auto it_inputs_x = signature_def.inputs().find("x");
    EXPECT_FALSE(it_inputs_x == signature_def.inputs().end());
    // Ensure the TensorInfo has name and dtype populated.
    const TensorInfo& tensor_info_x = it_inputs_x->second;
    EXPECT_EQ("x:0", tensor_info_x.name());
    EXPECT_EQ(DT_FLOAT, tensor_info_x.dtype());

    EXPECT_EQ(1, signature_def.outputs_size());
    const auto it_outputs_y = signature_def.outputs().find("y");
    EXPECT_FALSE(it_outputs_y == signature_def.outputs().end());
    // Ensure the TensorInfo has name and dtype populated.
    const TensorInfo& tensor_info_y = it_outputs_y->second;
    EXPECT_EQ("y:0", tensor_info_y.name());
    EXPECT_EQ(DT_FLOAT, tensor_info_y.dtype());
  }
  EXPECT_TRUE(found_named_signature);
}

// Checks a basic load for half plus two for SavedModelBundle.
TEST(BundleShimTest, BasicExportSavedModel) {
  const string saved_model_bundle_export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kSavedModelBundlePath);
  LoadAndValidateSavedModelBundle(saved_model_bundle_export_dir,
                                  {kSavedModelTagServe}, "regress_x_to_y");
}

// Checks a basic load fails with an invalid export path.
TEST(BundleShimTest, InvalidPath) {
  const string invalid_export_dir = testing::TensorFlowSrcRoot();
  SessionOptions session_options;
  RunOptions run_options;
  SavedModelBundle saved_model_bundle;
  Status status = LoadSessionBundleOrSavedModelBundle(
      session_options, run_options, invalid_export_dir, {kSavedModelTagServe},
      &saved_model_bundle);
  EXPECT_EQ(error::Code::NOT_FOUND, status.code());
}

// Checks that if loading a session bundle fails, the error is propagated to
// LoadSessionBundleOrSavedModelBundle().
TEST(BundleShimTest, LoadSessionBundleError) {
  const string session_bundle_export_dir =
      test_util::TestSrcDirPath(kSessionBundlePath);
  SessionOptions session_options;
  RunOptions run_options;
  // Invalid threadpool index to use for session-run calls.
  run_options.set_inter_op_thread_pool(100);
  SavedModelBundle saved_model_bundle;
  EXPECT_FALSE(LoadSessionBundleOrSavedModelBundle(session_options, run_options,
                                                   session_bundle_export_dir,
                                                   {"tag"}, &saved_model_bundle)
                   .ok());
}

}  // namespace
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
