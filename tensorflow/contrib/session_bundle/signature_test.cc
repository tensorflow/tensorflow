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

#include "tensorflow/contrib/session_bundle/signature.h"

#include <memory>

#include "google/protobuf/any.pb.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace serving {
namespace {

static bool HasSubstr(StringPiece base, StringPiece substr) {
  bool ok = str_util::StrContains(base, substr);
  EXPECT_TRUE(ok) << base << ", expected substring " << substr;
  return ok;
}

TEST(GetClassificationSignature, Basic) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  ClassificationSignature* input_signature =
      signatures.mutable_default_signature()
          ->mutable_classification_signature();
  input_signature->mutable_input()->set_tensor_name("flow");
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  ClassificationSignature signature;
  const Status status = GetClassificationSignature(meta_graph_def, &signature);
  TF_ASSERT_OK(status);
  EXPECT_EQ(signature.input().tensor_name(), "flow");
}

TEST(GetClassificationSignature, MissingSignature) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  signatures.mutable_default_signature();
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  ClassificationSignature signature;
  const Status status = GetClassificationSignature(meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Expected a classification signature"))
      << status.error_message();
}

TEST(GetClassificationSignature, WrongSignatureType) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  signatures.mutable_default_signature()->mutable_regression_signature();
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  ClassificationSignature signature;
  const Status status = GetClassificationSignature(meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Expected a classification signature"))
      << status.error_message();
}

TEST(GetNamedClassificationSignature, Basic) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  ClassificationSignature* input_signature =
      (*signatures.mutable_named_signatures())["foo"]
          .mutable_classification_signature();
  input_signature->mutable_input()->set_tensor_name("flow");
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  ClassificationSignature signature;
  const Status status =
      GetNamedClassificationSignature("foo", meta_graph_def, &signature);
  TF_ASSERT_OK(status);
  EXPECT_EQ(signature.input().tensor_name(), "flow");
}

TEST(GetNamedClassificationSignature, MissingSignature) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  ClassificationSignature signature;
  const Status status =
      GetNamedClassificationSignature("foo", meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Missing signature named \"foo\""))
      << status.error_message();
}

TEST(GetNamedClassificationSignature, WrongSignatureType) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  (*signatures.mutable_named_signatures())["foo"]
      .mutable_regression_signature();
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  ClassificationSignature signature;
  const Status status =
      GetNamedClassificationSignature("foo", meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(
      status.error_message(),
      "Expected a classification signature for name \"foo\""))
      << status.error_message();
}

TEST(GetRegressionSignature, Basic) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  RegressionSignature* input_signature =
      signatures.mutable_default_signature()->mutable_regression_signature();
  input_signature->mutable_input()->set_tensor_name("flow");
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  RegressionSignature signature;
  const Status status = GetRegressionSignature(meta_graph_def, &signature);
  TF_ASSERT_OK(status);
  EXPECT_EQ(signature.input().tensor_name(), "flow");
}

TEST(GetRegressionSignature, MissingSignature) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  signatures.mutable_default_signature();
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  RegressionSignature signature;
  const Status status = GetRegressionSignature(meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Expected a regression signature"))
      << status.error_message();
}

TEST(GetRegressionSignature, WrongSignatureType) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  signatures.mutable_default_signature()->mutable_classification_signature();
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  RegressionSignature signature;
  const Status status = GetRegressionSignature(meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Expected a regression signature"))
      << status.error_message();
}

TEST(GetNamedSignature, Basic) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  ClassificationSignature* input_signature =
      (*signatures.mutable_named_signatures())["foo"]
          .mutable_classification_signature();
  input_signature->mutable_input()->set_tensor_name("flow");
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  Signature signature;
  const Status status = GetNamedSignature("foo", meta_graph_def, &signature);
  TF_ASSERT_OK(status);
  EXPECT_EQ(signature.classification_signature().input().tensor_name(), "flow");
}

TEST(GetNamedSignature, MissingSignature) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  Signature signature;
  const Status status = GetNamedSignature("foo", meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Missing signature named \"foo\""))
      << status.error_message();
}

// MockSession used to test input and output interactions with a
// tensorflow::Session.
struct MockSession : public tensorflow::Session {
  ~MockSession() override = default;

  Status Create(const GraphDef& graph) override {
    return errors::Unimplemented("Not implemented for mock.");
  }

  Status Extend(const GraphDef& graph) override {
    return errors::Unimplemented("Not implemented for mock.");
  }

  // Sets the input and output arguments.
  Status Run(const std::vector<std::pair<string, Tensor>>& inputs_arg,
             const std::vector<string>& output_tensor_names_arg,
             const std::vector<string>& target_node_names_arg,
             std::vector<Tensor>* outputs_arg) override {
    inputs = inputs_arg;
    output_tensor_names = output_tensor_names_arg;
    target_node_names = target_node_names_arg;
    *outputs_arg = outputs;
    return status;
  }

  Status Close() override {
    return errors::Unimplemented("Not implemented for mock.");
  }

  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return errors::Unimplemented("Not implemented for mock.");
  }

  // Arguments stored on a Run call.
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> output_tensor_names;
  std::vector<string> target_node_names;

  // Output argument set by Run; should be set before calling.
  std::vector<Tensor> outputs;

  // Return value for Run; should be set before calling.
  Status status;
};

constexpr char kInputName[] = "in:0";
constexpr char kClassesName[] = "classes:0";
constexpr char kScoresName[] = "scores:0";

class RunClassificationTest : public ::testing::Test {
 public:
  void SetUp() override {
    signature_.mutable_input()->set_tensor_name(kInputName);
    signature_.mutable_classes()->set_tensor_name(kClassesName);
    signature_.mutable_scores()->set_tensor_name(kScoresName);
  }

 protected:
  ClassificationSignature signature_;
  Tensor input_tensor_;
  Tensor classes_tensor_;
  Tensor scores_tensor_;
  MockSession session_;
};

TEST_F(RunClassificationTest, Basic) {
  input_tensor_ = test::AsTensor<int>({99});
  session_.outputs = {test::AsTensor<int>({3}), test::AsTensor<int>({2})};
  const Status status = RunClassification(signature_, input_tensor_, &session_,
                                          &classes_tensor_, &scores_tensor_);

  // Validate outputs.
  TF_ASSERT_OK(status);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({3}), classes_tensor_);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({2}), scores_tensor_);

  // Validate inputs.
  ASSERT_EQ(1, session_.inputs.size());
  EXPECT_EQ(kInputName, session_.inputs[0].first);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({99}),
                               session_.inputs[0].second);

  ASSERT_EQ(2, session_.output_tensor_names.size());
  EXPECT_EQ(kClassesName, session_.output_tensor_names[0]);
  EXPECT_EQ(kScoresName, session_.output_tensor_names[1]);
}

TEST_F(RunClassificationTest, ClassesOnly) {
  input_tensor_ = test::AsTensor<int>({99});
  session_.outputs = {test::AsTensor<int>({3})};
  const Status status = RunClassification(signature_, input_tensor_, &session_,
                                          &classes_tensor_, nullptr);

  // Validate outputs.
  TF_ASSERT_OK(status);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({3}), classes_tensor_);

  // Validate inputs.
  ASSERT_EQ(1, session_.inputs.size());
  EXPECT_EQ(kInputName, session_.inputs[0].first);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({99}),
                               session_.inputs[0].second);

  ASSERT_EQ(1, session_.output_tensor_names.size());
  EXPECT_EQ(kClassesName, session_.output_tensor_names[0]);
}

TEST_F(RunClassificationTest, ScoresOnly) {
  input_tensor_ = test::AsTensor<int>({99});
  session_.outputs = {test::AsTensor<int>({2})};
  const Status status = RunClassification(signature_, input_tensor_, &session_,
                                          nullptr, &scores_tensor_);

  // Validate outputs.
  TF_ASSERT_OK(status);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({2}), scores_tensor_);

  // Validate inputs.
  ASSERT_EQ(1, session_.inputs.size());
  EXPECT_EQ(kInputName, session_.inputs[0].first);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({99}),
                               session_.inputs[0].second);

  ASSERT_EQ(1, session_.output_tensor_names.size());
  EXPECT_EQ(kScoresName, session_.output_tensor_names[0]);
}

TEST(RunClassification, RunNotOk) {
  ClassificationSignature signature;
  signature.mutable_input()->set_tensor_name("in:0");
  signature.mutable_classes()->set_tensor_name("classes:0");
  Tensor input_tensor = test::AsTensor<int>({99});
  MockSession session;
  session.status = errors::DataLoss("Data is gone");
  Tensor classes_tensor;
  const Status status = RunClassification(signature, input_tensor, &session,
                                          &classes_tensor, nullptr);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(), "Data is gone"))
      << status.error_message();
}

TEST(RunClassification, TooManyOutputs) {
  ClassificationSignature signature;
  signature.mutable_input()->set_tensor_name("in:0");
  signature.mutable_classes()->set_tensor_name("classes:0");
  Tensor input_tensor = test::AsTensor<int>({99});
  MockSession session;
  session.outputs = {test::AsTensor<int>({3}), test::AsTensor<int>({4})};

  Tensor classes_tensor;
  const Status status = RunClassification(signature, input_tensor, &session,
                                          &classes_tensor, nullptr);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(), "Expected 1 output"))
      << status.error_message();
}

TEST(RunClassification, WrongBatchOutputs) {
  ClassificationSignature signature;
  signature.mutable_input()->set_tensor_name("in:0");
  signature.mutable_classes()->set_tensor_name("classes:0");
  Tensor input_tensor = test::AsTensor<int>({99, 100});
  MockSession session;
  session.outputs = {test::AsTensor<int>({3})};

  Tensor classes_tensor;
  const Status status = RunClassification(signature, input_tensor, &session,
                                          &classes_tensor, nullptr);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(),
                            "Input batch size did not match output batch size"))
      << status.error_message();
}

constexpr char kRegressionsName[] = "regressions:0";

class RunRegressionTest : public ::testing::Test {
 public:
  void SetUp() override {
    signature_.mutable_input()->set_tensor_name(kInputName);
    signature_.mutable_output()->set_tensor_name(kRegressionsName);
  }

 protected:
  RegressionSignature signature_;
  Tensor input_tensor_;
  Tensor output_tensor_;
  MockSession session_;
};

TEST_F(RunRegressionTest, Basic) {
  input_tensor_ = test::AsTensor<int>({99, 100});
  session_.outputs = {test::AsTensor<float>({1, 2})};
  const Status status =
      RunRegression(signature_, input_tensor_, &session_, &output_tensor_);

  // Validate outputs.
  TF_ASSERT_OK(status);
  test::ExpectTensorEqual<float>(test::AsTensor<float>({1, 2}), output_tensor_);

  // Validate inputs.
  ASSERT_EQ(1, session_.inputs.size());
  EXPECT_EQ(kInputName, session_.inputs[0].first);
  test::ExpectTensorEqual<int>(test::AsTensor<int>({99, 100}),
                               session_.inputs[0].second);

  ASSERT_EQ(1, session_.output_tensor_names.size());
  EXPECT_EQ(kRegressionsName, session_.output_tensor_names[0]);
}

TEST_F(RunRegressionTest, RunNotOk) {
  input_tensor_ = test::AsTensor<int>({99});
  session_.status = errors::DataLoss("Data is gone");
  const Status status =
      RunRegression(signature_, input_tensor_, &session_, &output_tensor_);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(), "Data is gone"))
      << status.error_message();
}

TEST_F(RunRegressionTest, MismatchedSizeForBatchInputAndOutput) {
  input_tensor_ = test::AsTensor<int>({99, 100});
  session_.outputs = {test::AsTensor<float>({3})};

  const Status status =
      RunRegression(signature_, input_tensor_, &session_, &output_tensor_);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(),
                            "Input batch size did not match output batch size"))
      << status.error_message();
}

TEST(SetAndGetSignatures, RoundTrip) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  signatures.mutable_default_signature()
      ->mutable_classification_signature()
      ->mutable_input()
      ->set_tensor_name("in:0");
  TF_ASSERT_OK(SetSignatures(signatures, &meta_graph_def));
  Signatures read_signatures;
  TF_ASSERT_OK(GetSignatures(meta_graph_def, &read_signatures));

  EXPECT_EQ("in:0", read_signatures.default_signature()
                        .classification_signature()
                        .input()
                        .tensor_name());
}

TEST(GetSignatures, MissingSignature) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures read_signatures;
  const auto status = GetSignatures(meta_graph_def, &read_signatures);
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION, status.code());
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(), "Expected exactly one"))
      << status.error_message();
}

TEST(GetSignatures, WrongProtoInAny) {
  tensorflow::MetaGraphDef meta_graph_def;
  auto& collection_def = *(meta_graph_def.mutable_collection_def());
  auto* any =
      collection_def[kSignaturesKey].mutable_any_list()->mutable_value()->Add();
  // Put an unexpected type into the Signatures Any.
  any->PackFrom(TensorBinding());
  Signatures read_signatures;
  const auto status = GetSignatures(meta_graph_def, &read_signatures);
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION, status.code());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Expected Any type_url for: "
                                    "tensorflow.serving.Signatures"))
      << status.error_message();
}

TEST(GetSignatures, JunkInAny) {
  tensorflow::MetaGraphDef meta_graph_def;
  auto& collection_def = *(meta_graph_def.mutable_collection_def());
  auto* any =
      collection_def[kSignaturesKey].mutable_any_list()->mutable_value()->Add();
  // Create a valid Any then corrupt it.
  any->PackFrom(Signatures());
  any->set_value("junk junk");
  Signatures read_signatures;
  const auto status = GetSignatures(meta_graph_def, &read_signatures);
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION, status.code());
  EXPECT_TRUE(str_util::StrContains(status.error_message(), "Failed to unpack"))
      << status.error_message();
}

TEST(GetSignatures, DefaultAndNamedTogetherOK) {
  tensorflow::MetaGraphDef meta_graph_def;
  auto& collection_def = *(meta_graph_def.mutable_collection_def());
  auto* any =
      collection_def[kSignaturesKey].mutable_any_list()->mutable_value()->Add();
  Signatures signatures;
  signatures.mutable_default_signature()
      ->mutable_classification_signature()
      ->mutable_input()
      ->set_tensor_name("in:0");
  ClassificationSignature* input_signature =
      (*signatures.mutable_named_signatures())["foo"]
          .mutable_classification_signature();
  input_signature->mutable_input()->set_tensor_name("flow");

  any->PackFrom(signatures);
  Signatures read_signatures;
  const auto status = GetSignatures(meta_graph_def, &read_signatures);

  EXPECT_TRUE(status.ok());
}

// Check that we only have one 'Signatures' entry in the collection_def map.
// Note that each such object can have multiple named_signatures inside of it.
TEST(GetSignatures, MultipleSignaturesNotOK) {
  tensorflow::MetaGraphDef meta_graph_def;
  auto& collection_def = *(meta_graph_def.mutable_collection_def());
  auto* any =
      collection_def[kSignaturesKey].mutable_any_list()->mutable_value()->Add();
  Signatures signatures;
  signatures.mutable_default_signature()
      ->mutable_classification_signature()
      ->mutable_input()
      ->set_tensor_name("in:0");
  any->PackFrom(signatures);

  // Add another signatures object.
  any =
      collection_def[kSignaturesKey].mutable_any_list()->mutable_value()->Add();
  any->PackFrom(signatures);
  Signatures read_signatures;
  const auto status = GetSignatures(meta_graph_def, &read_signatures);
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION, status.code());
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(), "Expected exactly one"))
      << status.error_message();
}

// GenericSignature test fixture that contains a signature initialized with two
// bound Tensors.
class GenericSignatureTest : public ::testing::Test {
 protected:
  GenericSignatureTest() {
    TensorBinding binding;
    binding.set_tensor_name("graph_A");
    signature_.mutable_map()->insert({"logical_A", binding});

    binding.set_tensor_name("graph_B");
    signature_.mutable_map()->insert({"logical_B", binding});
  }

  // GenericSignature that contains two bound Tensors.
  GenericSignature signature_;
};

// GenericSignature tests.

TEST_F(GenericSignatureTest, GetGenericSignatureBasic) {
  Signature expected_signature;
  expected_signature.mutable_generic_signature()->MergeFrom(signature_);

  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  signatures.mutable_named_signatures()->insert(
      {"generic_bindings", expected_signature});
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  GenericSignature actual_signature;
  TF_ASSERT_OK(GetGenericSignature("generic_bindings", meta_graph_def,
                                   &actual_signature));
  ASSERT_EQ("graph_A", actual_signature.map().at("logical_A").tensor_name());
  ASSERT_EQ("graph_B", actual_signature.map().at("logical_B").tensor_name());
}

TEST(GetGenericSignature, MissingSignature) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  GenericSignature signature;
  const Status status =
      GetGenericSignature("generic_bindings", meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(HasSubstr(status.error_message(),
                        "Missing generic signature named \"generic_bindings\""))
      << status.error_message();
}

TEST(GetGenericSignature, WrongSignatureType) {
  tensorflow::MetaGraphDef meta_graph_def;
  Signatures signatures;
  (*signatures.mutable_named_signatures())["generic_bindings"]
      .mutable_regression_signature();
  (*meta_graph_def.mutable_collection_def())[kSignaturesKey]
      .mutable_any_list()
      ->add_value()
      ->PackFrom(signatures);

  GenericSignature signature;
  const Status status =
      GetGenericSignature("generic_bindings", meta_graph_def, &signature);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.error_message(),
                                    "Expected a generic signature:"))
      << status.error_message();
}

// BindGeneric Tests.

TEST_F(GenericSignatureTest, BindGenericInputsBasic) {
  const std::vector<std::pair<string, Tensor>> inputs = {
      {"logical_A", test::AsTensor<float>({-1.0})},
      {"logical_B", test::AsTensor<float>({-2.0})}};

  std::vector<std::pair<string, Tensor>> bound_inputs;
  TF_ASSERT_OK(BindGenericInputs(signature_, inputs, &bound_inputs));

  EXPECT_EQ("graph_A", bound_inputs[0].first);
  EXPECT_EQ("graph_B", bound_inputs[1].first);
  test::ExpectTensorEqual<float>(test::AsTensor<float>({-1.0}),
                                 bound_inputs[0].second);
  test::ExpectTensorEqual<float>(test::AsTensor<float>({-2.0}),
                                 bound_inputs[1].second);
}

TEST_F(GenericSignatureTest, BindGenericInputsMissingBinding) {
  const std::vector<std::pair<string, Tensor>> inputs = {
      {"logical_A", test::AsTensor<float>({-42.0})},
      {"logical_MISSING", test::AsTensor<float>({-43.0})}};

  std::vector<std::pair<string, Tensor>> bound_inputs;
  const Status status = BindGenericInputs(signature_, inputs, &bound_inputs);
  ASSERT_FALSE(status.ok());
}

TEST_F(GenericSignatureTest, BindGenericNamesBasic) {
  const std::vector<string> input_names = {"logical_B", "logical_A"};
  std::vector<string> bound_names;
  TF_ASSERT_OK(BindGenericNames(signature_, input_names, &bound_names));

  EXPECT_EQ("graph_B", bound_names[0]);
  EXPECT_EQ("graph_A", bound_names[1]);
}

TEST_F(GenericSignatureTest, BindGenericNamesMissingBinding) {
  const std::vector<string> input_names = {"logical_B", "logical_MISSING"};
  std::vector<string> bound_names;
  const Status status = BindGenericNames(signature_, input_names, &bound_names);
  ASSERT_FALSE(status.ok());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
