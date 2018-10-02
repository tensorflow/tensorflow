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

#include "tensorflow/contrib/saved_model/cc/saved_model/signature_def_utils.h"

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FindByKeyTest : public ::testing::Test {
 protected:
  MetaGraphDef MakeSampleMetaGraphDef() {
    MetaGraphDef result;
    (*result.mutable_signature_def())["blah"].set_method_name("foo");
    (*result.mutable_signature_def())[kSignatureKey] = MakeSampleSignatureDef();
    (*result.mutable_signature_def())["gnarl"].set_method_name("blah");
    return result;
  }

  void SetInputNameForKey(const string& key, const string& name,
                          SignatureDef* signature_def) {
    (*signature_def->mutable_inputs())[key].set_name(name);
  }

  void SetOutputNameForKey(const string& key, const string& name,
                           SignatureDef* signature_def) {
    (*signature_def->mutable_outputs())[key].set_name(name);
  }

  SignatureDef MakeSampleSignatureDef() {
    SignatureDef result;
    result.set_method_name(kMethodName);
    SetInputNameForKey(kInput1Key, kInput1Name, &result);
    SetInputNameForKey(kInput2Key, kInput2Name, &result);
    SetOutputNameForKey(kOutput1Key, kOutput1Name, &result);
    SetOutputNameForKey(kOutput2Key, kOutput2Name, &result);
    return result;
  }

  const string kSignatureKey = "my_signature";
  const string kMethodName = "my_method";
  const string kInput1Key = "input_one_key";
  const string kInput1Name = "input_one";
  const string kInput2Key = "input_two_key";
  const string kInput2Name = "input_two";
  const string kOutput1Key = "output_one_key";
  const string kOutput1Name = "output_one";
  const string kOutput2Key = "output_two_key";
  const string kOutput2Name = "output_two";
};

TEST_F(FindByKeyTest, FindSignatureDefByKey) {
  const MetaGraphDef meta_graph_def = MakeSampleMetaGraphDef();
  const SignatureDef* signature_def;
  // Succeeds for an existing signature.
  TF_ASSERT_OK(
      FindSignatureDefByKey(meta_graph_def, kSignatureKey, &signature_def));
  EXPECT_EQ(kMethodName, signature_def->method_name());
  // Fails for a missing signature.
  EXPECT_FALSE(
      FindSignatureDefByKey(meta_graph_def, "nonexistent", &signature_def)
          .ok());
}

TEST_F(FindByKeyTest, FindInputTensorNameByKey) {
  const SignatureDef signature_def = MakeSampleSignatureDef();
  string name;
  // Succeeds for an existing input.
  TF_ASSERT_OK(FindInputTensorNameByKey(signature_def, kInput2Key, &name));
  EXPECT_EQ(kInput2Name, name);
  // Fails for a missing input.
  EXPECT_FALSE(
      FindInputTensorNameByKey(signature_def, "nonexistent", &name).ok());
}

TEST_F(FindByKeyTest, FindOutputTensorNameByKey) {
  const SignatureDef signature_def = MakeSampleSignatureDef();
  string name;
  // Succeeds for an existing output.
  TF_ASSERT_OK(FindOutputTensorNameByKey(signature_def, kOutput2Key, &name));
  EXPECT_EQ(kOutput2Name, name);
  // Fails for a missing output.
  EXPECT_FALSE(
      FindOutputTensorNameByKey(signature_def, "nonexistent", &name).ok());
}

class IsValidSignatureTest : public ::testing::Test {
 protected:
  void SetInputDataTypeForKey(const string& key, DataType dtype) {
    (*signature_def_.mutable_inputs())[key].set_dtype(dtype);
  }

  void SetOutputDataTypeForKey(const string& key, DataType dtype) {
    (*signature_def_.mutable_outputs())[key].set_dtype(dtype);
  }

  void EraseOutputKey(const string& key) {
    (*signature_def_.mutable_outputs()).erase(key);
  }

  void ExpectInvalidSignature() {
    EXPECT_FALSE(IsValidSignature(signature_def_));
  }

  void ExpectValidSignature() { EXPECT_TRUE(IsValidSignature(signature_def_)); }

  SignatureDef signature_def_;
};

TEST_F(IsValidSignatureTest, IsValidPredictSignature) {
  signature_def_.set_method_name("not_kPredictMethodName");
  // Incorrect method name
  ExpectInvalidSignature();

  signature_def_.set_method_name(kPredictMethodName);
  // No inputs
  ExpectInvalidSignature();

  SetInputDataTypeForKey(kPredictInputs, DT_STRING);
  // No outputs
  ExpectInvalidSignature();

  SetOutputDataTypeForKey(kPredictOutputs, DT_STRING);
  ExpectValidSignature();
}

TEST_F(IsValidSignatureTest, IsValidRegressionSignature) {
  signature_def_.set_method_name("not_kRegressMethodName");
  // Incorrect method name
  ExpectInvalidSignature();

  signature_def_.set_method_name(kRegressMethodName);
  // No inputs
  ExpectInvalidSignature();

  SetInputDataTypeForKey(kRegressInputs, DT_STRING);
  // No outputs
  ExpectInvalidSignature();

  SetOutputDataTypeForKey(kRegressOutputs, DT_STRING);
  // Incorrect data type
  ExpectInvalidSignature();

  SetOutputDataTypeForKey(kRegressOutputs, DT_FLOAT);
  ExpectValidSignature();
}

TEST_F(IsValidSignatureTest, IsValidClassificationSignature) {
  signature_def_.set_method_name("not_kClassifyMethodName");
  // Incorrect method name
  ExpectInvalidSignature();

  signature_def_.set_method_name(kClassifyMethodName);
  // No inputs
  ExpectInvalidSignature();

  SetInputDataTypeForKey(kClassifyInputs, DT_STRING);
  // No outputs
  ExpectInvalidSignature();

  SetOutputDataTypeForKey("invalidKey", DT_FLOAT);
  // Invalid key
  ExpectInvalidSignature();

  EraseOutputKey("invalidKey");
  SetOutputDataTypeForKey(kClassifyOutputClasses, DT_FLOAT);
  // Invalid dtype for classes
  ExpectInvalidSignature();

  SetOutputDataTypeForKey(kClassifyOutputClasses, DT_STRING);
  // Valid without scores
  ExpectValidSignature();

  SetOutputDataTypeForKey(kClassifyOutputScores, DT_STRING);
  // Invalid dtype for scores
  ExpectInvalidSignature();

  SetOutputDataTypeForKey(kClassifyOutputScores, DT_FLOAT);
  // Valid with both classes and scores
  ExpectValidSignature();
}

}  // namespace tensorflow
