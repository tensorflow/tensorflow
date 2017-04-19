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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class SignatureDefUtilsTest : public ::testing::Test {
 protected:
  MetaGraphDef MakeSampleMetaGraphDef() {
    MetaGraphDef result;
    (*result.mutable_signature_def())["blah"].set_method_name("foo");
    (*result.mutable_signature_def())[kSignatureKey] = MakeSampleSignatureDef();
    (*result.mutable_signature_def())["gnarl"].set_method_name("blah");
    return result;
  }

  SignatureDef MakeSampleSignatureDef() {
    SignatureDef result;
    result.set_method_name(kMethodName);
    (*result.mutable_inputs())[kInput1Key].set_name(kInput1Name);
    (*result.mutable_inputs())[kInput2Key].set_name(kInput2Name);
    (*result.mutable_outputs())[kOutput1Key].set_name(kOutput1Name);
    (*result.mutable_outputs())[kOutput2Key].set_name(kOutput2Name);
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

TEST_F(SignatureDefUtilsTest, FindSignatureDefByKey) {
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

TEST_F(SignatureDefUtilsTest, FindInputTensorNameByKey) {
  const SignatureDef signature_def = MakeSampleSignatureDef();
  string name;
  // Succeeds for an existing input.
  TF_ASSERT_OK(FindInputTensorNameByKey(signature_def, kInput2Key, &name));
  EXPECT_EQ(kInput2Name, name);
  // Fails for a missing input.
  EXPECT_FALSE(
      FindInputTensorNameByKey(signature_def, "nonexistent", &name).ok());
}

TEST_F(SignatureDefUtilsTest, FindOutputTensorNameByKey) {
  const SignatureDef signature_def = MakeSampleSignatureDef();
  string name;
  // Succeeds for an existing output.
  TF_ASSERT_OK(FindOutputTensorNameByKey(signature_def, kOutput2Key, &name));
  EXPECT_EQ(kOutput2Name, name);
  // Fails for a missing output.
  EXPECT_FALSE(
      FindOutputTensorNameByKey(signature_def, "nonexistent", &name).ok());
}

}  // namespace tensorflow
