/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/experimental/saved_model/public/saved_model_api.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/experimental/saved_model/core/tf_saved_model_api.h"
#include "tensorflow/c/experimental/saved_model/internal/saved_model_api_type.h"
#include "tensorflow/c/experimental/saved_model/public/concrete_function.h"
#include "tensorflow/c/experimental/saved_model/public/signature_def_function.h"
#include "tensorflow/c/experimental/saved_model/public/signature_def_function_metadata.h"
#include "tensorflow/c/experimental/saved_model/public/signature_def_param.h"
#include "tensorflow/c/experimental/saved_model/public/signature_def_param_list.h"
#include "tensorflow/c/experimental/saved_model/public/tensor_spec.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_shape.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"

namespace {

using tensorflow::tstring;

constexpr char kTestData[] = "cc/saved_model/testdata";
const char* kServeTag[] = {"serve"};

std::string SavedModelPath(absl::string_view saved_model_dir) {
  return tensorflow::io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                                  kTestData, saved_model_dir);
}

// This value parameterized test allows us to test both TFRT
// and non TFRT runtimes.
// https://github.com/google/googletest/blob/dcc92d0ab6c4ce022162a23566d44f673251eee4/googletest/docs/advanced.md#value-parameterized-tests
class CSavedModelAPITest : public ::testing::TestWithParam<bool> {};

TEST_P(CSavedModelAPITest, LoadsSavedModelWithTags) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  TFE_ContextOptionsSetTfrt(opts, use_tfrt);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string model_dir = SavedModelPath("VarsAndArithmeticObjectGraph");

  TF_SavedModel* saved_model =
      TF_LoadSavedModelWithTags(model_dir.c_str(), ctx, kServeTag, 1, status);

  // TODO(bmzhao): Change this to expect TF_OK when loading is implemented.
  // That unblocks writing other tests that require a TF_SavedModel*,
  // like loading a ConcreteFunction. This test at least checks that the
  // C API builds and can be minimally run.
  EXPECT_EQ(TF_GetCode(status), TF_UNIMPLEMENTED);

  TF_DeleteSavedModel(saved_model);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
}

TEST_P(CSavedModelAPITest, LoadsSavedModel) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  TFE_ContextOptionsSetTfrt(opts, use_tfrt);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string model_dir = SavedModelPath("VarsAndArithmeticObjectGraph");

  TF_SavedModel* saved_model =
      TF_LoadSavedModel(model_dir.c_str(), ctx, status);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_ConcreteFunction* compute_fn =
      TF_GetSavedModelConcreteFunction(saved_model, "compute", status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> compute_fn_inputs;
  TFE_TensorHandle* input_a = TestScalarTensorHandle(ctx, 2.0f);
  TFE_TensorHandle* input_b = TestScalarTensorHandle(ctx, 1.0f);
  compute_fn_inputs.push_back(input_a);
  compute_fn_inputs.push_back(input_b);

  TFE_Op* compute_fn_op = TF_ConcreteFunctionMakeCallOp(
      compute_fn, compute_fn_inputs.data(), compute_fn_inputs.size(), status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  // TODO(bmzhao): Finish API on FunctionMetadata args, so we know how many
  // inputs + outputs a function has.
  TFE_TensorHandle* compute_fn_outputs[1] = {nullptr};
  int num_retvals = 1;

  TFE_Execute(compute_fn_op, &compute_fn_outputs[0], &num_retvals, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_Tensor* result = TFE_TensorHandleResolve(compute_fn_outputs[0], status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(TF_NumDims(result), 0);
  float output_value = *static_cast<float*>(TF_TensorData(result));
  // (1 + 2) * (2 + 1) / 3 + 5 should be 8
  EXPECT_FLOAT_EQ(output_value, 8.0);

  TF_DeleteTensor(result);
  TFE_DeleteTensorHandle(compute_fn_outputs[0]);
  TFE_DeleteTensorHandle(input_a);
  TFE_DeleteTensorHandle(input_b);
  TFE_DeleteOp(compute_fn_op);
  TF_DeleteSavedModel(saved_model);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
}

// This tests running the "serving_default" SignatureDefFunction from the
// VarsAndArithmeticObjectGraph savedmodel. Here's what the signature_defs
// protobuf in the metagraph looks like:
// signature_def: {
//   key  : "serving_default"
//   value: {
//     inputs: {
//       key  : "a"
//       value: {
//         name : "serving_default_a:0"
//         dtype: DT_FLOAT
//         tensor_shape: {
//         }
//       }
//     }
//     inputs: {
//       key  : "b"
//       value: {
//         name : "serving_default_b:0"
//         dtype: DT_FLOAT
//         tensor_shape: {
//         }
//       }
//     }
//     outputs: {
//       key  : "output_0"
//       value: {
//         name : "StatefulPartitionedCall:0"
//         dtype: DT_FLOAT
//         tensor_shape: {
//         }
//       }
//     }
//     method_name: "tensorflow/serving/predict"
//   }
// }
TEST_P(CSavedModelAPITest, RunsSignatureDefFunction) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  TFE_ContextOptionsSetTfrt(opts, use_tfrt);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string model_dir = SavedModelPath("VarsAndArithmeticObjectGraph");

  TF_SavedModel* saved_model =
      TF_LoadSavedModel(model_dir.c_str(), ctx, status);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_SignatureDefFunction* serving_default =
      TF_GetSavedModelSignatureDefFunction(saved_model, "serving_default",
                                           status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_SignatureDefFunctionMetadata* metadata =
      TF_SignatureDefFunctionGetMetadata(serving_default);

  const TF_SignatureDefParamList* args =
      TF_SignatureDefFunctionMetadataArgs(metadata);
  const TF_SignatureDefParamList* returns =
      TF_SignatureDefFunctionMetadataReturns(metadata);

  EXPECT_EQ(TF_SignatureDefParamListSize(args), 2);
  const TF_SignatureDefParam* param_a = TF_SignatureDefParamListGet(args, 0);
  const TF_TensorSpec* tensor_spec_a = TF_SignatureDefParamTensorSpec(param_a);
  const TF_Shape* shape_a = TF_TensorSpecShape(tensor_spec_a);

  // Input "a" is a scalar, float32 tensor
  EXPECT_EQ("a", std::string(TF_SignatureDefParamName(param_a)));
  EXPECT_EQ(TF_FLOAT, TF_TensorSpecDataType(tensor_spec_a));
  EXPECT_EQ(0, TF_ShapeDims(shape_a));

  const TF_SignatureDefParam* param_b = TF_SignatureDefParamListGet(args, 1);
  const TF_TensorSpec* tensor_spec_b = TF_SignatureDefParamTensorSpec(param_b);
  const TF_Shape* shape_b = TF_TensorSpecShape(tensor_spec_b);

  // Input "b" is a scalar, float32 tensor
  EXPECT_EQ("b", std::string(TF_SignatureDefParamName(param_b)));
  EXPECT_EQ(TF_FLOAT, TF_TensorSpecDataType(tensor_spec_b));
  EXPECT_EQ(0, TF_ShapeDims(shape_b));

  EXPECT_EQ(TF_SignatureDefParamListSize(returns), 1);

  const TF_SignatureDefParam* param_out =
      TF_SignatureDefParamListGet(returns, 0);
  const TF_TensorSpec* tensor_spec_out =
      TF_SignatureDefParamTensorSpec(param_out);
  const TF_Shape* shape_out = TF_TensorSpecShape(tensor_spec_out);

  // Output "output_0" is a scalar, float32 tensor
  EXPECT_EQ("output_0", std::string(TF_SignatureDefParamName(param_out)));
  EXPECT_EQ(TF_FLOAT, TF_TensorSpecDataType(tensor_spec_out));
  EXPECT_EQ(0, TF_ShapeDims(shape_out));

  std::vector<TFE_TensorHandle*> compute_fn_inputs;
  TFE_TensorHandle* input_a = TestScalarTensorHandle(ctx, 2.0f);
  TFE_TensorHandle* input_b = TestScalarTensorHandle(ctx, 1.0f);
  compute_fn_inputs.push_back(input_a);
  compute_fn_inputs.push_back(input_b);

  TFE_Op* serving_default_op = TF_SignatureDefFunctionMakeCallOp(
      serving_default, compute_fn_inputs.data(), compute_fn_inputs.size(),
      status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> compute_fn_outputs(
      TF_SignatureDefParamListSize(returns));
  int num_retvals = TF_SignatureDefParamListSize(returns);

  TFE_Execute(serving_default_op, compute_fn_outputs.data(), &num_retvals,
              status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_Tensor* result = TFE_TensorHandleResolve(compute_fn_outputs[0], status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(TF_NumDims(result), 0);
  float output_value = *static_cast<float*>(TF_TensorData(result));
  // (1 + 2) * (2 + 1) / 3 + 5 should be 8
  EXPECT_FLOAT_EQ(output_value, 8.0);

  TF_DeleteTensor(result);
  TFE_DeleteTensorHandle(compute_fn_outputs[0]);
  TFE_DeleteTensorHandle(input_a);
  TFE_DeleteTensorHandle(input_b);
  TFE_DeleteOp(serving_default_op);
  TF_DeleteSavedModel(saved_model);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
}

TEST_P(CSavedModelAPITest, LoadsAssetSavedModel) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  TFE_ContextOptionsSetTfrt(opts, use_tfrt);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string model_dir = SavedModelPath("AssetModule");

  TF_SavedModel* saved_model =
      TF_LoadSavedModel(model_dir.c_str(), ctx, status);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_ConcreteFunction* read_file_fn =
      TF_GetSavedModelConcreteFunction(saved_model, "read_file", status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_Op* read_file_op =
      TF_ConcreteFunctionMakeCallOp(read_file_fn, nullptr, 0, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  // TODO(bmzhao): Finish API on FunctionMetadata args, so we know how many
  // inputs + outputs a function has.
  TFE_TensorHandle* read_file_fn_outputs[1] = {nullptr};
  int num_retvals = 1;

  TFE_Execute(read_file_op, &read_file_fn_outputs[0], &num_retvals, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_Tensor* result = TFE_TensorHandleResolve(read_file_fn_outputs[0], status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(TF_NumDims(result), 0);
  tensorflow::tstring* output_value =
      static_cast<tensorflow::tstring*>(TF_TensorData(result));
  std::string file_contents(*output_value);
  EXPECT_NE(file_contents.find("TEST ASSET FILE CONTENTS"), std::string::npos);

  TF_DeleteTensor(result);
  TFE_DeleteTensorHandle(read_file_fn_outputs[0]);
  TFE_DeleteOp(read_file_op);
  TF_DeleteSavedModel(saved_model);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
}

TEST_P(CSavedModelAPITest, LoadsStaticHashtableSavedModel) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  TFE_ContextOptionsSetTfrt(opts, use_tfrt);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string model_dir = SavedModelPath("StaticHashTableModule");

  TF_SavedModel* saved_model =
      TF_LoadSavedModel(model_dir.c_str(), ctx, status);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_ConcreteFunction* lookup_fn =
      TF_GetSavedModelConcreteFunction(saved_model, "lookup", status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  // Note(bmzhao): Based on static_hashtable_asset.txt, we expect the following
  // mapping:
  // "foo" -> 0
  // "bar" -> 1
  // "baz" -> 2
  // "wombat" -> 3
  // all other strings -> -1

  // Call lookup function with input "foo", expecting an output of 0
  {
    std::vector<TFE_TensorHandle*> lookup_fn_inputs;
    TFE_TensorHandle* input_foo = TestScalarTensorHandle(ctx, tstring("foo"));
    lookup_fn_inputs.push_back(input_foo);

    TFE_Op* lookup_op = TF_ConcreteFunctionMakeCallOp(
        lookup_fn, lookup_fn_inputs.data(), lookup_fn_inputs.size(), status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    // TODO(bmzhao): Finish API on FunctionMetadata args, so we know how many
    // inputs + outputs a function has.
    TFE_TensorHandle* lookup_fn_outputs[1] = {nullptr};
    int num_retvals = 1;

    TFE_Execute(lookup_op, &lookup_fn_outputs[0], &num_retvals, status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    TF_Tensor* result = TFE_TensorHandleResolve(lookup_fn_outputs[0], status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    EXPECT_EQ(TF_NumDims(result), 0);
    int64_t* output_value = static_cast<int64_t*>(TF_TensorData(result));
    EXPECT_EQ(*output_value, 0);

    TF_DeleteTensor(result);
    TFE_DeleteTensorHandle(input_foo);
    TFE_DeleteTensorHandle(lookup_fn_outputs[0]);
    TFE_DeleteOp(lookup_op);
  }

  // Call lookup function with input "baz", expecting an output of 2
  {
    std::vector<TFE_TensorHandle*> lookup_fn_inputs;
    TFE_TensorHandle* input_foo = TestScalarTensorHandle(ctx, tstring("baz"));
    lookup_fn_inputs.push_back(input_foo);

    TFE_Op* lookup_op = TF_ConcreteFunctionMakeCallOp(
        lookup_fn, lookup_fn_inputs.data(), lookup_fn_inputs.size(), status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    // TODO(bmzhao): Finish API on FunctionMetadata args, so we know how many
    // inputs + outputs a function has.
    TFE_TensorHandle* lookup_fn_outputs[1] = {nullptr};
    int num_retvals = 1;

    TFE_Execute(lookup_op, &lookup_fn_outputs[0], &num_retvals, status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    TF_Tensor* result = TFE_TensorHandleResolve(lookup_fn_outputs[0], status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    EXPECT_EQ(TF_NumDims(result), 0);
    int64_t* output_value = static_cast<int64_t*>(TF_TensorData(result));
    EXPECT_EQ(*output_value, 2);

    TF_DeleteTensor(result);
    TFE_DeleteTensorHandle(input_foo);
    TFE_DeleteTensorHandle(lookup_fn_outputs[0]);
    TFE_DeleteOp(lookup_op);
  }

  // Call lookup function w/input "NON-EXISTENT-KEY", expecting an output of -1
  {
    std::vector<TFE_TensorHandle*> lookup_fn_inputs;
    TFE_TensorHandle* input_foo =
        TestScalarTensorHandle(ctx, tstring("NON-EXISTENT-KEY"));
    lookup_fn_inputs.push_back(input_foo);

    TFE_Op* lookup_op = TF_ConcreteFunctionMakeCallOp(
        lookup_fn, lookup_fn_inputs.data(), lookup_fn_inputs.size(), status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    // TODO(bmzhao): Finish API on FunctionMetadata args, so we know how many
    // inputs + outputs a function has.
    TFE_TensorHandle* lookup_fn_outputs[1] = {nullptr};
    int num_retvals = 1;

    TFE_Execute(lookup_op, &lookup_fn_outputs[0], &num_retvals, status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    TF_Tensor* result = TFE_TensorHandleResolve(lookup_fn_outputs[0], status);
    EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

    EXPECT_EQ(TF_NumDims(result), 0);
    int64_t* output_value = static_cast<int64_t*>(TF_TensorData(result));
    EXPECT_EQ(*output_value, -1);

    TF_DeleteTensor(result);
    TFE_DeleteTensorHandle(input_foo);
    TFE_DeleteTensorHandle(lookup_fn_outputs[0]);
    TFE_DeleteOp(lookup_op);
  }

  TF_DeleteSavedModel(saved_model);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
}

TEST_P(CSavedModelAPITest, LoadSavedModelWithUninitializedVariable) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  TFE_ContextOptionsSetTfrt(opts, use_tfrt);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string model_dir = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(),
      "c/experimental/saved_model/internal/testdata/UninitializedVariable");

  TF_SavedModel* saved_model =
      TF_LoadSavedModel(model_dir.c_str(), ctx, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::TFSavedModelAPI* model_api =
      tensorflow::down_cast<tensorflow::TFSavedModelAPI*>(
          tensorflow::unwrap(saved_model));
  tensorflow::Variable* uninitialized_variable;
  ASSERT_EQ(absl::OkStatus(), model_api->GetVariable("uninitialized_variable",
                                                     &uninitialized_variable));
  ASSERT_EQ(tensorflow::DT_FLOAT, uninitialized_variable->dtype());

  ASSERT_EQ(absl::OkStatus(),
            model_api->GetVariable("sub_module.uninitialized_variable",
                                   &uninitialized_variable));
  ASSERT_EQ(tensorflow::DT_INT64, uninitialized_variable->dtype());

  TF_DeleteSavedModel(saved_model);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
}

TEST_P(CSavedModelAPITest, LoadSavedModelWithWhileLoop) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  TFE_ContextOptionsSetTfrt(opts, use_tfrt);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string model_dir = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(),
      "c/experimental/saved_model/internal/testdata/SimpleWhileLoop");

  TF_SavedModel* saved_model =
      TF_LoadSavedModel(model_dir.c_str(), ctx, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_ConcreteFunction* while_fn =
      TF_GetSavedModelConcreteFunction(saved_model, "compute", status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> while_fn_inputs;
  while_fn_inputs.push_back(TestScalarTensorHandle(ctx, 10.0f));

  TFE_Op* while_fn_op = TF_ConcreteFunctionMakeCallOp(
      while_fn, while_fn_inputs.data(), while_fn_inputs.size(), status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* while_fn_outputs[1] = {nullptr};
  int num_retvals = 1;

  TFE_Execute(while_fn_op, &while_fn_outputs[0], &num_retvals, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TF_Tensor* result = TFE_TensorHandleResolve(while_fn_outputs[0], status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  ASSERT_EQ(TF_NumDims(result), 0);
  float output_value = *static_cast<float*>(TF_TensorData(result));
  ASSERT_FLOAT_EQ(output_value, 55);  // 10+9+...+1

  TF_DeleteTensor(result);
  TFE_DeleteTensorHandle(while_fn_outputs[0]);
  TFE_DeleteOp(while_fn_op);
  TFE_DeleteTensorHandle(while_fn_inputs[0]);
  TF_DeleteSavedModel(saved_model);
  TF_DeleteStatus(status);
  TFE_DeleteContext(ctx);
}

INSTANTIATE_TEST_SUITE_P(RuntimeAgnosticSavedModelTests, CSavedModelAPITest,
                         ::testing::Bool());

}  // namespace
