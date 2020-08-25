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
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/mnist_gradients_testutil.h"
#include "tensorflow/c/eager/resnet_gradients_testutil.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_SetTracingImplementation(std::get<0>(GetParam()));
  }
};

Status RegisterGradients(GradientRegistry* registry) {
  TF_RETURN_IF_ERROR(registry->Register("Add", AddRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("AddV2", AddRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Exp", ExpRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("MatMul", MatMulRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Relu", ReluRegisterer));
  TF_RETURN_IF_ERROR(
      registry->Register("SparseSoftmaxCrossEntropyWithLogits",
                         SparseSoftmaxCrossEntropyLossRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("BiasAdd", BiasAddRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Conv2D", Conv2DRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Sub", SubRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Neg", NegRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("FusedBatchNormV3", FusedBatchNormV3Registerer));
  TF_RETURN_IF_ERROR(registry->Register("Mul", MulRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Log1p", Log1pRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("DivNoNan", DivNoNanRegisterer));
  return Status::OK();
}

// ========================= Test Util Functions ==============================

// Get a scalar TensorHandle with given value
Status TestScalarTensorHandle(AbstractContext* ctx, float value,
                              AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager = TestScalarTensorHandle(eager_ctx, value);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return Status::OK();
}

// Get a Matrix TensorHandle with given float values and dimensions
Status TestTensorHandleWithDimsFloat(AbstractContext* ctx, float data[],
                                     int64_t dims[], int num_dims,
                                     AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager =
      TestTensorHandleWithDimsFloat(eager_ctx, data, dims, num_dims);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return Status::OK();
}

// Get a Matrix TensorHandle with given int values and dimensions
Status TestTensorHandleWithDimsInt(AbstractContext* ctx, int data[],
                                   int64_t dims[], int num_dims,
                                   AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager =
      TestTensorHandleWithDimsInt(eager_ctx, data, dims, num_dims);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return Status::OK();
}

Status GetValue(AbstractTensorHandle* t, TF_Tensor** result_tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(wrap(t), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  *result_tensor = TFE_TensorHandleResolve(result_t, status.get());
  return Status::OK();
}

AbstractTensorHandlePtr GetTensorHandleUtilFloat(AbstractContext* ctx,
                                                 float vals[], int64_t dims[],
                                                 int num_dims) {
  AbstractTensorHandlePtr A;
  AbstractTensorHandle* a_raw = nullptr;
  Status s = TestTensorHandleWithDimsFloat(ctx, vals, dims, num_dims, &a_raw);
  A.reset(a_raw);
  return A;
}

AbstractTensorHandlePtr GetTensorHandleUtilInt(AbstractContext* ctx, int vals[],
                                               int64_t dims[], int num_dims) {
  AbstractTensorHandlePtr A;
  AbstractTensorHandle* a_raw = nullptr;
  Status s = TestTensorHandleWithDimsInt(ctx, vals, dims, num_dims, &a_raw);
  A.reset(a_raw);
  return A;
}

// Verifies that given tensor matches expected values
void VerifyResult(AbstractTensorHandle* to_check, float* expected_vals, int num_elems) {

  TF_Tensor* tensor;
  Status s = GetValue(to_check, &tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float result_data[num_elems] = {0};
  memcpy(&result_data[0], TF_TensorData(tensor),
         TF_TensorByteSize(tensor));

  float tolerance = 1e-3;
  for (int j = 0; j < num_elems; j++) {
    ASSERT_NEAR(result_data[j], expected_vals[j], tolerance);
  }

  TF_DeleteTensor(tensor);
}

void printArr(auto data[], int n) {
 std::cout << std::endl << "[";
 for (int i = 0; i < n - 1; i++) {
   std::cout << data[i] << ", ";
 }
 std::cout << data[n - 1] << "]" << std::endl << std::endl;
}


void printTensor(AbstractTensorHandle* t, int size) {
 TF_Tensor* tensor;
 Status s = GetValue(t, &tensor);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
 float result_data[size] = {0};
 memcpy(&result_data[0], TF_TensorData(tensor), TF_TensorByteSize(tensor));
 printArr(result_data, size);
 
 TF_DeleteTensor(tensor);
}
 
void printTensorInt(AbstractTensorHandle* t, int size) {
 TF_Tensor* tensor;
 Status s = GetValue(t, &tensor);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
 int result_data[size] = {0};
 memcpy(&result_data[0], TF_TensorData(tensor), TF_TensorByteSize(tensor));
 printArr(result_data, size);
 
 TF_DeleteTensor(tensor);
}

// =========================== Start Tests ================================

TEST_P(CppGradients, TestBiasAddGrad) {
 std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
     TF_NewStatus(), TF_DeleteStatus);
 AbstractContextPtr ctx;
 {
   AbstractContext* ctx_raw = nullptr;
   Status s =
       BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
   ASSERT_EQ(errors::OK, s.code()) << s.error_message();
   ctx.reset(ctx_raw);
 }
 
 float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
 int64_t A_dims[] = {2, 2, 1, 2};
 int num_dims = 4;
 AbstractTensorHandlePtr A =
     GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, num_dims);
 
 float bias_vals[] = {-1.0f, 2.0f};
 int64_t bias_dims[] = {2};
 num_dims = 1;
 AbstractTensorHandlePtr bias =
     GetTensorHandleUtilFloat(ctx.get(), bias_vals, bias_dims, num_dims);
 
 GradientRegistry registry;
 Status s = RegisterGradients(&registry);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
 std::vector<AbstractTensorHandle*> outputs(2);
 s = RunModel(BiasAddGradModel, ctx.get(), {A.get(), bias.get()},
              absl::MakeSpan(outputs),
              /*use_function=*/!std::get<2>(GetParam()), registry);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();

// Verify grad for A
 TF_Tensor* out_tensor;
 s = GetValue(outputs[0], &out_tensor);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
 float result_data[8] = {0};
 memcpy(&result_data[0], TF_TensorData(out_tensor),
        TF_TensorByteSize(out_tensor));
 
 float expected_out_1[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
 float tolerance = 1e-3;
 for (int j = 0; j < 8; j++) {
   ASSERT_NEAR(result_data[j], expected_out_1[j], tolerance);
 }
 
 // Verify grad for bias
 s = GetValue(outputs[1], &out_tensor);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
 memcpy(&result_data[0], TF_TensorData(out_tensor),
        TF_TensorByteSize(out_tensor));
 
 float expected_out_2[2] = {4.0f, 4.0f};
 for (int j = 0; j < 2; j++) {
   ASSERT_NEAR(result_data[j], expected_out_2[j], tolerance);
 }
 
 outputs[0]->Unref();
 outputs[1]->Unref();
 TF_DeleteTensor(out_tensor);
}

TEST_P(CppGradients, TestConv2DGrad) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }
  
  float A_vals[] = {2.0f, 1.0f, 2.0f, 0.0f, 1.0f,
                    1.0f, 3.0f, 2.0f, 2.0f, 3.0f,
                    1.0f, 1.0f, 3.0f, 3.0f, 0.0f,
                    2.0f, 2.0f, 0.0f, 1.0f, 1.0f,
                    0.0f, 0.0f, 3.0f, 1.0f, 2.0f};
  int64_t A_dims[] = {1, 5, 5, 1}; // {N, height, width, in_channels}
  int num_dims = 4;
  
  AbstractTensorHandlePtr A =
      GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, num_dims);
  
  float filter_vals[] = {/*f1 =*/2.0f, 0.1f,
                                 3.0f, 0.2f,
                         /*f2 =*/0.0f, 0.3f,
                                 1.0f, 0.4f};
  
  int64_t filter_dims[] = {2, 2, 1, 2}; // {fil_h, fil_w, in_channels, out_channels}
  num_dims = 4;
  
  AbstractTensorHandlePtr filter =
      GetTensorHandleUtilFloat(ctx.get(), filter_vals, filter_dims, num_dims);
   
  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  
  std::vector<AbstractTensorHandle*> outputs(3); // [conv_out, dx, dfilter]
  s = RunModel(Conv2DGradModel, ctx.get(), {A.get(), filter.get()},
                absl::MakeSpan(outputs),
                /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  
  // check conv_out
  float expected_conv_out[] = {10, 1.9, 10, 2.2, 6, 1.6,
                               6, 2, 2, 1, 12, 1.4, 15, 
                               2.2, 13, 2.7, 13, 1.7, 6, 
                               0.3, 7, 1.7, 11, 1.3, 16, 1.3,
                               7, 1, 0, 0.3, 10, 0.6, 7, 1.4,
                               4, 1.5, 7, 1.4, 2, 0.7, 0, 0, 9,
                               0.6, 9, 0.5, 8, 0.5, 4, 0.2};
  VerifyResult(outputs[0], expected_conv_out, 50);
  
  // dInput = dA
  float expected_dInput[] = {2.1, 5.3, 5.3, 5.3, 5.3, 2.4,
                             7, 7, 7, 7, 2.4, 7, 7, 7, 7, 2.4,
                             7, 7, 7, 7, 2.4, 7, 7, 7, 7};
  VerifyResult(outputs[1], expected_dInput, 25);
  
  // dfilter
  float expected_dfilter[] = {37, 37, 31, 31, 31, 31, 27, 27};
  VerifyResult(outputs[2], expected_dfilter, 8);
 }

TEST_P(CppGradients, TestMathGrads) {
  /** Test for various math grads that include:
    * Sub, Mul, Neg, Log1p, and Div.
    */

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  float A_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t A_dims[] = {2, 2};
  float B_vals[] = {.5f, -1.0f, 2.0f, 6.0f};
  int64_t B_dims[] = {2, 2};
  int num_dims = 2;

  AbstractTensorHandlePtr A =
      GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, num_dims);
  AbstractTensorHandlePtr B =
      GetTensorHandleUtilFloat(ctx.get(), B_vals, B_dims, num_dims);

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // Check grads for Mul: out = A*B 
  std::vector<AbstractTensorHandle*> outputs(2);
  s = RunModel(MulGradModel, ctx.get(), {A.get(), B.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // dA = B_vals
  VerifyResult(outputs[0], B_vals, 4);

  // dB = A_vals
  VerifyResult(outputs[1], A_vals, 4);


  // Check grads for Sub: out = A - B 
  s = RunModel(SubGradModel, ctx.get(), {A.get(), B.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // dA
  float expected_dA_Sub[] = {1.0f, 1.0f, 1.0f, 1.0f};
  VerifyResult(outputs[0], expected_dA_Sub, 4);

  // dB
  float expected_dB_Sub[] = {-1.0f, -1.0f, -1.0f, -1.0f};
  VerifyResult(outputs[1], expected_dB_Sub, 4);
  
  // Check grads for Div: out = A / B
  s = RunModel(DivGradModel, ctx.get(), {A.get(), B.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  //dA
  float expected_dA_Div[] = {2.0f, -1.0f, 0.5f, 0.16667f};
  VerifyResult(outputs[0], expected_dA_Div, 4);

  // dB
  float expected_dB_Div[] = {-4.0f, -2.0f, -0.75f, -0.11111f};
  VerifyResult(outputs[1], expected_dB_Div, 4);

  // Free 2nd output as no longer needed.
  outputs[1]->Unref();

  // Check Grads for Neg: out = -A 
  outputs.resize(1);
  s = RunModel(NegGradModel, ctx.get(), {A.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // dA
  float expected_dA_Neg[] = {-1.0f, -1.0f, -1.0f, -1.0f};
  VerifyResult(outputs[0], expected_dA_Neg, 4);

  // Check Grads for Log1p: out = tf.math.log1p(A) 
  s = RunModel(Log1pGradModel, ctx.get(), {A.get()},
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  // dA
  float expected_dA_Log1p[] = {0.5f, 0.3333f, 0.25f, 0.2f};
  VerifyResult(outputs[0], expected_dA_Log1p, 4);

  outputs[0]->Unref();
}

//  Add tests for
//   - BatchnormGrad
//   - fix Conv2D attrs

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow