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
#include "tensorflow/c/eager/gradient_checker.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/gradients_util.h"
#include "tensorflow/c/eager/mnist_gradients_testutil.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/gradients/array_grad.h"
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

class GradientCheckerTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(errors::OK, s.code()) << s.error_message();
  }
};

Status RegisterGradients(GradientRegistry* registry) {
  TF_RETURN_IF_ERROR(registry->Register("MatMul", MatMulRegisterer));
  TF_RETURN_IF_ERROR(
      registry->Register("SparseSoftmaxCrossEntropyWithLogits",
                         SparseSoftmaxCrossEntropyLossRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Pad", PadRegisterer));
  return Status::OK();
}

void printArr(auto data[], int n) {
 std::cout << std::endl << "[";
 for (int i = 0; i < n - 1; i++) {

   std::cout << data[i] << ", ";

 }
 std::cout << data[n - 1] << "]" << std::endl << std::endl;
}


void printTensor(AbstractTensorHandle* t) {
 TF_Tensor* tensor;
 Status s = GetValue(t, &tensor);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();

 int size = TF_TensorElementCount(tensor);
 float result_data[size] = {0};
 memcpy(&result_data[0], TF_TensorData(tensor),   TF_TensorByteSize(tensor));
 printArr(result_data, size);
 TF_DeleteTensor(tensor);
}

void printTensorInt(AbstractTensorHandle* t) {
  TF_Tensor* tensor;
  Status s = GetValue(t, &tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  int size = TF_TensorElementCount(tensor);
  int result_data[size] = {0};
  memcpy(&result_data[0], TF_TensorData(tensor),TF_TensorByteSize(tensor));
  printArr(result_data, size);
  TF_DeleteTensor(tensor);

}



TEST_P(GradientCheckerTest, TestGradCheckMatMul) {
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
  float B_vals[] = {.5f, -1.0f, 1.0f, 1.0f};
  int64_t B_dims[] = {2, 2};
  int num_dims = 2;
  
  AbstractTensorHandlePtr A =
      GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, num_dims);
  AbstractTensorHandlePtr B =
      GetTensorHandleUtilFloat(ctx.get(), B_vals, B_dims, num_dims);

  std::vector<AbstractTensorHandle*> inputs;
  inputs.push_back(A.get());
  inputs.push_back(B.get());

  AbstractTensorHandle* grad_approx;
  Status s = CalcNumericalGrad(
      ctx.get(), MatMulModel, absl::MakeSpan(inputs), /*input_index=*/0,
      /*use_function=*/!std::get<2>(GetParam()), &grad_approx);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* gt;
  s = GetValue(grad_approx, &gt);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(gt), TF_TensorByteSize(gt));

  float expected_dA[4] = {-.5f, 2.0f, -.5f, 2.0f};
  float tolerance = 1e-2;
  for (int j = 0; j < 4; j++) {
    ASSERT_NEAR(expected_dA[j], result_data[j], tolerance);
  }
  TF_DeleteTensor(gt);
}

TEST_P(GradientCheckerTest, TestGradCheckMul) {
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

  AbstractTensorHandlePtr x;
  {
    AbstractTensorHandle* x_raw = nullptr;
    Status s = ScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  AbstractTensorHandlePtr y;
  {
    AbstractTensorHandle* y_raw = nullptr;
    Status s = ScalarTensorHandle(ctx.get(), 7.0f, &y_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  // Will perform z = x*y.
  // dz/dx = y

  std::vector<AbstractTensorHandle*> inputs;
  inputs.push_back(x.get());
  inputs.push_back(y.get());
  AbstractTensorHandle* g;

  Status s = CalcNumericalGrad(ctx.get(), MulModel, absl::MakeSpan(inputs),
                               /*input_index=*/0,
                               /*use_function=*/!std::get<2>(GetParam()), &g);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* gt;
  s = GetValue(g, &gt);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  float result_data[1] = {0};
  memcpy(&result_data[0], TF_TensorData(gt), TF_TensorByteSize(gt));

  ASSERT_NEAR(result_data[0], 7.0f, /*abs_error=*/1e-2);
  TF_DeleteTensor(gt);
}

TEST_P(GradientCheckerTest, TestGradCheckSoftmax) {
  bool use_function = !std::get<2>(GetParam());
  if (use_function) {
    // TODO(b/168850692): Enable this.
    GTEST_SKIP() << "Can't take gradient of "
                    "SparseSoftmaxCrossEntropyWithLogits in tracing mode.";
  }
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);

  /** Test to show how to use this API with analytical gradients:
   *
   *  We have `SoftmaxLossGradModel`, which is a wrapper for the
   *  Softmax analytical gradient found in c/experimental/nn_grads.
   *
   *  We will use the GradientChecker by applying finite differences
   *  to the forward pass wrapped in `SoftmaxModel` and verify that
   *  both the analytical and numerical gradients are relatively
   *  close.
   *
   */

  AbstractContextPtr ctx;
  {
    AbstractContext* ctx_raw = nullptr;
    Status s =
        BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  // X = scores
  float X_vals[] = {1.0f, 2.0f, 3.0f, -5.0f, -4.0f, -3.0f, 2.0f, 0.0f, 1.0f};
  int64_t X_dims[] = {3, 3};
  int num_dims = 2;
  AbstractTensorHandlePtr X =
      GetTensorHandleUtilFloat(ctx.get(), X_vals, X_dims, num_dims);

  // y = labels
  int y_vals[] = {1, 0, 1};
  int64_t y_dims[] = {3};
  num_dims = sizeof(y_dims) / sizeof(y_dims[0]);
  AbstractTensorHandlePtr y =
      GetTensorHandleUtilInt(ctx.get(), y_vals, y_dims, num_dims);

  GradientRegistry registry;
  Status s = RegisterGradients(&registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  std::vector<AbstractTensorHandle*> inputs;
  inputs.push_back(X.get());
  inputs.push_back(y.get());

  // Run analytical gradient and get its data.
  std::vector<AbstractTensorHandle*> outputs(2);
  s = RunModel(SoftmaxLossGradModel, ctx.get(), absl::MakeSpan(inputs),
               absl::MakeSpan(outputs),
               /*use_function=*/!std::get<2>(GetParam()), registry);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* dX_tensor;
  s = GetValue(outputs[0], &dX_tensor);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  float danalytical[9] = {0};  // Contains data from analytical gradient.
  memcpy(&danalytical[0], TF_TensorData(dX_tensor),
         TF_TensorByteSize(dX_tensor));

  // Run numerical gradient approximation using the GradientChecker API.
  AbstractTensorHandle* g;  // Will contain numerical approximation data.
  s = CalcNumericalGrad(ctx.get(), SoftmaxModel, absl::MakeSpan(inputs),
                        /*input_index=*/0,
                        /*use_function=*/!std::get<2>(GetParam()), &g);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  TF_Tensor* gt;
  s = GetValue(g, &gt);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  float dnumerical[9] = {0};
  memcpy(&dnumerical[0], TF_TensorData(gt), TF_TensorByteSize(gt));

  // Now compare the two implementations:
  for (int j = 0; j < 9; j++) {
    ASSERT_NEAR(dnumerical[j], danalytical[j], /*abs_error=*/1e-2);
  }

  // Only Unref() first output as 2nd is nullptr grad for labels
  outputs[0]->Unref();
  TF_DeleteTensor(dX_tensor);
  TF_DeleteTensor(gt);
}

TEST_P(GradientCheckerTest, TestPadGrad) {
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
 
 float A_vals[] = {2.0f, 1.0f, 1.0f, 3.0f};
 
 int64_t A_dims[] = {2, 2};  // {N, height, width, in_channels}
 int num_dims = 2;
 
 AbstractTensorHandlePtr A =
     GetTensorHandleUtilFloat(ctx.get(), A_vals, A_dims, num_dims);

 float B_vals[] = {5, 6, 7, 8};
 
 int64_t B_dims[] = {2, 2};  // {N, height, width, in_channels}

 
 AbstractTensorHandlePtr B =
     GetTensorHandleUtilFloat(ctx.get(), B_vals, B_dims, num_dims);
 
 int pad_vals[] = {1, 2,  // [left_pad, right_pad] per dimension
                   3, 4};
 int64_t pad_dims[] = {2, 2};
 
 AbstractTensorHandlePtr paddings =
     GetTensorHandleUtilInt(ctx.get(), pad_vals, pad_dims, num_dims);
 
 
 GradientRegistry registry;
 Status s = RegisterGradients(&registry);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
 std::vector<AbstractTensorHandle*> outputs(3);  // [pad_out, dA, dpaddings (null)]
 s = RunModel(PadGradModel, ctx.get(), {A.get(), paddings.get()}, absl::MakeSpan(outputs),
              /*use_function=*/!std::get<2>(GetParam()), registry);
 ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 AbstractTensorHandle* pad_out = outputs[0];
 printTensor(outputs[0]);
 printTensor(outputs[1]);
 
 // s = ops::Shape(ctx.get(), {outputs[0]}, absl::MakeSpan(outputs), "shape_out");
 // ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
 // printTensorInt(outputs[0], 2); // should be [4,4]
 
//  s = ops::Shape(ctx.get(), {A.get()}, absl::MakeSpan(outputs), "shape_out");
//  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
//  AbstractTensorHandle* orig_shape = outputs[0];
//  printTensorInt(orig_shape, 2); // should be [2,2]
 
 
//  int slice_vals[] = {pad_vals[0], pad_vals[2]};
//  int64_t slice_dims[] = {2};
 
//  AbstractTensorHandlePtr slice_begin =
//      GetTensorHandleUtilInt(ctx.get(), slice_vals, slice_dims, 1);
 
//   s = ops::Slice(ctx.get(), {pad_out, slice_begin.get(), orig_shape}, absl::MakeSpan(outputs), "slice_out");
//  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
//  AbstractTensorHandle* slice_out = outputs[0];
 
//  printTensor(slice_out);
//  s = ops::Shape(ctx.get(), {slice_out}, absl::MakeSpan(outputs), "shape_out");
//  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
 
//  printTensorInt(outputs[0], 2); // shoudl be 2,2



// std::cout << "rank";
// printTensorInt(outputs[0], 2); // shoudl be 2, 2

// AbstractTensorHandlePtr zero =  GetScalarTensorHandleUtilInt(ctx.get(), 0);

// std::cout << "concatting";
//  s = ops::ConcatV2(ctx.get(), {A.get(), B.get()}, {zero.get()}, absl::MakeSpan(outputs), "cc");
//  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
//   AbstractTensorHandle* cc = outputs[0];
// printTensor(outputs[0]);

//  s = ops::Shape(ctx.get(), {cc}, absl::MakeSpan(outputs), "shape_out");
//  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
// printTensorInt(outputs[0], 2);

// HEREEEE
    // s = ops::Rank(ctx.get(), {A.get()}, absl::MakeSpan(outputs), "ranker");
    // ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    // AbstractTensorHandle* Rank_x = outputs[0];

    // AbstractTensorHandlePtr zero = GetScalarTensorHandleUtilInt(ctx.get(), 0);
    // AbstractTensorHandlePtr one = GetScalarTensorHandleUtilInt(ctx.get(), 1);

    // s = ops::ConcatV2(ctx.get(), {Rank_x, one.get()}, {zero.get()},
    //                    absl::MakeSpan(outputs), "cocat");
    // ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    // AbstractTensorHandle* size = outputs[0];
    // std::cout << "slizzy";
    // printTensorInt(size);


    // s = ops::Shape(ctx.get(), {size}, absl::MakeSpan(outputs), "shape_out");
    // ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    // AbstractTensorHandle* size_shape = outputs[0];
    // std::cout << "slizzy shape";
    // printTensorInt(size_shape);

    // int b_vals[] = {0, 0};
    // int64_t b_dims[] = {2};
    // AbstractTensorHandlePtr beginners = GetTensorHandleUtilInt(ctx.get(), b_vals, b_dims, 1);

    //  s = ops::Slice(ctx.get(), {paddings.get(), beginners.get(), size}, absl::MakeSpan(outputs), "slice_out");
    //  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    //  AbstractTensorHandle* firstcol = outputs[0];
    // std::cout << "firstcol";
    // printTensorInt(firstcol);

    // s = ops::Shape(ctx.get(), {firstcol}, absl::MakeSpan(outputs), "shape_out");
    // ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    // AbstractTensorHandle* fc_shape = outputs[0];
    // std::cout << "firstcol shape";
    // printTensorInt(fc_shape);

    // AbstractTensorHandlePtr minus_one = GetScalarTensorHandleUtilInt(ctx.get(), -1);
    // s = ops::Reshape(ctx.get(), {firstcol, minus_one.get()}, absl::MakeSpan(outputs),
    //                   "resh");
    // ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    // AbstractTensorHandle* reshaped_fc = outputs[0];
    // std::cout << "firstcol_reshaped (should be same as above)";
    // printTensorInt(reshaped_fc);

    // s = ops::Shape(ctx.get(), {reshaped_fc}, absl::MakeSpan(outputs), "shape_out");
    // ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    // AbstractTensorHandle* rfc_shape = outputs[0];
    // std::cout << "firstcol_reshaped shape (should be [2])";
    // printTensorInt(rfc_shape);

 
}


#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, GradientCheckerTest,
    ::testing::Combine(::testing::Values("graphdef"),
                       /*tfrt*/ ::testing::Values(false),
                       /*executing_eagerly*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
