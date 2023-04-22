/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/c/experimental/ops/math_ops.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: Mul()
// Summary: Returns x * y element-wise.
//
// Description:
//   *NOTE*: `Multiply` supports broadcasting. More about broadcasting
//   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
Status Mul(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Mul", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}

// Op: Conj()
// Summary: Returns the complex conjugate of a complex number.
//
// Description:
//   Given a tensor `input` of complex numbers, this operation returns a tensor
//   of complex numbers that are the complex conjugate of each element in
//   `input`. The complex numbers in `input` must be of the form \\(a + bj\\),
//   where *a* is the real part and *b* is the imaginary part.
//
//   The complex conjugate returned by this operation is of the form \\(a -
//   bj\\).
//
//   For example:
//
//   ```
//   # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
//   tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
//   ```
Status Conj(AbstractContext* ctx, AbstractTensorHandle* const input,
            AbstractTensorHandle** output, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Conj", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: AddV2()
// Summary: Returns x + y element-wise.
//
// Description:
//   *NOTE*: `Add` supports broadcasting. `AddN` does not. More about
//   broadcasting
//   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
Status AddV2(AbstractContext* ctx, AbstractTensorHandle* const x,
             AbstractTensorHandle* const y, AbstractTensorHandle** z,
             const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("AddV2", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}

// Op: MatMul()
// Summary: Multiply the matrix "a" by the matrix "b".
//
// Description:
//   The inputs must be two-dimensional matrices and the inner dimension of
//   "a" (after being transposed if transpose_a is true) must match the
//   outer dimension of "b" (after being transposed if transposed_b is
//   true).
//
//   *Note*: The default kernel implementation for MatMul on GPUs uses
//   cublas.
Status MatMul(AbstractContext* ctx, AbstractTensorHandle* const a,
              AbstractTensorHandle* const b, AbstractTensorHandle** product,
              bool transpose_a, bool transpose_b, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("MatMul", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(a));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(b));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("transpose_a", transpose_a));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("transpose_b", transpose_b));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(product, 1), &num_retvals);
}

// Op: Neg()
// Summary: Computes numerical negative value element-wise.
//
// Description:
//   I.e., \\(y = -x\\).
Status Neg(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle** y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Neg", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

// Op: Sum()
// Summary: Computes the sum of elements across dimensions of a tensor.
//
// Description:
//   Reduces `input` along the dimensions given in `axis`. Unless
//   `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry
//   in `axis`. If `keep_dims` is true, the reduced dimensions are retained with
//   length 1.
Status Sum(AbstractContext* ctx, AbstractTensorHandle* const input,
           AbstractTensorHandle* const reduction_indices,
           AbstractTensorHandle** output, bool keep_dims, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sum", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(reduction_indices));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("keep_dims", keep_dims));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: Sub()
// Summary: Returns x - y element-wise.
//
// Description:
//   *NOTE*: `Subtract` supports broadcasting. More about broadcasting
//   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
Status Sub(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sub", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}

// Op: Div()
// Summary: Returns x / y element-wise.
//
// Description:
//   *NOTE*: `Div` supports broadcasting. More about broadcasting
//   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
Status Div(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Div", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}

// Op: DivNoNan()
// Summary: Returns 0 if the denominator is zero.
//
// Description:
//
//   *NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
//   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
Status DivNoNan(AbstractContext* ctx, AbstractTensorHandle* const x,
                AbstractTensorHandle* const y, AbstractTensorHandle** z,
                const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("DivNoNan", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}

// Op: Exp()
// Summary: Computes exponential of x element-wise.  \\(y = e^x\\).
//
// Description:
//     This function computes the exponential of every element in the input
//     tensor. i.e. `exp(x)` or `e^(x)`, where `x` is the input tensor. `e`
//     denotes Euler's number and is approximately equal to 2.718281. Output is
//     positive for any real input.
//
//     ```python
//     x = tf.constant(2.0)
//     tf.math.exp(x) ==> 7.389056
//
//     x = tf.constant([2.0, 8.0])
//     tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)
//     ```
//
//     For complex numbers, the exponential value is calculated as follows:
//
//     ```
//     e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
//     ```
//
//     Let's consider complex number 1+1j as an example.
//     e^1 * (cos 1 + i sin 1) = 2.7182818284590 * (0.54030230586+0.8414709848j)
//
//     ```python
//     x = tf.constant(1 + 1j)
//     tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
//     ```
Status Exp(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle** y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Exp", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

// Op: Sqrt()
// Summary: Computes square root of x element-wise.
//
// Description:
//   I.e., \\(y = \sqrt{x} = x^{1/2}\\).
Status Sqrt(AbstractContext* ctx, AbstractTensorHandle* const x,
            AbstractTensorHandle** y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sqrt", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

// Op: SqrtGrad()
// Summary: Computes the gradient for the sqrt of `x` wrt its input.
//
// Description:
//   Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
//   is the corresponding input gradient.
Status SqrtGrad(AbstractContext* ctx, AbstractTensorHandle* const y,
                AbstractTensorHandle* const dy, AbstractTensorHandle** z,
                const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("SqrtGrad", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(dy));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}

// Op: Log1p()
// Summary: Computes natural logarithm of (1 + x) element-wise.
//
// Description:
//   I.e., \\(y = \log_e (1 + x)\\).
//
//   Example:
//
//   ```python
//   x = tf.constant([0, 0.5, 1, 5])
//   tf.math.log1p(x) ==> [0., 0.4054651, 0.6931472, 1.7917595]
//   ```
Status Log1p(AbstractContext* ctx, AbstractTensorHandle* const x,
             AbstractTensorHandle** y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Log1p", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
