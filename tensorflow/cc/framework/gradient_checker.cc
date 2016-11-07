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

#include "tensorflow/cc/framework/gradient_checker.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

// TODO(andydavis) Support returning relative error (as opposed to max error)
// between theoretical and numerical jacobians:
//   fabs(jac_t - jac_n) / max(fabs(jac_t), fabs(jac_n))

// TODO(andydavis) Vectorize and/or multi-thread Jacobian computations if
// performance becomes an issue.

template <typename T>
Status ComputeTheoreticalJacobianTranspose(
    const Scope& scope, const ops::Output& x, const TensorShape& x_shape,
    const Tensor& x_data, const ops::Output& y, const TensorShape& y_shape,
    Tensor* jacobian_t) {
  // Call AddSymbolicGradients to get 'dx' (we will feed 'dy').
  auto dy = Cast(scope, Const(scope, 1.0, y_shape), x.type());
  std::vector<ops::Output> outputs;
  TF_RETURN_IF_ERROR(AddSymbolicGradients(scope, {y}, {x}, {dy}, &outputs));
  auto dx = outputs[0];

  // Initialize 'dy_data' to zeros.
  Tensor dy_data(y.type(), y_shape);
  auto dy_data_flat = dy_data.flat<T>();
  dy_data_flat.setZero();

  // Compute the theoretical Jacobian one row at a time by back propagating '1.0'
  // for each element of 'dy', while holding all other elements of 'dy' at zero.
  ClientSession session(scope);
  std::vector<Tensor> dxout;
  const int64 x_size = x_shape.num_elements();
  const int64 dy_size = y_shape.num_elements();
  auto jacobian = jacobian_t->matrix<T>();
  for (int c = 0; c < dy_size; ++c) {
    dy_data_flat(c) = 1.0;

    TF_RETURN_IF_ERROR(session.Run({{x, x_data}, {dy, dy_data}}, {dx}, &dxout));

    auto dx_flat = dxout[0].flat<T>();
    for (int r = 0; r < x_size; ++r) {
      jacobian(r, c) = dx_flat(r);
    }

    dy_data_flat(c) = 0.0;
  }
  return Status::OK();
}

template <typename T>
Tensor DeepCopyTensor(const Tensor& t) {
  Tensor ret = Tensor(t.dtype(), t.shape());
  auto ret_flat = ret.flat<T>();
  auto t_flat = t.flat<T>();
  for (int i = 0; i < t.NumElements(); i++) {
    ret_flat(i) = t_flat(i);
  }
  return ret;
}

template <typename T>
Status EvaluateGraph(ClientSession& session, const ops::Output& x,
                     const ops::Output& y, const Tensor* x_data,
                     Tensor* y_data) {
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run({{x, *x_data}}, {y}, &outputs));
  if (outputs[0].SharesBufferWith(*x_data)) {
    *y_data = DeepCopyTensor<T>(outputs[0]);
  } else {
    *y_data = outputs[0];
  }
  return Status::OK();
}

template <typename T>
Status ComputeNumericJacobianTranspose(const Scope& scope, const ops::Output& x,
                                       const TensorShape& x_shape,
                                       const ops::Output& y,
                                       const TensorShape& y_shape,
                                       const T delta, Tensor* x_data,
                                       Tensor* jacobian_t) {
  const int64 x_size = x_shape.num_elements();
  const int64 y_size = y_shape.num_elements();
  // Create copies of x_data since the underlying buffer of the input Tensor is
  // not copied for some operations (i.e. Identity), which can lead to incorrect
  // results for the centered difference calculation.
  auto x_data_flat = x_data->flat<T>();

  // Compute the numeric Jacobian one column at a time by perturbing each
  // element of 'x_data' (positively and negatively) by 'delta', and
  // updating the jacobian with the centered difference.
  ClientSession session(scope);
  auto jacobian = jacobian_t->matrix<T>();
  for (int r = 0; r < x_size; ++r) {
    // Store current value of 'x' at 'r'.
    T v = x_data_flat(r);
    // Evaluate at positive delta.
    x_data_flat(r) = v + delta;
    Tensor y_pos;
    TF_RETURN_IF_ERROR(EvaluateGraph<T>(session, x, y, x_data, &y_pos));
    // Evaluate at negative delta.
    x_data_flat(r) = v - delta;
    Tensor y_neg;
    TF_RETURN_IF_ERROR(EvaluateGraph<T>(session, x, y, x_data, &y_neg));
    // Compute element-wise centered difference and store in Jacobian.
    auto y_pos_flat = y_pos.flat<T>();
    auto y_neg_flat = y_neg.flat<T>();
    const T scale = 2 * delta;
    for (int c = 0; c < y_size; ++c) {
      jacobian(r, c) = (y_pos_flat(c) - y_neg_flat(c)) / scale;
    }
    // Restore pre-perturbation value.
    x_data_flat(r) = v;
  }
  return Status::OK();
}

template <typename T>
Status ComputeGradientErrorInternal(const Scope& scope, const ops::Output& x,
                                    const TensorShape& x_shape,
                                    const ops::Output& y,
                                    const TensorShape& y_shape, Tensor* x_data,
                                    T* max_error) {
  const int64 x_size = x_shape.num_elements();
  const int64 y_size = y_shape.num_elements();

  // Initialize theoretical Jacobian to zeros.
  Tensor jacobian_t(x.type(), {x_size, y_size});
  auto jacobian_t_flat = jacobian_t.flat<T>();
  jacobian_t_flat.setZero();

  // Compute theoretical Jacobian.
  TF_RETURN_IF_ERROR(ComputeTheoreticalJacobianTranspose<T>(
      scope, x, x_shape, *x_data, y, y_shape, &jacobian_t));

  // Initialize numeric Jacobian to zeros.
  Tensor jacobian_n(x.type(), {x_size, y_size});
  auto jacobian_n_flat = jacobian_n.flat<T>();
  jacobian_n_flat.setZero();

  // Compute numeric Jacobian.
  TF_RETURN_IF_ERROR(ComputeNumericJacobianTranspose<T>(
      scope, x, x_shape, y, y_shape, 1e-3, x_data, &jacobian_n));

  // Compute the maximum error between theoretical and numeric Jacobians.
  *max_error = 0.0;
  auto jac_t = jacobian_t.matrix<T>();
  auto jac_n = jacobian_n.matrix<T>();
  for (int r = 0; r < x_size; ++r) {
    for (int c = 0; c < y_size; ++c) {
      *max_error = std::max(*max_error, std::fabs(jac_t(r, c) - jac_n(r, c)));
    }
  }
  return Status::OK();
}

}  // namespace

template <typename T>
Status ComputeGradientError(const Scope& scope, const ops::Output& x,
                            const TensorShape& x_shape, const ops::Output& y,
                            const TensorShape& y_shape, T* max_error) {
  // Initialize 'x_data' to random values.
  Tensor x_data(x.type(), x_shape);
  auto x_data_flat = x_data.flat<T>();
  x_data_flat.setRandom();
  // Compute gradient error.
  return ComputeGradientErrorInternal(scope, x, x_shape, y, y_shape, &x_data,
                                      max_error);
}

template <typename T>
Status ComputeGradientError(const Scope& scope, const ops::Output& x,
                            const Tensor& x_init_value, const ops::Output& y,
                            const TensorShape& y_shape, T* max_error) {
  // Initialize 'x_data' from 'x_init_value'.
  Tensor x_data(x_init_value);
  // Compute gradient error.
  return ComputeGradientErrorInternal(scope, x, x_data.shape(), y, y_shape,
                                      &x_data, max_error);
}

#define INSTANTIATE_GRAD_ERR_TYPE(T)                                        \
  template Status ComputeGradientError<T>(                                  \
      const Scope& scope, const ops::Output& x, const TensorShape& x_shape, \
      const ops::Output& y, const TensorShape& y_shape, T* max_error);      \
  template Status ComputeGradientError<T>(                                  \
      const Scope& scope, const ops::Output& x, const Tensor& x_init_value, \
      const ops::Output& y, const TensorShape& y_shape, T* max_error);

INSTANTIATE_GRAD_ERR_TYPE(float);
INSTANTIATE_GRAD_ERR_TYPE(double);

}  // namespace tensorflow
