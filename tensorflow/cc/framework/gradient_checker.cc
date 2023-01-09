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

#include <algorithm>
#include <utility>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

// TODO(andydavis) Support returning relative error (as opposed to max error)
// between theoretical and numerical jacobians:
//   fabs(jac_t - jac_n) / max(fabs(jac_t), fabs(jac_n))

// TODO(andydavis) Vectorize and/or multi-thread Jacobian computations if
// performance becomes an issue.

// BaseUnitsForType provides a list of typed unit values for each basis in the
// requested type.
// When T is real,
// BaseUnitsForType<T>::values() is just a single-entry vector [1]
// When T is complex,
// BaseUnitsForType<T>::values() is a two-entry vector [1, i] - the unit
// values in each of its two bases.
template <typename T>
struct BaseUnitsForType {};  // Specializations below

// Template specialization for BaseUnitsForType
#define SET_BASE_UNITS_FOR_TYPE(TYPE, INIT)                         \
  template <>                                                       \
  struct BaseUnitsForType<TYPE> {                                   \
    static const std::vector<TYPE>& values() {                      \
      static std::vector<TYPE>* units = new std::vector<TYPE> INIT; \
      return *units;                                                \
    }                                                               \
  }

SET_BASE_UNITS_FOR_TYPE(float, {1});
SET_BASE_UNITS_FOR_TYPE(double, {1});
SET_BASE_UNITS_FOR_TYPE(complex64, ({{1, 0}, {0, 1}}));
SET_BASE_UNITS_FOR_TYPE(complex128, ({{1, 0}, {0, 1}}));

// SetJacobian sets the jacobian value at the provided row and column from a
// tensor entry with type T.
// When T is real, this is a simple assignment that casts the entry into the
// jacobian type.
// When T is complex, it assigns the real and complex values to successive rows
// or columns in the matrix depending on the expand_by_row parameter
template <typename T, typename JAC_T>
typename std::enable_if<std::is_floating_point<T>::value>::type SetJacobian(
    typename TTypes<JAC_T>::Matrix* jacobian, const int row, const int col,
    const T& value, const bool expand_by_row) {
  (*jacobian)(row, col) = JAC_T{value};
}

template <typename T, typename JAC_T>
typename std::enable_if<is_complex<T>::value>::type SetJacobian(
    typename TTypes<JAC_T>::Matrix* jacobian, const int row, const int col,
    const T& value, const bool expand_by_row) {
  (*jacobian)(row, col) = JAC_T{value.real()};
  if (expand_by_row) {
    (*jacobian)(row + 1, col) = JAC_T{value.imag()};
  } else {
    (*jacobian)(row, col + 1) = JAC_T{value.imag()};
  }
}

// JacobianStride<T>::value holds the number of Jacobian elements needed to
// represent one element of the given type.
// When T is real the stride is 1, and when T is complex the stride is 2.
template <typename T>
struct JacobianStride {};  // Specializations below

#define SET_JACOBIAN_STRIDE(TYPE, VALUE) \
  template <>                            \
  struct JacobianStride<TYPE> {          \
    static constexpr int value = VALUE;  \
  }

SET_JACOBIAN_STRIDE(float, 1);
SET_JACOBIAN_STRIDE(double, 1);
SET_JACOBIAN_STRIDE(complex64, 2);
SET_JACOBIAN_STRIDE(complex128, 2);

template <typename X_T, typename Y_T, typename JAC_T>
Status ComputeTheoreticalJacobianTranspose(
    const Scope& scope, const OutputList& xs,
    const std::vector<TensorShape>& x_shapes,
    const std::vector<Tensor>& x_datas, const OutputList& ys,
    const std::vector<TensorShape>& y_shapes,
    std::vector<Tensor>* jacobian_ts) {
  size_t y_num = y_shapes.size();
  size_t x_num = x_shapes.size();
  // Call AddSymbolicGradients to get 'dxs' (we will feed 'dys').
  OutputList dys;
  dys.reserve(y_shapes.size());
  for (const auto& y_shape : y_shapes) {
    // TODO(suharshs): This currently assumes that all y's are the same type.
    dys.push_back(
        ops::Cast(scope, ops::Const(scope, 1.0, y_shape), ys[0].type()));
  }
  OutputList dxs;
  TF_RETURN_IF_ERROR(AddSymbolicGradients(scope, ys, xs, dys, &dxs));

  // Initialize 'dy_data' to zeros.
  std::vector<Tensor> dy_datas(y_num);
  for (int i = 0; i < y_num; i++) {
    dy_datas[i] = Tensor(ys[i].type(), y_shapes[i]);
    auto dy_data_flat = dy_datas[i].flat<Y_T>();
    dy_data_flat.setZero();
  }

  // Create the feed list.
  ClientSession::FeedType feed_list;
  for (int i = 0; i < x_num; i++) {
    feed_list.insert({xs[i], x_datas[i]});
  }
  for (int i = 0; i < y_num; i++) {
    feed_list.insert({dys[i], dy_datas[i]});
  }

  // x_stride and y_stride are used to calculate the correct jacobian row and
  // column position for a pair of elements at positions r, c within the x and y
  // tensors respectively.
  const int x_stride = JacobianStride<X_T>::value;
  const int y_stride = JacobianStride<Y_T>::value;
  ClientSession session(scope);
  for (int y_idx = 0; y_idx < y_num; y_idx++) {
    auto dy_data_flat = dy_datas[y_idx].flat<Y_T>();
    const int64_t dy_size = y_shapes[y_idx].num_elements();

    // Compute the theoretical Jacobians one row at a time by back propagating
    // '1.0' (or '1' and 'i' if y is complex) for each element of 'dy', while
    // holding all other elements of 'dy' at zero.
    for (int c = 0; c < dy_size; ++c) {
      int unit_dimension = 0;
      for (Y_T unit : BaseUnitsForType<Y_T>::values()) {
        dy_data_flat(c) = unit;

        std::vector<Tensor> dxout;
        TF_RETURN_IF_ERROR(session.Run(feed_list, dxs, &dxout));

        for (int x_idx = 0; x_idx < x_num; x_idx++) {
          if (x_shapes[x_idx] != dxout[x_idx].shape()) {
            return errors::Internal("Gradient for input ", x_idx,
                                    " expected shape ",
                                    x_shapes[x_idx].DebugString(), " but was ",
                                    dxout[x_idx].shape().DebugString());
          }
          const int64_t x_size = x_shapes[x_idx].num_elements();
          auto jacobian = (*jacobian_ts)[x_idx * y_num + y_idx].matrix<JAC_T>();
          auto dx_flat = dxout[x_idx].flat<X_T>();
          for (int r = 0; r < x_size; ++r) {
            SetJacobian<X_T, JAC_T>(&jacobian, r * x_stride,
                                    c * y_stride + unit_dimension, dx_flat(r),
                                    true /* expand_by_row=true */);
          }
        }

        dy_data_flat(c) = Y_T{0};
        unit_dimension++;
      }
    }
  }
  return OkStatus();
}

Status EvaluateGraph(ClientSession* session, const OutputList& xs,
                     const OutputList& ys, std::vector<Tensor>* x_datas,
                     std::vector<Tensor>* y_datas) {
  // Create the feed list.
  ClientSession::FeedType feed_list;
  for (int i = 0; i < x_datas->size(); i++) {
    feed_list.insert({xs[i], (*x_datas)[i]});
  }

  TF_RETURN_IF_ERROR(session->Run(feed_list, ys, y_datas));
  for (int y_idx = 0; y_idx < y_datas->size(); y_idx++) {
    for (int x_idx = 0; x_idx < x_datas->size(); x_idx++) {
      Tensor y_data = (*y_datas)[y_idx];
      if (y_data.SharesBufferWith((*x_datas)[x_idx])) {
        // Create copies of outputs that share a buffer with any inputs since
        // the underlying buffer of the input Tensors are not copied for some
        // operations (i.e. Identity), which can lead to incorrect results for
        // the centered difference calculation.
        (*y_datas)[y_idx] = tensor::DeepCopy(y_data);
      }
    }
  }
  return OkStatus();
}

template <typename X_T, typename Y_T, typename JAC_T>
Status ComputeNumericJacobianTranspose(const Scope& scope, const OutputList& xs,
                                       const std::vector<TensorShape>& x_shapes,
                                       const OutputList& ys,
                                       const std::vector<TensorShape>& y_shapes,
                                       const JAC_T delta,
                                       std::vector<Tensor>* x_datas,
                                       std::vector<Tensor>* jacobian_ts) {
  size_t y_num = y_shapes.size();
  size_t x_num = x_shapes.size();
  // x_stride and y_stride are used to calculate the correct jacobian row and
  // column position for a pair of elements at positions r, c within the x and y
  // tensors respectively.
  const int x_stride = JacobianStride<X_T>::value;
  const int y_stride = JacobianStride<Y_T>::value;

  ClientSession session(scope);
  for (int x_idx = 0; x_idx < x_num; x_idx++) {
    auto x_data_flat = (*x_datas)[x_idx].flat<X_T>();
    const int64_t x_size = x_shapes[x_idx].num_elements();

    // Compute the numeric Jacobian one column at a time by perturbing each
    // element of 'x_data' (positively and negatively) by 'delta', and
    // updating the jacobian with the centered difference. When x_data is
    // complex-valued, we perturb its real and complex parts separately.
    for (int r = 0; r < x_size; ++r) {
      int unit_dimension = 0;
      for (X_T unit : BaseUnitsForType<X_T>::values()) {
        X_T x_delta = unit * X_T{delta};
        // Store current value of 'x' at 'r'.
        X_T v = x_data_flat(r);
        // Evaluate at positive delta.
        x_data_flat(r) = v + x_delta;
        std::vector<Tensor> y_pos;
        TF_RETURN_IF_ERROR(EvaluateGraph(&session, xs, ys, x_datas, &y_pos));
        // Evaluate at negative delta.
        x_data_flat(r) = v - x_delta;
        std::vector<Tensor> y_neg;
        TF_RETURN_IF_ERROR(EvaluateGraph(&session, xs, ys, x_datas, &y_neg));

        for (int y_idx = 0; y_idx < y_num; y_idx++) {
          // Compute element-wise centered difference and store in each
          // Jacobian.
          auto y_pos_flat = y_pos[y_idx].flat<Y_T>();
          auto y_neg_flat = y_neg[y_idx].flat<Y_T>();
          const int64_t y_size = y_shapes[y_idx].num_elements();
          const Y_T scale = 2 * delta;
          auto jacobian = (*jacobian_ts)[x_idx * y_num + y_idx].matrix<JAC_T>();
          for (int c = 0; c < y_size; ++c) {
            SetJacobian<Y_T, JAC_T>(&jacobian, r * x_stride + unit_dimension,
                                    c * y_stride,
                                    (y_pos_flat(c) - y_neg_flat(c)) / scale,
                                    false /* expand_by_row=false */);
          }
        }
        // Restore pre-perturbation value.
        x_data_flat(r) = v;
        unit_dimension++;
      }
    }
  }
  return OkStatus();
}

// The Jacobian is always a real-valued matrix.
// Given y = f(x) for tensors y and x, it contains the derivatives dy_i/dx_j for
// every pair y_i in y and x_j in x.  Note that the Jacobian is defined directly
// over the elements of tensors y and x, and doesn't depend on their shapes.
//
// If x = (x_1, x_2, ..., x_m) and y = (y_1, y_2, .., y_n) the matrix evaluated
// is actually the Jacobian transpose, defined as this mxn matrix:
// dy_1/d_x1 dy_2/dx_1 ... dy_n/dx_1
// dy_1/dx_2 dy_2/dx_2 ... dy_n/dx_2
//     .
//     .
//     .
// dy_1/dx_m dy_2/dx_m ... dy_n/dx_m
//
// If x or y is complex, each complex entry is "expanded" into a real and
// imaginary entry, and the Jacobian is organized as above on the expanded list.
// e.g.
// [y1, y2] = Square([x1, x2]) where x and y are complex.
// Writing
// x = [x1_real, x1_imag, x2_real, x2_imag]
// y = [y1_real, y1_imag, y2_real, y2_imag]
// the Jacobian transpose is
// the 4x4 matrix:
// dy1_real/dx1_real dy1_imag/dx1_real dy2_real/dx1_real dy2_imag/dx1_real
// dy1_real/dx1_imag dy1_imag/dx1_imag dy2_real/dx1_imag dy2_imag/dx1_imag
// dy1_real/dx2_real dy1_imag/dx2_real dy2_real/dx2_real dy2_imag/dx2_real
// dy1_real/dx2_imag dy1_imag/dx2_imag dy2_real/dx2_imag dy2_imag/dx2_imag
template <typename X_T, typename Y_T, typename JAC_T>
void InitJacobians(const OutputList& xs,
                   const std::vector<TensorShape>& x_shapes,
                   const std::vector<TensorShape>& y_shapes,
                   std::vector<Tensor>* jacobians) {
  const size_t y_num = y_shapes.size();
  const size_t x_num = x_shapes.size();
  const DataType jacobian_type = DataTypeToEnum<JAC_T>::v();

  jacobians->resize(y_num * x_num);
  for (int x_idx = 0; x_idx < x_num; x_idx++) {
    // The number of rows is the number of elements in the x tensor multiplied
    // by the number of Jacobian entries needed to represent each x type.
    const int64_t x_size =
        x_shapes[x_idx].num_elements() * JacobianStride<X_T>::value;
    for (int y_idx = 0; y_idx < y_num; y_idx++) {
      // The number of columns is the number of elements in the y tensor
      // multiplied by the number of Jacobian entries needed to represent each
      // y type.
      const int64_t y_size =
          y_shapes[y_idx].num_elements() * JacobianStride<Y_T>::value;
      Tensor jacobian_t(jacobian_type, {x_size, y_size});
      auto jacobian_t_flat = jacobian_t.flat<JAC_T>();
      jacobian_t_flat.setZero();
      (*jacobians)[x_idx * y_num + y_idx] = std::move(jacobian_t);
    }
  }
}

template <typename X_T, typename Y_T, typename JAC_T>
Status ComputeGradientErrorInternal(const Scope& scope, const OutputList& xs,
                                    const std::vector<TensorShape>& x_shapes,
                                    const OutputList& ys,
                                    const std::vector<TensorShape>& y_shapes,
                                    std::vector<Tensor>* x_datas,
                                    JAC_T* max_error) {
  // Initialize theoretical Jacobians to zeros.
  std::vector<Tensor> jacobian_ts;
  InitJacobians<X_T, Y_T, JAC_T>(xs, x_shapes, y_shapes, &jacobian_ts);

  // Compute theoretical Jacobian.
  TF_RETURN_IF_ERROR((ComputeTheoreticalJacobianTranspose<X_T, Y_T, JAC_T>(
      scope, xs, x_shapes, *x_datas, ys, y_shapes, &jacobian_ts)));

  // Initialize numeric Jacobian to zeros.
  std::vector<Tensor> jacobian_ns;
  InitJacobians<X_T, Y_T, JAC_T>(xs, x_shapes, y_shapes, &jacobian_ns);

  // Compute numeric Jacobian.
  TF_RETURN_IF_ERROR((ComputeNumericJacobianTranspose<X_T, Y_T, JAC_T>(
      scope, xs, x_shapes, ys, y_shapes, JAC_T{1e-3f}, x_datas, &jacobian_ns)));

  for (int i = 0; i < jacobian_ts.size(); i++) {
    // Compute the maximum error between theoretical and numeric Jacobians.
    *max_error = 0.0;
    auto jac_t = jacobian_ts[i].matrix<JAC_T>();
    auto jac_n = jacobian_ns[i].matrix<JAC_T>();
    for (int r = 0; r < jacobian_ts[i].dim_size(0); ++r) {
      for (int c = 0; c < jacobian_ts[i].dim_size(1); ++c) {
        auto cur_error = std::fabs(jac_t(r, c) - jac_n(r, c));
        // Treat any NaN as max_error and immediately return.
        // (Note that std::max may ignore NaN arguments.)
        if (std::isnan(cur_error)) {
          *max_error = cur_error;
          return OkStatus();
        }
        *max_error = std::max(*max_error, cur_error);
      }
    }
  }
  return OkStatus();
}

}  // namespace

template <typename X_T, typename Y_T, typename JAC_T>
Status ComputeGradientError(const Scope& scope, const OutputList& xs,
                            const std::vector<TensorShape>& x_shapes,
                            const OutputList& ys,
                            const std::vector<TensorShape>& y_shapes,
                            JAC_T* max_error) {
  if (xs.size() != x_shapes.size()) {
    return errors::InvalidArgument("xs(size ", xs.size(),
                                   ") and x_shapes(size ", x_shapes.size(),
                                   ") must be the same size.");
  }
  if (ys.size() != y_shapes.size()) {
    return errors::InvalidArgument("ys(size ", ys.size(),
                                   ") and y_shapes(size ", y_shapes.size(),
                                   ") must be the same size.");
  }
  // Initialize 'x_datas' to random values.
  std::vector<Tensor> x_datas(x_shapes.size());
  for (int i = 0; i < x_shapes.size(); i++) {
    x_datas[i] = Tensor(xs[i].type(), x_shapes[i]);
    auto x_data_flat = x_datas[i].flat<X_T>();
    x_data_flat.setRandom();
  }
  // Compute gradient error.
  return ComputeGradientErrorInternal<X_T, Y_T, JAC_T>(
      scope, xs, x_shapes, ys, y_shapes, &x_datas, max_error);
}

template <typename X_T, typename Y_T, typename JAC_T>
Status ComputeGradientError(const Scope& scope, const Output& x,
                            const Tensor& x_init_value, const Output& y,
                            const TensorShape& y_shape, JAC_T* max_error) {
  // Initialize 'x_data' from 'x_init_value'.
  std::vector<Tensor> x_datas(1, Tensor(x_init_value));
  // Compute gradient error.
  return ComputeGradientErrorInternal<X_T, Y_T, JAC_T>(
      scope, {x}, {x_datas[0].shape()}, {y}, {y_shape}, &x_datas, max_error);
}

#define INSTANTIATE_GRAD_ERR_TYPE(X_T, Y_T, JAC_T)                     \
  template Status ComputeGradientError<X_T, Y_T, JAC_T>(               \
      const Scope& scope, const OutputList& xs,                        \
      const std::vector<TensorShape>& x_shapes, const OutputList& ys,  \
      const std::vector<TensorShape>& y_shapes, JAC_T* max_error);     \
  template Status ComputeGradientError<X_T, Y_T, JAC_T>(               \
      const Scope& scope, const Output& x, const Tensor& x_init_value, \
      const Output& y, const TensorShape& y_shape, JAC_T* max_error);

INSTANTIATE_GRAD_ERR_TYPE(float, float, float);
INSTANTIATE_GRAD_ERR_TYPE(double, float, double);
INSTANTIATE_GRAD_ERR_TYPE(double, double, double);
INSTANTIATE_GRAD_ERR_TYPE(complex64, float, float);
INSTANTIATE_GRAD_ERR_TYPE(float, complex64, float);
INSTANTIATE_GRAD_ERR_TYPE(complex64, complex64, float);
INSTANTIATE_GRAD_ERR_TYPE(complex128, complex128, double);

}  // namespace tensorflow
