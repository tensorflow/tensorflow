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
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/errors.h"

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
    const Scope& scope, const ops::OutputList& xs,
    const std::vector<TensorShape>& x_shapes,
    const std::vector<Tensor>& x_datas, const ops::OutputList& ys,
    const std::vector<TensorShape>& y_shapes,
    std::vector<Tensor>& jacobian_ts) {
  int y_num = y_shapes.size();
  int x_num = x_shapes.size();
  // Call AddSymbolicGradients to get 'dxs' (we will feed 'dys').
  ops::OutputList dys;
  for (const auto& y_shape : y_shapes) {
    // TODO(suharshs): This currently assumes that all x's are the same type.
    dys.push_back(Cast(scope, Const(scope, 1.0, y_shape), xs[0].type()));
  }
  ops::OutputList dxs;
  TF_RETURN_IF_ERROR(AddSymbolicGradients(scope, ys, xs, dys, &dxs));

  // Initialize 'dy_data' to zeros.
  std::vector<Tensor> dy_datas(y_num);
  for (int i = 0; i < y_num; i++) {
    dy_datas[i] = Tensor(ys[i].type(), y_shapes[i]);
    auto dy_data_flat = dy_datas[i].flat<T>();
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

  ClientSession session(scope);
  for (int y_idx = 0; y_idx < y_num; y_idx++) {
    auto dy_data_flat = dy_datas[y_idx].flat<T>();
    const int64 dy_size = y_shapes[y_idx].num_elements();

    // Compute the theoretical Jacobians one row at a time by back propagating
    // '1.0' for each element of 'dy', while holding all other elements of 'dy'
    // at zero.
    for (int c = 0; c < dy_size; ++c) {
      dy_data_flat(c) = 1.0;

      std::vector<Tensor> dxout;
      TF_RETURN_IF_ERROR(session.Run(feed_list, dxs, &dxout));

      for (int x_idx = 0; x_idx < x_num; x_idx++) {
        const int64 x_size = x_shapes[x_idx].num_elements();
        auto jacobian = jacobian_ts[x_idx * y_num + y_idx].matrix<T>();
        auto dx_flat = dxout[x_idx].flat<T>();
        for (int r = 0; r < x_size; ++r) {
          jacobian(r, c) = dx_flat(r);
        }
      }

      dy_data_flat(c) = 0.0;
    }
  }
  return Status::OK();
}

Status EvaluateGraph(ClientSession& session, const ops::OutputList& xs,
                     const ops::OutputList& ys, std::vector<Tensor>& x_datas,
                     std::vector<Tensor>* y_datas) {
  // Create the feed list.
  ClientSession::FeedType feed_list;
  for (int i = 0; i < x_datas.size(); i++) {
    feed_list.insert({xs[i], x_datas[i]});
  }

  TF_RETURN_IF_ERROR(session.Run(feed_list, ys, y_datas));
  for (int y_idx = 0; y_idx < y_datas->size(); y_idx++) {
    for (int x_idx = 0; x_idx < x_datas.size(); x_idx++) {
      Tensor y_data = (*y_datas)[y_idx];
      if (y_data.SharesBufferWith(x_datas[x_idx])) {
        // Create copies of outputs that share a buffer with any inputs since
        // the underlying buffer of the input Tensors are not copied for some
        // operations (i.e. Identity), which can lead to incorrect results for
        // the centered difference calculation.
        (*y_datas)[y_idx] = tensor::DeepCopy(y_data);
      }
    }
  }
  return Status::OK();
}

template <typename T>
Status ComputeNumericJacobianTranspose(
    const Scope& scope, const ops::OutputList& xs,
    const std::vector<TensorShape>& x_shapes, const ops::OutputList& ys,
    const std::vector<TensorShape>& y_shapes, const T delta,
    std::vector<Tensor>& x_datas, std::vector<Tensor>& jacobian_ts) {
  int y_num = y_shapes.size();
  int x_num = x_shapes.size();

  ClientSession session(scope);
  for (int x_idx = 0; x_idx < x_num; x_idx++) {
    auto x_data_flat = x_datas[x_idx].flat<T>();
    const int64 x_size = x_shapes[x_idx].num_elements();

    // Compute the numeric Jacobian one column at a time by perturbing each
    // element of 'x_data' (positively and negatively) by 'delta', and
    // updating the jacobian with the centered difference.
    for (int r = 0; r < x_size; ++r) {
      // Store current value of 'x' at 'r'.
      T v = x_data_flat(r);
      // Evaluate at positive delta.
      x_data_flat(r) = v + delta;
      std::vector<Tensor> y_pos;
      TF_RETURN_IF_ERROR(EvaluateGraph(session, xs, ys, x_datas, &y_pos));
      // Evaluate at negative delta.
      x_data_flat(r) = v - delta;
      std::vector<Tensor> y_neg;
      TF_RETURN_IF_ERROR(EvaluateGraph(session, xs, ys, x_datas, &y_neg));

      for (int y_idx = 0; y_idx < y_num; y_idx++) {
        // Compute element-wise centered difference and store in each Jacobian.
        auto y_pos_flat = y_pos[y_idx].flat<T>();
        auto y_neg_flat = y_neg[y_idx].flat<T>();
        const int64 y_size = y_shapes[y_idx].num_elements();
        const T scale = 2 * delta;
        auto jacobian = jacobian_ts[x_idx * y_num + y_idx].matrix<T>();
        for (int c = 0; c < y_size; ++c) {
          jacobian(r, c) = (y_pos_flat(c) - y_neg_flat(c)) / scale;
        }
      }
      // Restore pre-perturbation value.
      x_data_flat(r) = v;
    }
  }
  return Status::OK();
}

template <typename T>
void InitJacobians(const ops::OutputList& xs,
                   const std::vector<TensorShape>& x_shapes,
                   const std::vector<TensorShape>& y_shapes,
                   std::vector<Tensor>& jacobians) {
  int y_num = y_shapes.size();
  int x_num = x_shapes.size();

  jacobians.resize(y_num * x_num);
  for (int x_idx = 0; x_idx < x_num; x_idx++) {
    const int64 x_size = x_shapes[x_idx].num_elements();
    for (int y_idx = 0; y_idx < y_num; y_idx++) {
      const int64 y_size = y_shapes[y_idx].num_elements();
      Tensor jacobian_t(xs[x_idx].type(), {x_size, y_size});
      auto jacobian_t_flat = jacobian_t.flat<T>();
      jacobian_t_flat.setZero();
      jacobians[x_idx * y_num + y_idx] = std::move(jacobian_t);
    }
  }
}

template <typename T>
Status ComputeGradientErrorInternal(const Scope& scope,
                                    const ops::OutputList& xs,
                                    const std::vector<TensorShape>& x_shapes,
                                    const ops::OutputList& ys,
                                    const std::vector<TensorShape>& y_shapes,
                                    std::vector<Tensor>& x_datas,
                                    T* max_error) {
  // Initialize theoretical Jacobians to zeros.
  std::vector<Tensor> jacobian_ts;
  InitJacobians<T>(xs, x_shapes, y_shapes, jacobian_ts);

  // Compute theoretical Jacobian.
  TF_RETURN_IF_ERROR(ComputeTheoreticalJacobianTranspose<T>(
      scope, xs, x_shapes, x_datas, ys, y_shapes, jacobian_ts));

  // Initialize numeric Jacobian to zeros.
  std::vector<Tensor> jacobian_ns;
  InitJacobians<T>(xs, x_shapes, y_shapes, jacobian_ns);

  // Compute numeric Jacobian.
  TF_RETURN_IF_ERROR(ComputeNumericJacobianTranspose<T>(
      scope, xs, x_shapes, ys, y_shapes, 1e-3, x_datas, jacobian_ns));

  for (int i = 0; i < jacobian_ts.size(); i++) {
    // Compute the maximum error between theoretical and numeric Jacobians.
    *max_error = 0.0;
    auto jac_t = jacobian_ts[i].matrix<T>();
    auto jac_n = jacobian_ns[i].matrix<T>();
    for (int r = 0; r < jacobian_ts[i].dim_size(0); ++r) {
      for (int c = 0; c < jacobian_ts[i].dim_size(1); ++c) {
        *max_error = std::max(*max_error, std::fabs(jac_t(r, c) - jac_n(r, c)));
      }
    }
  }
  return Status::OK();
}

}  // namespace

template <typename T>
Status ComputeGradientError(const Scope& scope, const ops::OutputList& xs,
                            const std::vector<TensorShape>& x_shapes,
                            const ops::OutputList& ys,
                            const std::vector<TensorShape>& y_shapes,
                            T* max_error) {
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
    auto x_data_flat = x_datas[i].flat<T>();
    x_data_flat.setRandom();
  }
  // Compute gradient error.
  return ComputeGradientErrorInternal(scope, xs, x_shapes, ys, y_shapes,
                                      x_datas, max_error);
}

template <typename T>
Status ComputeGradientError(const Scope& scope, const ops::Output& x,
                            const Tensor& x_init_value, const ops::Output& y,
                            const TensorShape& y_shape, T* max_error) {
  // Initialize 'x_data' from 'x_init_value'.
  std::vector<Tensor> x_datas(1, Tensor(x_init_value));
  // Compute gradient error.
  return ComputeGradientErrorInternal(scope, {x}, {x_datas[0].shape()}, {y},
                                      {y_shape}, x_datas, max_error);
}

#define INSTANTIATE_GRAD_ERR_TYPE(T)                                        \
  template Status ComputeGradientError<T>(                                  \
      const Scope& scope, const ops::OutputList& xs,                        \
      const std::vector<TensorShape>& x_shapes, const ops::OutputList& ys,  \
      const std::vector<TensorShape>& y_shapes, T* max_error);              \
  template Status ComputeGradientError<T>(                                  \
      const Scope& scope, const ops::Output& x, const Tensor& x_init_value, \
      const ops::Output& y, const TensorShape& y_shape, T* max_error);

INSTANTIATE_GRAD_ERR_TYPE(float);
INSTANTIATE_GRAD_ERR_TYPE(double);

}  // namespace tensorflow
