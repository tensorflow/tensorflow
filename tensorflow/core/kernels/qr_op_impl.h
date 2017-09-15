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

// See docs in ../ops/linalg_ops.cc.
//
// This header file is used by the individual qr_*op*.cc files for registering
// individual kernels. A separate file is used for each instantiated kernel to
// improve compilation times.
#include <algorithm>

#include "third_party/eigen3/Eigen/QR"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar>
class QrOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit QrOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  using TensorShapes = typename Base::TensorShapes;

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
    Base::ValidateSingleMatrix(context, input_matrix_shapes);
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    int64 m = input_matrix_shapes[0].dim_size(0);
    int64 n = input_matrix_shapes[0].dim_size(1);
    int64 min_size = std::min(m, n);
    if (full_matrices_) {
      return TensorShapes({TensorShape({m, m}), TensorShape({m, n})});
    } else {
      return TensorShapes(
          {TensorShape({m, min_size}), TensorShape({min_size, n})});
    }
  }

  int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double max_size = std::max(m, n);
    double min_size = std::min(m, n);
    double cost = 2 * max_size * min_size * min_size -
                  2 * min_size * min_size * min_size / 3.;
    // TODO(jpoulson): Increase the cost if full_matrices is true in a manner
    // that reflects the algorithm used for the expansion.
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64>(cost);
  }

  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    Eigen::HouseholderQR<Matrix> qr(inputs[0]);
    const int m = inputs[0].rows();
    const int n = inputs[0].cols();
    const int min_size = std::min(m, n);

    if (full_matrices_) {
      outputs->at(0) = qr.householderQ();
      outputs->at(1) = qr.matrixQR().template triangularView<Eigen::Upper>();
    } else {
      // TODO(jpoulson): Exploit the fact that Householder transformations can
      // be expanded faster than they can be applied to an arbitrary matrix
      // (Cf. LAPACK's DORGQR).
      Matrix tmp = Matrix::Identity(m, min_size);
      outputs->at(0) = qr.householderQ() * tmp;
      auto qr_top = qr.matrixQR().block(0, 0, min_size, n);
      outputs->at(1) = qr_top.template triangularView<Eigen::Upper>();
    }
  }

 private:
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(QrOp);
};

}  // namespace tensorflow
