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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_SVD_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_SVD_OP_IMPL_H_

// See docs in ../ops/linalg_ops.cc.
//
// This header file is used by the individual svd_*op*.cc files for registering
// individual kernels. A separate file is used for each instantiated kernel to
// improve compilation times.
#include <algorithm>

#include "Eigen/SVD"  // from @eigen_archive
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar>
class SvdOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit SvdOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compute_uv", &compute_uv_));
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
    int64_t m = input_matrix_shapes[0].dim_size(0);
    int64_t n = input_matrix_shapes[0].dim_size(1);
    int64_t min_size = std::min(m, n);
    if (compute_uv_) {
      return TensorShapes({TensorShape({min_size}),
                           TensorShape({m, full_matrices_ ? m : min_size}),
                           TensorShape({n, full_matrices_ ? n : min_size})});
    } else {
      return TensorShapes({TensorShape({min_size})});
    }
  }

  // TODO(rmlarsen): This should depend on compute_uv. See b/30409375.
  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double cost = 12 * std::max(m, n) * std::min(m, n) * std::min(m, n);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    int64_t n = inputs[0].cols();
    int64_t m = inputs[0].rows();
    const bool empty = (m == 0 || n == 0);
    int options = 0;  // Don't compute singular vectors;
    if (compute_uv_) {
      options = full_matrices_ ? Eigen::ComputeFullU | Eigen::ComputeFullV
                               : Eigen::ComputeThinU | Eigen::ComputeThinV;
    }

    if (empty) {
      // For an empty matrix where only one dimension is zero, we still set
      // U or V to the unit matrix for the dimension that is non-zero.
      if (compute_uv_ && full_matrices_) {
        if (m > 0) {
          outputs->at(1) = Matrix::Identity(m, m);
        } else {
          outputs->at(2) = Matrix::Identity(n, n);
        }
      }
      return;
    }

    Eigen::BDCSVD<Matrix> svd(inputs[0], options);
    if (svd.info() != Eigen::Success) {
      LOG(ERROR) << "Eigen::BDCSVD failed with error code " << svd.info();
      outputs->at(0).fill(std::numeric_limits<Scalar>::quiet_NaN());
      if (compute_uv_) {
        outputs->at(1).fill(std::numeric_limits<Scalar>::quiet_NaN());
        outputs->at(2).fill(std::numeric_limits<Scalar>::quiet_NaN());
      }
    } else {
      outputs->at(0) = svd.singularValues().template cast<Scalar>();
      if (compute_uv_) {
        outputs->at(1) = svd.matrixU();
        outputs->at(2) = svd.matrixV();
      }
    }
  }

 private:
  bool compute_uv_;
  bool full_matrices_;

  SvdOp(const SvdOp&) = delete;
  void operator=(const SvdOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_SVD_OP_IMPL_H_
