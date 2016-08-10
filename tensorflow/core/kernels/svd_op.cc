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
#include <algorithm>

#include "third_party/eigen3/Eigen/SVD"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperation>
class SvdOp : public LinearAlgebraOp<Scalar, SupportsBatchOperation> {
 public:
  typedef LinearAlgebraOp<Scalar, SupportsBatchOperation> Base;

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
    int64 m = input_matrix_shapes[0].dim_size(0);
    int64 n = input_matrix_shapes[0].dim_size(1);
    int64 min_size = std::min(m, n);
    if (compute_uv_) {
      return TensorShapes({TensorShape({min_size}),
                           TensorShape({m, full_matrices_ ? m : min_size}),
                           TensorShape({n, full_matrices_ ? n : min_size})});
    } else {
      return TensorShapes({TensorShape({min_size})});
    }
  }

  // TODO(rmlarsen): This should depend on compute_uv. See b/30409375.
  int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double cost = 12 * std::max(m, n) * std::min(m, n) * std::min(m, n);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64>(cost);
  }

  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    Eigen::JacobiSVD<Matrix, Eigen::HouseholderQRPreconditioner> svd;
    if (compute_uv_) {
      svd.compute(inputs[0],
                  (full_matrices_ ? Eigen::ComputeFullU | Eigen::ComputeFullV
                                  : Eigen::ComputeThinU | Eigen::ComputeThinV));
      outputs->at(0) = svd.singularValues();
      outputs->at(1) = svd.matrixU();
      outputs->at(2) = svd.matrixV();
    } else {
      svd.compute(inputs[0]);
      outputs->at(0) = svd.singularValues();
    }
  }

 private:
  bool compute_uv_;
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(SvdOp);
};

REGISTER_LINALG_OP("Svd", (SvdOp<float, false>), float);
REGISTER_LINALG_OP("Svd", (SvdOp<double, false>), double);
REGISTER_LINALG_OP("BatchSvd", (SvdOp<float, true>), float);
REGISTER_LINALG_OP("BatchSvd", (SvdOp<double, true>), double);

}  // namespace tensorflow
