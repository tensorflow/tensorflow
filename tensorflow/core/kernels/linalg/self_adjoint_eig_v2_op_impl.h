/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_SELF_ADJOINT_EIG_V2_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_SELF_ADJOINT_EIG_V2_OP_IMPL_H_

// See docs in ../ops/linalg_ops.cc.

#include "Eigen/Core"  // from @eigen_archive
#include "Eigen/Eigenvalues"  // from @eigen_archive
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar>
class SelfAdjointEigV2Op : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit SelfAdjointEigV2Op(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compute_v", &compute_v_));
  }

  using TensorShapes = typename Base::TensorShapes;
  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    int64_t n = input_matrix_shapes[0].dim_size(0);
    if (compute_v_) {
      return TensorShapes({TensorShape({n}), TensorShape({n, n})});
    } else {
      return TensorShapes({TensorShape({n})});
    }
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const int64_t rows = inputs[0].rows();
    if (rows == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      return;
    }

    // This algorithm relies on denormals, so switch them back on locally.
    port::ScopedDontFlushDenormal dont_flush_denormals;

    Eigen::SelfAdjointEigenSolver<Matrix> eig(
        inputs[0],
        compute_v_ ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
    // TODO(rmlarsen): Output more detailed error info on failure.
    OP_REQUIRES(
        context, eig.info() == Eigen::Success,
        errors::InvalidArgument("Self-adjoint eigen decomposition was not "
                                "successful. The input might not be valid."));

    outputs->at(0) = eig.eigenvalues().template cast<Scalar>();
    if (compute_v_) {
      outputs->at(1) = eig.eigenvectors();
    }
  }

 private:
  bool compute_v_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_SELF_ADJOINT_EIG_V2_OP_IMPL_H_
