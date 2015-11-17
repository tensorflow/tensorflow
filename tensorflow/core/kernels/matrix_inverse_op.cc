// See docs in ../ops/linalg_ops.cc.
#include <cmath>

#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperationT>
class MatrixInverseOp
    : public LinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixInverseOp(OpKernelConstruction* context)
      : LinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) {}
  ~MatrixInverseOp() override {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) override {
    return input_matrix_shape;
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return rows * rows * rows;
    }
  }

  using typename LinearAlgebraOp<Scalar, SupportsBatchOperationT>::Matrix;
  using typename LinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap;
  using
      typename LinearAlgebraOp<Scalar, SupportsBatchOperationT>::ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& input,
                     MatrixMap* output) override {
    OP_REQUIRES(context, input.rows() == input.cols(),
                errors::InvalidArgument("Input matrix must be square."));
    if (input.rows() == 0) {
      // By definition, an empty matrix's inverse is an empty matrix.
      return;
    }
    if (input.isApprox(input.transpose())) {
      // Matrix is symmetric, compute Cholesky factorization
      // input = L * L^T.
      Eigen::LLT<Matrix, Eigen::Lower> cholesky_decomposition(input);
      if (cholesky_decomposition.info() == Eigen::Success) {
        // Cholesky succeeded => Matrix was SPD.
        output->noalias() = cholesky_decomposition.solve(
            Matrix::Identity(input.rows(), input.cols()));
        return;
      }
    }
    Eigen::PartialPivLU<Matrix> lu_decomposition(input);
    // While PartialPivLU cannot give strong guarantees on invertability,
    // we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes, such as providing integer valued
    // matrices that are exacly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    // TODO(rmlarsen): Add check based on condition number estimation.
    const Scalar min_abs_pivot =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > Scalar(0),
                errors::InvalidArgument("Input is not invertible."));
    output->noalias() = lu_decomposition.inverse();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixInverseOp);
};

REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<float, false>), float);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<double, false>), double);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<float, true>), float);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<double, true>),
                   double);

}  // namespace tensorflow
